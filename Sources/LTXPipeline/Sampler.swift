// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The rectified-flow schedule, the guidance stack and the Euler loop.
///
/// This is the arithmetic *between* transformer calls. It owns no weights and no modules:
/// there is no matmul, and the one reduction — the `std` inside the rescale — runs in fp32
/// over the whole tensor, so two correct implementations have no summation order to
/// disagree about.
///
/// ## The three mistakes a difference cannot see
///
/// bf16 storage resolves to about 3.32e-03, so anything below that is invisible to a
/// comparison of two rendered tensors. Exactly three mistakes live down there, all of them
/// dtype discipline: leaving the velocity unrounded (1.56e-03), combining the guidance
/// terms in bf16 (1.97e-03), and the population-versus-sample `std` choice — which is
/// exactly 0.0, because the correction cancels algebraically. Each is written out at the
/// line where it would be made, so the reason the code is shaped that way sits next to the
/// shape.
///
/// Every *structural* mistake lands far outside: STG written as `scale - 1` is 1.30e-02,
/// two passes swapped 1.5e-02 to 1.9e-01, a flipped `dt` 8.19e-01, and the scaled timestep
/// in the x0 conversion 4.29e+02.
///
/// ## What is deliberately not here
///
/// **No noise generation.** Contract 1: the initial latent is a *recorded* tensor, never a
/// seed. `Pipeline.recordedNoise` is the only way in, and there is no `seed:` parameter
/// anywhere in this file. A bf16 CUDA draw is not portable, so redrawing from `seed: 0`
/// reproduces nothing — see `docs/FRAGILE_CONTRACTS.md` #1.
///
/// **No transformer.** `Sampler.Denoiser` is a seam. The sampler decides *which* passes run
/// and in what order; what a pass costs is the DiT's business, and keeping them apart is
/// what lets the whole guidance stack be exercised against a stub denoiser in milliseconds
/// rather than through a 21 B-parameter forward.
///
/// `Sendable` because it holds nothing but the schedule and the two guidance parameter
/// sets — all value types. No `MLXArray` is stored anywhere on it, which is what lets a
/// suite build one `static let` and share it across parallel tests.
public struct Sampler: Sendable {

    public var schedule: FlowSchedule
    public var video: GuidanceParams
    public var audio: GuidanceParams

    public init(schedule: FlowSchedule = FlowSchedule(),
                video: GuidanceParams, audio: GuidanceParams) {
        self.schedule = schedule
        self.video = video
        self.audio = audio
    }

    public enum Failure: Error, CustomStringConvertible {
        case steps(Int)
        case missingPass(Pass, stream: String)
        case zeroSigma(step: Int)
        /// `eta > 0` with no noise tensor. Raising beats substituting zeros: a silently
        /// deterministic "ancestral" step is a render that looks fine and is not the
        /// schedule it claims to be.
        case missingAncestralNoise(step: Int)
        case tokenCount(stream: String, got: Int, expected: Int)
        /// A latent arrived in the wrong rank on the way *into* token space. Distinct from
        /// `tokenCount` on purpose: that one means the right kind of tensor with the wrong
        /// extent, this one means the caller handed over something that is not a VAE
        /// latent at all — most likely a token-space stream that has already been
        /// patchified once.
        case latentRank(stream: String, got: [Int], expected: String)

        public var description: String {
            switch self {
            case let .steps(n):
                return "a schedule needs at least one step, not \(n)"
            case let .missingPass(pass, stream):
                return "the \(stream) guider asks for the \(pass) pass but no velocity "
                    + "was produced for it"
            case let .zeroSigma(step):
                return "sigma is 0 at step \(step); to_velocity divides by it"
            case let .missingAncestralNoise(step):
                return "the ancestral step at \(step) has eta > 0 and no noise tensor; "
                    + "it needs one to renoise from sigma_down back up to sigma_next"
            case let .tokenCount(stream, got, expected):
                return "the \(stream) latent carries \(got) tokens, not \(expected)"
            case let .latentRank(stream, got, expected):
                return "the \(stream) latent is \(got), expected \(expected) — a token-space "
                    + "stream cannot be patchified twice"
            }
        }
    }

    // MARK: - The passes

    /// One forward pass of the transformer, and its position in the batch.
    ///
    /// A step appends its passes in exactly this order and chunks the batch back apart
    /// afterwards, so the raw integer is the pass's identity. At a batch size of 1 each
    /// pass is its own transformer call, and the calls of a run walk straight through this
    /// list, once per enabled pass per step.
    public enum Pass: Int, Sendable, CaseIterable, CustomStringConvertible {
        /// The positive prompt, unperturbed.
        case conditional = 0
        /// The negative prompt. Present when either stream's CFG scale is not 1.
        case unconditional = 1
        /// Positive prompt, self-attention replaced by its value projection in the STG
        /// blocks. Present when either stream's STG scale is not 0.
        case perturbed = 2
        /// Positive prompt, **both cross-modal attentions severed**. Present when either
        /// stream's modality scale is not 1.
        case modality = 3

        public var description: String {
            switch self {
            case .conditional: return "conditional"
            case .unconditional: return "unconditional"
            case .perturbed: return "perturbed"
            case .modality: return "modality"
            }
        }
    }

    /// Which passes a step runs, and which streams are live for it.
    ///
    /// The two flags are not decoration. The pass list is decided by both guiders
    /// together, but the substitution is **per stream**: a step where only the audio
    /// guider skips still pays for every pass, still steps the audio latent, and steps it
    /// with a **stale** prediction. `Sampler.step` takes `enabled:` for exactly this and
    /// `run` fills it from here.
    public struct Plan: Sendable, Equatable {
        public var passes: [Pass]
        /// False when this stream's `skipStep` skips this step.
        public var videoEnabled: Bool
        public var audioEnabled: Bool
        /// Both streams skipped: no transformer call at all. The step still *happens* —
        /// both latents advance on the reused predictions.
        public var isEmpty: Bool { !videoEnabled && !audioEnabled }
    }

    /// The pass list, as a function of the two guiders, the step index, and this step's
    /// sigma (for the percent windows).
    ///
    /// `sigma` defaults to 0.5, which sits inside the default `[0, 1]` windows. Tests
    /// that do not name a window keep working; a narrowed window must pass the real
    /// schedule value.
    ///
    /// The union, not the intersection: a pass runs if **either** stream needs it, because
    /// one transformer call carries both streams. So an audio-only CFG still costs the
    /// video stream a pass, and the video guider's `calculate` will then combine an
    /// unconditioned prediction it did not ask for with a coefficient of exactly zero.
    /// That is what makes the pass count predictable from the guidance settings alone:
    /// guidance costs are additive.
    public func plan(step: Int, sigma: Float = 0.5) -> Plan {
        let vSkip = video.skipsStep(step), aSkip = audio.skipsStep(step)
        if vSkip && aSkip {
            return Plan(passes: [], videoEnabled: false, audioEnabled: false)
        }
        var passes: [Pass] = [.conditional]
        if video.runsUnconditional || audio.runsUnconditional { passes.append(.unconditional) }
        if video.runsPerturbed(atSigma: sigma) || audio.runsPerturbed(atSigma: sigma) {
            passes.append(.perturbed)
        }
        if video.runsIsolatedModality(atSigma: sigma)
            || audio.runsIsolatedModality(atSigma: sigma) {
            passes.append(.modality)
        }
        return Plan(passes: passes, videoEnabled: !vSkip, audioEnabled: !aSkip)
    }

    /// Transformer calls a whole render will make, without making any of them.
    ///
    /// Guidance behaviour is worth stating as a *pass delta* rather than as an output
    /// comparison, precisely because an implementation that computes a disabled guidance
    /// term and multiplies it by zero is byte-identical and twice the price.
    public func passCount(steps: Int) -> Int {
        (0..<steps).reduce(0) { $0 + plan(step: $1).passes.count }
    }

    // MARK: - The step

    /// The two latent streams, in the patchified `[B, T, C]` layout the loop works in.
    ///
    /// The sampler never sees `[B, C, F, H, W]`. The latent stays patchified from the
    /// initial state until the unpatchify at the very end, and `patchify_proj.in` observes
    /// it in exactly this layout — which is why a step can be replayed against a recorded
    /// tensor with no reshaping at all.
    public struct Streams {
        public var video: MLXArray
        public var audio: MLXArray
        public init(video: MLXArray, audio: MLXArray) {
            self.video = video
            self.audio = audio
        }
    }

    /// The x0 conversion: `sample - velocity * timesteps`.
    ///
    /// `timesteps` is the **unscaled** per-token noise level `denoiseMask * sigma`, built
    /// before anything scales it. The `timestep_scale_multiplier` of 1000 belongs to the
    /// AdaLN input and only there; using the scaled value here lands 4.29e+02 out, which
    /// is the largest error anything in this file can produce and among the easiest to
    /// make, because the AdaLN input is the scaled one and it is the timestep most
    /// visible from outside.
    ///
    /// Computed in fp32 and cast back.
    public static func denoised(latent: MLXArray, velocity: MLXArray,
                                timesteps: MLXArray) -> MLXArray {
        (latent.asType(.float32) - velocity.asType(.float32) * timesteps)
            .asType(latent.dtype)
    }

    /// The per-token noise level for an unconditioned render.
    ///
    /// `[1, tokens, 1]` rather than a scalar so it broadcasts against `[B, T, C]` the way
    /// a `[B, T, 1]` mask does. A scalar would broadcast too, and would keep
    /// broadcasting once a conditioned render makes the mask non-uniform — at which point
    /// every token would silently get the same noise level.
    public static func uniformTimesteps(sigma: Float, tokens: Int) -> MLXArray {
        MLXArray([Float](repeating: sigma, count: tokens), [1, tokens, 1])
    }

    /// The noiser's second `lerp` — the conditioned tokens of the **initial** latent start
    /// at the conditioning value, not at noise.
    ///
    /// `clean * (1 - mask) + noise * mask`, which is the same arithmetic as
    /// ``postProcess(denoised:denoiseMask:cleanLatent:)`` and why this delegates to it
    /// rather than restating it. Gaussian noise is added to a latent state **scaled by the
    /// denoise mask**.
    ///
    /// Omitting this is not a small error even though the tokens get overwritten a step
    /// later. Paired with the per-token timesteps it is an outright contradiction: step 0
    /// tells the transformer those tokens are clean, at timestep 0, while they actually
    /// hold random noise. Measured: an i2v render whose frame 1 dissolved the conditioning
    /// image into mush and whose frame 8 onward had no relationship to it at all.
    /// - Parameters:
    ///   - initial: the latent before noising — stage 2's upscaled latent, or `nil` for a
    ///     fresh render, which starts from zeros so the first `lerp` returns the noise
    ///     unchanged.
    ///   - noiseScale: `1.0` is a fresh render; the distilled pipeline's stage 2 passes its
    ///     first sigma, `0.909375`, which re-noises the upscaled
    ///     stage-1 result to that level instead of replacing it. This is the entire
    ///     mechanism by which the refine continues the draft rather than starting over, and
    ///     it is a `lerp` toward noise, not an addition of it.
    public static func seedConditioned(noise: MLXArray, denoiseMask: MLXArray?,
                                       cleanLatent: MLXArray?,
                                       initial: MLXArray? = nil,
                                       noiseScale: Float = 1.0) -> MLXArray {
        // A lerp is `a + w * (b - a)`, computed in fp32 both times.
        var latent = noise.asType(.float32)
        if let initial {
            let start = initial.asType(.float32)
            latent = start + noiseScale * (latent - start)
        } else if noiseScale != 1.0 {
            // Zeros as the start: a fresh render has no initial latent to lerp from.
            latent = latent * noiseScale
        }
        return postProcess(denoised: latent.asType(noise.dtype),
                           denoiseMask: denoiseMask, cleanLatent: cleanLatent)
    }

    /// The per-token noise level — the general case, conditioned or not.
    ///
    /// **This is what makes conditioning reach the model at all.** `postProcess` holds a
    /// conditioned token's *value*; this is what tells the transformer that the value is
    /// clean. A frozen token carries mask 0 and therefore timestep 0, which is the only
    /// signal in the whole forward pass that says "this is a reference, condition on it".
    ///
    /// Feeding a uniform sigma instead does not fail, and that is exactly the problem: the
    /// render completes, the conditioned tokens still show the supplied image or audio
    /// because they are overwritten after every step, and everything else is generated as
    /// if unconditioned. It presents as a hard cut at the first frame and a clip with no
    /// relationship to its own conditioning — measured on a 97-frame i2v render where
    /// frame 0 was the input still and frame 1 was already an unrelated shot.
    ///
    /// With an all-ones mask this is `uniformTimesteps` bit for bit: `1.0 * sigma` is exact
    /// in fp32, so the unconditioned path is unchanged rather than merely close.
    public static func timesteps(sigma: Float, mask: MLXArray?, tokens: Int) -> MLXArray {
        guard let mask else { return uniformTimesteps(sigma: sigma, tokens: tokens) }
        return mask.asType(.float32) * sigma
    }

    /// The guidance combination.
    ///
    /// ```
    /// pred = cond
    ///      + (cfgScale - 1)      * (cond - unconditional)
    ///      + stgScale            * (cond - perturbed)
    ///      + (modalityScale - 1) * (cond - modality)
    ///
    /// if rescaleScale != 0:
    ///     factor = std(cond) / std(pred)
    ///     factor = rescaleScale * factor + (1 - rescaleScale)
    ///     pred   = pred * factor
    /// ```
    ///
    /// **STG is `scale`; CFG and modality are `scale - 1`.** Three terms of the same shape
    /// and one of them is spelled differently, because STG's neutral value is 0 and the
    /// other two neutralise at 1. Writing all three alike costs 1.30e-02 — far outside the
    /// storage floor, so it is caught.
    ///
    /// **An absent pass is not a zero tensor.** Substituting `0.0` for a pass that did not
    /// run turns its term into `(scale - 1) * (cond - 0)`, which is harmless only while
    /// the coefficient happens to be zero. This refuses the combination instead, so a term
    /// with a live coefficient and no pass behind it is an error rather than a silent
    /// `cond * scale`.
    ///
    /// **`std` is the unbiased one** (`ddof: 1`); MLX defaults to `ddof: 0`, so it is
    /// passed explicitly.
    ///
    /// The choice is written from the definition and is **unobservable**, and not merely
    /// because the difference is too small to survive the bf16 round: the correction
    /// appears in the numerator and the denominator of `std(cond) / std(pred)` over the
    /// same element count, so `sqrt(n / (n - 1))` divides out *algebraically*. Switching
    /// to `ddof: 0` moves the output by exactly nothing, and
    /// `GuidanceAlgebraTests.ddofCancels` demonstrates the cancellation at 8 elements,
    /// where the two conventions are 7% apart before they cancel.
    public func combine(_ passes: [Pass: MLXArray], params: GuidanceParams,
                        stream: String) throws -> MLXArray {
        guard let cond = passes[.conditional] else {
            throw Failure.missingPass(.conditional, stream: stream)
        }
        let outDType = cond.dtype
        // Always fp32, never the stream dtype. Combining in bf16 also drags the rescale's
        // scalar arithmetic there, where `0.7 * factor + (1 - 0.7)` lands one ULP apart
        // depending on whether the scalars round before the multiply or after it.
        let work: DType = .float32

        func term(_ pass: Pass, _ coefficient: Double) throws -> MLXArray? {
            guard coefficient != 0 else { return nil }
            guard let other = passes[pass] else {
                throw Failure.missingPass(pass, stream: stream)
            }
            return Float(coefficient) * (cond.asType(work) - other.asType(work))
        }

        var pred = cond.asType(work)
        for t in [try term(.unconditional, params.cfgScale - 1),
                  try term(.perturbed, params.stgScale),
                  try term(.modality, params.modalityScale - 1)] {
            if let t { pred = pred + t }
        }

        let rescale = params.rescaleScale
        if rescale != 0 {
            var factor = MLX.std(cond.asType(work), ddof: 1)
                / MLX.std(pred, ddof: 1)
            factor = Float(rescale) * factor + Float(1 - rescale)
            pred = pred * factor
        }
        return pred.asType(outDType)
    }

    /// The ancestral step — stage 1 of the distilled two-stage pipeline.
    ///
    /// Advances deterministically to an intermediate `sigmaDown <= sigmaNext`, then renoises
    /// back up to `sigmaNext`, rescaling the signal by `alphaNext / alphaDown` so the
    /// transition stays variance-preserving. `eta` interpolates between a plain Euler step
    /// (`eta = 0`, no noise) and a fully ancestral one (`eta = 1`, which is what 2.5 uses).
    ///
    /// ## Three ways to get this wrong, all of which produce a plausible render
    ///
    /// **It is the rectified-flow parameterisation (`alpha = 1 - sigma`), not the DDIM /
    /// variance-exploding one.** The familiar k-diffusion ancestral helper gives a
    /// different `sigmaDown` and a different amount of injected noise for the same `eta`,
    /// and the two **agree only at `eta = 0`**. Reaching for it is the obvious mistake
    /// here, and it is wrong at exactly the `eta` this pipeline uses.
    ///
    /// **This one DOES short-circuit at `sigmaNext == 0`,** returning the denoised prediction
    /// outright — unlike ``eulerStep(sample:denoised:sigmas:index:)``, which has no special
    /// case because the plain formula collapses to it by itself. Adding the special case
    /// there would be wrong; omitting it here would divide by `alphaDown` at the terminal
    /// step.
    ///
    /// **The noise is a separate seeded stream,** offset by `+10000` from the render's
    /// seed. Without the offset the loop's first draw would be bit-identical to the initial
    /// latent's, both being a normal draw at the same shape and dtype from a freshly seeded
    /// generator. That is contract 1's second RNG stream.
    ///
    /// Computed in fp32 and cast back to the sample's dtype.
    public func ancestralStep(sample: MLXArray, denoised: MLXArray, sigmas: [Float],
                              index: Int, noise: MLXArray?,
                              eta: Float = 1.0, sNoise: Float = 1.0,
                              directionDenoised: MLXArray? = nil) throws -> MLXArray {
        let sigma = sigmas[index], sigmaNext = sigmas[index + 1]
        // The terminal short-circuit precedes CFG++ deliberately: at `sigmaNext == 0` the
        // step *is* the prediction, and the destination is the guided one either way.
        if sigmaNext == 0 { return denoised.asType(sample.dtype) }
        guard sigma != 0 else { throw Failure.zeroSigma(step: index) }
        guard eta <= 0 || noise != nil else { throw Failure.missingAncestralNoise(step: index) }

        let x = sample.asType(.float32)
        let x0 = denoised.asType(.float32)

        let downstepRatio = 1.0 + (sigmaNext / sigma - 1.0) * eta
        let sigmaDown = sigmaNext * downstepRatio
        let sigmaDownRatio = sigmaDown / sigma
        // `euler_ancestral_cfg_pp`: the deterministic half takes its direction from the
        // unconditional prediction, exactly as in ``eulerStep(sample:denoised:sigmas:index:directionDenoised:)``,
        // and the renoise below is untouched. Kept as a separate expression from the
        // standard one so the standard path stays bit-identical.
        // Same substitution as ``cfgPPStep(sample:denoised:direction:sigma:sigmaTarget:)``,
        // targeting `sigmaDown` — the deterministic half of the ancestral step. The renoise
        // below is untouched.
        var next = directionDenoised.map {
            Self.cfgPPStep(sample: sample, denoised: denoised, direction: $0,
                           sigma: sigma, sigmaTarget: sigmaDown).asType(.float32)
        } ?? (sigmaDownRatio * x + (1.0 - sigmaDownRatio) * x0)

        if eta > 0, let noise {
            let alphaNext = 1.0 - sigmaNext
            let alphaDown = 1.0 - sigmaDown
            let inner = sigmaNext * sigmaNext
                - sigmaDown * sigmaDown * alphaNext * alphaNext / (alphaDown * alphaDown)
            // Clamp at zero before the square root: the quantity is a variance and floating
            // point can push it a hair negative at the schedule's flat head, where sigma and
            // sigmaNext are within 6e-03 of each other.
            let renoise = Foundation.sqrt(Swift.max(inner, 0))
            next = (alphaNext / alphaDown) * next + noise.asType(.float32) * sNoise * renoise
        }
        return next.asType(sample.dtype)
    }

    /// The deterministic step: `sample + velocity * dt`, where
    /// `velocity = (sample - denoised) / sigma` and `dt = sigmaNext - sigma`.
    ///
    /// The cast in the middle is the one to keep. The velocity is rounded to the sample's
    /// dtype and then immediately re-widened for the update — a round trip that does
    /// nothing algebraically and 1.56e-03 of work numerically. That sits *inside* the
    /// bf16 storage floor, so nothing downstream would notice its absence; this comment
    /// is what holds it.
    ///
    /// There is no `sigmaNext == 0` special case here and there must not be: that belongs
    /// to ``ancestralStep(sample:denoised:sigmas:index:noise:eta:sNoise:)``, and this
    /// schedule's last step relies on the plain formula collapsing to the denoised
    /// prediction exactly — `SamplerLoopTests.terminalSigmaCollapses` pins that.
    /// One CFG++ step: the guided prediction as destination, the unconditional one as the
    /// **noise estimate**, in this schedule's own parameterisation.
    ///
    /// ## Why this is not k-diffusion's formula
    ///
    /// `euler_cfg_pp` is written `x = denoised + d·sigma_next` for `d = (x - x0_uncond)/sigma`.
    /// That is correct under the **variance-exploding** convention `x = x0 + sigma·eps`,
    /// where a step's x0 coefficient is 1 and swapping which prediction supplies `d` changes
    /// only the noise term.
    ///
    /// This schedule is **rectified flow**: `x = (1 - sigma)·x0 + sigma·eps`. Its ordinary
    /// step carries an x0 coefficient of `1 - sigma_next/sigma` — about **0.005** on the
    /// first step of a 30-step run — because the `-sigma_next·x0/sigma` inside the velocity
    /// cancels almost all of the leading `x0`. Take the direction from a *different*
    /// prediction and that cancellation stops happening, leaving x0 at coefficient 1: a
    /// ~200x amplification of the roughest prediction in the whole trajectory, applied at
    /// every step. Transcribed literally, it renders uniform mush — measured, at guidance
    /// scales 3.0 and 1.5 alike, which is what ruled out a scale mismatch as the cause.
    ///
    /// So the swap happens where it belongs. Recover the unconditional noise estimate,
    /// `eps_u = (x - (1 - sigma)·x0_u) / sigma`, and step to the target with the schedule's
    /// own coefficients: `(1 - sigma_target)·x0_guided + sigma_target·eps_u`. When the two
    /// predictions agree this reduces exactly to the ordinary step, which is the property
    /// that makes it a correction rather than a different sampler.
    static func cfgPPStep(sample: MLXArray, denoised: MLXArray, direction: MLXArray,
                          sigma: Float, sigmaTarget: Float) -> MLXArray {
        let x = sample.asType(.float32)
        let guided = denoised.asType(.float32)
        let uncond = direction.asType(.float32)
        let eps = (x - (1.0 - sigma) * uncond) / sigma
        return ((1.0 - sigmaTarget) * guided + sigmaTarget * eps).asType(sample.dtype)
    }

    public func eulerStep(sample: MLXArray, denoised: MLXArray, sigmas: [Float],
                          index: Int,
                          directionDenoised: MLXArray? = nil) throws -> MLXArray {
        let sigma = sigmas[index], sigmaNext = sigmas[index + 1]
        guard sigma != 0 else { throw Failure.zeroSigma(step: index) }

        // CFG++ takes the *direction* from a different prediction than the destination.
        //
        // The plain step is `x0 + r(sample - x0)` for `r = sigmaNext / sigma`, written
        // below as `sample + v·dt` — the same thing, and the spelling is load-bearing for
        // bit-exactness, so this is a separate branch rather than a generalisation of it.
        // CFG++ keeps the guided `denoised` as the destination and takes the direction from
        // the **unconditional** prediction: `x0_guided + r(sample - x0_uncond)`.
        //
        // What that buys: the guided x0 is a extrapolation away from the unconditional one,
        // and at a high CFG scale the step direction inherits the whole of that
        // extrapolation. Stepping along the unconditional direction while still landing on
        // the guided prediction keeps the target and drops the over-shoot, which is what
        // "CFG burn" — the crushed, over-saturated look at high guidance — actually is.
        if let directionDenoised {
            return Self.cfgPPStep(sample: sample, denoised: denoised,
                                  direction: directionDenoised,
                                  sigma: sigma, sigmaTarget: sigmaNext)
        }

        let dt = sigmaNext - sigma
        var velocity = (sample.asType(.float32) - denoised.asType(.float32)) / sigma
        velocity = velocity.asType(sample.dtype)
        return (sample.asType(.float32) + velocity.asType(.float32) * dt)
            .asType(sample.dtype)
    }

    /// Put the conditioned tokens back where the model overwrote them.
    ///
    /// The identity on a text-to-audio-video render — the mask is all ones, the clean
    /// latent all zeros. It is here because it is the seam image conditioning enters
    /// through, and a sampler that omits it works perfectly right up until the first
    /// keyframe, at which point the conditioned frames drift because the model's opinion
    /// about them is being kept.
    ///
    /// The working dtype is the **mask's**, not fp32: only `clean` is cast and
    /// `denoised * mask` promotes, so an fp32 mask computes the blend in fp32 and a bf16
    /// one computes it in bf16. Forcing fp32 here would be the more accurate arithmetic
    /// and the less faithful one.
    public static func postProcess(denoised: MLXArray, denoiseMask: MLXArray?,
                                   cleanLatent: MLXArray?) -> MLXArray {
        guard let mask = denoiseMask, let clean = cleanLatent else { return denoised }
        return (denoised * mask
                + clean.asType(.float32) * (1.0 - mask)).asType(denoised.dtype)
    }

    /// One step's two outputs: where the latents landed, and the prediction that took
    /// them there.
    ///
    /// The second is not a diagnostic. A skipped step reuses the previous **prediction** —
    /// not the velocities that produced it — so a loop that does not carry it forward
    /// cannot reproduce a render with `skipStep` set. It is the guider's raw output, taken
    /// **before** the conditioning blend, because the blend is re-applied on the way out
    /// of every step and applying it twice to a reused prediction is not the identity.
    public struct StepResult {
        public var latents: Streams
        /// The guidance combination's output, before the conditioning blend.
        public var denoised: Streams
    }

    /// Which diffusion step the loop takes.
    ///
    /// Resolved per stage rather than per pipeline: the distilled two-stage runs the
    /// ancestral step with `eta = 1, sNoise = 1` for stage 1 and the deterministic step
    /// for stage 2. It is a per-run choice, so it is a parameter and not a property of the
    /// sampler.
    public enum Stepper: Sendable, Equatable {
        case euler
        /// `seed` drives the renoise draws only, offset by 10000 from the render's seed —
        /// see ``ancestralStep(sample:denoised:sigmas:index:noise:eta:sNoise:)`` for why
        /// the offset is load-bearing rather than decorative.
        case ancestral(eta: Float, sNoise: Float, seed: UInt64)
    }

    /// One step, from the per-pass velocities to the next latent.
    ///
    /// This is the whole sampler minus the transformer: the velocities can come from a
    /// live DiT or straight out of recorded `proj_out.out` tensors, and the arithmetic in
    /// between is the same either way.
    ///
    /// - Parameters:
    ///   - enabled: false for a stream whose guider skips this step. That stream ignores
    ///     `velocities` entirely and steps on `lastDenoised` instead. It still steps: a
    ///     skipped stream is a stale *prediction*, not a frozen latent.
    ///   - lastDenoised: the previous step's `StepResult.denoised`. Required when either
    ///     stream is disabled.
    public func step(index: Int, sigmas: [Float], latents: Streams,
                     velocities: (video: [Pass: MLXArray], audio: [Pass: MLXArray]),
                     denoiseMask: Streams? = nil,
                     cleanLatent: Streams? = nil,
                     enabled: (video: Bool, audio: Bool) = (true, true),
                     lastDenoised: Streams? = nil,
                     stepper: Stepper = .euler,
                     stepNoise: Streams? = nil,
                     cfgPP: Float = 0) throws -> StepResult {
        let sigma = sigmas[index]
        let videoParams = video.effective(atSigma: sigma)
        let audioParams = audio.effective(atSigma: sigma)
        // The x0 conversion uses the plain sigma, not the 1000x-scaled one the AdaLN
        // modules consume.
        let x0Sigma = sigma

        func advance(_ latent: MLXArray, _ passes: [Pass: MLXArray],
                     _ params: GuidanceParams, _ stream: String,
                     _ mask: MLXArray?, _ clean: MLXArray?,
                     _ isEnabled: Bool, _ last: MLXArray?,
                     _ noise: MLXArray?) throws
            -> (next: MLXArray, denoised: MLXArray) {
            let combined: MLXArray
            // The unconditional x0, kept for CFG++. Only ever non-nil when a `.unconditional`
            // pass actually ran, which is exactly when `cfgScale != 1` — so a recipe with
            // guidance off cannot reach the CFG++ branch, and there is nothing sensible for
            // it to mean there anyway.
            var uncondX0: MLXArray?
            if isEnabled {
                let timesteps = Self.timesteps(sigma: x0Sigma, mask: mask,
                                               tokens: latent.dim(1))
                var predictions: [Pass: MLXArray] = [:]
                for (pass, velocity) in passes {
                    predictions[pass] = Self.denoised(latent: latent, velocity: velocity,
                                                      timesteps: timesteps)
                }
                uncondX0 = predictions[.unconditional]
                combined = try combine(predictions, params: params, stream: stream)
            } else {
                // There is no previous prediction until a step has run, and step 0 is
                // never skipped for any `skipStep`, so this branch is unreachable in
                // practice. It refuses rather than inventing a first prediction.
                guard let last else {
                    throw Failure.missingPass(.conditional, stream: stream)
                }
                combined = last
            }
            let processed = Self.postProcess(denoised: combined, denoiseMask: mask,
                                             cleanLatent: clean)
            // The CFG++ direction, blended and then put through the *same* conditioning
            // seam as the destination. Skipping the blend here would let a frozen token
            // take its direction from an unprocessed prediction and drift out from under
            // the mask on the deterministic path, which has no second blend to correct it.
            //
            // `nil` at `cfgPP == 0` rather than a blend that happens to be the identity:
            // the steppers branch on nil to keep the standard path bit-identical, and
            // `(1 - 0) * a + 0 * b` is not bit-identical to `a`.
            let direction: MLXArray? = {
                guard cfgPP > 0, let uncondX0 else { return nil }
                let blended = cfgPP >= 1
                    ? uncondX0
                    : (1 - cfgPP) * combined.asType(.float32)
                        + cfgPP * uncondX0.asType(.float32)
                return Self.postProcess(denoised: blended.asType(combined.dtype),
                                        denoiseMask: mask, cleanLatent: clean)
            }()
            var next: MLXArray
            switch stepper {
            case .euler:
                next = try eulerStep(sample: latent, denoised: processed, sigmas: sigmas,
                                     index: index, directionDenoised: direction)
            case let .ancestral(eta, sNoise, _):
                next = try ancestralStep(sample: latent, denoised: processed, sigmas: sigmas,
                                         index: index, noise: noise, eta: eta, sNoise: sNoise,
                                         directionDenoised: direction)
                // **The blend goes on TWICE on the ancestral path, and only there.**
                //
                // The ancestral loop blends once into the x0 prediction (`processed`
                // above) and then again into the *stepped* latent. The deterministic loop
                // blends and steps, full stop — so this must not leak onto `.euler`, and
                // `AncestralConditioningTests` holds the Euler path bit-identical to
                // blend-once-then-step, over every token of both streams.
                //
                // Why it matters, and why it stays invisible until it doesn't: with an
                // all-ones mask the second application is the identity, so an
                // unconditioned render cannot see it at all. The moment a token is frozen
                // it becomes load-bearing — `ancestralStep` renoises *every* token, frozen
                // ones included, so without this the conditioned tokens carry a fresh draw
                // of noise into the next step's forward while their timestep still says 0.
                // That is the "conditioning looks applied but isn't" failure in its
                // quietest form: the render completes, frame 0 still shows the image
                // because the *next* step's blend overwrites it again, and the model was
                // told something false at every step in between.
                //
                // Two conditions:
                //
                // * `eta > 0`. At `eta == 0` the step is a plain Euler step and takes no
                //   second blend.
                // * not the terminal step. At `sigmas[index + 1] == 0` the once-blended
                //   prediction *is* the final latent — `ancestralStep` short-circuits to
                //   `processed` — so blending again would apply the mask twice to it,
                //   which is NOT the identity at a fractional strength.
                if eta > 0, sigmas[index + 1] != 0 {
                    next = Self.postProcess(denoised: next, denoiseMask: mask,
                                            cleanLatent: clean)
                }
            }
            return (next, combined)
        }

        let v = try advance(latents.video, velocities.video, videoParams, "video",
                            denoiseMask?.video, cleanLatent?.video,
                            enabled.video, lastDenoised?.video, stepNoise?.video)
        let a = try advance(latents.audio, velocities.audio, audioParams, "audio",
                            denoiseMask?.audio, cleanLatent?.audio,
                            enabled.audio, lastDenoised?.audio, stepNoise?.audio)
        return StepResult(latents: Streams(video: v.next, audio: a.next),
                          denoised: Streams(video: v.denoised, audio: a.denoised))
    }

    // MARK: - The loop

    /// What produces one guidance pass's velocity. The transformer, or a fixture.
    ///
    /// Returns the **velocity**, not the denoised prediction: the transformer predicts
    /// velocity and the x0 conversion wraps it, so a denoiser that returned x0 would be off
    /// by the whole conversion.
    public protocol Denoiser {
        /// - Parameters:
        ///   - pass: which of the four the caller wants. A denoiser that cannot serve one
        ///     (no STG seam, say) must throw rather than substitute the conditional pass.
        ///   - sigma: the step's noise level, unscaled. Multiplying by
        ///     `FlowSchedule.timestepScaleMultiplier` is the transformer's entry, not the
        ///     sampler's business — but it happens in the sampler's fp32, which is why
        ///     `Sampler.scaledTimestep` lives here and not in `DiTForward`.
        ///   - denoiseMask: the run's mask, or `nil` when nothing is conditioned. Passed
        ///     per call rather than held on the denoiser deliberately: the sampler already
        ///     owns this mask for `postProcess`, and a second copy on the denoiser is two
        ///     places that can disagree about which tokens are frozen. They must be the
        ///     same mask, so there is only one.
        func velocity(pass: Sampler.Pass, step: Int, sigma: Float,
                      latents: Sampler.Streams,
                      denoiseMask: Sampler.Streams?) throws -> Sampler.Streams
    }

    /// What a completed run reports, beyond the latents.
    public struct Trajectory {
        /// The final latents, still patchified.
        public var latents: Streams
        /// Every step's starting latent, `steps + 1` entries, the last equal to `latents`.
        public var states: [Streams]
        /// Transformer calls actually made. Compare against `passCount(steps:)` — they
        /// must agree, and the difference between guidance settings is only visible in
        /// exactly this unit.
        public var passes: Int
        public var sigmas: [Float]
    }

    /// The denoising loop, over `sigmas[..<steps]`.
    ///
    /// The last iteration steps to `sigma = 0`, which is where the plain Euler update
    /// collapses to the denoised prediction. No special case: adding one would change the
    /// final latent.
    /// - Parameters:
    ///   - explicitSigmas: a fixed schedule, overriding `schedule.sigmas(steps:)`. The
    ///     distilled stages pass `FlowSchedule.distilledStage1` / `distilledStage2`, which
    ///     are constants and must not be run through the shift computation. `steps` is
    ///     ignored when this is supplied — the list decides, `count - 1`.
    ///   - stepper: `.euler`, or `.ancestral` for the distilled stage 1.
    ///   - retainHistory: keep every step's starting latent in ``Trajectory/states``.
    ///     Suites read that array. A live render never does, and 31 copies of the
    ///     production latent are 31 copies that only grow for the length of the run.
    public func run(steps: Int, initial: Streams, denoiser: Denoiser,
                    denoiseMask: Streams? = nil, cleanLatent: Streams? = nil,
                    explicitSigmas: [Float]? = nil,
                    stepper: Stepper = .euler,
                    retainHistory: Bool = true,
                    evalCadence: Int = 1,
                    cfgPP: Float = 0,
                    observer: ((Int, Streams) throws -> Void)? = nil) throws -> Trajectory {
        let sigmas: [Float]
        let stepCount: Int
        if let explicitSigmas {
            guard explicitSigmas.count >= 2 else { throw Failure.steps(explicitSigmas.count) }
            sigmas = explicitSigmas
            stepCount = explicitSigmas.count - 1
        } else {
            guard steps >= 1 else { throw Failure.steps(steps) }
            sigmas = try schedule.sigmas(steps: steps)
            stepCount = steps
        }
        // One key for the whole run, split per step, so the renoise draws are a
        // deterministic function of the seed and the step index and nothing else. Contract 1
        // applies as ever: reproducible here, and not comparable to any other RNG.
        var ancestralKey: MLXArray?
        if case let .ancestral(_, _, seed) = stepper { ancestralKey = MLX.key(seed) }
        var latents = initial
        var states: [Streams] = retainHistory ? [initial] : []
        var passes = 0
        // The previous step's prediction, carried across steps. **The prediction, not the
        // velocities that made it.** An earlier draft of this loop cached the four per-pass
        // velocities and re-ran the whole x0 conversion and guidance combination against
        // the *current* latent and the *current* sigma on a skipped step. That produces a
        // plausible tensor from stale inputs; a skipped step must reuse the prediction
        // verbatim and step the current latent with it.
        var lastDenoised: Streams?

        for index in 0..<stepCount {
            let plan = plan(step: index, sigma: sigmas[index])
            var velocities: (video: [Pass: MLXArray], audio: [Pass: MLXArray]) = ([:], [:])
            for pass in plan.passes {
                let out = try denoiser.velocity(pass: pass, step: index,
                                                sigma: sigmas[index], latents: latents,
                                                denoiseMask: denoiseMask)
                velocities.video[pass] = out.video
                velocities.audio[pass] = out.audio
                passes += 1
            }
            // Drawn before the step and only when the stepper asks for it, so the euler path
            // consumes nothing from the key and stays bit-identical to before this existed.
            var stepNoise: Streams?
            if let key = ancestralKey {
                let split = MLX.split(key: key, into: 3)
                ancestralKey = split[0]
                stepNoise = Streams(
                    video: MLX.normal(latents.video.shape, dtype: .float32, key: split[1])
                        .asType(latents.video.dtype),
                    audio: MLX.normal(latents.audio.shape, dtype: .float32, key: split[2])
                        .asType(latents.audio.dtype))
            }
            let result = try step(index: index, sigmas: sigmas, latents: latents,
                                  velocities: velocities, denoiseMask: denoiseMask,
                                  cleanLatent: cleanLatent,
                                  enabled: (plan.videoEnabled, plan.audioEnabled),
                                  lastDenoised: lastDenoised,
                                  stepper: stepper, stepNoise: stepNoise,
                                  cfgPP: cfgPP)
            latents = result.latents
            lastDenoised = result.denoised
            // A step of lazy graph is two live streams; forcing here keeps the working set
            // to one step rather than the whole trajectory. The predictions are forced too
            // — a skipped step keeps one alive across iterations, and an unforced one
            // would hold the entire producing graph with it.
            //
            // `evalCadence` trades that bound for fewer sync points. Each force is a barrier
            // where the GPU drains and the CPU rebuilds, and this port measures roughly
            // 0.4 s of per-forward cost that does not scale with tokens — 8% of a forward at
            // draft resolution. Forcing every n-th step amortises it across n steps.
            //
            // **It is bounded and not free.** n steps of graph stay live instead of one, so
            // the working set grows with the cadence. The default is 1, which is the
            // behaviour this loop has always had; raising it is a decision made against a
            // measured memory headroom rather than a default anybody inherits. The last step
            // always forces, because the caller is handed its result.
            let isLast = index == stepCount - 1
            if isLast || evalCadence <= 1 || (index + 1) % evalCadence == 0 {
                MLX.eval(latents.video, latents.audio,
                         lastDenoised!.video, lastDenoised!.audio)
            }
            // History forces regardless: appending unforced latents would hold every
            // producing graph alive at once, which is the opposite of what a cadence is for.
            //
            // The **observer** deliberately does not force. The one this port passes reports
            // a step number and never touches the latent, so forcing on its behalf would
            // cancel the cadence at every call site that logs progress — which is all of
            // them. An observer that does read the latent must force it itself.
            if retainHistory {
                MLX.eval(latents.video, latents.audio)
                states.append(latents)
            }
            try observer?(index, latents)
        }
        return Trajectory(latents: latents, states: states, passes: passes, sigmas: sigmas)
    }

    // MARK: - Layout

    /// `[1, T, C] -> [1, C, F, H, W]`. `VideoLatentPatchifier(patch_size=1).unpatchify`.
    ///
    /// A transpose and a reshape, and the token order it assumes — frame, then y, then x
    /// with x fastest — is the same one `DiTForward.videoPositions` builds the rotary grid
    /// in. Unpatchifying in the other order hands the VAE a correctly shaped, spatially
    /// transposed latent, which decodes to something that looks like a video.
    public static func unpatchifyVideo(_ x: MLXArray, latentFrames: Int, height: Int,
                                       width: Int) throws -> MLXArray {
        let expected = latentFrames * height * width
        guard x.dim(1) == expected else {
            throw Failure.tokenCount(stream: "video", got: x.dim(1), expected: expected)
        }
        return x.transposed(0, 2, 1)
            .reshaped([x.dim(0), x.dim(2), latentFrames, height, width])
    }

    /// `[1, T, C * F] -> [1, C, T, F]`. `AudioPatchifier(patch_size=1).unpatchify`.
    public static func unpatchifyAudio(_ x: MLXArray, channels: Int) -> MLXArray {
        x.reshaped([x.dim(0), x.dim(1), channels, x.dim(2) / channels])
            .transposed(0, 2, 1, 3)
    }

    // MARK: - Into token space

    /// `[1, C, F, H, W] -> [1, T, C]`. The exact inverse of ``unpatchifyVideo``.
    ///
    /// **What this is for.** Everything in the sampler works in token space, but a VAE
    /// encoder emits a latent in VAE space, and conditioning needs the two to meet:
    /// `Sampler.postProcess` blends a `cleanLatent` against the denoised stream, and both
    /// operands have to be `[B, T, C]`. Without this, an encoded image cannot become a
    /// conditioning latent at all.
    ///
    /// **The token order is the load-bearing part, and it is not free to choose.** Frame,
    /// then y, then x with x fastest — the same order ``unpatchifyVideo`` assumes and the
    /// same one `DiTForward.videoPositions` builds the rotary grid in. Its counterpart's
    /// doc records what the wrong order costs on the way out: a correctly shaped,
    /// spatially transposed latent that decodes to something that still looks like a
    /// video. Going *in*, the same mistake places conditioned content at the wrong tokens,
    /// so a conditioned frame would be honoured — at scrambled positions, at full
    /// strength, with no shape error anywhere.
    ///
    /// Guarding that with a test is cheap and exact: this and ``unpatchifyVideo`` compose
    /// to the identity, bit for bit, in both directions. A round trip needs no tolerance
    /// and no recorded answer, which makes it the rare check that is both free and
    /// complete.
    public static func patchifyVideo(_ x: MLXArray) throws -> MLXArray {
        guard x.ndim == 5 else {
            throw Failure.latentRank(stream: "video", got: x.shape, expected: "[B, C, F, H, W]")
        }
        let b = x.dim(0), c = x.dim(1)
        let tokens = x.dim(2) * x.dim(3) * x.dim(4)
        return x.reshaped([b, c, tokens]).transposed(0, 2, 1)
    }

    /// `[1, C, T, F] -> [1, T, C * F]`. The exact inverse of ``unpatchifyAudio``.
    ///
    /// The audio stream packs `C * F` per token — 8 channels x 16 features = 128 at
    /// production — which is why the channel axis here is a product and why
    /// ``unpatchifyAudio`` needs `channels` told to it while this does not: the split is
    /// recoverable from the input's own shape on the way in, and not on the way out.
    public static func patchifyAudio(_ x: MLXArray) throws -> MLXArray {
        guard x.ndim == 4 else {
            throw Failure.latentRank(stream: "audio", got: x.shape, expected: "[B, C, T, F]")
        }
        let b = x.dim(0), c = x.dim(1), t = x.dim(2), f = x.dim(3)
        return x.transposed(0, 2, 1, 3).reshaped([b, t, c * f])
    }
}

// MARK: - The schedule

/// The rectified-flow sigma schedule.
///
/// Computed in `Float` rather than `Double`, and the last two bits of it are load-bearing:
/// they are the difference between the timestep 999.999755859375 and a plausible, wrong
/// 1000.0.
///
/// Three details decide that, and none is visible in the formula:
///
/// * **The ramp is not `start + i * step` throughout.** The first half walks forwards from
///   `start` and the second half backwards from `end`, with the step in float32. At
///   `count = 4` that is `0x1.555554p-1` at index 1 where the naive expression gives
///   `0x1.555556p-1`. See ``linspace(from:to:count:)``.
/// * **The division is a reciprocal and then a multiply.** Taken that way,
///   `exp(shift) / (exp(shift) + 0)` is `0x1.fffffep-1`, one ulp *below* one — and the
///   stretch then carries that ulp into the first sigma. Divide the other way and the
///   first timestep is exactly 1000.0, which is not the value this schedule produces.
/// * **`tokens` is 4096, not the render's token count.** The shift anchors at the default
///   token count, so the schedule does not depend on the resolution at all. Passing the
///   real 240 tokens is the helpful mistake: it produces a different schedule for every
///   shape, all of them wrong.
///
/// Measured against `adaln_single.in` at every step: **0 ulps**.
public struct FlowSchedule: Sendable, Equatable {

    /// The shift anchors. Token counts, and the second doubles as the default token count
    /// a schedule computes at when no latent is supplied.
    public static let baseShiftAnchor = 1024
    public static let maxShiftAnchor = 4096

    /// `timestep_scale_multiplier`, 1000 on this checkpoint.
    ///
    /// Declared on the schedule rather than on `DiTForward` because the schedule is what
    /// it multiplies, and `FlowScheduleTests` checks the product against
    /// `adaln_single.in`. That is a placement choice: numerically it is the same single
    /// fp32 multiply of an fp32 sigma by an integer wherever it happens, which is why
    /// `adaln_single.in` lands on the recorded value from either side.
    ///
    /// The x0 conversion uses the **unscaled** sigma regardless — the per-token timesteps
    /// are built before any scaling.
    public static let timestepScaleMultiplier = 1000

    // MARK: - The distilled schedules, which are not computed

    /// Stage 1 of the distilled two-stage pipeline, 8 steps.
    ///
    /// **Literal, not derived.** Everything else in this type computes a shifted schedule
    /// from a token count; these are fixed constants. Running them through
    /// ``sigmas(steps:)`` would produce a different schedule and a different render, so the
    /// fixed entry point exists precisely so that the two cannot be confused.
    ///
    /// Note the flat head — `1.0, 0.99375, 0.9875, 0.98125, 0.975` are within `6.3e-03` of
    /// each other. Five of the eight steps barely move sigma; the trajectory does its work
    /// in the last three. That is also where `ancestralStep`'s `clamp(min: 0)` earns its
    /// keep, since the renoise variance is near zero across that head.
    ///
    /// `steps` is `count - 1`: nine sigmas, eight steps, ending at 0.
    public static let distilledStage1: [Float] = [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
    ]

    /// Stage 2, 3 steps, deterministic Euler.
    ///
    /// Its head `0.909375` is also stage 2's noise scale: the upscaled stage-1 latent is
    /// re-noised to exactly this level on the way in, which is what makes the refine a
    /// continuation rather than a fresh render. Same three values appear as the tail of
    /// ``distilledStage1``, which is not a coincidence — stage 2 re-runs the part of the
    /// schedule where the trajectory actually moves, at full resolution.
    public static let distilledStage2: [Float] = [0.909375, 0.725, 0.421875, 0.0]

    public var maxShift: Double
    public var baseShift: Double
    public var terminal: Double
    public var stretch: Bool
    /// The token count the shift anchors at. See the type's documentation for why this is
    /// not the render's token count.
    public var tokens: Int

    public init(maxShift: Double = 2.05, baseShift: Double = 0.95,
                terminal: Double = 0.1, stretch: Bool = true,
                tokens: Int = FlowSchedule.maxShiftAnchor) {
        self.maxShift = maxShift
        self.baseShift = baseShift
        self.terminal = terminal
        self.stretch = stretch
        self.tokens = tokens
    }

    /// A float32 linear ramp, walked from both ends:
    ///
    /// ```
    /// step    = (end - start) / (count - 1)
    /// halfway = count / 2
    /// data[i] = i < halfway ? start + step * i : end - step * (count - i - 1)
    /// ```
    ///
    /// The backwards half is not an optimisation — it pins the endpoint exactly — and
    /// computing only the forwards half puts index 2 of a 4-element ramp one ulp out.
    public static func linspace(from start: Float, to end: Float, count: Int) -> [Float] {
        precondition(count >= 1, "linspace needs at least one point")
        if count == 1 { return [start] }
        let step = (end - start) / Float(count - 1)
        let halfway = count / 2
        return (0..<count).map { i in
            i < halfway ? start + step * Float(i) : end - step * Float(count - 1 - i)
        }
    }

    /// `steps + 1` sigmas, descending, ending at 0.
    ///
    /// At `steps == 1` the schedule is `[0.1, 0.0]`, not `[1.0, 0.0]`: the stretch
    /// normalises the *last* non-zero sigma to `terminal`, and with one step that is also
    /// the first. Worth knowing before reading a one-step render as "start from pure
    /// noise" — it does not.
    public func sigmas(steps: Int) throws -> [Float] {
        guard steps >= 1 else { throw Sampler.Failure.steps(steps) }
        var sigmas = Self.linspace(from: 1.0, to: 0.0, count: steps + 1)

        let mm = (maxShift - baseShift)
            / Double(Self.maxShiftAnchor - Self.baseShiftAnchor)
        let b = baseShift - mm * Double(Self.baseShiftAnchor)
        // The exponential is evaluated in double and then rounded to float32, and it is the
        // rounded value that participates in every operation below.
        let shift = Float(Foundation.exp(Double(tokens) * mm + b))

        for i in sigmas.indices where sigmas[i] != 0 {
            let denominator = shift + (1.0 / sigmas[i] - 1.0)
            // Reciprocal then multiply: writing `shift / denominator` here loses the ulp
            // that the recorded timestep is made of.
            sigmas[i] = (1.0 / denominator) * shift
        }

        if stretch {
            let live = sigmas.indices.filter { sigmas[$0] != 0 }
            if !live.isEmpty {
                let oneMinusZ = live.map { 1.0 - sigmas[$0] }
                let scale = oneMinusZ[oneMinusZ.count - 1] / Float(1.0 - terminal)
                for (k, i) in live.enumerated() { sigmas[i] = 1.0 - oneMinusZ[k] / scale }
            }
        }
        return sigmas
    }

    /// `sigma * timestep_scale_multiplier`, in the sampler's float32.
    ///
    /// The quantity `adaln_single.in` records, and the one `DiTForward.Sampling.uniform`
    /// takes. Rounding order matters: `Float(sigma) * 1000` is not `Float(sigma * 1000)`
    /// evaluated in double, and the recorded value is the first.
    public static func scaledTimestep(_ sigma: Float) -> Float {
        sigma * Float(timestepScaleMultiplier)
    }
}

// MARK: - Guidance

/// The guidance parameters, per stream.
///
/// One for video and one for audio, differing in production only in `cfgScale` — 3.0
/// against 7.0. Swapping them costs 2.15e-01. They stay two separate values rather than
/// one shared struct because the *audio* one also drives the A2V gate's AdaLN (see
/// `DiTForward.Sampling`), and a design that shared a single params object would make that
/// impossible to express.
public struct GuidanceParams: Sendable, Equatable {

    /// Neutral at 1.0. Costs the `.unconditional` pass when it is not.
    public var cfgScale: Double
    /// Neutral at **0.0**, not 1.0 — see `Sampler.combine`. Costs the `.perturbed` pass.
    public var stgScale: Double
    /// Which blocks the STG pass replaces self-attention in. `[28]` in production.
    ///
    /// **An empty list does NOT disable STG.** The perturbed pass is scheduled on
    /// ``runsPerturbed`` — that is, on `stgScale` alone — and `stgBlocks` only decides
    /// where the perturbation lands. An empty list therefore runs a full-price pass that
    /// perturbs nothing and returns the conditional result.
    public var stgBlocks: [Int]
    /// Neutral at 0.0. Costs nothing: it is applied to the combination, not to a pass.
    public var rescaleScale: Double
    /// Neutral at 1.0. Costs the `.modality` pass when it is not.
    public var modalityScale: Double
    /// 0 runs every step. `n` runs one step in `n + 1` and reuses the previous prediction
    /// for the rest.
    public var skipStep: Int
    /// Percent window for the STG pass. `p = 0` at sigma 1 (start of sampling), `p = 1` at
    /// sigma 0, mapped as sigma in `[1 - end, 1 - start]`. The default `[0, 1]` is every
    /// sampled sigma.
    public var stgStartPercent: Double
    public var stgEndPercent: Double
    /// Same mapping, for the isolated-modality pass.
    public var modalityStartPercent: Double
    public var modalityEndPercent: Double

    public init(cfgScale: Double = 1.0, stgScale: Double = 0.0, stgBlocks: [Int] = [],
                rescaleScale: Double = 0.0, modalityScale: Double = 1.0,
                skipStep: Int = 0,
                stgStartPercent: Double = 0, stgEndPercent: Double = 1,
                modalityStartPercent: Double = 0, modalityEndPercent: Double = 1) {
        self.cfgScale = cfgScale
        self.stgScale = stgScale
        self.stgBlocks = stgBlocks
        self.rescaleScale = rescaleScale
        self.modalityScale = modalityScale
        self.skipStep = skipStep
        self.stgStartPercent = stgStartPercent
        self.stgEndPercent = stgEndPercent
        self.modalityStartPercent = modalityStartPercent
        self.modalityEndPercent = modalityEndPercent
    }

    /// The guidance the `tiny_v5` fixture was recorded with.
    ///
    /// `rescaleScale` and `skipStep` have no CLI surface, so these cannot be recovered
    /// from a preset name — and every one of them is load-bearing at bit precision. A
    /// wrong `rescaleScale` alone lands 2.47e-01 out.
    public static let tinyV5Video = GuidanceParams(
        cfgScale: 3.0, stgScale: 1.0, stgBlocks: [28], rescaleScale: 0.7,
        modalityScale: 3.0, skipStep: 0)
    public static let tinyV5Audio = GuidanceParams(
        cfgScale: 7.0, stgScale: 1.0, stgBlocks: [28], rescaleScale: 0.7,
        modalityScale: 3.0, skipStep: 0)

    /// Tolerant equality: `relTol = 1e-9`, `absTol = 0.0`.
    ///
    /// Used rather than `==` because the three enable predicates below must agree with
    /// each other. They differ only for a scale a user typed as `1.0000000001`, which is
    /// exactly the case where "did the extra forward pass run?" should not depend on which
    /// predicate asked.
    static func isClose(_ a: Double, _ b: Double) -> Bool {
        abs(a - b) <= 1e-9 * max(abs(a), abs(b))
    }

    /// True when the `.unconditional` pass runs.
    public var runsUnconditional: Bool { !Self.isClose(cfgScale, 1.0) }
    /// True when the `.perturbed` pass runs. Note the neutral value is 0.
    ///
    /// `stgBlocks` is deliberately *not* consulted: the pass is scheduled on the scale
    /// alone and the block list is handed to the perturbation, so an empty list runs a
    /// pass that perturbs nothing — a full-price forward whose result equals the
    /// conditional one.
    public var runsPerturbed: Bool { !Self.isClose(stgScale, 0.0) }
    /// True when the `.modality` pass runs.
    public var runsIsolatedModality: Bool { !Self.isClose(modalityScale, 1.0) }

    /// True when this stream's `skipStep` skips the given step.
    public func skipsStep(_ step: Int) -> Bool {
        skipStep != 0 && step % (skipStep + 1) != 0
    }

    /// Window `[start, end]` is sigma in `[1 - end, 1 - start]`.
    public static func sigmaIsInsideWindow(_ sigma: Float, start: Double, end: Double) -> Bool {
        let lo = Float(1 - end), hi = Float(1 - start)
        return lo <= sigma && sigma <= hi
    }

    public func runsPerturbed(atSigma sigma: Float) -> Bool {
        runsPerturbed && Self.sigmaIsInsideWindow(sigma, start: stgStartPercent, end: stgEndPercent)
    }

    public func runsIsolatedModality(atSigma sigma: Float) -> Bool {
        runsIsolatedModality
            && Self.sigmaIsInsideWindow(sigma, start: modalityStartPercent, end: modalityEndPercent)
    }

    /// Scales as this step's combination should see them. A pass that ran only
    /// because the *other* stream was inside its window must not apply this
    /// stream's term.
    public func effective(atSigma sigma: Float) -> GuidanceParams {
        var p = self
        if !runsPerturbed(atSigma: sigma) { p.stgScale = 0 }
        if !runsIsolatedModality(atSigma: sigma) { p.modalityScale = 1 }
        return p
    }
}

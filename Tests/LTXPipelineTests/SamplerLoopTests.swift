// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

import LTXModules
@testable import LTXPipeline

/// The Euler step, the loop's state carrying, and Contract 1.
///
/// Everything here is either an exact identity or a structural observation rather than a
/// numerical comparison, which is what lets it run against a stub denoiser in
/// milliseconds. What it covers is the behaviour around the step: what the loop carries
/// between iterations, what happens at the terminal sigma, and the one thing the sampler
/// must refuse to do.
@Suite("Sampler loop")
struct SamplerLoopTests {

    static func array(_ v: [Float], _ shape: [Int]? = nil) -> MLXArray {
        MLXArray(v, shape ?? [1, v.count, 1])
    }

    static func values(_ x: MLXArray) -> [Float] {
        x.asType(.float32).flattened().asArray(Float.self)
    }

    /// A denoiser that returns a fixed velocity per pass and records what it was asked for.
    ///
    /// The seam `Sampler.Denoiser` exists for. It cannot be a stand-in for the DiT's
    /// numerics, but the loop's control flow — which passes run, in what order, at which
    /// sigma, against which latent — is entirely the sampler's business and is exactly
    /// what this observes.
    final class RecordingDenoiser: Sampler.Denoiser, @unchecked Sendable {
        struct Call: Equatable {
            let pass: Sampler.Pass
            let step: Int
            let sigma: Float
            /// Whether the loop handed this call a mask. Recorded rather than ignored
            /// because the defect that made the parameter exist was precisely that the
            /// denoiser never saw one — a stub that accepts and discards it would let
            /// that regress silently.
            var sawMask: Bool = false
        }
        private(set) var calls: [Call] = []
        let videoTokens: Int
        let audioTokens: Int
        /// Per-pass velocity, so the four passes are distinguishable in the output.
        let scale: Float

        init(videoTokens: Int, audioTokens: Int, scale: Float = 0.125) {
            self.videoTokens = videoTokens
            self.audioTokens = audioTokens
            self.scale = scale
        }

        func velocity(pass: Sampler.Pass, step: Int, sigma: Float,
                      latents: Sampler.Streams,
                      denoiseMask: Sampler.Streams?) throws -> Sampler.Streams {
            calls.append(Call(pass: pass, step: step, sigma: sigma,
                              sawMask: denoiseMask != nil))
            let v = scale * Float(pass.rawValue + 1)
            return Sampler.Streams(
                video: MLXArray([Float](repeating: v, count: videoTokens),
                                [1, videoTokens, 1]).asType(.float32),
                audio: MLXArray([Float](repeating: v, count: audioTokens),
                                [1, audioTokens, 1]).asType(.float32))
        }
    }

    static func initialLatents(video: Int = 8, audio: Int = 4) -> Sampler.Streams {
        Sampler.Streams(
            video: array((0..<video).map { Float($0 + 1) }),
            audio: array((0..<audio).map { Float($0 + 1) }))
    }

    // MARK: - The Euler step

    /// At the terminal sigma the plain Euler update collapses to the denoised prediction.
    ///
    /// The plain Euler step has **no** `sigmaNext == 0` special case — that belongs to the
    /// ancestral step, which returns the prediction outright. The formula does it by
    /// itself:
    ///
    /// ```
    /// x + ((x - denoised) / sigma) * (0 - sigma) = denoised
    /// ```
    ///
    /// Checked in fp32 at values where the division and the multiplication are exact, so
    /// the collapse is exact. Adding the special case would change nothing here and would
    /// change the last step of any schedule whose arithmetic is not exact — which is why
    /// this step must not have one.
    @Test("the last step collapses to the denoised prediction without a special case")
    func terminalSigmaCollapses() throws {
        let sampler = Sampler(video: GuidanceParams(), audio: GuidanceParams())
        let sample = Self.array([1, 2, 3, 4])
        let denoised = Self.array([0.5, 0.25, -1, 8])
        // A power of two, so (x - d) / sigma * sigma is exact.
        let sigmas: [Float] = [0.5, 0.0]
        let out = try sampler.eulerStep(sample: sample, denoised: denoised,
                                        sigmas: sigmas, index: 0)
        print("terminal step: \(Self.values(out)) against denoised "
            + "\(Self.values(denoised))")
        #expect(Self.values(out) == Self.values(denoised),
                Comment(rawValue: "the terminal step must land on the denoised prediction "
                        + "exactly; got \(Self.values(out))"))
    }

    /// The velocity conversion refuses `sigma == 0`, and so does this.
    ///
    /// It matters because the schedule's last entry *is* zero: a loop that ran one
    /// iteration too many would divide by it, and an implementation that quietly returned
    /// the sample instead would produce a render one step short with no error anywhere.
    @Test("a zero sigma is refused rather than divided by")
    func zeroSigmaThrows() {
        let sampler = Sampler(video: GuidanceParams(), audio: GuidanceParams())
        let x = Self.array([1, 2])
        #expect(throws: Sampler.Failure.self) {
            _ = try sampler.eulerStep(sample: x, denoised: x, sigmas: [0.0, 0.0], index: 0)
        }
    }

    /// `dt` is `sigma_next - sigma`, so it is negative and the latent moves toward the
    /// prediction.
    ///
    /// Stated as a direction rather than a value: with a denoised prediction above the
    /// sample, a descending schedule must move the sample *up*. A flipped sign is among the
    /// largest structural errors available here, and it is invisible to any check that only
    /// looks at magnitudes.
    @Test("the step moves the sample toward the prediction")
    func stepMovesTowardThePrediction() throws {
        let sampler = Sampler(video: GuidanceParams(), audio: GuidanceParams())
        let sample = Self.array([0, 0, 0, 0])
        let denoised = Self.array([1, 1, 1, 1])
        let out = Self.values(try sampler.eulerStep(sample: sample, denoised: denoised,
                                                    sigmas: [1.0, 0.5], index: 0))
        print("sample 0 -> \(out) with prediction 1 over sigma 1.0 -> 0.5")
        #expect(out.allSatisfy { $0 > 0 && $0 < 1 },
                Comment(rawValue: "a half step from 0 toward 1 must land strictly between "
                        + "them; got \(out). Values below 0 mean dt has the wrong sign."))
        #expect(out == [0.5, 0.5, 0.5, 0.5],
                Comment(rawValue: "halving the sigma must cover exactly half the distance "
                        + "on a rectified-flow schedule; got \(out)"))
    }

    // MARK: - Contract 1

    /// The initial latent is a tensor the caller supplies, and there is no other way in.
    ///
    /// `docs/FRAGILE_CONTRACTS.md` #1: a bf16 normal draw is not portable across
    /// frameworks or devices, neither in the generator's stream nor in the rounding, and
    /// the trajectory saturates within two steps — so a different starting point is a
    /// different render rather than a slightly different one.
    ///
    /// When the noise is **absent it throws**. There is no fallback to a seed, and this is
    /// the assertion that keeps it that way: a future convenience overload that "helpfully"
    /// generated noise would turn this test red rather than pass silently.
    @Test("an absent noise tensor is an error, never a seed")
    func contractOneHasNoSeedFallback() {
        #expect(throws: Pipeline.Failure.self,
                Comment(rawValue: "an absent noise tap must throw. There is deliberately "
                        + "no seed overload -- see FRAGILE_CONTRACTS #1 -- and a fallback "
                        + "that redrew from seed 0 would reproduce nothing.")) {
            _ = try Pipeline.recordedNoise([:])
        }
    }

    @Test("noise equality folds signed zero but no other bf16 difference")
    func signedZeroIsTheOnlyFoldedNoiseDifference() {
        let minusZero = MLXArray([-0.0 as Float, 1.0], [1, 2]).asType(.bfloat16)
        let plusZero = MLXArray([0.0 as Float, 1.0], [1, 2]).asType(.bfloat16)
        let changed = MLXArray([0.0 as Float, 1.0078125], [1, 2]).asType(.bfloat16)
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(minusZero, plusZero),
                Comment(rawValue: "signed zero is an algebraic representation detail and "
                        + "must not fail the noiser equality contract"))
        #expect(!Pipeline.noiseExactlyMatchesInitialLatent(plusZero, changed),
                Comment(rawValue: "the signed-zero accommodation must not weaken exact "
                        + "equality for non-zero bf16 values"))
    }

    @Test("recorded noise prefers the independent source draw")
    func independentDrawTakesPrecedenceOverLegacyPatchifyTap() throws {
        let videoDraw = MLXArray([2.0 as Float], [1, 1, 1]).asType(.bfloat16)
        let audioDraw = MLXArray([3.0 as Float], [1, 1, 1]).asType(.bfloat16)
        let noise = try Pipeline.recordedNoise([
            "noise.video": videoDraw,
            "noise.audio": audioDraw,
            "patchify_proj.in.call000": MLXArray([9.0 as Float], [1, 1, 1]),
            "audio_patchify_proj.in.call000": MLXArray([9.0 as Float], [1, 1, 1]),
        ])
        #expect(Self.values(noise.video) == [2.0])
        #expect(Self.values(noise.audio) == [3.0])
    }

    // MARK: - The loop

    /// The loop calls the denoiser once per planned pass, in the planned order, at the
    /// step's own sigma.
    @Test("the loop walks the plan in order at the schedule's sigmas")
    func loopWalksThePlan() throws {
        let params = GuidanceParams(cfgScale: 3, stgScale: 1, stgBlocks: [28],
                                    rescaleScale: 0.7, modalityScale: 3)
        let sampler = Sampler(video: params, audio: params)
        let denoiser = RecordingDenoiser(videoTokens: 8, audioTokens: 4)
        let steps = 3
        let trajectory = try sampler.run(steps: steps, initial: Self.initialLatents(),
                                         denoiser: denoiser)
        let sigmas = try FlowSchedule().sigmas(steps: steps)

        #expect(trajectory.passes == sampler.passCount(steps: steps),
                Comment(rawValue: "\(trajectory.passes) calls made, "
                        + "\(sampler.passCount(steps: steps)) planned"))
        #expect(denoiser.calls.count == trajectory.passes)
        #expect(trajectory.states.count == steps + 1,
                Comment(rawValue: "states carries every step's starting latent plus the "
                        + "end; got \(trajectory.states.count) for \(steps) steps"))

        for (index, call) in denoiser.calls.enumerated() {
            let step = index / 4
            #expect(call.step == step)
            #expect(call.pass.rawValue == index % 4,
                    Comment(rawValue: "call \(index) asked for \(call.pass); "
                            + "_guided_denoise appends cond, uncond, ptb, mod in that "
                            + "order and the golden's callNNN counter follows it"))
            #expect(call.sigma == sigmas[step],
                    Comment(rawValue: "call \(index) ran at sigma \(call.sigma), not "
                            + "\(sigmas[step]) -- the schedule is indexed by step, not by "
                            + "call"))
        }
    }

    /// `skipStep` 0 runs every step; `n` skips all but one step in `n + 1`.
    @Test("skipsStep reproduces should_skip_step")
    func skipsStepMatchesTheReference() {
        var params = GuidanceParams()
        params.skipStep = 0
        #expect((0..<6).allSatisfy { !params.skipsStep($0) },
                Comment(rawValue: "skip_step 0 must run every step"))
        params.skipStep = 1
        #expect((0..<6).map { params.skipsStep($0) }
                == [false, true, false, true, false, true])
        params.skipStep = 2
        #expect((0..<6).map { params.skipsStep($0) }
                == [false, true, true, false, true, true])
        // Step 0 is never skipped, at any setting. That is what makes the "nothing to
        // reuse" branch in `step` unreachable in a real render.
        #expect((0...8).allSatisfy { n in
            var p = GuidanceParams(); p.skipStep = n; return !p.skipsStep(0)
        })
    }

    /// A skipped step reuses the previous **prediction**, not the previous velocities.
    ///
    /// The distinction is the whole content of this test. What is cached is the guider's
    /// output — the previous prediction — and the *current* latent is stepped with it. A
    /// loop that instead cached the four per-pass velocities and re-ran the x0 conversion
    /// would use the current latent and the current sigma against stale velocities, which
    /// is a different prediction: the conversion is `x - v * sigma`, and both `x` and
    /// `sigma` have moved.
    ///
    /// Asserted as an exact identity: the skipped step's output must equal `eulerStep`
    /// applied to the *previous* step's prediction, and must **not** equal what a
    /// velocity-reusing loop would produce. Both halves, because only asserting the first
    /// would also pass if the two happened to coincide.
    @Test("a skipped step reuses the previous prediction, not the previous velocities")
    func skippedStepReusesThePrediction() throws {
        var params = GuidanceParams(cfgScale: 3, stgScale: 1, rescaleScale: 0,
                                    modalityScale: 3)
        params.skipStep = 1
        let sampler = Sampler(video: params, audio: params)
        let steps = 2
        let sigmas = try FlowSchedule().sigmas(steps: steps)
        let initial = Self.initialLatents()
        let denoiser = RecordingDenoiser(videoTokens: 8, audioTokens: 4)

        let trajectory = try sampler.run(steps: steps, initial: initial,
                                         denoiser: denoiser)
        // Step 1 is skipped by both streams, so no transformer call is made for it.
        #expect(trajectory.passes == sampler.plan(step: 0).passes.count,
                Comment(rawValue: "\(trajectory.passes) passes for 2 steps with "
                        + "skip_step 1; the skipped step must cost nothing"))
        #expect(denoiser.calls.allSatisfy { $0.step == 0 })

        // Rebuild step 0 by hand to recover its prediction, then step 1 from it.
        var velocities: (video: [Sampler.Pass: MLXArray],
                         audio: [Sampler.Pass: MLXArray]) = ([:], [:])
        for pass in sampler.plan(step: 0).passes {
            let out = try RecordingDenoiser(videoTokens: 8, audioTokens: 4)
                .velocity(pass: pass, step: 0, sigma: sigmas[0], latents: initial,
                          denoiseMask: nil)
            velocities.video[pass] = out.video
            velocities.audio[pass] = out.audio
        }
        let first = try sampler.step(index: 0, sigmas: sigmas, latents: initial,
                                     velocities: velocities)
        let reusingPrediction = try sampler.eulerStep(
            sample: first.latents.video, denoised: first.denoised.video,
            sigmas: sigmas, index: 1)

        // What the old, wrong loop did: keep the velocities and re-derive a prediction
        // from the *current* latent and the *current* sigma.
        let reusingVelocities = try sampler.step(index: 1, sigmas: sigmas,
                                                 latents: first.latents,
                                                 velocities: velocities).latents.video

        let got = Self.values(trajectory.latents.video)
        print("skipped step: got \(got)")
        print("  reusing the prediction: \(Self.values(reusingPrediction))")
        print("  reusing the velocities: \(Self.values(reusingVelocities))")
        #expect(got == Self.values(reusingPrediction),
                Comment(rawValue: "a skipped step must step the current latent with the "
                        + "PREVIOUS prediction. got \(got), expected "
                        + "\(Self.values(reusingPrediction))"))
        #expect(got != Self.values(reusingVelocities),
                Comment(rawValue: "the two reuse strategies coincide on this fixture, so "
                        + "the test above proves nothing. Choose inputs where "
                        + "to_denoised's dependence on the latent and the sigma is "
                        + "visible."))
    }

    /// A stream that skips while the other does not still pays for every pass.
    ///
    /// The step returns early only when **both** guiders skip. Otherwise the full pass list
    /// runs — one transformer call carries both streams — and the substitution happens per
    /// stream afterwards. So an audio-only skip costs the video stream nothing and saves
    /// nothing.
    @Test("one stream skipping does not save a pass")
    func perStreamSkipCostsFullPrice() throws {
        var video = GuidanceParams(cfgScale: 3, stgScale: 1, modalityScale: 3)
        var audio = video
        audio.skipStep = 1
        let sampler = Sampler(video: video, audio: audio)

        let plan = sampler.plan(step: 1)
        #expect(plan.passes.count == 4,
                Comment(rawValue: "step 1 skips audio only, so all four passes still run; "
                        + "got \(plan.passes.count)"))
        #expect(plan.videoEnabled && !plan.audioEnabled)
        #expect(!plan.isEmpty, "only one stream skipped, so the step is not empty")

        // And when both skip, the step costs nothing.
        video.skipStep = 1
        let both = Sampler(video: video, audio: audio)
        #expect(both.plan(step: 1).passes.isEmpty)
        #expect(both.plan(step: 1).isEmpty)
        #expect(both.passCount(steps: 4) == 2 * 4,
                Comment(rawValue: "four steps, half of them skipped, four passes each: "
                        + "\(both.passCount(steps: 4))"))
    }

    /// A skipped stream still advances; only its prediction is stale.
    ///
    /// The failure this guards against is the intuitive reading of "skip": freezing the
    /// latent. The latent is not frozen — the step runs unconditionally on whatever
    /// prediction it was handed — so a skipped stream keeps moving along the schedule using
    /// the previous step's opinion.
    @Test("a skipped stream keeps stepping on a stale prediction")
    func skippedStreamStillAdvances() throws {
        var video = GuidanceParams(cfgScale: 3, stgScale: 1, modalityScale: 3)
        var audio = video
        audio.skipStep = 1
        video.skipStep = 0
        let sampler = Sampler(video: video, audio: audio)
        let initial = Self.initialLatents()
        let denoiser = RecordingDenoiser(videoTokens: 8, audioTokens: 4)
        let trajectory = try sampler.run(steps: 2, initial: initial, denoiser: denoiser)

        let before = Self.values(trajectory.states[1].audio)
        let after = Self.values(trajectory.states[2].audio)
        print("audio skipped at step 1: \(before) -> \(after)")
        #expect(before != after,
                Comment(rawValue: "the skipped audio stream did not move. A skip reuses "
                        + "the prediction, it does not freeze the latent."))
    }

    /// A first step with nothing to reuse is refused rather than invented.
    ///
    /// Unreachable through `run` — step 0 is never skipped, at any `skipStep` — so this
    /// addresses `step` directly. Leaving the state untouched would be the permissive
    /// reading; refusing is the stricter one, and the one that cannot be mistaken for a
    /// render.
    @Test("a disabled stream with no previous prediction throws")
    func disabledWithoutAPredictionThrows() throws {
        let sampler = Sampler(video: GuidanceParams(), audio: GuidanceParams())
        let initial = Self.initialLatents()
        let sigmas = try FlowSchedule().sigmas(steps: 2)
        #expect(throws: Sampler.Failure.self) {
            _ = try sampler.step(index: 0, sigmas: sigmas, latents: initial,
                                 velocities: ([:], [:]),
                                 enabled: (false, false), lastDenoised: nil)
        }
    }

    // MARK: - Layout

    /// The token count is checked, not assumed.
    ///
    /// `unpatchifyVideo` is a transpose and a reshape, and a reshape with the wrong product
    /// is the kind of error that surfaces three stages later as a VAE complaint about a
    /// shape nobody chose.
    @Test("unpatchify refuses a latent whose token count does not match the grid")
    func unpatchifyChecksTheTokenCount() {
        let x = MLXArray([Float](repeating: 0, count: 24), [1, 24, 1])
        #expect(throws: Sampler.Failure.self) {
            _ = try Sampler.unpatchifyVideo(x, latentFrames: 2, height: 3, width: 5)
        }
        #expect(throws: Never.self) {
            _ = try Sampler.unpatchifyVideo(x, latentFrames: 2, height: 3, width: 4)
        }
    }

    /// The video unpatchify puts x fastest, then y, then frame.
    ///
    /// Checked by construction: a latent whose single channel counts 0, 1, 2, ... in token
    /// order must come back as a grid counting along the width first. A port that
    /// unpatchified in the other order hands the VAE a correctly shaped, spatially
    /// transposed latent — which decodes to something that looks like a video.
    @Test("the video unpatchify puts width fastest")
    func unpatchifyOrdering() throws {
        let frames = 2, height = 3, width = 4
        let tokens = frames * height * width
        let x = MLXArray((0..<tokens).map { Float($0) }, [1, tokens, 1])
        let grid = try Sampler.unpatchifyVideo(x, latentFrames: frames, height: height,
                                               width: width)
        #expect(grid.shape == [1, 1, frames, height, width])
        let flat = Self.values(grid)
        // With one channel the [B, C, F, H, W] traversal is the same linear order as the
        // token axis, so any transposition shows up as a permutation of this.
        #expect(flat == (0..<tokens).map { Float($0) },
                Comment(rawValue: "token order and grid order disagree: \(flat)"))
    }

    /// The audio unpatchify splits the token channel into (channels, mel bins).
    ///
    /// The unpatchify is `b t (c f) -> b c t f`, so the channel axis is the **outer** half
    /// of the token's channel dimension. Splitting it the other way is a
    /// reshape of the right size onto the wrong axes.
    @Test("the audio unpatchify treats channels as the outer half")
    func unpatchifyAudioOrdering() {
        let time = 2, channels = 3, bins = 4
        let x = MLXArray((0..<(time * channels * bins)).map { Float($0) },
                         [1, time, channels * bins])
        let out = Sampler.unpatchifyAudio(x, channels: channels)
        #expect(out.shape == [1, channels, time, bins])
        // Token 0's channel block 0 is bins 0...3, which must become [0, c=0, t=0, :].
        let first = Self.values(out.take(MLXArray([0]), axis: 1))
        #expect(Array(first.prefix(bins)) == [0, 1, 2, 3],
                Comment(rawValue: "channel 0 of frame 0 should be the first \(bins) "
                        + "values of the token; got \(Array(first.prefix(bins)))"))
    }

    // MARK: - Conditioning reaches the model, not just the latent

    /// A frozen token must arrive at the transformer labelled **clean**.
    ///
    /// This is the structural check contract 15 argues for, and it exists because the
    /// failure it catches is invisible to any comparison of the numbers: `postProcess`
    /// holds a conditioned token's *value* whether or not the model was told the token is a
    /// reference, so a render with inert conditioning completes, shows the supplied image
    /// at frame 0, and generates the remaining frames as if unconditioned. Measured on a
    /// 97-frame i2v render before the fix: frame 0 was the input still, frame 1 was an
    /// unrelated shot.
    ///
    /// What is available instead is an equality, which has no tolerance floor at all.
    @Test("a frozen token gets timestep 0 and a free one gets sigma")
    func maskedTimestepsMarkFrozenTokensClean() {
        let sigma: Float = 0.7
        // 1 = free, 0 = frozen: the first two of six tokens are conditioned.
        let mask = MLXArray([Float](arrayLiteral: 0, 0, 1, 1, 1, 1), [1, 6, 1])

        let masked = Sampler.timesteps(sigma: sigma, mask: mask, tokens: 6)
        #expect(masked.shape == [1, 6, 1])
        let values = masked.asType(.float32).asArray(Float.self)
        #expect(values == [0, 0, sigma, sigma, sigma, sigma],
                Comment(rawValue: "frozen tokens must reach the model at timestep 0 — that "
                        + "zero is the only signal that says the token is a clean "
                        + "reference. Got \(values)"))

        // Partial strength is a real mode: strength 0.6 means mask 0.4, and the token is
        // held at 0.4 * sigma rather than at either extreme.
        let partial = MLXArray([Float](arrayLiteral: 0.4, 1), [1, 2, 1])
        let scaled = Sampler.timesteps(sigma: sigma, mask: partial, tokens: 2)
            .asType(.float32).asArray(Float.self)
        #expect(scaled == [0.4 * sigma, sigma])
    }

    /// The unconditioned path must be untouched, bit for bit.
    ///
    /// `1.0 * x` is exact in fp32, so an all-ones mask is not merely close to the uniform
    /// constructor — it is the same tensor. That is what lets the masked path ship without
    /// altering a single unconditioned render.
    @Test("an all-ones mask is bit-identical to the uniform timesteps")
    func allOnesMaskIsTheUniformPath() {
        let sigma: Float = 0.909375
        let ones = MLXArray([Float](repeating: 1, count: 32), [1, 32, 1])
        let viaMask = Sampler.timesteps(sigma: sigma, mask: ones, tokens: 32)
            .asType(.float32).asArray(Float.self)
        let viaUniform = Sampler.uniformTimesteps(sigma: sigma, tokens: 32)
            .asType(.float32).asArray(Float.self)
        #expect(viaMask == viaUniform)

        // And a nil mask takes the uniform constructor outright.
        let viaNil = Sampler.timesteps(sigma: sigma, mask: nil, tokens: 32)
            .asType(.float32).asArray(Float.self)
        #expect(viaNil == viaUniform)
    }

    /// `Sampling.masked` carries the same rule into the AdaLN input.
    ///
    /// Two timesteps drive one forward pass — the unscaled one the x0 conversion reads and
    /// the 1000x one AdaLN reads — and fixing only the first would leave the transformer
    /// still being told every token is noisy.
    @Test("the AdaLN timestep is per token too, and uniform when nothing is conditioned")
    func adaLNTimestepFollowsTheMask() {
        let scaled = FlowSchedule.scaledTimestep(0.5)
        let videoMask = MLXArray([Float](arrayLiteral: 0, 1, 1), [1, 3, 1])

        let conditioned = DiTForward.Sampling.masked(
            scaledTimestep: scaled, videoMask: videoMask, audioMask: nil,
            videoTokens: 3, audioTokens: 2)
        #expect(conditioned.videoTimestep.shape == [3])
        #expect(conditioned.videoTimestep.asType(.float32).asArray(Float.self)
                == [0, scaled, scaled])
        // A nil audio mask means that stream is unconditioned and stays uniform.
        #expect(conditioned.audioTimestep.asType(.float32).asArray(Float.self)
                == [scaled, scaled])
        // Prompt and gate timesteps are not token-space and must not be masked.
        #expect(conditioned.videoPromptTimestep.asType(.float32).asArray(Float.self) == [scaled])
        #expect(conditioned.a2vGateTimestep.asType(.float32).asArray(Float.self) == [scaled])

        // A partial mask is NOT frozen: strength 0.9 leaves mask 0.1, which is not zero.
        let partial = DiTForward.Sampling.masked(
            scaledTimestep: scaled,
            videoMask: MLXArray([Float](arrayLiteral: 0.1, 1, 1), [1, 3, 1]),
            audioMask: nil, videoTokens: 3, audioTokens: 2)
        #expect(partial.videoPromptTimestep.asType(.float32).asArray(Float.self) == [scaled])

        let uniform = DiTForward.Sampling.uniform(
            scaledTimestep: scaled, videoTokens: 3, audioTokens: 2)
        let bothNil = DiTForward.Sampling.masked(
            scaledTimestep: scaled, videoMask: nil, audioMask: nil,
            videoTokens: 3, audioTokens: 2)
        #expect(bothNil.videoTimestep.asType(.float32).asArray(Float.self)
                == uniform.videoTimestep.asType(.float32).asArray(Float.self))
    }


    /// A wholly frozen stream zeroes its scalar sigma, not just its per-token timesteps.
    ///
    /// A wholly frozen stream has its scalar sigma zeroed outright, so that prompt AdaLN
    /// and the cross-modality gates agree with the frozen conditioning: with audio frozen,
    /// both `audio_adaln_single.in` and `audio_prompt_adaln_single.in` are `0.0`.
    ///
    /// The gate mapping is the half of this that is easy to get backwards, and it is
    /// invisible whenever both streams sit at the same sigma. The A2V gate follows the
    /// **audio** sigma, so freezing audio zeroes `a2v` and leaves `v2a` on the schedule.
    @Test("freezing a stream zeroes its prompt AdaLN and the gate it drives")
    func aFrozenStreamZeroesItsScalarSigma() {
        let scaled = FlowSchedule.scaledTimestep(0.75)
        // a2v: audio frozen outright, video free.
        let s = DiTForward.Sampling.masked(
            scaledTimestep: scaled,
            videoMask: MLXArray([Float](repeating: 1, count: 4), [1, 4, 1]),
            audioMask: MLXArray([Float](repeating: 0, count: 3), [1, 3, 1]),
            videoTokens: 4, audioTokens: 3)

        #expect(s.audioTimestep.asType(.float32).asArray(Float.self) == [0, 0, 0])
        #expect(s.audioPromptTimestep.asType(.float32).asArray(Float.self) == [0],
                Comment(rawValue: "a frozen stream's prompt AdaLN must be 0; leaving it on "
                        + "the schedule diverges at AdaLN with no other symptom"))
        #expect(s.a2vGateTimestep.asType(.float32).asArray(Float.self) == [0],
                Comment(rawValue: "the A2V gate is driven by the AUDIO sigma, so freezing "
                        + "audio must zero it"))
        // The video stream is untouched by the audio freeze.
        #expect(s.videoTimestep.asType(.float32).asArray(Float.self)
                == [scaled, scaled, scaled, scaled])
        #expect(s.videoPromptTimestep.asType(.float32).asArray(Float.self) == [scaled])
        #expect(s.v2aGateTimestep.asType(.float32).asArray(Float.self) == [scaled],
                Comment(rawValue: "the V2A gate follows the VIDEO sigma and must NOT be "
                        + "zeroed by an audio freeze"))
    }

}

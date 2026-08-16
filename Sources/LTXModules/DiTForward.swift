// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// The DiT body: patchify, then all 48 blocks in sequence.
///
/// `DiTBlock` runs one block from inputs it is handed; this is what runs one block's
/// output into the next, and composition is where the remaining structural mistakes live:
/// a loop that re-feeds the block input, or threads video while dropping audio, or reuses
/// block 0's weights for all 48, produces correctly shaped tensors at every depth.
///
/// It stops before the output head (`norm_out` / `proj_out`).
///
/// ## What a forward pass has to build
///
/// `DiTBlock.Conditioning` is a long struct because a block reads far more than its
/// latent. Three of its fields are constructed here:
///
/// * **The timestep projections.** Eight `AdaLayerNormSingle` modules turn one scalar
///   noise level into the block modulation, the prompt K/V modulation, the cross-modal
///   scale/shift and the cross-modal gate — per stream. `DiTModules.adaLayerNormSingle`
///   implements the module; `Sampling` is what decides which scalar each of the eight
///   sees, and that mapping is not symmetric (see `Sampling`).
/// * **The rotary embeddings.** Built in closed form over the patch grid —
///   `frequencyGrid`, `videoPositions`, `audioPositions`, `rotary` below.
/// * **The keyframe marker.** `patchify_proj.out` is *not* the block-0 input. See
///   `applyKeyframeEmbedding`.
///
/// The text conditioning is taken as an argument rather than built here; the encoder path
/// lives outside this type.
///
/// ## What is deliberately not here
///
/// **No attention mask.** The context mask the DiT receives is all ones over all 1024
/// conditioning tokens, and an all-ones keep-mask is a no-op in either the multiplicative
/// or the additive-log convention. The *tokenizer's* mask (11 non-zero of 1024) is a
/// different tensor and never reaches the DiT; feeding it in here would mask 1013 tokens
/// the model is meant to attend to.
///
/// **No perturbation masks.** An empty perturbation config is all-keep, and `DiTBlock`
/// skips the multiply rather than multiplying by ones, which is exact rather than
/// approximately exact. STG perturbs a later guidance branch.
public struct DiTForward {

    public var block: DiTBlock
    public var modules: DiTModules
    public let topology: TransformerTopology
    public let weights: [String: MLXArray]

    /// Overlay `lora` on every adapted projection the forward pass runs.
    ///
    /// Applied to the block (its attention and feed-forward) and to the shared modules.
    /// An adapter that names a weight this forward never reaches is not an error here —
    /// `LoRAOverlay.validate(againstBaseWeights:)` is where that is caught, before any
    /// compute is spent.
    public mutating func attach(lora: LoRAOverlay?) {
        block.attach(lora: lora)
        modules.lora = lora
    }

    /// Gradient checkpointing for the block loop — training only.
    ///
    /// Inference never wants this: it holds nothing for a backward pass, so recomputation
    /// would be pure cost. Training does, because without it all 48 blocks' activations stay
    /// resident until the backward consumes them, which is ~102 MB per video token against
    /// ~39 GB of weights that do not grow with sequence length.
    ///
    /// Checkpointing has to be applied **per block**, not once around the stack: one wrap
    /// around the whole forward recomputes it in a single trace and saves nothing.
    ///
    /// The two closures exist because of the transform's one hard rule — only arrays passed
    /// across the boundary receive gradients, and captured ones are silently treated as
    /// constants. ``parameters`` hands the block's trainable arrays *in*, and ``overlay``
    /// rebuilds the adapter from them as they arrive back inside the recomputed trace. MLX
    /// Python's `nn.checkpoint` is arranged the same way, passing `trainable_parameters()` as
    /// its first argument for exactly this reason.
    public struct Checkpointing {
        /// The traced arrays for block `index`, in the order ``overlay`` expects them back.
        public var parameters: (Int) -> [MLXArray]
        /// Rebuilds block `index`'s adapter from arrays that came back through the boundary.
        public var overlay: (Int, [MLXArray]) -> LoRAOverlay?

        public init(parameters: @escaping (Int) -> [MLXArray],
                    overlay: @escaping (Int, [MLXArray]) -> LoRAOverlay?) {
            self.parameters = parameters
            self.overlay = overlay
        }
    }

    /// Non-nil enables per-block gradient checkpointing. The per-block observers do not
    /// fire on the checkpointed path: they would run twice — once forward, once on
    /// recompute — and the second set are not the values the forward produced.
    public var checkpointing: Checkpointing?

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)
        case tokenCount(stream: String, latent: Int, geometry: Int)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .tokenCount(stream, latent, geometry):
                return "the \(stream) latent carries \(latent) tokens but the geometry "
                    + "describes \(geometry); the rotary grid would be built for the "
                    + "wrong positions"
            }
        }
    }

    /// Share an already-loaded weight map. A 21 B-parameter dictionary is not something to
    /// load twice, and `DiTBlock`, `DiTAttention` and `DiTModules` all read from this one.
    /// - Parameter attentionPath: where every attention in every block accumulates its
    ///   softmax. `.explicit` is the default; a *render* wants `.fused`, whose peak is
    ///   linear in the token count rather than quadratic. ``DiTAttention/AttentionPath``
    ///   carries the table.
    public init(weights: [String: MLXArray], topology: TransformerTopology,
                attentionPath: DiTAttention.AttentionPath = .explicit) {
        self.weights = weights
        self.topology = topology
        self.block = DiTBlock(weights: weights, topology: topology,
                              attentionPath: attentionPath)
        self.modules = DiTModules(weights: weights, topology: topology)
    }

    public init(checkpoint url: URL, topology: TransformerTopology,
                attentionPath: DiTAttention.AttentionPath = .explicit) throws {
        self.init(weights: try MLX.loadArrays(url: url), topology: topology,
                  attentionPath: attentionPath)
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[topology.prefix + name] else {
            throw Failure.missing(topology.prefix + name)
        }
        return w
    }

    // MARK: - Geometry

    /// Everything the rotary grid and the token counts are derived from.
    ///
    /// Nothing here is a property of the checkpoint — it is the render request plus the
    /// VAE's compression — so it is passed in rather than read from the weights. The
    /// defaults are the model's own configuration values; a wrong `theta` or `maxPos`
    /// builds a well-formed rotary grid at the wrong angles.
    public struct Geometry: Sendable {
        /// Pixel frames requested, e.g. 25. The latent frame count follows the lattice.
        public var frames: Int
        public var height: Int
        public var width: Int
        public var frameRate: Double
        /// Supplies the spatial and temporal scale factors and the audio latent count.
        public var latent: LatentGeometry

        /// `AudioPatchifier` constants. The audio positions are wall-clock **seconds**,
        /// so the mel hop and sample rate are part of the position grid, not just of the
        /// VAE.
        public var audioHopLength: Int
        public var audioSampleRate: Int
        public var audioLatentDownsample: Int

        /// `positional_embedding_theta`.
        public var theta: Double
        /// `positional_embedding_max_pos` — 20 *seconds*, 2048 pixels, 2048 pixels. The
        /// time axis and the space axes are divided by wildly different numbers, which is
        /// what makes seconds and pixels commensurate in one embedding.
        public var videoMaxPos: [Int]
        public var audioMaxPos: [Int]
        /// `max(positional_embedding_max_pos[0], audio_positional_embedding_max_pos[0])`.
        public var crossMaxPos: Int

        public init(frames: Int, height: Int, width: Int, frameRate: Double,
                    latent: LatentGeometry = LatentGeometry(),
                    audioHopLength: Int = 160, audioSampleRate: Int = 16_000,
                    audioLatentDownsample: Int = 4,
                    theta: Double = 10_000,
                    videoMaxPos: [Int] = [20, 2048, 2048],
                    audioMaxPos: [Int] = [20], crossMaxPos: Int = 20) {
            self.frames = frames
            self.height = height
            self.width = width
            self.frameRate = frameRate
            self.latent = latent
            self.audioHopLength = audioHopLength
            self.audioSampleRate = audioSampleRate
            self.audioLatentDownsample = audioLatentDownsample
            self.theta = theta
            self.videoMaxPos = videoMaxPos
            self.audioMaxPos = audioMaxPos
            self.crossMaxPos = crossMaxPos
        }

        public var latentFrames: Int { latent.latentFrames(forFrames: frames) }
        /// `h * w` — and also the number of tokens the keyframe marker applies to.
        public var tokensPerLatentFrame: Int {
            (height / latent.spatialScale) * (width / latent.spatialScale)
        }
        public var videoTokens: Int { latentFrames * tokensPerLatentFrame }

        /// An IC-LoRA reference clip carried **in context** beside the generated tokens.
        ///
        /// The reference's tokens are appended to the video stream, and the thing that
        /// makes them a reference rather than more picture is that they are *clean*: the
        /// noisy latent gets zeros in their slots, the clean latent gets the real tokens,
        /// and their denoise mask is `1 - strength`, which lands them at timestep 0.
        ///
        /// **They are not separated in position space.** The reference is the same shot,
        /// so `downscaleFactor` multiplies its height and width coordinates back up into
        /// the target's frame — a half-resolution reference lands on exactly the target's
        /// pixel range. Skipping that scaling is the silent failure: the reference then
        /// occupies the target's top-left quadrant, every token is well-formed, and the
        /// model is being shown a crop it was never trained to see.
        public struct Reference: Sendable, Equatable {
            public var latentFrames: Int
            public var latentHeight: Int
            public var latentWidth: Int
            /// Target-over-reference spatial ratio. 2 for the x2 pixel spatial upscaler,
            /// read from the adapter's own `reference_downscale_factor` metadata.
            public var downscaleFactor: Int
            /// Target-over-reference temporal ratio. 1 for the spatial upscaler.
            public var temporalScaleFactor: Int
            /// A shift of this reference's whole time axis, in **pixel frames**, applied
            /// before the divide by the frame rate. Zero for every shipped IC-LoRA.
            ///
            /// This is the one dial that separates several references from each other.
            /// With it at 0 — the only value the single-reference adapters use — two
            /// references occupy *identical* coordinates and the model has nothing but
            /// content to tell them apart. Multiple-Subject-Reference gives slot `i` of `n`
            /// the offset `-(n - i)`, so the slots land at distinct negative times and the
            /// first of them is furthest from the target's own frame 0.
            ///
            /// Pixel frames, not seconds and not latent frames: the offset is added to
            /// the pixel coordinates and the frame rate divided out afterwards, so `-3` is
            /// three *source* frames — an eighth of a latent frame, not three of them.
            public var frameOffset: Int
            /// Which learned slot vector was added to this reference's latent channels, or
            /// nil if none was. Carried for reporting only — the addition happens on the
            /// latent before patchify, so nothing in the forward pass reads this.
            public var slotIndex: Int?

            public init(latentFrames: Int, latentHeight: Int, latentWidth: Int,
                        downscaleFactor: Int = 1, temporalScaleFactor: Int = 1,
                        frameOffset: Int = 0, slotIndex: Int? = nil) {
                self.latentFrames = latentFrames
                self.latentHeight = latentHeight
                self.latentWidth = latentWidth
                self.downscaleFactor = downscaleFactor
                self.temporalScaleFactor = temporalScaleFactor
                self.frameOffset = frameOffset
                self.slotIndex = slotIndex
            }

            public var tokens: Int { latentFrames * latentHeight * latentWidth }
        }

        /// Reference clips carried in context, **in the order their tokens are appended**.
        ///
        /// A list rather than an optional because Multiple-Subject-Reference conditions on
        /// up to five stills at once and their order is load-bearing twice over: it decides
        /// which slot embedding each one got, and it decides which block of appended tokens
        /// each one's positions describe. A set or a dictionary would lose that.
        ///
        /// Defaulted in the declaration rather than added to `init`, so every existing
        /// caller keeps compiling and keeps meaning what it meant.
        public var references: [Reference] = []

        /// The single-reference view, for the callers that have exactly one.
        ///
        /// Every IC-LoRA that shipped before MSR takes one reference, and reading `nil` or
        /// assigning one is what those call sites say. Setting it replaces the whole list;
        /// reading it returns nil when there is not exactly one, so a caller that grew a
        /// second reference cannot keep reading the first and believe it has them all.
        public var reference: Reference? {
            get { references.count == 1 ? references[0] : nil }
            set { references = newValue.map { [$0] } ?? [] }
        }

        public var referenceTokens: Int { references.reduce(0) { $0 + $1.tokens } }

        /// What the video stream actually carries. ``videoTokens`` stays the count of
        /// *generated* tokens — the noise draw, the conditioning masks and the unpatchify
        /// are all sized from it — and only the forward pass sees the longer sequence.
        public var videoTokensWithReference: Int { videoTokens + referenceTokens }
        public var audioTokens: Int {
            latent.audioLatentCount(frames: frames, frameRate: frameRate)
        }

        public func validate() throws {
            _ = try latent.videoLatentShape(width: width, height: height, frames: frames)
            _ = try latent.audioLatentShape(frames: frames, frameRate: frameRate)
        }
    }

    // MARK: - Sampling

    /// The scalar noise levels the eight AdaLN modules consume, already scaled.
    ///
    /// "Already scaled" means the timesteps have been multiplied by
    /// `timestep_scale_multiplier`. The multiplier is 1000 on this checkpoint and the
    /// multiplication happens in the *sampler's* fp32, before the DiT sees anything —
    /// folding it in here would round a second time.
    ///
    /// ## The mapping is not symmetric
    ///
    /// ```
    /// adaln_single                            <- video timesteps      per video token
    /// audio_adaln_single                      <- audio timesteps      per audio token
    /// prompt_adaln_single                     <- video sigma          scalar
    /// audio_prompt_adaln_single               <- audio sigma          scalar
    /// av_ca_video_scale_shift_adaln_single    <- video timesteps      per video token
    /// av_ca_audio_scale_shift_adaln_single    <- audio timesteps      per audio token
    /// av_ca_a2v_gate_adaln_single             <- AUDIO sigma          scalar
    /// av_ca_v2a_gate_adaln_single             <- VIDEO sigma          scalar
    /// ```
    ///
    /// The last two are the trap. Each stream's gate module is driven by the *other*
    /// stream's sigma, so **the A2V gate is conditioned on how noisy the *audio* is** and
    /// the V2A gate on the video. That is the sensible thing — the gate decides how much
    /// to trust what the other stream is saying — and it is invisible whenever both
    /// streams sit at the same sigma. The fields are kept separate so the mapping is
    /// stated in code rather than accidentally right.
    ///
    /// The gate input carries one further factor,
    /// `av_ca_timestep_scale_multiplier / timestep_scale_multiplier`. On this checkpoint
    /// it is 1, which is why the field is a plain value the caller supplies rather than a
    /// constant multiplied in.
    public struct Sampling {
        /// `[videoTokens]`.
        public var videoTimestep: MLXArray
        /// `[audioTokens]`.
        public var audioTimestep: MLXArray
        /// `[1]`.
        public var videoPromptTimestep: MLXArray
        public var audioPromptTimestep: MLXArray
        /// `[1]`, driven by the **audio** sigma.
        public var a2vGateTimestep: MLXArray
        /// `[1]`, driven by the **video** sigma.
        public var v2aGateTimestep: MLXArray

        public init(videoTimestep: MLXArray, audioTimestep: MLXArray,
                    videoPromptTimestep: MLXArray, audioPromptTimestep: MLXArray,
                    a2vGateTimestep: MLXArray, v2aGateTimestep: MLXArray) {
            self.videoTimestep = videoTimestep
            self.audioTimestep = audioTimestep
            self.videoPromptTimestep = videoPromptTimestep
            self.audioPromptTimestep = audioPromptTimestep
            self.a2vGateTimestep = a2vGateTimestep
            self.v2aGateTimestep = v2aGateTimestep
        }

        /// One scaled noise level for every token of both streams.
        ///
        /// The whole-clip text-to-audio-video case, where nothing is conditioned and every
        /// token is at the same point on the schedule. Per-token timesteps exist for
        /// conditioning and autoregressive extension, which is why the general initialiser
        /// stays.
        public static func uniform(scaledTimestep: Float, videoTokens: Int,
                                   audioTokens: Int) -> Sampling {
            func filled(_ n: Int) -> MLXArray {
                MLXArray([Float](repeating: scaledTimestep, count: n))
            }
            return Sampling(videoTimestep: filled(videoTokens),
                            audioTimestep: filled(audioTokens),
                            videoPromptTimestep: filled(1),
                            audioPromptTimestep: filled(1),
                            a2vGateTimestep: filled(1),
                            v2aGateTimestep: filled(1))
        }

        /// Per-token noise levels from a denoise mask — the **conditioned** case.
        ///
        /// The per-token timesteps are the denoise mask times sigma, so a frozen token
        /// reaches AdaLN at **0** rather than at the step's sigma. That zero is the
        /// signal that the token is a clean reference. Without it the transformer cannot
        /// tell a conditioning frame from a noisy one, conditions on nothing, and the
        /// render only appears conditioned because the sampler overwrites those tokens
        /// after each step — see ``Sampler/timesteps(sigma:mask:tokens:)`` for what that
        /// looks like on screen.
        ///
        /// Masks arrive `[1, tokens, 1]` from `RenderConditioning` and are flattened here,
        /// because this field is `[tokens]`. A `nil` mask means that stream is
        /// unconditioned and takes the uniform value.
        ///
        /// **A wholly frozen stream zeroes more than its per-token timesteps.** Its sigma
        /// is zeroed *before* anything is derived from it, so that the prompt AdaLN and
        /// the cross-modal gates match the frozen conditioning: a frozen stream's prompt
        /// AdaLN input is 0, and so is the gate that stream drives.
        ///
        /// Frozen is **derived** from the mask being entirely zero rather than passed as a
        /// flag, because a flag can disagree with the mask it describes and this one would
        /// disagree silently — the render would complete with the prompt AdaLN of a frozen
        /// stream sitting at the schedule value.
        ///
        /// Note which gate follows which stream: the A2V gate is driven by the **audio**
        /// sigma and the V2A gate by the **video** one, so freezing audio zeroes `a2v`, not
        /// `v2a`. Getting that backwards is invisible whenever both streams share a sigma.
        ///
        /// A partial mask — i2v at strength 0.9, say — is **not** frozen. Only the
        /// per-token timesteps move; the scalar stays on the schedule.
        ///
        /// With all-ones masks this equals ``uniform(scaledTimestep:videoTokens:audioTokens:)``
        /// bit for bit: `1.0 * x` is exact in fp32.
        public static func masked(scaledTimestep: Float,
                                  videoMask: MLXArray?, audioMask: MLXArray?,
                                  videoTokens: Int, audioTokens: Int) -> Sampling {
            func filled(_ n: Int, _ value: Float = scaledTimestep) -> MLXArray {
                MLXArray([Float](repeating: value, count: n))
            }
            func perToken(_ mask: MLXArray?, _ n: Int) -> MLXArray {
                guard let mask else { return filled(n) }
                return mask.asType(.float32).reshaped([n]) * scaledTimestep
            }
            func isFrozen(_ mask: MLXArray?) -> Bool {
                guard let mask else { return false }
                return mask.max().item(Float.self) == 0
            }
            let videoFrozen = isFrozen(videoMask)
            let audioFrozen = isFrozen(audioMask)
            let videoSigma: Float = videoFrozen ? 0 : scaledTimestep
            let audioSigma: Float = audioFrozen ? 0 : scaledTimestep

            return Sampling(videoTimestep: perToken(videoMask, videoTokens),
                            audioTimestep: perToken(audioMask, audioTokens),
                            videoPromptTimestep: filled(1, videoSigma),
                            audioPromptTimestep: filled(1, audioSigma),
                            a2vGateTimestep: filled(1, audioSigma),
                            v2aGateTimestep: filled(1, videoSigma))
        }
    }

    // MARK: - Rotary reconstruction

    /// The frequency grid, evaluated in **float64**, and it has to be float64.
    ///
    /// `frequencies_precision` asks for it, and the two precisions are not
    /// interchangeable: the powers span 1 to 10000 and are then scaled by up to
    /// `10000 * pi/2`, so a float32 exponent's error is multiplied into an angle of
    /// several radians.
    ///
    /// The double precision is confined to *this* function: it casts to float32 on the way
    /// out, and everything downstream — the fractional positions, the product, the cosine
    /// — is float32. Widening any of it would compute a different function, not a more
    /// accurate one.
    ///
    /// The linspace is evaluated as `i * step` with `step = 1 / (n - 1)`, with the last
    /// element *overwritten* by the endpoint. `Double(i) / Double(n - 1)` is a different
    /// expression and differs from it in the last bit at some `i`.
    public static func frequencyGrid(theta: Double, positionalDimensions: Int,
                                     innerDim: Int) -> MLXArray {
        let count = innerDim / (2 * positionalDimensions)
        precondition(count > 1, "a one-element grid has no linspace step")
        let step = 1.0 / Double(count - 1)
        var values = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let exponent = i == count - 1 ? 1.0 : Double(i) * step
            values[i] = Float(pow(theta, exponent) * Double.pi / 2)
        }
        return MLXArray(values)
    }

    /// The video patch grid: each token's bounds in pixel coordinates, with time in
    /// seconds.
    ///
    /// Returns `[1, 3, tokens, 2]`: the `[start, end)` bounds of each token's patch, per
    /// positional dimension, in float32. Three conventions, each of which yields a
    /// same-shaped wrong grid if missed:
    ///
    /// * The bounds are **pixel** coordinates. The latent index is multiplied back up by
    ///   the VAE scale factors `(8, 32, 32)` before anything else happens.
    /// * The causal correction rewrites the time axis as
    ///   `(t + 1 - temporalScale).clamp(min: 0)`, because the causal video VAE's first
    ///   latent frame covers **one** pixel frame and every later one covers eight. The
    ///   first token's bounds are `[0, 1)`, not `[0, 8)`.
    /// * Only time is divided by the frame rate; space stays in pixels.
    ///
    /// Token order is `(frame, y, x)` with `x` fastest, which is also the order
    /// `tokensPerLatentFrame` assumes when the keyframe marker takes "the first latent
    /// frame" to be the first `h * w` tokens.
    public static func videoPositions(latentFrames: Int, latentHeight: Int,
                                      latentWidth: Int, temporalScale: Int,
                                      spatialScale: Int, frameRate: Double) -> MLXArray {
        let tokens = latentFrames * latentHeight * latentWidth
        var values = [Float](repeating: 0, count: 3 * tokens * 2)
        let scales = [temporalScale, spatialScale, spatialScale]
        var token = 0
        for t in 0..<latentFrames {
            for y in 0..<latentHeight {
                for x in 0..<latentWidth {
                    let start = [t, y, x]
                    for dim in 0..<3 {
                        // Integer pixel bounds first, then the causal correction, then the
                        // float conversion: the clamp is meaningless once the value has
                        // been divided by the frame rate.
                        var lo = start[dim] * scales[dim]
                        var hi = (start[dim] + 1) * scales[dim]
                        if dim == 0 {
                            lo = max(lo + 1 - temporalScale, 0)
                            hi = max(hi + 1 - temporalScale, 0)
                        }
                        var loF = Float(lo), hiF = Float(hi)
                        if dim == 0 {
                            loF /= Float(frameRate)
                            hiF /= Float(frameRate)
                        }
                        values[(dim * tokens + token) * 2] = loF
                        values[(dim * tokens + token) * 2 + 1] = hiF
                    }
                    token += 1
                }
            }
        }
        return MLXArray(values, [1, 3, tokens, 2])
    }

    /// The reference clip's positions, translated into the target's coordinate frame.
    ///
    /// The order below is load-bearing, because the temporal shift reads coordinates the
    /// spatial scaling would have already changed if it ran first:
    ///
    /// 1. Build the reference's **own** patch grid. Same function as the target's — the
    ///    reference is a smaller clip, not a different kind of thing.
    /// 2. Put it on its own time spacing, `fps / temporalScaleFactor`. Folded into the
    ///    `frameRate` argument, because ``videoPositions`` already divides time by it.
    /// 3. Shift the whole time axis by `frameOffset` **pixel frames**, while it is still in
    ///    pixel frames. Zero for every shipped adapter; negative for an MSR slot.
    /// 4. If the reference is temporally sparse, slide it forward so its last patch ends
    ///    where the target's last patch ends, clamping the causal first patch's negative
    ///    start back to zero.
    /// 5. Scale height and width by `downscaleFactor`, so a half-resolution reference
    ///    covers the target's full pixel range rather than a corner of it.
    ///
    /// Step 3 happens **before** the divide by the frame rate rather than after: adding
    /// `frameOffset / fps` to an already-divided grid is the same number in exact
    /// arithmetic and a different one in float32, on a quantity that feeds a cosine scaled
    /// by up to `10000 * pi/2`.
    ///
    /// Step 4 is unexercised by the x2 spatial upscaler, whose
    /// `reference_temporal_scale_factor` is 1. It is written out rather than refused so
    /// that a temporally-sparse adapter is a configuration rather than a code change, but
    /// nothing here has run it.
    ///
    /// **Steps 3 and 4 do not compose**, and the precondition says so rather than letting
    /// them. Step 4's `clamp(min: 0)` would erase exactly the negative times step 3 exists
    /// to create — leaving every MSR slot stacked at time zero, which is a well-formed
    /// grid that has silently lost the separation.
    public static func referencePositions(_ reference: Geometry.Reference,
                                          target: MLXArray,
                                          geometry: Geometry) -> MLXArray {
        precondition(reference.frameOffset == 0 || reference.temporalScaleFactor == 1,
                     "a reference cannot carry both a frame offset (\(reference.frameOffset)) "
                     + "and a temporal scale factor (\(reference.temporalScaleFactor)): the "
                     + "sparse-reference shift clamps time at zero and would erase the offset")

        // Built at frame rate 1, so the time axis comes back in **pixel frames** and the
        // offset can be added in the units it is defined in. Dividing afterwards by the
        // same `Float` divisor `videoPositions` would have used is bit-identical to
        // letting it divide, so the no-offset path is unchanged.
        var positions = videoPositions(
            latentFrames: reference.latentFrames,
            latentHeight: reference.latentHeight,
            latentWidth: reference.latentWidth,
            temporalScale: geometry.latent.temporalScale,
            spatialScale: geometry.latent.spatialScale,
            frameRate: 1)

        var time = positions[0..., 0..<1, 0..., 0...]
        var space = positions[0..., 1..<3, 0..., 0...]

        if reference.frameOffset != 0 {
            time = time + Float(reference.frameOffset)
        }
        // Its own time spacing, `fps / temporalScaleFactor`.
        time = time / Float(geometry.frameRate / Double(reference.temporalScaleFactor))

        if reference.temporalScaleFactor != 1 {
            // `t_target` is the target's first patch end — one target frame period.
            let tTarget = target[0..., 0..<1, 0..<1, 1..<2]
            let shift = Float(reference.temporalScaleFactor - 1) * tTarget
            time = MLX.maximum(time - shift, MLXArray(Float(0)).asType(time.dtype))
        }
        if reference.downscaleFactor != 1 {
            space = space * Float(reference.downscaleFactor)
        }
        positions = MLX.concatenated([time, space], axis: 1)
        return positions
    }

    /// The audio patch grid — timestamps in **seconds**, not indices.
    ///
    /// Returns `[1, 1, tokens, 2]`. The same causal correction as video, one level down: a
    /// latent frame covers `audioLatentDownsample` mel frames and the first one does not,
    /// so the mel index is `(frame * 4 + 1 - 4).clamp(min: 0)` before
    /// `* hopLength / sampleRate`. The `[start, end)` pair is the same series offset by one
    /// latent frame, which is why token 0 spans `[0.00, 0.01)` and token 1 spans
    /// `[0.01, 0.05)` — a tenth of the width of every later token.
    ///
    /// The two scalings are kept as separate operations (`* hop` then `/ sampleRate`):
    /// fusing them into one constant is a different float32 rounding.
    public static func audioPositions(tokens: Int, hopLength: Int, sampleRate: Int,
                                      downsample: Int) -> MLXArray {
        func seconds(_ index: Int) -> Float {
            var frame = Float(index * downsample)
            frame = max(frame + 1 - Float(downsample), 0)
            return frame * Float(hopLength) / Float(sampleRate)
        }
        var values = [Float](repeating: 0, count: tokens * 2)
        for i in 0..<tokens {
            values[i * 2] = seconds(i)
            values[i * 2 + 1] = seconds(i + 1)
        }
        return MLXArray(values, [1, 1, tokens, 2])
    }

    /// The split-rotary cos/sin tables for one stream, from its patch bounds.
    ///
    /// Two details with no shape error to catch them:
    ///
    /// **The positional dimension varies fastest.** Channel `j` is index `j / nPos` of
    /// positional dimension `j % nPos`. The untransposed layout is the same width and a
    /// permutation of the same numbers.
    ///
    /// **The padding goes in front.** `dim / 2` need not be divisible by
    /// `nPos * indexCount`: video is 2048 against `3 * 682 = 2046`, so two **leading**
    /// channels get `cos = 1, sin = 0` — an identity rotation. Audio at 1024 against
    /// `1 * 1024` pads nothing. Padding at the tail leaves the shape intact and rotates
    /// every channel by the wrong angle.
    ///
    /// The `[start, end)` bounds are averaged, so a token's angle comes from its patch
    /// *centre*.
    ///
    /// The tables are cast to `dtype` — the latent's dtype, bf16 — at the end, because the
    /// rotation multiplies bf16 activations by a bf16 embedding. Handing it an fp32 table
    /// computes a *different* function, not a more accurate one.
    public static func rotary(positions: MLXArray, innerDim: Int, maxPos: [Int],
                              heads: Int, theta: Double,
                              dtype: DType) -> DiTAttention.Rotary {
        let positionalDimensions = positions.dim(1)
        let tokens = positions.dim(2)
        precondition(maxPos.count == positionalDimensions,
                     "maxPos has \(maxPos.count) entries for \(positionalDimensions) "
                     + "positional dimensions")

        let bounds = positions.asType(.float32)
        let middle = (bounds[0..., 0..., 0..., 0] + bounds[0..., 0..., 0..., 1]) / 2.0
        // Each dimension's centre over its own `maxPos`: [1, tokens, nPos].
        let fractional = MLX.stacked(
            (0..<positionalDimensions).map { middle[0..., $0, 0...] / Float(maxPos[$0]) },
            axis: -1)

        let indices = frequencyGrid(theta: theta,
                                    positionalDimensions: positionalDimensions,
                                    innerDim: innerDim)
        // [1, tokens, nPos, indexCount] -> transpose -> flatten from axis 2.
        let angles = indices * (fractional.expandedDimensions(axis: -1) * 2.0 - 1.0)
        let ordered = angles.transposed(0, 1, 3, 2)
            .reshaped([1, tokens, indices.size * positionalDimensions])

        var cos = MLX.cos(ordered), sin = MLX.sin(ordered)
        let pad = innerDim / 2 - ordered.dim(-1)
        precondition(pad >= 0, "inner dim \(innerDim) cannot hold \(ordered.dim(-1)) "
                     + "frequencies")
        if pad > 0 {
            cos = MLX.concatenated([MLX.ones([1, tokens, pad]), cos], axis: -1)
            sin = MLX.concatenated([MLX.zeros([1, tokens, pad]), sin], axis: -1)
        }
        func heads4(_ t: MLXArray) -> MLXArray {
            t.reshaped([1, tokens, heads, innerDim / 2 / heads]).transposed(0, 2, 1, 3)
        }
        return DiTAttention.Rotary(cos: heads4(cos).asType(dtype),
                                   sin: heads4(sin).asType(dtype))
    }

    /// The four rotary embeddings a block needs, built once for the whole stack.
    ///
    /// Positions do not change with depth, so one set serves all 48 blocks.
    ///
    /// The cross-modal pair is **temporal only and built at the audio cross-attention
    /// width for both streams**: the time axis alone, at `audio_cross_attention_dim`. So
    /// the same video tokens carry a
    /// `[1, 32, T, 64]` embedding in `attn1` and a `[1, 32, T, 32]` one in
    /// `audio_to_video_attn`. The audio self and audio cross embeddings coincide exactly —
    /// audio positions are already one-dimensional and the audio width already equals the
    /// cross width — which makes confusing them undetectable, so it is written down.
    public struct Rotaries {
        public let videoSelf: DiTAttention.Rotary
        public let audioSelf: DiTAttention.Rotary
        public let videoCross: DiTAttention.Rotary
        public let audioCross: DiTAttention.Rotary
    }

    public func rotaries(geometry: Geometry, dtype: DType) -> Rotaries {
        let heads = topology.numAttentionHeads
        var video = Self.videoPositions(
            latentFrames: geometry.latentFrames,
            latentHeight: geometry.height / geometry.latent.spatialScale,
            latentWidth: geometry.width / geometry.latent.spatialScale,
            temporalScale: geometry.latent.temporalScale,
            spatialScale: geometry.latent.spatialScale,
            frameRate: geometry.frameRate)
        if !geometry.references.isEmpty {
            // `target: video` is the *generated* grid every time, never the running
            // concatenation. The sparse-reference shift reads the target's first patch end
            // out of it, and feeding it a grid that already had a reference appended would
            // read a reference's coordinate instead — same shape, wrong shift, and the
            // error grows with each reference added.
            let target = video
            video = MLX.concatenated(
                [video] + geometry.references.map {
                    Self.referencePositions($0, target: target, geometry: geometry)
                },
                axis: 2)
        }
        let audio = Self.audioPositions(tokens: geometry.audioTokens,
                                        hopLength: geometry.audioHopLength,
                                        sampleRate: geometry.audioSampleRate,
                                        downsample: geometry.audioLatentDownsample)
        let time = video[0..., 0..<1, 0..., 0...]
        return Rotaries(
            videoSelf: Self.rotary(positions: video, innerDim: topology.videoWidth,
                                   maxPos: geometry.videoMaxPos, heads: heads,
                                   theta: geometry.theta, dtype: dtype),
            audioSelf: Self.rotary(positions: audio, innerDim: topology.audioWidth,
                                   maxPos: geometry.audioMaxPos, heads: heads,
                                   theta: geometry.theta, dtype: dtype),
            videoCross: Self.rotary(positions: time, innerDim: topology.audioWidth,
                                    maxPos: [geometry.crossMaxPos], heads: heads,
                                    theta: geometry.theta, dtype: dtype),
            audioCross: Self.rotary(positions: audio, innerDim: topology.audioWidth,
                                    maxPos: [geometry.crossMaxPos], heads: heads,
                                    theta: geometry.theta, dtype: dtype))
    }

    // MARK: - Entry

    /// The keyframe absolute-position embedding — **not** a no-op on this checkpoint.
    ///
    /// The parameter is zero on checkpoints that predate it, but this one carries a
    /// *trained* embedding, and the first latent frame is marked **unconditionally** — not
    /// only when the caller supplies keyframes — because the causal VAE makes that frame
    /// the same token class as a keyframe slot.
    ///
    /// Skipping it is quiet rather than loud: it touches only the first latent frame's
    /// tokens, and by less than the bf16 storage floor, so the first block simply starts
    /// from a slightly wrong stream and every later one inherits it.
    ///
    /// Video only: `keyframes_abs_pos_embedding` exists for the video stream alone, so the
    /// audio stream never receives one.
    ///
    /// **Reference tokens are never marked, and marking the first `tokensPerLatentFrame`
    /// tokens is what achieves that** — references are appended, so the head of the stream
    /// is always the target's own first latent frame. A reference's own first latent frame
    /// also spans a single pixel frame and would otherwise claim the marker on position
    /// alone. If the token order here ever became prepend, this line would silently mark a
    /// reference and leave the target's frame 0 bare.
    public func applyKeyframeEmbedding(_ x: MLXArray,
                                       tokensPerLatentFrame: Int) throws -> MLXArray {
        guard let embedding = weights[topology.prefix + "keyframes_abs_pos_embedding"]
        else {
            // A checkpoint without the parameter marks no token at all. Not an error.
            return x
        }
        let tokens = x.dim(1)
        precondition(tokensPerLatentFrame <= tokens,
                     "a latent frame cannot be wider than the stream")
        var mask = [Float](repeating: 0, count: tokens)
        for i in 0..<tokensPerLatentFrame { mask[i] = 1 }
        let marker = MLXArray(mask, [1, tokens, 1]).asType(x.dtype)
            * embedding.reshaped([1, 1, embedding.size]).asType(x.dtype)
        return x + marker
    }

    /// Everything the 48 blocks read, built once.
    ///
    /// Returned rather than kept private so a test can check the entry stage — the
    /// patchified streams and the modulation — before spending 48 blocks of compute on a
    /// stream that was already wrong.
    public struct Prepared {
        public var video: DiTBlock.Conditioning
        public var audio: DiTBlock.Conditioning
        /// `patchify_proj.out`, before the keyframe marker. `video.x` is after it.
        public var videoPatchified: MLXArray
        public var audioPatchified: MLXArray
        /// `AdaLayerNormSingle` returns this alongside the modulation; the output head
        /// consumes it and nothing in the block body does.
        public var videoEmbeddedTimestep: MLXArray
        public var audioEmbeddedTimestep: MLXArray
    }

    public func prepare(videoLatent: MLXArray, audioLatent: MLXArray,
                        videoContext: MLXArray, audioContext: MLXArray,
                        sampling: Sampling, geometry: Geometry,
                        adaLNInputObserver: ((String, MLXArray) throws -> Void)? = nil,
                        tapObserver: ((String, MLXArray) throws -> Void)? = nil)
        throws -> Prepared {
        // With no reference this is `videoTokens` and the check is the one it always was.
        guard videoLatent.dim(1) == geometry.videoTokensWithReference else {
            throw Failure.tokenCount(stream: "video", latent: videoLatent.dim(1),
                                     geometry: geometry.videoTokensWithReference)
        }
        guard audioLatent.dim(1) == geometry.audioTokens else {
            throw Failure.tokenCount(stream: "audio", latent: audioLatent.dim(1),
                                     geometry: geometry.audioTokens)
        }

        let videoPatchified = try modules.patchify(videoLatent, stream: .video)
        let audioPatchified = try modules.patchify(audioLatent, stream: .audio)
        // These are the exact module return values, before the keyframe addition and
        // before either can be threaded through a block. Observing them here is what keeps
        // them from ever being reconstructed out of a later residual.
        try tapObserver?("patchify_proj.out", videoPatchified)
        try tapObserver?("audio_patchify_proj.out", audioPatchified)
        let vx = try applyKeyframeEmbedding(
            videoPatchified, tokensPerLatentFrame: geometry.tokensPerLatentFrame)

        // Each module's `[N, k * dim]` output is reshaped to `[B, -1, k * dim]`.
        // `DiTBlock.adaValues` reshapes on axis 1, so a tensor left at `[N, k * dim]`
        // would have `k * dim` read as the token count and the modulation sliced out of
        // axes that mean nothing.
        func modulation(_ t: MLXArray, _ module: String) throws
            -> (MLXArray, MLXArray) {
            // The actual module input, after the sampler has selected the pass and the DiT
            // has constructed its per-stream sampling arguments — observed at the one
            // point the module receives it rather than rebuilt from the schedule.
            try adaLNInputObserver?(module, t)
            let (m, embedded) = try modules.adaLayerNormSingle(t, module: module)
            try tapObserver?(module + ".out", m)
            try tapObserver?(module + ".out.el1", embedded)
            return (m.reshaped([1, m.dim(0), m.dim(1)]),
                    embedded.reshaped([1, embedded.dim(0), embedded.dim(1)]))
        }

        let (videoTimesteps, videoEmbedded) =
            try modulation(sampling.videoTimestep, "adaln_single")
        let (audioTimesteps, audioEmbedded) =
            try modulation(sampling.audioTimestep, "audio_adaln_single")
        let (videoPrompt, _) =
            try modulation(sampling.videoPromptTimestep, "prompt_adaln_single")
        let (audioPrompt, _) =
            try modulation(sampling.audioPromptTimestep, "audio_prompt_adaln_single")
        let (videoCrossScaleShift, _) = try modulation(
            sampling.videoTimestep, "av_ca_video_scale_shift_adaln_single")
        let (audioCrossScaleShift, _) = try modulation(
            sampling.audioTimestep, "av_ca_audio_scale_shift_adaln_single")
        let (a2vGate, _) =
            try modulation(sampling.a2vGateTimestep, "av_ca_a2v_gate_adaln_single")
        let (v2aGate, _) =
            try modulation(sampling.v2aGateTimestep, "av_ca_v2a_gate_adaln_single")

        // The rotary tables are built at the stream's own dtype, which is the latent's.
        let pe = rotaries(geometry: geometry, dtype: vx.dtype)

        return Prepared(
            video: DiTBlock.Conditioning(
                x: vx, timesteps: videoTimesteps, context: videoContext,
                promptTimestep: videoPrompt,
                crossScaleShiftTimestep: videoCrossScaleShift,
                // The gate belongs to the direction that WRITES this stream.
                crossGateTimestep: a2vGate,
                rotary: pe.videoSelf, crossRotary: pe.videoCross),
            audio: DiTBlock.Conditioning(
                x: audioPatchified, timesteps: audioTimesteps, context: audioContext,
                promptTimestep: audioPrompt,
                crossScaleShiftTimestep: audioCrossScaleShift,
                crossGateTimestep: v2aGate,
                rotary: pe.audioSelf, crossRotary: pe.audioCross),
            videoPatchified: videoPatchified, audioPatchified: audioPatchified,
            videoEmbeddedTimestep: videoEmbedded, audioEmbeddedTimestep: audioEmbedded)
    }

    // MARK: - The loop

    public struct Output {
        /// The residual streams as the output head will receive them.
        public var video: MLXArray
        public var audio: MLXArray
        public var videoEmbeddedTimestep: MLXArray
        public var audioEmbeddedTimestep: MLXArray
    }

    /// Every block in sequence, from the latents to the input of the output head.
    ///
    /// - Parameters:
    ///   - observer: called after each block with `(index, video, audio)`, so a caller
    ///     that wants several depths gets them from one pass rather than re-running the
    ///     stack once per depth.
    ///   - blocks: how many blocks to run, defaulting to every block the checkpoint has.
    ///     Bounded by `topology.blockCount` — a caller that asks for 48 on a 32-block
    ///     checkpoint gets 32 rather than a missing-weight error 32 blocks in.
    ///
    /// **Both streams are threaded, and that is the whole point.** `DiTBlock` returns a
    /// pair and the audio half is not a by-product: A2V reads the audio stream to write
    /// video and V2A reads video to write audio, so dropping either feedback path leaves
    /// the other stream's *shape* untouched and its values wrong from block 1 onwards.
    public func callAsFunction(videoLatent: MLXArray, audioLatent: MLXArray,
                               videoContext: MLXArray, audioContext: MLXArray,
                               sampling: Sampling, geometry: Geometry,
                               blocks: Int? = nil,
                               videoSelfAttentionPassthroughBlocks: Set<Int> = [],
                               audioSelfAttentionPassthroughBlocks: Set<Int> = [],
                               crossPerturbationMask: MLXArray? = nil,
                               adaLNInputObserver: ((String, MLXArray) throws -> Void)? = nil,
                               tapObserver: ((String, MLXArray) throws -> Void)? = nil,
                               blockTapObserver: ((Int, String, MLXArray) throws -> Void)? = nil,
                               blockOutputObserver: ((Int, String, MLXArray) throws -> Void)? = nil,
                               observer: ((Int, MLXArray, MLXArray) throws -> Void)? = nil,
                               stepCache: StepCache? = nil,
                               stepIndex: Int = 0,
                               stepCount: Int = 0)
        throws -> Output {
        var prepared = try prepare(videoLatent: videoLatent, audioLatent: audioLatent,
                                   videoContext: videoContext,
                                   audioContext: audioContext,
                                   sampling: sampling, geometry: geometry,
                                   adaLNInputObserver: adaLNInputObserver,
                                   tapObserver: tapObserver)
        if let mask = crossPerturbationMask {
            prepared.video.crossPerturbationMask = mask
            prepared.audio.crossPerturbationMask = mask
        }
        var video = prepared.video, audio = prepared.audio
        let count = min(blocks ?? topology.blockCount, topology.blockCount)

        if let cache = stepCache, count > 1, stepCount > 0 {
            try runCachedBlocks(cache: cache, stepIndex: stepIndex, stepCount: stepCount,
                                count: count, video: &video, audio: &audio,
                                videoSelfAttentionPassthroughBlocks: videoSelfAttentionPassthroughBlocks,
                                audioSelfAttentionPassthroughBlocks: audioSelfAttentionPassthroughBlocks,
                                blockTapObserver: blockTapObserver,
                                blockOutputObserver: blockOutputObserver,
                                observer: observer)
        } else {
            for index in 0..<count {
                try runOneBlock(index, video: &video, audio: &audio,
                                videoSelfAttentionPassthroughBlocks: videoSelfAttentionPassthroughBlocks,
                                audioSelfAttentionPassthroughBlocks: audioSelfAttentionPassthroughBlocks,
                                blockTapObserver: blockTapObserver,
                                blockOutputObserver: blockOutputObserver,
                                observer: observer)
            }
        }

        return Output(video: video.x, audio: audio.x,
                      videoEmbeddedTimestep: prepared.videoEmbeddedTimestep,
                      audioEmbeddedTimestep: prepared.audioEmbeddedTimestep)
    }

    /// Block 0 always runs (it is the probe). The remaining 47 run only when
    /// the cache says the residual has moved enough to bother.
    ///
    /// Nil `stepCache` takes the plain loop above, which is bit-identical to
    /// this path on a render that never reuses — both eval after every block.
    /// The split exists so a skipped step does not build 47 blocks of graph.
    private func runCachedBlocks(
        cache: StepCache, stepIndex: Int, stepCount: Int, count: Int,
        video: inout DiTBlock.Conditioning, audio: inout DiTBlock.Conditioning,
        videoSelfAttentionPassthroughBlocks: Set<Int>,
        audioSelfAttentionPassthroughBlocks: Set<Int>,
        blockTapObserver: ((Int, String, MLXArray) throws -> Void)?,
        blockOutputObserver: ((Int, String, MLXArray) throws -> Void)?,
        observer: ((Int, MLXArray, MLXArray) throws -> Void)?
    ) throws {
        let hInVideo = video.x, hInAudio = audio.x
        try runOneBlock(0, video: &video, audio: &audio,
                        videoSelfAttentionPassthroughBlocks: videoSelfAttentionPassthroughBlocks,
                        audioSelfAttentionPassthroughBlocks: audioSelfAttentionPassthroughBlocks,
                        blockTapObserver: blockTapObserver,
                        blockOutputObserver: blockOutputObserver,
                        observer: observer)
        let videoProbe = video.x - hInVideo
        let audioProbe = audio.x - hInAudio
        MLX.eval(videoProbe, audioProbe)

        switch cache.decide(videoProbe: videoProbe, audioProbe: audioProbe,
                            step: stepIndex, totalSteps: stepCount) {
        case let .reuse(cachedVideo, cachedAudio):
            video.x = hInVideo + cachedVideo
            audio.x = hInAudio + cachedAudio
            MLX.eval(video.x, audio.x)
        case .runFull:
            for index in 1..<count {
                try runOneBlock(index, video: &video, audio: &audio,
                                videoSelfAttentionPassthroughBlocks: videoSelfAttentionPassthroughBlocks,
                                audioSelfAttentionPassthroughBlocks: audioSelfAttentionPassthroughBlocks,
                                blockTapObserver: blockTapObserver,
                                blockOutputObserver: blockOutputObserver,
                                observer: observer)
            }
            cache.record(videoResidual: video.x - hInVideo,
                         audioResidual: audio.x - hInAudio)
        }
    }

    private func runOneBlock(
        _ index: Int,
        video: inout DiTBlock.Conditioning, audio: inout DiTBlock.Conditioning,
        videoSelfAttentionPassthroughBlocks: Set<Int>,
        audioSelfAttentionPassthroughBlocks: Set<Int>,
        blockTapObserver: ((Int, String, MLXArray) throws -> Void)?,
        blockOutputObserver: ((Int, String, MLXArray) throws -> Void)?,
        observer: ((Int, MLXArray, MLXArray) throws -> Void)?
    ) throws {
        video.selfAttentionAllPerturbed = videoSelfAttentionPassthroughBlocks.contains(index)
        audio.selfAttentionAllPerturbed = audioSelfAttentionPassthroughBlocks.contains(index)

        if let checkpointing {
            // `video`/`audio` are `inout` and cannot be captured by an escaping closure, so
            // the conditioning is copied. Everything in it besides `x` — timesteps, masks,
            // rope tables — is constant through the backward, so capturing it is correct;
            // only the carried activations and the adapter factors cross the boundary.
            //
            // `x` is then dropped from the copy, and that detail is load-bearing. The graph
            // retains this closure for the backward, so a captured `x` would pin the previous
            // block's output, whose own node holds the block before it — 48 closures chained
            // into one retention graph, each with a full weight table. It defeats the memory
            // saving and unwinding it recurses 48 frames deep; past ~16 blocks that overflows
            // the test thread's stack and lands as SIGBUS with no diagnostic. The real `x`
            // arrives through `io[0]`/`io[1]` below.
            var carriedVideo = video, carriedAudio = audio
            let placeholder = MLXArray.zeros([1])
            carriedVideo.x = placeholder
            carriedAudio.x = placeholder
            let traced = checkpointing.parameters(index)
            let base = block

            let out = GradientCheckpoint.apply([video.x, audio.x] + traced) { io in
                var video = carriedVideo, audio = carriedAudio
                video.x = io[0]
                audio.x = io[1]
                var scoped = base
                scoped.attach(lora: checkpointing.overlay(index, Array(io.dropFirst(2))))
                // `try!`: the boundary is not throwing, and every error `callAsFunction`
                // raises is a missing or misshapen weight — structural, so it would have
                // fired on block 0 long before any checkpointed block ran.
                let result = try! scoped(block: index, video: video, audio: audio)
                return [result.video, result.audio]
            }
            video.x = out[0]
            audio.x = out[1]
            // No `MLX.eval` here. Forcing the block's output would materialise exactly what
            // the checkpoint exists to drop.
            try observer?(index, video.x, audio.x)
            return
        }

        let out = try block(block: index, video: video, audio: audio,
                            observer: { suffix, tensor in
                                try blockTapObserver?(index, suffix, tensor)
                            }, outputObserver: { suffix, tensor in
                                try blockOutputObserver?(index, suffix, tensor)
                            })
        video.x = out.video
        audio.x = out.audio
        MLX.eval(video.x, audio.x)
        try observer?(index, video.x, audio.x)
    }
}

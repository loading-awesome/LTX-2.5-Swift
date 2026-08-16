// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Dispatch
import LTXCatalog
import LTXFoundation
import LTXModules
import LTXRecipes
import MLX

/// The distilled two-stage pipeline — draft at half resolution, upsample the **video**
/// latent, refine at full resolution.
///
/// ## The shape of it
///
/// ```
/// stage 1  W/2 x H/2   distilled sigmas          9 sigmas -> 8 steps   ancestral, eta=1
/// upsample video only  [1,128,F,H',W'] -> [1,128,F,2H',2W']
/// stage 2  W x H       stage-2 distilled sigmas  4 sigmas -> 3 steps   deterministic Euler
/// ```
///
/// ## The thing that is easiest to get wrong, stated first
///
/// **Only the video latent goes through the upsampler.** There is no audio counterpart
/// anywhere in the pipeline; stage 2 receives stage 1's audio latent **untouched** as its
/// initial latent. The audio latent is `[1, 8, 101, 16]` at every resolution — it is a
/// function of the frame count and the frame rate, not of the picture size — so a
/// symmetric "upsample both streams" would fail on rank rather than silently, but a
/// symmetric *re-draw* of the audio would not. ``render(spec:checkpoints:text:)`` carries
/// stage 1's audio latent forward by value for exactly this reason, and
/// `DistilledTwoStageTests.theAudioLatentIsNotUpsampled` asserts it.
///
/// ## No guider is constructed, and that needs no new code
///
/// This pipeline denoises without a guider at all, which here is ``neutralGuidance`` —
/// `cfg 1.0, stg 0.0, modality 1.0, rescale 0.0` — for which `Sampler.plan(step:)` yields
/// `passes == [.conditional]` (every `runsUnconditional` / `runsPerturbed` /
/// `runsIsolatedModality` predicate is false) and `Sampler.combine` returns the
/// conditional prediction unmodified (every coefficient is zero, and `rescaleScale == 0`
/// skips the rescale outright). One transformer forward per step, eleven for the whole
/// render.
public enum DistilledRenderer {

    public enum Failure: Error, CustomStringConvertible {
        case missingCheckpoint(String, path: String)
        /// The two-stage resolution check.
        case resolution(width: Int, height: Int)
        /// A conditioning pair built for one stage's geometry handed to the other.
        case conditioningGeometry(stage: Int, what: String, got: [Int], expected: [Int])

        public var description: String {
            switch self {
            case let .missingCheckpoint(what, path):
                return "the \(what) checkpoint is not at \(path)"
            case let .resolution(width, height):
                return "a two-stage render needs \(width)x\(height) divisible by 64, not 32: "
                    + "stage 1 samples at half of it and that half must still sit on the "
                    + "VAE's 32-pixel latent grid (assert_resolution, is_two_stage=True)"
            case let .conditioningGeometry(stage, what, got, expected):
                return "stage \(stage)'s \(what) is \(got), expected \(expected) — a "
                    + "conditioning pair is built for one geometry and the two stages do "
                    + "not share one"
            }
        }
    }

    // MARK: - Constants

    /// Fully ancestral: the whole variance-preserving amount, every step.
    public static let ancestralEta: Float = 1.0
    /// The scale on each ancestral renoise draw.
    public static let ancestralSNoise: Float = 1.0
    /// The ancestral loop's seed offset. Without it the loop's first draw would be
    /// bit-identical to the initial latent's, both being a normal draw at the same shape
    /// and dtype from a freshly seeded generator.
    public static let ancestralNoiseSeedOffset: UInt64 = 10_000
    /// The first model version whose distilled stage 1 is ancestral.
    public static let ancestralSamplerSinceVersion = [2, 5]

    /// **A constant this implementation has to invent.**
    ///
    /// All four initial latents — stage 1 video, stage 1 audio, stage 2 video, stage 2
    /// audio, in that order — come from **one** generator that advances across both stages.
    /// MLX's keys are counter-based and do not advance, so there is no equivalent of "the
    /// same generator, later". Stage 2 therefore gets its own key.
    ///
    /// The offset only needs to differ from stage 1's key and from the ancestral loop's
    /// `seed + 10000`. Without it the two stages' *audio* draws would be bit-identical —
    /// same seed, same split, same `[1, 101, 128]` shape — a correlation nothing should
    /// have and no shape check would ever notice. The video draws differ regardless,
    /// because the token counts differ.
    ///
    /// It is named and recorded in the provenance so a render can be reproduced.
    public static let stage2NoiseSeedOffset: UInt64 = 20_000

    /// No guider, expressed as guidance parameters: one pass per step.
    ///
    /// Not a "disabled guider" in the sense of computing terms and multiplying them by
    /// zero — `Sampler.plan(step:)` never asks for the passes at all, so this is three
    /// transformer forwards *per step* cheaper than the production single-stage settings.
    public static let neutralGuidance = GuidanceParams(
        cfgScale: 1.0, stgScale: 0.0, stgBlocks: [], rescaleScale: 0.0,
        modalityScale: 1.0, skipStep: 0)

    /// Whether stage 1 takes the ancestral sampler, which is a property of the checkpoint.
    ///
    /// Reads `model_version` off the transformer checkpoint's `__metadata__` and compares
    /// it as numeric components: parsing stops at the first non-integer component (so
    /// `2.5.rc1` is `(2, 5)`), `-` is normalised to `.` first (so `2.5-rc2` is too), and an
    /// unset or unreadable version parses to the empty list, which compares below every
    /// real version and therefore selects the deterministic sampler.
    ///
    /// A 2.4-era distilled transformer run through this file takes the deterministic
    /// stage 1 instead, and getting
    /// that backwards is silent: the ancestral loop injects noise at every step, so the
    /// wrong answer produces a plausible render either way.
    public static func shouldUseAncestralSampler(transformer: URL) -> Bool {
        guard let header = try? SafetensorsHeader.read(from: transformer) else { return false }
        return isAtLeastVersion(parseModelVersion(header.metadata["model_version"]),
                                ancestralSamplerSinceVersion)
    }

    /// Version-component `>=`: lexicographic, with a shorter prefix comparing *below* a
    /// longer one sharing it — `[] < [2, 5]`, `[2, 5] >= [2, 5]`, `[2, 5, 1] >= [2, 5]`,
    /// `[3] >= [2, 5]`. `lexicographicallyPrecedes` is exactly that ordering, so the
    /// negation is the whole comparison — including the empty case, which precedes
    /// everything and therefore selects the deterministic sampler.
    public static func isAtLeastVersion(_ version: [Int], _ minimum: [Int]) -> Bool {
        !version.lexicographicallyPrecedes(minimum)
    }

    /// A version string's leading numeric components, with `-` normalised to `.` first.
    public static func parseModelVersion(_ version: String?) -> [Int] {
        guard let version, !version.isEmpty else { return [] }
        var parts: [Int] = []
        for component in version.replacingOccurrences(of: "-", with: ".").split(separator: ".") {
            guard component.allSatisfy(\.isNumber), let n = Int(component) else { break }
            parts.append(n)
        }
        return parts
    }

    // MARK: - Checkpoints

    /// The four a single-stage render needs, plus the upsampler.
    ///
    /// The video VAE is named once and used twice: the upsampler needs its
    /// `per_channel_statistics` and nothing else from it.
    /// ``LatentSpatialUpsampler/statistics(videoVAE:)`` reads those two `[128]` vectors out
    /// of the header instead of building an encoder, so the file is opened twice and
    /// mapped once.
    public struct Checkpoints {
        public var base: Renderer.Checkpoints
        public var upsampler: URL

        public init(base: Renderer.Checkpoints, upsampler: URL) {
            self.base = base
            self.upsampler = upsampler
        }

        public func verifyPresent() throws {
            try base.verifyPresent()
            guard FileManager.default.fileExists(atPath: upsampler.path) else {
                throw Failure.missingCheckpoint("latent spatial upsampler", path: upsampler.path)
            }
            // Refuse the temporal upscaler, which ships in the same directory under a
            // near-identical name and loads all the way to a rank mismatch deep inside
            // `spatialUpsample` otherwise.
            _ = try LatentSpatialUpsampler.readConfiguration(
                try SafetensorsHeader.read(from: upsampler))
        }
    }

    // MARK: - Spec

    /// What gets rendered. **No step count and no guidance**: both stages' schedules are
    /// literal constants and the pipeline builds no guider.
    ///
    /// Exposing a `steps:` knob here would be a knob that does nothing, or worse, one that
    /// silently ran `FlowSchedule.sigmas(steps:)` — a computed shifted schedule that is not
    /// this pipeline's and produces a different render.
    public struct Spec {
        /// The prompt. There is no negative prompt: with `cfgScale == 1` no unconditional
        /// pass runs, so a negative branch would be a second 12 B encoder forward producing
        /// a tensor nothing reads.
        public var prompt: String
        /// The **output** geometry. Stage 1 samples at half its width and height.
        public var geometry: DiTForward.Geometry
        public var seed: UInt64
        /// Adapters overlaid on the transformer, applied to **both** stages.
        ///
        /// Both stages run the same weights, so an adapter that applied to only one would
        /// be a draft and a refine from two different models. Empty is a render against
        /// the base weights alone.
        public var adapters: [LoRAOverlay.Request] = []

        /// Same choice, same reasoning, as ``Renderer/Spec/attentionPath`` — and it bites
        /// harder here, because stage 2 runs at the full output size with no guidance to
        /// hide behind, so the peak of a two-stage render *is* one stage-2 attention.
        public var attentionPath: DiTAttention.AttentionPath

        public init(prompt: String,
                    geometry: DiTForward.Geometry = Renderer.Spec.productionGeometry,
                    seed: UInt64 = 0,
                    attentionPath: DiTAttention.AttentionPath = .fused) {
            self.prompt = prompt
            self.geometry = geometry
            self.seed = seed
            self.attentionPath = attentionPath
        }

        /// Half the output width and half the output height.
        ///
        /// Everything else is carried across unchanged — frames, frame rate, the audio
        /// constants, the positional-embedding maxima. The audio stream in particular is
        /// identical in both stages: its latent shape comes from the frame count and the
        /// fps, never from the picture size.
        public var stage1Geometry: DiTForward.Geometry {
            var g = geometry
            g.width = geometry.width / 2
            g.height = geometry.height / 2
            return g
        }

        public var stage1Sigmas: [Float] { FlowSchedule.distilledStage1 }
        public var stage2Sigmas: [Float] { FlowSchedule.distilledStage2 }
        /// 8 and 3 — `count - 1`, which is what `Sampler.run(explicitSigmas:)` derives.
        public var stage1Steps: Int { stage1Sigmas.count - 1 }
        public var stage2Steps: Int { stage2Sigmas.count - 1 }

        /// Stage 2's noise scale: its first sigma, 0.909375, for **both** streams.
        public var stage2NoiseScale: Float { stage2Sigmas[0] }

        /// The two-stage resolution check, then both stages' own lattice checks.
        public func validate() throws {
            guard geometry.width % 64 == 0, geometry.height % 64 == 0 else {
                throw Failure.resolution(width: geometry.width, height: geometry.height)
            }
            try geometry.validate()
            try stage1Geometry.validate()
        }
    }

    // MARK: - Phase one: text

    /// One branch, not two.
    ///
    /// This pipeline encodes a single prompt and uses the result in both stages.
    /// `Renderer.conditioning` encodes two branches because a guided render needs the
    /// negative one; this path never runs an unconditional pass, so it encodes one and
    /// hands the same tensors to `DiTDenoiser`'s `positive` and `negative` slots. The
    /// `negative` slot is reachable only from `Sampler.Pass.unconditional`, which
    /// ``neutralGuidance`` never plans.
    public static func conditioning(spec: Spec, checkpoints: Checkpoints) throws
        -> Renderer.TextBranches {
        try spec.validate()
        let text = try TextConditioningPipeline(textEncoder: checkpoints.base.textEncoder,
                                                dit: checkpoints.base.dit)
        let encoding = try text.encode(prompt: spec.prompt, branch: 0)
        let conditioning = Pipeline.Conditioning(
            video: try Renderer.require(encoding.features[.video], "features.video"),
            audio: try Renderer.require(encoding.features[.audio], "features.audio"))
        MLX.eval(conditioning.video, conditioning.audio)
        return Renderer.TextBranches(positive: conditioning, negative: conditioning,
                                     validTokenCounts: [encoding.validTokenCount])
    }

    // MARK: - Phase one and a half: the image anchor, once per stage

    /// One conditioning image → **two** conditioning pairs, one per stage.
    ///
    /// The fact worth stating outright, because it is the one an inference would get
    /// wrong: the two pairs are built **separately**, each by resizing the source image to
    /// *its own stage's* pixel size and encoding that with the video VAE. There is no
    /// downscale of a latent anywhere in it.
    ///
    /// So the source image is resized twice — to `width/2, height/2` and to `width, height`
    /// — and encoded twice. Resizing once and pooling the latent would be cheaper, would
    /// produce the right shapes, and would be a different tensor: the VAE's spatial
    /// compression is not an average, and the half-resolution encode of an image is not the
    /// downsample of its full-resolution encode.
    ///
    /// The file is decoded once and resized from those original pixels for each stage.
    /// Resizing stage 2's output down to stage 1 would compose two filters and is not the
    /// same operator.
    ///
    /// **`--strength` applies to both stages.** The same conditioning list is applied in
    /// both, so the anchor is re-imposed at stage 2 rather than being a stage-1-only hint:
    /// `denoiseMask = 1 - strength` on the conditioned latent frame's tokens, in each
    /// stage's own state. Dropping it from stage 2 would let the three refinement steps
    /// walk off the anchor at full resolution, which is exactly where it would be visible.
    ///
    /// - Parameters:
    ///   - url: the still. See ``LTXPipeline/MediaInput/image(at:height:width:alpha:)`` for
    ///     the resize filter.
    ///   - encoder: loaded once and used for both stages.
    public static func imageConditioning(
        at url: URL, strength: Double, spec: Spec, encoder: VideoVAEEncoder
    ) throws -> (stage1: RenderConditioning.Built, stage2: RenderConditioning.Built) {
        guard let pair = try imageConditioning([ImageConditioning(url: url)],
                                               defaultStrength: strength,
                                               spec: spec, encoder: encoder) else {
            // Unreachable: the list above is never empty.
            throw Failure.conditioningGeometry(stage: 1, what: "image conditioning pair",
                                               got: [], expected: [1])
        }
        return pair
    }

    /// Several stills, each pinned to a latent frame, built for both stages.
    ///
    /// Stage 1 samples at half the spatial resolution and the **same** frame count, so a
    /// keyframe's position is identical in both stages and only its pixels are resized. Each
    /// file is decoded once and resized twice from those original pixels, never once and
    /// then downsampled again — composing two bilinear filters is a different operator and a
    /// visibly softer draft.
    ///
    /// Returns `nil` for an empty list, so a caller can hand it whatever the request had.
    public static func imageConditioning(
        _ images: [ImageConditioning], defaultStrength: Double, spec: Spec,
        encoder: VideoVAEEncoder
    ) throws -> (stage1: RenderConditioning.Built, stage2: RenderConditioning.Built)? {
        guard !images.isEmpty else { return nil }

        var stage1Keys: [RenderConditioning.Keyframe] = []
        var stage2Keys: [RenderConditioning.Keyframe] = []
        for image in images {
            let pixels = try conditioningPixels(at: image.url, spec: spec)
            let strength = image.strength ?? defaultStrength
            stage1Keys.append(RenderConditioning.Keyframe(
                latent: try RenderConditioning.encodeImage(pixels.stage1, encoder: encoder),
                pixelFrame: image.pixelFrame, strength: strength))
            stage2Keys.append(RenderConditioning.Keyframe(
                latent: try RenderConditioning.encodeImage(pixels.stage2, encoder: encoder),
                pixelFrame: image.pixelFrame, strength: strength))
        }

        func built(_ keys: [RenderConditioning.Keyframe], _ geometry: DiTForward.Geometry,
                   _ stage: Int) throws -> RenderConditioning.Built {
            guard let pair = try RenderConditioning.build(keyframes: keys,
                                                          strength: defaultStrength,
                                                          geometry: geometry) else {
                // Unreachable: `build` returns nil only when every conditioning input is nil.
                throw Failure.conditioningGeometry(stage: stage,
                                                   what: "image conditioning pair",
                                                   got: [], expected: [1])
            }
            return pair
        }

        return (stage1: try built(stage1Keys, spec.stage1Geometry, 1),
                stage2: try built(stage2Keys, spec.geometry, 2))
    }

    /// The two resized conditioning frames, before either meets the VAE.
    ///
    /// Split out from ``imageConditioning(at:strength:spec:encoder:)`` for the same reason
    /// ``stage2Initial(upscaledVideoTokens:stage1Audio:noise:noiseScale:conditioning:)`` is
    /// split out of ``render(spec:checkpoints:text:stage1Conditioning:stage2Conditioning:observer:)``:
    /// it is the part of the per-stage conditioning that can be **executed without a 1.4 GB
    /// checkpoint**, and therefore the part a test can hold. What a test can then assert is
    /// the whole claim that distinguishes this implementation from the wrong one — that
    /// there are two frames, at two sizes, from one decode of one file.
    ///
    /// Returns `[1, H, W, 3]` in `[0, 1]` per stage, ready for
    /// ``LTXModules/VideoVAEEncoder/pixels(fromRGB:)``.
    public static func conditioningPixels(at url: URL, spec: Spec) throws
        -> (stage1: MLXArray, stage2: MLXArray) {
        try spec.validate()
        // Decoded ONCE and resized from those original pixels twice. Resizing stage 2's
        // frame down to stage 1's would compose two bilinear filters, which is a different
        // operator and a visibly softer draft.
        let decoded = try MediaInput.image(at: url)
        let g1 = spec.stage1Geometry, g2 = spec.geometry
        return (stage1: try MediaInput.resizeAndCenterCrop(decoded, height: g1.height,
                                                           width: g1.width),
                stage2: try MediaInput.resizeAndCenterCrop(decoded, height: g2.height,
                                                           width: g2.width))
    }

    // MARK: - Phase two: the two stages

    /// Called once per completed step, with the 1-based stage and the step's index within it.
    public typealias Observer = (_ stage: Int, _ step: Int, _ stepsInStage: Int) throws -> Void

    /// Both stages, the upsample between them, and both decodes.
    ///
    /// - Parameters:
    ///   - stage1Conditioning: built at ``Spec/stage1Geometry``.
    ///   - stage2Conditioning: built at the full geometry, and **rebuilt** rather than
    ///     reused: the same source image re-encoded at the output resolution. Passing
    ///     stage 1's pair here would be a token-count mismatch on video and — because the
    ///     audio pair is the same shape in both stages — a silent success on audio.
    public static func render(spec: Spec, checkpoints: Checkpoints,
                              text: Renderer.TextBranches,
                              stage1Conditioning: RenderConditioning.Built? = nil,
                              stage2Conditioning: RenderConditioning.Built? = nil,
                              observer: Observer? = nil) throws -> Renderer.Result {
        try spec.validate()
        try checkpoints.verifyPresent()
        try checkConditioning(stage1Conditioning, geometry: spec.stage1Geometry, stage: 1)
        try checkConditioning(stage2Conditioning, geometry: spec.geometry, stage: 2)

        let stage1Geometry = spec.stage1Geometry
        let sampler = Sampler(video: neutralGuidance, audio: neutralGuidance)
        let ancestral = shouldUseAncestralSampler(transformer: checkpoints.base.dit)
        let stepper: Sampler.Stepper = ancestral
            ? .ancestral(eta: ancestralEta, sNoise: ancestralSNoise,
                         seed: spec.seed &+ ancestralNoiseSeedOffset)
            : .euler

        // The four initial draws. See ``stage2NoiseSeedOffset`` for why stage 2 gets its
        // own key rather than a later position in one stream.
        let stage1Noise = try RenderNoise(seed: spec.seed, geometry: stage1Geometry)
        let stage2Noise = try RenderNoise(seed: spec.seed &+ stage2NoiseSeedOffset,
                                          geometry: spec.geometry)

        let (videoLatent, audioLatent): (MLXArray, MLXArray) = try {
            // **One transformer load for both stages.** Building and freeing the
            // transformer per stage would pay 42 GB of load twice and hold the upsampler
            // alone in between. That is a memory strategy, not arithmetic: the two stages
            // run the same weights on different shapes. Holding them across the upsample
            // costs 42 + 1 GB resident and saves a second read of a 42 GB file.
            let header = try SafetensorsHeader.read(from: checkpoints.base.dit)
            let topology = try TransformerTopology.read(header)
            try topology.verifyCrossModalWiring(header)
            let weights = try MLX.loadArrays(url: checkpoints.base.dit)
            var forward = DiTForward(weights: weights, topology: topology,
                                     attentionPath: spec.attentionPath)
            // Both stages share this forward, so attaching here attaches to both — which is
            // the only correct answer: a draft and a refine from different weights is not a
            // two-stage render of anything.
            if !spec.adapters.isEmpty {
                let overlay = try LoRAOverlay.load(spec.adapters)
                try overlay.validate(againstBaseWeights: weights)
                forward.attach(lora: overlay)
            }
            let head = DiTOutputHead(weights: weights, topology: topology)

            // Prefault the upsampler file and pull the two VAE statistic
            // vectors while stage 1 occupies the GPU. Do **not** construct
            // `LatentSpatialUpsampler` here: that calls `MLX.loadArrays` and
            // MLX is not safe to share with the sampler thread. The upsample
            // itself scales with shape and dominates at 7.56 s, so this only
            // hides page-in of a 327 MB mmap.
            let prefetch = UpsamplerPrefetch()
            let upsamplerGate = DispatchSemaphore(value: 0)
            DispatchQueue.global(qos: .userInitiated).async {
                prefetch.result = Result { try Self.prefetchUpsampler(checkpoints) }
                upsamplerGate.signal()
            }

            func denoiser(_ geometry: DiTForward.Geometry) -> Pipeline.DiTDenoiser {
                Pipeline.DiTDenoiser(forward: forward, head: head, geometry: geometry,
                                     positive: text.positive, negative: text.negative,
                                     videoSTGBlocks: [], audioSTGBlocks: [])
            }

            // ---- Stage 1 -------------------------------------------------------------
            //
            // A fresh noised state at scale 1.0 with no initial latent: zeros as the start,
            // so the first `lerp` returns the noise unchanged, then the mask blend.
            let drawn1 = stage1Noise.streamsForRenderOnly()
            let seeded1 = Sampler.Streams(
                video: Sampler.seedConditioned(
                    noise: drawn1.video,
                    denoiseMask: stage1Conditioning?.denoiseMask.video,
                    cleanLatent: stage1Conditioning?.cleanLatent.video),
                audio: Sampler.seedConditioned(
                    noise: drawn1.audio,
                    denoiseMask: stage1Conditioning?.denoiseMask.audio,
                    cleanLatent: stage1Conditioning?.cleanLatent.audio))

            let stage1: Sampler.Trajectory
            do {
                stage1 = try sampler.run(
                    steps: spec.stage1Steps, initial: seeded1,
                    denoiser: denoiser(stage1Geometry),
                    denoiseMask: stage1Conditioning?.denoiseMask,
                    cleanLatent: stage1Conditioning?.cleanLatent,
                    explicitSigmas: spec.stage1Sigmas,
                    stepper: stepper,
                    retainHistory: false,
                    observer: { step, _ in try observer?(1, step, spec.stage1Steps) })
            } catch {
                upsamplerGate.wait()
                throw error
            }
            upsamplerGate.wait()

            // ---- Between the stages --------------------------------------------------
            //
            // VIDEO ONLY.
            // The sampler works in token space and the upsampler wants NCDHW, so the round
            // trip is unpatchify -> upsample -> patchify. `PatchifyRoundTripTests` shows the
            // two are bit-exact inverses, so the detour costs nothing numerically.
            //
            // The audio latent below is stage 1's, carried by value. It does not go near
            // the upsampler and is not redrawn.
            let stage1VideoLatent = try Sampler.unpatchifyVideo(
                stage1.latents.video, latentFrames: stage1Geometry.latentFrames,
                height: stage1Geometry.height / stage1Geometry.latent.spatialScale,
                width: stage1Geometry.width / stage1Geometry.latent.spatialScale)
            let upscaled: MLXArray = try {
                let stats = try prefetch.result!.get()
                let upsampler = try LatentSpatialUpsampler(
                    checkpoint: checkpoints.upsampler,
                    statistics: (std: MLXArray(stats.std), mean: MLXArray(stats.mean)))
                let out = try upsampler.upsample(stage1VideoLatent)
                MLX.eval(out)
                return out
            }()
            let upscaledTokens = try Sampler.patchifyVideo(upscaled)

            // ---- Stage 2 -------------------------------------------------------------
            //
            // Stage 2's noise scale applies to BOTH streams: video from the upsampler,
            // audio from stage 1 unchanged. A noiser at a non-unit scale is
            // `lerp(initial, noise, 0.909375)` — a move *toward* noise, not an addition of
            // it, and the entire mechanism by which the refine continues the draft instead
            // of starting over.
            let seeded2 = stage2Initial(
                upscaledVideoTokens: upscaledTokens,
                stage1Audio: stage1.latents.audio,
                noise: stage2Noise.streamsForRenderOnly(),
                noiseScale: spec.stage2NoiseScale,
                conditioning: stage2Conditioning)

            // Deterministic Euler, always: a 3-step refinement schedule is too short to
            // remove freshly injected noise.
            let stage2 = try sampler.run(
                steps: spec.stage2Steps, initial: seeded2,
                denoiser: denoiser(spec.geometry),
                denoiseMask: stage2Conditioning?.denoiseMask,
                cleanLatent: stage2Conditioning?.cleanLatent,
                explicitSigmas: spec.stage2Sigmas,
                stepper: .euler,
                retainHistory: false,
                observer: { step, _ in try observer?(2, step, spec.stage2Steps) })

            return (try Sampler.unpatchifyVideo(
                        stage2.latents.video, latentFrames: spec.geometry.latentFrames,
                        height: spec.geometry.height / spec.geometry.latent.spatialScale,
                        width: spec.geometry.width / spec.geometry.latent.spatialScale),
                    Sampler.unpatchifyAudio(stage2.latents.audio,
                                            channels: spec.geometry.latent.audioChannels))
        }()
        MLX.Memory.clearCache()

        // ---- The decodes, which are the single-stage path's -------------------------
        let videoRGB: MLXArray = try {
            let videoDecoder = try VideoVAEDecoder(checkpoint: checkpoints.base.videoVAE)
            let rgb = try Renderer.decodeVideoOnRecordedPolicy(videoDecoder, videoLatent,
                                                                geometry: spec.geometry)
            MLX.eval(rgb)
            return rgb
        }()
        MLX.Memory.clearCache()

        let audioMel: MLXArray = try {
            let audioDecoder = try AudioVAEDecoder(checkpoint: checkpoints.base.audioVAE)
            let mel = try audioDecoder.decode(audioLatent)
            MLX.eval(mel)
            return mel
        }()
        MLX.Memory.clearCache()

        let vocoderConfig = try AudioVocoder.configuration(
            from: try SafetensorsHeader.read(from: checkpoints.base.audioVAE))
        let audioWaveform: MLXArray = try {
            let vocoder = try AudioVocoder(checkpoint: checkpoints.base.audioVAE)
            let wave = try vocoder.waveform(fromMel: audioMel)
            MLX.eval(wave)
            return wave
        }()

        return Renderer.Result(
            videoRGB: videoRGB, audioMel: audioMel, audioWaveform: audioWaveform,
            audioSampleRate: vocoderConfig.outputSamplingRate,
            provenance: provenance(spec: spec, checkpoints: checkpoints, text: text,
                                   ancestral: ancestral,
                                   stage1Noise: stage1Noise, stage2Noise: stage2Noise))
    }

    // MARK: - The handoff

    /// Stage 2's initial latents: a noised state for both streams, at stage 2's noise
    /// scale, each with its own initial latent.
    ///
    /// A separate function because it is the one piece of the two-stage feature that can be
    /// **executed without a 42 GB checkpoint**, and therefore the one piece a test can hold.
    /// `DistilledTwoStageTests` drives it with marker tensors and checks two things a whole
    /// render cannot: that the arithmetic is `lerp(initial, noise, 0.909375)` to the bit,
    /// and that the **audio** stream is a function of `stage1Audio` and never of
    /// `upscaledVideoTokens`.
    ///
    /// That second check is the point. The two streams are noised in one breath with the
    /// same scale, and only their initial latents differ — video from the upsampler, audio
    /// from stage 1. Making both read the upscaled tensor is a two-character symmetry error
    /// that a shape check catches only by luck: at production the video stream is
    /// `[1, 3120, 128]` and the audio `[1, 101, 128]`, so it would throw — but at a
    /// geometry where the token counts happened to agree it would not, and the render would
    /// be an image sequence played as sound.
    ///
    /// - Parameters:
    ///   - upscaledVideoTokens: the upsampler's output, patchified. **Video only.**
    ///   - stage1Audio: stage 1's final audio latent, in token space, **unmodified**. It has
    ///     not been through the upsampler and there is no audio upsampler to put it through.
    ///   - noise: stage 2's own draw, at stage 2's shapes.
    ///   - noiseScale: stage 2's first sigma, 0.909375, the same for both streams.
    public static func stage2Initial(upscaledVideoTokens: MLXArray,
                                     stage1Audio: MLXArray,
                                     noise: Sampler.Streams,
                                     noiseScale: Float,
                                     conditioning: RenderConditioning.Built?)
        -> Sampler.Streams {
        Sampler.Streams(
            video: Sampler.seedConditioned(
                noise: noise.video,
                denoiseMask: conditioning?.denoiseMask.video,
                cleanLatent: conditioning?.cleanLatent.video,
                initial: upscaledVideoTokens.asType(noise.video.dtype),
                noiseScale: noiseScale),
            audio: Sampler.seedConditioned(
                noise: noise.audio,
                denoiseMask: conditioning?.denoiseMask.audio,
                cleanLatent: conditioning?.cleanLatent.audio,
                initial: stage1Audio.asType(noise.audio.dtype),
                noiseScale: noiseScale))
    }

    /// A conditioning pair is built for one geometry; refuse a mismatched one at the seam
    /// rather than deep inside `postProcess`, where a broadcast can hide it.
    static func checkConditioning(_ built: RenderConditioning.Built?,
                                  geometry: DiTForward.Geometry, stage: Int) throws {
        guard let built else { return }
        let video = [1, geometry.videoTokens, 1]
        guard built.denoiseMask.video.shape == video else {
            throw Failure.conditioningGeometry(stage: stage, what: "video denoise mask",
                                               got: built.denoiseMask.video.shape,
                                               expected: video)
        }
        let audio = [1, geometry.audioTokens, 1]
        guard built.denoiseMask.audio.shape == audio else {
            throw Failure.conditioningGeometry(stage: stage, what: "audio denoise mask",
                                               got: built.denoiseMask.audio.shape,
                                               expected: audio)
        }
    }

    // MARK: - Provenance

    static func provenance(spec: Spec, checkpoints: Checkpoints,
                           text: Renderer.TextBranches, ancestral: Bool,
                           stage1Noise: RenderNoise, stage2Noise: RenderNoise)
        -> [String: String] {
        let g1 = spec.stage1Geometry
        let g2 = spec.geometry
        func sigmas(_ s: [Float]) -> String { s.map { String($0) }.joined(separator: ",") }
        func latentShape(_ g: DiTForward.Geometry) -> String {
            let h: Int = g.height / g.latent.spatialScale
            let w: Int = g.width / g.latent.spatialScale
            return "[1, \(g.latent.videoChannels), \(g.latentFrames), \(h), \(w)]"
        }

        let stage1Sampler: String
        if ancestral {
            stage1Sampler = "EulerAncestralDiffusionStep(eta=\(ancestralEta), "
                + "s_noise=\(ancestralSNoise)), loop noise seed = \(spec.seed) + "
                + "\(ancestralNoiseSeedOffset)"
        } else {
            let minimum = ancestralSamplerSinceVersion.map { String($0) }.joined(separator: ".")
            stage1Sampler = "deterministic Euler (the checkpoint's model_version is below "
                + minimum + ")"
        }

        var manifest: [String: String] = [:]
        manifest["kind"] = "a rendered clip — judged by eye, not by measurement"
        manifest["pipeline"] = "two-stage distilled"
        manifest["seed_note"] = "Both initial latents and the stage-1 ancestral noise were "
            + "drawn by MLX's own RNG, so a render is reproducible from its seed on this "
            + "machine."
        manifest["prompt"] = spec.prompt
        manifest["negative_prompt"] = "none — no guider is constructed and no unconditional "
            + "pass runs, so only one text branch is encoded (text_branches: 1)"
        manifest["valid_token_counts"] = text.validTokenCounts
            .map { String($0) }.joined(separator: ",")
        manifest["seed"] = String(spec.seed)
        manifest["stage_1"] = "\(g1.width)x\(g1.height), \(spec.stage1Steps) steps, sigmas "
            + sigmas(spec.stage1Sigmas) + ", " + stage1Sampler
        manifest["stage_1_noise"] = stage1Noise.provenance
        manifest["upsample"] = "video latent only, x2 spatial: " + latentShape(g1) + " -> "
            + latentShape(g2) + ". The AUDIO latent passes through untouched."
        manifest["upsampler_checkpoint"] = checkpoints.upsampler.lastPathComponent
        manifest["stage_2"] = "\(g2.width)x\(g2.height), \(spec.stage2Steps) steps, sigmas "
            + sigmas(spec.stage2Sigmas) + ", deterministic Euler, noise_scale "
            + "\(spec.stage2NoiseScale) on BOTH streams (video from the upsampler, audio "
            + "from stage 1)"
        manifest["stage_2_noise"] = stage2Noise.provenance + " (key offset "
            + "\(stage2NoiseSeedOffset), so stage 2 draws independently of stage 1)"
        manifest["guidance"] = "none. SimpleDenoiser: cfg 1.0, stg 0.0, modality 1.0, "
            + "rescale 0.0 -> exactly one transformer forward per step, "
            + "\(spec.stage1Steps + spec.stage2Steps) for the render"
        manifest["geometry"] = "\(g2.width)x\(g2.height)x\(g2.frames) @ \(g2.frameRate) fps"
        manifest["attention"] = Renderer.attentionProvenance(spec.attentionPath)
        manifest["video_decode"] = Renderer.videoDecodeProvenance(
            spec: Renderer.Spec(prompt: spec.prompt, geometry: g2))
        manifest["audio"] = "mel -> vocoder -> 48 kHz stereo waveform. PERCEPTUAL ONLY: "
            + "contract 11 forbids diffing a vocoder waveform, and no gate consumes it."
        return manifest
    }

    /// File I/O only — no `MLXArray`. Touches every page of the upsampler mmap
    /// and materialises the two 128-float VAE statistic vectors the constructor
    /// would otherwise read after stage 1.
    private static func prefetchUpsampler(_ checkpoints: Checkpoints)
        throws -> (std: [Float], mean: [Float]) {
        try prefault(checkpoints.upsampler)
        let reader = try SafetensorsReader(url: checkpoints.base.videoVAE)
        let prefix = LatentSpatialUpsampler.statisticsPrefix
        return (try reader.floats(prefix + "std-of-means"),
                try reader.floats(prefix + "mean-of-means"))
    }

    private static func prefault(_ url: URL) throws {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        var offset = 0
        while offset < data.count {
            _ = data[offset]
            offset += 4096
        }
    }
}

/// Host for I/O overlapped with stage 1. Must not hold MLX types.
private final class UpsamplerPrefetch: @unchecked Sendable {
    var result: Result<(std: [Float], mean: [Float]), Error>?
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX

/// Prompt in, frames out — the whole trajectory, composed.
///
/// ## Why the encoder is loaded, used, and dropped before the DiT arrives
///
/// Gemma is 26 GB and the transformer is 42 GB. Both fit in 256 GB alongside production
/// activations, but only just, and holding the encoder for the length of a 30-step render
/// buys nothing: the text features are computed once, before the first sampler pass, and
/// are never consulted again. `conditioning(...)` is therefore a separate call that
/// returns tensors rather than a stored encoder, and `render(...)` takes those tensors.
/// Fusing the two into one function that held both checkpoints would work on this machine
/// and fail on a smaller one, for no gain.
///
/// ## Audio is vocoded on this path
///
/// Contract 11 forbids *diffing* the vocoder waveform, not omitting the vocoder.
/// `Result.audioWaveform` is written to WAV and muxed.
public enum Renderer {

    public enum Failure: Error, CustomStringConvertible {
        case missingCheckpoint(String, path: String)
        case steps(Int)
        case unexpectedLatentShape(String, got: [Int], expected: [Int])

        public var description: String {
            switch self {
            case let .missingCheckpoint(what, path):
                return "the \(what) checkpoint is not at \(path)"
            case let .steps(n):
                return "steps must be positive, got \(n)"
            case let .unexpectedLatentShape(what, got, expected):
                return "\(what) is \(got), expected \(expected)"
            }
        }
    }

    /// The checkpoints a render needs, and nothing else.
    ///
    /// Paths rather than loaded weights: the whole point of the two-phase structure above
    /// is that the encoder's 26 GB is resident only while it is being used.
    public struct Checkpoints {
        public var textEncoder: URL
        public var dit: URL
        public var videoVAE: URL
        public var audioVAE: URL

        public init(textEncoder: URL, dit: URL, videoVAE: URL, audioVAE: URL) {
            self.textEncoder = textEncoder
            self.dit = dit
            self.videoVAE = videoVAE
            self.audioVAE = audioVAE
        }

        /// Fails now, naming the file and verifying it is a valid safetensors file
        /// containing the expected components, rather than 26 GB into a load.
        public func verifyPresent() throws {
            let fm = FileManager.default
            for (what, url, expectedComponent) in [
                ("text encoder", textEncoder, LTXComponent.textEncoder),
                ("DiT", dit, LTXComponent.transformer),
                ("video VAE", videoVAE, nil as LTXComponent?),
                ("audio VAE", audioVAE, LTXComponent.audioVAE)
            ] {
                guard fm.fileExists(atPath: url.path) else {
                    throw Failure.missingCheckpoint(what, path: url.path)
                }
                do {
                    let file = try CheckpointIdentity.identify(fileAt: url)
                    if let expected = expectedComponent {
                        guard file.components.contains(expected) else {
                            throw Failure.missingCheckpoint("valid \(what) (missing \(expected.rawValue) tensors)", path: url.path)
                        }
                    } else if what == "video VAE" {
                        if file.components.contains(.videoVAENADiffusion) {
                            throw Failure.missingCheckpoint(
                                "causal video VAE (this file is the NA-diffusion decoder; v1 has no na3d kernel)",
                                path: url.path)
                        }
                        guard file.components.contains(.videoVAECausal) else {
                            throw Failure.missingCheckpoint("valid video VAE (missing video VAE tensors)", path: url.path)
                        }
                    }
                } catch {
                    throw Failure.missingCheckpoint("readable \(what) safetensors file (error: \(error.localizedDescription))", path: url.path)
                }
            }
        }
    }

    /// Everything that decides what gets rendered.
    ///
    /// The defaults are the production settings: 640x384x97 at 24 fps, 30 steps, video
    /// cfg 3.0 / audio cfg 7.0, STG 1.0 on block 28, modality 3.0, rescale 0.7. Changing
    /// any of them is legitimate; they are defaults so that a render with nothing
    /// specified lands on the settings this model was tuned at.
    public struct Spec {
        public var prompt: String
        public var negativePrompt: String
        public var geometry: DiTForward.Geometry
        public var steps: Int
        public var seed: UInt64
        public var videoGuidance: GuidanceParams
        public var audioGuidance: GuidanceParams
        /// Where attention accumulates its softmax, defaulted **the other way from the
        /// rest of this package**.
        ///
        /// `DiTAttention.attentionPath` defaults to `.explicit` because at small shapes it
        /// is marginally the more accurate of the two. A render is a different regime: it
        /// runs the same modules at 3x to 6x the token count, and `.explicit` materialises
        /// a `[1, 32, T, T]` fp32 score matrix whose size is quadratic in `T`. At
        /// 576x1024x241 that is 81.6 GB for a single self-attention call and the render
        /// dies on SIGKILL; at production shape it is 2.5 GB and nobody notices.
        ///
        /// So this defaults to `.fused`. The measured disagreement between the two paths
        /// is `1.11e-04` on `attn1` and exactly zero on `audio_attn1`.
        ///
        /// Deliberately **not** a CLI flag. `LTXRecipes` is the module the CLI's `Request`
        /// lives in and it is MLX-free by design, so exposing this there would mean either
        /// dragging MLX across that boundary or minting a second, stringly-typed copy of
        /// the enum. The setting is reachable from code, which is where the two paths get
        /// compared, and no render wants `.explicit` at a shape where the choice is
        /// visible.
        public var attentionPath: DiTAttention.AttentionPath
        /// 0 disables the cache (the default, until a quality sweep fills it in).
        /// The measured knee is 0.10; that is a starting point, not a claim.
        public var cacheThreshold: Double
        public var cacheMaxSkips: Int

        /// Adapters applied as a residual overlay on top of the base weights. Defaults to
        /// empty, which is a render against the base weights alone.
        ///
        /// Overlay, never bake — `LoRAOverlay` explains the 9.7 GB reason. The resident cost
        /// is the adapter file itself, which for the rank-450 distilled LoRA is 8.29 GB on
        /// top of the DiT's 42.06 and is **not** modelled by `MemoryPlan`.
        public var adapters: [LoRAOverlay.Request]

        public static let productionGeometry = DiTForward.Geometry(
            frames: 97, height: 384, width: 640, frameRate: 24)

        /// The production video guidance.
        public static let productionVideoGuidance = GuidanceParams(
            cfgScale: 3.0, stgScale: 1.0, stgBlocks: [28],
            rescaleScale: 0.7, modalityScale: 3.0, skipStep: 0)

        /// The production audio guidance — the same shape, a different cfg scale. Dual CFG
        /// is two independent scales, not one applied twice; see contract 7.
        public static let productionAudioGuidance = GuidanceParams(
            cfgScale: 7.0, stgScale: 1.0, stgBlocks: [28],
            rescaleScale: 0.7, modalityScale: 3.0, skipStep: 0)

        public init(prompt: String, negativePrompt: String = "",
                    geometry: DiTForward.Geometry = Spec.productionGeometry,
                    steps: Int = 30, seed: UInt64 = 0,
                    videoGuidance: GuidanceParams = Spec.productionVideoGuidance,
                    audioGuidance: GuidanceParams = Spec.productionAudioGuidance,
                    attentionPath: DiTAttention.AttentionPath = .fused,
                    cacheThreshold: Double = 0,
                    cacheMaxSkips: Int = StepCachePolicy.measuredCap,
                    adapters: [LoRAOverlay.Request] = []) {
            self.attentionPath = attentionPath
            self.cacheThreshold = cacheThreshold
            self.cacheMaxSkips = cacheMaxSkips
            self.adapters = adapters
            self.prompt = prompt
            self.negativePrompt = negativePrompt
            self.geometry = geometry
            self.steps = steps
            self.seed = seed
            self.videoGuidance = videoGuidance
            self.audioGuidance = audioGuidance
        }
    }

    /// Both text branches, as the sampler wants them.
    public struct TextBranches {
        public var positive: Pipeline.Conditioning
        public var negative: Pipeline.Conditioning
        /// How many of the 1024 padded positions carried a real token, per branch. Worth
        /// carrying into the manifest: contract 9 forces `min_length = 1024`, so a prompt
        /// that tokenised to nothing at all still produces full-shaped features and a
        /// perfectly ordinary-looking render.
        public var validTokenCounts: [Int]
    }

    public struct Result: @unchecked Sendable {
        /// `[F, H, W, 3]`, channels-last, `[0, 1]` — what `RenderOutput.writeVideo` takes.
        public var videoRGB: MLXArray
        /// The decoded mel, `[B, 2, T, 64]`. Still returned alongside the waveform
        /// because contract 11 forbids diffing the waveform: anything that wants to check
        /// the audio numerically must reach for this tensor and not the one below it.
        public var audioMel: MLXArray
        /// `[B, 2, samples]` at 48 kHz, from the vocoder.
        ///
        /// **Perceptual only.** Contract 11: identical mel inputs are not guaranteed to
        /// produce identical waveforms, because the transposed convolutions' algorithm
        /// selection is not pinned. It exists so a person can listen.
        public var audioWaveform: MLXArray
        /// 48000. Read from the vocoder's configuration rather than assumed, because the
        /// bandwidth-extension stage is what sets it and a checkpoint could ship without
        /// one.
        public var audioSampleRate: Int
        /// Everything needed to re-render this exact clip.
        public var provenance: [String: String]
    }

    // MARK: - Phase one: text

    /// Encode both guidance branches, then let the 26 GB encoder go.
    ///
    /// `TextConditioningPipeline` reads the connector weights from the transformer
    /// checkpoint and keeps only those keys. The rest of the 42 GB map is dropped
    /// before Gemma runs, so this phase's resident cost is the encoder's.
    public static func conditioning(spec: Spec, checkpoints: Checkpoints) throws
        -> TextBranches {
        try spec.geometry.validate()
        let text = try TextConditioningPipeline(textEncoder: checkpoints.textEncoder,
                                                dit: checkpoints.dit)
        let branches = try text(prompt: spec.prompt, negativePrompt: spec.negativePrompt)

        // Branch order is fixed: 0 positive, 1 negative. `branchPrompts` owns that
        // convention — including that an empty negative prompt is not an empty
        // sequence but the pipeline's default negative at 223 valid tokens (contract 9).
        func conditioning(_ e: TextConditioningPipeline.Encoding) throws
            -> Pipeline.Conditioning {
            Pipeline.Conditioning(video: try require(e.features[.video], "features.video"),
                                  audio: try require(e.features[.audio], "features.audio"))
        }
        let positive = try conditioning(branches[0])
        let negative = try conditioning(branches[1])
        // Force the features off the connector graph so the 26 GB encoder (and the
        // connector slice of the DiT) can actually leave when this function returns.
        MLX.eval(positive.video, positive.audio, negative.video, negative.audio)
        return TextBranches(positive: positive, negative: negative,
                            validTokenCounts: branches.map(\.validTokenCount))
    }

    // MARK: - Phase two: the trajectory and the decode

    /// The sampling loop and both VAE decodes, from encoded text.
    ///
    /// `observer` is called once per completed step with the step index, so a caller can
    /// print progress. A production render is 30 steps of four passes through 48 blocks;
    /// without it the process is silent for a long time and indistinguishable from hung.
    public static func render(spec: Spec, checkpoints: Checkpoints, text: TextBranches,
                               conditioning: RenderConditioning.Built? = nil,
                               observer: ((Int, Int) throws -> Void)? = nil) throws -> Result {
        try spec.geometry.validate()
        guard spec.steps > 0 else { throw Failure.steps(spec.steps) }
        try checkpoints.verifyPresent()

        let noise = try RenderNoise(seed: spec.seed, geometry: spec.geometry)
        let sampler = Sampler(video: spec.videoGuidance, audio: spec.audioGuidance)

        let (videoLatent, audioLatent, cacheNote): (MLXArray, MLXArray, String) = try {
            let topology = try TransformerTopology.read(
                try SafetensorsHeader.read(from: checkpoints.dit))
            try topology.verifyCrossModalWiring(try SafetensorsHeader.read(from: checkpoints.dit))
            let weights = try MLX.loadArrays(url: checkpoints.dit)

            let caches: [Sampler.Pass: StepCache] = spec.cacheThreshold > 0
                ? Dictionary(uniqueKeysWithValues: Sampler.Pass.allCases.map {
                    ($0, StepCache(pass: $0.description, threshold: spec.cacheThreshold,
                                   maxConsecutiveSkips: spec.cacheMaxSkips))
                })
                : [:]

            // Validated against the base weights before a single forward. An adapter naming
            // a weight this checkpoint does not have is a wrong-adapter mistake, and reading
            // the key set costs less than discovering it thirty steps in — the same argument
            // `VideoUpscaler.referenceGuided` makes for its IC-LoRA.
            var forward = DiTForward(weights: weights, topology: topology,
                                     attentionPath: spec.attentionPath)
            if !spec.adapters.isEmpty {
                let overlay = try LoRAOverlay.load(spec.adapters)
                try overlay.validate(againstBaseWeights: weights)
                forward.attach(lora: overlay)
            }

            let denoiser = Pipeline.DiTDenoiser(
                forward: forward,
                head: DiTOutputHead(weights: weights, topology: topology),
                geometry: spec.geometry,
                positive: text.positive, negative: text.negative,
                videoSTGBlocks: spec.videoGuidance.stgBlocks,
                audioSTGBlocks: spec.audioGuidance.stgBlocks,
                stepCaches: caches,
                cacheStepCount: spec.steps)

            // `denoiseMask` and `cleanLatent` are forwarded as a pair or not at all —
            // `Sampler.postProcess` short-circuits unless both are present, so a half-supplied
            // pair would silently disable conditioning rather than fail.
            //
            // The initial latent is seeded through the same pair first: the conditionings
            // are applied *before* the noiser runs, and the noiser then blends by the mask
            // — so a frozen token's starting value is the conditioning latent itself.
            // Feeding pure noise here while telling the transformer those tokens sit
            // at timestep 0 is a contradiction, and it looks like an i2v render that dissolves
            // its own reference frame.
            let drawn = noise.streamsForRenderOnly()
            let seeded = Sampler.Streams(
                video: Sampler.seedConditioned(noise: drawn.video,
                                               denoiseMask: conditioning?.denoiseMask.video,
                                               cleanLatent: conditioning?.cleanLatent.video),
                audio: Sampler.seedConditioned(noise: drawn.audio,
                                               denoiseMask: conditioning?.denoiseMask.audio,
                                               cleanLatent: conditioning?.cleanLatent.audio))

            let rendered = try Pipeline.render(
                sampler: sampler, steps: spec.steps,
                initial: seeded,
                denoiser: denoiser, geometry: spec.geometry,
                denoiseMask: conditioning?.denoiseMask,
                cleanLatent: conditioning?.cleanLatent,
                observer: { step, _ in try observer?(step, spec.steps) })
            let cacheNote = caches.isEmpty
                ? "off"
                : caches.values.map(\.summary).sorted().joined(separator: "; ")
            return (rendered.videoLatent, rendered.audioLatent, cacheNote)
        }()
        // The 42 GB transformer is out of scope. Hand its buffers back before either
        // VAE is opened — without this the allocator keeps the DiT working set and
        // the decode peak is DiT-residue plus decoder.
        MLX.Memory.clearCache()

        let videoRGB: MLXArray = try {
            let videoDecoder = try VideoVAEDecoder(checkpoint: checkpoints.videoVAE)
            let rgb = try decodeVideoOnRecordedPolicy(videoDecoder, videoLatent,
                                                     geometry: spec.geometry)
            MLX.eval(rgb)
            return rgb
        }()
        MLX.Memory.clearCache()

        let audioMel: MLXArray = try {
            let audioDecoder = try AudioVAEDecoder(checkpoint: checkpoints.audioVAE)
            let mel = try audioDecoder.decode(audioLatent)
            MLX.eval(mel)
            return mel
        }()
        MLX.Memory.clearCache()

        // The vocoder lives in the *same* file as the audio VAE decoder — 1227 of that
        // checkpoint's 1329 tensors are `vocoder.*` — so there is no fifth checkpoint to
        // configure and no way for the two halves to come from different exports.
        // Loaded in its own scope so the decoder's map is gone before this one opens
        // the file a second time and materialises every weight as fp32.
        let vocoderConfig = try AudioVocoder.configuration(
            from: try SafetensorsHeader.read(from: checkpoints.audioVAE))
        let audioWaveform: MLXArray = try {
            let vocoder = try AudioVocoder(checkpoint: checkpoints.audioVAE)
            let wave = try vocoder.waveform(fromMel: audioMel)
            MLX.eval(wave)
            return wave
        }()

        return Result(videoRGB: videoRGB, audioMel: audioMel,
                      audioWaveform: audioWaveform,
                      audioSampleRate: vocoderConfig.outputSamplingRate,
                      provenance: provenance(spec: spec, noise: noise, text: text,
                                             cache: cacheNote))
    }

    // MARK: - The decode policy

    /// The recorded `TileSizeConfig` decode policy, at whatever length this render is.
    ///
    /// At 13 latent frames this reproduces the two-chunk composition **bit for bit** —
    /// asserted by `TiledDecodeTests.generalPathReproducesTheRecordedTwoChunkComposition`:
    /// latent tiles `[0, 10)` and `[6, 13)`, a 25-frame output seam at `[48, 73)`, fp32
    /// trapezoidal masks accumulated into bf16. That bit-exactness is the safety argument
    /// for generalising at all; a generalisation that moved the arithmetic would have
    /// nothing behind it.
    ///
    /// Everything else — how many tiles a longer clip gets — is memory strategy, not a
    /// contract. ``TiledDecode`` documents which is which, and why its seam arithmetic
    /// deviates by one rounding from a streamed decode.
    ///
    /// A render decoded in one shot would still produce a plausible video, through a path
    /// nothing has measured. This goes through the measured one at every length.
    static func decodeVideoOnRecordedPolicy(_ decoder: VideoVAEDecoder, _ latent: MLXArray,
                                            geometry: DiTForward.Geometry) throws -> MLXArray {
        let latentFrames = geometry.latentFrames
        let latentHeight = geometry.height / geometry.latent.spatialScale
        let latentWidth = geometry.width / geometry.latent.spatialScale
        let expected = [1, geometry.latent.videoChannels, latentFrames,
                        latentHeight, latentWidth]
        guard latent.shape == expected else {
            throw Failure.unexpectedLatentShape("the sampled video latent",
                                                got: latent.shape, expected: expected)
        }

        // The grid comes off the checkpoint, not off `geometry`: the tiling has to be
        // built on the factors the decoder will actually apply.
        let factors = decoder.config.scaleFactors
        let layout = try TiledDecode.plan(
            latentFrames: latentFrames, latentHeight: latentHeight, latentWidth: latentWidth,
            policy: .conv(height: geometry.height, width: geometry.width),
            scale: TiledDecode.Scale(time: factors.time, height: factors.height,
                                     width: factors.width))
        let raw = try TiledDecode.decode(latent, layout: layout) { try decoder.decode($0) }
        return decoder.rgb(raw)
    }

    /// The tile layout a render *would* use, without decoding anything — for `--dry-run`
    /// style reporting and for saying, before 42 GB of transformer is loaded, that a
    /// frame is too wide for the spatial blend this port has not measured.
    public static func videoTileLayout(geometry: DiTForward.Geometry,
                                       scale: TiledDecode.Scale = .video) throws
        -> TiledDecode.Layout {
        try TiledDecode.plan(
            latentFrames: geometry.latentFrames,
            latentHeight: geometry.height / geometry.latent.spatialScale,
            latentWidth: geometry.width / geometry.latent.spatialScale,
            policy: .conv(height: geometry.height, width: geometry.width),
            scale: scale)
    }

    // MARK: - Provenance

    /// The sidecar. Written next to the mp4 so that a clip found later can be told what
    /// produced it and re-rendered exactly.
    static func provenance(spec: Spec, noise: RenderNoise, text: TextBranches,
                           cache: String = "off")
        -> [String: String] {
        [
            "kind": "a rendered clip — judged by eye, not by measurement",
            "seed_note": "The initial latent was drawn by MLX's own RNG. This model's "
                + "trajectory separates at production shape at every step count, so two "
                + "seeds are two different renders rather than variations on one.",
            "prompt": spec.prompt,
            "negative_prompt": spec.negativePrompt,
            "valid_token_counts": text.validTokenCounts.map(String.init).joined(separator: ","),
            "seed": String(spec.seed),
            "noise": noise.provenance,
            "steps": String(spec.steps),
            "geometry": "\(spec.geometry.width)x\(spec.geometry.height)x\(spec.geometry.frames)"
                + " @ \(spec.geometry.frameRate) fps",
            "attention": attentionProvenance(spec.attentionPath),
            "cache": cache,
            "cache_threshold": String(spec.cacheThreshold),
            "cache_max_skips": String(spec.cacheMaxSkips),
            "video_guidance": describe(spec.videoGuidance),
            "audio_guidance": describe(spec.audioGuidance),
            "video_decode": videoDecodeProvenance(spec: spec),
            "audio": "mel -> vocoder -> 48 kHz stereo waveform. JUDGE BY EAR: contract "
                + "11 explains why a vocoder waveform cannot be compared at all. The "
                + "comparable audio boundary is the mel, upstream of this.",
        ]
    }

    /// The realised tile layout, named in the sidecar so a clip found later can be told
    /// which decode composition produced it.
    static func videoDecodeProvenance(spec: Spec) -> String {
        let head = "reference TileSizeConfig, temporal 80/24 + aspect-coupled 768/64 "
            + "spatial (from_long_side)"
        guard let layout = try? videoTileLayout(geometry: spec.geometry) else { return head }
        let tiles = layout.frames.tiles.map { "[\($0.latent.lowerBound),\($0.latent.upperBound))" }
        let seams = (0 ..< max(0, layout.frames.tiles.count - 1))
            .map { String(layout.frames.overlap(after: $0)) }
        return head + "; latent tiles \(tiles.joined(separator: " ")); "
            + "output seams \(seams.isEmpty ? "none" : seams.joined(separator: "/")) frames; "
            + "\(layout.height.tiles.count)x\(layout.width.tiles.count) spatial; "
            + "fp32 masks into a bf16 buffer"
    }

    /// Which softmax this clip was made with — recorded because it is the one setting a
    /// render deliberately does *not* share with the rest of the package, and a clip found
    /// later should not have to be guessed about.
    static func attentionProvenance(_ path: DiTAttention.AttentionPath) -> String {
        switch path {
        case .fused:
            return "fused (MLX steel SDPA), the render default. Chosen because the "
                + "explicit path materialises a [1, 32, T, T] fp32 score matrix, which is "
                + "quadratic in the token count and does not fit at long or large shapes. "
                + "Measured disagreement against explicit: 1.11e-04 on attn1, exactly "
                + "zero on audio_attn1."
        case .explicit:
            return "explicit fp32 softmax. Peak memory is quadratic in the token "
                + "count; this is not the render default and will be killed at long or "
                + "large shapes."
        }
    }

    static func describe(_ g: GuidanceParams) -> String {
        "cfg=\(g.cfgScale) stg=\(g.stgScale) stg_blocks=\(g.stgBlocks) "
            + "stg_window=[\(g.stgStartPercent),\(g.stgEndPercent)] "
            + "rescale=\(g.rescaleScale) modality=\(g.modalityScale) "
            + "modality_window=[\(g.modalityStartPercent),\(g.modalityEndPercent)] "
            + "skip_step=\(g.skipStep)"
    }

    static func require(_ x: MLXArray?, _ name: String) throws -> MLXArray {
        guard let x else { throw Pipeline.Failure.missingTap(name) }
        return x
    }
}

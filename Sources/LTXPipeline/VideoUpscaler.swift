// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import LTXRecipes
import MLX

/// x2 spatial upscale of a video that already exists.
///
/// ## Why this is a separate entry point and not a flag on `render`
///
/// `DistilledRenderer` samples a draft, upsamples its latent, and refines. The upsample in
/// the middle is a learned latent→latent operator that does not care where its input latent
/// came from — `LatentSpatialUpsampler.upsample` takes `[1, 128, F, H, W]` and returns
/// `[1, 128, F, 2H, 2W]`, and a VAE encode of a finished clip produces exactly the same
/// kind of object as a completed stage 1: a *clean* latent at sigma 0. So the whole of
/// "upscale an existing file" is: replace stage 1 with an encode.
///
/// It is a separate command rather than `render --two-stage --video` because the two share
/// no request: there is no prompt, no seed, no schedule and no output geometry to choose —
/// the input file supplies all four, and the output size is `2x` by construction. Threading
/// a video through `Render`'s twenty guidance flags, every one of which is already refused
/// under `--two-stage`, would be a wider seam for less.
///
/// ## The two modes, and what separates them
///
/// ``Mode/plain`` is encode → upsample → decode. Three checkpoints totalling 3.4 GB, **no
/// transformer and no text encoder**, and the upsampler is the only learned thing in the
/// path. It is the honest floor: whatever the x2 operator does on its own.
///
/// ``Mode/refined`` adds the recipe's second stage — the *documented* one, not an invented
/// schedule: `[0.909375, 0.725, 0.421875, 0.0]`, so 3 steps, deterministic Euler, no
/// guider, on the distilled checkpoint. `FlowSchedule.distilledStage2` and
/// ``DistilledRenderer`` already carry it, and this reuses both rather than restating them.
/// That costs the 42 GB DiT and a 26 GB text-encoder pass for the prompt, for 3 forwards.
///
/// The `sigmas[0] = 0.909375` matters for reading the output: stage 2 does not merely clean
/// the upsample, it **re-noises to 91% and re-denoises**. On a generated draft that is the
/// mechanism by which the refine continues the draft. On a *finished* clip it is also
/// licence to move away from it, which is the thing the A/B has to look for.
///
/// ## What it will refuse
///
/// The VAE's spatial grid is 32 and the upsampler doubles, so an input must be a multiple
/// of **32** on both axes and the output is then automatically a multiple of 64 — which is
/// what `Grid` and `DistilledRenderer.Spec.validate` demand of a two-stage size. 864x480 in
/// gives 1728x960 out and both hold. 432x240 in does not: 432 is 13.5 latent columns.
///
/// Frames must sit on the VAE's `8k+1` lattice, and an arbitrary clip's frame count almost
/// never does, so nearly every shot is off-grid. The tail is **padded up** by cloning the
/// last frame and the padding is trimmed off after decoding, so a 192-frame shot goes in as
/// 193 and comes back as 192. Nothing is dropped and nothing extra is written.
///
/// The input's audio is carried over unchanged and trimmed to the picture. It is never
/// re-synthesised: every mode here changes the picture and none of them has anything to say
/// about the sound.
public enum VideoUpscaler {

    public enum Mode: String, Sendable, CaseIterable {
        /// Encode, upsample, decode. No transformer.
        case plain
        /// The above plus the recipe's 3-step stage-2 refine on the distilled checkpoint.
        case refined
        /// **Encode and decode, no upsample.** Output is the same size as the input, and in
        /// a perfect world the same picture.
        ///
        /// This is a control, not a feature. Every other mode's output has been through the
        /// VAE, so any artifact it shows is either the VAE's or the upsampler's and the two
        /// are indistinguishable from the result alone. This isolates the first: whatever
        /// `roundtrip` costs is the floor beneath `plain`, and only what `plain` adds on top
        /// belongs to the x2 operator.
        case roundtrip
        /// **No latent upsample. Encode, refine at the input's own size, decode.**
        ///
        /// The caller supplies a clip that is *already* at the output size, resampled in
        /// pixel space — `ffmpeg ... scale=W:H:flags=lanczos`. This is the shape of a
        /// delivery upscale — a **pixel**-spatial resample followed by a v2v denoise —
        /// rather than of the distilled recipe's stage 2, and it exists because the two
        /// paths fail differently.
        ///
        /// The latent operator damages the picture and then needs a *strong* refine to
        /// repair it — measured here as visible crinkle surviving at denoise 0.30, and
        /// identity drifting by 0.60. A Lanczos resample damages nothing; it is merely
        /// soft. So this path can run the refine weak, where it adds detail without
        /// licence to reinvent, which is the setting the other path cannot reach.
        case refineOnly
        /// **The real upscaler.** The LTX-2.5 22B IC-LoRA Pixel Spatial Upscaler.
        ///
        /// The other three modes all drive `latent-spatial-upscaler-x2`, which is the
        /// operator that sits *between* the two stages of a distilled render and hands its
        /// output to a denoiser, never to a decoder. This drives
        /// `ic-lora-pixel-spatial-upscaler-x2`, which is the one built to upscale a clip.
        ///
        /// It is not an upsampler at all. It is a rank-32 adapter on the 22B transformer
        /// plus an *in-context reference*: the input clip is encoded and its tokens ride in
        /// the video sequence beside the noisy ones, clean and at timestep 0, occupying the
        /// target's own coordinate range. The model then renders the shot at 2x from noise,
        /// carrying composition, motion and identity off the reference and inventing the
        /// high-frequency detail that was never in it.
        ///
        /// Which is why there is no denoise dial here. Fidelity to the reference is traded
        /// through **steps and guidance**, per the model card, not through a starting sigma.
        ///
        /// In its documented use the reference conditions **stage 1 only** — stage 2 there
        /// is the ordinary latent upsample and 3-step refine, and it adds a *second* 2x.
        /// This runs stage 1 alone at the target size, because that is the single place the
        /// adapter's trained 2x reference-to-target relationship actually holds.
        case icLoRA
    }

    public enum Failure: Error, CustomStringConvertible {
        case offGrid(width: Int, height: Int)
        case tooFewFrames(got: Int)
        case missingCheckpoint(String, path: String)
        case badShape(String, got: [Int])

        public var description: String {
            switch self {
            case let .offGrid(width, height):
                return "the input is \(width)x\(height); both axes must be a multiple of 32 "
                    + "(the video VAE's spatial compression), so that the x2 upsample lands "
                    + "on the multiple-of-64 grid a two-stage size requires. "
                    + "\(width - width % 32)x\(height - height % 32) is the nearest at or below"
            case let .tooFewFrames(got):
                return "\(got) frame(s) decoded; padding up onto the 8k+1 lattice needs at "
                    + "least a frame to clone"
            case let .missingCheckpoint(what, path):
                return "the \(what) checkpoint is not at \(path)"
            case let .badShape(what, got):
                return "\(what) came back as \(got), which is not the rank this path expects"
            }
        }
    }

    /// What a run did, in the units a caller is going to want to compare.
    public struct Report: Sendable {
        public var inputWidth: Int
        public var inputHeight: Int
        public var outputWidth: Int
        public var outputHeight: Int
        /// Frames actually processed, after the lattice snap.
        public var frames: Int
        /// Frames the source had. `framesDropped` is the difference.
        public var sourceFrames: Int
        public var framesDropped: Int { sourceFrames - frames }
        public var mode: Mode
        public var decodeSeconds: Double
        public var encodeSeconds: Double
        public var upsampleSeconds: Double
        public var refineSeconds: Double
        public var vaeDecodeSeconds: Double
        public var writeSeconds: Double
        public var totalSeconds: Double

        public var summary: String {
            var lines = [
                "\(inputWidth)x\(inputHeight) -> \(outputWidth)x\(outputHeight), "
                    + "\(frames) frames, mode \(mode.rawValue)",
                String(format: "  read      %6.1fs", decodeSeconds),
                String(format: "  encode    %6.1fs", encodeSeconds),
                String(format: "  upsample  %6.1fs", upsampleSeconds),
            ]
            if mode == .refined {
                lines.append(String(format: "  refine    %6.1fs", refineSeconds))
            }
            lines.append(String(format: "  decode    %6.1fs", vaeDecodeSeconds))
            lines.append(String(format: "  write     %6.1fs", writeSeconds))
            lines.append(String(format: "  TOTAL     %6.1fs", totalSeconds))
            if framesDropped != 0 {
                lines.append("  NOTE: frame count \(sourceFrames) -> \(frames)")
            }
            return lines.joined(separator: "\n")
        }
    }

    /// Where the checkpoints are. `dit` and `textEncoder` are read only by ``Mode/refined``.
    public struct Checkpoints {
        public var videoVAE: URL
        public var upsampler: URL
        public var dit: URL?
        public var textEncoder: URL?
        public var audioVAE: URL?
        /// The IC-LoRA adapter, read only by ``Mode/icLoRA``. NOT the latent spatial
        /// upscaler in ``upsampler`` — different file, different mechanism, and the two
        /// are easy to confuse because both are ~327 MB and both say "spatial upscaler".
        public var icLoRA: URL?

        public init(videoVAE: URL, upsampler: URL, dit: URL? = nil,
                    textEncoder: URL? = nil, audioVAE: URL? = nil, icLoRA: URL? = nil) {
            self.videoVAE = videoVAE
            self.upsampler = upsampler
            self.dit = dit
            self.textEncoder = textEncoder
            self.audioVAE = audioVAE
            self.icLoRA = icLoRA
        }

        func requireExists() throws {
            try require(videoVAE, "video VAE")
            try require(upsampler, "latent spatial upsampler")
        }

        private func require(_ url: URL, _ what: String) throws {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Failure.missingCheckpoint(what, path: url.path)
            }
        }
    }

    // MARK: - The run

    /// Read `input`, upscale x2, write `output`.
    ///
    /// - Parameters:
    ///   - fps: written into the output container. It does **not** change the picture; it is
    ///     the frame rate the file declares. Read it off the source rather than defaulting.
    ///   - maxFrames: cap before the lattice snap, for a cheap probe of a long clip.
    ///   - note: progress, one line per phase.
    /// The recipe's stage-2 schedule, rescaled to start at `denoise`.
    ///
    /// **This is a generalisation of the recipe, not the recipe.** There is exactly one
    /// recorded stage-2 schedule, `[0.909375, 0.725, 0.421875, 0.0]`, and at
    /// `denoise == 0.909375` this returns it unchanged and bit-identically. Any lower value
    /// multiplies the whole curve by `denoise / 0.909375`, preserving the recorded *shape*
    /// — the relative spacing of the three steps — while starting nearer the input.
    ///
    /// It exists because the recorded σ₀ is wrong for this input by construction. 0.909375
    /// is chosen for a *draft*: an 8-step ancestral sketch at half size, which is supposed
    /// to be overwritten. A finished clip is not a sketch, and re-noising it to 91% and
    /// re-denoising against a photoreal-biased base model does not upscale it, it re-renders
    /// it — measured here at PSNR 20.2 dB with a different actor in the frame.
    ///
    /// Nothing anchors a rescaled curve. Treat the number as a dial that was turned by eye.
    public static func stage2Sigmas(denoise: Float) -> [Float] {
        let recorded = FlowSchedule.distilledStage2
        guard let first = recorded.first, first > 0 else { return recorded }
        if denoise >= first { return recorded }
        let k = denoise / first
        return recorded.map { $0 * k }
    }

    /// Which curve a reference-guided stage samples on.
    ///
    /// Extracted from ``referenceGuided(references:geometry:checkpoints:prompt:seed:steps:referenceStrength:initialVideo:explicitSigmas:guided:overlay:note:)``
    /// so the precedence is checkable without a 22 B checkpoint in the room, because getting
    /// it wrong is silent.
    ///
    /// **An explicit curve wins, guided or not.** The guided branch used to take priority,
    /// and that combination — a guided stage 2 — had never been run. Stage 2 supplies its own
    /// curve precisely because it starts partway down the trajectory: it continues stage 1
    /// rather than beginning again. A schedule derived from a step count always starts at
    /// 1.0, and the upsampled latent is lerped to `sigmas[0]` on the way in, so a 1.0 head
    /// lerps it *entirely* to noise. The refine becomes a fresh render at full resolution,
    /// at four forwards a step, with nothing left of the draft it was handed. It completes,
    /// it looks like a render, and it is wrong.
    static func schedule(explicit: [Float]?, guided: Guided?,
                         requestedSteps: Int) throws -> [Float] {
        if let explicit { return explicit }
        // The dev transformer's own continuous schedule, not a thinning of a recorded
        // table. `FlowSchedule` is what `prod` samples on and takes any step count.
        if let guided { return try FlowSchedule().sigmas(steps: guided.steps) }
        return Self.stage1Sigmas(steps: requestedSteps)
    }

    /// A stage-2 refine curve for the **dev** transformer, on its own continuous schedule.
    ///
    /// ``stage2Sigmas(denoise:)`` rescales the *recorded distilled* table, which is the
    /// right curve only on the weights it was distilled onto. `prod` runs dev, and dev
    /// samples on `FlowSchedule` — so this builds the continuous curve at the refine's own
    /// step count and scales the whole of it to start at `denoise`, which is the same
    /// operation ``stage2Sigmas(denoise:)`` performs on the recorded one: the shape is the
    /// schedule's, the starting point is the re-noise level.
    ///
    /// `denoise` defaults to 0.909375, the distilled stage 2's own head, because that is
    /// the level the whole two-stage arrangement is built around — stage 1 hands over a
    /// draft and stage 2 re-noises to 91% and re-derives at full resolution. Nothing
    /// anchors the pairing of that number with *this* curve; it is the recorded
    /// arrangement's re-noise level applied to the schedule dev actually samples on.
    ///
    /// Not a thinning and not a tail. Taking the tail of a 30-step continuous curve below
    /// 0.909 would be roughly 27 steps, which is not a refine; taking its last four sigmas
    /// would start near 0.09 and barely move the picture.
    public static func productionStage2Sigmas(
        steps: Int,
        denoise: Float = FlowSchedule.distilledStage2[0]) throws -> [Float] {
        let curve = try FlowSchedule().sigmas(steps: steps)
        guard let first = curve.first, first > 0 else { return curve }
        let k = denoise / first
        return curve.map { $0 * k }
    }

    /// How many steps a stage-2 refine takes, which depends on *which curve* it runs.
    ///
    /// Not a preference. The distilled stage-2 table is four sigmas long, so three steps is
    /// that schedule itself rather than a sample of it. ``productionStage2Sigmas(steps:_:)``
    /// has no such length, and `FlowSchedule`'s stretch pins its last live sigma to
    /// `terminal` — so three steps of it is `0.909 -> 0.668 -> 0.091 -> 0`, one jump
    /// covering four fifths of the range it is descending.
    ///
    /// Measured, on a fixed draft and seed: at three steps a black coat arrives as a coarse
    /// stipple, and at eight as fabric. It is the same defect an over-thinned stage 1 has,
    /// read from the other end — a sigma missing from the middle of the working range. The
    /// eight costs 32 forwards against 12, beside stage 1's 120.
    public static func refineSteps(guided: Bool) -> Int { guided ? 8 : 3 }

    /// The recorded stage-1 schedule, thinned to `steps`.
    ///
    /// That schedule is `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725,
    /// 0.421875, 0.0]` — and the shape of that list is the whole reason a step count is a
    /// meaningful dial rather than a quality slider. **The first five sigmas span 1.0 to
    /// 0.975.** Four of the eight steps move the noise level by two and a half percent
    /// between them; the last four do all the work.
    ///
    /// So thinning takes from the head and never from the tail. `0.909375, 0.725, 0.421875,
    /// 0.0` survives every step count down to 4, which is why 4 steps is not "half the
    /// render" — it is the same trajectory with the near-stationary preamble removed.
    ///
    /// At `steps >= 8` this returns the recorded schedule unchanged and bit-identically.
    /// Below it, this is a thinning of a recorded curve and nothing anchors it.
    public static func stage1Sigmas(steps: Int) -> [Float] {
        let recorded = FlowSchedule.distilledStage1
        let required = steps + 1
        guard required < recorded.count else { return recorded }

        // Where the schedule stops loitering near 1.0 and starts descending.
        let tailStart = 5
        let tail = Array(recorded[tailStart...])
        guard required > tail.count else {
            // Fewer steps than the working tail has: sample the tail itself, keeping both
            // ends, rather than eating into it from an arbitrary side.
            let head = recorded[0]
            let picks = (1..<required).map { i -> Float in
                let t = Double(i - 1) / Double(max(required - 2, 1))
                return tail[Int((t * Double(tail.count - 1)).rounded())]
            }
            return [head] + picks
        }
        let headCount = required - tail.count
        let head = (0..<headCount).map { i -> Float in
            let t = Double(i) / Double(max(headCount - 1, 1))
            return recorded[Int((t * Double(tailStart - 1)).rounded())]
        }
        return head + tail
    }

    public static func upscale(input: URL,
                               output: URL,
                               checkpoints: Checkpoints,
                               mode: Mode = .plain,
                               prompt: String = "",
                               denoise: Float = FlowSchedule.distilledStage2[0],
                               seed: UInt64 = 0,
                               steps: Int = FlowSchedule.distilledStage1.count - 1,
                               keepAudio: Bool = true,
                               fps: Double = 24,
                               maxFrames: Int? = nil,
                               note: ((String) -> Void)? = nil) async throws -> Report {
        try checkpoints.requireExists()
        let wall = Date()

        // ---- Read ------------------------------------------------------------------
        var t = Date()
        note?("reading \(input.lastPathComponent)")
        let decoded = try await MediaInput.videoFrames(at: input, maxFrames: maxFrames)
        guard decoded.ndim == 4, decoded.shape[3] == 3 else {
            throw Failure.badShape("the decoded clip", got: decoded.shape)
        }
        let sourceFrames = decoded.shape[0]
        let h = decoded.shape[1], w = decoded.shape[2]
        guard w % 32 == 0, h % 32 == 0 else { throw Failure.offGrid(width: w, height: h) }

        let latentGeometry = LatentGeometry()
        let lattice = latentGeometry.lattice

        // **Padded up, then trimmed back — never snapped down.**
        //
        // The VAE only accepts `8k+1` frames and an incoming clip's frame count almost
        // never lands on that lattice: 192 wants 193, 243 wants 249. Snapping *down* is
        // the cheap answer and it silently shortens the shot by up to seven frames —
        // which, muxed against its own untouched audio and cut beside its neighbours,
        // reads as the next shot coming in early. That is the same class of desync this
        // project has already paid for once.
        //
        // So the tail is extended by cloning the last frame up to the next lattice point,
        // and the extra frames are dropped from the *output* after decoding. The clip keeps
        // every frame it arrived with, and the padding never reaches the file.
        let frames = lattice.contains(sourceFrames) ? sourceFrames : lattice.snapUp(sourceFrames)
        guard sourceFrames >= 2 else { throw Failure.tooFewFrames(got: sourceFrames) }
        let keptFrames = sourceFrames
        let rgb: MLXArray
        if frames == sourceFrames {
            rgb = decoded
        } else {
            let pad = frames - sourceFrames
            let last = decoded[(sourceFrames - 1) ..< sourceFrames]
            rgb = MLX.concatenated([decoded] + Array(repeating: last, count: pad), axis: 0)
            note?("padded \(sourceFrames) -> \(frames) frames onto the 8k+1 lattice by "
                + "cloning the last frame; the \(pad) extra frame(s) are trimmed after decode")
        }
        MLX.eval(rgb)
        let decodeSeconds = -t.timeIntervalSinceNow

        // ---- Encode ----------------------------------------------------------------
        //
        // bf16 on the way in, deliberately. `VideoVAEEncoder.encode` documents the trap:
        // an fp32 clip promotes all 42 convolutions and computes more precisely than the
        // encoder is meant to, which is the expensive way to be wrong.
        t = Date()
        note?("encoding to latent")
        let latent: MLXArray = try {
            let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
            let pixels = encoder.pixels(fromRGB: rgb).asType(.bfloat16)
            let l = try encoder.encode(pixels)
            MLX.eval(l)
            return l
        }()
        MLX.Memory.clearCache()
        let encodeSeconds = -t.timeIntervalSinceNow
        note?("latent \(latent.shape)")

        // ---- Upsample --------------------------------------------------------------
        //
        // The statistics come from the video VAE's own `per_channel_statistics.*`, which is
        // where `DistilledRenderer.prefetchUpsampler` reads them: the upsampler denormalises
        // on the way in and renormalises on the way out, and that round trip is part of the
        // operator rather than something a caller applies around it.
        // The output scale, and for `.icLoRA` the reference scale with it.
        //
        // **Read from the adapter, not assumed.** Both numbers used to be the literal `2`,
        // written when the x2 pixel spatial upscaler was the only adapter this port could
        // run. Across the sixteen adapters on disk that constant is right for three: the
        // upscalers are 2 and 4, the control adapters 2, and the ten editing adapters —
        // ingredients, in-outpainting, colorization, deblur and the rest — are **1**, because
        // they transform a clip in place rather than enlarging it.
        //
        // Getting it wrong is silent both ways. Too large an output invents a size the
        // adapter was not trained to produce; too large a reference factor lands the
        // reference on a fraction of the target's pixel range, which
        // `DiTForward.Geometry.Reference` describes as showing the model a crop it never
        // saw — well-formed tokens, plausible output, wrong render.
        var icLoRAReference: ICLoRAReference?
        if mode == .icLoRA {
            guard let loraURL = checkpoints.icLoRA else {
                throw Failure.missingCheckpoint("IC-LoRA (required by --mode ic-lora)",
                                                path: "(not supplied)")
            }
            let reference = try ICLoRAReference.read(contentsOf: loraURL)
            icLoRAReference = reference
            note?("adapter declares reference_downscale_factor \(reference.downscaleFactor)"
                + ", temporal \(reference.temporalScaleFactor)"
                + (reference.downscaleFactor == 1
                    ? " — an in-place transform, so the output is the input's size"
                    : " — a x\(reference.downscaleFactor) enlargement"))
        }

        let scale: Int
        switch mode {
        case .roundtrip, .refineOnly: scale = 1
        // The latent upsampler is x2 by construction; these modes do not consult an adapter.
        case .plain, .refined: scale = 2
        case .icLoRA: scale = icLoRAReference?.downscaleFactor ?? 1
        }
        let outWidth = w * scale, outHeight = h * scale
        var geometry = DiTForward.Geometry(frames: frames, height: outHeight, width: outWidth,
                                           frameRate: fps, latent: latentGeometry)
        if let reference = icLoRAReference {
            geometry.reference = DiTForward.Geometry.Reference(
                latentFrames: latent.dim(2),
                latentHeight: latent.dim(3),
                latentWidth: latent.dim(4),
                downscaleFactor: reference.downscaleFactor,
                temporalScaleFactor: reference.temporalScaleFactor)
        }

        t = Date()
        let upscaled: MLXArray
        switch mode {
        case .roundtrip, .refineOnly:
            note?(mode == .refineOnly
                ? "no latent upsample (refine-only; the input is already at output size)"
                : "no upsample (roundtrip control)")
            upscaled = latent
        case .icLoRA:
            // The generation IS the upscale here, so it happens in this slot and what comes
            // out is already the target latent. Nothing downstream can tell the difference.
            note?("IC-LoRA reference-guided render at \(outWidth)x\(outHeight)")
            // `upscale` carries the source clip's own waveform, so the generated audio is
            // dropped here deliberately — see ``referenceGuided``.
            upscaled = try referenceGuided(references: [latent], geometry: geometry,
                                           checkpoints: checkpoints, prompt: prompt,
                                           seed: seed, steps: steps, note: note).video
        case .plain, .refined:
            note?("x2 latent upsample")
            upscaled = try {
                let upsampler = try LatentSpatialUpsampler(checkpoint: checkpoints.upsampler,
                                                           videoVAE: checkpoints.videoVAE)
                let u = try upsampler.upsample(latent)
                MLX.eval(u)
                return u
            }()
        }
        MLX.Memory.clearCache()
        let upsampleSeconds = -t.timeIntervalSinceNow
        note?("upscaled \(upscaled.shape)")

        // ---- Refine, optionally ------------------------------------------------------
        var refined = upscaled
        var refineSeconds: Double = 0
        if mode == .refined || mode == .refineOnly {
            t = Date()
            let sigmas = Self.stage2Sigmas(denoise: denoise)
            note?("stage-2 refine at \(outWidth)x\(outHeight): \(sigmas.count - 1) Euler "
                + "steps from sigma \(sigmas[0])")
            refined = try refine(upscaled, geometry: geometry, checkpoints: checkpoints,
                                 prompt: prompt, sigmas: sigmas, note: note)
            MLX.Memory.clearCache()
            refineSeconds = -t.timeIntervalSinceNow
        }

        // ---- Decode ------------------------------------------------------------------
        t = Date()
        note?("decoding")
        let out: MLXArray = try {
            let decoder = try VideoVAEDecoder(checkpoint: checkpoints.videoVAE)
            let pixels = try Renderer.decodeVideoOnRecordedPolicy(decoder, refined,
                                                                  geometry: geometry)
            MLX.eval(pixels)
            return pixels
        }()
        MLX.Memory.clearCache()
        // The cloned tail never reaches the file.
        let trimmed = keptFrames == frames ? out : out[0 ..< keptFrames]
        let vaeDecodeSeconds = -t.timeIntervalSinceNow

        // ---- Write -------------------------------------------------------------------
        //
        // The source's audio rides along unchanged — see ``sourceAudio(_:frames:fps:note:)``
        // for why it is carried rather than re-synthesised, and why it is trimmed.
        t = Date()
        note?("writing \(output.lastPathComponent)")
        let videoSpec = RenderOutput.VideoSpec(fps: fps)
        var carried = false
        if keepAudio {
            do {
                let (samples, rate) = try await sourceAudio(input, frames: keptFrames, fps: fps,
                                                            note: note)
                try RenderOutput.write(rgb: trimmed, audio: samples, to: output, spec: videoSpec,
                                       audioSpec: RenderOutput.AudioSpec(sampleRate: rate))
                carried = true
            } catch {
                // A clip with no audio track is the ordinary case for a picture-only
                // render, not a failure. Anything else is worth saying out loud rather
                // than silently shipping a mute file.
                note?("no audio carried over: \(error)")
            }
        }
        if !carried {
            try RenderOutput.writeVideo(rgb: trimmed, to: output, spec: videoSpec)
        }
        let writeSeconds = -t.timeIntervalSinceNow

        return Report(inputWidth: w, inputHeight: h,
                      outputWidth: outWidth, outputHeight: outHeight,
                      frames: keptFrames, sourceFrames: sourceFrames, mode: mode,
                      decodeSeconds: decodeSeconds, encodeSeconds: encodeSeconds,
                      upsampleSeconds: upsampleSeconds, refineSeconds: refineSeconds,
                      vaeDecodeSeconds: vaeDecodeSeconds, writeSeconds: writeSeconds,
                      totalSeconds: -wall.timeIntervalSinceNow)
    }

    /// The input's own audio, trimmed to the output's picture duration.
    ///
    /// **Carried, not regenerated.** Every mode here either resamples or re-renders the
    /// *picture*; none of them has anything to say about the sound, and the source's
    /// waveform is already the best version of it. Re-synthesising audio through the model
    /// would replace a finished dialogue take with a fresh guess.
    ///
    /// Read at the source's own rate and channel count, so no resample happens in the
    /// common case — `MediaInput.audio(at:)` is explicit that a rate conversion invents
    /// samples and is the caller's choice, and here there is no reason to make it.
    ///
    /// The trim is the part that is easy to skip and wrong to skip. Frames are snapped
    /// **down** onto the 8k+1 lattice, so a 243-frame shot renders as 241 and its picture
    /// is 83 ms shorter than the audio it came with. Muxing the full waveform against the
    /// shorter picture leaves the tail hanging past the last frame, which in a cut sequence
    /// reads as the next shot starting late.
    private static func sourceAudio(_ input: URL, frames: Int, fps: Double,
                                    note: ((String) -> Void)?) async throws -> (MLXArray, Int) {
        var (samples, rate) = try await MediaInput.audio(at: input)
        var channels = samples.dim(1)
        if channels != RenderOutput.audioChannels {
            note?("source audio is \(channels)ch; converting to stereo at \(rate) Hz")
            samples = try await MediaInput.audio(at: input, sampleRate: rate,
                                                 channels: RenderOutput.audioChannels)
            channels = samples.dim(1)
        }

        let wanted = Int((Double(frames) / fps * Double(rate)).rounded())
        let have = samples.dim(2)
        if have > wanted {
            note?("audio \(have) -> \(wanted) samples to match the picture")
            samples = samples[0..., 0..., 0 ..< wanted]
        } else if have < wanted {
            note?("audio padded \(have) -> \(wanted) samples to match the picture")
            samples = MLX.concatenated(
                [samples, MLXArray.zeros([1, channels, wanted - have], dtype: samples.dtype)],
                axis: 2)
        }
        return (samples, rate)
    }

    // MARK: - IC-LoRA

    /// Render the target at 2x with the reference clip carried in context.
    ///
    /// The IC-LoRA stage-1 render, with the reference clip's latent carried in context.
    /// Every piece of this already existed here; what did not was the sequence being longer
    /// than the picture.
    ///
    /// ## The three tensors that make a token a reference
    ///
    /// Appended after the generated tokens, in this order:
    ///
    /// * **latent** — `zeros_like(tokens)`. A placeholder. Never read, because the mask
    ///   overwrites it before the first forward.
    /// * **cleanLatent** — the reference's own patchified tokens.
    /// * **denoiseMask** — `1 - strength`, so `0` at full strength.
    ///
    /// The mask is doing two jobs at once and only one of them is obvious.
    /// ``Sampler/postProcess(denoised:denoiseMask:cleanLatent:)`` uses it to *hold the
    /// value*, and ``Sampler/timesteps(sigma:mask:tokens:)`` uses it to hand those tokens
    /// **timestep 0** — which is the only signal in the entire forward pass that says "this
    /// is given, condition on it". Hold the value without the timestep and the render
    /// completes, the reference tokens still contain the reference because they are
    /// rewritten after every step, and the picture is generated as though unconditioned.
    ///
    /// ## What the adapter has to reach
    ///
    /// 960 tensors: `attn1`/`attn2` `to_{q,k,v,out.0}` and `ff.net.{0.proj,2}`, on all 48
    /// blocks, video stream only — the file contains no audio tensors at all. Eight of
    /// those ten modules per block live in `DiTAttention`, which is why
    /// ``LTXModules/DiTForward/attach(lora:)`` reaches the block rather than only the
    /// shared modules.
    ///
    /// ## Schedule
    ///
    /// The distilled stage-1 sigmas, ancestral, no guider — the same arrangement the
    /// distilled renderer uses, so this borrows ``DistilledRenderer/neutralGuidance`` and
    /// its ancestral constants rather than restating them.
    /// - Parameters:
    ///   - references: reference latents, **in the order their tokens are appended**, which
    ///     must be the order `geometry.references` describes. One for every IC-LoRA that
    ///     shipped before MSR; up to five for MSR, each already carrying its slot embedding.
    ///   - referenceStrength: how clean the reference is held. 1.0 keeps it perfectly
    ///     clean, which is what every shipped IC-LoRA expects; lower
    ///     values let the sampler denoise it. The reference fills its denoise mask with
    ///     `1 - strength`, so this is that subtraction and not a rescaling of the tokens.
    ///   - overlay: a pre-built adapter overlay. Supplied by callers whose adapter file
    ///     holds more than LoRA factors — MSR's five slot tensors have to be lifted out
    ///     before ``LTXModules/LoRAOverlay`` sees the map, so that caller builds the overlay
    ///     itself and hands it over. Nil loads `checkpoints.icLoRA` whole, as before.
    /// The production arrangement, for a reference-conditioned render that wants the dev
    /// transformer's continuous schedule instead of the distilled tables.
    ///
    /// The distilled default is one forward per step on a nine-sigma table. This is four
    /// forwards per step on a schedule of any length, which is what `prod` is — and the
    /// reference tokens ride through it unchanged, held clean in **both** guidance
    /// branches. That last part is the whole reason this is a type rather than a step
    /// count: the conditional and unconditional passes must see the same reference, or
    /// CFG subtracts a render that never saw the character from one that did, and the
    /// conditioning quietly weakens instead of failing.
    public struct Guided: Sendable {
        public let negativePrompt: String
        public let video: GuidanceParams
        public let audio: GuidanceParams
        public let steps: Int

        /// The production arrangement, matching `RecipeRegistry.production`.
        ///
        /// Defined once, here, rather than in each command that offers it: these five
        /// numbers are what `prod` *means*, and two commands carrying their own copies is
        /// how one of them ends up a version behind.
        public static func production(negativePrompt: String = "") -> Guided {
            Guided(negativePrompt: negativePrompt,
                   video: GuidanceParams(cfgScale: 3.0, stgScale: 1.0, stgBlocks: [28],
                                         rescaleScale: 0.7, modalityScale: 3.0),
                   audio: GuidanceParams(cfgScale: 7.0, stgScale: 1.0, stgBlocks: [28],
                                         rescaleScale: 0.7, modalityScale: 3.0),
                   steps: 30)
        }

        public init(negativePrompt: String = "", video: GuidanceParams,
                    audio: GuidanceParams, steps: Int) {
            self.negativePrompt = negativePrompt
            self.video = video
            self.audio = audio
            self.steps = steps
        }

        /// The same guidance at a different step count, for a stage 2.
        ///
        /// The five guidance numbers carry across unchanged — they are what `prod` means and
        /// stage 2 is still `prod`. Only the count differs, and it differs because stage 2 is
        /// a refine: it re-derives the last stretch of the trajectory at full resolution
        /// rather than walking the whole of it.
        public func refining(steps: Int) -> Guided {
            Guided(negativePrompt: negativePrompt, video: video, audio: audio, steps: steps)
        }
    }

    static func referenceGuided(references: [MLXArray],
                                        geometry: DiTForward.Geometry,
                                        checkpoints: Checkpoints,
                                        prompt: String,
                                        seed: UInt64,
                                        steps requestedSteps: Int,
                                        referenceStrength: Float = 1.0,
                                        initialVideo: MLXArray? = nil,
                                        explicitSigmas: [Float]? = nil,
                                        guided: Guided? = nil,
                                        overlay preloaded: LoRAOverlay? = nil,
                                        sampler samplerKind: SamplerKind? = nil,
                                        evalCadence: Int = 1,
                                        cacheThreshold: Double = 0,
                                        cacheMaxSkips: Int = StepCachePolicy.measuredCap,
                                        note: ((String) -> Void)?) throws
        -> (video: MLXArray, audio: MLXArray) {
        guard let ditURL = checkpoints.dit, let textURL = checkpoints.textEncoder,
              let loraURL = checkpoints.icLoRA else {
            throw Failure.missingCheckpoint("DiT, text encoder and IC-LoRA (all three are "
                + "required by --mode ic-lora)", path: "(not supplied)")
        }
        for (what, url) in [("DiT", ditURL), ("text encoder", textURL),
                            ("IC-LoRA", loraURL)] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Failure.missingCheckpoint(what, path: url.path)
            }
        }

        // Patchified one at a time and concatenated in order, rather than concatenating the
        // latents first. `patchify` flattens `(F, H, W)` per clip, so a single patchify of
        // stacked references would only agree with this when every reference has the same
        // grid — and would interleave their tokens rather than block them when it does not.
        // The position grid is built per reference and appended in this same order, so a
        // disagreement here shows up as references wearing each other's coordinates.
        guard references.count == geometry.references.count else {
            throw Failure.badShape("\(references.count) reference latent(s) against a geometry "
                + "describing \(geometry.references.count)", got: [references.count])
        }
        guard !references.isEmpty else {
            // Not a degenerate case worth supporting: with no reference this path is an
            // ordinary render with an adapter attached, and `Renderer` is that path.
            throw Failure.badShape("no reference latents; this entry point exists to carry "
                + "at least one", got: [])
        }
        let referenceTokens =
            MLX.concatenated(try references.map { try Sampler.patchifyVideo($0) }, axis: 1)
        let refCount = referenceTokens.dim(1)
        guard refCount == geometry.referenceTokens else {
            throw Failure.badShape("the reference token block (geometry expected "
                + "\(geometry.referenceTokens))", got: referenceTokens.shape)
        }
        note?("\(references.count) reference(s), \(refCount) tokens beside "
            + "\(geometry.videoTokens) generated")

        note?("encoding the prompt")
        let text = try TextConditioningPipeline(textEncoder: textURL, dit: ditURL)
        let branches: Renderer.TextBranches
        if let guided {
            // A real second branch. The unguided path reuses the positive encoding for
            // both, which is correct only while `cfgScale` is 1.0 and the unconditional
            // pass is never consulted; under guidance that would subtract the prompt from
            // itself and neutralise CFG while still paying for the pass.
            let encoded = try text(prompt: prompt, negativePrompt: guided.negativePrompt)
            func conditioning(_ e: TextConditioningPipeline.Encoding) throws
                -> Pipeline.Conditioning {
                Pipeline.Conditioning(
                    video: try Renderer.require(e.features[.video], "features.video"),
                    audio: try Renderer.require(e.features[.audio], "features.audio"))
            }
            let positive = try conditioning(encoded[0])
            let negative = try conditioning(encoded[1])
            MLX.eval(positive.video, positive.audio, negative.video, negative.audio)
            branches = Renderer.TextBranches(positive: positive, negative: negative,
                                             validTokenCounts: encoded.map(\.validTokenCount))
        } else {
            let encoding = try text.encode(prompt: prompt, branch: 0)
            let conditioning = Pipeline.Conditioning(
                video: try Renderer.require(encoding.features[.video], "features.video"),
                audio: try Renderer.require(encoding.features[.audio], "features.audio"))
            MLX.eval(conditioning.video, conditioning.audio)
            branches = Renderer.TextBranches(positive: conditioning, negative: conditioning,
                                             validTokenCounts: [encoding.validTokenCount])
        }
        MLX.Memory.clearCache()

        let header = try SafetensorsHeader.read(from: ditURL)
        let topology = try TransformerTopology.read(header)
        try topology.verifyCrossModalWiring(header)
        let weights = try MLX.loadArrays(url: ditURL)

        // Validated against the base before a single forward: a LoRA naming a weight this
        // checkpoint does not have is a wrong-adapter mistake, and finding it after eight
        // steps at the target shape costs more than reading the key set costs.
        let overlay = try preloaded ?? LoRAOverlay.load(url: loraURL, strength: 1.0)
        try overlay.validate(againstBaseWeights: weights)
        note?("IC-LoRA attached: \(overlay.reports.map(\.description).joined(separator: "; "))")

        var forward = DiTForward(weights: weights, topology: topology, attentionPath: .fused)
        forward.attach(lora: overlay)
        let head = DiTOutputHead(weights: weights, topology: topology)
        // One cache per guidance pass, never one shared across them: a conditional residual
        // and an unconditional one are not comparable, and a shared cache would reuse across
        // that gap. Threshold 0 means the dense loop, which is the default and what every
        // measurement in this repo was taken with.
        // Built for every pass rather than for the planned ones: the plan needs a `Sampler`
        // that does not exist yet at this point, and a cache for a pass that never runs is
        // never queried. Cheap, and it cannot go stale against a guidance change.
        var stepCaches: [Sampler.Pass: StepCache] = [:]
        let plannedSteps = explicitSigmas.map { $0.count - 1 } ?? requestedSteps
        if cacheThreshold > 0 {
            for pass in Sampler.Pass.allCases {
                stepCaches[pass] = StepCache(pass: pass.description,
                                             threshold: cacheThreshold,
                                             maxConsecutiveSkips: cacheMaxSkips)
            }
            note?("step cache: threshold \(cacheThreshold), cap \(cacheMaxSkips) — a step "
                  + "whose block-0 residual moved less than the threshold reuses the previous")
        }
        let denoiser = Pipeline.DiTDenoiser(forward: forward, head: head, geometry: geometry,
                                            positive: branches.positive,
                                            negative: branches.negative,
                                            videoSTGBlocks: guided?.video.stgBlocks ?? [],
                                            audioSTGBlocks: guided?.audio.stgBlocks ?? [],
                                            stepCaches: stepCaches,
                                            cacheStepCount: stepCaches.isEmpty ? 0 : plannedSteps)

        // Stage 2 supplies its own schedule and its own starting point. Everything else —
        // the adapter, the appended reference, the clean mask — stays, which is the whole
        // difference between this and `refine`.
        let sigmas = try Self.schedule(explicit: explicitSigmas, guided: guided,
                                       requestedSteps: requestedSteps)
        let steps = sigmas.count - 1
        note?("schedule: \(steps) steps, sigmas \(sigmas)")
        let sampler = Sampler(video: guided?.video ?? DistilledRenderer.neutralGuidance,
                              audio: guided?.audio ?? DistilledRenderer.neutralGuidance)
        if let guided {
            note?("guidance: cfg \(guided.video.cfgScale)/\(guided.audio.cfgScale), "
                + "stg \(guided.video.stgScale) on \(guided.video.stgBlocks), "
                + "modality \(guided.video.modalityScale) -> "
                + "\(sampler.passCount(steps: steps)) forwards for the render")
        }
        // `RenderNoise` sizes itself from `videoTokens`, which stays the *generated* count,
        // so the draw is the same one an unconditioned render of this shape would make.
        let drawn = try RenderNoise(seed: seed, geometry: geometry).streamsForRenderOnly()

        let channels = drawn.video.dim(2)

        // A move *toward* noise, not an addition of it. Stage 2 starts
        // from the upsampled latent lerped to the schedule's first sigma; stage 1 starts from
        // noise alone, which is the same expression with nothing to lerp from.
        let startVideo: MLXArray
        if let initialVideo {
            let sigma = MLXArray(sigmas[0]).asType(drawn.video.dtype)
            startVideo = initialVideo.asType(drawn.video.dtype) * (1 - sigma)
                + drawn.video * sigma
        } else {
            startVideo = drawn.video
        }
        let videoNoise = MLX.concatenated(
            [startVideo, MLXArray.zeros([1, refCount, channels], dtype: drawn.video.dtype)],
            axis: 1)
        let videoMask = MLX.concatenated(
            [MLXArray.ones([1, geometry.videoTokens, 1], dtype: .float32),
             MLXArray.full([1, refCount, 1],
                           values: MLXArray(1 - referenceStrength)).asType(.float32)], axis: 1)
        let videoClean = MLX.concatenated(
            [MLXArray.zeros([1, geometry.videoTokens, channels], dtype: referenceTokens.dtype),
             referenceTokens], axis: 1)

        // The audio stream has no reference and must not acquire one. An all-ones mask
        // leaves `postProcess` returning the denoised audio unchanged and `timesteps`
        // reducing to the uniform schedule bit for bit.
        let audioMask = MLXArray.ones([1, drawn.audio.dim(1), 1], dtype: .float32)
        let audioClean = MLXArray.zeros(drawn.audio.shape, dtype: drawn.audio.dtype)

        let denoiseMask = Sampler.Streams(video: videoMask, audio: audioMask)
        let cleanLatent = Sampler.Streams(video: videoClean, audio: audioClean)
        let seeded = Sampler.Streams(
            video: Sampler.seedConditioned(noise: videoNoise, denoiseMask: videoMask,
                                           cleanLatent: videoClean),
            audio: Sampler.seedConditioned(noise: drawn.audio, denoiseMask: audioMask,
                                           cleanLatent: audioClean))

        let trajectory = try sampler.run(
            steps: steps, initial: seeded, denoiser: denoiser,
            denoiseMask: denoiseMask, cleanLatent: cleanLatent,
            explicitSigmas: sigmas,
            // The recipe's declared sampler when it named one, else the historical default:
            // ancestral for the distilled path, deterministic Euler for a guided one.
            stepper: (samplerKind?.isAncestral ?? (guided == nil))
                ? .ancestral(eta: DistilledRenderer.ancestralEta,
                             sNoise: DistilledRenderer.ancestralSNoise,
                             seed: seed &+ DistilledRenderer.ancestralNoiseSeedOffset)
                : .euler,
            retainHistory: false, evalCadence: evalCadence,
            cfgPP: (samplerKind?.usesCFGPP ?? false) ? 1 : 0,
            observer: { step, _ in note?("  step \(step + 1)/\(steps)") })

        // Drop the reference. It was never picture — decoding it would append the input
        // clip to the end of the output as extra frames' worth of tokens.
        let generated = trajectory.latents.video[0..., 0 ..< geometry.videoTokens, 0...]
        let video = try Sampler.unpatchifyVideo(
            generated,
            latentFrames: geometry.latentFrames,
            height: geometry.height / geometry.latent.spatialScale,
            width: geometry.width / geometry.latent.spatialScale)

        // The audio stream is returned rather than dropped. It is denoised on every step
        // regardless — the cross-modal bridges read it in every block — so discarding it
        // throws away a finished result. `upscale` keeps the source clip's own waveform and
        // ignores this; a render from noise has no source waveform and needs it.
        let audio = Sampler.unpatchifyAudio(trajectory.latents.audio,
                                            channels: geometry.latent.audioChannels)
        return (video: video, audio: audio)
    }

    // MARK: - Stage 2

    /// The recipe's second stage, on a latent that came from a file instead of a draft.
    ///
    /// Everything here is `DistilledRenderer`'s: `FlowSchedule.distilledStage2` for the
    /// sigmas, ``DistilledRenderer/neutralGuidance`` so `Sampler.plan` yields one
    /// conditional pass per step, and `.euler` because a 3-step schedule is too short to
    /// remove freshly injected noise and ancestral would inject some.
    ///
    /// The audio stream is the part that has no analogue in a draft-fed run. Stage 2 carries
    /// stage 1's audio latent, and there is no stage 1 here — so this runs the video stream
    /// against an audio latent of the right shape drawn from the same seed and then throws
    /// the audio away. The caller keeps the source's own waveform, so nothing downstream
    /// reads it; what it must not do is change *shape*, because the DiT's cross-modal
    /// attention is sized from it.
    static func refine(_ upscaled: MLXArray,
                               geometry: DiTForward.Geometry,
                               checkpoints: Checkpoints,
                               prompt: String,
                               sigmas: [Float],
                               note: ((String) -> Void)?) throws -> MLXArray {
        guard let ditURL = checkpoints.dit, let textURL = checkpoints.textEncoder else {
            throw Failure.missingCheckpoint("DiT or text encoder (both are required by "
                + "--mode refined)", path: "(not supplied)")
        }
        guard FileManager.default.fileExists(atPath: ditURL.path) else {
            throw Failure.missingCheckpoint("DiT", path: ditURL.path)
        }
        guard FileManager.default.fileExists(atPath: textURL.path) else {
            throw Failure.missingCheckpoint("text encoder", path: textURL.path)
        }

        // One text branch. There is no guider on this path, so the negative slot is
        // unreachable and gets the same tensors.
        // The prompt is not decoration here. Whatever the refine re-derives, it re-derives
        // toward its conditioning, and an empty prompt points at the base model's own
        // prior — which for this checkpoint is photographic. Handing it the film's style
        // string is what keeps the re-derived detail inside the film's look.
        note?(prompt.isEmpty
            ? "encoding an EMPTY prompt — the refine will drift toward the base model's prior"
            : "encoding the prompt (one branch, no negative)")
        let text = try TextConditioningPipeline(textEncoder: textURL, dit: ditURL)
        let encoding = try text.encode(prompt: prompt, branch: 0)
        let conditioning = Pipeline.Conditioning(
            video: try Renderer.require(encoding.features[.video], "features.video"),
            audio: try Renderer.require(encoding.features[.audio], "features.audio"))
        MLX.eval(conditioning.video, conditioning.audio)
        let branches = Renderer.TextBranches(positive: conditioning, negative: conditioning,
                                             validTokenCounts: [encoding.validTokenCount])
        MLX.Memory.clearCache()

        let header = try SafetensorsHeader.read(from: ditURL)
        let topology = try TransformerTopology.read(header)
        try topology.verifyCrossModalWiring(header)
        let weights = try MLX.loadArrays(url: ditURL)
        let forward = DiTForward(weights: weights, topology: topology, attentionPath: .fused)
        let head = DiTOutputHead(weights: weights, topology: topology)
        let denoiser = Pipeline.DiTDenoiser(forward: forward, head: head, geometry: geometry,
                                            positive: branches.positive,
                                            negative: branches.negative,
                                            videoSTGBlocks: [], audioSTGBlocks: [])

        let steps = sigmas.count - 1
        let sampler = Sampler(video: DistilledRenderer.neutralGuidance,
                              audio: DistilledRenderer.neutralGuidance)
        let noise = try RenderNoise(seed: DistilledRenderer.stage2NoiseSeedOffset,
                                    geometry: geometry)
        let drawn = noise.streamsForRenderOnly()

        // A noiser at a non-unit scale: lerp(initial, noise, 0.909375). A move *toward*
        // noise, not an addition of it.
        let tokens = try Sampler.patchifyVideo(upscaled)
        let scale = MLXArray(sigmas[0]).asType(tokens.dtype)
        let one = MLXArray(Float(1)).asType(tokens.dtype)
        let seeded = Sampler.Streams(
            video: tokens * (one - scale) + drawn.video.asType(tokens.dtype) * scale,
            audio: drawn.audio)

        let trajectory = try sampler.run(
            steps: steps, initial: seeded, denoiser: denoiser,
            denoiseMask: nil, cleanLatent: nil,
            explicitSigmas: sigmas, stepper: .euler, retainHistory: false,
            observer: { step, _ in note?("  step \(step + 1)/\(steps)") })

        return try Sampler.unpatchifyVideo(
            trajectory.latents.video,
            latentFrames: geometry.latentFrames,
            height: geometry.height / geometry.latent.spatialScale,
            width: geometry.width / geometry.latent.spatialScale)
    }
}

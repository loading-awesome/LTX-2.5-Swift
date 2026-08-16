// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import LTXRecipes
import MLX

/// Render from a prompt and **a cast** — up to five reference stills, each held apart.
///
/// The Multiple-Subject-Reference adapter is the answer to the thing single-reference
/// IC-LoRAs cannot do: put a named character, a second character, a prop and a location in
/// one shot and have each stay itself. Ingredients takes one sheet and the model decides
/// what on it is what. MSR takes the subjects separately and gives each one an identity the
/// transformer can address.
///
/// ## The two mechanisms, and why one without the other is useless
///
/// **A learned slot vector**, added to a reference's latent channels before it is
/// patchified — ``LTXModules/ReferenceSlotEmbedding``, five tensors that ship inside the
/// adapter beside its 960 LoRA weights. This is the tag.
///
/// **A distinct negative time offset**, `-(n - i)` pixel frames for slot `i` of `n`. This is
/// the separation. Without it every reference occupies the *same* coordinates — which is
/// what every pre-MSR reference does, deliberately, because there is only ever one of them.
/// Two references at identical positions are two things the attention cannot address
/// independently, and the render blends them.
///
/// Neither is checkable from the output: drop the slot vectors and the render still shows
/// the subjects, drop the offsets and it still completes. What goes wrong in both cases is
/// that the cast bleeds together, which reads as "the adapter is not very good".
///
/// ## What this follows, and where it diverges
///
/// The carrying mechanism is the one the single-reference adapters already use: tokens
/// appended to the video stream, a denoise mask of `1 - strength` holding them clean,
/// positions replaced by the reference's own grid. This port had that machinery already
/// and reuses it rather than growing a second one.
///
/// Two divergences worth stating rather than discovering:
///
/// * The adapter's metadata says `reference_token_order = prepend` and the tokens are
///   nevertheless **appended** — the transformer reads the guide block off the tail. See
///   contract 18. Following the metadata would put the references in front of the target,
///   which also moves which tokens the keyframe marker claims. This appends.
/// * The metadata carries no `reference_downscale_factor`, so the default of 1 applies and
///   no dilation happens. References are encoded at the target's own resolution.
public enum MSRRenderer {

    /// How many stills the adapter has slots for.
    ///
    /// Five, and it is a property of the trained weights rather than a limit chosen here:
    /// slot `i` is looked up through a Fourier feature of `(i + 1) / 16`, and only five
    /// slots were ever trained. A sixth reference would produce a perfectly well-formed
    /// vector from a region of that function no training ever visited.
    public static let maximumReferences = 5

    /// Pixel frames each still is repeated to before encoding.
    ///
    /// A still, repeated — not a single frame. 25 or 33, defaulting to 33, and the count is
    /// load-bearing twice: it decides how many latent frames the
    /// reference occupies (5 at 33, 4 at 25, through the causal VAE's `8k+1`), and with it
    /// how many tokens of the sequence the cast consumes. At five references and 33 frames
    /// the references outweigh a short target.
    public enum ReferenceFrames: Int, Sendable, CaseIterable {
        case short = 25
        case standard = 33
    }

    public struct Options {
        /// The reference stills, in slot order. `pic1` first — it takes the furthest-back
        /// time offset.
        public var references: [URL]
        /// A reference that is a location or a plate rather than a subject.
        ///
        /// It occupies the last slot and is resized differently: a background is always
        /// centre-crop-filled, because it is scenery and the framing is what matters,
        /// while a subject is padded rather than cropped when
        /// its aspect ratio disagrees with the target's — cropping a portrait to a landscape
        /// frame is how a character reference becomes a reference to that character's chin.
        public var background: URL?
        public var prompt: String
        public var output: URL
        /// The **output** width — what gets written.
        public var width: Int
        /// The **output** height.
        public var height: Int
        public var frames: Int
        public var frameRate: Double
        public var steps: Int
        public var seed: UInt64
        /// Conditioning strength; `1 - strength` fills the references' denoise mask, so 1.0
        /// holds them perfectly clean, which is the default.
        public var strength: Float
        public var referenceFrames: ReferenceFrames
        /// Non-nil runs on the dev transformer's continuous schedule with real guidance
        /// instead of the distilled draft table. Four forwards per step rather than one;
        /// the slot conditioning is unchanged.
        public var guided: VideoUpscaler.Guided?
        /// Whether the recipe declared a refine stage after the draft.
        ///
        /// For the reason ``IngredientsRenderer/Options/twoStage`` gives: it makes a larger
        /// output cheaper than rendering it natively, and does not make a given output
        /// faster. The `msr-prod` recipes are single stage for that reason.
        ///
        /// **It costs more here than it does there.** Stage 2 re-prepares, re-encodes and
        /// re-tags every slot at the output resolution, because `reference_downscale_factor`
        /// is 1 and a reference must sit on its own stage's grid — so a five-slot cast pays
        /// five encodes twice. That is real, and it is still the cheaper half of the trade:
        /// the encodes are seconds and the steps they save are minutes.
        public var twoStage: Bool
        /// Stage-2 steps. The stage-2 table is four sigmas long, so 3 is the schedule
        /// rather than a sample of it.
        public var refineSteps: Int

        /// The width stage 1 samples at, from the first stage's `scale`.
        public var sampledWidth: Int
        /// The height stage 1 samples at, from the first stage's `scale`.
        public var sampledHeight: Int
        /// Stage 1's sigmas, from the recipe's own `SigmaPlan`.
        public var stage1Sigmas: [Float]
        /// Stage 2's sigmas when a refine follows, scaled to the re-noise level.
        public var stage2Sigmas: [Float]?

        /// Every shape, schedule and guidance setting from a resolved recipe. See
        /// ``IngredientsRenderer/Options/init(reference:prompt:output:plan:seed:strength:negativePrompt:)``
        /// for why the geometry is not a parameter.
        public init(references: [URL], background: URL? = nil, prompt: String, output: URL,
                    plan: ResolvedRecipe,
                    seed: UInt64 = 42, strength: Float = 1.0,
                    referenceFrames: ReferenceFrames = .standard,
                    negativePrompt: String = "") throws {
            guard let first = plan.stages.first else {
                throw Failure.noStages(recipe: plan.recipeID)
            }
            self.references = references
            self.background = background
            self.prompt = prompt
            self.output = output
            self.width = plan.output.width
            self.height = plan.output.height
            self.sampledWidth = first.shape.width
            self.sampledHeight = first.shape.height
            self.frames = plan.frames
            self.frameRate = plan.frameRate
            self.seed = seed
            self.strength = strength
            self.referenceFrames = referenceFrames
            self.twoStage = plan.stages.count > 1
            self.steps = first.stage.sigmas.steps
            self.stage1Sigmas = try first.stage.sigmas.sigmas()
            let refine = plan.stages.dropFirst().first
            self.stage2Sigmas = try refine.map {
                try $0.stage.sigmas.sigmas(denoise: FlowSchedule.distilledStage2[0])
            }
            self.refineSteps = refine?.stage.sigmas.steps ?? 0
            let videoGuide = first.stage.videoGuidance
            self.guided = videoGuide == .distilled ? nil : VideoUpscaler.Guided(
                negativePrompt: negativePrompt,
                video: GuidanceParams(videoGuide),
                audio: GuidanceParams(first.stage.audioGuidance),
                steps: first.stage.sigmas.steps)
        }

        /// Every reference in slot order, with the background last, and whether each is one.
        var slots: [(url: URL, isBackground: Bool)] {
            references.map { ($0, false) } + (background.map { [($0, true)] } ?? [])
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case noReferences
        case tooManyReferences(Int)
        case notAnMSRAdapter(path: String, underlying: String)
        case referenceLatentFrames(slot: Int, got: Int, want: Int)
        case checkpointMissing(String, path: String)
        case audioCheckpointMissing
        case noStages(recipe: String)

        public var description: String {
            switch self {
            case let .noStages(recipe):
                // The grid check that used to live here is `RecipeStage.shape(forOutput:)`
                // now, run during resolution against every stage including a halved one.
                return "recipe '\(recipe)' declares no stages to sample"
            case .noReferences:
                return "MSR needs at least one reference still. With none, this is an "
                    + "ordinary render and `ltx render` is that path."
            case let .tooManyReferences(count):
                return "\(count) references, but the adapter carries \(maximumReferences) "
                    + "trained slots (counting the background as one). A sixth slot index "
                    + "would produce a well-formed embedding from a region of the slot "
                    + "function that was never trained."
            case let .notAnMSRAdapter(path, underlying):
                return "\(path) is not a Multiple-Subject-Reference adapter: \(underlying)"
            case let .referenceLatentFrames(slot, got, want):
                return "reference \(slot + 1) encoded to \(got) latent frames, not \(want). "
                    + "Every slot must occupy the same grid, because their token blocks are "
                    + "concatenated and their positions are built per block."
            case let .checkpointMissing(what, path):
                return "the \(what) is not at \(path)"
            case .audioCheckpointMissing:
                return "no audio VAE checkpoint. Every render here carries sound, so this "
                    + "is a missing checkpoint rather than an option to skip."
            }
        }
    }

    public struct Report {
        public let slots: Int
        public let referenceLatentShape: [Int]
        public let referenceTokens: Int
        public let generatedTokens: Int
        public let frameOffsets: [Int]
        public let slotNorms: [Float]
        public let framesWritten: Int
        /// What was written. Stage 1 sampled at half these when ``Options/twoStage``.
        public let outputWidth: Int
        public let outputHeight: Int
        public let encodeSeconds: Double
        public let sampleSeconds: Double
        /// The x2 upsample, the stage-2 re-encode of every slot, and the refine. Zero
        /// when single-stage.
        public let refineSeconds: Double
        public let decodeSeconds: Double
        public let peakGB: Double
    }

    // MARK: - Reference preparation

    /// One still, resized for its slot and repeated onto the VAE's frame lattice.
    ///
    /// **Not** the aspect-fill-then-crop every other conditioning image here goes through.
    /// It branches:
    ///
    /// * a background is always centre-crop-filled;
    /// * a subject whose aspect *family* matches the target and which is not smaller than it
    ///   is also centre-crop-filled;
    /// * everything else is fit inside the target and padded with **white**.
    ///
    /// The padding branch is the one that matters and the one a port would skip. A 3:4
    /// character portrait centre-cropped into a 16:9 frame is a reference to that
    /// character's face at the exclusion of everything the crop removed, and the model is
    /// then conditioned on a subject it can only partly see. White rather than black or
    /// edge-replicate, because the fill colour is something the adapter saw during
    /// training.
    static func prepared(image url: URL, width: Int, height: Int,
                         isBackground: Bool, frames: Int) throws -> MLXArray {
        let source = try MediaInput.image(at: url)
        let srcH = source.dim(1), srcW = source.dim(2)

        func family(_ w: Int, _ h: Int) -> String {
            let ratio = Double(w) / Double(h)
            if ratio >= 1.25 { return "landscape" }
            if ratio <= 0.8 { return "portrait" }
            return "square"
        }
        let sameFamily = family(srcW, srcH) == family(width, height)
        let sourceIsSmaller = srcW <= width && srcH <= height

        var still: MLXArray
        if isBackground || (sameFamily && !sourceIsSmaller) {
            still = try MediaInput.resizeAndCenterCrop(source, height: height, width: width)
        } else {
            // `min`, not `max`: fit the whole image inside the target and pad the remainder.
            let scale = Swift.min(Double(width) / Double(srcW), Double(height) / Double(srcH))
            let fitW = Swift.max(1, Swift.min(width, Int((Double(srcW) * scale).rounded())))
            let fitH = Swift.max(1, Swift.min(height, Int((Double(srcH) * scale).rounded())))
            var resized = source
            // Width first, then height — bilinear is separable only in the kernel's order.
            if fitW != srcW { resized = MediaInput.bilinearAlongAxis(resized, axis: 2,
                                                                     outSize: fitW) }
            if fitH != srcH { resized = MediaInput.bilinearAlongAxis(resized, axis: 1,
                                                                     outSize: fitH) }
            let left = (width - fitW) / 2, top = (height - fitH) / 2
            var canvas = MLXArray.ones([source.dim(0), height, width, source.dim(3)],
                                       dtype: resized.dtype)
            canvas[0..., top ..< (top + fitH), left ..< (left + fitW), 0...] = resized
            still = canvas
        }

        // Repeated, not held. The reference is a clip of a motionless subject, which is what
        // the adapter was shown; a single frame would encode to one latent frame and occupy
        // a fifth of the temporal extent its slot offset was chosen for.
        still = MLX.concatenated(Array(repeating: still, count: frames), axis: 0)
        return still
    }

    /// Every slot prepared, encoded and tagged **at one stage's resolution**.
    ///
    /// Called once per stage rather than once per render. `reference_downscale_factor` is 1
    /// on this adapter, so a reference's latent grid is the stage's own dimensions over the
    /// VAE's 32 — carrying stage 1's half-size latents into a full-size stage 2 would append
    /// a token block whose coordinates describe a grid the target no longer has.
    ///
    /// The slot vector is re-applied here rather than carried, for the same reason: it is
    /// added on the channel axis *before* patchify so it broadcasts across that reference's
    /// spatial positions, and stage 2 has four times as many of them.
    static func encodedSlots(_ options: Options, width: Int, height: Int,
                             embedding: ReferenceSlotEmbedding, videoVAE: URL,
                             note: ((String) -> Void)?) throws
        -> (latents: [MLXArray], norms: [Float]) {
        let slots = options.slots
        var latents: [MLXArray] = []
        var norms: [Float] = []
        let encoder = try VideoVAEEncoder(checkpoint: videoVAE)
        for (index, slot) in slots.enumerated() {
            let rgb = try prepared(image: slot.url, width: width, height: height,
                                   isBackground: slot.isBackground,
                                   frames: options.referenceFrames.rawValue)
            // bf16 in, for the reason every encode here uses it: an fp32 clip promotes all
            // 42 convolutions and computes more precisely than the encoder is meant to.
            var latent = try encoder.encode(encoder.pixels(fromRGB: rgb).asType(.bfloat16))
            let vector = embedding(slotIndex: index)
            MLX.eval(vector)
            norms.append(MLX.sqrt((vector * vector).sum()).item(Float.self))
            latent = try embedding.applied(to: latent, slotIndex: index)
            MLX.eval(latent)
            latents.append(latent)
            note?("  slot \(index + 1)"
                + (slot.isBackground ? " (background)" : "")
                + " \(slot.url.lastPathComponent) -> \(latent.shape),"
                + String(format: " embedding norm %.4f", norms[index]))
        }
        MLX.Memory.clearCache()

        // Every slot on the same grid. They are concatenated into one token block and their
        // positions are built per block from the geometry's own record of each; a slot that
        // encoded to a different number of latent frames would shift every later block's
        // coordinates onto the wrong tokens, with no shape disagreeing.
        let want = latents[0].dim(2)
        for (index, latent) in latents.enumerated() where latent.dim(2) != want {
            throw Failure.referenceLatentFrames(slot: index, got: latent.dim(2), want: want)
        }
        return (latents, norms)
    }

    /// The geometry for one stage, with each reference's slot vector and time offset recorded.
    static func geometry(_ options: Options, width: Int, height: Int,
                         references: [MLXArray]) -> DiTForward.Geometry {
        var geometry = DiTForward.Geometry(frames: options.frames, height: height,
                                           width: width, frameRate: options.frameRate)
        let count = references.count
        geometry.references = references.enumerated().map { index, latent in
            DiTForward.Geometry.Reference(
                latentFrames: latent.dim(2),
                latentHeight: latent.dim(3),
                latentWidth: latent.dim(4),
                downscaleFactor: 1,
                temporalScaleFactor: 1,
                // `-(num_slots - slot_index)`: pic1 furthest back, the last slot at -1.
                frameOffset: -(count - index),
                slotIndex: index)
        }
        return geometry
    }

    // MARK: - The run

    public static func render(_ options: Options,
                              checkpoints: VideoUpscaler.Checkpoints,
                              note: ((String) -> Void)? = nil) throws -> Report {
        MLX.GPU.resetPeakMemory()

        let slots = options.slots
        guard !slots.isEmpty else { throw Failure.noReferences }
        guard slots.count <= maximumReferences else {
            throw Failure.tooManyReferences(slots.count)
        }
        // Every checkpoint first, before any work. Learned on the Ingredients path: a guard
        // that sits at the decode fires after both stages have run.
        // No grid check: every stage's shape was validated during recipe resolution.
        guard let audioVAE = checkpoints.audioVAE else { throw Failure.audioCheckpointMissing }
        guard let adapterURL = checkpoints.icLoRA else {
            throw Failure.checkpointMissing("MSR adapter", path: "(not supplied)")
        }
        for (what, url) in [("video VAE", checkpoints.videoVAE), ("audio VAE", audioVAE),
                            ("MSR adapter", adapterURL)]
            + (options.twoStage ? [("upsampler", checkpoints.upsampler)] : []) {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Failure.checkpointMissing(what, path: url.path)
            }
        }

        // ---- The adapter, split -------------------------------------------------------
        //
        // Read once and taken apart here rather than inside the sampler: the slot
        // embeddings are needed *before* the transformer is loaded, because they are applied
        // to the reference latents, and `LoRAOverlay` refuses a map containing them.
        note?("reading the adapter")
        let (loraTensors, slotTensors) = ReferenceSlotEmbedding.split(
            try MLX.loadArrays(url: adapterURL))
        let embedding: ReferenceSlotEmbedding
        do {
            embedding = try ReferenceSlotEmbedding.load(from: slotTensors)
        } catch {
            throw Failure.notAnMSRAdapter(path: adapterURL.path, underlying: "\(error)")
        }
        let overlay = try LoRAOverlay.compose(name: adapterURL.lastPathComponent,
                                              tensors: loraTensors,
                                              strength: 1.0)
        note?("adapter: \(overlay.reports.map(\.description).joined(separator: "; "))")
        note?("slot embedding: \(slotTensors.count) tensors, "
            + "\(embedding.latentChannels) channels")

        // ---- References, at stage 1's resolution ----------------------------------------
        var t = Date()
        note?("encoding \(slots.count) reference(s) at "
            + "\(options.sampledWidth)x\(options.sampledHeight), "
            + "\(options.referenceFrames.rawValue) frames each")
        let (referenceLatents, slotNorms) = try encodedSlots(
            options, width: options.sampledWidth, height: options.sampledHeight,
            embedding: embedding, videoVAE: checkpoints.videoVAE, note: note)
        let encodeSeconds = -t.timeIntervalSinceNow
        let count = referenceLatents.count

        // ---- Geometry ------------------------------------------------------------------
        let geometry = Self.geometry(options, width: options.sampledWidth,
                                     height: options.sampledHeight,
                                     references: referenceLatents)
        let offsets = geometry.references.map(\.frameOffset)
        note?("slot time offsets \(offsets) pixel frames "
            + "(\(offsets.map { String(format: "%.4f", Double($0) / options.frameRate) }) s)")
        note?("\(geometry.referenceTokens) reference tokens beside "
            + "\(geometry.videoTokens) generated")

        // ---- Stage 1 -------------------------------------------------------------------
        t = Date()
        note?(options.twoStage
              ? "stage 1 at \(geometry.width)x\(geometry.height), half of the"
                  + " \(options.width)x\(options.height) output"
              : "single stage at \(geometry.width)x\(geometry.height)")
        let sampled = try VideoUpscaler.referenceGuided(
            references: referenceLatents, geometry: geometry, checkpoints: checkpoints,
            prompt: options.prompt, seed: options.seed, steps: options.steps,
            referenceStrength: options.strength,
            explicitSigmas: options.stage1Sigmas, guided: options.guided,
            overlay: overlay, note: note)
        MLX.eval(sampled.video, sampled.audio)
        MLX.Memory.clearCache()
        let sampleSeconds = -t.timeIntervalSinceNow

        // ---- Stage 2: x2 upsample, re-encode the cast, refine ---------------------------
        //
        // The Ingredients path's shape, with the one difference that matters: the whole cast
        // is re-prepared and re-tagged at the output resolution, not one sheet. The refine
        // goes back through `referenceGuided` carrying the same slot conditioning stage 1
        // had — `VideoUpscaler.refine` attaches no adapter and appends no reference, so it
        // would denoise with a transformer that had never seen the cast and lose exactly the
        // identities this command exists to hold.
        var videoLatent = sampled.video
        var audioLatent = sampled.audio
        var outGeometry = geometry
        var refineSeconds: Double = 0
        if options.twoStage {
            t = Date()
            note?("x2 latent upsample")
            let upscaled: MLXArray = try {
                let upsampler = try LatentSpatialUpsampler(checkpoint: checkpoints.upsampler,
                                                           videoVAE: checkpoints.videoVAE)
                let out = try upsampler.upsample(videoLatent)
                MLX.eval(out)
                return out
            }()
            MLX.Memory.clearCache()

            note?("re-encoding \(count) reference(s) at \(options.width)x\(options.height)")
            let (stage2References, _) = try encodedSlots(
                options, width: options.width, height: options.height,
                embedding: embedding, videoVAE: checkpoints.videoVAE, note: note)
            outGeometry = Self.geometry(options, width: options.width, height: options.height,
                                        references: stage2References)

            // Each schedule on the weights it belongs to — see the same seam on the
            // Ingredients path. Both curves start at 0.909375, so stage 2 continues stage 1
            // rather than re-rendering it, whichever recipe is running.
            note?("stage 2 refine at \(outGeometry.width)x\(outGeometry.height)")
            let stage2 = try VideoUpscaler.referenceGuided(
                references: stage2References, geometry: outGeometry,
                checkpoints: checkpoints, prompt: options.prompt,
                seed: options.seed &+ DistilledRenderer.stage2NoiseSeedOffset,
                steps: options.refineSteps, referenceStrength: options.strength,
                initialVideo: try Sampler.patchifyVideo(upscaled),
                explicitSigmas: options.stage2Sigmas,
                guided: options.guided?.refining(steps: options.refineSteps),
                overlay: overlay, note: note)
            MLX.eval(stage2.video, stage2.audio)
            MLX.Memory.clearCache()
            videoLatent = stage2.video
            // Stage 2 denoises the audio further even though it never upsamples it — the
            // audio latent is a different rank and never meets the upsampler. Keeping stage
            // 1's would throw away a refinement pass on the sound for no reason.
            audioLatent = stage2.audio
            refineSeconds = -t.timeIntervalSinceNow
        }

        // ---- Decode --------------------------------------------------------------------
        t = Date()
        note?("decoding video")
        let pixels: MLXArray = try {
            let decoder = try VideoVAEDecoder(checkpoint: checkpoints.videoVAE)
            let out = try Renderer.decodeVideoOnRecordedPolicy(decoder, videoLatent,
                                                               geometry: outGeometry)
            MLX.eval(out)
            return out
        }()
        MLX.Memory.clearCache()

        // Audio is not optional here either. The cross-modal bridges denoise it on every
        // step regardless, so a video-only write discards a finished result.
        note?("decoding audio")
        let mel: MLXArray = try {
            let decoder = try AudioVAEDecoder(checkpoint: audioVAE)
            let out = try decoder.decode(audioLatent)
            MLX.eval(out)
            return out
        }()
        MLX.Memory.clearCache()
        let vocoderConfig = try AudioVocoder.configuration(
            from: try SafetensorsHeader.read(from: audioVAE))
        let waveform: MLXArray = try {
            let vocoder = try AudioVocoder(checkpoint: audioVAE)
            let out = try vocoder.waveform(fromMel: mel)
            MLX.eval(out)
            return out
        }()
        MLX.Memory.clearCache()
        let decodeSeconds = -t.timeIntervalSinceNow

        // ---- Write ---------------------------------------------------------------------
        //
        // No lead-in trim, unlike the Ingredients path. That detector exists because a
        // reference sharing the target's own coordinates gets reproduced at the head of the
        // clip; these references are the one kind that provably do not share them. If a
        // bleed shows up anyway it is evidence the offsets are not doing their job, and
        // hiding it behind a trim would be the wrong thing to learn from.
        note?("writing \(options.output.lastPathComponent)")
        let rate = vocoderConfig.outputSamplingRate
        try RenderOutput.write(rgb: pixels, audio: waveform, to: options.output,
                               spec: RenderOutput.VideoSpec(fps: options.frameRate),
                               audioSpec: RenderOutput.AudioSpec(sampleRate: rate))

        return Report(slots: count,
                      referenceLatentShape: referenceLatents[0].shape,
                      referenceTokens: geometry.referenceTokens,
                      generatedTokens: geometry.videoTokens,
                      frameOffsets: offsets,
                      slotNorms: slotNorms,
                      framesWritten: pixels.dim(0),
                      outputWidth: outGeometry.width,
                      outputHeight: outGeometry.height,
                      encodeSeconds: encodeSeconds,
                      sampleSeconds: sampleSeconds,
                      refineSeconds: refineSeconds,
                      decodeSeconds: decodeSeconds,
                      peakGB: Double(MLX.GPU.peakMemory) / 1e9)
    }
}

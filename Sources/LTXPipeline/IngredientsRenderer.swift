// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX

/// Render from a prompt **and a reference sheet**, through an in-context LoRA.
///
/// The Ingredients IC-LoRA takes a single reference image — a sheet holding the characters,
/// objects and locations a shot should contain — and generates video that uses them. The
/// mechanism is the one ``VideoUpscaler/referenceGuided(reference:geometry:checkpoints:prompt:seed:steps:note:)``
/// already implements: the reference's latent tokens are appended to the video stream, held
/// clean by the denoise mask, and dropped before the decode. Only the *shape* of the job
/// differs — the target is generated from noise at a size the caller chooses, rather than
/// derived from an input clip.
///
/// ## What is settled, and what is not
///
/// Settled by measurement: all 480 of the adapter's modules resolve against the 2.5
/// transformer with matching shapes, and its `reference_downscale_factor` is **1**, so the
/// sheet is encoded at the target's own resolution rather than a fraction of it.
///
/// Not settled: the adapter is a **2.3** checkpoint and the base here is 2.5. It attaches,
/// which is a statement about key sets and nothing more. Whether the features it learned
/// transfer is an empirical question this type exists to let you answer.
///
/// ## Schedule
///
/// The IC-LoRA pipeline is built on the two-stage **distilled** schedule, so this runs on
/// the distilled transformer at the distilled sigmas — the same arrangement
/// `ltx upscale --mode ic-lora` uses. Pointing it at the dev transformer would need a
/// different schedule and different guidance.
public enum IngredientsRenderer {

    public struct Options {
        /// The reference sheet. A still image; a video's first frame would also encode, but
        /// the adapter was trained on sheets.
        public var reference: URL
        public var prompt: String
        public var output: URL
        public var width: Int
        public var height: Int
        public var frames: Int
        public var frameRate: Double
        public var steps: Int
        public var seed: UInt64
        /// Strength on the IC-LoRA overlay. 1.0 is how the adapter is meant to be applied.
        public var strength: Float
        /// Run the second stage: x2 latent upsample then refine, doubling both dimensions.
        ///
        /// The IC-LoRA pipeline is two-stage and so is this: stage 1 at
        /// `--width x --height`, then a x2 latent upsample and a refine at double both.
        ///
        /// The refine goes back through `referenceGuided`, **not**
        /// ``VideoUpscaler/refine(_:geometry:checkpoints:prompt:sigmas:note:)``. That was the
        /// first attempt and it was wrong: `refine` is the `--mode refined` path, attaching no
        /// adapter and appending no reference, so it re-noised to sigma 0.909 and denoised
        /// with a bare transformer that had never seen the sheet — terrain melted and drifted,
        /// visibly worse than stage 1 alone. Stage 2 must carry the same conditioning stage 1
        /// did.
        public var upsample: Bool
        /// Stage-2 steps. The stage-2 schedule is shorter than stage 1's.
        public var refineSteps: Int

        public init(reference: URL, prompt: String, output: URL,
                    width: Int, height: Int, frames: Int, frameRate: Double = 24,
                    steps: Int = 3, seed: UInt64 = 42, strength: Float = 1.0,
                    upsample: Bool = false, refineSteps: Int = 3) {
            self.reference = reference
            self.prompt = prompt
            self.output = output
            self.width = width
            self.height = height
            self.frames = frames
            self.frameRate = frameRate
            self.steps = steps
            self.seed = seed
            self.strength = strength
            self.upsample = upsample
            self.refineSteps = refineSteps
        }
    }

    /// Leading pixel frames that reproduce the reference sheet, measured in **latent space**.
    ///
    /// The reference block's tokens carry the same positional coordinates as the target's
    /// first latent frames. That is not an implementation artefact — with `downscaleFactor`
    /// and `temporalScaleFactor` both 1 neither adjustment branch fires at all, so the
    /// coordinates genuinely coincide. The adapter is meant to keep them apart through the
    /// attention it was trained
    /// on; a 2.3 checkpoint on a 2.5 base does not fully manage it, and the sheet is
    /// reproduced at the head of the clip.
    ///
    /// ## Two wrong versions preceded this one
    ///
    /// **A constant of 2.** That is what a 97-frame clip shows. A 361-frame clip reproduced
    /// the sheet for 18 frames, so the constant shipped nearly a second of contact sheet.
    ///
    /// **Pixel difference against the sheet, after stage 2.** The refine pass rewrites the
    /// copied sheet into a degraded, colour-shifted grid — still obviously a contact sheet to
    /// look at, but no longer a near copy, so it scored *above* the threshold and the
    /// detector reported nothing to drop.
    ///
    /// Latent space before the refine is where the signal actually is: there the copy is a
    /// near copy of the reference latent we already hold, so the comparison costs one
    /// reduction and no decode.
    ///
    /// The latent-to-pixel conversion is the causal VAE's: latent frame 0 is a single pixel
    /// frame, every later one covers 8. Both prior measurements fall out of it — one
    /// contaminated latent frame predicts 2 pixel frames (the 97-frame case), three predict
    /// 18 (the 361-frame case).
    static func contaminatedLeadIn(videoLatent: MLXArray, reference: MLXArray,
                                   temporalScale: Int) -> Int {
        let frames = videoLatent.dim(2)
        guard frames > 4 else { return 0 }

        // `[1, C, F, H, W] -> [F]`, one reduction over the whole clip.
        let difference = MLX.abs(videoLatent - reference).mean(axes: [1, 3, 4]).reshaped([-1])
        MLX.eval(difference)
        let scores = difference.asArray(Float.self)

        // "Clean" is the clip\'s own second half, which cannot be contaminated — the artefact
        // is a lead-in. A median there keeps a long contaminated head from dragging the
        // reference level toward itself.
        let late = Array(scores[(frames / 2)...]).sorted()
        let clean = late[late.count / 2]
        // 0.9, not 0.6. The artefact does not cut — it *dissolves*: the opaque copies give
        // way to several frames of ghosted sheet cross-fading into the scene, so the score
        // ramps toward the clean level rather than stepping to it. A threshold tuned to the
        // opaque frames leaves the ghosting on screen, which is what 0.6 did. Over-trimming
        // costs a fraction of a second; under-trimming opens the clip on a contact sheet.
        let threshold = clean * 0.9

        var latentFrames = 0
        while latentFrames < frames / 2, scores[latentFrames] < threshold { latentFrames += 1 }
        guard latentFrames > 0 else { return 0 }

        // Causal VAE: latent 0 -> 1 pixel frame, each later latent -> `temporalScale`. The
        // extra frame is the decoder\'s smear across the transition.
        return 1 + (latentFrames - 1) * temporalScale + 1
    }

    public struct Report {
        public let referenceLatentShape: [Int]
        public let referenceTokens: Int
        public let generatedTokens: Int
        public let outputWidth: Int
        public let outputHeight: Int
        public let framesDropped: Int
        public let framesWritten: Int
        public let encodeSeconds: Double
        public let sampleSeconds: Double
        public let refineSeconds: Double
        public let decodeSeconds: Double
        public let peakGB: Double
    }

    public enum Failure: Error, CustomStringConvertible {
        case referenceFrames(Int)
        case audioCheckpointMissing
        case checkpointMissing(String, path: String)

        public var description: String {
            switch self {
            case let .referenceFrames(count):
                return "the reference encoded to \(count) latent frames; a sheet should be a "
                    + "single still and encode to 1. A multi-frame reference is a different "
                    + "conditioning shape than the adapter was trained on."
            case let .checkpointMissing(what, path):
                return "the \(what) is not at \(path)"
            case .audioCheckpointMissing:
                return "no audio VAE checkpoint. Every render here carries sound, so this "
                    + "is a missing checkpoint rather than an option to skip."
            }
        }
    }

    public static func render(_ options: Options,
                              checkpoints: VideoUpscaler.Checkpoints,
                              note: ((String) -> Void)? = nil) throws -> Report {
        MLX.GPU.resetPeakMemory()

        // Every checkpoint this render will need, checked before a single step runs.
        //
        // Learned the expensive way: the audio VAE was missing from the CLI's checkpoint set,
        // the guard sat at the decode, and a 14-minute two-stage render at 15 seconds threw
        // after both stages had finished. A precondition that fires at the end of the work it
        // was meant to protect is not a precondition.
        guard let audioVAE = checkpoints.audioVAE else {
            throw Failure.audioCheckpointMissing
        }
        for (what, url) in [("video VAE", checkpoints.videoVAE), ("audio VAE", audioVAE)]
            + (options.upsample ? [("upsampler", checkpoints.upsampler)] : []) {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Failure.checkpointMissing(what, path: url.path)
            }
        }

        // ---- Reference ---------------------------------------------------------------
        //
        // Encoded at the *target's* resolution. `reference_downscale_factor` is 1 on this
        // adapter, and the sheet's grid is the output dimensions divided by that factor —
        // so for Ingredients the sheet occupies a full frame's worth of latent grid, not a
        // thumbnail's.
        var t = Date()
        note?("encoding the reference sheet at \(options.width)x\(options.height)")
        let referenceLatent: MLXArray = try {
            let rgb = try MediaInput.image(at: options.reference,
                                           height: options.height, width: options.width)
            let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
            // bf16 for the same reason every other encode here uses it: an fp32 clip
            // promotes all 42 convolutions and computes more precisely than the encoder is
            // meant to.
            let latent = try encoder.encode(encoder.pixels(fromRGB: rgb).asType(.bfloat16))
            MLX.eval(latent)
            return latent
        }()
        MLX.Memory.clearCache()
        let encodeSeconds = -t.timeIntervalSinceNow

        guard referenceLatent.dim(2) == 1 else {
            throw Failure.referenceFrames(referenceLatent.dim(2))
        }

        var geometry = DiTForward.Geometry(frames: options.frames,
                                           height: options.height,
                                           width: options.width,
                                           frameRate: options.frameRate)
        geometry.reference = DiTForward.Geometry.Reference(
            latentFrames: referenceLatent.dim(2),
            latentHeight: referenceLatent.dim(3),
            latentWidth: referenceLatent.dim(4),
            downscaleFactor: 1,
            temporalScaleFactor: 1)
        note?("reference latent \(referenceLatent.shape) -> \(geometry.referenceTokens) tokens"
            + " beside \(geometry.videoTokens) generated")

        // ---- Stage 1 -----------------------------------------------------------------
        t = Date()
        let stage1 = try VideoUpscaler.referenceGuided(
            references: [referenceLatent], geometry: geometry, checkpoints: checkpoints,
            prompt: options.prompt, seed: options.seed, steps: options.steps,
            referenceStrength: options.strength, note: note)
        MLX.eval(stage1.video, stage1.audio)
        MLX.Memory.clearCache()
        let sampleSeconds = -t.timeIntervalSinceNow

        // ---- Stage 2: x2 upsample, then refine -----------------------------------------
        //
        // The IC-LoRA pipeline's own shape: the two-stage distilled schedule plus the
        // latent upsampler.
        //
        // **Only the video latent is upsampled.** The audio latent is a different rank and
        // never meets the upsampler. So the audio written below is stage 1's, at full
        // length and already final — refining video does not change what the scene sounds
        // like.
        var videoLatent = stage1.video
        // Stage 2 denoises the audio further even though it never upsamples it, and
        // `DistilledRenderer` writes that second-stage result rather than the draft's —
        // `Sampler.unpatchifyAudio(stage2.latents.audio, ...)`. Keeping stage 1's would throw
        // away a refinement pass on the sound for no reason.
        var audioLatent = stage1.audio
        var outGeometry = geometry
        var refineSeconds: Double = 0
        if options.upsample {
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

            outGeometry = DiTForward.Geometry(frames: options.frames,
                                              height: options.height * 2,
                                              width: options.width * 2,
                                              frameRate: options.frameRate)

            // The sheet again, at the doubled size. `reference_downscale_factor` is 1, so the
            // reference always matches the stage's own resolution — reusing stage 1's latent
            // here would append a half-size grid whose coordinates no longer line up.
            note?("re-encoding the reference at \(outGeometry.width)x\(outGeometry.height)")
            let stage2Reference: MLXArray = try {
                let rgb = try MediaInput.image(at: options.reference,
                                               height: outGeometry.height,
                                               width: outGeometry.width)
                let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
                let out = try encoder.encode(encoder.pixels(fromRGB: rgb).asType(.bfloat16))
                MLX.eval(out)
                return out
            }()
            MLX.Memory.clearCache()
            outGeometry.reference = DiTForward.Geometry.Reference(
                latentFrames: stage2Reference.dim(2),
                latentHeight: stage2Reference.dim(3),
                latentWidth: stage2Reference.dim(4),
                downscaleFactor: 1, temporalScaleFactor: 1)

            // `referenceGuided` again, not `VideoUpscaler.refine`.
            //
            // That distinction is the whole fix. `refine` is the `--mode refined` path: it
            // attaches no adapter and appends no reference, so it re-noised to sigma 0.909 and
            // then denoised with a bare transformer that had never seen the sheet. Measured
            // result: terrain that melted and drifted, visibly worse than stage 1 alone.
            // Stage 2 has to carry the same conditioning stage 1 did, or it is not refining
            // the same picture.
            note?("stage 2 refine at \(outGeometry.width)x\(outGeometry.height)")
            let stage2 = try VideoUpscaler.referenceGuided(
                references: [stage2Reference], geometry: outGeometry, checkpoints: checkpoints,
                prompt: options.prompt,
                seed: options.seed &+ DistilledRenderer.stage2NoiseSeedOffset,
                steps: options.refineSteps, referenceStrength: options.strength,
                initialVideo: try Sampler.patchifyVideo(upscaled),
                explicitSigmas: FlowSchedule.distilledStage2, note: note)
            MLX.eval(stage2.video, stage2.audio)
            MLX.Memory.clearCache()
            videoLatent = stage2.video
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

        // Audio is not optional. The DiT denoises it on every step regardless — the
        // cross-modal bridges read it in every block — so a video-only write throws away a
        // finished result and ships a silent file.
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

        // ---- Trim and write ------------------------------------------------------------
        let dropped = min(Self.contaminatedLeadIn(videoLatent: stage1.video,
                                                  reference: referenceLatent,
                                                  temporalScale: geometry.latent.temporalScale),
                          pixels.dim(0) / 2)
        let trimmed = dropped > 0 ? pixels[dropped...] : pixels
        note?(dropped > 0
              ? "dropping \(dropped) frames that reproduce the reference sheet"
                  + " -> \(trimmed.dim(0)) frames"
              : "no reference bleed detected")

        // The audio is trimmed by the same wall-clock span, or picture and sound drift apart
        // by exactly the length of the discarded head.
        let rate = vocoderConfig.outputSamplingRate
        let audioOffset = Int((Double(dropped) / options.frameRate) * Double(rate))
        let audio = audioOffset > 0 && audioOffset < waveform.dim(-1)
            ? waveform[0..., 0..., audioOffset...]
            : waveform

        note?("writing \(options.output.lastPathComponent)")
        try RenderOutput.write(rgb: trimmed, audio: audio, to: options.output,
                               spec: RenderOutput.VideoSpec(fps: options.frameRate),
                               audioSpec: RenderOutput.AudioSpec(sampleRate: rate))

        return Report(referenceLatentShape: referenceLatent.shape,
                      referenceTokens: geometry.referenceTokens,
                      generatedTokens: geometry.videoTokens,
                      outputWidth: outGeometry.width,
                      outputHeight: outGeometry.height,
                      framesDropped: dropped,
                      framesWritten: trimmed.dim(0),
                      encodeSeconds: encodeSeconds,
                      sampleSeconds: sampleSeconds,
                      refineSeconds: refineSeconds,
                      decodeSeconds: decodeSeconds,
                      peakGB: Double(MLX.GPU.peakMemory) / 1e9)
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import LTXModules
import MLX

/// Image, video and audio conditioning: what to freeze, and what to freeze it to.
///
/// ## The one seam all three share
///
/// `Sampler.postProcess` is `denoised * mask + clean * (1 - mask)`, applied **after every
/// step**. So conditioning is two token-space tensors per stream — a `cleanLatent` saying
/// what the conditioned tokens should be, and a `denoiseMask` saying how strongly to
/// insist. Everything below is the business of building that pair; the sampler already
/// knows what to do with it.
///
/// That the blend is re-imposed *every step* rather than seeded once is the fact behind
/// the surprising behaviour of strength: a high image strength does not merely start the
/// render near your image, it holds it there for all thirty steps, which is why a locked
/// frame-0 anchor stops a camera move from happening at all.
///
/// ## Mask polarity, which is easy to get backwards
///
/// **1 means free, 0 means frozen.** `mask = 1 - strength`: a strength of 1.0 pins the
/// tokens exactly, 0.0 leaves them untouched, and the interesting values are in between.
/// Getting this inverted produces a render that ignores the conditioning it was given and
/// conditions on the parts you left alone — which looks like a broken encoder rather than
/// a flipped sign.
///
/// ## Mask dtype is load-bearing
///
/// `postProcess` computes the blend in the **mask's** dtype: only `clean` is cast, and
/// `denoised * denoiseMask` promotes. These masks are built in fp32, so the blend is fp32
/// and rounds once on the way out. A bf16 mask would round twice.
public enum RenderConditioning {

    public enum Failure: Error, CustomStringConvertible {
        case conflicting(String)
        case strengthOutOfRange(Double)
        case imageFrames(Int)
        case videoLatentShape(got: [Int], expected: [Int])
        case audioLatentShape(got: [Int], expected: [Int])
        case audioFromWaveformUnsupported
        case keyframeOffLattice(pixelFrame: Int, temporalScale: Int,
                                nearestBelow: Int, nearestAbove: Int)
        case keyframeOutOfRange(pixelFrame: Int, frames: Int)
        case duplicateKeyframe(pixelFrame: Int)

        public var description: String {
            switch self {
            case let .conflicting(why): return why
            case let .strengthOutOfRange(s):
                return "conditioning strength must be in 0...1, got \(s)"
            case let .keyframeOffLattice(pixelFrame, scale, below, above):
                return "a keyframe at pixel frame \(pixelFrame) does not land on a latent "
                     + "frame boundary. The VAE is causal: frame 0 encodes alone, then every "
                     + "\(scale) pixel frames become one latent. Placing a still inside a "
                     + "chunk would freeze all \(scale) frames of it to that one image, "
                     + "which is not what was asked for. The nearest boundaries are "
                     + "\(below) and \(above)."
            case let .keyframeOutOfRange(pixelFrame, frames):
                return "a keyframe at pixel frame \(pixelFrame) is outside this render's "
                     + "0...\(frames - 1)"
            case let .duplicateKeyframe(pixelFrame):
                return "two keyframes both condition pixel frame \(pixelFrame); the second "
                     + "would overwrite the first"
            case let .imageFrames(n):
                return "an image conditions one latent frame; this encoded to \(n). Pass a "
                     + "still, or use video conditioning."
            case let .videoLatentShape(got, expected):
                return "the conditioning video encoded to \(got), but this render's latent "
                     + "is \(expected). Conditioning must match the render's geometry."
            case let .audioLatentShape(got, expected):
                return "the conditioning audio encoded to \(got), but this render's audio "
                     + "latent is \(expected)."
            case .audioFromWaveformUnsupported:
                return "audio-to-video needs a mel spectrogram, and this port cannot yet "
                     + "produce one from a waveform: the required front end is "
                     + "n_fft 1024, hop 160, slaney mel, log with a 1e-5 clamp, "
                     + "which is a DIFFERENT transform from the "
                     + "vocoder's MelSTFT (512/80) and is not implemented or measured "
                     + "here. Supply a recorded mel instead."
            }
        }
    }

    /// Which conditioning a render is doing.
    ///
    /// **Derived from what was supplied, never passed as a flag.** This has been learned
    /// the expensive way: DiT partitions that share an architecture and differ only in
    /// weights render under the wrong mode without error and without the training that
    /// mode relied on, so a runner with a hard-coded path cost a whole day of wrong
    /// renders. A mode that cannot disagree with its inputs cannot be set wrong.
    public enum Mode: String, Sendable {
        case textToAV
        case imageToVideo
        case videoToVideo
        case audioToVideo
    }

    public struct Built {
        public var cleanLatent: Sampler.Streams
        public var denoiseMask: Sampler.Streams
        public var mode: Mode
        /// One line for the provenance sidecar. A conditioned render that does not say
        /// what it was conditioned on is not reproducible.
        public var provenance: String
    }

    // MARK: - Building

    /// One still, pinned to one latent frame.
    ///
    /// A first frame, a last frame, or any set in between — the write is token-indexed, so
    /// nothing about frame 0 was ever special except that it was the only index the caller
    /// could name. `latentIndex(forPixelFrame:geometry:)` is what decides where it lands.
    ///
    /// **Strength is per keyframe.** An anchored opening frame and a loosely-suggested
    /// closing one is a normal request, and a single render-wide strength cannot express it.
    /// Not `Sendable`: `MLXArray` is not, which is why ``Built`` is not either. Conditioning
    /// is assembled and consumed inside one render task.
    public struct Keyframe {
        /// `[1, C, 1, H/32, W/32]` — one latent frame, as ``encodeImage(_:encoder:)`` returns.
        public let latent: MLXArray
        /// The **pixel** frame this still occupies. Must land on a latent boundary.
        public let pixelFrame: Int
        public let strength: Double

        public init(latent: MLXArray, pixelFrame: Int, strength: Double) {
            self.latent = latent
            self.pixelFrame = pixelFrame
            self.strength = strength
        }
    }

    /// Which latent frame a pixel frame belongs to, refusing anything mid-chunk.
    ///
    /// The VAE is causal, so frame 0 encodes to latent 0 on its own and every `temporalScale`
    /// pixel frames after it collapse into one more. Latent `k >= 1` therefore covers pixel
    /// frames `scale(k-1)+1 ... scale·k`, and the only pixel frames that name a latent
    /// *exactly* are 0 and the multiples of `scale`.
    ///
    /// Rounding instead of refusing is the tempting shortcut and it is wrong in a way nobody
    /// would see: a still requested at frame 5 would silently freeze frames 1 through 8.
    /// This is the same argument ``LTXRecipes/RecipeRequest/resolvedFrames(lattice:)`` makes
    /// for validating an explicit frame count rather than snapping it.
    public static func latentIndex(forPixelFrame pixelFrame: Int,
                                   geometry: DiTForward.Geometry) throws -> Int {
        guard pixelFrame >= 0, pixelFrame < geometry.frames else {
            throw Failure.keyframeOutOfRange(pixelFrame: pixelFrame, frames: geometry.frames)
        }
        let scale = geometry.latent.temporalScale
        guard pixelFrame % scale == 0 else {
            let below = (pixelFrame / scale) * scale
            throw Failure.keyframeOffLattice(pixelFrame: pixelFrame, temporalScale: scale,
                                             nearestBelow: below, nearestAbove: below + scale)
        }
        let index = pixelFrame / scale
        guard index < geometry.latentFrames else {
            throw Failure.keyframeOutOfRange(pixelFrame: pixelFrame, frames: geometry.frames)
        }
        return index
    }

    /// Assemble the conditioning pair, or `nil` for an unconditioned render.
    ///
    /// `nil` rather than an identity pair on purpose: `Sampler.postProcess` short-circuits
    /// on a nil mask, so a text-to-AV render does no blend arithmetic at all rather than
    /// multiplying by ones thirty times.
    ///
    /// Encoders are passed in already loaded. They are 84 and 44 tensors against the DiT's
    /// 4349, but a caller that renders a batch should not reload them per clip.
    /// One still at frame 0 — the original surface, kept so every existing call site keeps
    /// compiling and keeps meaning exactly what it meant.
    public static func build(
        imageLatent: MLXArray?,
        videoLatent: MLXArray? = nil,
        audioLatent: MLXArray? = nil,
        strength: Double,
        geometry: DiTForward.Geometry
    ) throws -> Built? {
        try build(keyframes: imageLatent.map {
                      [Keyframe(latent: $0, pixelFrame: 0, strength: strength)]
                  } ?? [],
                  videoLatent: videoLatent, audioLatent: audioLatent,
                  strength: strength, geometry: geometry)
    }

    public static func build(
        keyframes: [Keyframe] = [],
        videoLatent: MLXArray? = nil,
        audioLatent: MLXArray? = nil,
        strength: Double,
        geometry: DiTForward.Geometry
    ) throws -> Built? {
        guard (0.0 ... 1.0).contains(strength) else {
            throw Failure.strengthOutOfRange(strength)
        }
        for keyframe in keyframes where !(0.0 ... 1.0).contains(keyframe.strength) {
            throw Failure.strengthOutOfRange(keyframe.strength)
        }
        // Image and video both condition the video stream and would overwrite each other.
        // Refusing beats picking one: "there is no checkpoint that could honour both" is a
        // better error than a render that silently honoured the second.
        if !keyframes.isEmpty && videoLatent != nil {
            throw Failure.conflicting("an image and a video both condition the video "
                + "stream; supply one or the other")
        }
        if keyframes.isEmpty, videoLatent == nil, audioLatent == nil { return nil }

        let latentFrames = geometry.latentFrames
        let perFrame = geometry.tokensPerLatentFrame
        let videoTokens = geometry.videoTokens
        let videoChannels = geometry.latent.videoChannels
        let audioTokens = geometry.audioTokens
        let audioChannels = geometry.latent.audioChannels * geometry.latent.audioFeatures

        // Free everywhere by default; the branches below clamp what they condition.
        var cleanVideo = MLXArray.zeros([1, videoTokens, videoChannels], dtype: .float32)
        var maskVideo = MLXArray.ones([1, videoTokens, 1], dtype: .float32)
        var cleanAudio = MLXArray.zeros([1, audioTokens, audioChannels], dtype: .float32)
        var maskAudio = MLXArray.ones([1, audioTokens, 1], dtype: .float32)

        var mode = Mode.textToAV
        var detail: [String] = []
        let held = Float(1.0 - strength)

        if !keyframes.isEmpty {
            // Each still is one latent frame written into an otherwise-zero latent. The VAE
            // is causal, so a single pixel frame encodes to exactly one latent frame — and
            // latent frame 0 is also the one carrying the keyframe marker (contract 15),
            // which is applied inside the DiT and is not this layer's business.
            let expected = [1, videoChannels, 1,
                            geometry.height / geometry.latent.spatialScale,
                            geometry.width / geometry.latent.spatialScale]
            var written = Set<Int>()
            // Sorted so the provenance line reads in time order regardless of argument
            // order, and so a duplicate is reported against the earlier of the pair.
            for keyframe in keyframes.sorted(by: { $0.pixelFrame < $1.pixelFrame }) {
                guard keyframe.latent.shape == expected else {
                    guard keyframe.latent.ndim == 5 else {
                        throw Failure.videoLatentShape(got: keyframe.latent.shape,
                                                       expected: expected)
                    }
                    throw Failure.imageFrames(keyframe.latent.dim(2))
                }
                let index = try latentIndex(forPixelFrame: keyframe.pixelFrame,
                                            geometry: geometry)
                guard written.insert(index).inserted else {
                    throw Failure.duplicateKeyframe(pixelFrame: keyframe.pixelFrame)
                }
                let range = (index * perFrame) ..< ((index + 1) * perFrame)
                let tokens = try Sampler.patchifyVideo(keyframe.latent).asType(.float32)
                cleanVideo[0..., range, 0...] = tokens
                maskVideo[0..., range, 0...] = MLXArray(Float(1.0 - keyframe.strength))
                detail.append("image at pixel frame \(keyframe.pixelFrame) "
                    + "(latent \(index)), strength \(keyframe.strength)")
            }
            mode = .imageToVideo
        }

        if let video = videoLatent {
            let expected = [1, videoChannels, latentFrames,
                            geometry.height / geometry.latent.spatialScale,
                            geometry.width / geometry.latent.spatialScale]
            guard video.shape == expected else {
                throw Failure.videoLatentShape(got: video.shape, expected: expected)
            }
            cleanVideo = try Sampler.patchifyVideo(video).asType(.float32)
            maskVideo = MLXArray.full([1, videoTokens, 1], values: MLXArray(held))
            mode = .videoToVideo
            detail.append("video over all \(latentFrames) latent frames, strength \(strength)")
        }

        if let audio = audioLatent {
            let expected = [1, geometry.latent.audioChannels, audioTokens,
                            geometry.latent.audioFeatures]
            guard audio.shape == expected else {
                throw Failure.audioLatentShape(got: audio.shape, expected: expected)
            }
            cleanAudio = try Sampler.patchifyAudio(audio).asType(.float32)
            // Audio-to-video freezes the audio stream ENTIRELY and denoises video against
            // it. Partial audio strength is not a supported mode, so the strength argument
            // is deliberately ignored here rather than silently reinterpreted — and this
            // is said out loud in the provenance.
            maskAudio = MLXArray.zeros([1, audioTokens, 1], dtype: .float32)
            if mode == .textToAV { mode = .audioToVideo }
            detail.append("audio frozen entirely (all \(audioTokens) latents); the "
                + "strength argument does not apply to it")
        }

        return Built(
            cleanLatent: Sampler.Streams(video: cleanVideo, audio: cleanAudio),
            denoiseMask: Sampler.Streams(video: maskVideo, audio: maskAudio),
            mode: mode,
            provenance: "\(mode.rawValue): " + detail.joined(separator: "; ")
                + ". Mask polarity 1=free 0=frozen, blend recomputed every step. "
                + "Mask polarity, token range and timesteps = mask * sigma are checked; "
                + "the trajectory consuming them is not.")
    }

    /// Encode a still image to a one-frame latent, ready for ``build(imageLatent:...)``.
    ///
    /// Takes the deterministic path — the distribution's mode, not a sample. A sampled
    /// conditioning latent would make a fixed seed stop reproducing its own render, and a
    /// render nobody can re-run is an anecdote.
    public static func encodeImage(_ rgb: MLXArray, encoder: VideoVAEEncoder) throws -> MLXArray {
        try encoder.encode(encoder.pixels(fromRGB: rgb))
    }

    /// Encode video frames to a latent covering the whole clip.
    public static func encodeVideo(_ rgb: MLXArray, encoder: VideoVAEEncoder) throws -> MLXArray {
        try encoder.encode(encoder.pixels(fromRGB: rgb))
    }

    /// Encode a **mel spectrogram** to an audio latent.
    ///
    /// Not a waveform. See ``Failure/audioFromWaveformUnsupported`` for why the front end
    /// is missing and what it would take.
    public static func encodeAudioMel(_ mel: MLXArray, encoder: AudioVAEEncoder) throws -> MLXArray {
        try encoder.encode(mel)
    }
}

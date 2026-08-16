// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Latent shapes for a render request.
///
/// Arithmetic, no MLX, no checkpoint — so it is tested in milliseconds and its
/// errors cannot hide. Getting a latent shape wrong produces a confusing failure
/// several stages downstream, which is exactly the class of bug this layer exists
/// to make impossible.
///
/// ## Every rule here is checked against a measured shape
///
/// At 320×192, 25 frames, 24 fps, LTX 2.5 produces:
///
/// ```
/// video latent  (1, 128, 4, 6, 10)
/// audio latent  (1, 8, 26, 16)
/// ```
///
/// which pins every factor below: 320/32 = 10, 192/32 = 6, (25−1)/8 + 1 = 4,
/// 128 channels from the checkpoint's `in_channels`, 8 from the audio VAE's
/// `latent_channels`, and `round(25/24 × 25) = 26` audio latents.
///
/// The spatial and temporal factors are **read from the checkpoint**, not
/// declared: config that the checkpoint carries should come from the checkpoint
/// and never from a constant, because a hardcoded dimension is right until the
/// day a checkpoint disagrees with it and then wrong in silence. The defaults
/// below are what 2.5 reports; a caller that has a probed config should pass it.
public struct LatentGeometry: Sendable, Equatable {

    /// Spatial downscale of the video VAE. 32 for 2.3 and 2.5.
    public let spatialScale: Int
    /// Temporal compression, and the same `8` as the frame lattice.
    public let temporalScale: Int
    /// Video latent channels — the checkpoint's `in_channels`/`out_channels`.
    public let videoChannels: Int
    /// Audio VAE latent channels.
    public let audioChannels: Int
    /// Audio latent feature dimension.
    public let audioFeatures: Int
    /// Audio latents per second of wall-clock audio.
    public let audioLatentsPerSecond: Double

    public init(spatialScale: Int = 32,
                temporalScale: Int = 8,
                videoChannels: Int = 128,
                audioChannels: Int = 8,
                audioFeatures: Int = 16,
                audioLatentsPerSecond: Double = 25.0) {
        self.spatialScale = spatialScale
        self.temporalScale = temporalScale
        self.videoChannels = videoChannels
        self.audioChannels = audioChannels
        self.audioFeatures = audioFeatures
        self.audioLatentsPerSecond = audioLatentsPerSecond
    }

    public var lattice: FrameLattice { FrameLattice(timeScale: temporalScale) }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case offLattice(frames: Int, nearest: Int)
        case notMultipleOfScale(dimension: String, value: Int, scale: Int)
        case nonPositive(dimension: String, value: Int)

        public var description: String {
            switch self {
            case let .offLattice(frames, nearest):
                return "\(frames) frames is not on the 8k+1 lattice; nearest at or below "
                    + "is \(nearest). Off-lattice counts are not renderable (contract 4)"
            case let .notMultipleOfScale(dimension, value, scale):
                return "\(dimension) \(value) is not a multiple of \(scale); the VAE "
                    + "downscales spatially by \(scale)"
            case let .nonPositive(dimension, value):
                return "\(dimension) must be positive, got \(value)"
            }
        }
    }

    /// Video latent shape `[1, C, T, H/scale, W/scale]`.
    ///
    /// Validates rather than rounding. A caller who asks for 100 frames or a
    /// width of 300 has made a mistake that is cheap to report here and expensive
    /// to diagnose after a 40-second model load — and silently snapping would
    /// render something other than what was asked for.
    public func videoLatentShape(width: Int, height: Int, frames: Int) throws -> [Int] {
        guard width > 0 else { throw Failure.nonPositive(dimension: "width", value: width) }
        guard height > 0 else { throw Failure.nonPositive(dimension: "height", value: height) }
        guard frames > 0 else { throw Failure.nonPositive(dimension: "frames", value: frames) }
        guard width % spatialScale == 0 else {
            throw Failure.notMultipleOfScale(dimension: "width", value: width,
                                             scale: spatialScale)
        }
        guard height % spatialScale == 0 else {
            throw Failure.notMultipleOfScale(dimension: "height", value: height,
                                             scale: spatialScale)
        }
        guard lattice.contains(frames) else {
            throw Failure.offLattice(frames: frames, nearest: lattice.snapDown(frames))
        }
        return [1, videoChannels, latentFrames(forFrames: frames),
                height / spatialScale, width / spatialScale]
    }

    /// Latent frames for a pixel frame count on the lattice: `(n−1)/scale + 1`.
    public func latentFrames(forFrames frames: Int) -> Int {
        guard frames > 1 else { return 1 }
        return (frames - 1) / temporalScale + 1
    }

    /// Audio latent shape `[1, C, T, F]`.
    ///
    /// `T = round(seconds × latentsPerSecond)`, and **`round`, not `ceil`** —
    /// contract 3. 2.3 used `ceil`, 2.5 changed it, and the difference is one
    /// latent at some frame counts. It surfaces as a shape mismatch rather than a
    /// value diff, so a port carrying 2.3's `ceil` looks correct until a frame
    /// count lands near a boundary.
    public func audioLatentShape(frames: Int, frameRate: Double) throws -> [Int] {
        guard frames > 0 else { throw Failure.nonPositive(dimension: "frames", value: frames) }
        guard frameRate > 0 else {
            throw Failure.nonPositive(dimension: "frameRate", value: Int(frameRate))
        }
        return [1, audioChannels, audioLatentCount(frames: frames, frameRate: frameRate),
                audioFeatures]
    }

    /// Audio latent count. `round`, not `ceil` (contract 3).
    public func audioLatentCount(frames: Int, frameRate: Double) -> Int {
        let seconds = Double(frames) / frameRate
        return Int((seconds * audioLatentsPerSecond).rounded())
    }

    /// Token counts per stream. **There is no packed sequence.**
    ///
    /// LTX 2.5's DiT carries video and audio as two separate streams of different
    /// widths, bridged by cross-modal attention. Measured on the tiny fixture:
    ///
    /// ```
    /// patchify_proj        (1, 240, 128) -> (1, 240, 4096)    video, inner_dim
    /// audio_patchify_proj  (1,  26, 128) -> (1,  26, 2048)    audio, audio_inner_dim
    /// block_00.attn1.in                     (1, 240, 4096)
    /// block_00.audio_attn1.in               (1,  26, 2048)
    /// block_00.audio_to_video_attn.in       (1, 240, 4096)    writes the VIDEO stream
    /// block_00.video_to_audio_attn.in       (1,  26, 2048)    writes the AUDIO stream
    /// ```
    ///
    /// Text appears in neither: it enters by cross-attention (`attn2`,
    /// `audio_attn2`), which is what `cross_attention_dim` 4096 / 2048 describes.
    ///
    /// Multimodal DiTs of this shape are not all built this way — some do
    /// concatenate text, video and audio into a single packed sequence — and a
    /// port built on that assumption here would be structurally wrong at every
    /// block, without a shape check catching it: 240 happens to be exactly the
    /// video token count, so the wrong model still produces a plausible-looking
    /// tensor.
    public func streamTokenCounts(width: Int, height: Int, frames: Int,
                                  frameRate: Double) throws -> (video: Int, audio: Int) {
        let video = try videoLatentShape(width: width, height: height, frames: frames)
        return (video: video[2] * video[3] * video[4],
                audio: audioLatentCount(frames: frames, frameRate: frameRate))
    }
}

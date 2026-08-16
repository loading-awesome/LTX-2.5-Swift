// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The conv video VAE encoder's configuration and its frame lattice, both read out of the
/// checkpoint.
///
/// Two kinds of fact live here. The first is what the checkpoint states: the patch size,
/// the `129 = 128 + 1` head split, the channel ladder whose last `compress_all_res`
/// carries a multiplier of 1, and — unlike the decoder's — a block list that is built
/// **forwards**. A port that got any of them wrong would still load, still run and still
/// produce a correctly shaped latent.
///
/// The second is the frame lattice that follows from those blocks. `97 → 13` is derived
/// two independent ways, arithmetically and by walking the temporal stages, and the two
/// must agree; then the encoder's forward map and the decoder's inverse — both read from
/// the same file — are composed and required to return where they started. Neither
/// derivation follows from the other, so a frame duplication that landed on the wrong
/// side, or was skipped, breaks one and not the rest.
@Suite("Video VAE encoder configuration")
struct VideoVAEEncoderTests {

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-video-vae-conv-bf16.safetensors"

    /// Config-only load, for the tests that need no weights.
    static func configuration() throws -> VideoVAEEncoder.Configuration? {
        guard FileManager.default.fileExists(atPath: checkpointPath) else {
            print("SKIP VideoVAEEncoder: no checkpoint at \(checkpointPath)")
            return nil
        }
        return try VideoVAEEncoder.readConfiguration(
            SafetensorsHeader.read(from: URL(fileURLWithPath: checkpointPath)))
    }

    // MARK: - Architecture, asserted rather than assumed

    /// Everything about this encoder that a copy of the decoder gets wrong.
    ///
    /// Asserted against the *checkpoint* itself rather than against any fixture, so this
    /// test still means something when the tap file is absent.
    @Test("the config is the conv VAE, uniform log-var, forward block order")
    func configurationIsWhatTheCheckpointSays() throws {
        guard let config = try Self.configuration() else { return }

        #expect(config.latentChannels == 128)
        #expect(config.inChannels == 3)
        #expect(config.patchSize == 4)
        // 48 = 3 x 4 x 4: RGB space-to-depth'd by 4, no temporal patching.
        #expect(config.patchedInputChannels == 48)
        #expect(config.logVariance == .uniform)
        // 129 = 128 means + 1 shared log-variance channel. The "+1" is the tell, and the
        // split is confirmed by the taps: `logvar` is channel 128 alone.
        #expect(config.headChannels == 129)

        // `encoder_blocks` FORWARDS, unlike the decoder's reversed ladder.
        #expect(config.downBlocks == [
            .res(layers: 4),
            .downsample(stride: (1, 2, 2), multiplier: 2),      // compress_space_res
            .res(layers: 6),
            .downsample(stride: (2, 1, 1), multiplier: 2),      // compress_time_res
            .res(layers: 4),
            .downsample(stride: (2, 2, 2), multiplier: 2),      // compress_all_res
            .res(layers: 2),
            .downsample(stride: (2, 2, 2), multiplier: 1),      // compress_all_res, x1
            .res(layers: 2),
        ])
        // The ladder is not palindromic, so "forwards" is a claim this test can fail on.
        #expect(config.downBlocks != Array(config.downBlocks.reversed()))

        let scale = config.scaleFactors
        #expect(scale.time == 8)
        #expect(scale.height == 32)          // patch 4 x three spatial halvings
        #expect(scale.width == 32)

        // The widths start at `latent_channels`, and the last `compress_all_res` carries
        // `multiplier: 1` — it folds 8 sub-voxels into the *same* width, which is the
        // entry a "multiplier defaults to 2" reading gets wrong.
        #expect(config.channelLadder
                == [128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 1024])
    }

    // MARK: - Shape algebra

    /// `97 × 384 × 640` → `[1, 128, 13, 12, 20]`, derived arithmetically and
    /// geometrically, and the two must agree.
    ///
    /// Arithmetically `(97 − 1)/8 + 1 = 13`. Geometrically, three temporal stages each
    /// duplicating frame 0 and then halving: `97 → 98 → 49 → 50 → 25 → 26 → 13`. Neither
    /// follows from the other, and a duplication that landed on the wrong side or was
    /// skipped would break one and not the other.
    @Test("97 x 384 x 640 encodes to [1, 128, 13, 12, 20], both ways")
    func frameLatticeArithmetic() throws {
        guard let config = try Self.configuration() else { return }

        #expect(config.latentFrames(pixelFrames: 97) == 13)
        #expect(config.isOnLattice(pixelFrames: 97))
        #expect(384 / config.scaleFactors.height == 12)
        #expect(640 / config.scaleFactors.width == 20)

        // A single image is one latent frame: 1 -> 2 -> 1 at each temporal stage. This
        // is the whole reason contract 15's keyframe marker exists — the first latent
        // frame covers one pixel frame where every later one covers eight.
        #expect(config.latentFrames(pixelFrames: 1) == 1)
        #expect(config.isOnLattice(pixelFrames: 1))
        #expect(!config.isOnLattice(pixelFrames: 2))

        for frames in [1, 9, 17, 25, 97, 257] {
            var geometric = frames
            for block in config.downBlocks {
                if case let .downsample(stride, _) = block, stride.0 == 2 {
                    geometric = (geometric + 1) / 2
                }
            }
            #expect(config.latentFrames(pixelFrames: frames) == geometric,
                    Comment(rawValue: "\(frames) frames: arithmetic "
                        + "\(config.latentFrames(pixelFrames: frames)) vs geometric "
                        + "\(geometric)"))
            // And it is not a plain division: the naive answer differs everywhere but 1.
            if frames > 1 {
                #expect(config.latentFrames(pixelFrames: frames) != frames / 8)
            }
        }

        // The two halves of the VAE agree about the lattice, in both directions. Both
        // ladders are read from the same checkpoint, so this composes the file's own
        // forward map with its own inverse rather than with a restatement of either.
        let decoder = try VideoVAEDecoder.readConfiguration(
            SafetensorsHeader.read(from: URL(fileURLWithPath: Self.checkpointPath)))
        for latentFrames in [1, 2, 4, 13] {
            let pixels = decoder.targetFrames(latentFrames: latentFrames)
            #expect(config.latentFrames(pixelFrames: pixels) == latentFrames,
                    Comment(rawValue: "\(latentFrames) latent -> \(pixels) pixel -> "
                        + "\(config.latentFrames(pixelFrames: pixels)) latent"))
        }
    }
}

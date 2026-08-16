// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The audio VAE encoder: log-mel → latent.
///
/// What this suite checks:
///
/// * **The configuration**, read out of the checkpoint header rather than assumed,
///   including `double_z` and `in_channels` — the two fields
///   ``AudioVAEDecoder/Configuration`` does not carry, which is why it could not be
///   reused here.
/// * **Key coverage**, which is checkable exactly: 44 encoder tensors exist and 44 are
///   consumed. The shortcuts, the downsamples and the two head shapes are named
///   individually, so a misread config fails with a countable difference instead of
///   loading a smaller network that runs.
/// * **Shape algebra**, both directions, including contract 3's `round`, and the
///   agreement between the geometric and arithmetic derivations at frame counts that are
///   *not* on the `4T-3` lattice.
/// * **Negative controls** on rank, channel count, mel bins and empty input, plus a
///   mis-derived config that must demand tensors the checkpoint does not carry.
///
/// The shape algebra is pure arithmetic and always runs. Everything else needs the audio
/// VAE checkpoint, which is not in the repository; when it is absent those tests print a
/// `SKIP` line naming the path and assert nothing.
@Suite("Audio VAE encoder")
struct AudioVAEEncoderTests {

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-audio-vae-bf16.safetensors"

    static func skip(_ what: String, _ path: String) {
        print("SKIP AudioVAEEncoder/\(what): fixture absent at \(path)")
    }

    static func encoder() throws -> AudioVAEEncoder? {
        guard FileManager.default.fileExists(atPath: checkpointPath) else { return nil }
        return try AudioVAEEncoder(checkpoint: URL(fileURLWithPath: checkpointPath))
    }

    // MARK: - Configuration, read and not assumed

    @Test("the config is read from the checkpoint, double_z and in_channels included")
    func configurationIsWhatTheCheckpointSays() throws {
        guard FileManager.default.fileExists(atPath: Self.checkpointPath) else {
            return Self.skip("configurationIsWhatTheCheckpointSays", Self.checkpointPath)
        }
        let header = try SafetensorsHeader.read(
            from: URL(fileURLWithPath: Self.checkpointPath))
        let config = try AudioVAEEncoder.readConfiguration(header)

        #expect(config.ch == 128)
        #expect(config.chMult == [1, 2, 4])
        #expect(config.numResBlocks == 2)          // 2 blocks per level, NOT 3
        #expect(config.zChannels == 8)
        #expect(config.inChannels == 2)            // the mel is stereo
        #expect(config.melBins == 64)
        #expect(config.numResolutions == 3)
        // The two fields ``AudioVAEDecoder/Configuration`` does not carry, which is why
        // it could not be reused.
        #expect(config.doubleZ)
        #expect(config.headChannels == 16)         // 2 x 8: the distribution head exists
        #expect(config.midChannels == 512)
        #expect(config.latentColumns == 16)        // 64 -> 32 -> 16

        // Reaching here at all means `readConfiguration` did not throw on group norm,
        // width causality, attention, or `resamp_with_conv: false`.
    }

    /// 44 tensors in the file, 44 consumed. The unconsumed-keys check, from both sides.
    ///
    /// The initialiser already throws on a missing or extra key; this states the number
    /// independently so that a config change which silently drops a whole level (say
    /// `num_res_blocks` misread as 1) fails here with a countable difference rather than
    /// loading a smaller network that runs.
    @Test("every encoder tensor in the checkpoint is consumed, and no other")
    func keyCoverageIsExact() throws {
        guard let encoder = try Self.encoder() else {
            return Self.skip("keyCoverageIsExact", Self.checkpointPath)
        }
        let expected = AudioVAEEncoder.expectedShapes(encoder.config)
        let present = Set(encoder.weights.keys)

        #expect(present.count == 44,
                Comment(rawValue: "checkpoint carries \(present.count) encoder tensors"))
        #expect(Set(expected.keys) == present,
                Comment(rawValue: "unconsumed: \(present.subtracting(expected.keys).sorted()); "
                    + "missing: \(Set(expected.keys).subtracting(present).sorted())"))

        // No attention, no learned norm, and exactly the two channel-changing shortcuts
        // — `down.1.block.0` and `down.2.block.0`, derived from `in_ch_mult` rather than
        // predicted from the level index.
        #expect(!present.contains { $0.contains("attn") })
        #expect(!present.contains { $0.contains("norm") })
        #expect(!present.contains { $0.contains("conv_shortcut") })
        let shortcuts = present.filter { $0.hasSuffix("nin_shortcut.conv.weight") }.sorted()
        #expect(shortcuts == ["down.1.block.0.nin_shortcut.conv.weight",
                              "down.2.block.0.nin_shortcut.conv.weight"])
        // Downsample at levels 0 and 1 only: the *last* level has none, mirroring the
        // decoder's missing `up.0.upsample`.
        #expect(encoder.weights["down.0.downsample.conv.weight"] != nil)
        #expect(encoder.weights["down.1.downsample.conv.weight"] != nil)
        #expect(encoder.weights["down.2.downsample.conv.weight"] == nil)
        // No `block.2` anywhere: `num_res_blocks` and not `num_res_blocks + 1`.
        #expect(!present.contains { $0.contains(".block.2.") })

        // The stereo claim, sourced where it actually lives: `conv_in` is the only
        // tensor that knows the mel has two channels.
        #expect(encoder.weights["conv_in.conv.weight"]!.shape == [128, 2, 3, 3])
        // And the head: 16 out, which is the only tensor that knows `double_z` is true.
        #expect(encoder.weights["conv_out.conv.weight"]!.shape == [16, 512, 3, 3])
    }

    // MARK: - Shape algebra, both directions

    /// The two derivations of the latent frame count must agree, and they must invert
    /// the decoder's.
    @Test("mel frames and latent frames invert each other, geometrically and arithmetically")
    func frameAlgebraBothDirections() {
        let config = AudioVAEEncoder.Configuration(
            ch: 128, chMult: [1, 2, 4], numResBlocks: 2, zChannels: 8, inChannels: 2,
            melBins: 64, doubleZ: true, latentDownsampleFactor: 4)
        let decoderConfig = AudioVAEDecoder.Configuration(
            ch: 128, chMult: [1, 2, 4], numResBlocks: 2, zChannels: 8, outChannels: 2,
            melBins: 64, latentDownsampleFactor: 4)

        // The two sizes that appear in practice: production, and the small fixture.
        #expect(config.latentFrames(melFrames: 401) == 101)
        #expect(config.latentFrames(melFrames: 101) == 26)
        #expect(config.melFrames(latentFrames: 101) == 401)
        #expect(config.melFrames(latentFrames: 26) == 101)

        // Encoder and decoder agree on the forward map. If they ever stop, a round trip
        // changes shape and this says which side moved.
        for latent in [1, 2, 7, 13, 26, 64, 101] {
            #expect(config.melFrames(latentFrames: latent)
                == decoderConfig.targetFrames(latentFrames: latent))
        }

        // Round trip through both directions, and the geometric derivation independently.
        for latent in [1, 2, 7, 13, 26, 64, 101] {
            let mel = config.melFrames(latentFrames: latent)
            #expect(config.latentFrames(melFrames: mel) == latent,
                    Comment(rawValue: "latent \(latent) -> mel \(mel) -> "
                        + "\(config.latentFrames(melFrames: mel))"))
            #expect(config.latentFramesArithmetic(melFrames: mel) == latent)
        }

        // And the two derivations agree on mel frame counts that are *not* on the
        // 4T-3 lattice, which is where a floor/ceil confusion would show.
        for mel in 1...200 {
            #expect(config.latentFrames(melFrames: mel)
                == config.latentFramesArithmetic(melFrames: mel),
                    Comment(rawValue: "mel \(mel): geometric "
                        + "\(config.latentFrames(melFrames: mel)) vs arithmetic "
                        + "\(config.latentFramesArithmetic(melFrames: mel))"))
        }

        // Frequency: 64 -> 32 -> 16. This is where the latent's 16 columns come from.
        #expect(config.latentColumns(melBins: 64) == 16)
        #expect(config.latentColumns(melBins: 32) == 8)
    }

    /// Contract 3: `round`, not `ceil`, and 97 frames at 24 fps is the case that tells
    /// them apart.
    ///
    /// `97 / 24 x 25 = 101.0417`. `round` gives 101 and `ceil` gives 102 — a *shape*
    /// mismatch rather than a small numeric one, which is exactly why an implementation
    /// carrying 2.3's `ceil` looks correct until a frame count lands near a boundary. The
    /// 401 that follows is the production mel's frame count, and it is reached only from
    /// the 101.
    @Test("97 frames at 24 fps needs 401 mel frames, via round and not ceil")
    func contractThreeRoundNotCeil() throws {
        guard let encoder = try Self.encoder() else {
            return Self.skip("contractThreeRoundNotCeil", Self.checkpointPath)
        }
        let geometry = LatentGeometry()

        #expect(geometry.audioLatentCount(frames: 97, frameRate: 24) == 101)
        #expect(encoder.requiredMelFrames(videoFrames: 97, frameRate: 24) == 401)
        #expect(try geometry.audioLatentShape(frames: 97, frameRate: 24)
            == [1, 8, 101, 16])
        // The whole point: `ceil` would have produced a different, equally plausible mel.
        let ceiled = Int((97.0 / 24.0 * 25.0).rounded(.up))
        #expect(ceiled == 102)
        #expect(encoder.config.melFrames(latentFrames: ceiled) == 405)

        // And the encoder's own prediction agrees with `LatentGeometry` end to end.
        #expect(try encoder.latentShape(forMel: [1, 2, 401, 64]) == [1, 8, 101, 16])
        #expect(try encoder.latentShape(forMel: [1, 2, 101, 64]) == [1, 8, 26, 16])
    }

    // MARK: - Negative controls

    @Test("a rank-3 tensor is rejected")
    func rejectsWrongRank() throws {
        guard let encoder = try Self.encoder() else {
            return Self.skip("rejectsWrongRank", Self.checkpointPath)
        }
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.encode(MLXArray.zeros([2, 101, 64], dtype: .float32))
        }
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.encode(MLXArray.zeros([1, 2, 101, 64, 1], dtype: .float32))
        }
    }

    /// A mono mel is the likeliest caller mistake and the one MLX would report as an
    /// unhelpful convolution failure deep inside `conv_in`.
    @Test("a mono mel is rejected where stereo is required")
    func rejectsMonoMel() throws {
        guard let encoder = try Self.encoder() else {
            return Self.skip("rejectsMonoMel", Self.checkpointPath)
        }
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.encode(MLXArray.zeros([1, 1, 101, 64], dtype: .float32))
        }
        // And a mel with the wrong number of bins, which is the same class of error on
        // the other axis and would otherwise produce a differently shaped latent.
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.encode(MLXArray.zeros([1, 2, 101, 80], dtype: .float32))
        }
    }

    @Test("a zero-length mel is rejected")
    func rejectsEmptyMel() throws {
        guard let encoder = try Self.encoder() else {
            return Self.skip("rejectsEmptyMel", Self.checkpointPath)
        }
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.encode(MLXArray.zeros([1, 2, 0, 64], dtype: .float32))
        }
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.encode(MLXArray.zeros([0, 2, 101, 64], dtype: .float32))
        }
        #expect(throws: AudioVAEEncoder.Failure.self) {
            _ = try encoder.latentShape(forMel: [1, 2, 0, 64])
        }
    }

    /// A checkpoint with a tensor this port would not read must fail loudly at load.
    ///
    /// Exercised through ``AudioVAEEncoder/expectedShapes(_:)`` rather than by
    /// fabricating a file: a config claiming three residual blocks per level demands
    /// `down.L.block.2.*`, which the real checkpoint does not have, so the initialiser's
    /// missing-key branch is what would fire. The symmetric case — a key the config does
    /// not predict — is covered by ``keyCoverageIsExact()`` asserting set equality.
    @Test("a mis-derived config demands tensors the checkpoint does not have")
    func configMismatchIsDetectable() throws {
        guard let encoder = try Self.encoder() else {
            return Self.skip("configMismatchIsDetectable", Self.checkpointPath)
        }
        var wrong = encoder.config
        wrong = AudioVAEEncoder.Configuration(
            ch: wrong.ch, chMult: wrong.chMult, numResBlocks: 3, zChannels: wrong.zChannels,
            inChannels: wrong.inChannels, melBins: wrong.melBins, doubleZ: wrong.doubleZ,
            latentDownsampleFactor: wrong.latentDownsampleFactor)
        let demanded = Set(AudioVAEEncoder.expectedShapes(wrong).keys)
        let present = Set(encoder.weights.keys)
        #expect(!demanded.subtracting(present).isEmpty,
                "a 3-block config must demand tensors the checkpoint does not carry")

        // And `double_z: false` would halve the head, leaving conv_out's real 16 output
        // channels mis-shaped rather than merely surplus.
        let single = AudioVAEEncoder.Configuration(
            ch: 128, chMult: [1, 2, 4], numResBlocks: 2, zChannels: 8, inChannels: 2,
            melBins: 64, doubleZ: false, latentDownsampleFactor: 4)
        #expect(AudioVAEEncoder.expectedShapes(single)["conv_out.conv.weight"]
            == [8, 512, 3, 3])
        #expect(encoder.weights["conv_out.conv.weight"]!.shape == [16, 512, 3, 3])
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The audio VAE decoder's configuration, taken from the checkpoint header.
///
/// Every number here is read out of `ltx-2.5-audio-vae-bf16.safetensors` rather than
/// assumed, because a reasonable guess gets most of them wrong. This decoder uses pixel
/// norm rather than group norm, is causal on the time axis only, and carries no attention
/// at any level; `readConfiguration` throws on each of those rather than quietly building
/// a network that runs and is wrong. The stereo output channel count lives in `conv_out`
/// alone, so nothing upstream would object to it being 1.
///
/// The checkpoint is not in the repository. When it is absent the test asserts nothing.
@Suite("Audio VAE decoder configuration")
struct AudioVAEDecoderTests {

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-audio-vae-bf16.safetensors"

    /// Everything about this decoder that a reasonable guess gets wrong.
    @Test("the config is pixel norm, height causality, and no attention")
    func configurationIsWhatTheCheckpointSays() throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.checkpointPath) else { return }
        let header = try SafetensorsHeader.read(from: URL(fileURLWithPath: Self.checkpointPath))
        let config = try AudioVAEDecoder.readConfiguration(header)

        #expect(config.ch == 128)
        #expect(config.chMult == [1, 2, 4])
        #expect(config.numResBlocks == 2)          // 3 blocks per upsampling stage
        #expect(config.zChannels == 8)
        #expect(config.outChannels == 2)           // stereo, and only conv_out says so
        #expect(config.melBins == 64)
        #expect(config.numResolutions == 3)

        // `readConfiguration` throws on group norm, width causality or any attention;
        // reaching here at all means none of those is set.
    }
}

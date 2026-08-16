// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The conv video VAE decoder's configuration, read out of the checkpoint header.
///
/// Every fact below is one the checkpoint states and a port can get wrong without
/// producing an error anywhere: which of the two video VAEs this is, whether time is
/// padded causally, what the patch size and the scale factors are, and — the expensive
/// one — which direction the block ladder runs. `decoder_blocks` is listed in *encoder*
/// order and construction mirrors it, so a port that reads the list forwards builds a
/// decoder that upsamples space where it should upsample time, at exactly the right
/// output shape.
///
/// Asserted against the checkpoint rather than against any decoded tensor, so this is a
/// statement about what the file says and not about arithmetic.
@Suite("Video VAE decoder configuration")
struct VideoVAEDecoderTests {

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-video-vae-conv-bf16.safetensors"

    /// Everything about this decoder that a reasonable guess — or a copy of the audio
    /// VAE — gets wrong.
    @Test("the config is the conv VAE, pixel norm, non-causal, no timestep conditioning")
    func configurationIsWhatTheCheckpointSays() throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: Self.checkpointPath) else { return }
        let header = try SafetensorsHeader.read(from: URL(fileURLWithPath: Self.checkpointPath))
        let config = try VideoVAEDecoder.readConfiguration(header)

        #expect(config.latentChannels == 128)
        #expect(config.outChannels == 3)
        #expect(config.patchSize == 4)
        // Non-causal: time is padded symmetrically, by replication.
        #expect(config.causal == false)
        #expect(config.baseChannels == 128)

        // `decoder_blocks` REVERSED. The config lists
        // [res_x 4, compress_space, res_x 6, compress_time, res_x 4, compress_all,
        //  res_x 2, compress_all, res_x 2]; construction order is the mirror of that.
        #expect(config.upBlocks == [
            .res(layers: 2),
            .upsample(stride: (2, 2, 2)),     // compress_all
            .res(layers: 2),
            .upsample(stride: (2, 2, 2)),     // compress_all
            .res(layers: 4),
            .upsample(stride: (2, 1, 1)),     // compress_time
            .res(layers: 6),
            .upsample(stride: (1, 2, 2)),     // compress_space
            .res(layers: 4),
        ])
        // Read forwards, the first upsample would be `compress_space` and the last
        // `compress_all`. Assert the two orders differ, so this test cannot pass by
        // the ladder being palindromic.
        #expect(config.upBlocks != Array(config.upBlocks.reversed()))

        let scale = config.scaleFactors
        #expect(scale.time == 8)
        #expect(scale.height == 32)          // 2 x 2 x 2 upsamples x patch 4
        #expect(scale.width == 32)

        // `readConfiguration` throws on the diffusion VAE, group norm, non-zero spatial
        // padding, timestep conditioning, a residual compress_all or any unimplemented
        // block; reaching here means none of those is set.
    }
}

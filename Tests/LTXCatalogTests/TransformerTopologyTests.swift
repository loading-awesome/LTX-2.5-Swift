// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXCatalog
@testable import LTXFoundation

/// Checked against the real checkpoint where one is present, and against
/// synthetic headers for the rejection paths — a wrong topology cannot be produced
/// on demand from a 39 GiB file.
@Suite("Transformer topology")
struct TransformerTopologyTests {

    static let ditPath =
        LTXConfiguration.resolved.checkpoints.root! + "/diffusion_models/"
        + "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    static var dit: SafetensorsHeader? {
        let url = URL(fileURLWithPath: ditPath)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? SafetensorsHeader.read(from: url)
    }

    // MARK: - Against the real checkpoint

    @Test("reads the measured 2.5 topology: 48 blocks, 4096 / 2048 streams")
    func realTopology() throws {
        guard let header = Self.dit else { return }
        let t = try TransformerTopology.read(header)

        #expect(t.blockCount == 48)
        #expect(t.videoWidth == 4096)
        #expect(t.audioWidth == 2048)
        #expect(t.latentChannels == 128)     // agrees with LatentGeometry.videoChannels
        #expect(t.prefix == TransformerTopology.comfyPrefix)
        #expect(t.carriesEmbeddingsConnectors)
        #expect(t.numAttentionHeads == 32)
    }

    /// The asymmetry a shared feed-forward code path gets wrong. Video has no
    /// biases; audio has them. Build both one way and one stream is wrong, with no
    /// shape error to catch it.
    @Test("feed-forward bias differs between the streams")
    func feedForwardAsymmetry() throws {
        guard let header = Self.dit else { return }
        let t = try TransformerTopology.read(header)

        #expect(t.videoFeedForwardHasBias == false)
        #expect(t.audioFeedForwardHasBias == true)
        // 4x each stream width, measured.
        #expect(t.videoFeedForwardWidth == 16384)
        #expect(t.audioFeedForwardWidth == 8192)
    }

    /// Swapping the two cross-modal bridges is the natural mistake, and it leaves every
    /// weight correctly shaped, so the wiring has to be read off the names directly.
    @Test("cross-modal attention is wired as the reference wires it")
    func crossModalWiring() throws {
        guard let header = Self.dit else { return }
        let t = try TransformerTopology.read(header)
        try t.verifyCrossModalWiring(header)      // throws on any miswiring
    }

    /// The encode stage and the DiT must agree on conditioning width, and they
    /// agree through the checkpoint rather than through a constant on either side.
    @Test("text cross-attention widths match the measured encode taps")
    func textWidths() throws {
        guard let header = Self.dit else { return }
        let t = try TransformerTopology.read(header)
        let widths = try t.textCrossAttentionWidths(header)

        // Text conditioning arrives 4096 wide for video and 2048 wide for audio.
        #expect(widths.video == 4096)
        #expect(widths.audio == 2048)
    }

    // MARK: - Rejections, on synthetic headers

    /// Build a minimal header with the given tensor shapes.
    private func header(_ shapes: [String: [Int]]) throws -> SafetensorsHeader {
        var entries: [String: Any] = [:]
        var offset = 0
        for (name, shape) in shapes.sorted(by: { $0.key < $1.key }) {
            let bytes = shape.reduce(1, *) * 2       // BF16
            entries[name] = ["dtype": "BF16", "shape": shape,
                             "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        let json = try JSONSerialization.data(withJSONObject: entries)
        return try SafetensorsHeader.parse(json: json, dataStart: 8 + json.count,
                                          dataSize: offset)
    }

    /// Names for a well-formed minimal DiT, which each rejection test then breaks.
    private func wellFormed(videoWidth v: Int = 4096, audioWidth a: Int = 2048,
                           blocks: Int = 2) -> [String: [Int]] {
        var s: [String: [Int]] = [
            "patchify_proj.weight": [v, 128],
            "audio_patchify_proj.weight": [a, 128],
        ]
        for b in 0..<blocks {
            s["transformer_blocks.\(b).ff.net.0.proj.weight"] = [v * 4, v]
            s["transformer_blocks.\(b).audio_ff.net.0.proj.weight"] = [a * 4, a]
            s["transformer_blocks.\(b).audio_ff.net.0.proj.bias"] = [a * 4]
            s["transformer_blocks.\(b).audio_to_video_attn.to_q.weight"] = [a, v]
            s["transformer_blocks.\(b).audio_to_video_attn.to_out.0.weight"] = [v, a]
            s["transformer_blocks.\(b).video_to_audio_attn.to_k.weight"] = [a, v]
            s["transformer_blocks.\(b).video_to_audio_attn.to_out.0.weight"] = [a, a]
        }
        return s
    }

    @Test("a header with no blocks is refused")
    func noBlocks() throws {
        let h = try header(["patchify_proj.weight": [4096, 128]])
        #expect(throws: TransformerTopology.Failure.self) {
            try TransformerTopology.read(h)
        }
    }

    /// Reading video and audio as a single packed sequence would give them one shared
    /// width, which the two-width structure rules out.
    @Test("equal stream widths are refused as a packed-sequence misreading")
    func packedSequenceRefused() throws {
        let h = try header(wellFormed(videoWidth: 4096, audioWidth: 4096))
        #expect(throws: TransformerTopology.Failure.self) {
            try TransformerTopology.read(h)
        }
    }

    @Test("non-contiguous block indices are refused")
    func nonContiguous() throws {
        var s = wellFormed(blocks: 1)
        // A block 7 with no blocks 1..6 -- a partial or renamed checkpoint.
        s["transformer_blocks.7.ff.net.0.proj.weight"] = [16384, 4096]
        #expect(throws: TransformerTopology.Failure.self) {
            try TransformerTopology.read(try header(s))
        }
    }

    /// Swapping the two bridges is the mistake this exists to catch, so prove the
    /// check actually fails on it rather than trusting that it would.
    @Test("swapped cross-modal bridges are caught")
    func swappedBridges() throws {
        var s = wellFormed(blocks: 1)
        // Make a2v query read the AUDIO stream instead of the video stream.
        s["transformer_blocks.0.audio_to_video_attn.to_q.weight"] = [2048, 2048]
        let h = try header(s)
        let t = try TransformerTopology.read(h)
        #expect(throws: TransformerTopology.Failure.self) {
            try t.verifyCrossModalWiring(h)
        }
    }

    @Test("a bridge writing the wrong stream is caught")
    func wrongOutputStream() throws {
        var s = wellFormed(blocks: 1)
        // a2v must write video (4096); make it write audio.
        s["transformer_blocks.0.audio_to_video_attn.to_out.0.weight"] = [2048, 2048]
        let h = try header(s)
        let t = try TransformerTopology.read(h)
        #expect(throws: TransformerTopology.Failure.self) {
            try t.verifyCrossModalWiring(h)
        }
    }

    @Test("the unprefixed naming is accepted too")
    func unprefixed() throws {
        // Checkpoints circulate both with and without the `model.diffusion_model.`
        // prefix, so neither form may be assumed.
        let t = try TransformerTopology.read(try header(wellFormed()))
        #expect(t.prefix == "")
        #expect(t.videoWidth == 4096)
        #expect(t.audioWidth == 2048)
    }
}

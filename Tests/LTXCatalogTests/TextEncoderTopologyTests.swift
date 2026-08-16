// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXCatalog
@testable import LTXFoundation

/// Checked against the real text-encoder checkpoint where present, and against
/// synthetic headers for the rejection paths.
@Suite("Text encoder topology")
struct TextEncoderTopologyTests {

    static let path =
        LTXConfiguration.resolved.checkpoints.root! + "/text_encoders/"
        + "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

    static var encoder: SafetensorsHeader? {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? SafetensorsHeader.read(from: url)
    }

    // MARK: - Against the real checkpoint

    @Test("config is read from checkpoint metadata, since no config.json ships")
    func configFromMetadata() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)

        #expect(t.hiddenSize == 3840)
        #expect(t.layerCount == 48)
        #expect(t.intermediateSize == 15360)
        #expect(t.vocabularySize == 262144)
        #expect(t.rmsNormEpsilon == 1e-6)
        #expect(t.hiddenActivation == "gelu_pytorch_tanh")
    }

    @Test("48 layers in exactly two kinds, full attention every sixth layer")
    func layerPattern() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)

        #expect(t.layerTypes.count == 48)
        let full = t.layerTypes.enumerated().filter { $0.element == .full }.map(\.offset)
        #expect(full == [5, 11, 17, 23, 29, 35, 41, 47])
        #expect(t.layerTypes.filter { $0 == .sliding }.count == 40)
    }

    /// The structural difference a port must carry: two head geometries, and one
    /// kind that does not project V at all.
    @Test("the two layer kinds have different heads, and full shares K as V")
    func attentionShapes() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)

        let sliding = try #require(t.attention[.sliding])
        #expect(sliding.headDim == 256)
        #expect(sliding.queryHeads == 16)
        #expect(sliding.keyValueHeads == 8)          // GQA 2:1
        #expect(sliding.sharesKeyAndValue == false)

        let full = try #require(t.attention[.full])
        #expect(full.headDim == 512)                 // global_head_dim, not head_dim
        #expect(full.queryHeads == 16)
        #expect(full.keyValueHeads == 1)             // one KV head
        #expect(full.sharesKeyAndValue)              // attention_k_eq_v -> no v_proj
    }

    /// RoPE differs per layer kind. One theta for all 48 layers is wrong in either
    /// 8 or 40 of them, and produces correctly-shaped output either way.
    @Test("RoPE parameters are per layer kind, including partial rotation")
    func ropePerKind() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)

        let sliding = try #require(t.rope[.sliding])
        #expect(sliding.theta == 10_000.0)
        #expect(sliding.partialRotaryFactor == 1.0)   // full rotation

        let full = try #require(t.rope[.full])
        #expect(full.theta == 1_000_000.0)
        #expect(full.partialRotaryFactor == 0.25)     // only 128 of 512 dims rotate
        #expect(full.ropeType == "proportional")

        #expect(sliding.theta != full.theta)
    }

    /// Every layer's weights are checked against the kind `layer_types` claims,
    /// which is what makes the config trustworthy instead of assumed.
    @Test("all 48 layers' weight shapes agree with their declared kind")
    func weightsAgreeWithConfig() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)
        try t.verify(h)          // throws on any disagreement, including v_proj presence
    }

    @Test("text attention is causal — bidirectionality is scoped to vision")
    func causality() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)
        #expect(t.bidirectionalAttention == "vision")
        #expect(t.textAttentionIsCausal)
    }

    /// The coincidence, made explicit so nobody relies on it silently.
    @Test("at 1024 tokens the two masks coincide, but the kinds still differ")
    func maskCoincidence() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)

        #expect(t.slidingWindow == 1024)
        // Contract 9 forces exactly 1024 tokens, so the masks are the same mask.
        #expect(t.maskingIsIdenticalAcrossKinds(sequenceLength: 1024))
        // Above the window they are not.
        #expect(!t.maskingIsIdenticalAcrossKinds(sequenceLength: 2048))
        // And the kinds differ regardless of length -- head dim and RoPE.
        #expect(t.attention[.sliding] != t.attention[.full])
        #expect(t.rope[.sliding] != t.rope[.full])
    }

    /// Contract 8, checked against the projection weights themselves.
    @Test("projection widths come from the weights and confirm 188160")
    func projection() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)
        let w = try t.projectionWidths(h)

        #expect(w.video == 4096)
        #expect(w.audio == 2048)
        #expect(w.flat == 188160)
        #expect(w.flat == t.hiddenSize * (t.layerCount + 1))   // the +1 embedding output
        #expect(t.projectionHasBias(h))                        // V2 has biases; V1 does not
    }

    /// Both extra branches are inert here, and must be refused rather than ignored
    /// if a future checkpoint enables them.
    @Test("the unimplemented branches are inert on this checkpoint")
    func inertBranches() throws {
        guard let h = Self.encoder else { return }
        let t = try TextEncoderTopology.read(h)
        #expect(t.hiddenSizePerLayerInput == 0)
        #expect(!t.mixtureOfExpertsEnabled)
        try t.rejectUnimplementedVariants()
    }

    // MARK: - Rejections

    @Test("a checkpoint with no embedded config is refused, not defaulted")
    func noConfig() throws {
        let json = try JSONSerialization.data(withJSONObject: [
            "a": ["dtype": "BF16", "shape": [1], "data_offsets": [0, 2]],
        ])
        let h = try SafetensorsHeader.parse(json: json, dataStart: 8 + json.count, dataSize: 2)
        #expect(throws: TextEncoderTopology.Failure.self) {
            try TextEncoderTopology.read(h)
        }
    }

    @Test("MoE and per-layer-input variants are refused as unimplemented")
    func unsupportedVariants() {
        let moe = TextEncoderTopology(
            hiddenSize: 3840, layerCount: 1, intermediateSize: 15360, rmsNormEpsilon: 1e-6,
            vocabularySize: 262144, slidingWindow: 1024, layerTypes: [.sliding],
            attention: [:], rope: [:], bidirectionalAttention: "vision",
            hiddenActivation: "gelu_pytorch_tanh", hiddenSizePerLayerInput: 0,
            mixtureOfExpertsEnabled: true)
        #expect(throws: TextEncoderTopology.Failure.self) {
            try moe.rejectUnimplementedVariants()
        }

        let perLayer = TextEncoderTopology(
            hiddenSize: 3840, layerCount: 1, intermediateSize: 15360, rmsNormEpsilon: 1e-6,
            vocabularySize: 262144, slidingWindow: 1024, layerTypes: [.sliding],
            attention: [:], rope: [:], bidirectionalAttention: "vision",
            hiddenActivation: "gelu_pytorch_tanh", hiddenSizePerLayerInput: 256,
            mixtureOfExpertsEnabled: false)
        #expect(throws: TextEncoderTopology.Failure.self) {
            try perLayer.rejectUnimplementedVariants()
        }
    }

    /// A `v_proj` on a K-shares-V layer means the config flag does not describe the
    /// weights. Both directions of that mismatch are errors.
    @Test("an unexpected v_proj on a sharing layer is caught")
    func unexpectedVProj() throws {
        var entries: [String: Any] = [:]
        var offset = 0
        func add(_ name: String, _ shape: [Int]) {
            let bytes = shape.reduce(1, *) * 2
            entries[name] = ["dtype": "BF16", "shape": shape,
                             "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        let h = 64, hd = 128, qh = 4
        add("model.layers.0.self_attn.q_proj.weight", [qh * hd, h])
        add("model.layers.0.self_attn.k_proj.weight", [hd, h])
        add("model.layers.0.self_attn.o_proj.weight", [h, qh * hd])
        add("model.layers.0.self_attn.q_norm.weight", [hd])
        add("model.layers.0.self_attn.k_norm.weight", [hd])
        add("model.layers.0.self_attn.v_proj.weight", [hd, h])   // <- should not exist
        for n in ["input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            add("model.layers.0.\(n).weight", [h])
        }
        add("model.layers.0.layer_scalar", [1])
        add("model.layers.0.mlp.gate_proj.weight", [256, h])
        add("model.layers.0.mlp.up_proj.weight", [256, h])
        add("model.layers.0.mlp.down_proj.weight", [h, 256])

        let json = try JSONSerialization.data(withJSONObject: entries)
        let header = try SafetensorsHeader.parse(json: json, dataStart: 8 + json.count,
                                                dataSize: offset)
        let t = TextEncoderTopology(
            hiddenSize: h, layerCount: 1, intermediateSize: 256, rmsNormEpsilon: 1e-6,
            vocabularySize: 100, slidingWindow: 1024, layerTypes: [.full],
            attention: [.full: .init(headDim: hd, queryHeads: qh, keyValueHeads: 1,
                                     sharesKeyAndValue: true)],
            rope: [:], bidirectionalAttention: "vision",
            hiddenActivation: "gelu_pytorch_tanh", hiddenSizePerLayerInput: 0,
            mixtureOfExpertsEnabled: false)

        #expect(throws: TextEncoderTopology.Failure.self) {
            try t.verify(header)
        }
    }

    /// The inverse: a non-sharing layer whose v_proj is missing must not be silently
    /// filled from K.
    @Test("a missing v_proj on a non-sharing layer is caught")
    func missingVProj() throws {
        var entries: [String: Any] = [:]
        var offset = 0
        func add(_ name: String, _ shape: [Int]) {
            let bytes = shape.reduce(1, *) * 2
            entries[name] = ["dtype": "BF16", "shape": shape,
                             "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        let h = 64, hd = 128, qh = 4
        add("model.layers.0.self_attn.q_proj.weight", [qh * hd, h])
        add("model.layers.0.self_attn.k_proj.weight", [2 * hd, h])
        add("model.layers.0.self_attn.o_proj.weight", [h, qh * hd])
        add("model.layers.0.self_attn.q_norm.weight", [hd])
        add("model.layers.0.self_attn.k_norm.weight", [hd])
        // no v_proj, and this layer does not share
        for n in ["input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            add("model.layers.0.\(n).weight", [h])
        }
        add("model.layers.0.layer_scalar", [1])
        add("model.layers.0.mlp.gate_proj.weight", [256, h])
        add("model.layers.0.mlp.up_proj.weight", [256, h])
        add("model.layers.0.mlp.down_proj.weight", [h, 256])

        let json = try JSONSerialization.data(withJSONObject: entries)
        let header = try SafetensorsHeader.parse(json: json, dataStart: 8 + json.count,
                                                dataSize: offset)
        let t = TextEncoderTopology(
            hiddenSize: h, layerCount: 1, intermediateSize: 256, rmsNormEpsilon: 1e-6,
            vocabularySize: 100, slidingWindow: 1024, layerTypes: [.sliding],
            attention: [.sliding: .init(headDim: hd, queryHeads: qh, keyValueHeads: 2,
                                        sharesKeyAndValue: false)],
            rope: [:], bidirectionalAttention: "vision",
            hiddenActivation: "gelu_pytorch_tanh", hiddenSizePerLayerInput: 0,
            mixtureOfExpertsEnabled: false)

        #expect(throws: TextEncoderTopology.Failure.self) {
            try t.verify(header)
        }
    }
}

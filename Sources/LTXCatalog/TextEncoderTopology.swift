// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// The Gemma text encoder's structure, read from the checkpoint and its embedded
/// config.
///
/// ## Config comes from the checkpoint, and here that is not a preference
///
/// The LTX text-encoder file ships **no `config.json`**. The full HuggingFace
/// config is JSON-encoded in the safetensors metadata under `gemma_config`, so
/// reading config from the checkpoint rather than from a constant is not a
/// preference here — it is the only way this model can be configured at all.
///
/// The tokenizer is in there too, as a 32 MB `tokenizer_json` tensor, along with the
/// chat template and generation config. Nothing external is required.
///
/// ## This model has two different layer variants
///
/// Measured on `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (686 tensors):
/// 48 layers in exactly two shapes, selected by `layer_types`.
///
/// ```
///                    sliding (40)              full / "global" (8)
/// q_proj             [4096, 3840]  16x256      [8192, 3840]  16x512
/// k_proj             [2048, 3840]   8x256      [ 512, 3840]   1x512
/// v_proj             [2048, 3840]               ABSENT
/// o_proj             [3840, 4096]              [3840, 8192]
/// q_norm / k_norm    [256]                     [512]
/// rope               theta 1e4, full           theta 1e6, partial 0.25
/// ```
///
/// The global layers have no `v_proj` because `attention_k_eq_v` is true for them:
/// V *is* K. A port that builds 48 identical layers fails loudly on the shapes, so
/// that much is self-correcting — but applying one RoPE configuration to all 48, or
/// silently substituting K for V without knowing why, is arithmetic that produces
/// correctly-shaped output.
///
/// ## One coincidence not to lean on
///
/// `sliding_window` is 1024 and contract 9 forces the tokenizer to exactly 1024
/// tokens, so at this sequence length the sliding mask and the full mask are the
/// same mask. That makes the *masking* difference a no-op today. It does not make
/// the layer types a no-op — the head geometry and RoPE differ regardless — and if
/// `min_length` ever rose above 1024, 40 of 48 layers would change behaviour with
/// nothing to announce it.
public struct TextEncoderTopology: Sendable, Equatable {

    public enum LayerKind: String, Sendable, Equatable {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    public struct RoPEParameters: Sendable, Equatable {
        public let theta: Double
        /// Fraction of `headDim` that is rotated. `1.0` means all of it.
        public let partialRotaryFactor: Double
        public let ropeType: String
    }

    public struct AttentionShape: Sendable, Equatable {
        public let headDim: Int
        public let queryHeads: Int
        public let keyValueHeads: Int
        /// True when V is not projected separately — `attention_k_eq_v`.
        public let sharesKeyAndValue: Bool
    }

    public let hiddenSize: Int
    public let layerCount: Int
    public let intermediateSize: Int
    public let rmsNormEpsilon: Double
    public let vocabularySize: Int
    public let slidingWindow: Int
    /// Per-layer kind, `layerTypes.count == layerCount`.
    public let layerTypes: [LayerKind]
    public let attention: [LayerKind: AttentionShape]
    public let rope: [LayerKind: RoPEParameters]
    /// `use_bidirectional_attention`. `"vision"` means **text attention is causal**.
    public let bidirectionalAttention: String?
    public let hiddenActivation: String
    /// Non-zero would add a per-layer input branch this port does not implement.
    public let hiddenSizePerLayerInput: Int
    public let mixtureOfExpertsEnabled: Bool

    public enum Failure: Error, CustomStringConvertible {
        case noEmbeddedConfig
        case missingConfigField(String)
        case layerTypeCountMismatch(types: Int, layers: Int)
        case unknownLayerType(String)
        case shapeDisagreesWithConfig(layer: Int, kind: LayerKind, tensor: String,
                                     shape: [Int], expected: [Int])
        case unexpectedTensor(layer: Int, kind: LayerKind, name: String, why: String)
        case missingTensor(String)
        case unsupportedVariant(String)

        public var description: String {
            switch self {
            case .noEmbeddedConfig:
                return "the checkpoint carries no `gemma_config` metadata. This file ships "
                    + "no config.json either, so there is nothing to configure the model "
                    + "from -- do not substitute defaults, they would be a guess about a "
                    + "48-layer model with two layer variants"
            case let .missingConfigField(name):
                return "embedded gemma_config has no \(name)"
            case let .layerTypeCountMismatch(types, layers):
                return "layer_types lists \(types) entries but the checkpoint has \(layers) "
                    + "layers; the per-layer attention pattern cannot be applied"
            case let .unknownLayerType(name):
                return "unknown layer type \(name). Only sliding_attention and "
                    + "full_attention are implemented, and each has its own head "
                    + "geometry and RoPE -- a third kind is not a variation to guess at"
            case let .shapeDisagreesWithConfig(layer, kind, tensor, shape, expected):
                return "layer \(layer) (\(kind.rawValue)) \(tensor) is \(shape) but the "
                    + "config implies \(expected). Either layer_types is wrong for this "
                    + "layer or the head geometry is not what the config says"
            case let .unexpectedTensor(layer, kind, name, why):
                return "layer \(layer) (\(kind.rawValue)) has \(name), which it should not: \(why)"
            case let .missingTensor(name):
                return "required tensor \(name) is absent"
            case let .unsupportedVariant(why):
                return "this checkpoint needs a code path that does not exist yet: \(why)"
            }
        }
    }

    // MARK: - Reading

    public static func read(_ header: SafetensorsHeader) throws -> TextEncoderTopology {
        guard let config = header.metadataJSON("gemma_config") else {
            throw Failure.noEmbeddedConfig
        }
        // The text config is nested for the unified (text + vision + audio) model.
        let text = (config["text_config"] as? [String: Any]) ?? config

        func int(_ key: String) throws -> Int {
            guard let v = text[key] as? Int else { throw Failure.missingConfigField(key) }
            return v
        }
        func double(_ key: String) throws -> Double {
            if let v = text[key] as? Double { return v }
            if let v = text[key] as? Int { return Double(v) }
            throw Failure.missingConfigField(key)
        }

        let hiddenSize = try int("hidden_size")
        let layerCount = try int("num_hidden_layers")
        let headDim = try int("head_dim")
        let queryHeads = try int("num_attention_heads")
        let kvHeads = try int("num_key_value_heads")

        guard let rawTypes = text["layer_types"] as? [String] else {
            throw Failure.missingConfigField("layer_types")
        }
        guard rawTypes.count == layerCount else {
            throw Failure.layerTypeCountMismatch(types: rawTypes.count, layers: layerCount)
        }
        let types = try rawTypes.map { raw -> LayerKind in
            guard let k = LayerKind(rawValue: raw) else { throw Failure.unknownLayerType(raw) }
            return k
        }

        // Global layers carry their own head dim and KV head count, and share K as V.
        let globalHeadDim = (text["global_head_dim"] as? Int) ?? headDim
        let globalKVHeads = (text["num_global_key_value_heads"] as? Int) ?? kvHeads
        let kEqualsV = (text["attention_k_eq_v"] as? Bool) ?? false

        var attention: [LayerKind: AttentionShape] = [
            .sliding: AttentionShape(headDim: headDim, queryHeads: queryHeads,
                                     keyValueHeads: kvHeads, sharesKeyAndValue: false),
        ]
        attention[.full] = AttentionShape(headDim: globalHeadDim, queryHeads: queryHeads,
                                         keyValueHeads: globalKVHeads,
                                         sharesKeyAndValue: kEqualsV)

        // RoPE is per layer *type*, which is the detail most likely to be dropped:
        // one theta applied to all 48 layers is wrong in either 8 or 40 of them.
        var rope: [LayerKind: RoPEParameters] = [:]
        if let params = text["rope_parameters"] as? [String: Any] {
            for kind in [LayerKind.sliding, .full] {
                guard let entry = params[kind.rawValue] as? [String: Any] else { continue }
                let theta = (entry["rope_theta"] as? Double)
                    ?? Double((entry["rope_theta"] as? Int) ?? 10_000)
                let partial = (entry["partial_rotary_factor"] as? Double) ?? 1.0
                rope[kind] = RoPEParameters(theta: theta, partialRotaryFactor: partial,
                                            ropeType: (entry["rope_type"] as? String) ?? "default")
            }
        }

        let topology = TextEncoderTopology(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: try int("intermediate_size"),
            rmsNormEpsilon: try double("rms_norm_eps"),
            vocabularySize: try int("vocab_size"),
            slidingWindow: (text["sliding_window"] as? Int) ?? 0,
            layerTypes: types,
            attention: attention,
            rope: rope,
            bidirectionalAttention: text["use_bidirectional_attention"] as? String,
            hiddenActivation: (text["hidden_activation"] as? String) ?? "gelu_pytorch_tanh",
            hiddenSizePerLayerInput: (text["hidden_size_per_layer_input"] as? Int) ?? 0,
            mixtureOfExpertsEnabled: (text["enable_moe_block"] as? Bool) ?? false)

        try topology.rejectUnimplementedVariants()
        return topology
    }

    /// Refuse configurations whose extra branches this port does not implement.
    ///
    /// Both branches exist in `Gemma4DecoderLayer.forward` and both are inert on
    /// this checkpoint (`hidden_size_per_layer_input` 0, `enable_moe_block` false).
    /// A checkpoint that enabled either would run through a port that ignores them
    /// and produce plausible, wrong hidden states — so it is rejected rather than
    /// silently approximated.
    public func rejectUnimplementedVariants() throws {
        if hiddenSizePerLayerInput != 0 {
            throw Failure.unsupportedVariant(
                "hidden_size_per_layer_input is \(hiddenSizePerLayerInput), which adds a "
                + "per_layer_input_gate / per_layer_projection branch before layer_scalar")
        }
        if mixtureOfExpertsEnabled {
            throw Failure.unsupportedVariant(
                "enable_moe_block is true, which adds a router, experts and two more "
                + "feed-forward norms per layer")
        }
    }

    /// True when text attention is causal.
    ///
    /// `use_bidirectional_attention: "vision"` scopes bidirectionality to vision
    /// tokens, leaving text causal. A port that makes the text encoder bidirectional
    /// because "it is an encoder" changes every hidden state.
    public var textAttentionIsCausal: Bool {
        bidirectionalAttention != "text" && bidirectionalAttention != "all"
    }

    /// Whether the sliding mask and the full mask coincide at this length.
    ///
    /// True when `sequenceLength <= slidingWindow`. It says nothing about head
    /// geometry or RoPE, which differ between the layer kinds at every length.
    public func maskingIsIdenticalAcrossKinds(sequenceLength: Int) -> Bool {
        slidingWindow > 0 && sequenceLength <= slidingWindow
    }

    // MARK: - Verification against the weights

    /// Check every layer's weight shapes against the kind `layer_types` assigns it.
    ///
    /// This is what makes `layer_types` trustworthy rather than assumed: the config
    /// says layer 5 is a full-attention layer, and its `q_proj` is `[8192, 3840]`
    /// only if that is true. Disagreement means one of the two is wrong, and picking
    /// which by preference is how a port acquires an unexplainable bug.
    public func verify(_ header: SafetensorsHeader, prefix: String = "model.layers.") throws {
        for (index, kind) in layerTypes.enumerated() {
            guard let shape = attention[kind] else { continue }
            let stem = "\(prefix)\(index)."
            let qWidth = shape.queryHeads * shape.headDim
            let kvWidth = shape.keyValueHeads * shape.headDim

            func expect(_ name: String, _ expected: [Int]) throws {
                guard let e = header.tensors[stem + name] else {
                    throw Failure.missingTensor(stem + name)
                }
                guard e.shape == expected else {
                    throw Failure.shapeDisagreesWithConfig(
                        layer: index, kind: kind, tensor: name,
                        shape: e.shape, expected: expected)
                }
            }

            try expect("self_attn.q_proj.weight", [qWidth, hiddenSize])
            try expect("self_attn.k_proj.weight", [kvWidth, hiddenSize])
            try expect("self_attn.o_proj.weight", [hiddenSize, qWidth])
            try expect("self_attn.q_norm.weight", [shape.headDim])
            try expect("self_attn.k_norm.weight", [shape.headDim])

            // v_proj is present exactly when K and V are not shared. Both directions
            // are errors: a missing v_proj on a non-sharing layer would be silently
            // filled from K, and a present one on a sharing layer means the config
            // flag is lying about what the weights do.
            let vName = stem + "self_attn.v_proj.weight"
            let hasV = header.tensors[vName] != nil
            if shape.sharesKeyAndValue && hasV {
                throw Failure.unexpectedTensor(
                    layer: index, kind: kind, name: "self_attn.v_proj.weight",
                    why: "attention_k_eq_v is set, so V is K and no separate V is projected")
            }
            if !shape.sharesKeyAndValue && !hasV {
                throw Failure.missingTensor(vName)
            }
            if hasV { try expect("self_attn.v_proj.weight", [kvWidth, hiddenSize]) }

            // The four sandwich norms. Gemma normalises the attention and
            // feed-forward *outputs* before each residual add, on top of the usual
            // pre-norms -- a two-norm LLaMA-style layer silently drops half of them.
            for norm in ["input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                try expect("\(norm).weight", [hiddenSize])
            }

            // The per-layer output scale. One element, easy to miss in a name sweep,
            // and it multiplies the layer's whole output including the residual
            // stream -- measured range 0.0045 to 0.926 on this checkpoint, so
            // treating it as 1.0 misscales a layer by up to 220x.
            try expect("layer_scalar", [1])

            try expect("mlp.gate_proj.weight", [intermediateSize, hiddenSize])
            try expect("mlp.up_proj.weight", [intermediateSize, hiddenSize])
            try expect("mlp.down_proj.weight", [hiddenSize, intermediateSize])
        }
    }

    /// The projection that turns the layer stack into conditioning.
    ///
    /// Lives in this file on the 22B (contract 8): `text_embedding_projection.*`.
    /// Returns the widths so `TextFeatureProjection` can be configured from the
    /// checkpoint rather than from constants.
    public func projectionWidths(_ header: SafetensorsHeader) throws
        -> (video: Int, audio: Int, flat: Int) {
        let base = "text_embedding_projection."
        guard let v = header.tensors[base + "video_aggregate_embed.weight"] else {
            throw Failure.missingTensor(base + "video_aggregate_embed.weight")
        }
        guard let a = header.tensors[base + "audio_aggregate_embed.weight"] else {
            throw Failure.missingTensor(base + "audio_aggregate_embed.weight")
        }
        // Contract 8: the flat width is hidden_size * (num_hidden_layers + 1) -- the
        // +1 being the embedding output. Checked against the weights rather than
        // asserted, because this is the exact constant the 2.5 PR removed.
        let expectedFlat = hiddenSize * (layerCount + 1)
        guard v.shape.count == 2, v.shape[1] == expectedFlat else {
            throw Failure.shapeDisagreesWithConfig(
                layer: -1, kind: .full, tensor: "video_aggregate_embed.weight",
                shape: v.shape, expected: [v.shape.first ?? 0, expectedFlat])
        }
        guard a.shape.count == 2, a.shape[1] == expectedFlat else {
            throw Failure.shapeDisagreesWithConfig(
                layer: -1, kind: .full, tensor: "audio_aggregate_embed.weight",
                shape: a.shape, expected: [a.shape.first ?? 0, expectedFlat])
        }
        return (video: v.shape[0], audio: a.shape[0], flat: expectedFlat)
    }

    /// Both projections carry biases on this variant (V1's single one does not).
    public func projectionHasBias(_ header: SafetensorsHeader) -> Bool {
        header.tensors["text_embedding_projection.video_aggregate_embed.bias"] != nil
            && header.tensors["text_embedding_projection.audio_aggregate_embed.bias"] != nil
    }
}

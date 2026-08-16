// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich
//
// swiftlint:disable file_length

import Foundation
import LTXCatalog
import LTXFoundation
import MLX
import MLXFast

/// The Gemma-4 text encoder, in MLX.
///
/// Produces the 49 hidden states the conditioning projection consumes (contract 8).
///
/// Every non-obvious value here is pinned by `LTXFoundation.GemmaAttentionSpec` and
/// contract 13, and the structure by `LTXCatalog.TextEncoderTopology`, which verifies
/// all 48 layers' weight shapes against the config before this type runs.
public final class GemmaTextEncoder {

    /// Which attention implementation to use.
    ///
    /// Not a tuning knob — a diagnostic seam, and one that has already earned its
    /// keep. A fused kernel and its written-out equivalent should agree; here they do
    /// not, and being able to switch between them is what pins the difference on the
    /// kernel rather than on the layer.
    ///
    /// The difference is where the softmax accumulates. `explicit` computes logits and
    /// softmax in fp32; MLX's fused kernel evidently does not, and over a 1024-key
    /// reduction a bf16 softmax loses enough of the tail to move a layer's output. So
    /// "fusion is a performance choice, not a numerical one" (`FRAGILE_CONTRACTS.md`)
    /// does not hold for this kernel, and the claim has to be checked per backend
    /// rather than carried.
    public enum AttentionPath: Sendable {
        case fused
        case explicit
    }

    /// Defaults to `.explicit` **because it is the wider accumulation**, not because
    /// it is simpler. Revisit if MLX's fused kernel changes: it would be the faster
    /// path if it agreed.
    public var attentionPath: AttentionPath = .explicit

    public let topology: TextEncoderTopology
    private let weights: [String: MLXArray]
    private let prefix: String
    /// `sqrt(hidden_size)` — contract 13. Applied to the embedding lookup.
    private let embedScale: Float
    /// Per layer kind, `[headDim/2]`, fp32.
    private let inverseFrequencies: [TextEncoderTopology.LayerKind: MLXArray]

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case badTokenCount(Int)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .badTokenCount(n): return "token count must be positive, got \(n)"
            }
        }
    }

    public convenience init(checkpoint url: URL, topology: TextEncoderTopology,
                            prefix: String = "model.") throws {
        self.init(weights: try MLX.loadArrays(url: url), topology: topology, prefix: prefix)
    }

    /// Share an already-loaded encoder map — `TextConditioningPipeline` also needs the
    /// projection tensors from this file, and two `MLX.loadArrays` of 26 GB is two maps.
    public init(weights: [String: MLXArray], topology: TextEncoderTopology,
                prefix: String = "model.") {
        self.topology = topology
        self.prefix = prefix
        self.embedScale = Float(Double(topology.hiddenSize).squareRoot())
        self.weights = weights

        // RoPE frequencies, once, in fp32: the position grid must not be narrowed.
        let spec = GemmaAttentionSpec.ltx25
        var freqs: [TextEncoderTopology.LayerKind: MLXArray] = [:]
        for (kind, specKind) in [(TextEncoderTopology.LayerKind.sliding,
                                 GemmaAttentionSpec.LayerKind.sliding),
                                (.full, .full)] {
            let inv = spec.inverseFrequencies(for: specKind).map { Float($0) }
            freqs[kind] = MLXArray(inv)
        }
        self.inverseFrequencies = freqs
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[prefix + name] else { throw Failure.missing(prefix + name) }
        return w
    }

    private func optional(_ name: String) -> MLXArray? { weights[prefix + name] }

    // MARK: - Primitives

    /// Gemma-4's RMSNorm: `normed * weight`, **no `1 +`**.
    ///
    /// Gemma-2 and Gemma-3 use `(1 + weight)`, so this is the detail that gets
    /// inherited wrongly from the obvious place — under that reading every norm in the
    /// model is roughly doubled. The checkpoint's weights centre on 1.0, which settles
    /// it; see `GemmaAttentionSpec.rmsNormAddsOneToWeight`.
    ///
    /// Computed in **fp32 and cast back**, and with `pow(v, -0.5)` rather than
    /// `rsqrt` — the two differ in the last bit and Gemma's norm is specified with the
    /// `pow` spelling. See `GemmaAttentionSpec.rmsNormUsesPowNotRsqrt`.
    private func rmsNorm(_ x: MLXArray, weight w: MLXArray?) -> MLXArray {
        let f = x.asType(.float32)
        let meanSquares = MLX.mean(f * f, axis: -1, keepDims: true)
        var normed = f * MLX.pow(meanSquares + Float(topology.rmsNormEpsilon), -0.5)
        if let w { normed = normed * w.asType(.float32) }
        return normed.asType(x.dtype)
    }

    /// Rotate the second half into the first, Gemma/LLaMA style.
    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let a = x[.ellipsis, 0 ..< half]
        let b = x[.ellipsis, half ..< (2 * half)]
        return MLX.concatenated([-b, a], axis: -1)
    }

    /// `(x * cos) + (rotate_half(x) * sin)`.
    ///
    /// The non-rotated tail of a partial-rotary layer needs no special case: its
    /// inverse frequencies are **zero**, so `cos = 1` and `sin = 0` and the rotation
    /// is the identity there. Partial rotation is expressed as zero-padded frequencies
    /// rather than a tensor split, so the tail cannot drift from a slicing mistake.
    private func applyRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        (x * cos) + (rotateHalf(x) * sin)
    }

    /// `cos` / `sin` for one layer kind, shaped `[1, tokens, 1, headDim]`.
    private func positionEmbeddings(kind: TextEncoderTopology.LayerKind,
                                   tokens: Int) -> (cos: MLXArray, sin: MLXArray) {
        let inv = inverseFrequencies[kind]!                       // [headDim/2]
        let positions = MLXArray(0..<tokens).asType(.float32).reshaped([tokens, 1])
        let angles = positions * inv.reshaped([1, inv.dim(0)])    // [tokens, headDim/2]
        // Duplicated, not interleaved, to match `rotate_half`'s pairing.
        let full = MLX.concatenated([angles, angles], axis: -1)   // [tokens, headDim]
        return (MLX.cos(full).reshaped([1, tokens, 1, full.dim(-1)]),
                MLX.sin(full).reshaped([1, tokens, 1, full.dim(-1)]))
    }

    /// The tanh approximation to GELU, not the erf form.
    private func geluTanh(_ x: MLXArray) -> MLXArray {
        let c = Float(0.7978845608028654)      // sqrt(2/pi)
        let inner = c * (x + Float(0.044715) * x * x * x)
        return Float(0.5) * x * (1.0 + MLX.tanh(inner))
    }

    // MARK: - One layer

    private func attention(_ x: MLXArray, layer: Int,
                          kind: TextEncoderTopology.LayerKind,
                          cos: MLXArray, sin: MLXArray,
                          mask: MLXArray) throws -> MLXArray {
        let shape = topology.attention[kind]!
        let stem = "layers.\(layer).self_attn."
        let tokens = x.dim(1)
        let headDim = shape.headDim

        func project(_ name: String, heads: Int) throws -> MLXArray {
            let w = try weight(stem + name + ".weight")           // [out, in]
            return MLX.matmul(x, w.transposed(1, 0)).reshaped([1, tokens, heads, headDim])
        }

        // Q: project -> q_norm -> RoPE.
        var q = try project("q_proj", heads: shape.queryHeads)
        q = rmsNorm(q, weight: try weight(stem + "q_norm.weight"))
        q = applyRoPE(q, cos: cos, sin: sin)

        // K, and V from the SAME projection when they are shared -- taken BEFORE
        // k_norm. V binds to the raw projection, and K then goes on through its own
        // norm and rotation:
        //
        //     k = k_proj(h); v = v_proj(h) if there is one, else that same k
        //     k = k_norm(k); k = rope(k)
        //     v = v_norm(v)
        //
        // Taking V after computing K would hand it the k_norm scale (~0.06) and the
        // rotation. Neither is visible in the shape.
        let kProjected = try project("k_proj", heads: shape.keyValueHeads)
        var v: MLXArray
        if shape.sharesKeyAndValue {
            v = kProjected
        } else {
            v = try project("v_proj", heads: shape.keyValueHeads)
        }
        var k = rmsNorm(kProjected, weight: try weight(stem + "k_norm.weight"))
        k = applyRoPE(k, cos: cos, sin: sin)

        // V is normalised by a WEIGHTLESS norm and never rotated. `v_norm` is a
        // scale-free RMS norm over `head_dim`, so it allocates no parameter and has
        // no tensor in the checkpoint at all -- code assembled by enumerating
        // checkpoint tensors cannot discover this operation.
        v = rmsNorm(v, weight: nil)

        // [1, heads, tokens, headDim] for attention.
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // GQA: repeat the KV heads to match the query heads.
        let repeats = shape.queryHeads / shape.keyValueHeads
        if repeats > 1 {
            k = MLX.repeated(k, count: repeats, axis: 1)
            v = MLX.repeated(v, count: repeats, axis: 1)
        }

        // **Scale 1.0, not 1/sqrt(headDim).** Gemma-4's attention scaling is 1.0; the
        // softmax scale is folded into the learned `k_norm` weights, which is why they
        // measure ~0.06 where `q_norm` measures ~1.03. Applying the conventional scale
        // saturates the softmax.
        let out: MLXArray
        switch attentionPath {
        case .fused:
            out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: GemmaAttentionSpec.attentionScaling,
                mask: .array(mask))
        case .explicit:
            // The same arithmetic written out, for isolating the fused kernel. The
            // softmax runs in fp32: a bf16 softmax over 1024 keys loses enough of the
            // tail to matter.
            var logits = MLX.matmul(q.asType(.float32), k.asType(.float32)
                .transposed(0, 1, 3, 2)) * GemmaAttentionSpec.attentionScaling
            logits = logits + mask.asType(.float32)
            let weights = MLX.softmax(logits, axis: -1)
            out = MLX.matmul(weights, v.asType(.float32)).asType(q.dtype)
        }

        let merged = out.transposed(0, 2, 1, 3)
            .reshaped([1, tokens, shape.queryHeads * headDim])
        let o = try weight(stem + "o_proj.weight")
        return MLX.matmul(merged, o.transposed(1, 0))
    }

    private func feedForward(_ x: MLXArray, layer: Int) throws -> MLXArray {
        let stem = "layers.\(layer).mlp."
        let gate = MLX.matmul(x, try weight(stem + "gate_proj.weight").transposed(1, 0))
        let up = MLX.matmul(x, try weight(stem + "up_proj.weight").transposed(1, 0))
        let activated = geluTanh(gate) * up
        return MLX.matmul(activated, try weight(stem + "down_proj.weight").transposed(1, 0))
    }

    /// One decoder layer.
    ///
    /// Gemma's **sandwich** norms: the attention and feed-forward *outputs* are
    /// normalised before each residual add, on top of the usual pre-norms. A
    /// LLaMA-style two-norm layer silently drops half of them.
    ///
    /// `layer_scalar` multiplies the whole layer output **after both residual adds**,
    /// so it rescales the residual stream and not just this block's contribution.
    /// It is a buffer rather than a parameter, which is exactly how it comes to be
    /// overlooked; its range on this checkpoint is 0.0045 to 0.926, so substituting
    /// 1.0 overscales the worst layer by 220x.
    private func decoderLayer(_ x: MLXArray, layer: Int,
                             kind: TextEncoderTopology.LayerKind,
                             cos: MLXArray, sin: MLXArray,
                             mask: MLXArray) throws -> MLXArray {
        let stem = "layers.\(layer)."

        var h = rmsNorm(x, weight: try weight(stem + "input_layernorm.weight"))
        h = try attention(h, layer: layer, kind: kind, cos: cos, sin: sin, mask: mask)
        h = rmsNorm(h, weight: try weight(stem + "post_attention_layernorm.weight"))
        var residual = x + h

        h = rmsNorm(residual, weight: try weight(stem + "pre_feedforward_layernorm.weight"))
        h = try feedForward(h, layer: layer)
        h = rmsNorm(h, weight: try weight(stem + "post_feedforward_layernorm.weight"))
        residual = residual + h

        let scalar = try weight(stem + "layer_scalar")
        return residual * scalar.asType(residual.dtype)
    }

    // MARK: - The forward

    /// Attention mask for one layer kind.
    ///
    /// Text attention is **causal** — `use_bidirectional_attention: "vision"` scopes
    /// bidirectionality to vision tokens. Padding is masked too, and the pad value is
    /// `-finfo.max` rather than `-infinity`: `-inf` times a zero attention weight is
    /// `NaN`, while `-max` is finite and merely saturates.
    ///
    /// For a sliding layer, positions further apart than `slidingWindow` are also
    /// masked. At exactly 1024 tokens with a 1024 window the two masks coincide, which
    /// is a coincidence of this tokenizer's `min_length` and not a property to rely on.
    private func attentionMask(kind: TextEncoderTopology.LayerKind, tokens: Int,
                              valid: MLXArray) -> MLXArray {
        let big = -Float.greatestFiniteMagnitude
        let rows = MLXArray(0..<tokens).reshaped([tokens, 1])
        let cols = MLXArray(0..<tokens).reshaped([1, tokens])

        var allowed = cols .<= rows                                  // causal
        if kind == .sliding && topology.slidingWindow > 0 {
            allowed = allowed .&& (rows - cols .< MLXArray(Int32(topology.slidingWindow)))
        }
        // Padding: a key that is padding is never attended to.
        let keyValid = valid.reshaped([1, tokens]) .!= 0
        allowed = allowed .&& keyValid

        // **Fully-masked rows must be unmasked**, or softmax divides zero by zero.
        //
        // With left padding, query 0 is a pad token: causality allows it only key 0,
        // and key 0 is padding, so every position in its row is masked. Softmax over
        // an all-`-max` row is 0/0 = NaN, and one NaN propagates through 48 layers
        // into every element. Without this, layer 0's output is entirely NaN while
        // the embedding it was computed from is perfectly finite.
        //
        // Letting such a row attend uniformly produces finite garbage at that
        // position, and that is harmless: a padded position can never influence a
        // valid one, because pad keys are masked for every query. So the values there
        // are don't-care by construction, not by hope.
        let anyAllowed = MLX.any(allowed, axes: [-1], keepDims: true)
        allowed = allowed .|| MLX.broadcast(anyAllowed .== false, to: allowed.shape)

        let zero = MLXArray.zeros([tokens, tokens], dtype: .float32)
        return MLX.where(allowed, zero, MLXArray(big)).reshaped([1, 1, tokens, tokens])
    }

    /// Run the encoder and return the **49** hidden states the projection consumes.
    ///
    /// The last element is `final_norm(layer 47's output)`, **not** the raw output —
    /// layer 47's unnormalised output never appears in the list. See contract 13;
    /// returning `(embeds, out0...out47)` instead leaves a tensor whose RMS is ten
    /// times wrong in 1/49th of the projection's input.
    public func hiddenStates(inputIDs: [Int32], tokenWeights: [Float]) throws
        -> [MLXArray] {
        let tokens = inputIDs.count
        guard tokens > 0 else { throw Failure.badTokenCount(tokens) }

        let embedding = try weight("embed_tokens.weight")
        let ids = MLXArray(inputIDs)
        // The embedding lookup is scaled by sqrt(hidden_size).
        var h = embedding[ids].reshaped([1, tokens, topology.hiddenSize])
            * MLXArray(embedScale).asType(embedding.dtype)

        var states: [MLXArray] = [h]
        let valid = MLXArray(tokenWeights)

        // cos/sin and masks are per layer KIND, computed once each.
        var position: [TextEncoderTopology.LayerKind: (cos: MLXArray, sin: MLXArray)] = [:]
        var masks: [TextEncoderTopology.LayerKind: MLXArray] = [:]
        for kind in [TextEncoderTopology.LayerKind.sliding, .full] {
            position[kind] = positionEmbeddings(kind: kind, tokens: tokens)
            masks[kind] = attentionMask(kind: kind, tokens: tokens, valid: valid)
        }

        for layer in 0..<topology.layerCount {
            let kind = topology.layerTypes[layer]
            let p = position[kind]!
            h = try decoderLayer(h, layer: layer, kind: kind,
                                cos: p.cos.asType(h.dtype), sin: p.sin.asType(h.dtype),
                                mask: masks[kind]!.asType(h.dtype))
            MLX.eval(h)                 // bound peak memory: 48 layers of 1024x3840

            if layer < topology.layerCount - 1 {
                states.append(h)
            } else {
                // The last state is the FINAL-NORMED output. Contract 13.
                states.append(rmsNorm(h, weight: try weight("norm.weight")))
            }
        }
        return states
    }
}

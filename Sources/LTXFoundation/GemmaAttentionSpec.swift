// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The arithmetic of one Gemma-4 attention layer.
///
/// Pure math and no MLX, because every value here is a place where a reasonable
/// implementation is wrong. The structural facts are enforced by
/// `LTXCatalog.TextEncoderTopology` against the weights; this is the part the
/// weights cannot tell you.
public struct GemmaAttentionSpec: Sendable, Equatable {

    /// **There is no `1/sqrt(head_dim)`.**
    ///
    /// Gemma 4 sets the attention scale to 1.0 and passes it straight through to
    /// the attention itself. The softmax scale is folded into the
    /// learned `k_norm` weights instead, which is visible in the checkpoint: they sit
    /// at ≈0.122 for the 256-dim heads and ≈0.061 for the 512-dim ones, both near
    /// `1/16`, where a bare RMSNorm scale would sit near 1.0 like `q_norm` does.
    ///
    /// A port that applies the conventional scale double-scales the logits by 16× to
    /// 22×, which saturates the softmax into near-one-hot attention. Shapes are
    /// unaffected.
    public static let attentionScaling: Float = 1.0

    /// Gemma-4's RMSNorm is `normed * weight` — **not** `normed * (1 + weight)`.
    ///
    /// Gemma-2 and Gemma-3 use the `+1` convention, so this is exactly the sort of
    /// detail a port inherits from the wrong family. Confirmed two ways: Gemma 4's
    /// RMSNorm multiplies by its weight directly, and the checkpoint's norm weights
    /// centre on 1.0 rather than 0.0 (`q_norm` is a
    /// uniform 1.0312; `post_attention_layernorm` averages 1.349). Under the `+1`
    /// reading every norm in the model would be roughly doubled.
    public static let rmsNormAddsOneToWeight = false

    /// The norm is computed in **fp32** and cast back, even in a bf16 model.
    /// Normalising in bf16 is a measurably different result.
    public static let rmsNormAccumulatesInFloat32 = true

    /// The norm is defined as `pow(meanSquares, -0.5)` rather than `rsqrt`. The two
    /// can differ in the last bits, so this is recorded rather than treated as
    /// interchangeable.
    public static let rmsNormUsesPowNotRsqrt = true

    public enum LayerKind: String, Sendable, CaseIterable {
        case sliding = "sliding_attention"
        case full = "full_attention"
    }

    public struct RoPESpec: Sendable, Equatable {
        public let headDim: Int
        public let theta: Double
        public let partialRotaryFactor: Double
        /// `"default"` or `"proportional"` — and they are *not* the same formula.
        public let ropeType: String
    }

    public let sliding: RoPESpec
    public let full: RoPESpec

    public init(sliding: RoPESpec, full: RoPESpec) {
        self.sliding = sliding
        self.full = full
    }

    /// The measured 2.5 configuration.
    public static let ltx25 = GemmaAttentionSpec(
        sliding: RoPESpec(headDim: 256, theta: 10_000, partialRotaryFactor: 1.0,
                          ropeType: "default"),
        full: RoPESpec(headDim: 512, theta: 1_000_000, partialRotaryFactor: 0.25,
                       ropeType: "proportional"))

    public func spec(for kind: LayerKind) -> RoPESpec {
        kind == .sliding ? sliding : full
    }

    /// Inverse frequencies for one layer kind, length `headDim / 2`.
    ///
    /// ## The two formulas differ in their denominator
    ///
    /// This is the trap. Both compute `1 / theta^(2i / denominator)`, but:
    ///
    /// - **`default`** rotates the whole head, and its denominator is
    ///   `dim = headDim * partialRotaryFactor` — which equals `headDim` when the
    ///   factor is 1.
    /// - **`proportional`** divides by the **full `headDim`** even though only
    ///   `partialRotaryFactor` of it rotates. It produces `headDim/2` frequencies of
    ///   which the first `partialRotaryFactor * headDim / 2` are real and the rest
    ///   are **zero**.
    ///
    /// Using one formula for both gives wrong frequencies in the eight global
    /// layers: with `headDim` 512 and factor 0.25, the `default` reading would
    /// divide by 128 where `proportional` divides by 512.
    ///
    /// ## Partial rotation is zero frequencies, not a tensor split
    ///
    /// The non-rotated tail is padded with zeros rather than sliced out. A zero
    /// frequency gives `cos = 1, sin = 0`, so `x*cos + rotate_half(x)*sin` is the
    /// identity on those dimensions — the whole head goes through one uniform
    /// rotation and the tail passes through unchanged. A port that splits the tensor
    /// and concatenates must reproduce this exactly; matching the zeros is simpler
    /// and cannot drift.
    public func inverseFrequencies(for kind: LayerKind) -> [Double] {
        let s = spec(for: kind)
        let half = s.headDim / 2

        switch s.ropeType {
        case "proportional":
            // Denominator is the FULL head dim. `int(factor * headDim // 2)` real
            // angles, zero-padded to `half`.
            let angles = Int(s.partialRotaryFactor * Double(s.headDim) / 2.0)
            var out = [Double](repeating: 0, count: half)
            for i in 0..<min(angles, half) {
                out[i] = 1.0 / pow(s.theta, Double(2 * i) / Double(s.headDim))
            }
            return out

        default:
            // Denominator is `dim`, the rotated width.
            let dim = Int(Double(s.headDim) * s.partialRotaryFactor)
            var out = [Double](repeating: 0, count: dim / 2)
            for i in 0..<(dim / 2) {
                out[i] = 1.0 / pow(s.theta, Double(2 * i) / Double(dim))
            }
            return out
        }
    }

    /// How many of a kind's frequencies are non-zero — i.e. actually rotate.
    public func rotatingFrequencyCount(for kind: LayerKind) -> Int {
        inverseFrequencies(for: kind).reduce(into: 0) { $0 += $1 != 0 ? 1 : 0 }
    }

    // MARK: - The ordering that produces Q, K and V

    /// Where each projection's output goes, in order:
    ///
    /// ```
    /// q = q_norm(q_proj(h));  q = rope(q)
    /// k = k_proj(h)
    /// v = v_proj(h) if there is one, else k           // <- binds the PRE-norm k
    /// k = k_norm(k);  k = rope(k)
    /// v = v_norm(v)                                   // no rope on V
    /// ```
    ///
    /// Two things there:
    ///
    /// **`v_norm` exists and has no weights.** It is an RMSNorm over `head_dim` with
    /// no scale, so V is RMS-normalised with no learned weight — and because a
    /// scale-free norm allocates no parameter, **there is no `v_norm` tensor in the
    /// checkpoint at all** (verified: zero matches).
    /// A port assembled by enumerating checkpoint tensors cannot discover this
    /// operation, and omitting it leaves V unnormalised at every one of the 48
    /// layers.
    ///
    /// **On the K-shares-V layers, V aliases the *pre-norm* projection.** The
    /// binding `v = k` happens before `k = k_norm(k)` rebinds the name, so V is
    /// `v_norm(k_proj(h))` while K is `rope(k_norm(k_proj(h)))`. They share the
    /// projection and nothing else. Writing `v = k` after computing K — the natural
    /// reading of "K equals V" — gives V the k_norm scale and the rotation, both
    /// wrong.
    public struct ProjectionOrder: Sendable, Equatable {
        public let queryIsNormalizedThenRotated = true
        public let keyIsNormalizedThenRotated = true
        /// V is normalised and **never** rotated.
        public let valueIsRotated = false
        /// V is normalised — by a norm with no learnable weight.
        public let valueIsNormalized = true
        public let valueNormHasWeight = false
        /// When K is reused as V, the alias is taken before `k_norm`.
        public let sharedValueAliasesPreNormKey = true

        public init() {}
    }

    public static let projectionOrder = ProjectionOrder()

    /// Whether a layer kind projects V separately.
    ///
    /// `attention_k_eq_v && !is_sliding`, so only the global layers share. Checked against the weights by `TextEncoderTopology.verify`: exactly
    /// layers 5, 11, 17, 23, 29, 35, 41, 47 have no `v_proj`.
    public func projectsValueSeparately(for kind: LayerKind, keyEqualsValue: Bool) -> Bool {
        !(keyEqualsValue && kind == .full)
    }
}

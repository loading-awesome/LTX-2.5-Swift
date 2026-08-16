// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX
import MLXFast

/// The six attention modules of a DiT block.
///
/// A block runs `attn1`, `attn2`, `audio_attn1`, `audio_attn2`, `audio_to_video_attn` and
/// `video_to_audio_attn`, and they are six configurations of **one** operation, differing
/// only in their widths and in which of `context` / `pe` / `k_pe` they are handed. So this
/// is one implementation with a `Variant` selecting the weight stem, rather than six
/// near-copies that can drift apart.
///
/// ## What one attention module does
///
/// ```
/// context = x when no context is given
/// v = to_v(context)
/// q = to_q(x)
/// k = to_k(context)
/// q, k = q_norm(q), k_norm(k), then RoPE
/// out = sdpa(q, k, v, heads)                      # scale 1/sqrt(dim_head)
/// out = out * 2 * sigmoid(to_gate_logits(x))      # per head
/// return to_out.0(out)
/// ```
///
/// Three things in that are easy to get subtly wrong and produce a correctly-shaped
/// tensor either way; each is documented at the operation that avoids it.
///
/// ## The structure, read from the checkpoint rather than assumed
///
/// Every variant has 32 heads. The head dim does **not** follow:
///
/// ```
/// variant                query   context   inner   heads x dim   rope
/// attn1                   4096      4096    4096     32 x 128     self
/// attn2                   4096      4096    4096     32 x 128     none
/// audio_attn1             2048      2048    2048     32 x  64     self
/// audio_attn2             2048      2048    2048     32 x  64     none
/// audio_to_video_attn     4096      2048    2048     32 x  64     q + k
/// video_to_audio_attn     2048      4096    2048     32 x  64     q + k
/// ```
///
/// The two cross-modal modules are the interesting rows. `audio_to_video_attn` projects a
/// 4096-wide video query **down** to a 2048-wide inner dim to meet the audio context, and
/// `to_out.0` projects back up (`[4096, 2048]`). So "inner dim equals query dim" — true for
/// the four same-stream modules — is false exactly where it is most tempting to write the
/// code by analogy. The head dim is derived from `to_q.weight.shape[0] / heads` here,
/// never from the query width.
public struct DiTAttention {

    public let weights: [String: MLXArray]
    /// `model.diffusion_model.` on a ComfyUI export, `""` on a stripped one. Both occur.
    public let prefix: String
    public let topology: TransformerTopology

    /// `num_attention_heads`, which the config requires to equal
    /// `audio_num_attention_heads` — so one constant covers both streams and all six
    /// variants. The *head dim* is what differs, and that is read per module.
    public static let heads = 32
    /// `norm_eps` from the model config.
    public static let normEpsilon = Float(1e-6)

    /// Which of the six, as the weight stem the checkpoint uses.
    public enum Variant: String, CaseIterable, Sendable {
        case videoSelf = "attn1"
        case videoCross = "attn2"
        case audioSelf = "audio_attn1"
        case audioCross = "audio_attn2"
        case audioToVideo = "audio_to_video_attn"
        case videoToAudio = "video_to_audio_attn"

        /// Whether this variant is handed a rotary embedding at all.
        ///
        /// **`attn2` and `audio_attn2` are not.** When no rotary embedding is supplied
        /// the rotation is skipped outright — there is no fallback to an identity
        /// rotation and no reuse of the self-attention embedding. Their context is the
        /// text conditioning, which has no position in the latent grid. Rotating them
        /// "for symmetry with the other four" is silently wrong.
        public var takesRotary: Bool {
            switch self {
            case .videoCross, .audioCross: return false
            default: return true
            }
        }
    }

    /// One rotary embedding: the `(cos, sin)` pair, each `[B, heads, tokens, dim/2]`.
    ///
    /// **Both halves are required, and neither can be recovered from the other** —
    /// `cos(a*u)` is even in `u`, so the cosine is invariant under exactly the sign flip
    /// the sine is not. This type takes the pair and asks no questions about where it
    /// came from.
    public struct Rotary {
        public let cos: MLXArray
        public let sin: MLXArray
        public init(cos: MLXArray, sin: MLXArray) {
            self.cos = cos
            self.sin = sin
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)
        case rotaryWithoutPosition(Variant)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .rotaryWithoutPosition(v):
                return "\(v.rawValue) takes no rotary embedding; supplying one would "
                    + "rotate a stream that has no latent position"
            }
        }
    }

    public init(checkpoint url: URL, topology: TransformerTopology,
                attentionPath: AttentionPath = .explicit) throws {
        self.topology = topology
        self.prefix = topology.prefix
        self.weights = try MLX.loadArrays(url: url)
        self.attentionPath = attentionPath
    }

    /// Share an already-loaded weight dictionary, so a test that also builds `DiTModules`
    /// does not pay for a second 21 B-parameter map.
    public init(weights: [String: MLXArray], topology: TransformerTopology,
                attentionPath: AttentionPath = .explicit) {
        self.topology = topology
        self.prefix = topology.prefix
        self.weights = weights
        self.attentionPath = attentionPath
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[prefix + name] else { throw Failure.missing(prefix + name) }
        return w
    }

    private func bias(_ name: String) -> MLXArray? { weights[prefix + name] }

    // MARK: - Primitives

    /// A Linear, `[out, in]` in the checkpoint's layout.
    ///
    /// Deliberately duplicated from `DiTModules.linear` rather than shared: a common
    /// helper would let a numerical change made for one module silently move every other
    /// module's output.
    ///
    /// **The input is cast to the weight's dtype, and that is load-bearing.** The block
    /// runs bf16 end to end. Hand this matmul an fp32 input and MLX promotes the whole
    /// product to fp32, so the block computes *more precisely* than its bf16 weights
    /// encode — a different function, not a better one. The cast is lossless whenever the
    /// incoming values are already bf16 held in a wider container, which is the case it
    /// exists for.
    ///
    /// All four of `to_q`, `to_k`, `to_v` and `to_out.0` carry biases here
    /// (`attention_bias: true`), unlike the video feed-forward in the same block. The
    /// bias is looked up rather than assumed, so a checkpoint that drops one is a
    /// numerical difference this code follows rather than a crash.
    /// Low-rank adapters overlaid on this module's four projections.
    ///
    /// Set rather than baked, for the reason `DiTModules.lora` gives. It matters more here
    /// than there: an IC-LoRA's tensors are almost entirely `attn1`/`attn2` `to_{q,k,v,out}`
    /// — 8 of the 10 modules the pixel spatial upscaler adapts per block — so an overlay
    /// wired only into `DiTModules` reaches the two feed-forward projections and *nothing
    /// else*, and the render completes looking like the base model with a faint flavour of
    /// the adapter. That failure has no shape check anywhere.
    public var lora: LoRAOverlay?

    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let w = try weight(name + ".weight")
        var y = MLX.matmul(x.asType(w.dtype), w.transposed(1, 0))
        // Before the bias, and on the linear map rather than the affine one — the same
        // ordering `DiTModules.linear` documents, for the same reason.
        if let lora, let delta = lora.delta(forWeightKey: prefix + name + ".weight", input: x) {
            y = y + delta.asType(y.dtype)
        }
        if let b = bias(name + ".bias") { y = y + b.asType(y.dtype) }
        return y
    }

    /// RMS norm over the **whole inner dim, not per head**.
    ///
    /// This is the single most inviting mistake in the file. The Gemma encoder in this
    /// same repo norms `q` and `k` over `head_dim`, after the head split, because that is
    /// what Gemma-4 does. The DiT does not: the norm runs before any reshape, and the
    /// checkpoint agrees — `q_norm.weight` is `[4096]` for video and `[2048]` for audio,
    /// one weight per inner channel, not one per head channel.
    ///
    /// Dispatched to `rms_norm.metal` (precompiled into `mlx.metallib`). A hand-rolled
    /// mean-square / rsqrt graph is the same arithmetic as a pile of JIT kernels, and
    /// that graph is the off-kernel remainder of a sampling step. Gemma keeps its own
    /// `pow` path; this is not that.
    private func rmsNorm(_ x: MLXArray, weight w: MLXArray) -> MLXArray {
        MLX.rmsNorm(x, weight: w.asType(x.dtype), eps: Self.normEpsilon)
    }

    /// Split rotary — `rope_type: SPLIT`, which is this model's setting.
    ///
    /// The head dim splits into two **contiguous halves** and rotates as
    ///
    /// ```
    /// out[..., :D/2] = x1 * cos - x2 * sin
    /// out[..., D/2:] = x2 * cos + x1 * sin
    /// ```
    ///
    /// where `cos` and `sin` are `[B, heads, tokens, D/2]` and shared by both halves.
    ///
    /// There is also a legacy `INTERLEAVED` rope type, which pairs *adjacent* channels
    /// instead. The two coincide only when `D == 2`; at `D = 128` they are a permutation
    /// apart, with the same shape and a plausible-looking output. `rope_type` defaults to
    /// `SPLIT` and this checkpoint does not override it.
    ///
    /// **Not upcast.** The rotary tables are narrowed to the latent's dtype before they
    /// reach here, so this is bf16 times bf16. Rotating in fp32 would be more accurate
    /// and compute something other than what the weights encode — the same trap as
    /// `linear`, one operation later.
    private func applySplitRoPE(_ x: MLXArray, _ rotary: Rotary) -> MLXArray {
        let d = x.dim(-1)
        let half = d / 2
        let x1 = x[.ellipsis, 0 ..< half]
        let x2 = x[.ellipsis, half ..< d]
        let cos = rotary.cos.asType(x.dtype)
        let sin = rotary.sin.asType(x.dtype)
        return MLX.concatenated([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis: -1)
    }

    /// `x * sigmoid(x)`'s cousin: the sigmoid alone, in fp32 and cast back.
    ///
    /// Same rule as `DiTModules.silu` and `geluTanh`: an elementwise nonlinearity on a
    /// bf16 tensor is evaluated in fp32 and cast back.
    private func sigmoid(_ x: MLXArray) -> MLXArray {
        MLX.sigmoid(x.asType(.float32)).asType(x.dtype)
    }

    /// Where the softmax accumulates.
    ///
    /// MLX's fused kernel accumulates narrower than an explicit fp32 softmax, which
    /// matters enough in `GemmaTextEncoder` to decide its default. That is a fact about a
    /// kernel rather than about attention, so it is checked here rather than assumed, and
    /// here it barely moves: the two paths differ by `1.11e-04` on `attn1` and are
    /// bit-identical on `audio_attn1`. The reductions are short — 26 to 1024 keys — so
    /// there is far less softmax tail to lose than in Gemma's 1024-key layers.
    ///
    /// `.explicit` remains the default anyway — it is never worse here, and it is the
    /// path whose arithmetic is visible.
    ///
    /// ## The cost that comparison did not see, and that decides which path a *render* takes
    ///
    /// The two paths were compared on **accuracy at short sequence lengths**, 26 to 1024
    /// tokens. Neither number says anything about memory, and the memory difference is not
    /// a constant factor — it is a change of asymptotic class.
    ///
    /// `.explicit` materialises `[B, heads, T, T]` **in fp32**, twice over: the logits and
    /// the softmax of them. At 32 heads that is `2 x 32 x T x T x 4` bytes of live
    /// allocation for one self-attention call:
    ///
    /// | render | video tokens `T` | one self-attention |
    /// |---|---|---|
    /// | 640x384x97 (production) | 3,120 | 2.5 GB |
    /// | 576x1024x121 | 9,216 | 21.7 GB |
    /// | 576x1024x241 | 17,856 | **81.6 GB** |
    ///
    /// The last row does not fit beside a 42 GB transformer, and 576x1024x241 was observed
    /// to die on SIGKILL during sampling. It reads like a leak — the same render at half
    /// the length peaks at 55.5 GB and completes — and it is not one: nothing accumulates
    /// across steps, a single attention call is simply quadratic in the token count, so
    /// doubling the clip quadruples the peak.
    ///
    /// `.fused` reaches MLX's steel flash-attention kernel, which streams the reduction and
    /// never materialises the score matrix. The dispatch conditions are checked, not
    /// assumed: `ScaledDotProductAttention::use_fallback` takes the full path when
    /// `query_head_dim == value_head_dim`, that dim is 64 / 80 / 128, there is no mask, and
    /// `T > 8`. Every attention in this block satisfies all four at every render shape: the
    /// two video modules project 4096 into 128-wide heads, and the audio pair and both
    /// cross-modal modules project 2048 into 64-wide ones. A head dim outside `{64, 80,
    /// 128}` would silently fall back to the unfused implementation and hand back the
    /// quadratic peak with no error and no shape change, which is why the set is named here
    /// rather than trusted.
    ///
    /// So the default *here* stays `.explicit`, because the accuracy comparison above is
    /// the reason to prefer it wherever it fits — and `Renderer.Spec` and
    /// `DistilledRenderer.Spec` default to `.fused`, because at render shapes the choice
    /// is between a clip and a SIGKILL.
    public enum AttentionPath: Sendable {
        case fused
        case explicit
    }

    public var attentionPath: AttentionPath = .explicit

    // MARK: - Forward

    /// One attention module.
    ///
    /// - Parameters:
    ///   - x: the query stream, `[B, T, queryDim]`.
    ///   - context: the key/value stream. **`nil` means self-attention** — passing `x`
    ///     explicitly is equivalent, and passing the wrong stream is a shape error
    ///     everywhere except `attn2` / `audio_attn2`, where the conditioning happens to
    ///     be the same width as the query.
    ///   - rotary: `pe`, applied to **both** `q` and `k` unless `keyRotary` is given.
    ///   - keyRotary: `k_pe`. Present only on the two cross-modal modules, where the
    ///     query and key streams have different token counts and therefore different
    ///     positions. The key falls back to `pe` when this is `nil`, so `nil` means "use
    ///     `pe` for both", not "do not rotate the key".
    public func callAsFunction(_ x: MLXArray, context: MLXArray? = nil,
                               rotary: Rotary? = nil, keyRotary: Rotary? = nil,
                               block: Int, variant: Variant,
                               allPerturbed: Bool = false) throws -> MLXArray {
        if rotary != nil && !variant.takesRotary {
            throw Failure.rotaryWithoutPosition(variant)
        }
        let stem = "transformer_blocks.\(block).\(variant.rawValue)."
        let heads = topology.numAttentionHeads

        // Self-attention means the context is the query stream itself, decided before
        // anything is projected.
        let kv = context ?? x

        // Head dim from `to_q`, never from the query width: the two cross-modal modules
        // project 4096 -> 2048 and 2048 -> 2048 into the same 64-wide heads.
        let qWeight = try weight(stem + "to_q.weight")
        guard qWeight.shape.count == 2, qWeight.shape[0] % heads == 0 else {
            throw Failure.shape(stem + "to_q.weight", got: qWeight.shape,
                                expected: "[k * \(heads), queryDim]")
        }
        let inner = qWeight.shape[0]
        let dimHead = inner / heads

        // `v` is projected from the context and then **left alone** — never normed, never
        // rotated. Only the query/key path takes the norms and RoPE; running `v` through
        // the same helper as `q` and `k` gives a same-shaped, wrong result.
        let v = try linear(kv, stem + "to_v")

        // [B, T, inner] -> [B, heads, T, dimHead]; both the STG passthrough and normal
        // attention need this shape for the optional per-head gate.
        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([t.dim(0), t.dim(1), heads, dimHead]).transposed(0, 2, 1, 3)
        }
        let vh = split(v)
        var out: MLXArray
        if allPerturbed {
            // Fully perturbed: project V, then bypass the q/k projections, both RMS
            // norms, RoPE and SDPA.  Do not approximate this by running attention and
            // discarding it — the q/k work is observable in memory, and skipping it is
            // half of what this branch is for.
            out = vh
        } else {
            var q = try linear(x, stem + "to_q")
            var k = try linear(kv, stem + "to_k")
            q = rmsNorm(q, weight: try weight(stem + "q_norm.weight"))
            k = rmsNorm(k, weight: try weight(stem + "k_norm.weight"))
            var qh = split(q), kh = split(k)

            if let rotary {
                qh = applySplitRoPE(qh, rotary)
                kh = applySplitRoPE(kh, keyRotary ?? rotary)
            }

            // SDPA with the conventional `1/sqrt(dim_head)`. Unlike the Gemma encoder — whose
            // scale is 1.0, with the scaling folded into its `k_norm` weights — nothing
            // here overrides the default.
            let scale = Float(1.0 / Double(dimHead).squareRoot())
            switch attentionPath {
            case .fused:
                out = MLXFast.scaledDotProductAttention(queries: qh, keys: kh, values: vh,
                                                        scale: scale, mask: .none)
            case .explicit:
                let logits = MLX.matmul(qh.asType(.float32),
                                        kh.asType(.float32).transposed(0, 1, 3, 2)) * scale
                let weights = MLX.softmax(logits, axis: -1)
                out = MLX.matmul(weights, vh.asType(.float32)).asType(qh.dtype)
            }
        }

        // Per-head gating.
        //
        // **The logits come from `x`, the query stream — not from the attention output.**
        // The checkpoint is unambiguous: `to_gate_logits.weight` is `[32, 4096]` for
        // `audio_to_video_attn`, whose attention output is 2048 wide. Reading the gate off
        // the output would not even load there, and on the four same-stream modules it
        // would load fine and be wrong.
        //
        // `2.0 * sigmoid` is centred on **1**, so an untrained gate is pass-through and a
        // head can be amplified as well as suppressed. A plain sigmoid halves everything.
        if weights[prefix + stem + "to_gate_logits.weight"] != nil {
            let gates = 2.0 * sigmoid(try linear(x, stem + "to_gate_logits"))
            // out is [B, heads, T, dimHead]; gates are [B, T, heads].
            out = out * gates.transposed(0, 2, 1).expandedDimensions(axis: -1)
                .asType(out.dtype)
        }

        let merged = out.transposed(0, 2, 1, 3)
            .reshaped([out.dim(0), out.dim(2), inner])
        return try linear(merged, stem + "to_out.0")
    }
}

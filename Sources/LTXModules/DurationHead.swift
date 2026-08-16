// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The "automatic length" head: predicts a shot duration in seconds from the text
/// conditioning, which then snaps to the frame lattice.
///
/// A regression head, not a generation path. It reads the **connector token outputs** —
/// the same `enc.features.video` / `enc.features.audio` pair
/// ``TextConditioningPipeline/connect(projected:attentionMask:inputIDs:tokenWeights:prompt:branch:)``
/// already returns — projects each modality into a shared 256-wide space, tags them with
/// learnable modality embeddings, pools the concatenation with a single cross-attention
/// query, and pushes the pooled vector through a 2-layer MLP whose output is
/// exponentiated. 15 tensors, 3.8 MB, shipped as its own `model_patches` file and loaded
/// separately from the DiT.
///
/// ```
/// enc.features.video [1, 1024, 4096] --video_input_proj--> [1, 1024, 256] + video_emb ─┐
///                                                                                      ├─ cat(dim 1)
/// enc.features.audio [1, 1024, 2048] --audio_input_proj--> [1, 1024, 256] + audio_emb ─┘
///                                                                                      │
///                              [1, 2048, 256] --attention_pooler(1 query, 4 heads)--> [1, 1, 256]
///                                             --mlp_hidden--> gelu(tanh) --mlp_out--> [1]
///                                             --exp--> seconds
/// ```
///
/// ## What it consumes
///
/// The **positive branch only**. The negative branch is a different prompt and predicts a
/// visibly different duration — 8.05 s where the positive branch gives 4.99 s — so feeding
/// the wrong branch is a silent wrong answer rather than a crash.
///
/// Both modalities are optional and that is kept, but the shipped pipelines pass both. The
/// single-modality predictions are 15.67 s (video alone) and 3.05 s (audio alone) against
/// 4.99 s for the pair — so "pass whichever you have" is not a graceful degradation, it is
/// a different question.
///
/// ## No mask, and none is needed
///
/// The text sequence is padded to 1024 (`FRAGILE_CONTRACTS.md` #9) and a pooler that
/// attended over padding would produce a duration influenced by padding. It does not,
/// because there is no padding left to attend to: ``EmbeddingsConnector`` overwrites every
/// masked position with a learnable register and returns an all-zero mask, so
/// `enc.features.*` is 1024 valid tokens end to end. The pooler correspondingly has no
/// key-padding-mask input to pass one through even if a caller wanted to.
///
/// This is the one place where "the connector's register substitution" stops being an
/// internal detail of the connector and becomes load-bearing for a *different* module. A
/// bug in the registers would produce a plausible duration here, not a shape error.
///
/// ## The head count is invisible in the checkpoint
///
/// `in_proj_weight` is `[3 * 256, 256]` for **any** `num_heads` that divides 256, and
/// `out_proj.weight` is `[256, 256]` for any of them too. Nothing in the 15 tensors, and
/// nothing in the metadata (`"duration_head": {}` is empty), records it. So it comes from
/// the default, `num_pooler_heads=4`.
///
/// Getting it wrong still runs and still returns a plausible number. In fp32, one prompt,
/// all else equal:
///
/// | `num_pooler_heads` | predicted |
/// |---|---|
/// | 1 | 5.990 s |
/// | 2 | 4.556 s |
/// | **4** | **4.991 s** |
/// | 8 | 6.009 s |
///
/// Four plausible shot lengths. See ``Configuration/poolerHeads``.
///
/// ## Seconds, not log-seconds — but log-seconds internally
///
/// The regression target is trained in log-seconds so the loss spreads evenly across orders
/// of magnitude, and `exp` is applied before returning. So ``seconds(video:audio:)``
/// is seconds, directly, and there is **no clamping inside the head at all** — the head will
/// happily return 0.01 s or 400 s. Every clamp lives in the caller
/// (``frames(video:audio:frameRate:minSeconds:maxSeconds:lattice:)``), which is why the two
/// are kept separate here.
///
/// ## Precision
///
/// Weights are bf16 and the matmuls run at the weight's dtype, per the repo rule. The
/// softmax and the GELU compute their statistics in fp32 and **narrow back to bf16**, which
/// is `FRAGILE_CONTRACTS.md` #16 applied to this module: a reduction over bf16 yields bf16,
/// and computing in fp32 and *staying* there runs the head at a precision it was never
/// built for. It is measurable — with bf16 weights, narrowing the attention weights gives
/// 4.9997 s and not narrowing gives 4.9608 s, which is a wider move than the whole bf16
/// quantisation staying in fp32 was trying to avoid.
///
/// The final `exp` is narrowed the same way, so the returned `Double` carries bf16
/// precision — about ±0.016 s at a 5-second prediction. That is deliberate.
///
/// ## What a test can and cannot pin down
///
/// The predictions are a regression, not a lattice: there is no ground-truth mapping from
/// prompts to durations, so nothing asserts that a duration is *sensible* for a given
/// prompt, and the tests deliberately assert nothing about monotonicity or plausibility.
///
/// The concatenation order is video-first, and it is *not observable in the output*: the
/// pooler is a single query cross-attending over the concatenation, and attention is
/// permutation-invariant in its keys and values. Swapping to audio-first moves the
/// prediction from 4.990764 s to 4.990763 s — float noise. So no test can catch a
/// transposed order.
public struct DurationHead {

    // MARK: - Configuration

    /// Every dimension the head needs, read from the checkpoint.
    ///
    /// Read from `__metadata__.config`, and cross-checked against the tensor shapes where a
    /// shape records the same fact — the two disagreeing means the file is not what its
    /// metadata says it is, which is worth a loud failure rather than a shape error four
    /// operations later.
    public struct Configuration: Equatable, Sendable {
        /// `transformer.cross_attention_dim`, 4096. The video connector's output width.
        public let videoWidth: Int
        /// `transformer.audio_cross_attention_dim`, 2048.
        public let audioWidth: Int
        /// `duration_head.pooler_hidden_dim`, 256. The shared space both modalities project
        /// into and the pooler's `embed_dim`.
        public let poolerHidden: Int
        /// `duration_head.num_queries`, 1. Learnable pooling queries; the MLP's input width
        /// is `poolerHidden * queryCount`, so this is not free to change independently.
        public let queryCount: Int

        /// `duration_head.num_pooler_heads`, 4.
        ///
        /// **The one value no tensor in this checkpoint constrains**, and the one the type
        /// comment's table shows is worth getting right. `in_proj_weight` is `[3E, E]`
        /// whether the 256 channels are split into 1, 2, 4, 8 or 256 heads, so a wrong
        /// value here loads cleanly, runs cleanly, and returns a different number of
        /// seconds.
        ///
        /// Sourced from `duration_head.num_pooler_heads` in the metadata if the checkpoint
        /// records it. This one does not — its `duration_head` dict is `{}` — so the value
        /// falls back to the default of 4, see ``DurationHead/referencePoolerHeads``.
        public let poolerHeads: Int

        /// `duration_head.mlp_hidden`, 256.
        public let mlpHidden: Int

        /// The pooler's per-head channel count, 64.
        public var headDim: Int { poolerHidden / poolerHeads }

        public init(videoWidth: Int, audioWidth: Int, poolerHidden: Int, queryCount: Int,
                    poolerHeads: Int, mlpHidden: Int) {
            self.videoWidth = videoWidth
            self.audioWidth = audioWidth
            self.poolerHidden = poolerHidden
            self.queryCount = queryCount
            self.poolerHeads = poolerHeads
            self.mlpHidden = mlpHidden
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case unconsumed([String])
        case shape(String, got: [Int], expected: String)
        case configuration(String)
        case noModality
        case rank(String, got: [Int])
        case featureWidth(String, got: Int, expected: Int)
        case emptySequence(String)
        case batch(String, got: Int)

        public var description: String {
            switch self {
            case let .missing(name):
                return "weight \(name) is absent"
            case let .unconsumed(names):
                return "\(names.count) checkpoint tensors are not consumed by the duration "
                    + "head: \(names.sorted().joined(separator: ", ")). All 15 are accounted "
                    + "for on a 2.5 duration-head patch; an extra one means this is a "
                    + "different module's file or a newer head with parameters this port "
                    + "silently ignores"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .configuration(why):
                return "checkpoint config: \(why)"
            case .noModality:
                return "the duration head needs at least one of video / audio; both were nil"
            case let .rank(name, got):
                return "\(name) is rank \(got.count) \(got), expected rank 3 "
                    + "[batch, tokens, width] — this head takes connector *token* outputs, "
                    + "not a pooled vector"
            case let .featureWidth(name, got, expected):
                return "\(name) is \(got) wide but this checkpoint's input projection takes "
                    + "\(expected). Video is the 4096-wide stream and audio the 2048-wide "
                    + "one; passing them in the wrong order lands here"
            case let .emptySequence(name):
                return "\(name) has zero tokens. The pooler would take a softmax over an "
                    + "empty key axis and return NaN seconds"
            case let .batch(name, got):
                return "\(name) has batch \(got); this head returns one scalar and the "
                    + "reference's own DurationPredictor raises on any batch but 1"
            }
        }
    }

    // MARK: - Weight names

    /// The 15 parameters, with the `duration_head.` prefix already stripped.
    ///
    /// Enumerated rather than derived so the consumed/missing accounting in
    /// ``init(weights:configuration:)`` is against a written-down list. A head that grew a
    /// 16th tensor should fail loudly here, not load 15 of 16 and run.
    static func parameterShapes(for c: Configuration) -> [String: [Int]] {
        [
            "video_input_proj.weight": [c.poolerHidden, c.videoWidth],
            "video_input_proj.bias": [c.poolerHidden],
            "audio_input_proj.weight": [c.poolerHidden, c.audioWidth],
            "audio_input_proj.bias": [c.poolerHidden],
            "video_modality_emb": [c.poolerHidden],
            "audio_modality_emb": [c.poolerHidden],
            "attention_pooler.query_tokens": [c.queryCount, c.poolerHidden],
            // PyTorch `nn.MultiheadAttention`'s packed QKV: rows 0..<E are Q, E..<2E are K,
            // 2E..<3E are V. Packed, not three separate tensors, and the order is not
            // recorded anywhere in the file.
            "attention_pooler.cross_attn.in_proj_weight": [3 * c.poolerHidden, c.poolerHidden],
            "attention_pooler.cross_attn.in_proj_bias": [3 * c.poolerHidden],
            "attention_pooler.cross_attn.out_proj.weight": [c.poolerHidden, c.poolerHidden],
            "attention_pooler.cross_attn.out_proj.bias": [c.poolerHidden],
            "mlp_hidden.weight": [c.mlpHidden, c.poolerHidden * c.queryCount],
            "mlp_hidden.bias": [c.mlpHidden],
            "mlp_out.weight": [1, c.mlpHidden],
            "mlp_out.bias": [1],
        ]
    }

    /// The prefixes stripped from checkpoint keys, in order.
    ///
    /// The patch file ships with `duration_head.`; a head folded into a whole-model
    /// checkpoint carries `model.diffusion_model.duration_head.`. Longest first, because
    /// the second is a suffix of the first and testing the short one first would leave
    /// `model.diffusion_model.` glued to every name.
    static let checkpointPrefixes = ["model.diffusion_model.duration_head.", "duration_head."]

    // MARK: - State

    /// The 15 parameters, prefix stripped, at the checkpoint's own dtype.
    ///
    /// **Not widened.** ``AudioVocoder`` widens everything to fp32 because that whole model
    /// runs in fp32; this one does not, and the matmuls are meant to run at bf16.
    public let weights: [String: MLXArray]
    public let config: Configuration

    // MARK: - Loading

    /// Load the duration-head patch.
    ///
    /// The file is 3.8 MB and holds nothing else, so unlike the DiT and the VAEs there is no
    /// prefix filtering to do beyond stripping the one the exporter wrote.
    public init(checkpoint url: URL) throws {
        let header = try SafetensorsHeader.read(from: url)
        let config = try Self.configuration(from: header)

        let all = try MLX.loadArrays(url: url)
        var stripped: [String: MLXArray] = [:]
        var matched = false
        for prefix in Self.checkpointPrefixes where !matched {
            for (name, value) in all where name.hasPrefix(prefix) {
                stripped[String(name.dropFirst(prefix.count))] = value
                matched = true
            }
        }
        // Fall through to the raw dict when neither prefix hits, for a head exported bare.
        // Accepted rather than failed, because the alternative is a `.missing` naming a
        // prefix the file was never supposed to have.
        try self.init(weights: matched ? stripped : all, configuration: config)
    }

    /// Build from an already-stripped weight dictionary.
    ///
    /// This is where the whole consumed/missing accounting happens — not in a test — because
    /// a structurally wrong head runs to completion and returns a plausible number of
    /// seconds. There is no downstream shape error waiting to catch it.
    public init(weights: [String: MLXArray], configuration: Configuration) throws {
        self.config = configuration

        guard configuration.poolerHidden % configuration.poolerHeads == 0 else {
            throw Failure.configuration(
                "pooler_hidden_dim \(configuration.poolerHidden) is not divisible by "
                + "num_pooler_heads \(configuration.poolerHeads)")
        }

        let expected = Self.parameterShapes(for: configuration)
        var kept: [String: MLXArray] = [:]
        kept.reserveCapacity(expected.count)
        for (name, shape) in expected {
            guard let value = weights[name] else { throw Failure.missing(name) }
            guard value.shape == shape else {
                throw Failure.shape(name, got: value.shape, expected: "\(shape)")
            }
            kept[name] = value
        }
        let extra = Set(weights.keys).subtracting(expected.keys).sorted()
        guard extra.isEmpty else { throw Failure.unconsumed(extra) }
        self.weights = kept
    }

    /// Read ``Configuration`` from the file's `__metadata__.config`.
    ///
    /// The widths come from `transformer.*` because the head's input projections mirror the
    /// main transformer's connector dims — the same key the DiT itself reads, which is why
    /// a duration head is only valid against the checkpoint it shipped beside. The head's
    /// own hyperparameters live in a `duration_head` sub-dict with no settled schema; on
    /// this checkpoint it is empty.
    ///
    /// Where a tensor shape records the same number, the two are cross-checked. That is not
    /// redundant: a metadata block copied from a neighbouring checkpoint would otherwise
    /// build a `Configuration` that disagrees with the file and surface as a `.shape`
    /// failure naming a tensor that is perfectly fine.
    public static func configuration(from header: SafetensorsHeader) throws -> Configuration {
        guard let root = header.metadataJSON("config") else {
            throw Failure.configuration("__metadata__.config is absent or not JSON")
        }
        guard let transformer = root["transformer"] as? [String: Any] else {
            throw Failure.configuration("config.transformer is absent; the head's input "
                + "widths are the transformer's connector dims and are not stored anywhere "
                + "else in this file")
        }
        // Present but empty on this checkpoint. Absent is also fine, and treated as empty,
        // but a *non-dict* is a malformed file.
        let headCfg: [String: Any]
        if let raw = root["duration_head"] {
            guard let dict = raw as? [String: Any] else {
                throw Failure.configuration("config.duration_head is not an object")
            }
            headCfg = dict
        } else {
            headCfg = [:]
        }

        // The header is read before any prefix is stripped, so a shape lookup has to try the
        // same prefixes ``init(checkpoint:)`` does. Looking the bare name up directly finds
        // nothing on the shipped patch — every key there is `duration_head.*` — and the
        // failure would be a `.missing` naming a tensor that is plainly in the file.
        func shape(_ name: String) throws -> [Int] {
            for prefix in checkpointPrefixes + [""] {
                if let entry = header.tensors[prefix + name] { return entry.shape }
            }
            throw Failure.missing(name)
        }
        func require(_ dict: [String: Any], _ key: String, _ label: String) throws -> Int {
            guard let v = dict[key] as? Int else {
                throw Failure.configuration("\(label) is absent or not an integer")
            }
            return v
        }
        func agree(_ configured: Int, _ observed: Int, _ what: String) throws {
            guard configured == observed else {
                throw Failure.configuration(
                    "\(what): metadata says \(configured), the tensors say \(observed)")
            }
        }

        let videoWidth = try require(transformer, "cross_attention_dim",
                                     "config.transformer.cross_attention_dim")
        let audioWidth = try require(transformer, "audio_cross_attention_dim",
                                     "config.transformer.audio_cross_attention_dim")
        let videoProj = try shape("video_input_proj.weight")
        let audioProj = try shape("audio_input_proj.weight")
        guard videoProj.count == 2, audioProj.count == 2 else {
            throw Failure.shape("*_input_proj.weight", got: videoProj + audioProj,
                                expected: "two rank-2 tensors")
        }
        try agree(videoWidth, videoProj[1], "video connector width")
        try agree(audioWidth, audioProj[1], "audio connector width")

        // `query_tokens` is `[num_queries, pooler_hidden_dim]`, so both are readable from a
        // shape. Prefer the metadata when it records them, then check the two agree.
        let queries = try shape("attention_pooler.query_tokens")
        guard queries.count == 2 else {
            throw Failure.shape("attention_pooler.query_tokens", got: queries,
                                expected: "[num_queries, pooler_hidden_dim]")
        }
        let queryCount = (headCfg["num_queries"] as? Int) ?? queries[0]
        let poolerHidden = (headCfg["pooler_hidden_dim"] as? Int) ?? queries[1]
        try agree(queryCount, queries[0], "num_queries")
        try agree(poolerHidden, queries[1], "pooler_hidden_dim")

        let mlp = try shape("mlp_hidden.weight")
        guard mlp.count == 2 else {
            throw Failure.shape("mlp_hidden.weight", got: mlp,
                                expected: "[mlp_hidden, pooler_hidden_dim * num_queries]")
        }
        let mlpHidden = (headCfg["mlp_hidden"] as? Int) ?? mlp[0]
        try agree(mlpHidden, mlp[0], "mlp_hidden")

        // The one value with no observable shape. See `Configuration.poolerHeads`.
        let poolerHeads = (headCfg["num_pooler_heads"] as? Int) ?? referencePoolerHeads

        return Configuration(videoWidth: videoWidth, audioWidth: audioWidth,
                             poolerHidden: poolerHidden, queryCount: queryCount,
                             poolerHeads: poolerHeads, mlpHidden: mlpHidden)
    }

    /// The `num_pooler_heads` fallback, used when the checkpoint's config does not say.
    ///
    /// Named rather than inlined because it is the only number in this file that is not read
    /// from the checkpoint, and it should be obvious in a diff if it ever moves.
    public static let referencePoolerHeads = 4

    // MARK: - Primitives

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(name) }
        return w
    }

    /// A Linear in the checkpoint's PyTorch `[out, in]` layout.
    ///
    /// **The input is cast to the weight's dtype**, the same rule as `DiTAttention.linear`
    /// and for the same reason: `enc.features.*` arrives as fp32 from
    /// ``TextConditioningPipeline``, and letting MLX promote the matmul to fp32 would run
    /// the head at a precision it was never built for. This head runs at the pipeline
    /// dtype, which is bf16.
    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let w = try weight(name + ".weight")
        var y = MLX.matmul(x.asType(w.dtype), w.transposed(1, 0))
        if let b = weights[name + ".bias"] { y = y + b.asType(y.dtype) }
        return y
    }

    /// `gelu(x, approximate: "tanh")`, evaluated in fp32 and cast back.
    ///
    /// Identical to ``DiTModules``' and ``EmbeddingsConnector``'s, deliberately duplicated
    /// rather than shared for the same reason those two are duplicated from each other: a
    /// shared helper would let a change made for one module silently move another's
    /// output.
    private func geluTanh(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let inner = Float(0.7978845608028654) * (f + Float(0.044715) * f * f * f)
        return (Float(0.5) * f * (1.0 + MLX.tanh(inner))).asType(x.dtype)
    }

    // MARK: - Forward

    /// Predicted seconds, before any lattice snapping.
    ///
    /// No clamping — the head does not clamp and neither does this. A caller wanting the
    /// `[min_seconds, max_seconds]` behaviour wants
    /// ``frames(video:audio:frameRate:minSeconds:maxSeconds:lattice:)``; a caller debugging
    /// a surprising length wants both numbers, which is why they are separate calls.
    ///
    /// - Parameters:
    ///   - video: `enc.features.video`, `[1, tokens, videoWidth]`, or `nil`.
    ///   - audio: `enc.features.audio`, `[1, tokens, audioWidth]`, or `nil`.
    ///     The two need not have the same token count — they are concatenated along the
    ///     token axis, not stacked.
    ///
    /// Pass the **positive** branch. The negative branch is a different prompt and predicts
    /// a different duration.
    public func seconds(video: MLXArray?, audio: MLXArray?) throws -> Double {
        MLX.exp(try logSeconds(video: video, audio: audio).asType(.float32))
            .asType(try logSecondsDType()).item(Float.self).asDouble
    }

    /// The MLP's raw output: **log**-seconds, the space the head was trained in.
    ///
    /// Exposed because the exponential compresses the interesting part. A head that has gone
    /// wrong by a factor of 20 is a shift of 3 here and an eye-catching 100 seconds there,
    /// and the two are the same defect.
    public func logSeconds(video: MLXArray?, audio: MLXArray?) throws -> MLXArray {
        guard video != nil || audio != nil else { throw Failure.noModality }

        var groups: [MLXArray] = []
        var batch: Int?

        // Video first, then audio. The order is not observable in the output — the pooler
        // is one query cross-attending over the concatenation, and attention is
        // permutation-invariant in its keys and values — so nothing downstream would catch
        // a transposed order.
        if let video {
            try check(video, name: "video", width: config.videoWidth, batch: &batch)
            groups.append(try linear(video, "video_input_proj")
                + (try weight("video_modality_emb")))
        }
        if let audio {
            try check(audio, name: "audio", width: config.audioWidth, batch: &batch)
            groups.append(try linear(audio, "audio_input_proj")
                + (try weight("audio_modality_emb")))
        }

        let tokens = groups.count == 1 ? groups[0] : MLX.concatenated(groups, axis: 1)
        let pooled = try pool(tokens)

        // Flatten the query axis into the feature axis. With `num_queries == 1` this is a
        // squeeze, which is why `mlp_hidden.weight`
        // is `[256, 256]` rather than `[256, 256 * num_queries]` on this checkpoint. A head
        // trained with more queries would concatenate them here, and reading that as a mean
        // or taking query 0 would load fine at `num_queries == 1` and be wrong beyond it.
        let flat = pooled.reshaped([pooled.dim(0), config.poolerHidden * config.queryCount])
        let hidden = geluTanh(try linear(flat, "mlp_hidden"))
        return try linear(hidden, "mlp_out").squeezed(axis: -1)
    }

    private func logSecondsDType() throws -> DType { try weight("mlp_out.weight").dtype }

    private func check(_ x: MLXArray, name: String, width: Int, batch: inout Int?) throws {
        guard x.ndim == 3 else { throw Failure.rank(name, got: x.shape) }
        guard x.dim(2) == width else {
            throw Failure.featureWidth(name, got: x.dim(2), expected: width)
        }
        guard x.dim(1) > 0 else { throw Failure.emptySequence(name) }
        guard x.dim(0) == 1 else { throw Failure.batch(name, got: x.dim(0)) }
        batch = x.dim(0)
    }

    /// The attention pooler — multi-head attention of `(queries, tokens, tokens)`.
    ///
    /// The learnable queries are the *query* side and the connector tokens are both key and
    /// value, which is the direction that makes the output shape independent of sequence
    /// length. Reversed, it would produce `[1, tokens, 256]` and fail at the MLP — the one
    /// mistake in this file that does not silently succeed.
    private func pool(_ tokens: MLXArray) throws -> MLXArray {
        let b = tokens.dim(0)
        let e = config.poolerHidden
        let heads = config.poolerHeads
        let dim = config.headDim

        // The learnable queries, broadcast across the batch.
        let queries = try weight("attention_pooler.query_tokens")
            .reshaped([1, config.queryCount, e])
        let q0 = b == 1 ? queries : MLX.broadcast(queries, to: [b, config.queryCount, e])

        // Unpack `in_proj_weight` `[3E, E]`. The packed order is Q, K, V and it is recorded
        // nowhere in the file — reading it as K, Q, V would load and run.
        let inW = try weight("attention_pooler.cross_attn.in_proj_weight")
        let inB = try weight("attention_pooler.cross_attn.in_proj_bias")
        func project(_ x: MLXArray, _ slot: Int) -> MLXArray {
            let w = inW[(slot * e) ..< ((slot + 1) * e)]
            let bias = inB[(slot * e) ..< ((slot + 1) * e)]
            return MLX.matmul(x.asType(w.dtype), w.transposed(1, 0)) + bias.asType(w.dtype)
        }
        let q = project(q0, 0)
        let k = project(tokens, 1)
        let v = project(tokens, 2)

        func split(_ t: MLXArray) -> MLXArray {
            t.reshaped([t.dim(0), t.dim(1), heads, dim]).transposed(0, 2, 1, 3)
        }
        let qh = split(q), kh = split(k), vh = split(v)

        // `head_dim ** -0.5`, the only scale a packed multi-head attention offers. Not
        // `embed_dim ** -0.5` — at 4 heads those differ by 2x and the wrong one merely
        // sharpens or flattens the pooling.
        let scale = Float(1.0 / Double(dim).squareRoot())

        // Explicit rather than fused, matching `DiTAttention`'s default, because the
        // accumulation width is the whole point here and a fused kernel's is not visible.
        //
        // The statistic accumulates in fp32 and the result **narrows back to the weight
        // dtype** before it multiplies V — contract 16. The attention weights are bf16 by
        // the time they reach the value matmul. With bf16 weights that is worth 4.9997 s
        // narrowed against 4.9608 s not narrowed: staying in fp32 is the more accurate arm
        // and it moves the prediction by more than the whole bf16 quantisation it was
        // trying to avoid.
        let logits = MLX.matmul(qh.asType(.float32),
                                kh.asType(.float32).transposed(0, 1, 3, 2)) * scale
        let attention = MLX.softmax(logits, axis: -1).asType(qh.dtype)
        let out = MLX.matmul(attention, vh)

        let merged = out.transposed(0, 2, 1, 3).reshaped([b, config.queryCount, e])
        return try linear(merged, "attention_pooler.cross_attn.out_proj")
    }

    // MARK: - The lattice

    /// `min_seconds` — the default floor a caller clamps a prediction to.
    public static let defaultMinSeconds = 1.0
    /// `max_seconds` — the default ceiling. Not a model constant: it is a guard against a
    /// misbehaving prediction requesting an OOM-sized generation, and offline analysis of
    /// raw predictions is meant to widen it.
    public static let defaultMaxSeconds = 20.0

    /// Predicted seconds, snapped to the `8k + 1` frame lattice.
    ///
    /// The snapping is `FRAGILE_CONTRACTS.md` #4 and belongs to ``FrameLattice``, including
    /// its one asymmetry — floor onto the lattice, but raise a result that lands below the
    /// minimum. Nothing here reimplements it; this call exists so a caller does not have to
    /// remember the clamp defaults.
    ///
    /// The minimum itself is `max(1, round(minSeconds * frameRate))`. The floor of 1 only
    /// bites when `minSeconds * frameRate < 0.5`, and since the lattice's smallest point is
    /// 1 anyway it changes the intermediate rather than the result.
    public func frames(video: MLXArray?, audio: MLXArray?, frameRate: Double,
                       minSeconds: Double = DurationHead.defaultMinSeconds,
                       maxSeconds: Double = DurationHead.defaultMaxSeconds,
                       lattice: FrameLattice = FrameLattice()) throws -> Int {
        Self.snap(seconds: try seconds(video: video, audio: audio), frameRate: frameRate,
                  minSeconds: minSeconds, maxSeconds: maxSeconds, lattice: lattice)
    }

    /// The seconds-to-frames half on its own, so it is testable without a checkpoint.
    ///
    /// A one-line delegation and deliberately so: this half is arithmetic that can be
    /// asserted exactly, while the prediction needs a checkpoint and cannot be, so the seam
    /// between them is where the test suite splits.
    public static func snap(seconds: Double, frameRate: Double,
                            minSeconds: Double = DurationHead.defaultMinSeconds,
                            maxSeconds: Double = DurationHead.defaultMaxSeconds,
                            lattice: FrameLattice = FrameLattice()) -> Int {
        lattice.frames(forSeconds: seconds, frameRate: frameRate,
                       minSeconds: minSeconds, maxSeconds: maxSeconds)
    }
}

private extension Float {
    var asDouble: Double { Double(self) }
}

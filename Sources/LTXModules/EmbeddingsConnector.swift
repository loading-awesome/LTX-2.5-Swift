// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich
//
// swiftlint:disable file_length

import Foundation
import LTXFoundation
import MLX

/// The 8-layer 1-D transformer between the text projection and the DiT, in MLX.
///
/// The projection (`TextProjectionMLX`) turns 49 Gemma hidden states into one 4096-wide
/// video stream and one 2048-wide audio stream. This is what happens to them next, and it
/// is not a thin adapter: eight full pre-norm transformer blocks per stream, with gated
/// attention and split RoPE, ~1.6 B parameters on the video side alone.
///
/// ## Four things about this module that are easy to get wrong
///
/// **1. The weights are in the DiT checkpoint, not the text encoder.** The projection this
/// module consumes lives in the text encoder checkpoint, so every instinct says to look
/// there for these too. The 258 tensors are instead in
/// `ltx-2.5-22b-dev-transformer-bf16.safetensors`, under
/// `model.diffusion_model.{video,audio}_embeddings_connector.`. Nothing in the text
/// encoder file will tell you they are missing — you get a `nil` lookup at a path you had
/// no reason to doubt.
///
/// **2. `learnable_registers` does not extend the sequence.** The name reads like the
/// register tokens of the ViT literature, which are *prepended* — and prepending 128 rows
/// produces a `[1, 1152, 4096]` tensor that is wrong in a way the next module happily
/// accepts. It is a **substitution**: the `[128, width]` block is tiled `S / 128` times
/// and every position the mask calls padding is overwritten with the corresponding
/// learned row. The sequence length never changes. A 1024-position row with 11 real text
/// tokens is 1013 registers, so this decides almost the entire tensor.
///
/// **3. The connector returns a ZEROED mask, and that mask is what its own blocks use.**
/// The incoming mask is zeroed before the block loop: once padding has been replaced by
/// registers there is nothing left to hide, so attention inside the stack is unmasked.
/// Keeping the incoming `-finfo.max` mask masks out the registers the module just went to
/// the trouble of inserting — no shape error, no NaN, just a quietly different answer.
/// This type therefore takes a mask, uses it *only* to choose register positions, and
/// runs attention with no mask at all.
///
/// **4. RoPE here is `split`, not interleaved, and its tables are cast to bf16 before
/// use.** Split rotates the two contiguous *halves* of each head against each other;
/// interleaved rotates adjacent pairs. Both produce the same shape. The tables are built
/// from a **float64** frequency grid (`frequencies_precision: "float64"`) and then
/// narrowed — first to fp32, then to the hidden dtype — so keeping them in fp32 is not
/// more accurate, it is computing something else.
///
/// ## Dtype
///
/// The rules `DiTModules` establishes hold here and are worth restating because this
/// module has 48 matmuls per stream to get them wrong in: **matmuls take the weight's
/// dtype** (bf16), while **elementwise nonlinearities and norm statistics run in fp32 and
/// cast back**, because a bf16 reduction has no tail left to keep.
public struct EmbeddingsConnector {

    public typealias Stream = TextConditioningLayout.Stream

    /// The whole checkpoint, memory-mapped. Both streams can share one of these.
    public let weights: [String: MLXArray]
    /// `model.diffusion_model.` on a ComfyUI export, `""` on a stripped one — detected
    /// from the tensors rather than assumed, the same two cases `DiTModules` handles.
    public let prefix: String
    public let stream: Stream

    /// Everything below is **read from the weights**, never assumed.
    ///
    /// The head count in particular: audio is 2048 wide and the obvious guess is 16 heads
    /// of 128 to match video's head size. It is 32 heads of **64** — the checkpoint's
    /// `audio_connector_attention_head_dim: 64`. `to_gate_logits.weight` is `[heads,
    /// width]`, so the tensor states the head count outright and there is no reason to
    /// infer it. Getting it wrong reshapes the attention output into the wrong lanes and
    /// still type-checks.
    public let width: Int
    public let heads: Int
    public let headDim: Int
    public let layerCount: Int
    /// Rows in `learnable_registers`; 128 here. The sequence must be a multiple of it.
    public let registerCount: Int

    /// `connector_positional_embedding_max_pos`. 4096 with 1024 tokens, so the sequence
    /// occupies the first quarter of the positional range — not a normalisation to [0, 1).
    public static let positionalMaxPosition = 4096
    /// `positional_embedding_theta`. Nothing in the config overrides it for the
    /// connector, so this is the connector's own default rather than the DiT's value.
    public static let positionalTheta = 10_000.0

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case sequenceNotTiled(sequence: Int, registers: Int)
        case shape(String, got: [Int], expected: String)

        public var description: String {
            switch self {
            case let .missing(name):
                return "weight \(name) is absent"
            case let .sequenceNotTiled(sequence, registers):
                return "sequence \(sequence) is not a multiple of \(registers) registers"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            }
        }
    }

    public init(checkpoint url: URL, stream: Stream) throws {
        try self.init(weights: try MLX.loadArrays(url: url), stream: stream)
    }

    /// Shares one loaded checkpoint between the two streams — the file is 26 GiB and
    /// `MLX.loadArrays` maps it, but two maps is still two maps.
    public init(weights: [String: MLXArray], stream: Stream) throws {
        self.weights = weights
        self.stream = stream

        let stem = "\(stream == .video ? "video" : "audio")_embeddings_connector."
        let registers = "learnable_registers"
        if weights["model.diffusion_model." + stem + registers] != nil {
            self.prefix = "model.diffusion_model." + stem
        } else if weights[stem + registers] != nil {
            self.prefix = stem
        } else {
            throw Failure.missing("model.diffusion_model." + stem + registers)
        }

        guard let registerBlock = weights[self.prefix + registers],
              registerBlock.shape.count == 2
        else { throw Failure.missing(self.prefix + registers) }
        self.registerCount = registerBlock.shape[0]
        self.width = registerBlock.shape[1]

        let gate = self.prefix + "transformer_1d_blocks.0.attn1.to_gate_logits.weight"
        guard let gateWeight = weights[gate], gateWeight.shape.count == 2 else {
            throw Failure.missing(gate)
        }
        guard gateWeight.shape[1] == self.width, self.width % gateWeight.shape[0] == 0 else {
            throw Failure.shape(gate, got: gateWeight.shape,
                                expected: "[heads, \(self.width)] with heads dividing width")
        }
        self.heads = gateWeight.shape[0]
        self.headDim = self.width / self.heads

        var blocks = 0
        while weights[self.prefix + "transformer_1d_blocks.\(blocks).attn1.to_q.weight"]
            != nil { blocks += 1 }
        guard blocks > 0 else {
            throw Failure.missing(self.prefix + "transformer_1d_blocks.0.attn1.to_q.weight")
        }
        self.layerCount = blocks
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[prefix + name] else { throw Failure.missing(prefix + name) }
        return w
    }

    // MARK: - Primitives

    /// A Linear, `[out, in]` in the checkpoint's layout, **with the input cast to the
    /// weight's dtype**. See `DiTModules.linear` for why the cast goes that direction.
    ///
    /// Every Linear in this module has a bias — `ff_bias` defaults to `true` for the
    /// connector, unlike the DiT's video feed-forward, which has none. The bias is looked
    /// up rather than assumed so a checkpoint that disagreed would fail loudly.
    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let w = try weight(name + ".weight")
        var y = MLX.matmul(x.asType(w.dtype), w.transposed(1, 0))
        if let b = weights[prefix + name + ".bias"] { y = y + b.asType(y.dtype) }
        return y
    }

    /// RMS norm with `eps = 1e-6`; the weight is optional.
    ///
    /// **The block norms have no weight tensor at all.** Each block norms with no
    /// parameter before attention and again before the feed-forward, and the stack norms
    /// once more on the way out (`connector_norm_output: true`). Only `q_norm` and
    /// `k_norm` are learned. Code assembled by enumerating the checkpoint's tensors
    /// cannot discover the weightless norms — there is nothing in the file to enumerate —
    /// and dropping the final one leaves the output off by its own RMS, a scale error
    /// rather than a rotation, so it survives every direction-only sanity check.
    ///
    /// Statistics in fp32 and cast back: a 4096-wide mean of squares in bf16 loses the
    /// tail.
    ///
    /// Stay on this graph rather than dispatching to `MLX.rmsNorm`. That kernel is fine
    /// in the DiT, but it accumulates narrower than this and the eight-block stack is
    /// sensitive enough that the difference reaches the output.
    private func rmsNorm(_ x: MLXArray, weight w: MLXArray? = nil) -> MLXArray {
        let f = x.asType(.float32)
        let meanSquares = MLX.mean(f * f, axis: -1, keepDims: true)
        var normed = f * MLX.rsqrt(meanSquares + Float(1e-6))
        if let w { normed = normed * w.asType(.float32) }
        return normed.asType(x.dtype)
    }

    /// `gelu(x, approximate: "tanh")`, evaluated in fp32 and cast back.
    ///
    /// Same reasoning as `DiTModules.geluTanh`: the cubic term is what a bf16 evaluation
    /// destroys, and a feed-forward is where that shows up first.
    private func geluTanh(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let c = Float(0.7978845608028654)                       // sqrt(2/pi)
        let inner = c * (f + Float(0.044715) * f * f * f)
        return (Float(0.5) * f * (1.0 + MLX.tanh(inner))).asType(x.dtype)
    }

    // MARK: - RoPE

    /// The **float64** frequency grid, selected by `frequencies_precision: "float64"` in
    /// the checkpoint config.
    ///
    /// `theta ** linspace(0, 1, width/2) * pi/2`, spanning `pi/2` to `5000pi`. Computed
    /// in `Double` on the CPU because MLX has no float64: the top of the grid lands near
    /// 1.57e4 radians, where the low bits of an fp32 exponentiation are already gone.
    ///
    /// `linspace` is spelled `i * (1 / (n - 1))` with the final sample forced to exactly
    /// `1`, not `i / (n - 1)`, because the two differ in the last bit and the last bit is
    /// the one being multiplied by 10000.
    static func frequencyGrid(width: Int, positionDimensions: Int = 1,
                              theta: Double = positionalTheta) -> [Float] {
        let count = width / (2 * positionDimensions)
        guard count > 1 else { return [Float(Double.pi / 2)] }
        let step = 1.0 / Double(count - 1)
        return (0..<count).map { i in
            let t = i == count - 1 ? 1.0 : Double(i) * step
            return Float(pow(theta, t) * Double.pi / 2)
        }
    }

    /// The split-RoPE tables for this module's 1-D index grid.
    ///
    /// Positions are `arange(seq)` scaled by `max_pos = 4096`, giving an angle argument of
    /// `2p/4096 - 1` — the tokens sit in the first quarter of the range, and the argument
    /// is *negative* throughout. `pad_size` is 0 because one position dimension already
    /// fills `width/2` frequencies, so there is no `cos = 1, sin = 0` prefix to prepend.
    ///
    /// Returned as `[1, heads, seq, headDim/2]` **in `dtype`**: the tables are narrowed
    /// to the hidden dtype before they are applied, not after.
    func ropeTables(sequence: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let grid = MLXArray(Self.frequencyGrid(width: width))       // [width/2], fp32
        let positions = MLXArray(0..<sequence).asType(.float32).reshaped([sequence, 1])
        let fractional = positions / Float(Self.positionalMaxPosition)
        let angles = (fractional * Float(2) - Float(1)) * grid.reshaped([1, width / 2])
        func table(_ v: MLXArray) -> MLXArray {
            v.reshaped([1, sequence, heads, headDim / 2])
                .transposed(0, 2, 1, 3)
                .asType(dtype)
        }
        return (table(MLX.cos(angles)), table(MLX.sin(angles)))
    }

    /// Split rotary: rotate the two contiguous HALVES of each head.
    ///
    /// The head dimension splits into a leading pair of halves, so lane `i` of the first
    /// half pairs with lane `i` of the second — `(a, b) -> (a cos - b sin, b cos +
    /// a sin)`. The interleaved form pairs adjacent elements instead. Same shape, same
    /// norm, different tensor; there is no error to catch it downstream.
    ///
    /// Input and output are `[B, T, heads * headDim]`; the head split happens here so the
    /// caller never has to deal with the unflatten / transpose / flatten dance.
    private func applySplitRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let batch = x.dim(0), tokens = x.dim(1), half = headDim / 2
        let perHead = x.reshaped([batch, tokens, heads, headDim])
            .transposed(0, 2, 1, 3)                                  // [B, H, T, headDim]
        let a = perHead[.ellipsis, 0..<half]
        let b = perHead[.ellipsis, half..<headDim]
        let rotated = MLX.concatenated([a * cos - b * sin, b * cos + a * sin], axis: -1)
        return rotated.transposed(0, 2, 1, 3)
            .reshaped([batch, tokens, self.heads * headDim])
    }

    // MARK: - One block

    /// One pre-norm block, with gated attention.
    ///
    /// ```
    /// n = rms_norm(h)                      <- weightless
    /// h = h + to_out(gate(attn(q, k, v)))  <- q, k get q_norm/k_norm then split RoPE
    /// h = h + ff(rms_norm(h))              <- weightless again
    /// ```
    ///
    /// **The gate reads `n`, not `h`.** Its logits come from the same tensor attention
    /// itself consumes, which at this call site is the normalised hidden states — not the
    /// residual stream. `gates = 2 * sigmoid(to_gate_logits(n))`, one scalar per head per
    /// token, and the factor of 2 means an unlearned gate is the identity rather than a
    /// halving. Feeding it the residual stream instead is a same-shaped, silently
    /// different modulation of every head.
    private func block(_ hidden: MLXArray, index: Int,
                       cos: MLXArray, sin: MLXArray) throws -> MLXArray {
        let stem = "transformer_1d_blocks.\(index)."
        let batch = hidden.dim(0), tokens = hidden.dim(1)

        let normed = rmsNorm(hidden)

        var q = try linear(normed, stem + "attn1.to_q")
        q = rmsNorm(q, weight: try weight(stem + "attn1.q_norm.weight"))
        q = applySplitRoPE(q, cos: cos, sin: sin)

        var k = try linear(normed, stem + "attn1.to_k")
        k = rmsNorm(k, weight: try weight(stem + "attn1.k_norm.weight"))
        k = applySplitRoPE(k, cos: cos, sin: sin)

        let v = try linear(normed, stem + "attn1.to_v")
            .reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3)
        let qh = q.reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3)
        let kh = k.reshaped([batch, tokens, heads, headDim]).transposed(0, 2, 1, 3)

        // **No mask.** The connector zeroed it before the block loop; see the type
        // docstring. The conventional scale `1/sqrt(headDim)` applies — it differs
        // between the streams because their head sizes do (128 video, 64 audio), so it is
        // derived rather than written down.
        //
        // Softmax in fp32: MLX's fused attention kernel accumulates in bf16, and over a
        // 1024-key reduction that loses enough of the tail to matter. `GemmaTextEncoder`
        // makes the same choice for the same reason.
        let scale = Float(1.0 / Double(headDim).squareRoot())
        var logits = MLX.matmul(qh.asType(.float32), kh.asType(.float32)
            .transposed(0, 1, 3, 2)) * scale
        logits = MLX.softmax(logits, axis: -1)
        var attended = MLX.matmul(logits, v.asType(.float32)).asType(hidden.dtype)

        attended = attended.transposed(0, 2, 1, 3)
            .reshaped([batch, tokens, heads, headDim])
        let gateLogits = try linear(normed, stem + "attn1.to_gate_logits")
        let gates = gateSigmoid(gateLogits).reshaped([batch, tokens, heads, 1])
        attended = (attended * gates).reshaped([batch, tokens, width])

        var h = try linear(attended, stem + "attn1.to_out.0") + hidden

        let ff = geluTanh(try linear(rmsNorm(h), stem + "ff.net.0.proj"))
        h = try linear(ff, stem + "ff.net.2") + h
        return h
    }

    /// `2 * sigmoid(x)`, in fp32 and cast back — an elementwise nonlinearity, so the same
    /// accumulate-type rule as `geluTanh` and `DiTModules.silu`.
    private func gateSigmoid(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (2.0 * MLX.sigmoid(f)).asType(x.dtype)
    }

    // MARK: - Forward

    /// Run the connector.
    ///
    /// - Parameters:
    ///   - hidden: `[B, S, width]`, already **right-padded** — the connector requires
    ///     `[valid..., pad...]` and does not sort for you. `EmbeddingsProcessor` is where
    ///     that permutation happens.
    ///   - additiveMask: `[B, 1, 1, S]`, `0` for valid and `-finfo(dtype).max` for pad.
    ///     Used *only* to decide which positions become registers.
    /// - Returns: the encoded stream and the mask that goes with it — all zeros whenever
    ///   registers are enabled, and consumed downstream by `EmbeddingsProcessor` as
    ///   `(mask < 1e-6)`.
    public func callAsFunction(_ hidden: MLXArray, additiveMask: MLXArray) throws
        -> (encoded: MLXArray, mask: MLXArray) {
        let tokens = hidden.dim(1)
        guard tokens % registerCount == 0 else {
            throw Failure.sequenceNotTiled(sequence: tokens, registers: registerCount)
        }

        var h = try substituteRegisters(hidden, additiveMask: additiveMask)
        let tables = ropeTables(sequence: tokens, dtype: h.dtype)
        for i in 0..<layerCount {
            h = try block(h, index: i, cos: tables.cos, sin: tables.sin)
            MLX.eval(h)             // bound peak memory: 8 x [1, 1024, 4096] plus a 32 x
                                    // 1024 x 1024 fp32 score matrix per block
        }
        // `connector_norm_output: true` — one more weightless RMS norm on the way out.
        h = rmsNorm(h)

        return (h, MLXArray.zeros(additiveMask.shape, dtype: additiveMask.dtype))
    }

    /// Overwrite every padded position with a learnable register.
    ///
    /// **The tiling is a repeat, not a repeat-interleave.** The 128-row block is laid
    /// down end to end — row `p` of the sequence gets register `p % 128`.
    /// `MLX.repeated(_:count:axis:)` is the *interleave*: it would give each register 8
    /// consecutive positions, `p / 8`. Both produce `[1024, width]` and both look
    /// plausible; only one is right. Expressed here as a broadcast so the distinction is
    /// structural rather than a function name one has to have checked.
    func substituteRegisters(_ hidden: MLXArray, additiveMask: MLXArray) throws -> MLXArray {
        let batch = hidden.dim(0), tokens = hidden.dim(1)
        let registers = try weight("learnable_registers").asType(hidden.dtype)
        let tiles = tokens / registerCount
        let tiled = MLX.broadcast(registers.reshaped([1, registerCount, width]),
                                  to: [tiles, registerCount, width])
            .reshaped([1, tokens, width])

        // Valid positions are `additiveMask[:, 0, 0, :] >= 0`. A blend written as
        // `b * h + (1 - b) * regs` in the hidden dtype gives the same tensor; with `b`
        // exactly 0 or 1 a `where` gets there without the two multiplies.
        let valid = additiveMask.reshaped([batch, tokens, 1]) .>= MLXArray(Float(0))
        return MLX.where(valid, hidden, tiled)
    }
}

/// The layout work either side of the two connectors.
///
/// Small, but every step of it is a place this can go silently wrong.
public struct EmbeddingsProcessor {

    public let video: EmbeddingsConnector
    public let audio: EmbeddingsConnector?

    /// `finfo(bfloat16).max` — the magnitude the additive mask multiplies by, **not**
    /// `finfo(float32).max`.
    ///
    /// The two are within 0.4% of each other and both are "effectively infinite" to a
    /// softmax, so nothing here can tell them apart today: this value is only ever
    /// emitted into a mask whose validity is read from its *sign*, and the connector
    /// zeroes its own mask before attention. It is spelled correctly anyway, because the
    /// one place the difference bites is a mask that survives into a bf16 tensor —
    /// `finfo(float32).max` is not representable in bf16 and becomes `inf`, and `inf`
    /// times a zero attention weight is `NaN` rather than a masked-out token.
    public static let bf16Max = Float(3.3895313892515355e+38)

    public init(video: EmbeddingsConnector, audio: EmbeddingsConnector?) {
        self.video = video
        self.audio = audio
    }

    /// One checkpoint, both streams.
    ///
    /// The DiT file is 42 GB. Only the `*embeddings_connector*` keys are kept — the
    /// rest of the map is dropped before this returns, so a text-encode does not hold
    /// the transformer for the length of a Gemma forward.
    public init(checkpoint url: URL) throws {
        let all = try MLX.loadArrays(url: url)
        let connector = Dictionary(uniqueKeysWithValues:
            all.filter { $0.key.contains("embeddings_connector") })
        self.video = try EmbeddingsConnector(weights: connector, stream: .video)
        self.audio = try EmbeddingsConnector(weights: connector, stream: .audio)
    }

    /// The permutation that moves valid tokens ahead of padding, stably.
    ///
    /// The connectors require right-padded input and the tokenizer produces **left**-
    /// padded batches — a row with 11 real tokens has them at indices 1013..1023, so this
    /// is emphatically not the identity, and skipping it feeds the connector 1013
    /// registers followed by 11 stranded embeddings.
    ///
    /// Computed on the CPU from the mask rather than with a device argsort. Stability is
    /// load-bearing — valid tokens must keep their relative order — and MLX's `argSort`
    /// makes no such guarantee. The mask is `[B, S]` of 0/1, so a counting pass is exact,
    /// obviously stable, and costs nothing next to the 1.6 B-parameter stack it feeds.
    ///
    /// - Returns: `order` as `[B, S]` gather indices, and the reordered additive mask.
    public static func rightPadOrder(additiveMask: MLXArray)
        -> (order: MLXArray, mask: MLXArray) {
        let batch = additiveMask.dim(0), tokens = additiveMask.dim(-1)
        let flat = additiveMask.reshaped([batch, tokens]).asType(.float32)
            .asArray(Float.self)

        var order = [Int32](repeating: 0, count: batch * tokens)
        var reordered = [Float](repeating: 0, count: batch * tokens)
        let pad = -EmbeddingsProcessor.bf16Max
        for b in 0..<batch {
            var write = 0
            for t in 0..<tokens where flat[b * tokens + t] >= 0 {
                order[b * tokens + write] = Int32(t)
                reordered[b * tokens + write] = 0
                write += 1
            }
            for t in 0..<tokens where flat[b * tokens + t] < 0 {
                order[b * tokens + write] = Int32(t)
                reordered[b * tokens + write] = pad
                write += 1
            }
        }
        return (MLXArray(order, [batch, tokens]),
                MLXArray(reordered, [batch, 1, 1, tokens]).asType(additiveMask.dtype))
    }

    /// Apply a permutation from `rightPadOrder` to a `[B, S, D]` feature tensor.
    public static func reorder(_ features: MLXArray, order: MLXArray) -> MLXArray {
        let indices = MLX.broadcast(order.reshaped([order.dim(0), order.dim(1), 1]),
                                    to: features.shape)
        return MLX.takeAlong(features, indices, axis: 1)
    }

    /// `(connectorMask < 1e-6)` as `[B, S, 1]`.
    ///
    /// Note the direction. The connector's returned mask is **all zeros**, so this is all
    /// **ones**. Reading it as "positions that were valid" gets the polarity backwards
    /// and zeroes the entire video stream, which is a very loud failure; reading it as a
    /// no-op is the quiet one, and is why the multiply below is written out rather than
    /// elided.
    public static func binaryMask(connectorMask: MLXArray) -> MLXArray {
        let batch = connectorMask.dim(0), tokens = connectorMask.dim(-1)
        return (connectorMask.reshaped([batch, tokens, 1]) .< MLXArray(Float(1e-6)))
            .asType(.int32)
    }

    /// Right-pad, run both connectors, and apply the masking asymmetry.
    ///
    /// **Video is multiplied by the binary mask; audio is not.** The asymmetry is
    /// deliberate, not an oversight in a shared loop — audio's returned mask is discarded
    /// outright. Today it costs nothing: the binary mask is all ones because the
    /// connector zeroes its own mask, so the video features come out bit-identical to the
    /// connector's output. It is written out anyway. "Simplifying" it away would look
    /// identical right up until the connector stopped zeroing its mask — exactly the
    /// class of change that ships without a note.
    public func callAsFunction(video videoFeatures: MLXArray,
                               audio audioFeatures: MLXArray?,
                               additiveMask: MLXArray) throws
        -> (video: MLXArray, audio: MLXArray?, mask: MLXArray) {
        let (order, mask) = Self.rightPadOrder(additiveMask: additiveMask)

        let (videoEncoded, videoMask) = try video(Self.reorder(videoFeatures, order: order),
                                                  additiveMask: mask)
        let binary = Self.binaryMask(connectorMask: videoMask)
        let videoOut = videoEncoded * binary.asType(videoEncoded.dtype)

        var audioOut: MLXArray?
        if let audio, let audioFeatures {
            // The projection keeps fp32 for its own output, but the audio residual
            // stream is bf16 from the connector boundary onward. See the matching seam
            // in `TextConditioningPipeline.connect`, which has to stay identical to
            // this one.
            audioOut = try audio(Self.reorder(audioFeatures.asType(.bfloat16), order: order),
                                 additiveMask: mask).encoded
        }
        return (videoOut, audioOut, binary.reshaped([binary.dim(0), binary.dim(1)]))
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// One `BasicAVTransformerBlock`: two streams, six attentions, and the modulation that
/// wires them together.
///
/// `DiTModules` and `DiTAttention` each hold one piece in isolation. This is where the
/// pieces stop being independent — the block is the first thing whose output depends on
/// getting the *order* right, and almost every way to get it wrong produces a correctly
/// shaped tensor.
///
/// ## The shape of the computation
///
/// ```
/// video:  norm -> attn1 -> gate -> renorm -> attn2(text) -> +
///                                              |
/// audio:  norm -> audio_attn1 -> gate -> renorm -> audio_attn2(text) -> +
///                                              |
/// cross:  a2v (video queries audio) and v2a (audio queries video), BOTH from the
///         pre-cross snapshot of the other stream
///                                              |
/// video:  norm -> ff -> gate -> +          audio: norm -> audio_ff -> gate -> +
/// ```
///
/// ## Four decisions that leave the shapes intact
///
/// **The cross-modal pair reads a snapshot.** `vx_pre_av` and `ax_pre_av` are captured
/// before either direction runs, so `v2a` sees the video stream as it was *before* `a2v`
/// added to it. Using the live `vx` would make the result depend on which direction was
/// evaluated first, and the two orders differ by roughly the size of one cross-attention
/// output. Nothing about the shapes changes.
///
/// **The modulation triples unpack shift before scale — except where they do not.** The
/// self-attention and feed-forward triples unpack as `(shift, scale, gate)`. The
/// cross-modal pairs unpack as `(scale, shift)`, the other way round. Both are
/// `[B, T, dim]` and swapping them is arithmetic, not a crash.
///
/// **The modulation index ranges are not contiguous per use.** The 9-row table is sliced
/// `0..<3` for self-attention, `3..<6` for the feed-forward, and `6..<9` for the text
/// cross-attention — but the *cross-modal* modulation comes from two entirely different
/// 5-row tables, and the prompt K/V modulation from a third 2-row one. Reading nine rows
/// and spending them in order runs out at the wrong place.
///
/// **The A2V gate is the video table's row 4 and the V2A gate is the audio table's.**
/// Each direction's gate is indexed out of the table belonging to the stream it *writes*,
/// while the scale and shift for the same call come from both tables. They are different
/// widths (4096 and 2048), so this one at least fails loudly.
public struct DiTBlock {

    public var attention: DiTAttention
    public var modules: DiTModules
    public let topology: TransformerTopology
    public let prefix: String
    public let weights: [String: MLXArray]

    /// Overlay `lora` on every adapted projection in this block.
    ///
    /// Both sub-modules, because an adapter's per-block tensors are split across them:
    /// `attn1`/`attn2` `to_{q,k,v,out.0}` live in ``DiTAttention`` and `ff.net.{0.proj,2}`
    /// in ``DiTModules``. The pixel spatial upscaler carries all ten per block, so
    /// attaching to one and not the other silently applies 20% of the adapter.
    public mutating func attach(lora: LoRAOverlay?) {
        attention.lora = lora
        modules.lora = lora
    }

    /// `norm_eps` from the model config, shared by every norm in the block.
    public static let normEpsilon = Float(1e-6)

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            }
        }
    }

    public init(checkpoint url: URL, topology: TransformerTopology,
                attentionPath: DiTAttention.AttentionPath = .explicit) throws {
        let weights = try MLX.loadArrays(url: url)
        self.weights = weights
        self.topology = topology
        self.prefix = topology.prefix
        self.attention = DiTAttention(weights: weights, topology: topology,
                                      attentionPath: attentionPath)
        self.modules = DiTModules(weights: weights, topology: topology)
    }

    /// Share an already-loaded weight map rather than paying for a second 21 B-parameter
    /// dictionary.
    public init(weights: [String: MLXArray], topology: TransformerTopology,
                attentionPath: DiTAttention.AttentionPath = .explicit) {
        self.weights = weights
        self.topology = topology
        self.prefix = topology.prefix
        self.attention = DiTAttention(weights: weights, topology: topology,
                                      attentionPath: attentionPath)
        self.modules = DiTModules(weights: weights, topology: topology)
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[prefix + name] else { throw Failure.missing(prefix + name) }
        return w
    }

    /// The dtype the stream computes in, read from a projection weight.
    ///
    /// Deliberately **not** read from a modulation table: those are the checkpoint's only
    /// fp32 tensors, so `scale_shift_table.dtype` would report fp32 and every caller that
    /// trusted it would run the block at the wrong precision.
    public func computeDType(block: Int) throws -> DType {
        try weight("transformer_blocks.\(block).ff.net.0.proj.weight").dtype
    }

    // MARK: - Per-stream inputs

    /// Everything one stream carries into a block.
    ///
    /// It is a long list, and that is the point: a block reads far more than its latent.
    /// The timesteps come from the AdaLN modules' outputs, the context from the text
    /// encoder, and the rotary embeddings from the position grids — the latent alone is
    /// about a tenth of what a block is handed.
    public struct Conditioning {
        /// The latent stream, `[B, T, dim]`.
        public var x: MLXArray
        /// `adaln_single.out` reshaped to `[B, T, 9 * dim]`.
        ///
        /// The module emits `[T, 9 * dim]` — the batch folded away — and ``adaValues``
        /// reshapes on axis 1, so passing it through unreshaped silently reinterprets
        /// `9 * dim` as the token count.
        public var timesteps: MLXArray
        /// The text conditioning this stream cross-attends to.
        public var context: MLXArray
        /// `prompt_adaln_single.out` as `[B, 1, 2 * dim]`, or nil.
        ///
        /// Nil means `use_prompt_adaln_single=False`, where the K/V modulation is the
        /// static per-block table alone and is therefore timestep-independent. On this
        /// checkpoint the module exists and this is not nil.
        public var promptTimestep: MLXArray?
        /// `av_ca_<stream>_scale_shift_adaln_single.out` as `[B, T, 4 * dim]`.
        public var crossScaleShiftTimestep: MLXArray
        /// The gate for the cross-modal direction that **writes this stream**:
        /// `av_ca_a2v_gate_adaln_single.out` for video, `av_ca_v2a_gate_...` for audio,
        /// as `[B, 1, dim]`.
        public var crossGateTimestep: MLXArray
        /// `pe` for this stream's self-attention.
        public var rotary: DiTAttention.Rotary
        /// `SKIP_VIDEO_SELF_ATTN` / `SKIP_AUDIO_SELF_ATTN` for this block and stream.
        /// When true the attention keeps `to_v`, optional per-head gating and `to_out`,
        /// while bypassing q/k projection, their norms, RoPE and SDPA.
        public var selfAttentionAllPerturbed: Bool
        /// `cross_positional_embeddings` — temporal-only, and computed at the **audio**
        /// width for both streams. Not the same tensor as `rotary`.
        public var crossRotary: DiTAttention.Rotary
        /// `cross_attn_perturbation_mask`, applied to this stream's cross-modal result.
        ///
        /// Nil is the unperturbed path and multiplies by nothing at all rather than by a
        /// ones tensor — STG perturbs a *later* guidance branch, so on the unperturbed
        /// branch there is no mask to apply and materialising one would only invite a
        /// broadcast bug.
        public var crossPerturbationMask: MLXArray?

        public init(x: MLXArray, timesteps: MLXArray, context: MLXArray,
                    promptTimestep: MLXArray?, crossScaleShiftTimestep: MLXArray,
                    crossGateTimestep: MLXArray, rotary: DiTAttention.Rotary,
                    crossRotary: DiTAttention.Rotary,
                    selfAttentionAllPerturbed: Bool = false,
                    crossPerturbationMask: MLXArray? = nil) {
            self.x = x
            self.timesteps = timesteps
            self.context = context
            self.promptTimestep = promptTimestep
            self.crossScaleShiftTimestep = crossScaleShiftTimestep
            self.crossGateTimestep = crossGateTimestep
            self.rotary = rotary
            self.crossRotary = crossRotary
            self.selfAttentionAllPerturbed = selfAttentionAllPerturbed
            self.crossPerturbationMask = crossPerturbationMask
        }

        /// Every tensor at the stream's compute dtype. See `callAsFunction`.
        func cast(to dtype: DType) -> Conditioning {
            var c = self
            c.x = x.asType(dtype)
            c.timesteps = timesteps.asType(dtype)
            c.context = context.asType(dtype)
            c.promptTimestep = promptTimestep?.asType(dtype)
            c.crossScaleShiftTimestep = crossScaleShiftTimestep.asType(dtype)
            c.crossGateTimestep = crossGateTimestep.asType(dtype)
            c.rotary = DiTAttention.Rotary(cos: rotary.cos.asType(dtype),
                                           sin: rotary.sin.asType(dtype))
            c.crossRotary = DiTAttention.Rotary(cos: crossRotary.cos.asType(dtype),
                                                sin: crossRotary.sin.asType(dtype))
            c.crossPerturbationMask = crossPerturbationMask?.asType(dtype)
            return c
        }
    }

    // MARK: - Primitives

    /// RMS norm with **no weight**.
    ///
    /// Every norm inside the block is unparameterised; the only learned norms here are
    /// `q_norm` / `k_norm` inside the attentions. Code that reaches for a weight at this
    /// point finds none in the checkpoint and, if it defaults to ones, is accidentally
    /// correct — which is worse than failing, because the same code is wrong the moment it
    /// is pointed at a model that does carry one.
    ///
    /// The statistic is accumulated in fp32 inside `rms_norm.metal` (one of the
    /// nine kernels `tools/build_mlx_metallib.sh` compiles). A decomposed
    /// mean-square graph is the same numbers as a pile of JIT launches.
    public func rmsNorm(_ x: MLXArray) -> MLXArray {
        MLX.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype),
                    eps: Self.normEpsilon)
    }

    /// A slice of the static table plus the matching slice of the timestep projection,
    /// unbound along the parameter axis.
    ///
    /// **`num` comes from the table passed in, not from the block's full table.**
    /// ``avCrossAdaValues`` calls this twice on *slices* of a 5-row table — once with the
    /// first four rows and once with the last one — so the same timestep tensor is
    /// reshaped as 4 parameters in one call and 1 in the other. Hardcoding 9 (or 5) makes
    /// both wrong while keeping every shape plausible.
    ///
    /// - Parameters:
    ///   - table: the already-sliced static table, `[num, dim]`.
    ///   - timestep: `[B, T, num * dim]`.
    ///   - indices: which of the `num` rows to return.
    func adaValues(table: MLXArray, timestep: MLXArray,
                           _ indices: Range<Int>) -> [MLXArray] {
        let num = table.dim(0), dim = table.dim(1)
        let batch = timestep.dim(0), tokens = timestep.dim(1)
        let projected = timestep.reshaped([batch, tokens, num, dim])
        // The table is DOWNCAST to the stream's dtype before the add, and on this
        // checkpoint that is a real conversion rather than a formality. The 290 modulation
        // tables are the only **fp32** tensors in an otherwise bf16 file (4059 bf16 to 290
        // fp32, and the 290 are exactly the modulation tables), so letting MLX promote the
        // sum to fp32 would compute every modulation at the wrong precision.
        let t = table.asType(timestep.dtype)
        return indices.map { i in
            t[i].reshaped([1, 1, dim]) + projected[0..., 0..., i, 0...]
        }
    }

    /// `rmsNorm(x) * (1 + scale) + shift`.
    ///
    /// The `1 +` is not decoration. The AdaLN projection is zero-initialised at training
    /// time so an untrained block is the identity, which only works because the scale is
    /// an *offset* from one. Dropping it multiplies the normed stream by a number near
    /// zero and the block emits almost pure `shift`.
    func adaZero(_ x: MLXArray, scale: MLXArray, shift: MLXArray) -> MLXArray {
        let normed = rmsNorm(x)
        return normed * (1.0 + scale).asType(normed.dtype) + shift.asType(normed.dtype)
    }

    // MARK: - Forward

    /// Called with each intermediate's **suffix** and the tensor at the point it is
    /// produced.
    ///
    /// The suffix names a position inside the block — `in.kw_video`,
    /// `attn1.in.kw_pe.el1`, `video_to_audio_attn.in.kw_context` — so a caller prefixes
    /// its own block name and gets a stable key.
    public typealias Observer = (String, MLXArray) throws -> Void

    /// One block, both streams. Video first, audio second.
    ///
    /// - Parameter crossModalSnapshot: read the two cross-modal directions from the
    ///   pre-cross snapshot. Passing `false` runs V2A against the video stream A2V has
    ///   already written to — the plausible wrong version, kept as a parameter so a test
    ///   can *measure* that the snapshot matters instead of asserting that it does. Never
    ///   pass `false` in production; there is no configuration in which it is correct.
    /// - Parameter observer: receives the block's 26 intermediates as they are produced.
    ///   Nil — the default, and what `DiTForward` passes — costs nothing: every emission
    ///   is behind the optional, so no tensor is retained and no evaluation is forced.
    ///   Non-nil holds a reference to every intermediate until the observer lets go, which
    ///   is why the forward stack does not pass one.
    ///
    /// ## Why inputs are emitted here rather than reconstructed afterwards
    ///
    /// Each `observer` call below sits immediately before the call that consumes the
    /// tensor and passes the identical binding. Rebuilding an input afterwards from the
    /// same expression that produced it agrees with it by construction, so it stops being
    /// an independent observation of what the block was actually handed.
    ///
    /// The two returned latents are deliberately **not** emitted. They are the return
    /// value; routing them through the observer as well would give a caller two spellings
    /// of one tensor and an opportunity to disagree with itself.
    public func callAsFunction(block: Int, video: Conditioning, audio: Conditioning,
                               crossModalSnapshot: Bool = true,
                               observer: Observer? = nil,
                               outputObserver: Observer? = nil)
        throws -> (video: MLXArray, audio: MLXArray) {
        let stem = "transformer_blocks.\(block)."
        let videoTable = try weight(stem + "scale_shift_table")
        let audioTable = try weight(stem + "audio_scale_shift_table")
        let a2vVideoTable = try weight(stem + "scale_shift_table_a2v_ca_video")
        let a2vAudioTable = try weight(stem + "scale_shift_table_a2v_ca_audio")

        // Everything the block was handed is cast to the stream's own dtype first.
        //
        // A caller that assembles its conditioning in fp32 would otherwise run the block
        // at the wrong precision *everywhere at once*. `DiTModules.linear` already pins
        // matmuls to the weight's dtype, but a block is mostly elementwise — norms,
        // gates, `(1 + scale)` — and none of that goes near a weight. Left alone, the
        // whole residual stream would run in fp32, which is `FRAGILE_CONTRACTS.md`'s
        // "more accurate is not matching" applied to a hundred operations rather than
        // one.
        //
        // The cast is lossless in the direction that matters: a value that already holds
        // bf16 loses nothing on the way down.
        let dtype = try computeDType(block: block)
        let video = video.cast(to: dtype), audio = audio.cast(to: dtype)

        /// Both halves of a rotary pair: the cosine under `tap`, the sine under
        /// `tap.el1`.
        ///
        /// The sine is emitted separately because it is not recoverable from the cosine,
        /// which is even in the angle. Emitting only the cosine would look complete and
        /// leave the sine's sign unobserved.
        func emitRotary(_ tap: String, _ pe: DiTAttention.Rotary) throws {
            try observer?(tap, pe.cos)
            try observer?(tap + ".el1", pe.sin)
        }

        // Each stream's latent as the block will actually use it. Emitted after the dtype
        // cast rather than before it: pre-cast would report whatever precision the
        // *caller* happened to hand in rather than the one the block computes at.
        try observer?("in.kw_video", video.x)
        try observer?("in.kw_audio", audio.x)

        // ---- video self-attention -------------------------------------------------
        var vx = video.x
        let vmsa = adaValues(table: videoTable, timestep: video.timesteps, 0..<3)
        let normVX = adaZero(vx, scale: vmsa[1], shift: vmsa[0])
        try observer?("attn1.in", normVX)
        try emitRotary("attn1.in.kw_pe", video.rotary)
        let vmsaOut = try attention(normVX, rotary: video.rotary,
                                    block: block, variant: .videoSelf,
                                    allPerturbed: video.selfAttentionAllPerturbed)
        try outputObserver?("attn1.out", vmsaOut)
        // The self-attention stage produces BOTH the updated residual and a fresh norm of
        // it. The text cross-attention consumes the normed copy and the residual add
        // consumes the un-normed one; using either for both is a silent divergence.
        vx = vx + vmsaOut * vmsa[2].asType(vmsaOut.dtype)
        let vxNormed = rmsNorm(vx)

        // ---- audio self-attention -------------------------------------------------
        var ax = audio.x
        let amsa = adaValues(table: audioTable, timestep: audio.timesteps, 0..<3)
        let normAX = adaZero(ax, scale: amsa[1], shift: amsa[0])
        try observer?("audio_attn1.in", normAX)
        try emitRotary("audio_attn1.in.kw_pe", audio.rotary)
        let amsaOut = try attention(normAX, rotary: audio.rotary,
                                    block: block, variant: .audioSelf,
                                    allPerturbed: audio.selfAttentionAllPerturbed)
        try outputObserver?("audio_attn1.out", amsaOut)
        ax = ax + amsaOut * amsa[2].asType(amsaOut.dtype)
        let axNormed = rmsNorm(ax)

        // ---- text cross-attention -------------------------------------------------
        vx = vx + (try textCrossAttention(
            xNormed: vxNormed, conditioning: video, table: videoTable,
            promptTable: try weight(stem + "prompt_scale_shift_table"),
            block: block, variant: .videoCross, observer: observer,
            outputObserver: outputObserver))
        ax = ax + (try textCrossAttention(
            xNormed: axNormed, conditioning: audio, table: audioTable,
            promptTable: try weight(stem + "audio_prompt_scale_shift_table"),
            block: block, variant: .audioCross, observer: observer,
            outputObserver: outputObserver))

        // ---- cross-modal ----------------------------------------------------------
        //
        // Snapshot first. `v2a` must query the video stream as it stood before `a2v`
        // wrote to it, or the result depends on evaluation order.
        let vxPreAV = vx, axPreAV = ax

        // A2V: video queries audio. Scale/shift for the query come from the video table,
        // for the key/value from the audio table; the gate is the video table's row 4.
        let (a2vVideoScale, a2vVideoShift, a2vGate) = avCrossAdaValues(
            table: a2vVideoTable, scaleShiftTimestep: video.crossScaleShiftTimestep,
            gateTimestep: video.crossGateTimestep, 0..<2)
        let (a2vAudioScale, a2vAudioShift, _) = avCrossAdaValues(
            table: a2vAudioTable, scaleShiftTimestep: audio.crossScaleShiftTimestep,
            gateTimestep: audio.crossGateTimestep, 0..<2)
        let a2vQuery = adaZero(vxPreAV, scale: a2vVideoScale, shift: a2vVideoShift)
        let a2vContext = adaZero(axPreAV, scale: a2vAudioScale, shift: a2vAudioShift)
        try observer?("audio_to_video_attn.in", a2vQuery)
        try observer?("audio_to_video_attn.in.kw_context", a2vContext)
        // **`pe` is this direction's query grid and `k_pe` the other stream's.** The two
        // directions share both grids, crossed over, so A2V's `k_pe` is exactly V2A's
        // `pe`: swapping the pair emits four tensors of the right shape carrying the wrong
        // positions, and only the naming distinguishes them.
        try emitRotary("audio_to_video_attn.in.kw_pe", video.crossRotary)
        try emitRotary("audio_to_video_attn.in.kw_k_pe", audio.crossRotary)
        var a2v = try attention(
            a2vQuery, context: a2vContext,
            rotary: video.crossRotary, keyRotary: audio.crossRotary,
            block: block, variant: .audioToVideo)
        try outputObserver?("audio_to_video_attn.out", a2v)
        a2v = a2v * a2vGate.asType(a2v.dtype)
        if let mask = video.crossPerturbationMask { a2v = a2v * mask.asType(a2v.dtype) }
        vx = vx + a2v

        // V2A: audio queries video, from rows 2..<4 of the SAME two tables.
        let (v2aAudioScale, v2aAudioShift, v2aGate) = avCrossAdaValues(
            table: a2vAudioTable, scaleShiftTimestep: audio.crossScaleShiftTimestep,
            gateTimestep: audio.crossGateTimestep, 2..<4)
        let (v2aVideoScale, v2aVideoShift, _) = avCrossAdaValues(
            table: a2vVideoTable, scaleShiftTimestep: video.crossScaleShiftTimestep,
            gateTimestep: video.crossGateTimestep, 2..<4)
        let v2aQuery = adaZero(axPreAV, scale: v2aAudioScale, shift: v2aAudioShift)
        let v2aContext = adaZero(crossModalSnapshot ? vxPreAV : vx,
                                 scale: v2aVideoScale, shift: v2aVideoShift)
        try observer?("video_to_audio_attn.in", v2aQuery)
        try observer?("video_to_audio_attn.in.kw_context", v2aContext)
        try emitRotary("video_to_audio_attn.in.kw_pe", audio.crossRotary)
        try emitRotary("video_to_audio_attn.in.kw_k_pe", video.crossRotary)
        var v2a = try attention(
            v2aQuery, context: v2aContext,
            rotary: audio.crossRotary, keyRotary: video.crossRotary,
            block: block, variant: .videoToAudio)
        try outputObserver?("video_to_audio_attn.out", v2a)
        v2a = v2a * v2aGate.asType(v2a.dtype)
        if let mask = audio.crossPerturbationMask { v2a = v2a * mask.asType(v2a.dtype) }
        ax = ax + v2a

        // ---- feed-forward ---------------------------------------------------------
        let vmlp = adaValues(table: videoTable, timestep: video.timesteps, 3..<6)
        let vffIn = adaZero(vx, scale: vmlp[1], shift: vmlp[0])
        try observer?("ff.in", vffIn)
        let vff = try modules.feedForward(vffIn, block: block, stream: .video)
        try outputObserver?("ff.out", vff)
        vx = vx + vff * vmlp[2].asType(vff.dtype)

        let amlp = adaValues(table: audioTable, timestep: audio.timesteps, 3..<6)
        let affIn = adaZero(ax, scale: amlp[1], shift: amlp[0])
        try observer?("audio_ff.in", affIn)
        let aff = try modules.feedForward(affIn, block: block, stream: .audio)
        try outputObserver?("audio_ff.out", aff)
        ax = ax + aff * amlp[2].asType(aff.dtype)

        return (vx, ax)
    }

    /// Text cross-attention with `cross_attention_adaln` enabled.
    ///
    /// Two modulations, from two different tables:
    ///
    /// - the **query** side from rows `6..<9` of the block's own 9-row table, unpacked as
    ///   `(shift, scale, gate)`;
    /// - the **key/value** side from the 2-row prompt table plus, when the prompt AdaLN
    ///   module exists, its timestep projection.
    ///
    /// `xNormed` arrives already RMS-normalised from the self-attention stage, so this
    /// applies the affine only. Normalising again here is a plausible-looking extra step
    /// that changes the result and nothing else.
    private func textCrossAttention(xNormed: MLXArray, conditioning: Conditioning,
                                    table: MLXArray, promptTable: MLXArray,
                                    block: Int,
                                    variant: DiTAttention.Variant,
                                    observer: Observer? = nil,
                                    outputObserver: Observer? = nil) throws -> MLXArray {
        let q = adaValues(table: table, timestep: conditioning.timesteps, 6..<9)
        let qShift = q[0], qScale = q[1], qGate = q[2]

        let dim = promptTable.dim(1)
        // Downcast to the stream's dtype, same as `adaValues`: the prompt table is one of
        // the checkpoint's fp32 tensors.
        let table = promptTable.asType(xNormed.dtype)
        var shiftKV = table[0].reshaped([1, 1, dim])
        var scaleKV = table[1].reshaped([1, 1, dim])
        if let prompt = conditioning.promptTimestep {
            let projected = prompt.reshaped([prompt.dim(0), prompt.dim(1), 2, dim])
            shiftKV = shiftKV + projected[0..., 0..., 0, 0...]
            scaleKV = scaleKV + projected[0..., 0..., 1, 0...]
        }

        let input = xNormed * (1.0 + qScale).asType(xNormed.dtype)
            + qShift.asType(xNormed.dtype)
        let context = conditioning.context
            * (1.0 + scaleKV).asType(conditioning.context.dtype)
            + shiftKV.asType(conditioning.context.dtype)
        // The context emitted here is the one **after** this affine, not the raw encoder
        // features. The two are the same shape, so forwarding the unmodulated features
        // instead is a wiring error nothing structural catches.
        //
        // `variant.rawValue` is the attention's own module name (`attn2`, `audio_attn2`)
        // — taken from the enum rather than passed in, so the emitted name cannot drift
        // from the attention the tensor was actually handed to.
        try observer?(variant.rawValue + ".in", input)
        try observer?(variant.rawValue + ".in.kw_context", context)
        let out = try attention(input, context: context, block: block, variant: variant)
        try outputObserver?(variant.rawValue + ".out", out)
        return out * qGate.asType(out.dtype)
    }

    /// The cross-modal modulation: `(scale, shift, gate)` — **scale first**.
    ///
    /// The scale/shift pair comes from the first four rows of the 5-row table indexed by
    /// `indices` (`0..<2` for A2V, `2..<4` for V2A), reshaping the timestep as 4
    /// parameters. The gate comes from row 4 alone, reshaping a *different* timestep
    /// tensor as 1 parameter. Two tables' worth of bookkeeping in one call, and the
    /// returned order is the reverse of the `(shift, scale, gate)` used everywhere else
    /// in this block.
    private func avCrossAdaValues(table: MLXArray, scaleShiftTimestep: MLXArray,
                                  gateTimestep: MLXArray, _ indices: Range<Int>)
        -> (scale: MLXArray, shift: MLXArray, gate: MLXArray) {
        let scaleShift = adaValues(table: table[0..<4], timestep: scaleShiftTimestep,
                                   indices)
        let gate = adaValues(table: table[4...], timestep: gateTimestep, 0..<1)
        return (scaleShift[0], scaleShift[1], gate[0])
    }
}

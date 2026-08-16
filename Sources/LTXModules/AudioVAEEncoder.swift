// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The audio VAE **encoder**: log-mel spectrogram → audio latent.
///
/// This is the front door for audio-to-video conditioning, and it is the exact
/// counterpart of ``AudioVAEDecoder`` — not its reverse. The two are read from the same
/// `ddconfig`, share `per_channel_statistics`, and meet at the same tensor; everything
/// between differs, and several of the differences are same-shape/different-value traps
/// that a "run the decoder backwards" port would sail straight past.
///
/// ## What this consumes
///
/// **A stereo log-mel spectrogram `[B, 2, melFrames, 64]`, not a waveform.** The
/// checkpoint settles it without being asked: `conv_in.conv.weight` is `[128, 2, 3, 3]`,
/// so the first convolution accepts exactly `in_channels: 2` planes. A waveform has no
/// second spatial axis to convolve over at all.
///
/// The waveform→mel step lives *outside* the encoder, in ``AudioMelProcessor``. That is
/// why this type takes a tensor and never a file path: the module boundary is at the mel.
///
/// **The mel is not the vocoder's mel.** This matters because ``AudioVocoder`` also owns
/// a mel transform and it would be a natural thing to reach for. They are different
/// transforms with different parameters and their outputs are not interchangeable:
///
/// | | ``AudioMelProcessor`` (the encoder's) | the vocoder's BWE analysis mel |
/// |---|---|---|
/// | n_fft / win | 1024 | 512 |
/// | hop | 160 | 80 |
/// | frames for 4.04 s @ 16 kHz | 401 | 802 |
///
/// The encoder's transform is a mel spectrogram at `sample_rate: 16000, n_fft: 1024,
/// win_length: 1024, hop_length: 160, f_min: 0, f_max: 8000, n_mels: 64, window: hann
/// (periodic), center: true, pad_mode: "reflect", power: 1.0, mel_scale: "slaney",
/// norm: "slaney"`, followed by `log(clamp(mel, min: 1e-5))` and a transpose that puts
/// time before frequency. **`power: 1.0` is magnitude, not power** — a `power: 2.0` mel
/// is the same shape and every value doubled after the log, which would look like a
/// scale bug thirty layers downstream.
///
/// That transform is **not implemented here**, and this type is not the place for it: it
/// is a signal-processing stage with no weights in this checkpoint. Feeding this encoder
/// is the caller's job, and ``AudioMelProcessor`` is what does it.
///
/// ## How stereo becomes eight channels
///
/// Contract 9b says the mel is stereo and the latent is 8-channel, and neither follows
/// from the other. The bridge is that **the two ears stop being an axis at `conv_in`**:
/// `[B, 2, T, F] → [B, 128, T, F]` mixes both ears into `ch` feature channels in the
/// very first convolution and nothing downstream ever separates them again. The latent's
/// 8 channels come from `z_channels` at the far end, via `conv_out`. There is no
/// per-ear latent and no channel of the latent that "is" the left ear — a port that
/// tried to encode each ear separately and stack the results would produce a correctly
/// shaped latent from entirely the wrong arithmetic.
///
/// ## The shape ladder
///
/// ```
/// mel               [1,   2, 401, 64]
/// conv_in           [1, 128, 401, 64]
/// down[0] 2 x resnet[1, 128, 401, 64] -> downsample -> [1, 128, 201, 32]
/// down[1] 2 x resnet[1, 256, 201, 32] -> downsample -> [1, 256, 101, 16]
/// down[2] 2 x resnet[1, 512, 101, 16]     no downsample at the last level
/// mid  2 x resnet   [1, 512, 101, 16]     no attention between them
/// conv_out          [1,  16, 101, 16]     16 = 2 x z_channels, double_z is true
/// chunk()[0]        [1,   8, 101, 16]     the mean half; the other half is discarded
/// ```
///
/// Two facts worth stating because they are not symmetric with the decoder:
///
/// * The **downsampling path runs `num_res_blocks` blocks per level, not
///   `num_res_blocks + 1`.** The decoder's `+1` is a decoder-only asymmetry. Two here,
///   three there, and the checkpoint has `down.L.block.{0,1}` and no `block.2`.
/// * The level **without** a resampler is the *last* one on the way down and the
///   *first* one on the way up. `down.2` has no `downsample`; `up.0` has no `upsample`.
///
/// ## Layout
///
/// MLX convolves `NHWC` with weights `[out, kh, kw, in]`; the checkpoint stores `NCHW`
/// with `[out, in, kh, kw]`. Same discipline as the sibling: the transposes
/// happen at the two ends of ``encode(_:)`` and once per kernel, **the channel axis is
/// `-1` inside this type and `1` at its boundary**, and `PixelNorm` reduces over
/// channels. Getting that axis wrong normalises over frequency instead — same shape,
/// different network, no error.
public struct AudioVAEEncoder {

    /// `ddconfig`, read from the checkpoint rather than declared.
    ///
    /// ### Why this is not ``AudioVAEDecoder/Configuration``
    ///
    /// It was checked first, and it is two fields short of sufficient — both of them
    /// load-bearing, and neither derivable from what it does carry:
    ///
    /// * **`in_channels`.** The decoder carries `out_ch`. On this checkpoint both are
    ///   `2`, which is exactly why substituting one for the other would work here and
    ///   fail silently on a mono checkpoint.
    /// * **`double_z`.** This decides whether `conv_out` emits `2 * z_channels` or
    ///   `z_channels`, i.e. whether the distribution head exists at all. A decoder has
    ///   no opinion on it, so the field is absent there.
    ///
    /// Every field is read from the checkpoint, never declared: each has a plausible
    /// wrong value that type-checks and loads.
    public struct Configuration: Equatable, Sendable {
        public let ch: Int
        public let chMult: [Int]
        /// `num_res_blocks`. The **downsampling** path runs exactly this many blocks per
        /// level — no `+1`. See the type's ladder.
        public let numResBlocks: Int
        public let zChannels: Int
        /// `in_channels`: the mel's channel count, `2` (stereo) here.
        public let inChannels: Int
        public let melBins: Int
        /// `double_z`. True on this checkpoint, so `conv_out` emits `2 * zChannels`.
        public let doubleZ: Bool
        /// 4. The latent is 4× coarser than the mel in time, before causal trimming.
        public let latentDownsampleFactor: Int

        public var numResolutions: Int { chMult.count }

        /// Channels `conv_out` emits: `2 * zChannels` under `double_z`, else `zChannels`.
        public var headChannels: Int { doubleZ ? 2 * zChannels : zChannels }

        /// Channels entering `mid` and leaving the downsampling path.
        public var midChannels: Int { ch * (chMult.last ?? 1) }

        /// Latent frames for a mel of `melFrames`, derived **geometrically** — the exact
        /// output length of each strided downsample, composed.
        ///
        /// One `Downsample` is `pad(top: 2, bottom: 0)` then a 3×3 stride-2 convolution,
        /// so `H → floor((H + 2 − 3) / 2) + 1 = floor((H − 1) / 2) + 1`.
        ///
        /// Cross-checked against ``latentFramesArithmetic(melFrames:)`` in
        /// ``encode(_:)``. The two are independent derivations and must agree.
        public func latentFrames(melFrames: Int) -> Int {
            var frames = melFrames
            for _ in 0 ..< max(numResolutions - 1, 0) {
                frames = (frames - 1) / 2 + 1
            }
            return max(frames, 1)
        }

        /// The same number derived from the latent factor alone: `(M − 1) / factor + 1`.
        ///
        /// This is the algebraic inverse of ``melFrames(latentFrames:)``. It happens to
        /// agree with the geometric derivation for **two** stride-2 downsamples and a
        /// factor of 4; on a checkpoint
        /// where `len(ch_mult) − 1` and `log2(factor)` disagreed it would not, and
        /// ``encode(_:)`` refuses rather than picking one.
        public func latentFramesArithmetic(melFrames: Int) -> Int {
            max((melFrames - 1) / latentDownsampleFactor + 1, 1)
        }

        /// Mel frames the decoder produces from `latentFrames`: `4T − 3`, floored at 1.
        ///
        /// Identical to ``AudioVAEDecoder/Configuration/targetFrames(latentFrames:)`` and
        /// deliberately restated rather than borrowed, so that the encoder's shape
        /// algebra is checkable without constructing a decoder. The round-trip tests
        /// assert the two agree.
        public func melFrames(latentFrames: Int) -> Int {
            max(latentFrames * latentDownsampleFactor - (latentDownsampleFactor - 1), 1)
        }

        /// Latent columns for a mel of `bins`, again geometrically.
        ///
        /// The frequency axis is padded `(left: 0, right: 1)` by `Downsample`, so
        /// `W → floor((W + 1 − 3) / 2) + 1 = floor((W − 2) / 2) + 1`. For 64 that is
        /// 32 then 16, which is where the latent's 16 columns come from — **not** from
        /// anything about `z_channels`.
        public func latentColumns(melBins bins: Int) -> Int {
            var columns = bins
            for _ in 0 ..< max(numResolutions - 1, 0) {
                columns = (columns - 2) / 2 + 1
            }
            return max(columns, 1)
        }

        /// Latent columns for this checkpoint's own `melBins`. 16 here.
        public var latentColumns: Int { latentColumns(melBins: melBins) }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case unexpected(String)
        case shape(String, got: [Int], expected: String)
        case unsupported(String)
        case configuration(String)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .unexpected(name):
                return "checkpoint carries \(name), which this port never reads"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .unsupported(why): return why
            case let .configuration(why): return "checkpoint config: \(why)"
            }
        }
    }

    /// Encoder tensors, keys stripped of `audio_vae.encoder.`.
    public let weights: [String: MLXArray]
    /// `per_channel_statistics.std-of-means`, `[ch]` = `[128]`.
    public let stdOfMeans: MLXArray
    /// `per_channel_statistics.mean-of-means`, `[ch]` = `[128]`.
    public let meanOfMeans: MLXArray
    public let config: Configuration

    public static let encoderPrefix = "audio_vae.encoder."
    public static let statisticsPrefix = "audio_vae.per_channel_statistics."

    /// Load the encoder half of a file that also contains the decoder and the vocoder.
    ///
    /// 1329 tensors, of which 44 are the encoder and 1227 the vocoder. Two prefixes
    /// separate them — `audio_vae.encoder.` and `audio_vae.per_channel_statistics.`,
    /// and nothing else.
    public init(checkpoint url: URL) throws {
        let header = try SafetensorsHeader.read(from: url)
        self.config = try Self.readConfiguration(header)

        let all = try MLX.loadArrays(url: url)
        var encoder: [String: MLXArray] = [:]
        for (name, value) in all where name.hasPrefix(Self.encoderPrefix) {
            encoder[String(name.dropFirst(Self.encoderPrefix.count))] = value
        }
        guard !encoder.isEmpty else { throw Failure.missing(Self.encoderPrefix + "*") }

        // The config claims no attention and an unparameterised norm. Believe the
        // tensors, not the config: either would mean this type silently skips a stage.
        if let stray = encoder.keys.first(where: {
            $0.contains(".attn") || $0.hasSuffix(".q.weight") || $0.hasSuffix(".k.weight")
                || $0.hasSuffix(".v.weight") || $0.contains("proj_out")
        }) {
            throw Failure.unsupported(
                "encoder carries an attention tensor (\(stray)) but mid_block_add_attention "
                + "is false and attn_resolutions is empty; this port implements no attention")
        }
        if let stray = encoder.keys.first(where: { $0.contains("norm") }) {
            throw Failure.unsupported(
                "encoder carries \(stray); norm_type is \"pixel\", which is unparameterised")
        }

        // Every tensor this type will read, with the shape it must have, derived from
        // the config alone. Both directions are checked: a tensor this port needs and
        // cannot find, and a tensor the checkpoint has that this port never consumes.
        // The second is the one that hides a whole missing block — 44 keys loaded and
        // 40 used is a silent architecture mismatch, not a warning.
        let expected = Self.expectedShapes(config)
        for (name, shape) in expected {
            guard let tensor = encoder[name] else {
                throw Failure.missing(Self.encoderPrefix + name)
            }
            guard tensor.shape == shape else {
                throw Failure.shape(Self.encoderPrefix + name, got: tensor.shape,
                                    expected: "\(shape)")
            }
        }
        if let stray = encoder.keys.sorted().first(where: { expected[$0] == nil }) {
            throw Failure.unexpected(Self.encoderPrefix + stray)
        }
        self.weights = encoder

        guard let std = all[Self.statisticsPrefix + "std-of-means"],
              let mean = all[Self.statisticsPrefix + "mean-of-means"]
        else { throw Failure.missing(Self.statisticsPrefix + "{std,mean}-of-means") }
        // The statistics index the *patchified* latent, `z_channels x latentColumns`.
        // See ``normalize(_:)``.
        let statisticsCount = config.zChannels * config.latentColumns
        guard std.shape == [statisticsCount], mean.shape == [statisticsCount] else {
            throw Failure.shape(Self.statisticsPrefix + "{std,mean}-of-means",
                                got: std.shape, expected: "[\(statisticsCount)]")
        }
        self.stdOfMeans = std
        self.meanOfMeans = mean
    }

    /// Parse `ddconfig` and reject anything this type does not implement, loudly, at
    /// load time rather than as a wrong number later.
    ///
    /// `downsample_time: false` is present in this checkpoint's `ddconfig` and is
    /// **inert**: nothing reads it, and both axes downsample regardless. It is named
    /// here so a future reader does not spend an afternoon deciding whether the time
    /// axis should have been left alone.
    static func readConfiguration(_ header: SafetensorsHeader) throws -> Configuration {
        guard let config = header.metadataJSON("config"),
              let audio = config["audio_vae"] as? [String: Any],
              let model = audio["model"] as? [String: Any],
              let params = model["params"] as? [String: Any],
              let dd = params["ddconfig"] as? [String: Any]
        else { throw Failure.configuration("audio_vae.model.params.ddconfig is absent") }

        func int(_ key: String) throws -> Int {
            guard let v = dd[key] as? Int else {
                throw Failure.configuration("ddconfig.\(key) is absent or not an integer")
            }
            return v
        }

        if let norm = dd["norm_type"] as? String, norm != "pixel" {
            throw Failure.unsupported("norm_type is \"\(norm)\"; this port implements "
                + "PixelNorm only, and GroupNorm would need affine weights the "
                + "checkpoint does not carry")
        }
        // The causal axis decides which dimension gets the one-sided padding. For the
        // encoder it also decides which dimension the *downsampler* pads one-sidedly,
        // and the two paddings differ — see ``downsample(_:stem:)``. "width" would make
        // frequency causal and time symmetric: same shapes, every frame misaligned.
        if let axis = dd["causality_axis"] as? String, axis != "height" {
            throw Failure.unsupported("causality_axis is \"\(axis)\"; this port "
                + "implements \"height\" (time on axis 2) only")
        }
        if let add = dd["mid_block_add_attention"] as? Bool, add {
            throw Failure.unsupported("mid_block_add_attention is true; this port "
                + "implements no attention")
        }
        if let attn = dd["attn_resolutions"] as? [Int], !attn.isEmpty {
            throw Failure.unsupported("attn_resolutions is \(attn); this port "
                + "implements no attention")
        }
        // `resamp_with_conv: false` would replace every `Downsample` with a 2x2 average
        // pool, which cannot be made causal under any causality axis. The checkpoint's
        // downsample weights would then be dead. Reject it here rather than load and
        // ignore them.
        if let resamp = dd["resamp_with_conv"] as? Bool, !resamp {
            throw Failure.unsupported("resamp_with_conv is false; Downsample would be an "
                + "average pool, which is not valid for a non-NONE causality axis and "
                + "which would leave the checkpoint's downsample weights unused")
        }

        guard let mult = dd["ch_mult"] as? [Int], !mult.isEmpty else {
            throw Failure.configuration("ddconfig.ch_mult is absent or empty")
        }
        // Two sources, in order, and neither is assumed to be 64.
        let melBins: Int
        if let m = dd["mel_bins"] as? Int {
            melBins = m
        } else if let pre = audio["preprocessing"] as? [String: Any],
                  let mel = pre["mel"] as? [String: Any],
                  let m = mel["n_mel_channels"] as? Int {
            melBins = m
        } else {
            throw Failure.configuration("neither ddconfig.mel_bins nor "
                + "preprocessing.mel.n_mel_channels is present")
        }

        // `double_z` defaults to true, so absence means true — not false.
        let doubleZ = (dd["double_z"] as? Bool) ?? true

        return Configuration(
            ch: try int("ch"), chMult: mult, numResBlocks: try int("num_res_blocks"),
            zChannels: try int("z_channels"), inChannels: try int("in_channels"),
            melBins: melBins, doubleZ: doubleZ, latentDownsampleFactor: 4)
    }

    /// Every tensor this port reads, and the shape it must have. Config in, keys out.
    ///
    /// The channel ladder is `in_ch_mult = (1, *ch_mult)`: level `L`
    /// enters at `ch * in_ch_mult[L]` and leaves at `ch * ch_mult[L]`, and only a level's
    /// **first** block changes channel count and therefore only that block has a
    /// shortcut. That is derived here, not predicted from the level index — on this
    /// checkpoint it yields `down.1.block.0` and `down.2.block.0` and nothing else,
    /// which is exactly what the file carries.
    static func expectedShapes(_ c: Configuration) -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        func conv(_ stem: String, out: Int, inChannels: Int, kernel: Int) {
            shapes[stem + ".conv.weight"] = [out, inChannels, kernel, kernel]
            shapes[stem + ".conv.bias"] = [out]
        }

        conv("conv_in", out: c.ch, inChannels: c.inChannels, kernel: 3)

        let inMult = [1] + c.chMult
        for level in 0 ..< c.numResolutions {
            var current = c.ch * inMult[level]
            let blockOut = c.ch * c.chMult[level]
            for block in 0 ..< c.numResBlocks {
                let stem = "down.\(level).block.\(block)"
                conv(stem + ".conv1", out: blockOut, inChannels: current, kernel: 3)
                conv(stem + ".conv2", out: blockOut, inChannels: blockOut, kernel: 3)
                // `conv_shortcut` is false at every construction site, so the skip is
                // always the 1x1 `nin_shortcut` when it exists at all.
                if current != blockOut {
                    conv(stem + ".nin_shortcut", out: blockOut, inChannels: current,
                         kernel: 1)
                }
                current = blockOut
            }
            if level != c.numResolutions - 1 {
                conv("down.\(level).downsample", out: blockOut, inChannels: blockOut,
                     kernel: 3)
            }
        }

        for stem in ["mid.block_1", "mid.block_2"] {
            conv(stem + ".conv1", out: c.midChannels, inChannels: c.midChannels, kernel: 3)
            conv(stem + ".conv2", out: c.midChannels, inChannels: c.midChannels, kernel: 3)
        }
        conv("conv_out", out: c.headChannels, inChannels: c.midChannels, kernel: 3)
        return shapes
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(Self.encoderPrefix + name) }
        return w
    }

    // MARK: - Encode

    /// Mel `[B, inChannels, melFrames, melBins]` → latent `[B, zChannels, frames,
    /// columns]`, both `NCHW`.
    ///
    /// **This is the deterministic path, and it is the only path there is.** See
    /// ``distribution(_:)`` for why that is a fact about the architecture and not a
    /// choice made here.
    ///
    /// **The mel is cast to the weight dtype (bf16) on the way in**, for the reason the
    /// sibling documents: handing this an fp32 mel makes MLX promote the convolutions to
    /// fp32, changing the arithmetic of every stage. The activation dtype between stages
    /// is part of the architecture, not an accident. Each convolution still keeps its
    /// higher-precision accumulator internally; only the stage boundary is bf16.
    ///
    /// Takes a tensor, never a path: the module boundary is at the mel.
    public func encode(_ mel: MLXArray) throws -> MLXArray {
        try encode(mel, observer: nil)
    }

    /// Internal diagnostic observer, called with each intermediate as it is computed. It
    /// has no path to alter encode arithmetic or state.
    func encode(_ mel: MLXArray, observer: ((String, MLXArray) -> Void)?) throws -> MLXArray {
        let head = try self.head(mel, observer: observer)
        let mean = try mode(ofHead: head)
        observer?("mean", mean)
        let latent = normalize(mean)
        observer?("normalized", latent)
        return latent
    }

    /// The raw distribution head, **before** per-channel normalisation.
    ///
    /// `conv_out` emits `2 * zChannels` under `double_z`, split down the channel axis:
    /// the first half is the mean, the second half the log-variance.
    ///
    /// **Nothing on the audio path ever reads the second half.** There is no diagonal
    /// Gaussian sampling here — the only one in the tree belongs to the *video* VAE — so
    /// **there is no sampled variant to expose and no reparameterisation trick to
    /// mirror.** Inventing one would make a2v non-reproducible for a fixed seed for no
    /// gain. The second half is returned raw, for inspection only.
    ///
    /// The mean returned here is *not* what ``encode(_:)`` returns — normalisation has
    /// not been applied yet.
    public func distribution(_ mel: MLXArray) throws
        -> (mean: MLXArray, logVariance: MLXArray?) {
        let head = try self.head(mel, observer: nil)
        let mean = try mode(ofHead: head)
        guard config.doubleZ else { return (mean, nil) }
        return (mean, head[0..., config.zChannels ..< (2 * config.zChannels)])
    }

    /// `conv_out`'s output, `[B, headChannels, frames, columns]`, in `NCHW`.
    func head(_ mel: MLXArray, observer: ((String, MLXArray) -> Void)?) throws -> MLXArray {
        guard mel.ndim == 4 else {
            throw Failure.shape("mel", got: mel.shape,
                                expected: "[B, \(config.inChannels), melFrames, "
                                    + "\(config.melBins)] (rank 4)")
        }
        guard mel.shape[1] == config.inChannels else {
            // A mono mel here is the single most likely caller error, and it is one MLX
            // would not catch: `conv2d` with a 1-channel input against a 2-channel
            // kernel fails, but a caller who "fixed" it by broadcasting would get a
            // plausible latent from a spectrogram that never existed.
            throw Failure.shape("mel", got: mel.shape,
                                expected: "\(config.inChannels) channels on axis 1; the "
                                    + "mel is stereo (in_channels: \(config.inChannels), "
                                    + "preprocessing.audio.stereo: true)")
        }
        guard mel.shape[3] == config.melBins else {
            throw Failure.shape("mel", got: mel.shape,
                                expected: "\(config.melBins) mel bins on axis 3")
        }
        guard mel.shape[0] > 0, mel.shape[2] > 0 else {
            throw Failure.shape("mel", got: mel.shape,
                                expected: "at least one batch item and one mel frame")
        }
        // A mel too short to survive the downsamples would produce a zero- or
        // one-frame latent by accident. Refuse below the point where the geometric
        // and arithmetic frame counts still describe the same thing.
        let melFrames = mel.shape[2]
        let geometric = config.latentFrames(melFrames: melFrames)
        let arithmetic = config.latentFramesArithmetic(melFrames: melFrames)
        guard geometric == arithmetic else {
            throw Failure.unsupported(
                "mel of \(melFrames) frames gives \(geometric) latent frames "
                + "geometrically and \(arithmetic) arithmetically; the two derivations "
                + "disagree, which means len(ch_mult) - 1 = \(config.numResolutions - 1) "
                + "downsamples and a latent factor of \(config.latentDownsampleFactor) "
                + "do not describe the same compression")
        }

        let dtype = try weight("conv_in.conv.weight").dtype
        // NCHW -> NHWC. Everything below is channels-last; channel axis is -1.
        var h = mel.asType(dtype).transposed(0, 2, 3, 1)

        h = try causalConv(h, "conv_in")
        observer?("conv_in", h)

        for level in 0 ..< config.numResolutions {
            for block in 0 ..< config.numResBlocks {        // num_res_blocks, no +1
                h = try resnet(h, "down.\(level).block.\(block)")
                observer?("down.\(level).block.\(block)", h)
            }
            // The last level has no downsample: the checkpoint carries
            // `down.0.downsample` and `down.1.downsample` and no `down.2.downsample`.
            if level != config.numResolutions - 1 {
                h = try downsample(h, stem: "down.\(level).downsample")
                observer?("down.\(level).downsample", h)
            }
        }

        // mid: block_1 -> Identity -> block_2, the identity being `mid.attn_1` under
        // `mid_block_add_attention: false`. The decoder's `mid` is the same block, which
        // is why nothing sits between them there either.
        h = try resnet(h, "mid.block_1")
        observer?("mid.block_1", h)
        h = try resnet(h, "mid.block_2")
        observer?("mid.block_2", h)

        // The tail: norm -> silu -> conv. The encoder has no `give_pre_end` and no
        // `tanh_out` — those are decoder-only flags — so this tail is unconditional.
        // Split across three statements rather than nested, so the two most
        // dtype-sensitive operations in the whole encoder — the fp32 PixelNorm reduction
        // and the fp32 SiLU — are separately observable rather than sharing a boundary.
        let normed = pixelNorm(h)
        observer?("norm_out", normed)
        let activated = silu(normed)
        observer?("non_linearity", activated)
        h = try causalConv(activated, "conv_out")
        observer?("conv_out", h)

        h = h.transposed(0, 3, 1, 2)                        // NHWC -> NCHW

        let expectedFrames = config.latentFrames(melFrames: melFrames)
        let expectedColumns = config.latentColumns(melBins: mel.shape[3])
        guard h.shape[1] == config.headChannels, h.shape[2] == expectedFrames,
              h.shape[3] == expectedColumns
        else {
            throw Failure.shape("encoder head", got: h.shape,
                                expected: "[B, \(config.headChannels), \(expectedFrames), "
                                    + "\(expectedColumns)]")
        }
        return h
    }

    /// The deterministic latent: the first `zChannels` of the head.
    ///
    /// Named `mode` because for a diagonal Gaussian the mode and the mean coincide, and
    /// "mode" is the word the surrounding codebase uses for the non-sampled path.
    func mode(ofHead head: MLXArray) throws -> MLXArray {
        guard head.shape[1] == config.headChannels else {
            throw Failure.shape("encoder head", got: head.shape,
                                expected: "\(config.headChannels) channels on axis 1")
        }
        guard config.doubleZ else { return head }
        return head[0..., 0 ..< config.zChannels]
    }

    /// Apply the per-channel normalisation: `(x − mean) / std`.
    ///
    /// The exact inverse of ``AudioVAEDecoder/denormalize(_:)``, which is
    /// `x * std + mean`.
    ///
    /// **The `[128]` statistics against an `8`-channel latent are not a mismatch.** The
    /// latent is patchified downstream — `b c t f -> b t (c f)` with `c = 8, f = 16` —
    /// so the statistics index the flattened `(channel, mel column)` pair,
    /// `8 × 16 = 128`. Broadcasting over `NCHW` with a `[1, C, 1, F]` reshape is the same
    /// arithmetic without the two rearranges, and the reshape is **channel-major**:
    /// `[1, F, 1, C]` is a same-shaped latent scaled by the wrong 128 numbers, with
    /// nothing to catch it until the DiT is conditioned on nonsense.
    ///
    /// The statistics are cast to the latent's dtype, so this stays in bf16 — the
    /// division is not silently promoted.
    ///
    /// Order matters: chunk, *then* normalise. Normalising the full 16-channel head
    /// first would apply 128 statistics to 256 values.
    public func normalize(_ latent: MLXArray) -> MLXArray {
        let channels = latent.shape[1], columns = latent.shape[3]
        let std = stdOfMeans.reshaped([1, channels, 1, columns]).asType(latent.dtype)
        let mean = meanOfMeans.reshaped([1, channels, 1, columns]).asType(latent.dtype)
        return (latent - mean) / std
    }

    // MARK: - Shape algebra

    /// The latent shape ``encode(_:)`` will return for a mel of `melShape`, without
    /// running it. Validates the same way `encode` does.
    public func latentShape(forMel melShape: [Int]) throws -> [Int] {
        guard melShape.count == 4, melShape[1] == config.inChannels,
              melShape[3] == config.melBins, melShape[0] > 0, melShape[2] > 0
        else {
            throw Failure.shape("mel", got: melShape,
                                expected: "[B, \(config.inChannels), melFrames, "
                                    + "\(config.melBins)]")
        }
        return [melShape[0], config.zChannels,
                config.latentFrames(melFrames: melShape[2]),
                config.latentColumns(melBins: melShape[3])]
    }

    /// Mel frames needed to encode a render of `videoFrames` at `frameRate` into the
    /// latent that render requires.
    ///
    /// Composes the two rules that own this number, and neither is this type's:
    ///
    /// 1. `LatentGeometry.audioLatentCount` — `round(seconds × 25)`, **`round`, not
    ///    `ceil`** (contract 3). 2.3 used `ceil` and 2.5 changed it, and the difference
    ///    is one latent at some frame counts. At 97 frames / 24 fps it is
    ///    `round(101.04) = 101` against `ceil(101.04) = 102`, which is a shape mismatch
    ///    rather than a value diff — a port carrying 2.3's `ceil` looks correct until a
    ///    frame count lands near a boundary. The 25 is not a constant either: it is
    ///    `sample_rate / hop_length / factor = 16000 / 160 / 4`.
    /// 2. ``Configuration/melFrames(latentFrames:)`` — `4T − 3`.
    ///
    /// For 97 frames at 24 fps: 101 latents, 401 mel frames. Those are the production
    /// numbers, `[1, 8, 101, 16]` and `[1, 2, 401, 64]`.
    public func requiredMelFrames(videoFrames: Int, frameRate: Double,
                                  geometry: LatentGeometry = LatentGeometry()) -> Int {
        config.melFrames(
            latentFrames: geometry.audioLatentCount(frames: videoFrames,
                                                    frameRate: frameRate))
    }

    // MARK: - Blocks

    /// `ResnetBlock`: `norm → silu → conv1 → norm → silu → conv2`, plus the skip.
    ///
    /// The same block the decoder uses, so this mirrors ``AudioVAEDecoder/resnet(_:_:)``
    /// deliberately rather than coincidentally. `temb_channels` is 0 for the whole VAE,
    /// so there is no timestep-embedding branch to run. Dropout is 0.0 and the model is
    /// only ever evaluated.
    ///
    /// The shortcut's presence is *read* from the checkpoint, not predicted, for the same
    /// reason as in the sibling; ``expectedShapes(_:)`` independently derives where it
    /// should be and the initialiser rejects any disagreement.
    func resnet(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        var h = try causalConv(silu(pixelNorm(x)), stem + ".conv1")
        h = try causalConv(silu(pixelNorm(h)), stem + ".conv2")
        var skip = x
        if weights[stem + ".nin_shortcut.conv.weight"] != nil {
            skip = try causalConv(x, stem + ".nin_shortcut")
        }
        return skip + h
    }

    /// `Downsample`: asymmetric pad, then a 3×3 **stride-2** convolution.
    ///
    /// This is the one place the encoder is not a mirror of anything, and the trap is
    /// that its key grammar is identical to a `CausalConv2d`'s while its padding is not.
    ///
    /// `Downsample`'s convolution is a **plain** one, not a `CausalConv2d`. Both store
    /// their kernel at `<stem>.conv.weight`, so `down.0.downsample.conv.weight` and
    /// `conv_in.conv.weight` are indistinguishable by name — but the padding rules
    /// differ, and `Downsample` fixes its own tuple per causality axis rather than
    /// deriving one from the kernel size:
    ///
    /// ```
    /// causality axis = height:  pad = (0, 1, 2, 0)   # (left, right, top, bottom)
    /// ```
    ///
    /// So time gets `(2, 0)` — the same full causal extent a `CausalConv2d` uses — but
    /// **frequency gets `(0, 1)`, not `(1, 1)`**. That is the difference that has no
    /// shape consequence whatsoever: 64 columns padded `(1,1)` to 66 and 64 padded
    /// `(0,1)` to 65 both convolve down to 32 at stride 2. Only the alignment differs,
    /// by half a bin, on every frequency in the spectrogram. Nothing downstream can see
    /// it; it is written out here because reading it off `CausalConv2d` by analogy is
    /// the obvious mistake and it is invisible.
    ///
    /// Axis 1 is time and axis 2 is frequency because the tensor is NHWC here.
    func downsample(_ x: MLXArray, stem: String) throws -> MLXArray {
        let w = try weight(stem + ".conv.weight")
        let b = try weight(stem + ".conv.bias")
        guard w.ndim == 4, w.shape[2] == 3, w.shape[3] == 3 else {
            throw Failure.shape(stem + ".conv.weight", got: w.shape,
                                expected: "[out, in, 3, 3]; Downsample's padding tuple is "
                                    + "hardcoded for a 3x3 kernel and does not generalise")
        }
        let input = MLX.padded(x.asType(w.dtype), widths: [
            IntOrPair(0),           // batch
            IntOrPair((2, 0)),      // time, causal: the full extent above
            IntOrPair((0, 1)),      // frequency: right only, NOT symmetric
            IntOrPair(0),           // channels
        ])
        let kernel = w.transposed(0, 2, 3, 1)
        // Same accumulator discipline as ``causalConv(_:_:)``.
        let accumulated = MLX.conv2d(input.asType(.float32), kernel.asType(.float32),
                                     stride: 2) + b.asType(.float32)
        return accumulated.asType(w.dtype)
    }

    // MARK: - Primitives

    /// A `CausalConv2d` from the checkpoint: `<stem>.conv.{weight,bias}`.
    ///
    /// The doubled `conv` in the key is the wrapper's inner convolution; the wrapper
    /// holds no parameters, only the padding:
    ///
    /// ```
    /// (pad_left, pad_right, pad_top, pad_bottom) = (pad_w/2, pad_w - pad_w/2, pad_h, 0)
    /// ```
    ///
    /// for `causality_axis = height`. **Time is padded entirely at the top, frequency
    /// symmetrically.** For a 3×3 kernel that is `(1, 1, 2, 0)` — *two* rows above, not
    /// one, because the full `(k − 1)` extent goes on the causal side. Splitting it 1/1
    /// gives the same output shape and lets every frame see one frame of its own future,
    /// which is what causal convolution exists to prevent and which no shape check can
    /// detect. Contrast ``downsample(_:stem:)``, whose frequency padding is *not* this.
    ///
    /// MLX's `conv2d` takes one symmetric `padding` per axis and cannot express `(2, 0)`,
    /// so the padding is materialised with `padded(widths:)` and the convolution runs
    /// unpadded. That is not an approximation — it is the same zero padding, written out.
    ///
    /// Weight layout: the checkpoint stores `[out, in, kh, kw]`, MLX convolves `[out, kh,
    /// kw, in]` over `NHWC`, hence the `(0, 2, 3, 1)` transpose.
    func causalConv(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        let w = try weight(stem + ".conv.weight")
        let b = try weight(stem + ".conv.bias")
        guard w.ndim == 4 else {
            throw Failure.shape(stem + ".conv.weight", got: w.shape,
                                expected: "[out, in, kh, kw]")
        }
        let kh = w.shape[2], kw = w.shape[3]
        let padHeight = kh - 1, padWidth = kw - 1

        var input = x.asType(w.dtype)
        if padHeight > 0 || padWidth > 0 {
            input = MLX.padded(input, widths: [
                IntOrPair(0),                                        // batch
                IntOrPair((padHeight, 0)),                           // time, causal
                IntOrPair((padWidth / 2, padWidth - padWidth / 2)),  // frequency
                IntOrPair(0),                                        // channels
            ])
        }
        let kernel = w.transposed(0, 2, 3, 1)
        // A convolution accumulates its BF16 products and its bias together before
        // storing the BF16 result. MLX exposes convolution and bias as separate ops;
        // retain the accumulator precision across that seam before the one required
        // boundary rounding, instead of rounding the convolution and then adding bias.
        // This is the mirror of the note in ``AudioVAEDecoder/conv(_:_:)`` and it is
        // load-bearing in the same way: rounding twice is a different number.
        let accumulated = MLX.conv2d(input.asType(.float32), kernel.asType(.float32))
            + b.asType(.float32)
        return accumulated.asType(w.dtype)
    }

    /// `x / sqrt(mean(x², over channels) + 1e-6)`, no affine, channel axis last.
    ///
    /// Duplicated from ``AudioVAEDecoder/pixelNorm(_:eps:)`` rather than shared: the
    /// sibling's is an instance method on a type this one does not own, and constructing
    /// a decoder to borrow it would load 56 tensors to call a pure function.
    ///
    /// ## It rounds once more than the sibling
    ///
    /// The norm is three statements and each one stores a bf16 result — fp32 accumulate,
    /// bf16 store:
    ///
    /// ```
    /// meanSq = mean(x², over channels)   -> bf16
    /// rms    = sqrt(meanSq + eps)        -> bf16, twice: once for the add, once for sqrt
    /// return x / rms                     -> bf16
    /// ```
    ///
    /// ``AudioVAEDecoder/pixelNorm(_:eps:)`` folds the epsilon add into the square root
    /// and rounds once there instead. This form is the one that follows the repo's rule
    /// of casting back after each op. There are 13 PixelNorm sites on this path, which is
    /// why a single rounding point moves the output at all.
    ///
    /// **The epsilon is 1e-6, not 1e-8.** 1e-8 is the natural default for a norm of this
    /// shape and is overridden at every construction site; only the override is ever
    /// built.
    func pixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let squared = x * x
        let meanSquare = squared.asType(.float32).mean(axis: -1, keepDims: true).asType(x.dtype)
        let shifted = (meanSquare.asType(.float32) + eps).asType(x.dtype)
        let rms = MLX.sqrt(shifted.asType(.float32)).asType(x.dtype)
        return x / rms
    }

    /// `x * sigmoid(x)`, evaluated in fp32 and cast back.
    ///
    /// Duplicated for the same reason as ``pixelNorm(_:eps:)``. An elementwise
    /// nonlinearity on a bf16 tensor takes an fp32 accumulate type; evaluating it in bf16
    /// throughout is a real error, not a rounding one. There are 17 SiLUs on this
    /// encoder's path — two per resnet across six down-blocks and two mid-blocks, plus
    /// the one before `conv_out`.
    func silu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * MLX.sigmoid(f)).asType(x.dtype)
    }
}

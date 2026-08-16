// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The audio VAE **decoder**: audio latent → log-mel spectrogram.
///
/// It stops deliberately short of the waveform.
///
/// ## Why the vocoder is not here
///
/// `FRAGILE_CONTRACTS.md` contract 11. cuDNN picks the algorithm for the vocoder's
/// transposed convolutions per run, so two runs given a **bit-identical** mel produce
/// different waveforms. The mel is where the audio path stops being reproducible, so
/// the mel is where this type ends — at `conv_out` — and the waveform is judged
/// perceptually. Adding the vocoder here would append a stage nothing can check.
///
/// ## What the architecture actually is
///
/// Read off `ddconfig` in the checkpoint's own metadata rather than assumed, because two
/// of the LDM-family defaults are wrong here and both would load every tensor the port
/// went looking for:
///
/// * **No attention anywhere.** `mid_block_add_attention: false` and
///   `attn_resolutions: []`, so `mid.attn_1` is an identity and no stage has an `attn`
///   list. The 56 decoder tensors contain no `q`/`k`/`v`/`proj_out`. That is why the
///   choice of attention backend makes no difference through this stage — not that the
///   kernels agree, but that none of them runs.
/// * **PixelNorm, not GroupNorm**, and unparameterised: `x / sqrt(mean(x², over
///   channels) + 1e-6)` with no weight and no bias. The checkpoint has zero `norm*`
///   tensors. A port that reached for `GroupNorm(32, affine: true)` would find nothing
///   to load and could silently run with ones and zeros: same shapes, different
///   arithmetic, no error. Note the epsilon is **1e-6**, not the `1e-8` that is the
///   natural default for a norm of this shape; it is set explicitly at every
///   construction site.
///
/// The shape ladder, with the two facts that do not follow from the latent:
///
/// ```
/// latent            [1,   8,  26, 16]
/// conv_in           [1, 512,  26, 16]
/// mid  2 x resnet   [1, 512,  26, 16]     no attention between them
/// up[2] 3 x resnet  [1, 512,  26, 16] -> upsample -> [1, 512,  51, 32]
/// up[1] 3 x resnet  [1, 256,  51, 32] -> upsample -> [1, 256, 101, 64]
/// up[0] 3 x resnet  [1, 128, 101, 64]     no upsample at level 0
/// conv_out          [1,   2, 101, 64]
/// ```
///
/// **26 latent frames become 101 mel frames, not 104.** Each upsample is
/// nearest-neighbour ×2 followed by a causal conv and then a drop of the *first* time
/// row, so `T → 2T − 1`: `26 → 51 → 101`.
/// ``Configuration/targetFrames(latentFrames:)`` derives the same number independently as
/// `max(26 × 4 − 3, 1)`. See ``upsample(_:stem:)`` for why the drop is on the first row
/// and not the last.
///
/// **The mel is stereo because `conv_out.conv.weight` is `[2, 128, 3, 3]`**, matching
/// `out_ch: 2` and `preprocessing.audio.stereo: true`. Nothing about the latent's 8
/// channels or 16 columns implies it. Contract 9b says the same thing: neither the
/// channel count nor the frame count follows from the latent shape, so both are read
/// off the checkpoint.
///
/// ## Layout
///
/// MLX convolves in `NHWC` with weights `[out, kh, kw, in]`; the checkpoint stores
/// `NCHW` with `[out, in, kh, kw]`. The transposes happen at the two ends and once per
/// kernel, and everything between ``decode(_:)``'s two `transposed` calls is NHWC — so
/// **the channel axis is `-1` inside this type and `1` at its boundary**. `PixelNorm`
/// reduces over channels; getting that axis wrong normalises over frequency instead,
/// which is a same-shaped tensor and a silently different network.
public struct AudioVAEDecoder {

    /// `ddconfig`, read from the checkpoint rather than declared.
    ///
    /// Every field here has a plausible wrong value that type-checks and loads, which is
    /// why none of them is written down in the source.
    public struct Configuration: Equatable, Sendable {
        public let ch: Int
        public let chMult: [Int]
        /// `num_res_blocks` from the config. The **upsampling** path runs
        /// `num_res_blocks + 1` blocks per stage — the decoder is not symmetric with
        /// the encoder here, and 2 becomes 3.
        public let numResBlocks: Int
        public let zChannels: Int
        public let outChannels: Int
        public let melBins: Int
        /// 4. The latent is 4× coarser than the mel in time, before causal trimming.
        public let latentDownsampleFactor: Int

        public var numResolutions: Int { chMult.count }

        /// Frames the decoder is required to produce from `latentFrames`.
        ///
        /// `latentFrames * factor - (factor - 1)`, floored at 1, which is the causal
        /// form. For 26 that is 101, and it agrees with what the two causal upsamples
        /// produce geometrically. The two are derived independently and cross-checked
        /// in ``decode(_:)``.
        public func targetFrames(latentFrames: Int) -> Int {
            max(latentFrames * latentDownsampleFactor - (latentDownsampleFactor - 1), 1)
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)
        case unsupported(String)
        case configuration(String)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .unsupported(why): return why
            case let .configuration(why): return "checkpoint config: \(why)"
            }
        }
    }

    /// Decoder tensors, keys stripped of `audio_vae.decoder.`.
    public let weights: [String: MLXArray]
    /// `per_channel_statistics.std-of-means`, `[ch]` = `[128]`.
    public let stdOfMeans: MLXArray
    /// `per_channel_statistics.mean-of-means`, `[ch]` = `[128]`.
    public let meanOfMeans: MLXArray
    public let config: Configuration

    public static let decoderPrefix = "audio_vae.decoder."
    public static let statisticsPrefix = "audio_vae.per_channel_statistics."

    /// Load the decoder half of a file that also contains the vocoder.
    ///
    /// The checkpoint is 1329 tensors, 1227 of which are the vocoder. The two prefixes
    /// above separate them, and nothing else is read. Loading the vocoder's weights
    /// would cost far more memory than the decoder itself.
    public init(checkpoint url: URL) throws {
        let header = try SafetensorsHeader.read(from: url)
        self.config = try Self.readConfiguration(header)

        let all = try MLX.loadArrays(url: url)
        var decoder: [String: MLXArray] = [:]
        for (name, value) in all where name.hasPrefix(Self.decoderPrefix) {
            decoder[String(name.dropFirst(Self.decoderPrefix.count))] = value
        }
        guard !decoder.isEmpty else { throw Failure.missing(Self.decoderPrefix + "*") }

        // The config claims no attention. Believe the tensors, not the config: an
        // attention tensor here would mean this type silently skips a whole block.
        if let stray = decoder.keys.first(where: {
            $0.contains(".attn") || $0.hasSuffix(".q.weight") || $0.hasSuffix(".k.weight")
                || $0.hasSuffix(".v.weight") || $0.contains("proj_out")
        }) {
            throw Failure.unsupported(
                "decoder carries an attention tensor (\(stray)) but mid_block_add_attention "
                + "is false and attn_resolutions is empty; this port implements no attention")
        }
        // Same for a learned norm. PixelNorm has no parameters; if the checkpoint has
        // any, the norm is not the one this type implements.
        if let stray = decoder.keys.first(where: { $0.contains("norm") }) {
            throw Failure.unsupported(
                "decoder carries \(stray); norm_type is \"pixel\", which is unparameterised")
        }
        self.weights = decoder

        guard let std = all[Self.statisticsPrefix + "std-of-means"],
              let mean = all[Self.statisticsPrefix + "mean-of-means"]
        else { throw Failure.missing(Self.statisticsPrefix + "{std,mean}-of-means") }
        self.stdOfMeans = std
        self.meanOfMeans = mean
    }

    /// Parse `ddconfig` out of the safetensors metadata and reject anything this type
    /// does not implement, loudly, at load time rather than as a wrong number later.
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
        // The causal axis decides *which* dimension gets the one-sided padding and the
        // dropped row. "width" would make frequency causal and time symmetric — same
        // shapes throughout, every frame misaligned.
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

        guard let mult = dd["ch_mult"] as? [Int], !mult.isEmpty else {
            throw Failure.configuration("ddconfig.ch_mult is absent or empty")
        }
        // `mel_bins` is in ddconfig on this checkpoint, with preprocessing.mel.
        // n_mel_channels as the fallback. Both are read; neither is assumed to be 64,
        // which is what the latent's 16 columns times two upsamples happens to give.
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

        return Configuration(
            ch: try int("ch"), chMult: mult, numResBlocks: try int("num_res_blocks"),
            zChannels: try int("z_channels"), outChannels: try int("out_ch"),
            melBins: melBins, latentDownsampleFactor: 4)
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(Self.decoderPrefix + name) }
        return w
    }

    // MARK: - Decode

    /// Latent `[B, zChannels, frames, columns]` → mel `[B, outChannels, mel frames,
    /// melBins]`, both `NCHW`.
    ///
    /// The latent is the whole input: there is no hidden state, no cache and no second
    /// argument.
    ///
    /// **The latent is cast to the weight dtype (bf16) on the way in.** Handing this an
    /// fp32 latent makes MLX promote the convolutions to fp32, changing the arithmetic of
    /// every stage; the activation dtype between stages is part of the architecture, not
    /// an accident. That fixes the stage boundary precision only — each Conv2d still uses
    /// its normal higher-precision accumulator before storing a bf16 activation.
    public func decode(_ latent: MLXArray) throws -> MLXArray {
        try decode(latent, observer: nil)
    }

    /// Internal diagnostic observer, called with each intermediate as it is computed. It
    /// has no path to alter decode arithmetic or state.
    func decode(_ latent: MLXArray,
                observer: ((String, MLXArray) -> Void)?) throws -> MLXArray {
        guard latent.ndim == 4, latent.shape[1] == config.zChannels else {
            throw Failure.shape("latent", got: latent.shape,
                                expected: "[B, \(config.zChannels), frames, columns]")
        }
        let dtype = try weight("conv_in.conv.weight").dtype
        let latentFrames = latent.shape[2]
        let targetFrames = config.targetFrames(latentFrames: latentFrames)

        var h = denormalize(latent.asType(dtype))
        observer?("denormalized", h)
        // NCHW -> NHWC. Everything below is channels-last; channel axis is -1.
        h = h.transposed(0, 2, 3, 1)

        h = try conv(h, "conv_in")
        observer?("conv_in", h)

        // mid: block_1 -> Identity -> block_2. The identity is `mid.attn_1` with
        // `mid_block_add_attention: false`, which is why nothing sits between them.
        h = try resnet(h, "mid.block_1")
        observer?("mid.block_1", h)
        h = try resnet(h, "mid.block_2")
        observer?("mid.block_2", h)

        for level in stride(from: config.numResolutions - 1, through: 0, by: -1) {
            for block in 0...config.numResBlocks {          // num_res_blocks + 1
                h = try resnet(h, "up.\(level).block.\(block)")
                observer?("up.\(level).block.\(block)", h)
            }
            // Level 0 has no upsample: the checkpoint carries `up.1.upsample` and
            // `up.2.upsample` and no `up.0.upsample`. The guard checks both the level
            // and the weight's presence.
            if level != 0, weights["up.\(level).upsample.conv.conv.weight"] != nil {
                h = try upsample(h, stem: "up.\(level).upsample")
                observer?("up.\(level).upsample", h)
            }
        }

        // `give_pre_end: false`, so the norm/activation/conv tail runs; `tanh_out:
        // false`, so the mel leaves unsquashed. Both flags are fixed rather than
        // configurable, and a `tanh` here would clamp a spectrogram whose real range is
        // about [-10.9, -2.9].
        h = try conv(silu(pixelNorm(h)), "conv_out")
        observer?("conv_out", h)

        h = h.transposed(0, 3, 1, 2)                        // NHWC -> NCHW

        // The geometric result and the arithmetic target must already agree. They are
        // derived two different ways — two `T -> 2T-1` upsamples versus
        // `4T - 3` — so a disagreement means the causal drop landed on the wrong row,
        // and ``adjustOutputShape(_:targetFrames:)`` would paper over it by cropping or
        // zero-padding. That crop-and-pad exists below, but reaching it is a bug and
        // this says so.
        guard h.shape[2] == targetFrames, h.shape[3] == config.melBins,
              h.shape[1] == config.outChannels
        else {
            throw Failure.shape("decoded mel", got: h.shape,
                                expected: "[B, \(config.outChannels), \(targetFrames), "
                                    + "\(config.melBins)] before shape adjustment")
        }
        return adjustOutputShape(h, targetFrames: targetFrames)
    }

    /// Undo the encoder's per-channel normalisation: `x * std + mean`.
    ///
    /// **The `[128]` statistics against an `8`-channel latent are not a mismatch.** The
    /// latent patchifies as `b c t f -> b t (c f)` with `c = 8, f = 16`, so the
    /// statistics index the flattened `(channel, mel column)` pair — `8 × 16 = 128`,
    /// which is also why they are `[ch]`-sized in a decoder whose latent is `z_channels`
    /// wide. The reshape is therefore `[1, C, 1, F]`, **channel-major**. Transposing it
    /// to `[1, F, 1, C]` is a same-shaped latent scaled by the wrong 128 numbers, with
    /// nothing to catch it until the mel is wrong everywhere.
    ///
    /// Broadcasting over `NCHW` is the same arithmetic as patchify/denormalise/unpatchify
    /// without the two rearranges, and the statistics are cast to the latent's dtype, so
    /// this stays in bf16 too.
    func denormalize(_ latent: MLXArray) -> MLXArray {
        let channels = latent.shape[1], columns = latent.shape[3]
        let std = stdOfMeans.reshaped([1, channels, 1, columns]).asType(latent.dtype)
        let mean = meanOfMeans.reshaped([1, channels, 1, columns]).asType(latent.dtype)
        return latent * std + mean
    }

    // MARK: - Blocks

    /// `ResnetBlock`: `norm → silu → conv1 → norm → silu → conv2`, plus the skip.
    ///
    /// `temb_channels` is 0 for the whole VAE, so there is no timestep-embedding branch
    /// to run. Dropout is 0.0 and the model is only ever evaluated.
    ///
    /// The skip is always `nin_shortcut` (1×1) and never `conv_shortcut` (3×3):
    /// `conv_shortcut` is `false` at every construction site. Only a
    /// stage's channel-changing first block has one at all — the checkpoint carries
    /// `up.0.block.0.nin_shortcut` and `up.1.block.0.nin_shortcut` and nothing else,
    /// which is exactly the two places `in_channels != out_channels`. So its presence
    /// is *read*, not predicted from the level index.
    func resnet(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        var h = try conv(silu(pixelNorm(x)), stem + ".conv1")
        h = try conv(silu(pixelNorm(h)), stem + ".conv2")
        var skip = x
        if weights[stem + ".nin_shortcut.conv.weight"] != nil {
            skip = try conv(x, stem + ".nin_shortcut")
        }
        return skip + h
    }

    /// Nearest ×2 on both axes, causal conv, then drop the **first** time row.
    ///
    /// Two things here are one-line changes that keep every shape and break every value:
    ///
    /// * **`mode: "nearest"`, not bilinear.** Nearest duplicates each element;
    ///   `repeated(count: 2, axis:)` is exactly that. Bilinear interpolates and hands
    ///   every downstream block a different, entirely plausible tensor.
    /// * **The drop is row 0, not row `n-1`.** After the duplication `[0,0,1,1,2,2]`,
    ///   the causal conv's two-row left pad makes output rows 0 and 1 both depend on
    ///   input element 0 alone, while every later row spans two inputs. Dropping row 0
    ///   removes the redundant one and restores the `1 + 2n` length the encoder's
    ///   padding implies. Dropping the last row gives the identical shape and shifts
    ///   every frame by one — an error a shape assertion cannot see and a
    ///   frame-count check cannot see either.
    ///
    /// Axis 1 is time and axis 2 is frequency because the tensor is NHWC here.
    func upsample(_ x: MLXArray, stem: String) throws -> MLXArray {
        var y = MLX.repeated(x, count: 2, axis: 1)
        y = MLX.repeated(y, count: 2, axis: 2)
        y = try conv(y, stem + ".conv")
        return y[0..., 1...]                        // axis 1 is time in NHWC
    }

    /// Crop-then-pad to the frame count ``Configuration/targetFrames(latentFrames:)``
    /// derived.
    ///
    /// A no-op whenever the causal drops landed correctly, which ``decode(_:)`` asserts
    /// before calling this. It exists anyway because omitting it would turn a future
    /// off-by-one into a shape crash instead of a crop — a behavioural difference at a
    /// boundary, not just a missing convenience.
    func adjustOutputShape(_ x: MLXArray, targetFrames: Int) -> MLXArray {
        var y = x[0...,
                  0 ..< min(x.shape[1], config.outChannels),
                  0 ..< min(x.shape[2], targetFrames),
                  0 ..< min(x.shape[3], config.melBins)]
        let timePad = targetFrames - y.shape[2]
        let freqPad = config.melBins - y.shape[3]
        if timePad > 0 || freqPad > 0 {
            y = MLX.padded(y, widths: [IntOrPair(0), IntOrPair(0),
                                       IntOrPair((0, max(timePad, 0))),
                                       IntOrPair((0, max(freqPad, 0)))])
        }
        return y
    }

    // MARK: - Primitives

    /// A `CausalConv2d` from the checkpoint: `<stem>.conv.{weight,bias}`.
    ///
    /// The doubled `conv` in the key is the wrapper's inner convolution; the wrapper
    /// holds no parameters of its own, only the padding — which is the whole point of
    /// it:
    ///
    /// ```
    /// (pad_left, pad_right, pad_top, pad_bottom) = (pad_w/2, pad_w - pad_w/2, pad_h, 0)
    /// ```
    ///
    /// for `causality_axis = height`. **Time is padded entirely at the top, frequency
    /// symmetrically.** For a 3×3 kernel that is `(1, 1, 2, 0)` — *two* rows above, not
    /// one, because the full `(k − 1)` extent goes on the causal side. Splitting it 1/1
    /// produces the same output shape and lets every frame see one frame of its own
    /// future, which is precisely the defect causal convolution exists to prevent and
    /// which no shape check can detect.
    ///
    /// MLX's `conv2d` takes a single symmetric `padding` per axis, so it cannot express
    /// `(2, 0)`; the padding is materialised with `padded(widths:)` first and the
    /// convolution runs unpadded. That is not an approximation — it is the same zero
    /// padding, written out.
    ///
    /// Weight layout: the checkpoint stores `[out, in, kh, kw]` and MLX convolves
    /// with `[out, kh, kw, in]` over `NHWC`, hence the `(0, 2, 3, 1)` transpose.
    func conv(_ x: MLXArray, _ stem: String) throws -> MLXArray {
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
        // storing the BF16 result.  MLX exposes convolution and bias as separate ops;
        // retain the accumulator precision across that seam before the one required
        // boundary rounding instead of rounding the convolution and then adding bias.
        let accumulated = MLX.conv2d(input.asType(.float32), kernel.asType(.float32))
            + b.asType(.float32)
        return accumulated.asType(w.dtype)
    }

    /// `x / sqrt(mean(x², over channels) + 1e-6)`, no affine, channel axis last.
    ///
    /// The dtype split is the one `DiTModules` documents, applied to a reduction:
    ///
    /// * The **square** stays in the tensor's dtype, because it is its own op and its
    ///   bf16 result is what the mean then sees.
    /// * The **reduction widens to fp32 and casts back** — fp32 accumulate, bf16 result.
    ///   Left implicit, MLX's accumulator is not something to rely on; 512 channels is
    ///   enough summands for a bf16 accumulator to lose real precision.
    /// * `sqrt(· + eps)` runs in fp32 and casts back, the elementwise rule.
    ///
    /// **The epsilon is 1e-6, not 1e-8.** 1e-8 is the natural default for a norm of this
    /// shape and is overridden at every construction site; only the override is ever
    /// built.
    func pixelNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let squared = x * x
        let meanSquare = squared.asType(.float32).mean(axis: -1, keepDims: true)
            .asType(x.dtype)                                    // contract 16
        let rms = MLX.sqrt(meanSquare.asType(.float32) + eps).asType(x.dtype)
        return x / rms
    }

    /// `x * sigmoid(x)`, evaluated in fp32 and cast back.
    ///
    /// An elementwise nonlinearity on a bf16 tensor takes an fp32 accumulate type;
    /// evaluating it in bf16 throughout is a real error, not a rounding one. There are
    /// 19 SiLUs on this decoder's path.
    func silu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * MLX.sigmoid(f)).asType(x.dtype)
    }
}

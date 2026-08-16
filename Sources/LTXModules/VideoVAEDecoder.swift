// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The **conv** video VAE decoder: video latent → RGB frames.
///
/// `[B, 128, F', H', W']` in, `[B, 3, 8(F'−1)+1, 32H', 32W']` out, with a separate
/// ``rgb(_:)`` tail — `c f h w → f h w c`, `(x + 1) × 0.5`, clamp — because that, and
/// not the raw decoder output, is the `[0, 1]` video a caller wants.
///
/// ## Which decoder this is
///
/// 2.5 ships two video VAEs. `ltx-2.5-video-vae-conv-bf16.safetensors`, whose config
/// says `_class_name: "CausalVideoAutoencoder"`, is the **convolutional** decoder this
/// type implements; `ltx-2.5-video-vae-bf16.safetensors` sits next to it in the same
/// directory and is the diffusion VAE, `CausalDiffusionVAE`. They are different
/// architectures behind the same interface, so ``readConfiguration(_:)`` rejects the
/// other one by name rather than discovering it as a missing tensor.
///
/// ## What the architecture actually is
///
/// Read off `__metadata__.config.vae` rather than assumed. Six facts here have a
/// plausible wrong value that still loads every tensor the port asks for:
///
/// * **`decoder_blocks` is applied REVERSED**, so ``upBlocks``'s entry 0 is the
///   config's *last* entry. Read forwards, the ladder has the same tensor count and the
///   same channel widths at the wrong depths, and nothing raises.
/// * **PixelNorm's epsilon is 1e-8** — *not* the audio VAE's 1e-6. Carrying
///   ``AudioVAEDecoder``'s constant across is the obvious move and it is wrong.
/// * **`causal_decoder` is false, and time is padded by REPLICATION**: frame 0 once in
///   front and the last frame once behind. Not zeros — zeros keep every shape and pull
///   the first and last output frames toward the origin. Not causal either: causal
///   would put the whole `k − 1` extent in front and nothing behind.
/// * **Spatial padding is zeros while temporal padding is replicate.** Two padding
///   modes on two axes of the same kernel: the inner convolution zero-pads `(0, 1, 1)`
///   inside a wrapper that has already replicate-padded time.
/// * **No attention anywhere.** `decoder_blocks` contains no `attn` or `attn_res_x`
///   entry, so no attention block is ever constructed and the 84 decoder tensors carry
///   no `to_qkv`, `proj` or `gamma`.
/// * **No `conv_shortcut`, no `norm3`, no `scale_shift_table`.** Every resnet has
///   `in == out` — the channel changes happen inside the depth-to-space rearrange,
///   not in a resnet — so both are identities; and `timestep_conditioning` is false,
///   so there is no ada-LN and no noise injection and hence **no generator**. The
///   manifest's `decode_generator_seed: 1234` is consumed by nothing on this path,
///   which is why this decode is reproducible at all.
///
/// ## The ladder
///
/// ```
/// latent                       [1,  128,  4,  6,  10]
/// conv_in                      [1, 1024,  4,  6,  10]
/// up_blocks.0  res_x  2        [1, 1024,  4,  6,  10]
/// up_blocks.1  compress_all    [1, 4096,  4,  6,  10] -> d2s [1, 512,  8, 12, 20] -> 7
/// up_blocks.2  res_x  2        [1,  512,  7, 12,  20]
/// up_blocks.3  compress_all    [1, 4096,  7, 12,  20] -> d2s [1, 512, 14, 24, 40] -> 13
/// up_blocks.4  res_x  4        [1,  512, 13, 24,  40]
/// up_blocks.5  compress_time   [1,  512, 13, 24,  40] -> d2s [1, 256, 26, 24, 40] -> 25
/// up_blocks.6  res_x  6        [1,  256, 25, 24,  40]
/// up_blocks.7  compress_space  [1,  512, 25, 24,  40] -> d2s [1, 128, 25, 48, 80]
/// up_blocks.8  res_x  4        [1,  128, 25, 48,  80]
/// conv_out                     [1,   48, 25, 48,  80]
/// unpatchify(4)                [1,    3, 25, 192, 320]
/// ```
///
/// `4 → 7 → 13 → 25` is `8(F' − 1) + 1`, and it comes from **dropping the first frame**
/// after each temporal depth-to-space — not from any rate that divides. See
/// ``upsample(_:stem:stride:)``.
///
/// ## Tiling
///
/// A tiled decode splits the latent, calls this same forward pass per tile and blends
/// the overlaps with trapezoidal masks. Tile *count* is a memory strategy and
/// explicitly **not** a contract — it varies from one tile at small shapes to a hundred
/// at production ones — so this type deliberately implements the *decode*, not the
/// splitting. It is shape-agnostic: every dimension is derived from the latent it is
/// handed, nothing is baked to `4 × 6 × 10`, and a caller that wants tiles can slice
/// the latent, call ``decode(_:)`` per slice and blend.
///
/// That is not merely a convenience boundary. Temporal padding here is *replication of
/// the edge frames*, so tiles are genuinely not independent — which is why they have to
/// be overlapped and blended, and why two tilings of the same clip produce different
/// numbers while both being correct.
///
/// ## Layout
///
/// MLX convolves `NDHWC` with weights `[out, kd, kh, kw, in]`; the checkpoint is
/// PyTorch's `NCDHW` with `[out, in, kd, kh, kw]`. The transposes happen at the two
/// ends of ``decode(_:)`` and once per kernel, and everything between is
/// channels-last — so **the channel axis is `-1` inside this type and `1` at its
/// boundary**. ``pixelNorm(_:eps:)`` reduces over channels; getting that axis wrong
/// normalises over width instead, which is a same-shaped tensor and a silently
/// different network.
public struct VideoVAEDecoder {

    /// `config.vae`, read from the checkpoint rather than declared.
    public struct Configuration: Equatable, Sendable {
        public let latentChannels: Int
        public let outChannels: Int
        /// 4. The final spatial expansion `unpatchify` performs, on top of the
        /// upsample ladder's 8×. Together they are the 32× the config's scale factors
        /// report.
        public let patchSize: Int
        /// False on this checkpoint: time is padded **symmetrically by replication**,
        /// one frame each side for a 3-tap kernel.
        public let causal: Bool
        /// The `decoder_blocks` ladder, already reversed into construction order.
        public let upBlocks: [Block]
        /// `decoder_base_channels`, 128.
        public let baseChannels: Int

        public enum Block: Equatable, Sendable {
            /// `res_x`: `n` resnets, no resampling.
            case res(layers: Int)
            /// `compress_time` / `compress_space` / `compress_all`: a depth-to-space
            /// upsample with this `(time, height, width)` stride.
            case upsample(stride: (Int, Int, Int))

            public static func == (a: Block, b: Block) -> Bool {
                switch (a, b) {
                case let (.res(x), .res(y)): return x == y
                case let (.upsample(x), .upsample(y)): return x == y
                default: return false
                }
            }
        }

        /// Channels at the bottleneck: `base × Π multipliers`, which for this ladder
        /// is `128 × 8 = 1024`.
        ///
        /// The ladder derives it from the block list rather than reading it off
        /// `conv_in`. ``VideoVAEDecoder/init(checkpoint:)`` then cross-checks the two
        /// against `conv_in.conv.weight`'s own out-channel count, so a config and a
        /// checkpoint that disagree fail at load.
        public func bottleneckChannels(_ weights: [String: MLXArray]) -> Int? {
            weights["conv_in.conv.weight"]?.shape[0]
        }

        /// Total upsampling, `(time, height, width)`, patch size included.
        ///
        /// `(8, 32, 32)` here: three temporal doublings, two spatial doublings from
        /// `compress_all` plus one from `compress_space`, times `patchSize` on the two
        /// spatial axes.
        public var scaleFactors: (time: Int, height: Int, width: Int) {
            var t = 1, h = patchSize, w = patchSize
            for block in upBlocks {
                if case let .upsample(stride) = block {
                    t *= stride.0; h *= stride.1; w *= stride.2
                }
            }
            return (t, h, w)
        }

        /// Frames the decoder produces from `latentFrames`.
        ///
        /// `scale × (F' − 1) + 1`, floored at 1. Derived here arithmetically and
        /// *independently* produced by the geometry of the three first-frame drops;
        /// ``decode(_:)`` requires the two to agree.
        public func targetFrames(latentFrames: Int) -> Int {
            max(scaleFactors.time * (latentFrames - 1) + 1, 1)
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

    /// Decoder tensors, keys stripped of `decoder.`.
    public let weights: [String: MLXArray]
    /// `per_channel_statistics.std-of-means`, `[latentChannels]`.
    public let stdOfMeans: MLXArray
    /// `per_channel_statistics.mean-of-means`, `[latentChannels]`.
    public let meanOfMeans: MLXArray
    public let config: Configuration

    public static let decoderPrefix = "decoder."
    public static let statisticsPrefix = "per_channel_statistics."

    /// Load the decoder half of a file that also carries the encoder.
    ///
    /// 170 tensors: 84 decoder, 84 encoder, 2 statistics. Checkpoints ship either
    /// monolithic, under a `vae.decoder.` prefix, or split, under a bare `decoder.`;
    /// this file is the split form, so the bare prefix is the one that matches. Loading
    /// the encoder as well would double the resident weights for nothing.
    public init(checkpoint url: URL) throws {
        let header = try SafetensorsHeader.read(from: url)
        self.config = try Self.readConfiguration(header)

        let all = try MLX.loadArrays(url: url)
        var decoder: [String: MLXArray] = [:]
        for (name, value) in all where name.hasPrefix(Self.decoderPrefix) {
            decoder[String(name.dropFirst(Self.decoderPrefix.count))] = value
        }
        guard !decoder.isEmpty else { throw Failure.missing(Self.decoderPrefix + "*") }

        // The config claims no attention, no channel-changing resnet, no noise and no
        // timestep conditioning. Believe the tensors, not the config: any of these
        // would mean this type silently skips a whole computation.
        for marker in ["to_qkv", "gamma", "norm3", "conv_shortcut", "scale_shift_table",
                       "per_channel_scale", "time_embedder"] {
            if let stray = decoder.keys.first(where: { $0.contains(marker) }) {
                throw Failure.unsupported(
                    "decoder carries \(stray), which this port does not implement; the "
                    + "config claims no attention, no channel-changing resnet, no noise "
                    + "injection and no timestep conditioning")
            }
        }
        // PixelNorm is unparameterised. A `norm*.weight` here would mean the norm is
        // not the one this type implements — the failure mode ``AudioVAEDecoder``
        // documents, where a GroupNorm port finds nothing to load and quietly runs
        // with ones and zeros.
        if let stray = decoder.keys.first(where: { $0.contains("norm") }) {
            throw Failure.unsupported(
                "decoder carries \(stray); norm_layer is \"pixel_norm\", which is "
                + "unparameterised")
        }
        self.weights = decoder

        guard let std = all[Self.statisticsPrefix + "std-of-means"],
              let mean = all[Self.statisticsPrefix + "mean-of-means"]
        else { throw Failure.missing(Self.statisticsPrefix + "{std,mean}-of-means") }
        guard std.size == config.latentChannels, mean.size == config.latentChannels else {
            throw Failure.shape(Self.statisticsPrefix + "*-of-means",
                                got: std.shape, expected: "[\(config.latentChannels)]")
        }
        self.stdOfMeans = std
        self.meanOfMeans = mean

        // The block ladder and the checkpoint must agree about the bottleneck.
        // `conv_in` widens the latent to `base × Π reduction factors`, and the ladder's
        // upsamples divide that back down to `base` — so the product of the reduction
        // factors recovered from the checkpoint's own upsample tensors has to be
        // exactly `conv_in`'s out-channels over `decoder_base_channels`. Reading
        // `decoder_blocks` in the wrong order is the error this type is most exposed
        // to; it changes neither tensor count nor any channel width, and this is the
        // check that sees it.
        guard let convIn = decoder["conv_in.conv.weight"] else {
            throw Failure.missing(Self.decoderPrefix + "conv_in.conv.weight")
        }
        guard convIn.shape[1] == config.latentChannels else {
            throw Failure.shape("conv_in.conv.weight", got: convIn.shape,
                                expected: "[bottleneck, \(config.latentChannels), 3, 3, 3]")
        }
        var channels = convIn.shape[0]
        for (index, block) in config.upBlocks.enumerated() {
            guard case let .upsample(stride) = block else { continue }
            guard let w = decoder["up_blocks.\(index).conv.conv.weight"] else {
                throw Failure.missing(
                    Self.decoderPrefix + "up_blocks.\(index).conv.conv.weight")
            }
            guard w.shape[1] == channels else {
                throw Failure.shape("up_blocks.\(index).conv.conv.weight", got: w.shape,
                                    expected: "[*, \(channels), 3, 3, 3] — the ladder "
                                        + "and the checkpoint disagree about this "
                                        + "block's input width, which is what reading "
                                        + "decoder_blocks in the wrong order looks like")
            }
            channels = w.shape[0] / (stride.0 * stride.1 * stride.2)
        }
        guard channels == config.baseChannels else {
            throw Failure.configuration(
                "the upsample ladder ends at \(channels) channels, but "
                + "decoder_base_channels is \(config.baseChannels)")
        }
    }

    /// Parse `config.vae` and reject anything this type does not implement, loudly, at
    /// load time rather than as a wrong number later.
    public static func readConfiguration(_ header: SafetensorsHeader) throws -> Configuration {
        guard let config = header.metadataJSON("config"),
              let vae = config["vae"] as? [String: Any]
        else { throw Failure.configuration("config.vae is absent") }

        // The two 2.5 video VAEs sit in the same directory and load through the same
        // protocol. Name the one this file implements.
        if let name = vae["_class_name"] as? String, name != "CausalVideoAutoencoder" {
            throw Failure.unsupported("_class_name is \"\(name)\"; this port implements "
                + "the convolutional CausalVideoAutoencoder only, not the diffusion VAE")
        }
        if let dims = vae["dims"] as? Int, dims != 3 {
            throw Failure.unsupported("dims is \(dims); this port implements 3D "
                + "convolutions only, not the 2D or DualConv3d variants")
        }
        if let norm = vae["norm_layer"] as? String, norm != "pixel_norm" {
            throw Failure.unsupported("norm_layer is \"\(norm)\"; this port implements "
                + "PixelNorm only, and GroupNorm would need affine weights the "
                + "checkpoint does not carry")
        }
        if let pad = vae["spatial_padding_mode"] as? String, pad != "zeros" {
            throw Failure.unsupported("spatial_padding_mode is \"\(pad)\"; this port "
                + "implements zero spatial padding only")
        }
        // Timestep conditioning would inject noise from a generator this port cannot
        // reproduce, and would add an ada-LN whose table is not in the file.
        if let ts = vae["timestep_conditioning"] as? Bool, ts {
            throw Failure.unsupported("timestep_conditioning is true; this port "
                + "implements no timestep conditioning and cannot reproduce the "
                + "reference's noise draw")
        }
        if let scaling = vae["scaling_factor"] as? Double, scaling != 1.0 {
            throw Failure.unsupported("scaling_factor is \(scaling); this port applies "
                + "per-channel statistics only")
        }

        guard let raw = vae["decoder_blocks"] as? [[Any]], !raw.isEmpty else {
            throw Failure.configuration("vae.decoder_blocks is absent or empty")
        }
        // Reversed: the ladder is built from `decoder_blocks` back to front.
        var blocks: [Configuration.Block] = []
        for entry in raw.reversed() {
            guard entry.count == 2, let name = entry[0] as? String else {
                throw Failure.configuration("decoder_blocks entry is not [name, params]")
            }
            let params = (entry[1] as? [String: Any]) ?? ["num_layers": entry[1]]
            switch name {
            case "res_x":
                guard let layers = params["num_layers"] as? Int else {
                    throw Failure.configuration("res_x carries no num_layers")
                }
                blocks.append(.res(layers: layers))
            case "compress_time": blocks.append(.upsample(stride: (2, 1, 1)))
            case "compress_space": blocks.append(.upsample(stride: (1, 2, 2)))
            case "compress_all":
                // `residual` defaults to false and is false here. A residual
                // depth-to-space adds a duplicated skip path this type does not have.
                if let residual = params["residual"] as? Bool, residual {
                    throw Failure.unsupported("compress_all has residual: true; this "
                        + "port implements the non-residual DepthToSpaceUpsample only")
                }
                blocks.append(.upsample(stride: (2, 2, 2)))
            default:
                throw Failure.unsupported("decoder block \"\(name)\" is not implemented; "
                    + "this checkpoint's ladder is res_x / compress_time / "
                    + "compress_space / compress_all and carries no attention or res_x_y")
            }
        }

        func int(_ key: String, _ fallback: Int) -> Int { (vae[key] as? Int) ?? fallback }
        return Configuration(
            latentChannels: int("latent_channels", 128),
            outChannels: int("out_channels", 3),
            patchSize: int("patch_size", 4),
            causal: (vae["causal_decoder"] as? Bool) ?? false,
            upBlocks: blocks,
            baseChannels: int("decoder_base_channels", 128))
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(Self.decoderPrefix + name) }
        return w
    }

    // MARK: - Decode

    /// Latent `[B, latentChannels, F', H', W']` → video `[B, 3, F, H, W]`, both `NCDHW`.
    ///
    /// The output is the decoder's raw range, roughly `[-1, 1]`. ``rgb(_:)`` is the
    /// separate `[0, 1]` tail.
    ///
    /// **The latent is cast to the weight dtype (bf16) on the way in.** A latent handed
    /// in as fp32 would make MLX promote every convolution to fp32, spending memory and
    /// time on a precision the bf16 weights do not carry. The cast is lossless for any
    /// latent that came out of the bf16 pipeline.
    public func decode(_ latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 5, latent.shape[1] == config.latentChannels else {
            throw Failure.shape("latent", got: latent.shape,
                                expected: "[B, \(config.latentChannels), F, H, W]")
        }
        let dtype = try weight("conv_in.conv.weight").dtype
        let latentFrames = latent.shape[2]
        let targetFrames = config.targetFrames(latentFrames: latentFrames)
        let scale = config.scaleFactors

        var h = denormalize(latent.asType(dtype))
        // NCDHW -> NDHWC. Everything below is channels-last; channel axis is -1.
        h = h.transposed(0, 2, 3, 4, 1)

        h = try conv(h, "conv_in")

        for (index, block) in config.upBlocks.enumerated() {
            switch block {
            case let .res(layers):
                for layer in 0 ..< layers {
                    h = try resnet(h, "up_blocks.\(index).res_blocks.\(layer)")
                }
            case let .upsample(stride):
                h = try upsample(h, stem: "up_blocks.\(index)", stride: stride)
            }
        }

        // `conv_norm_out` is PixelNorm and `conv_act` is SiLU. `timestep_conditioning`
        // is false, so nothing sits between them — no ada-LN, and the video leaves
        // unsquashed (there is no `tanh_out` on this decoder at all).
        h = try conv(silu(pixelNorm(h)), "conv_out")

        h = h.transposed(0, 4, 1, 2, 3)                     // NDHWC -> NCDHW
        h = unpatchify(h)

        // The geometric result and the arithmetic target are derived two different
        // ways — three first-frame drops versus `8(F' − 1) + 1` — so a disagreement
        // means a drop landed on the wrong side or was skipped, and neither a shape
        // assertion downstream nor the frame count alone would catch it.
        guard h.shape[1] == config.outChannels, h.shape[2] == targetFrames,
              h.shape[3] == latent.shape[3] * scale.height,
              h.shape[4] == latent.shape[4] * scale.width
        else {
            throw Failure.shape("decoded video", got: h.shape,
                                expected: "[B, \(config.outChannels), \(targetFrames), "
                                    + "\(latent.shape[3] * scale.height), "
                                    + "\(latent.shape[4] * scale.width)]")
        }
        return h
    }

    /// The display tail: `[B, C, F, H, W]` → `[F, H, W, C]` in `[0, 1]`, by
    /// `c f h w -> f h w c`, `(x + 1) × 0.5`, clamp.
    ///
    /// **Batch element 0 only** — the rest are dropped, so this is not a reshape and a
    /// `B > 1` decode loses everything but the first clip here.
    ///
    /// It stays in the decoder's dtype rather than widening. `x + 1` lands in `[1, 2)`,
    /// where a bf16 ULP is `2^-8 = 3.906e-03`; that quantum, and not this step's
    /// arithmetic, is what bounds the output, so widening here would buy a precision
    /// the frames do not have.
    ///
    /// The clamp is usually inert; typical output sits well inside `[0, 1]`. It is
    /// applied anyway, because omitting it would only ever show up on the inputs where
    /// clipping matters most.
    public func rgb(_ decoded: MLXArray) -> MLXArray {
        let video = decoded[0].transposed(1, 2, 3, 0)       // c f h w -> f h w c
        let one = MLXArray(Float(1)).asType(video.dtype)
        let half = MLXArray(Float(0.5)).asType(video.dtype)
        return MLX.clip((video + one) * half,
                        min: MLXArray(Float(0)).asType(video.dtype),
                        max: one)
    }

    /// Undo the encoder's per-channel normalisation: `x * std + mean`.
    ///
    /// A plain per-latent-channel affine — 128 statistics against 128 latent channels,
    /// reshaped `[1, C, 1, 1, 1]`. Unlike ``AudioVAEDecoder/denormalize(_:)``, where
    /// `[128]` indexes a flattened `(channel, mel column)` pair and the reshape has an
    /// orientation to get wrong, there is nothing to transpose here.
    ///
    /// The statistics are cast to the latent's dtype rather than the latent widened to
    /// theirs, so this stays in bf16.
    public func denormalize(_ latent: MLXArray) -> MLXArray {
        let channels = latent.shape[1]
        let std = stdOfMeans.reshaped([1, channels, 1, 1, 1]).asType(latent.dtype)
        let mean = meanOfMeans.reshaped([1, channels, 1, 1, 1]).asType(latent.dtype)
        return latent * std + mean
    }

    // MARK: - Blocks

    /// `ResnetBlock3D`: `norm → silu → conv1 → norm → silu → conv2`, plus the skip.
    ///
    /// The skip is the **raw input**. Every resnet in this decoder has
    /// `in_channels == out_channels` — the channel counts change inside
    /// ``upsample(_:stem:stride:)``'s rearrange, never in a resnet — so `conv_shortcut`
    /// and `norm3` are both identities. That is read off the checkpoint (84 tensors,
    /// all `conv{1,2}.conv.{weight,bias}` plus four for `conv_in`/`conv_out`), and
    /// ``init(checkpoint:)`` refuses to load a file that contradicts it.
    ///
    /// `inject_noise` and `timestep_conditioning` are false, `dropout` is 0.0 and the
    /// model is in `eval`, so none of those branches exists.
    public func resnet(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        var h = try conv(silu(pixelNorm(x)), stem + ".conv1")
        h = try conv(silu(pixelNorm(h)), stem + ".conv2")
        return x + h
    }

    /// `DepthToSpaceUpsample`: conv, depth-to-space, then drop the **first** frame.
    ///
    /// Two things here keep every shape and change every value:
    ///
    /// * **The channel axis decomposes c-major, then time, then height, then width.**
    ///   `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)` means flat channel
    ///   `((c·P1 + p1)·P2 + p2)·P3 + p3`. Any other ordering of the three sub-axes is
    ///   the same tensor with the sub-voxels permuted — a checkerboard that no shape
    ///   check and no channel count can see.
    /// * **The drop is frame 0, not frame `n−1`**, and only when the temporal stride is
    ///   2. It is what turns `4 → 8` into `4 → 7`, and three of them turn `8F'` into
    ///   `8F' − 7`. Dropping the last frame gives the identical shape and shifts every
    ///   frame by one — invisible to the frame-count check in ``decode(_:)``, which
    ///   would still see 25.
    ///
    /// `residual` is false for every block in this config (the `compress_all` entries
    /// carry only a `multiplier`), so there is no duplicated skip path;
    /// ``readConfiguration(_:)`` rejects a config that sets it.
    ///
    /// Axes are `NDHWC` here: 1 is time, 2 height, 3 width, 4 channels.
    public func upsample(_ x: MLXArray, stem: String,
                         stride: (Int, Int, Int)) throws -> MLXArray {
        var y = try conv(x, stem + ".conv")
        let (p1, p2, p3) = stride
        let b = y.shape[0], d = y.shape[1], h = y.shape[2], w = y.shape[3]
        let outChannels = y.shape[4] / (p1 * p2 * p3)

        // NDHWC with the channel axis split c-major: [b, d, h, w, c, p1, p2, p3].
        y = y.reshaped([b, d, h, w, outChannels, p1, p2, p3])
        // -> [b, d, p1, h, p2, w, p3, c]
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        y = y.reshaped([b, d * p1, h * p2, w * p3, outChannels])

        if p1 == 2 { y = y[0..., 1...] }                    // axis 1 is time in NDHWC
        return y
    }

    /// `unpatchify`: `[B, C·q·r, F, H, W]` → `[B, C, F, H·q, W·r]`, `NCDHW`.
    ///
    /// `b (c p r q) f h w -> b c (f p) (h q) (w r)`, with `p = 1` and `q = r = 4`.
    ///
    /// The channel order is `c, p, r, q` while the spatial pairing is `(h q)` and
    /// `(w r)`: **`r` — width — is the more significant sub-axis and `q` — height —
    /// the less significant one.** Swapping them transposes every 4×4 patch. Same
    /// shape, same per-channel statistics, a visibly different image; only the values
    /// say which way round it goes.
    ///
    /// `patch_size_t` is 1 everywhere in this decoder, so the `p` axis is a
    /// degenerate 1 and the frame count is untouched.
    public func unpatchify(_ x: MLXArray) -> MLXArray {
        let q = config.patchSize, r = config.patchSize
        let b = x.shape[0], f = x.shape[2], h = x.shape[3], w = x.shape[4]
        let outChannels = x.shape[1] / (q * r)
        // [b, c, r, q, f, h, w] -> [b, c, f, h, q, w, r]
        var y = x.reshaped([b, outChannels, r, q, f, h, w])
        y = y.transposed(0, 1, 4, 5, 3, 6, 2)
        return y.reshaped([b, outChannels, f, h * q, w * r])
    }

    // MARK: - Primitives

    /// A `CausalConv3d` from the checkpoint: `<stem>.conv.{weight,bias}`.
    ///
    /// The doubled `conv` in the key is the wrapper's inner convolution; the wrapper
    /// holds no parameters, only the temporal padding — which is the whole point of
    /// it, and which is **replication, not zeros**. See
    /// ``temporalPad(_:timeKernel:)``.
    ///
    /// `causal_decoder` is **false** on this checkpoint, so a 3-tap kernel gets one
    /// copy of frame 0 in front and one copy of the last frame behind. Zero-padding
    /// time instead produces the same shape and pulls the first and last output frames
    /// toward the origin — an error that is largest exactly at the clip boundaries a
    /// viewer looks at first.
    ///
    /// The inner convolution then pads `(0, k/2, k/2)` with zeros, so **height and
    /// width are zero-padded symmetrically while time is replicate-padded outside the
    /// convolution.** Two padding modes on two axes of the same kernel. MLX's `conv3d`
    /// takes one symmetric padding per axis and no mode, which happens to express
    /// exactly the spatial half; the temporal half is materialised with `concatenated`
    /// first, and the depth padding is left at 0.
    ///
    /// Weight layout: the checkpoint is PyTorch `[out, in, kd, kh, kw]` and MLX
    /// convolves with `[out, kd, kh, kw, in]` over `NDHWC`, hence `(0, 2, 3, 4, 1)`.
    public func conv(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        let w = try weight(stem + ".conv.weight")
        let b = try weight(stem + ".conv.bias")
        guard w.ndim == 5 else {
            throw Failure.shape(stem + ".conv.weight", got: w.shape,
                                expected: "[out, in, kd, kh, kw]")
        }
        let kd = w.shape[2], kh = w.shape[3], kw = w.shape[4]

        let input = temporalPad(x.asType(w.dtype), timeKernel: kd)
        let kernel = w.transposed(0, 2, 3, 4, 1)
        return MLX.conv3d(input, kernel, padding: IntOrTriple((0, kh / 2, kw / 2)))
            + b.asType(w.dtype)
    }

    /// `CausalConv3d`'s temporal padding: replicate the edge frames.
    ///
    /// Symmetric — `(k−1)/2` copies of frame 0 in front and of the last frame behind —
    /// when `causal` is false, which it is here; the whole `k−1` extent in front and
    /// nothing behind when it is true. Both are implemented because the class supports
    /// both and the flag is read from the config, and because the two differ by
    /// exactly one frame of look-ahead: a same-shaped tensor in which every frame can
    /// see its own future.
    public func temporalPad(_ x: MLXArray, timeKernel: Int) -> MLXArray {
        let pad = timeKernel - 1
        guard pad > 0 else { return x }
        if config.causal {
            let first = MLX.repeated(x[0..., ..<1], count: pad, axis: 1)
            return MLX.concatenated([first, x], axis: 1)
        }
        let half = pad / 2
        guard half > 0 else { return x }
        let first = MLX.repeated(x[0..., ..<1], count: half, axis: 1)
        let last = MLX.repeated(x[0..., (x.shape[1] - 1)...], count: half, axis: 1)
        return MLX.concatenated([first, x, last], axis: 1)
    }

    /// `x / sqrt(mean(x², over channels) + eps)`, no affine, channel axis last.
    ///
    /// **The epsilon is 1e-8, not the audio VAE's 1e-6.**
    /// ``AudioVAEDecoder/pixelNorm(_:eps:)`` uses the other value, and copying the
    /// constant across is the mistake this comment exists to prevent.
    ///
    /// The dtype split is the one `DiTModules` documents, applied to a reduction:
    ///
    /// * The **square** stays in the tensor's dtype, so its bf16 result is what the
    ///   mean then sees.
    /// * The **reduction widens to fp32 and casts back**: 1024 channels at the
    ///   bottleneck is enough summands for a bf16 accumulator to lose real precision,
    ///   and MLX's accumulator is not something to leave implicit at a seam.
    /// * `sqrt(· + eps)` runs in fp32 and casts back, one rounding for the whole
    ///   elementwise step.
    public func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
        let squared = x * x
        let meanSquare = squared.asType(.float32).mean(axis: -1, keepDims: true)
            .asType(x.dtype)                                    // EXPERIMENT contract 16
        let rms = MLX.sqrt(meanSquare.asType(.float32) + eps).asType(x.dtype)
        return x / rms
    }

    /// `x * sigmoid(x)`, evaluated in fp32 and cast back.
    ///
    /// An elementwise nonlinearity on a bf16 tensor is evaluated at fp32 and rounded
    /// once, not evaluated in bf16 throughout. There are **45** SiLUs on this decoder's
    /// path — 22 resnets × 2, plus `conv_act` — so a hand-rolled bf16 sigmoid here is
    /// visible in the decoded frames.
    public func silu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * MLX.sigmoid(f)).asType(x.dtype)
    }
}

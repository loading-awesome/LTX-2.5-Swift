// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The **conv** video VAE encoder: RGB frames → video latent.
///
/// `[B, 3, F, H, W]` in — pixels already in `[-1, 1]` — `[B, 128, F', H', W']` out, the
/// per-channel-normalised **means**. It is the front door ``VideoVAEDecoder`` is the back
/// door of: image-to-video and video-to-video conditioning both need a pixel clip turned
/// into a latent.
///
/// ## The one thing to know: there is no sampling path
///
/// `conv_out` produces **129** channels against a 128-channel latent, which looks like a
/// variational head with a shared log-variance, and it is one. The log-variance is
/// discarded: the head reduces to "take the first 128 channels", no distribution object
/// is ever constructed, and no `randn` is ever drawn.
///
/// So ``encode(_:)`` is not "the deterministic option among two" — it is the *only*
/// behaviour there is, and adding a sampler here would invent a noise draw that makes
/// i2v non-reproducible for a fixed seed for no reason. ``logVariance(_:)`` exposes the
/// discarded channel for diagnostics and is deliberately not wired into anything.
///
/// ## What the architecture actually is
///
/// Read off `__metadata__.config.vae`, never declared. The traps here are not the
/// decoder's traps mirrored — a VAE encoder is not the decoder run backwards, and four
/// of these five differ from the decoder in a way that keeps every shape:
///
/// * **The encoder is CAUSAL; the decoder is not.** `causal_decoder` is `false` and
///   ``VideoVAEDecoder`` correctly replicate-pads time *symmetrically*. Every encoder
///   convolution is causal **structurally**, not by config flag, so carrying
///   `causal_decoder` across is the single easiest mistake available here. Causal means
///   the whole `k − 1` extent is frame 0 repeated in front and **nothing behind**: same
///   output shape, and a symmetric pad would let every latent frame see one pixel frame
///   of its own future.
/// * **`encoder_blocks` is applied FORWARDS**, where the decoder's ladder is built from
///   `decoder_blocks` reversed. Reversing it here — the reflex after writing the decoder
///   — gives a ladder with the same tensor count and different channel widths, which
///   *does* fail at load, but only because ``init(checkpoint:)`` checks every shape.
/// * **The downsamples are `*_res`: space-to-depth with an averaged skip.**
///   `compress_space_res` / `compress_time_res` / `compress_all_res` build
///   `SpaceToDepthDownsample`, not a strided convolution. The tell is in the keys:
///   `down_blocks.1.conv.conv.weight` has **two** levels of `conv` (the module's inner
///   `CausalConv3d`, whose own inner convolution is the second), where a bare
///   `compress_space` would be a `CausalConv3d` directly and give
///   `down_blocks.1.conv.weight`. Its skip path is not a plain identity: the *input* is
///   space-to-depth'd too and then **mean-pooled in groups** down to the output width.
///   Dropping that skip keeps every shape.
/// * **Channel widths start from `latent_channels`, not `encoder_base_channels`.**
///   `conv_in` widens the patchified 48 to 128, the latent width, and every later width
///   is that scaled by the downsamples' multipliers. The config *also* carries
///   `encoder_base_channels: 128`, which nothing on this path reads. The two agree
///   numerically on this checkpoint, which is exactly why reading the wrong one would
///   keep working until it silently did not.
/// * **PixelNorm's epsilon is 1e-8** — same as the video decoder, and *not* the audio
///   VAE's 1e-6. Copying that constant across from ``AudioVAEDecoder`` is the obvious
///   move and it is wrong.
///
/// ## The ladder
///
/// ```
/// pixels                            [1,   3, 97, 384, 640]
/// patchify(4)                       [1,  48, 97,  96, 160]
/// conv_in                           [1, 128, 97,  96, 160]
/// down_blocks.0  res_x 4            [1, 128, 97,  96, 160]
/// down_blocks.1  compress_space_res [1,  64, 97,  96, 160] -> s2d [1, 256, 97, 48, 80]
/// down_blocks.2  res_x 6            [1, 256, 97,  48,  80]
/// down_blocks.3  compress_time_res  97 -> 98 -> s2d          [1, 512, 49, 48, 80]
/// down_blocks.4  res_x 4            [1, 512, 49,  48,  80]
/// down_blocks.5  compress_all_res   49 -> 50 -> s2d          [1,1024, 25, 24, 40]
/// down_blocks.6  res_x 2            [1,1024, 25,  24,  40]
/// down_blocks.7  compress_all_res   25 -> 26 -> s2d          [1,1024, 13, 12, 20]
/// down_blocks.8  res_x 2            [1,1024, 13,  12,  20]
/// conv_out                          [1, 129, 13,  12,  20]
/// means = [:128], normalise         [1, 128, 13,  12,  20]
/// ```
///
/// `97 → 49 → 25 → 13` is `(F − 1)/8 + 1`, and it comes from **duplicating frame 0**
/// before each temporal space-to-depth, not from a rate that divides. It is the mirror
/// of the decoder's first-frame *drop*, and it is what makes the VAE causal in time: the
/// first latent frame is built from one pixel frame (itself, duplicated) where every
/// later one covers eight. A single image encodes to exactly one latent frame —
/// `1 → 2 → 1` at each of the three temporal stages — which is why contract 15's
/// keyframe marker can exist at all.
///
/// ## Tiling
///
/// A tiled encode splits on the *latent* grid, maps each interval back to pixels,
/// encodes each tile through this same forward pass and blends with trapezoidal masks.
/// As with the decoder, tile count is a memory strategy and not a contract, so this type
/// implements the *encode* and leaves the splitting to a caller. Tiles are genuinely not
/// independent here either — causal padding replicates each tile's own first frame — so
/// tiling needs ≥16 frames and ≥64 pixels of overlap to be usable at all.
///
/// ## Layout
///
/// Same convention as ``VideoVAEDecoder``: MLX convolves `NDHWC` with weights
/// `[out, kd, kh, kw, in]` where the checkpoint is PyTorch `NCDHW` with
/// `[out, in, kd, kh, kw]`. ``patchify(_:)`` runs in `NCDHW` — the only layout in which
/// its index algebra is checkable by eye — and everything from ``conv(_:_:)`` onward is
/// channels-last, so **the channel axis is `-1` inside this type and `1` at its
/// boundary**.
///
/// ``encode(_:)`` takes a tensor, in `NCDHW`, rather than a path.
/// ``pixels(fromRGB:)`` is the named conversion from ``VideoVAEDecoder/rgb(_:)``'s
/// channels-last `[F, H, W, 3]`, because a silent layout assumption at that seam is the
/// transposed-patchify failure wearing a different costume.
public struct VideoVAEEncoder {

    /// `config.vae`, read from the checkpoint rather than declared.
    ///
    /// Deliberately **not** ``VideoVAEDecoder/Configuration``, which cannot express this
    /// half: it has no channel multipliers (the decoder's widths are recovered from the
    /// upsample strides alone, the encoder's are not), no log-variance mode, its block
    /// list is already reversed into the decoder's construction order, and its `causal`
    /// is `causal_decoder` — `false` — where this encoder is structurally causal.
    /// Reusing it would mean four fields that are silently about the other half.
    public struct Configuration: Equatable, Sendable {
        /// `latent_channels`, 128. Also the width `conv_in` produces: the channel
        /// ladder starts here rather than at `encoder_base_channels`.
        public let latentChannels: Int
        /// `in_channels`, 3 — RGB, *before* patchify. The first convolution sees
        /// ``patchedInputChannels`` = 48.
        public let inChannels: Int
        /// 4. Space-to-depth applied to `(H, W)` before `conv_in`, on top of the ladder's
        /// 8× — together the 32× the config's scale factors report.
        public let patchSize: Int
        /// `encoder_blocks`, in the config's own order.
        public let downBlocks: [Block]
        /// `latent_log_var`. Governs `conv_out`'s width and nothing else that this type
        /// uses, because the log-variance is discarded.
        public let logVariance: LogVariance

        public enum Block: Equatable, Sendable {
            /// `res_x`: `n` resnets, no resampling, `in == out`.
            case res(layers: Int)
            /// `compress_*_res`: a `SpaceToDepthDownsample` with this `(time, height,
            /// width)` stride, widening the channel count by `multiplier`.
            case downsample(stride: (Int, Int, Int), multiplier: Int)

            public static func == (a: Block, b: Block) -> Bool {
                switch (a, b) {
                case let (.res(x), .res(y)): return x == y
                case let (.downsample(sx, mx), .downsample(sy, my)):
                    return sx == sy && mx == my
                default: return false
                }
            }
        }

        /// How `conv_out`'s extra channels are meant to be read.
        ///
        /// Only ``uniform`` is implemented, because it is what this checkpoint declares
        /// and the others cannot be tested against anything here. They are named so an
        /// unsupported checkpoint fails with the mode in the message.
        public enum LogVariance: String, Equatable, Sendable {
            /// `N` means plus **one** shared log-variance channel: 129 for 128.
            case uniform
            /// `2N` channels, a per-channel log-variance.
            case perChannel = "per_channel"
            /// `N + 1`, with the extra channel dropped and replaced by a literal −30.
            case constant
            /// `N`, no variance at all: the head is the means and nothing else.
            case none
        }

        /// `in_channels × patch_size²`: 48, and `48 = 3 × 4 × 4` is RGB space-to-depth'd
        /// by 4 with no temporal patching (`patch_size_t` is 1).
        public var patchedInputChannels: Int { inChannels * patchSize * patchSize }

        /// `conv_out`'s output width, derived from the mode rather than read off the
        /// tensor, so ``VideoVAEEncoder/init(checkpoint:)`` can cross-check the two.
        public var headChannels: Int {
            switch logVariance {
            case .uniform, .constant: return latentChannels + 1
            case .perChannel: return latentChannels * 2
            case .none: return latentChannels
            }
        }

        /// Total downsampling, `(time, height, width)`, patch size included.
        ///
        /// `(8, 32, 32)` here. A `compress_*` block contributes a factor of 2 per
        /// resampled axis **independent of any multiplier**, which is why this reads the
        /// stride and not the channel widths.
        public var scaleFactors: (time: Int, height: Int, width: Int) {
            var t = 1, h = patchSize, w = patchSize
            for block in downBlocks {
                if case let .downsample(stride, _) = block {
                    t *= stride.0; h *= stride.1; w *= stride.2
                }
            }
            return (t, h, w)
        }

        /// Latent frames produced from `pixelFrames`, `(F − 1)/8 + 1`.
        ///
        /// The inverse of ``VideoVAEDecoder/Configuration/targetFrames(latentFrames:)``,
        /// and only exact on the `8k + 1` lattice — see ``isOnLattice(pixelFrames:)``.
        public func latentFrames(pixelFrames: Int) -> Int {
            max((pixelFrames - 1) / scaleFactors.time + 1, 1)
        }

        /// Whether `F` sits on the `8k + 1` lattice the temporal stages require.
        public func isOnLattice(pixelFrames: Int) -> Bool {
            pixelFrames >= 1 && (pixelFrames - 1) % scaleFactors.time == 0
        }

        /// The channel width at the input of every block, and after the last one.
        ///
        /// `count + 1` entries. Derived from the block list alone — the encoder's widths
        /// are `latent_channels` scaled by each downsample's `multiplier`, which is
        /// **not** recoverable from the strides the way the decoder's are, and is
        /// therefore the thing ``VideoVAEEncoder/init(checkpoint:)`` cross-checks against
        /// every tensor it loads.
        public var channelLadder: [Int] {
            var widths = [latentChannels]
            for block in downBlocks {
                if case let .downsample(_, multiplier) = block {
                    widths.append(widths[widths.count - 1] * multiplier)
                } else {
                    widths.append(widths[widths.count - 1])
                }
            }
            return widths
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)
        case unsupported(String)
        case configuration(String)
        case unconsumed([String])

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .unsupported(why): return why
            case let .configuration(why): return "checkpoint config: \(why)"
            case let .unconsumed(names):
                return "\(names.count) encoder tensor(s) the module tree never reads, "
                    + "so the tree is wrong in a way that would still produce a "
                    + "plausible latent: \(names.sorted().prefix(8).joined(separator: ", "))"
            }
        }
    }

    /// Encoder tensors, keys stripped of `encoder.`.
    public let weights: [String: MLXArray]
    /// `per_channel_statistics.std-of-means`, `[latentChannels]`.
    public let stdOfMeans: MLXArray
    /// `per_channel_statistics.mean-of-means`, `[latentChannels]`.
    public let meanOfMeans: MLXArray
    public let config: Configuration

    public static let encoderPrefix = "encoder."
    public static let statisticsPrefix = "per_channel_statistics."

    /// Load the encoder half of a file that also carries the decoder.
    ///
    /// 170 tensors: 84 encoder, 84 decoder, 2 statistics. Checkpoints ship either
    /// monolithic, under a `vae.encoder.` prefix, or split, under a bare `encoder.`;
    /// this file is the split form. The decoder half is left unread rather than loaded
    /// and ignored, so an encode-only caller does not pay for it twice.
    ///
    /// Every tensor is checked against the shape the block ladder says it must have,
    /// **and** the set of keys the ladder touches is checked against the set the file
    /// carries. The second half is the one that matters: a module tree that is wrong but
    /// self-consistent — a reversed ladder, a missing skip, a `compress_all` read as
    /// `compress_all_res` — leaves tensors unused, runs to completion, and emits a latent
    /// that looks entirely reasonable.
    public init(checkpoint url: URL) throws {
        let header = try SafetensorsHeader.read(from: url)
        self.config = try Self.readConfiguration(header)

        let all = try MLX.loadArrays(url: url)
        var encoder: [String: MLXArray] = [:]
        for (name, value) in all where name.hasPrefix(Self.encoderPrefix) {
            encoder[String(name.dropFirst(Self.encoderPrefix.count))] = value
        }
        guard !encoder.isEmpty else { throw Failure.missing(Self.encoderPrefix + "*") }

        // The config claims no attention, no channel-changing resnet, no noise and no
        // timestep conditioning. Believe the tensors, not the config: each of these
        // would be a whole computation this type silently skips.
        for marker in ["to_qkv", "gamma", "norm3", "conv_shortcut", "scale_shift_table",
                       "per_channel_scale", "time_embedder", "quant_conv"] {
            if let stray = encoder.keys.first(where: { $0.contains(marker) }) {
                throw Failure.unsupported(
                    "encoder carries \(stray), which this port does not implement; the "
                    + "config claims no attention, no channel-changing resnet, no noise "
                    + "injection, no timestep conditioning and no quant conv")
            }
        }
        // PixelNorm is unparameterised. A `norm*.weight` would mean the norm is not the
        // one this type implements — ``AudioVAEDecoder``'s documented failure mode, where
        // a GroupNorm port finds nothing to load and quietly runs with ones and zeros.
        if let stray = encoder.keys.first(where: { $0.contains("norm") }) {
            throw Failure.unsupported(
                "encoder carries \(stray); norm_layer is \"pixel_norm\", which is "
                + "unparameterised")
        }
        self.weights = encoder

        guard let std = all[Self.statisticsPrefix + "std-of-means"],
              let mean = all[Self.statisticsPrefix + "mean-of-means"]
        else { throw Failure.missing(Self.statisticsPrefix + "{std,mean}-of-means") }
        guard std.size == config.latentChannels, mean.size == config.latentChannels else {
            throw Failure.shape(Self.statisticsPrefix + "*-of-means",
                                got: std.shape, expected: "[\(config.latentChannels)]")
        }
        self.stdOfMeans = std
        self.meanOfMeans = mean

        // Walk the ladder, demanding each tensor at exactly the width the config says,
        // and record what was touched.
        var consumed = Set<String>()
        func require(_ name: String, _ expected: [Int]) throws {
            guard let w = encoder[name] else {
                throw Failure.missing(Self.encoderPrefix + name)
            }
            guard w.shape == expected else {
                throw Failure.shape(name, got: w.shape, expected: "\(expected)")
            }
            consumed.insert(name)
        }
        func requireConv(_ stem: String, out: Int, into input: Int) throws {
            try require(stem + ".conv.weight", [out, input, 3, 3, 3])
            try require(stem + ".conv.bias", [out])
        }

        try requireConv("conv_in", out: config.latentChannels,
                        into: config.patchedInputChannels)

        let widths = config.channelLadder
        for (index, block) in config.downBlocks.enumerated() {
            let input = widths[index]
            switch block {
            case let .res(layers):
                for layer in 0 ..< layers {
                    let stem = "down_blocks.\(index).res_blocks.\(layer)"
                    try requireConv(stem + ".conv1", out: input, into: input)
                    try requireConv(stem + ".conv2", out: input, into: input)
                }
            case let .downsample(stride, _):
                let output = widths[index + 1]
                let fold = stride.0 * stride.1 * stride.2
                guard output % fold == 0 else {
                    throw Failure.configuration(
                        "down_blocks.\(index) folds \(fold) sub-voxels into \(output) "
                        + "channels, which does not divide")
                }
                // The inner convolution produces `out / fold` channels and the
                // space-to-depth multiplies them back up. Reading this as a strided
                // `compress_*` instead would expect `[out, in, 3, 3, 3]` here — a
                // different shape, and the reason this check catches that mis-read.
                try requireConv("down_blocks.\(index).conv", out: output / fold,
                                into: input)
                guard (input * fold) % output == 0 else {
                    throw Failure.configuration(
                        "down_blocks.\(index)'s skip folds \(input * fold) channels into "
                        + "\(output), which does not group evenly")
                }
            }
        }
        try requireConv("conv_out", out: config.headChannels,
                        into: widths[widths.count - 1])

        let unconsumed = Set(encoder.keys).subtracting(consumed)
        guard unconsumed.isEmpty else { throw Failure.unconsumed(Array(unconsumed)) }
    }

    /// Parse `config.vae` and reject anything this type does not implement, loudly, at
    /// load time rather than as a wrong number later.
    public static func readConfiguration(_ header: SafetensorsHeader) throws -> Configuration {
        guard let config = header.metadataJSON("config"),
              let vae = config["vae"] as? [String: Any]
        else { throw Failure.configuration("config.vae is absent") }

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
        // The two padding modes genuinely differ on this path: zeros on `(H, W)`,
        // first-frame replication on `F`.
        if let pad = vae["spatial_padding_mode"] as? String, pad != "zeros" {
            throw Failure.unsupported("spatial_padding_mode is \"\(pad)\"; this port "
                + "implements zero spatial padding only")
        }
        if let scaling = vae["scaling_factor"] as? Double, scaling != 1.0 {
            throw Failure.unsupported("scaling_factor is \(scaling); this port applies "
                + "per-channel statistics only")
        }
        if let quant = vae["use_quant_conv"] as? Bool, quant {
            throw Failure.unsupported("use_quant_conv is true; VideoEncoder builds no "
                + "quant conv and this port has nowhere to put its weights")
        }

        let rawMode = (vae["latent_log_var"] as? String) ?? "uniform"
        guard let mode = Configuration.LogVariance(rawValue: rawMode) else {
            throw Failure.configuration("latent_log_var is \"\(rawMode)\", which is not "
                + "one of per_channel / uniform / constant / none")
        }
        guard mode == .uniform else {
            throw Failure.unsupported("latent_log_var is \"\(rawMode)\"; this port "
                + "implements \"uniform\" — N means plus one shared log-variance channel "
                + "— which is what this checkpoint declares, and there is no checkpoint "
                + "here to check the other modes' head width against")
        }

        guard let raw = vae["encoder_blocks"] as? [[Any]], !raw.isEmpty else {
            throw Failure.configuration("vae.encoder_blocks is absent or empty")
        }
        // FORWARDS: `encoder_blocks` is taken in order, unlike the decoder's ladder,
        // which is built from `decoder_blocks` reversed.
        var blocks: [Configuration.Block] = []
        for entry in raw {
            guard entry.count == 2, let name = entry[0] as? String else {
                throw Failure.configuration("encoder_blocks entry is not [name, params]")
            }
            let params = (entry[1] as? [String: Any]) ?? ["num_layers": entry[1]]
            func multiplier() -> Int { (params["multiplier"] as? Int) ?? 2 }
            switch name {
            case "res_x":
                guard let layers = params["num_layers"] as? Int else {
                    throw Failure.configuration("res_x carries no num_layers")
                }
                blocks.append(.res(layers: layers))
            case "compress_time_res":
                blocks.append(.downsample(stride: (2, 1, 1), multiplier: multiplier()))
            case "compress_space_res":
                blocks.append(.downsample(stride: (1, 2, 2), multiplier: multiplier()))
            case "compress_all_res":
                blocks.append(.downsample(stride: (2, 2, 2), multiplier: multiplier()))
            case "compress_time", "compress_space", "compress_all", "compress_all_x_y":
                // A bare `compress_*` is a *strided* CausalConv3d with `out == in`, whose
                // key is `down_blocks.N.conv.weight` — one `conv` level, not two — and
                // which has no averaged skip. Nothing about it is expressible by
                // pretending it is the `_res` variant, so say so rather than load it.
                throw Failure.unsupported("encoder block \"\(name)\" is a strided "
                    + "convolution, not a SpaceToDepthDownsample; this port implements "
                    + "the compress_*_res family only, which is what this checkpoint "
                    + "carries")
            default:
                throw Failure.unsupported("encoder block \"\(name)\" is not implemented; "
                    + "this checkpoint's ladder is res_x / compress_space_res / "
                    + "compress_time_res / compress_all_res and carries no attention or "
                    + "res_x_y")
            }
        }

        func int(_ key: String, _ fallback: Int) -> Int { (vae[key] as? Int) ?? fallback }
        return Configuration(
            latentChannels: int("latent_channels", 128),
            // `in_channels` is the encoder's RGB width. The config's top-level
            // `out_channels` is the *decoder's* RGB width and happens to be 3 as well,
            // so the two are easy to confuse and only one of them belongs here.
            inChannels: int("in_channels", 3),
            patchSize: int("patch_size", 4),
            downBlocks: blocks,
            logVariance: mode)
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(Self.encoderPrefix + name) }
        return w
    }

    // MARK: - Encode

    /// Video `[B, 3, F, H, W]` in `[-1, 1]` → latent `[B, 128, F', H', W']`, both
    /// `NCDHW`.
    ///
    /// This is the deterministic path and the only one: see the type's note on why no
    /// sampler exists. Same input twice is bit-identical output.
    ///
    /// **The clip is cast to the weight dtype (bf16) on the way in.** Handing in an fp32
    /// clip without the cast makes MLX promote all 42 convolutions to fp32, spending
    /// memory and time on a precision the bf16 weights do not carry.
    ///
    /// **Off-lattice frame counts throw rather than crop.** Silently dropping the
    /// trailing `(F − 1) mod 8` frames is a bad thing for a library to do quietly: the
    /// dropped frames change the conditioning's duration, and the keyframe positions a
    /// caller computes from its own frame count would then address the wrong latent
    /// frames. ``cropToLattice(_:)`` performs that crop when a caller asks for it in
    /// writing.
    public func encode(_ pixels: MLXArray) throws -> MLXArray {
        let head = try self.head(pixels)
        return normalize(means(head))
    }

    /// `conv_out`'s raw output, `[B, 129, F', H', W']`, before the head split and before
    /// the per-channel normalisation.
    ///
    /// Exposed because it is the natural boundary: it is the last thing that is purely
    /// the convolutional stack, and everything after it is three lines of index
    /// arithmetic. Exposing it separately lets the stack, the channel split and the
    /// normalisation each be inspected on their own.
    public func head(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 5, pixels.shape[1] == config.inChannels else {
            throw Failure.shape("pixels", got: pixels.shape,
                                expected: "[B, \(config.inChannels), F, H, W]")
        }
        let frames = pixels.shape[2], height = pixels.shape[3], width = pixels.shape[4]
        let scale = config.scaleFactors
        guard config.isOnLattice(pixelFrames: frames) else {
            throw Failure.shape("pixels", got: pixels.shape,
                                expected: "F on the \(scale.time)k + 1 lattice; \(frames) "
                                    + "is not, and cropping it here would silently drop "
                                    + "the trailing \((frames - 1) % scale.time) frame(s) — "
                                    + "call cropToLattice(_:) to ask for that in writing")
        }
        guard height % scale.height == 0, width % scale.width == 0 else {
            throw Failure.shape("pixels", got: pixels.shape,
                                expected: "H and W divisible by \(scale.height) and "
                                    + "\(scale.width); patchify and the space-to-depth "
                                    + "ladder have no padding to absorb a remainder")
        }

        let dtype = try weight("conv_in.conv.weight").dtype
        var h = patchify(pixels.asType(dtype))

        // NCDHW -> NDHWC. Everything below is channels-last; channel axis is -1.
        h = h.transposed(0, 2, 3, 4, 1)
        h = try conv(h, "conv_in")

        for (index, block) in config.downBlocks.enumerated() {
            switch block {
            case let .res(layers):
                for layer in 0 ..< layers {
                    h = try resnet(h, "down_blocks.\(index).res_blocks.\(layer)")
                }
            case let .downsample(stride, _):
                h = try downsample(h, stem: "down_blocks.\(index)", stride: stride,
                                   outChannels: config.channelLadder[index + 1])
            }
        }

        // `conv_norm_out` is PixelNorm and `conv_act` is SiLU, exactly as on the decoder;
        // `timestep_conditioning` never reaches the encoder at all.
        h = try conv(silu(pixelNorm(h)), "conv_out")
        h = h.transposed(0, 4, 1, 2, 3)                     // NDHWC -> NCDHW

        // The geometric result and the arithmetic target are derived two different ways
        // — three frame-0 duplications before halving, versus `(F − 1)/8 + 1` — so a
        // disagreement means a duplication landed on the wrong side or was skipped.
        let expected = [pixels.shape[0], config.headChannels,
                        config.latentFrames(pixelFrames: frames),
                        height / scale.height, width / scale.width]
        guard h.shape == expected else {
            throw Failure.shape("encoded head", got: h.shape, expected: "\(expected)")
        }
        return h
    }

    /// The head's first `latentChannels` channels.
    ///
    /// The one channel left over is the log-variance, and this is the whole of what
    /// happens to the variational head: a slice. The head looks like it does something
    /// with the variance; it does not.
    public func means(_ head: MLXArray) -> MLXArray {
        head[0..., ..<config.latentChannels]
    }

    /// The discarded log-variance channel, `[B, 1, F', H', W']`, **unnormalised**.
    ///
    /// Nothing consumes it — it is not multiplied into the latent, it is not used to
    /// sample, and the per-channel statistics are never applied to it. It exists so that
    /// a diagnostic can look at what the model thinks its own uncertainty is without
    /// that ever becoming a code path that changes the latent.
    public func logVariance(_ head: MLXArray) -> MLXArray {
        head[0..., config.latentChannels...]
    }

    /// Drop the trailing `(F − 1) mod 8` frames, putting `F` back on the lattice.
    ///
    /// The forgiving behaviour ``head(_:)`` refuses to perform silently. A caller that
    /// genuinely wants it calls this, and can see at the call site that frames were
    /// dropped.
    public func cropToLattice(_ pixels: MLXArray) -> MLXArray {
        let frames = pixels.shape[2]
        let excess = (frames - 1) % config.scaleFactors.time
        guard excess > 0 else { return pixels }
        return pixels[0..., 0..., ..<(frames - excess)]
    }

    /// Apply the encoder's per-channel normalisation: `(x − mean) / std`.
    ///
    /// The inverse of ``VideoVAEDecoder/denormalize(_:)`` — a plain per-latent-channel
    /// affine, 128 statistics against 128 channels, reshaped `[1, C, 1, 1, 1]` — but not
    /// its bit-exact inverse, because this divides where that multiplies and both round
    /// in bf16. A round trip through the two therefore does not return the input
    /// exactly, and the residual is a property of the arithmetic, not a bug.
    ///
    /// The statistics are cast to the latent's dtype rather than the latent widened to
    /// theirs, so this stays in bf16. And it is applied **after** the head split, to the
    /// means only — the log-variance channel never sees it.
    public func normalize(_ means: MLXArray) -> MLXArray {
        let channels = means.shape[1]
        let std = stdOfMeans.reshaped([1, channels, 1, 1, 1]).asType(means.dtype)
        let mean = meanOfMeans.reshaped([1, channels, 1, 1, 1]).asType(means.dtype)
        return (means - mean) / std
    }

    /// `[F, H, W, C]` in `[0, 1]` → `[1, C, F, H, W]` in `[-1, 1]`.
    ///
    /// The algebraic inverse of ``VideoVAEDecoder/rgb(_:)``, and the same map that takes
    /// `uint8` frames to an encoder input (`x / 127.5 − 1`). It exists so an
    /// image-to-video caller holding an ordinary `[0, 1]` image does not have to
    /// rediscover the convention, and so the round-trip test can close the loop without
    /// inventing one.
    ///
    /// The batch axis is *added*, mirroring `rgb`'s dropping of it.
    public func pixels(fromRGB rgb: MLXArray) -> MLXArray {
        let two = MLXArray(Float(2)).asType(rgb.dtype)
        let one = MLXArray(Float(1)).asType(rgb.dtype)
        return (rgb * two - one).transposed(3, 0, 1, 2).expandedDimensions(axis: 0)
    }

    // MARK: - Blocks

    /// `patchify`: `[B, C, F, H, W]` → `[B, C·q·r, F, H/q, W/r]`, `NCDHW`.
    ///
    /// `b c (f p) (h q) (w r) -> b (c p r q) f h w`, with `p = 1` and `q = r = 4`.
    ///
    /// Exactly the inverse permutation of ``VideoVAEDecoder/unpatchify(_:)``, and it
    /// carries the same trap: the channel order is `c, p, r, q` while the spatial
    /// pairing is `(h q)` and `(w r)`, so **`r` — width — is the more significant
    /// sub-axis and `q` — height — the less significant one.** Swapping them transposes
    /// every 4×4 patch. The shape is identical, the per-channel statistics are
    /// identical, the latent is the right shape and decodes to a spatially scrambled
    /// image — which is the failure that has no shape check and no channel count to
    /// catch it.
    ///
    /// `patch_size_t` is 1, so the `p` axis is degenerate and the frame count is
    /// untouched. That is also why `48 = 3 × 4 × 4` and not `3 × 4 × 4 × something`.
    public func patchify(_ x: MLXArray) -> MLXArray {
        let q = config.patchSize, r = config.patchSize
        guard q > 1 || r > 1 else { return x }
        let b = x.shape[0], c = x.shape[1], f = x.shape[2]
        let h = x.shape[3] / q, w = x.shape[4] / r
        // [b, c, f, h, q, w, r] -> [b, c, r, q, f, h, w]
        var y = x.reshaped([b, c, f, h, q, w, r])
        y = y.transposed(0, 1, 6, 4, 2, 3, 5)
        return y.reshaped([b, c * r * q, f, h, w])
    }

    /// `ResnetBlock3D`: `norm → silu → conv1 → norm → silu → conv2`, plus the skip.
    ///
    /// Structurally identical to ``VideoVAEDecoder/resnet(_:_:)`` — every encoder resnet
    /// also has `in == out`, so `conv_shortcut` and `norm3` are both identities, and
    /// `inject_noise` / `timestep_conditioning` are false — but the convolutions
    /// underneath it are **causal** where the decoder's are symmetric. Nothing in this
    /// function shows that; it is in ``temporalPad(_:timeKernel:)``, which is why that is
    /// where the comment lives.
    public func resnet(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        var h = try conv(silu(pixelNorm(x)), stem + ".conv1")
        h = try conv(silu(pixelNorm(h)), stem + ".conv2")
        return x + h
    }

    /// `SpaceToDepthDownsample`: duplicate frame 0, then a convolved branch and an
    /// averaged skip branch, both space-to-depth'd, summed.
    ///
    /// Both branches fold by `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`, and the
    /// skip's group-average is `b (c g) d h w -> b c d h w`, down to the output width.
    ///
    /// Four things here keep every shape and change every value:
    ///
    /// * **The frame-0 duplication comes first, and both branches see it.** It is the
    ///   causal counterpart of the decoder's first-frame *drop*: `F → F + 1 → (F + 1)/2`
    ///   is what turns `97 → 49` rather than `97 → 48` or `49`-with-a-different-phase.
    ///   Appending a copy of the *last* frame instead gives the identical shape and
    ///   shifts the temporal phase of every latent frame.
    /// * **The channel axis composes c-major, then time, then height, then width** —
    ///   `(c p1 p2 p3)`, flat index `((c·P1 + p1)·P2 + p2)·P3 + p3`. Any other ordering
    ///   is the same tensor with its sub-voxels permuted, exactly as in the decoder.
    /// * **The skip's group-mean composes `(c g)`, so `g` is the *fastest* axis.** The
    ///   space-to-depth output is folded `[out, group]` and averaged over `group`;
    ///   folding it `[group, out]` averages a different set of channels together and is
    ///   again shape-preserving. Group sizes on this checkpoint are 2, 1, 4, 8 — note
    ///   the 1, where `compress_time_res` doubles both the fold and the width, so its
    ///   skip is a pure rearrangement and this particular error is invisible there.
    /// * **The skip is added, not concatenated.** `out_channels` already accounts for
    ///   the multiplier, so a concatenation would change the width and be caught; the
    ///   real risk is dropping the skip entirely, which is not.
    ///
    /// Axes are `NDHWC` here: 1 is time, 2 height, 3 width, 4 channels.
    public func downsample(_ x: MLXArray, stem: String, stride: (Int, Int, Int),
                           outChannels: Int) throws -> MLXArray {
        var input = x
        if stride.0 == 2 {
            input = MLX.concatenated([input[0..., ..<1], input], axis: 1)
        }

        let folded = spaceToDepth(input, stride: stride)
        let group = folded.shape[4] / outChannels
        guard group >= 1, folded.shape[4] == group * outChannels else {
            throw Failure.shape(stem + " skip", got: folded.shape,
                                expected: "a channel count divisible by \(outChannels)")
        }
        // The groups are 8 elements at most, so this is not where precision is won or
        // lost; the widen-accumulate-round is spelled out so MLX's accumulator is not
        // left implicit at a seam.
        var skip = folded.reshaped(folded.shape.dropLast() + [outChannels, group])
        skip = skip.asType(.float32).mean(axis: -1).asType(folded.dtype)

        let convolved = spaceToDepth(try conv(input, stem + ".conv"), stride: stride)
        guard convolved.shape == skip.shape else {
            throw Failure.shape(stem, got: convolved.shape, expected: "\(skip.shape)")
        }
        return convolved + skip
    }

    /// `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`, expressed in `NDHWC`.
    ///
    /// The mirror of ``VideoVAEDecoder/upsample(_:stem:stride:)``'s rearrange, with the
    /// same c-major channel composition. Kept separate from ``downsample(_:stem:stride:outChannels:)``
    /// because both of that function's branches apply it and a divergence between them
    /// would be undetectable in the sum.
    public func spaceToDepth(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
        let (p1, p2, p3) = stride
        guard p1 * p2 * p3 > 1 else { return x }
        let b = x.shape[0]
        let d = x.shape[1] / p1, h = x.shape[2] / p2, w = x.shape[3] / p3
        let c = x.shape[4]
        // [b, d, p1, h, p2, w, p3, c] -> [b, d, h, w, c, p1, p2, p3]
        var y = x.reshaped([b, d, p1, h, p2, w, p3, c])
        y = y.transposed(0, 1, 3, 5, 7, 2, 4, 6)
        return y.reshaped([b, d, h, w, c * p1 * p2 * p3])
    }

    // MARK: - Primitives

    /// A `CausalConv3d` from the checkpoint: `<stem>.conv.{weight,bias}`.
    ///
    /// The doubled `conv` in the key is the wrapper's inner convolution; the wrapper
    /// holds no parameters, only the temporal padding. Spatially the inner convolution
    /// pads `(0, k/2, k/2)` with zeros, so **height and width are zero-padded
    /// symmetrically while time is replicate-padded, causally, outside the convolution.**
    /// MLX's `conv3d` takes one symmetric padding per axis and no mode, which happens to
    /// express exactly the spatial half; the temporal half is materialised with
    /// `concatenated` first and the depth padding left at 0.
    ///
    /// Every convolution on the encoder path has `stride=1` — the `_res` downsamples put
    /// their stride in the space-to-depth rearrange, not in the kernel — so no stride is
    /// passed here, and ``init(checkpoint:)`` checking every kernel at `[·, ·, 3, 3, 3]`
    /// is what would catch a checkpoint that disagreed.
    ///
    /// The convolution runs in the **weight's dtype**, bf16, matching
    /// ``VideoVAEDecoder/conv(_:_:)`` rather than ``AudioVAEDecoder``'s
    /// fp32-accumulate-through-bias seam. Encoding in one convention and decoding in the
    /// other would put a dtype discrepancy in the middle of every round trip.
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

    /// `CausalConv3d`'s temporal padding, on the **causal** branch: the whole `k − 1`
    /// extent is frame 0 repeated in front, and nothing behind.
    ///
    /// Unlike the decoder, this is not read from a config flag: every convolution on the
    /// encoder path is causal, `conv_in` and `conv_out` and the downsamples and the
    /// resnets alike. `causal_decoder: false` in the config describes the *decoder*, and
    /// applying it here would give the symmetric pad: same output shape, and every
    /// latent frame able to see one pixel frame of its own future, which is precisely
    /// the property that makes this VAE streamable and makes contract 15's keyframe
    /// marker meaningful.
    ///
    /// **What this means for a single image.** With `F = 1`, the first convolution sees
    /// `[f0, f0, f0]` and every temporal downsample duplicates the one frame to two and
    /// halves it back to one — so a single pixel frame encodes to exactly one latent
    /// frame, built entirely from itself. That is not a special case in the code; it
    /// falls out of the causal padding, and it is the reason the first latent frame of a
    /// clip covers one pixel frame where every later one covers eight.
    public func temporalPad(_ x: MLXArray, timeKernel: Int) -> MLXArray {
        let pad = timeKernel - 1
        guard pad > 0 else { return x }
        let first = MLX.repeated(x[0..., ..<1], count: pad, axis: 1)
        return MLX.concatenated([first, x], axis: 1)
    }

    /// `x / sqrt(mean(x², over channels) + eps)`, no affine, channel axis last.
    ///
    /// **The epsilon is 1e-8**, the same value the video decoder uses and not the audio
    /// VAE's 1e-6.
    ///
    /// The dtype split has one seam worth spelling out. The square stays in the tensor's
    /// dtype; the mean-square accumulates in fp32 — 1024 channels at the bottom of the
    /// ladder is enough summands for a bf16 accumulator to lose real precision — and is
    /// then **rounded back to the tensor's dtype** before the square root, which itself
    /// runs in fp32 and rounds once more.
    ///
    /// That round back is not cosmetic. There are 37 of these norms on the path and
    /// PixelNorm *divides* by the result, so a half-ULP on the denominator compounds
    /// across the ladder — it is the largest of the dtype choices in this file.
    /// ``VideoVAEDecoder/pixelNorm(_:eps:)`` keeps the fp32 form, so the two halves do
    /// differ here — noted because the next person to compare the two files will see it.
    ///
    /// A positive epsilon is also what makes an all-zero clip encode to something finite
    /// rather than NaN: `0 / sqrt(0 + 1e-8)` is 0, and 1e-4 is representable in bf16.
    public func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
        let squared = x * x
        let meanSquare = squared.asType(.float32).mean(axis: -1, keepDims: true)
            .asType(x.dtype)
        let rms = MLX.sqrt(meanSquare.asType(.float32) + eps).asType(x.dtype)
        return x / rms
    }

    /// `x * sigmoid(x)`, evaluated in fp32 and cast back.
    ///
    /// An elementwise nonlinearity on a bf16 tensor is evaluated at fp32 and rounded
    /// once, not evaluated in bf16 throughout. There are **37** SiLUs on this encoder's
    /// path — 18 resnets × 2, plus `conv_act` — so a hand-rolled bf16 sigmoid here is
    /// visible at the output.
    public func silu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * MLX.sigmoid(f)).asType(x.dtype)
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The x2 latent spatial upsampler that sits between stage 1 and stage 2 of the
/// distilled pipeline: `[B, 128, F, H, W]` in, `[B, 128, F, 2H, 2W]` out.
///
/// The entry point is ``upsample(_:)``, not ``forward(_:)``: **the normalisation round
/// trip around the network is part of the feature.** Stopping at the network alone
/// produces a correctly shaped latent from the wrong distribution.
///
/// ## What the architecture is, and where each number came from
///
/// The checkpoint carries its own config in `__metadata__.config`. Everything structural
/// is read from there and cross-checked against the 72 tensors:
///
/// | field | value | source |
/// |---|---|---|
/// | `in_channels` | 128 | `__metadata__.config`, checked against `initial_conv.weight[1]` |
/// | `mid_channels` | 1024 | `__metadata__.config`, checked against `initial_conv.weight[0]` |
/// | `num_blocks_per_stage` | 4 | `__metadata__.config`, checked by exact key-set equality |
/// | `dims` | 3 | `__metadata__.config`, checked against the 5-rank conv weights |
/// | `spatial_upsample` / `temporal_upsample` | true / false | `__metadata__.config` |
/// | `spatial_scale` | 2.0 | `__metadata__.config`, checked against `upsampler.0.weight[0] / mid` |
/// | `rational_resampler` | false | `__metadata__.config`, checked by `upsampler.0` being a `Sequential` member |
/// | GroupNorm groups | **32** | **declared here, not carried by the checkpoint** |
///
/// That last row is the one honest gap. The checkpoint stores only a `[1024]` weight and
/// a `[1024]` bias, which are identical for any group count that divides 1024. No shape
/// check anywhere can distinguish 32 groups from 16 or 64, and the numbers differ.
/// ``Configuration/normGroups`` names the constant rather than pretending it was derived.
///
/// ## Which configuration this checkpoint is
///
/// The config admits several different networks over the same key names, and this
/// checkpoint is exactly one of them. The discriminating facts, all read rather than
/// assumed:
///
/// * **`dims == 3`, so there is no fold of frames into the batch around the whole
///   network.** The `dims == 2` variant folds for *everything*, which makes every
///   GroupNorm per-frame instead of per-clip. Same tensors, same shapes, different
///   statistics.
/// * **The upsampler itself is a 2-D convolution plus a pixel shuffle of 2, applied per
///   frame.** `upsampler.0.weight` is rank 4 — `[4096, 1024, 3, 3]` — while every other
///   conv in the file is rank 5. That is the checkpoint agreeing with the spatial-only
///   variant, and it is why the fold happens around three lines rather than around the
///   network.
/// * **No frame is dropped.** Dropping the first frame belongs to the *temporal* variant.
///   This is the spatial one, and frame count is invariant end to end. Dropping a frame
///   here would still produce a plausible latent and would desynchronise it from the
///   audio stream that stage 2 carries alongside.
/// * **`SpatialRationalResampler` is not constructed.** It would add a `conv` and a
///   `blur_down` buffer, and at `scale = 2.0` it degenerates to the same shape ladder
///   with different weights under different names. The key set is what rules it out:
///   `upsampler.0.*` is a `Sequential` member 0, not `.conv.*`.
///
/// ## The ladder, at the production shape
///
/// ```
/// latent (normalised)          [1,  128, 13,  6, 10]
/// denormalize                  [1,  128, 13,  6, 10]
/// initial_conv  3x3x3          [1, 1024, 13,  6, 10]
/// initial_norm  GroupNorm(32)  [1, 1024, 13,  6, 10]
/// SiLU                         [1, 1024, 13,  6, 10]
/// res_blocks.0..3              [1, 1024, 13,  6, 10]
/// fold frames into batch       [13, 1024,     6, 10]
/// upsampler.0   3x3            [13, 4096,     6, 10]
/// pixel shuffle 2              [13, 1024,    12, 20]
/// unfold                       [1, 1024, 13, 12, 20]
/// post_upsample_res_blocks.0..3[1, 1024, 13, 12, 20]
/// final_conv    3x3x3          [1,  128, 13, 12, 20]
/// renormalize                  [1,  128, 13, 12, 20]
/// ```
///
/// ## Padding: zeros on all three axes, including time
///
/// These are plain 3-D convolutions with `padding=1`, **not** the video VAE's causal
/// convolutions. There is no wrapper, no replicate padding, and no causality: time is
/// zero-padded symmetrically exactly like height and width. ``VideoVAEDecoder/conv(_:_:)``
/// documents the opposite convention for the decoder and the two must not be confused —
/// carrying the decoder's `temporalPad` across would keep every shape and pull the first
/// and last latent frames toward the origin.
///
/// ## Layout
///
/// This module computes in `NDHWC`, but its boundary is `NCDHW`, because that is what the
/// sampler's latent is; everything between the two transposes in ``forward(_:)`` is
/// channels-last, so **the channel axis is `-1` inside and `1` outside**.
/// ``groupNorm(_:_:)`` reduces over *both* the spatial-temporal axes and a channel
/// sub-range, so an axis error there is not merely a wrong normaliser — it is a
/// differently shaped reduction that still broadcasts.
///
/// ## Dtype
///
/// The pipeline builds this model in bf16 and hands it a bf16 latent. Two reductions live
/// on this path and they narrow differently, which is the `docs/FRAGILE_CONTRACTS.md` #16
/// question asked of a module that has no `PixelNorm`:
///
/// * **`GroupNorm` computes entirely in fp32 and narrows once, at its output** — mean,
///   variance, normalisation *and* the affine. The three cheaper spellings — fp32
///   statistics with a bf16 affine, bf16-narrowed statistics, all-bf16 — each give a
///   visibly different answer.
/// * That is **the opposite of what contract 16 asks of `PixelNorm`**, and for a reason
///   worth stating: `PixelNorm` narrows because its mean is a standalone op producing a
///   bf16 *tensor* which the subsequent ops then see. `GroupNorm` has no such
///   intermediate to narrow. The contract's actual rule is about where a value is
///   materialised; applying its *conclusion* by analogy would produce the wrong answer
///   here.
/// * `eps` is `1e-5`, the GroupNorm default. Not `1e-6` and not `1e-8`; both of those are
///   live constants elsewhere in this repo.
///
/// The convolutions follow ``AudioVAEDecoder/conv(_:_:)``'s accumulator discipline —
/// fp32 products, fp32 bias, one narrowing — rather than ``VideoVAEDecoder/conv(_:_:)``'s
/// bf16 add. The two disagree today; see the note on ``conv3d(_:_:)``.
public struct LatentSpatialUpsampler {

    /// `__metadata__.config`, read from the checkpoint rather than declared.
    public struct Configuration: Equatable, Sendable {
        /// Latent channels in and out, 128. The network is channel-preserving.
        public let inChannels: Int
        /// Working width, 1024. Every conv between `initial_conv` and `final_conv` is
        /// `mid → mid`.
        public let midChannels: Int
        /// ResBlocks per stage, 4 — and there are two stages, so eight blocks and 32
        /// conv tensors.
        public let numBlocksPerStage: Int
        /// 3. The `dims == 2` branch is a different network over the same weights;
        /// ``readConfiguration(_:)`` refuses it rather than silently folding frames.
        public let dims: Int
        /// 2. Held as an `Int` because the only value this branch implements is the
        /// integer pixel-shuffle factor; `spatial_scale` is a `float` in the config
        /// because `SpatialRationalResampler` supports 0.75 and 1.5, and this file
        /// implements neither.
        public let spatialScale: Int
        /// **32, declared here rather than read from the checkpoint.**
        ///
        /// The group count appears nowhere in the config and in no tensor shape: a
        /// `[1024]` weight is consistent with 8, 16, 32, 64 or 128 groups, and only the
        /// arithmetic differs. This field exists so that the constant is named and
        /// overridable in a test rather than buried at a call site.
        public let normGroups: Int

        public init(inChannels: Int, midChannels: Int, numBlocksPerStage: Int, dims: Int,
                    spatialScale: Int, normGroups: Int = 32) {
            self.inChannels = inChannels
            self.midChannels = midChannels
            self.numBlocksPerStage = numBlocksPerStage
            self.dims = dims
            self.spatialScale = spatialScale
            self.normGroups = normGroups
        }

        /// `4 × mid`: `upsampler.0`'s output width, `scale² × mid` for a 2D shuffle.
        public var shuffledChannels: Int { spatialScale * spatialScale * midChannels }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)
        case unexpected(String)
        case unsupported(String)
        case configuration(String)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected): return "\(name) is \(got), expected \(expected)"
            case let .unexpected(name):
                return "checkpoint carries \(name), which the derived topology does not "
                    + "account for"
            case let .unsupported(why): return why
            case let .configuration(why): return "checkpoint config: \(why)"
            }
        }
    }

    /// The upsampler's 72 tensors, under their checkpoint names (this file has no
    /// component prefix to strip).
    public let weights: [String: MLXArray]
    public let config: Configuration
    /// `per_channel_statistics.std-of-means` from the **video VAE**, `[inChannels]`.
    public let stdOfMeans: MLXArray
    /// `per_channel_statistics.mean-of-means` from the **video VAE**, `[inChannels]`.
    public let meanOfMeans: MLXArray

    public static let statisticsPrefix = "per_channel_statistics."

    // MARK: - Loading

    /// Load the upsampler, taking the normalisation statistics from a video VAE file.
    ///
    /// Two checkpoints, because the feature genuinely spans two: the normalisation
    /// statistics belong to the **video VAE**, not to the upsampler, and only two `[128]`
    /// vectors of it are needed. They are read straight out of the VAE header rather than
    /// by building an encoder whose weights would never be used. See
    /// ``statistics(videoVAE:)``.
    ///
    /// Pass the VAE the pipeline is using. On the two 2.5 video VAEs that ship side by
    /// side — `ltx-2.5-video-vae-conv-bf16` and `ltx-2.5-video-vae-bf16` — the statistics
    /// are **identical**, so today the choice does not change the numbers; that is an
    /// observation about these two files, not a property to rely on.
    public init(checkpoint: URL, videoVAE: URL) throws {
        try self.init(checkpoint: checkpoint, statistics: Self.statistics(videoVAE: videoVAE))
    }

    /// Designated initialiser: upsampler weights plus explicit `(std, mean)`.
    ///
    /// Separated so a test can supply identity statistics and isolate the network from
    /// the normalisation round trip, and so a caller that already holds a loaded VAE
    /// does not re-read the file.
    public init(checkpoint: URL, statistics: (std: MLXArray, mean: MLXArray)) throws {
        let header = try SafetensorsHeader.read(from: checkpoint)
        let config = try Self.readConfiguration(header)
        self.config = config

        let loaded = try MLX.loadArrays(url: checkpoint)
        guard !loaded.isEmpty else { throw Failure.missing("*") }

        // Exact key-set equality in both directions. The derived topology must demand
        // every tensor the file has and no others: a `num_blocks_per_stage` misread as 3
        // loads a smaller network that runs, and an extra tensor means the checkpoint is
        // a variant this branch does not implement. `AudioVAEEncoder` establishes this
        // pattern and the reason — a config and a checkpoint that disagree must fail at
        // load, not as a wrong number later.
        let expected = Self.expectedShapes(config)
        for (name, shape) in expected {
            guard let tensor = loaded[name] else { throw Failure.missing(name) }
            guard tensor.shape == shape else {
                throw Failure.shape(name, got: tensor.shape, expected: "\(shape)")
            }
        }
        if let stray = loaded.keys.sorted().first(where: { expected[$0] == nil }) {
            throw Failure.unexpected(stray)
        }
        self.weights = loaded

        guard statistics.std.size == config.inChannels,
              statistics.mean.size == config.inChannels else {
            throw Failure.shape(Self.statisticsPrefix + "{std,mean}-of-means",
                                got: statistics.std.shape,
                                expected: "[\(config.inChannels)]")
        }
        self.stdOfMeans = statistics.std
        self.meanOfMeans = statistics.mean
    }

    /// Read `per_channel_statistics.{std,mean}-of-means` out of a video VAE checkpoint.
    ///
    /// Via ``SafetensorsReader`` rather than `MLX.loadArrays`: the conv video VAE is
    /// 1.4 GB and 170 tensors, and this needs two vectors of 128 floats. The reader
    /// mmaps and materialises only the ranges asked for.
    ///
    /// bf16 → fp32 here is exact (a 16-bit shift), and the arrays are cast back to the
    /// latent's dtype at each use in ``denormalize(_:)`` / ``renormalize(_:)``.
    public static func statistics(videoVAE: URL) throws -> (std: MLXArray, mean: MLXArray) {
        let reader = try SafetensorsReader(url: videoVAE)
        func vector(_ suffix: String) throws -> MLXArray {
            let name = statisticsPrefix + suffix
            guard reader.header.tensors[name] != nil else { throw Failure.missing(name) }
            return MLXArray(try reader.floats(name))
        }
        return (try vector("std-of-means"), try vector("mean-of-means"))
    }

    /// Parse `__metadata__.config` and refuse every branch this type does not implement.
    ///
    /// The refusals are not defensive padding. Each one names a configuration that would
    /// load all 72 tensors and compute something else:
    ///
    /// * `dims: 2` folds frames into the batch for the whole network.
    /// * `temporal_upsample: true` needs an 8× or 2× `Conv3d` this file does not carry,
    ///   and drops frame 0 afterwards.
    /// * `rational_resampler: true` is a different module under different key names.
    /// * `spatial_scale` other than 2.0 only exists on the rational path.
    public static func readConfiguration(_ header: SafetensorsHeader) throws -> Configuration {
        guard let config = header.metadataJSON("config") else {
            throw Failure.configuration("__metadata__.config is absent")
        }
        if let name = config["_class_name"] as? String, name != "LatentUpsampler" {
            throw Failure.unsupported("_class_name is \"\(name)\"; this port implements "
                + "LatentUpsampler only")
        }

        func int(_ key: String, _ fallback: Int) -> Int { (config[key] as? Int) ?? fallback }
        func bool(_ key: String, _ fallback: Bool) -> Bool {
            (config[key] as? Bool) ?? ((config[key] as? NSNumber)?.boolValue ?? fallback)
        }

        // Defaults for the keys a config is allowed to omit, so a partial config resolves
        // to the configuration these checkpoints are written with rather than to a guess.
        let dims = int("dims", 3)
        guard dims == 3 else {
            throw Failure.unsupported("dims is \(dims); this port implements the 3D "
                + "branch only. The 2D branch folds frames into the batch for the whole "
                + "network, which makes every GroupNorm per-frame instead of per-clip")
        }
        guard bool("spatial_upsample", true) else {
            throw Failure.unsupported("spatial_upsample is false; this port implements "
                + "the spatial branch only")
        }
        guard !bool("temporal_upsample", false) else {
            throw Failure.unsupported("temporal_upsample is true; that branch builds a "
                + "Conv3d upsampler and drops the first frame afterwards, and this file "
                + "carries a rank-4 upsampler weight")
        }
        guard !bool("rational_resampler", false) else {
            throw Failure.unsupported("rational_resampler is true; SpatialRationalResampler "
                + "is a different module with a blur-downsample buffer this port does "
                + "not implement")
        }
        let scale = (config["spatial_scale"] as? NSNumber)?.doubleValue ?? 2.0
        guard scale == 2.0 else {
            throw Failure.unsupported("spatial_scale is \(scale); only 2.0 reaches the "
                + "plain PixelShuffle branch, and the others are rational_resampler-only")
        }

        let mid = int("mid_channels", 512)
        let groups = 32
        guard mid % groups == 0 else {
            throw Failure.configuration("mid_channels \(mid) is not divisible by the "
                + "\(groups) GroupNorm groups this module uses")
        }
        return Configuration(inChannels: int("in_channels", 128), midChannels: mid,
                             numBlocksPerStage: int("num_blocks_per_stage", 4), dims: dims,
                             spatialScale: Int(scale), normGroups: groups)
    }

    /// Every tensor the derived topology accounts for, with its exact shape.
    ///
    /// 72 for this checkpoint: `4 + 32 + 2 + 32 + 2`. Both the missing-key and the
    /// surplus-key direction are checked against this in ``init(checkpoint:statistics:)``,
    /// which is what makes a mis-derived `num_blocks_per_stage` a load failure with a
    /// countable difference instead of a smaller network that runs.
    public static func expectedShapes(_ c: Configuration) -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        func conv3(_ stem: String, out: Int, inChannels: Int) {
            shapes[stem + ".weight"] = [out, inChannels, 3, 3, 3]
            shapes[stem + ".bias"] = [out]
        }
        func norm(_ stem: String, _ channels: Int) {
            shapes[stem + ".weight"] = [channels]
            shapes[stem + ".bias"] = [channels]
        }
        func stage(_ prefix: String) {
            for i in 0 ..< c.numBlocksPerStage {
                conv3("\(prefix).\(i).conv1", out: c.midChannels, inChannels: c.midChannels)
                norm("\(prefix).\(i).norm1", c.midChannels)
                conv3("\(prefix).\(i).conv2", out: c.midChannels, inChannels: c.midChannels)
                norm("\(prefix).\(i).norm2", c.midChannels)
            }
        }
        conv3("initial_conv", out: c.midChannels, inChannels: c.inChannels)
        norm("initial_norm", c.midChannels)
        stage("res_blocks")
        // Rank 4, and that is the tell: `upsampler.0` is a Conv2d applied per frame even
        // though every other conv in the file is a Conv3d.
        shapes["upsampler.0.weight"] = [c.shuffledChannels, c.midChannels, 3, 3]
        shapes["upsampler.0.bias"] = [c.shuffledChannels]
        stage("post_upsample_res_blocks")
        conv3("final_conv", out: c.inChannels, inChannels: c.midChannels)
        return shapes
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(name) }
        return w
    }

    // MARK: - The feature

    /// Un-normalise, upsample, re-normalise. **This is the entry point the pipeline
    /// uses.**
    ///
    /// The sampler works in normalised latent space; the upsampler was trained in the
    /// VAE's raw latent space. Skipping either half leaves the network reading, or
    /// stage 2 receiving, a latent scaled by 128 per-channel constants — a tensor of
    /// exactly the right shape and dtype, finite, with a plausible histogram. Nothing
    /// downstream would raise.
    ///
    /// Both halves stay in the latent's dtype: the statistics are cast to the tensor, and
    /// the arithmetic is done in the tensor's own precision.
    public func upsample(_ latent: MLXArray) throws -> MLXArray {
        try renormalize(forward(denormalize(latent)))
    }

    /// `x * std + mean`, statistics broadcast over `[1, C, 1, 1, 1]`, `NCDHW`.
    ///
    /// The same operation ``VideoVAEDecoder/denormalize(_:)`` performs before decoding,
    /// from the same two buffers in the same file. It is *here* as well because the
    /// upsampler sits upstream of the decoder and the sampler's latent is normalised at
    /// both seams.
    public func denormalize(_ latent: MLXArray) -> MLXArray {
        let channels = latent.shape[1]
        let std = stdOfMeans.reshaped([1, channels, 1, 1, 1]).asType(latent.dtype)
        let mean = meanOfMeans.reshaped([1, channels, 1, 1, 1]).asType(latent.dtype)
        return latent * std + mean
    }

    /// `(x - mean) / std`, the exact inverse of ``denormalize(_:)``.
    ///
    /// A *division*, not a multiply by a reciprocal: on 128 per-channel constants the two
    /// spellings differ by a rounding each.
    public func renormalize(_ latent: MLXArray) -> MLXArray {
        let channels = latent.shape[1]
        let std = stdOfMeans.reshaped([1, channels, 1, 1, 1]).asType(latent.dtype)
        let mean = meanOfMeans.reshaped([1, channels, 1, 1, 1]).asType(latent.dtype)
        return (latent - mean) / std
    }

    /// The network itself — the `dims == 3` spatial configuration — without the
    /// normalisation round trip.
    ///
    /// `[B, C, F, H, W]` in, `[B, C, F, 2H, 2W]` out, both `NCDHW`. **The frame count
    /// and the channel count are invariant**; only height and width move, and they move
    /// by exactly the shuffle factor. The guard at the end states all four because three
    /// of them are the ones a wrong branch would break silently.
    ///
    /// The latent is cast to the weight dtype (bf16) on the way in, for the reason
    /// ``VideoVAEDecoder/decode(_:)`` documents: an fp32 input promotes every convolution
    /// to fp32, which is a different network's worth of arithmetic on the same weights.
    public func forward(_ latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 5, latent.shape[1] == config.inChannels else {
            throw Failure.shape("latent", got: latent.shape,
                                expected: "[B, \(config.inChannels), F, H, W]")
        }
        let dtype = try weight("initial_conv.weight").dtype
        let frames = latent.shape[2], height = latent.shape[3], width = latent.shape[4]

        // NCDHW -> NDHWC. Everything below is channels-last; channel axis is -1.
        var x = latent.asType(dtype).transposed(0, 2, 3, 4, 1)

        x = try conv3d(x, "initial_conv")
        x = try groupNorm(x, "initial_norm")
        x = silu(x)

        for i in 0 ..< config.numBlocksPerStage {
            x = try resBlock(x, "res_blocks.\(i)")
        }

        x = try spatialUpsample(x)

        for i in 0 ..< config.numBlocksPerStage {
            x = try resBlock(x, "post_upsample_res_blocks.\(i)")
        }

        x = try conv3d(x, "final_conv")
        x = x.transposed(0, 4, 1, 2, 3)                 // NDHWC -> NCDHW

        let scale = config.spatialScale
        guard x.shape == [latent.shape[0], config.inChannels, frames,
                          height * scale, width * scale] else {
            throw Failure.shape("upsampled latent", got: x.shape,
                                expected: "[\(latent.shape[0]), \(config.inChannels), "
                                    + "\(frames), \(height * scale), \(width * scale)]")
        }
        return x
    }

    // MARK: - Blocks

    /// `ResBlock`: `conv1 → norm1 → SiLU → conv2 → norm2 → SiLU(x + residual)`.
    ///
    /// **Norm follows conv here.** The video and audio VAEs are norm-then-conv
    /// pre-activation blocks; this is the post-activation arrangement, and the
    /// difference is a whole different function of the same weights — so it is worth
    /// checking rather than assuming from the neighbouring modules.
    ///
    /// **The residual is added before the second SiLU, not after it.** The sum goes
    /// *through* the nonlinearity. Adding after would be the more familiar shape
    /// (`silu(h) + x`) and would keep every dimension.
    ///
    /// `mid_channels` defaults to `channels`, so both convs are `1024 → 1024` and there
    /// is no shortcut projection; ``expectedShapes(_:)`` asserts that against the file.
    public func resBlock(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        var h = try conv3d(x, stem + ".conv1")
        h = try groupNorm(h, stem + ".norm1")
        h = silu(h)
        h = try conv3d(h, stem + ".conv2")
        h = try groupNorm(h, stem + ".norm2")
        return silu(h + x)
    }

    /// The spatial upsampler: fold frames into the batch, 2-D convolution to `4 × mid`
    /// channels, pixel shuffle of 2, unfold.
    ///
    /// Two things to get right, neither of which a shape check sees:
    ///
    /// * **The fold is frame-major within the batch**, i.e. index `b·F + f`. In `NDHWC`
    ///   the batch and frame axes are already adjacent and in that order, so the fold is
    ///   a pure reshape — but only because the layout puts them there. It would be a
    ///   transpose in `NCDHW`.
    /// * **The shuffle decomposes the channel axis c-major:** flat channel
    ///   `(c·2 + p1)·2 + p2` with `p1` on **height** and `p2` on **width**. Any other
    ///   ordering permutes the 2×2 sub-pixels — a checkerboard with identical shape,
    ///   identical channel count and identical per-channel statistics. This is the same
    ///   decomposition ``VideoVAEDecoder/upsample(_:stem:stride:)`` documents, one axis
    ///   shorter.
    public func spatialUpsample(_ x: MLXArray) throws -> MLXArray {
        let w = try weight("upsampler.0.weight")
        let b = try weight("upsampler.0.bias")
        guard w.ndim == 4 else {
            throw Failure.shape("upsampler.0.weight", got: w.shape,
                                expected: "[out, in, kh, kw] — a rank-5 weight here means "
                                    + "the checkpoint is a temporal or spatiotemporal "
                                    + "upsampler, not this branch")
        }
        let batch = x.shape[0], frames = x.shape[1]
        let height = x.shape[2], width = x.shape[3], channels = x.shape[4]

        // Fold frames into the batch, which in NDHWC is a reshape.
        var y = x.reshaped([batch * frames, height, width, channels])

        let kernel = w.transposed(0, 2, 3, 1)           // [out, in, kh, kw] -> [out, kh, kw, in]
        y = (MLX.conv2d(y.asType(.float32), kernel.asType(.float32),
                        padding: IntOrPair((w.shape[2] / 2, w.shape[3] / 2)))
            + b.asType(.float32)).asType(w.dtype)

        // Pixel shuffle of 2, channel axis last: [n, h, w, c·p1·p2] -> [n, h·2, w·2, c].
        let p = config.spatialScale
        let outChannels = y.shape[3] / (p * p)
        y = y.reshaped([batch * frames, height, width, outChannels, p, p])
        y = y.transposed(0, 1, 4, 2, 5, 3)              // [n, h, p1, w, p2, c]
        y = y.reshaped([batch * frames, height * p, width * p, outChannels])

        return y.reshaped([batch, frames, height * p, width * p, outChannels])
    }

    // MARK: - Primitives

    /// A 3×3×3 convolution with `padding=1`, from the checkpoint.
    ///
    /// **Zero padding on all three axes, time included.** These are plain convolutions,
    /// not the video VAE's causal ones: there is no wrapper, no replicate padding and no
    /// causality anywhere in this module. The doubled `.conv.` segment that appears in
    /// every VAE key is absent here for the same reason — the weight is
    /// `initial_conv.weight`, not `initial_conv.conv.weight`.
    ///
    /// Weight layout: stored `[out, in, kd, kh, kw]`, MLX convolves `[out, kd, kh, kw,
    /// in]` over `NDHWC`, hence `(0, 2, 3, 4, 1)`.
    ///
    /// Products and bias accumulate in fp32 and narrow once, following
    /// ``AudioVAEDecoder/conv(_:_:)``: the bias belongs inside the convolution's
    /// epilogue, before the bf16 store. ``VideoVAEDecoder/conv(_:_:)`` currently adds
    /// its bias in bf16 instead — the two modules disagree, and reconciling them is not
    /// this file's to make.
    public func conv3d(_ x: MLXArray, _ stem: String) throws -> MLXArray {
        let w = try weight(stem + ".weight")
        let b = try weight(stem + ".bias")
        guard w.ndim == 5 else {
            throw Failure.shape(stem + ".weight", got: w.shape,
                                expected: "[out, in, kd, kh, kw]")
        }
        let kernel = w.transposed(0, 2, 3, 4, 1)
        let padding = IntOrTriple((w.shape[2] / 2, w.shape[3] / 2, w.shape[4] / 2))
        let accumulated = MLX.conv3d(x.asType(.float32), kernel.asType(.float32),
                                     padding: padding) + b.asType(.float32)
        return accumulated.asType(w.dtype)
    }

    /// GroupNorm over `NDHWC`: 32 groups, `eps = 1e-5`, affine.
    ///
    /// Each of the 32 groups spans `C/32` **contiguous** channels and the *whole*
    /// spatial-temporal volume of one batch element, so the reduction is over
    /// `(D, H, W, C/32)` — four axes, of which one is a slice of the channel axis. In
    /// `NCDHW` that is a single `reshape(N, G, -1)`; in `NDHWC` the channel sub-axis is
    /// separated from the rest, hence the two reduction axes here.
    ///
    /// Reducing over the wrong set is the failure this comment exists for. Dropping the
    /// channel sub-axis gives a per-channel instance norm; dropping the temporal axis
    /// gives a per-frame group norm — both broadcast, both keep the shape, both are a
    /// different network.
    ///
    /// **fp32 throughout, narrowed once at the end**, including the affine. See this
    /// type's dtype note for why that differs from `PixelNorm`'s narrowing.
    ///
    /// The variance is the **biased** one, `mean((x - mean)²)`. `eps` is added to the
    /// variance inside the reciprocal square root, not to the deviation.
    public func groupNorm(_ x: MLXArray, _ stem: String, eps: Float = 1e-5) throws -> MLXArray {
        let gamma = try weight(stem + ".weight")
        let beta = try weight(stem + ".bias")
        let batch = x.shape[0], frames = x.shape[1]
        let height = x.shape[2], width = x.shape[3], channels = x.shape[4]
        let groups = config.normGroups
        guard channels % groups == 0 else {
            throw Failure.shape(stem, got: x.shape,
                                expected: "a channel count divisible by \(groups)")
        }
        guard gamma.size == channels, beta.size == channels else {
            throw Failure.shape(stem + ".weight", got: gamma.shape,
                                expected: "[\(channels)]")
        }

        // [B, D, H, W, C] -> [B, D·H·W, G, C/G]. Contiguous in both split axes, so group
        // membership is the contiguous channel split GroupNorm is defined over.
        let f = x.asType(.float32)
            .reshaped([batch, frames * height * width, groups, channels / groups])
        let mean = f.mean(axes: [1, 3], keepDims: true)
        let centred = f - mean
        let variance = (centred * centred).mean(axes: [1, 3], keepDims: true)
        var normed = centred * MLX.rsqrt(variance + eps)

        normed = normed.reshaped([batch, frames, height, width, channels])
        let scale = gamma.reshaped([1, 1, 1, 1, channels]).asType(.float32)
        let shift = beta.reshaped([1, 1, 1, 1, channels]).asType(.float32)
        return (normed * scale + shift).asType(x.dtype)
    }

    /// `x * sigmoid(x)`, evaluated in fp32 and cast back.
    ///
    /// The same rule ``VideoVAEDecoder/silu(_:)`` documents: an elementwise nonlinearity
    /// on a bf16 tensor is evaluated with an fp32 accumulator. There are **17** SiLUs on
    /// this path — 8 res blocks × 2, plus the initial activation.
    public func silu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * MLX.sigmoid(f)).asType(x.dtype)
    }
}

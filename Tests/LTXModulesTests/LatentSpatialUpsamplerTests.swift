// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The x2 latent spatial upsampler, stage 1 → stage 2 of the distilled pipeline.
///
/// What this suite checks, in descending order of strength:
///
/// 1. **Structural equality against hand-computed arithmetic**, on a synthetic
///    checkpoint small enough that the expected values are written out by hand: the
///    pixel-shuffle sub-pixel ordering, the GroupNorm reduction set, and the residual
///    passing *through* the second activation. These are the three things that keep
///    every shape and change every value.
/// 2. **Exact key-set agreement** between the topology derived from
///    `__metadata__.config` and the 72 tensors in the file, in both directions, plus
///    explicit refusal of a checkpoint that disagrees.
/// 3. **Shape and dtype contracts** on the real checkpoint at the production input, and
///    the presence of the per-channel normalisation round trip the network runs inside.
@Suite("Latent spatial upsampler", .serialized)
struct LatentSpatialUpsamplerTests {

    static let checkpointPath = LTXConfiguration.resolved.checkpoints.root!
        + "/latent_upscale_models/"
        + "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    static let videoVAEPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-video-vae-conv-bf16.safetensors"

    static func skip(_ what: String, _ path: String) {
        print("SKIP \(what): \(path) is absent")
    }

    static func module() throws -> LatentSpatialUpsampler? {
        guard FileManager.default.fileExists(atPath: checkpointPath),
              FileManager.default.fileExists(atPath: videoVAEPath) else { return nil }
        return try LatentSpatialUpsampler(checkpoint: URL(fileURLWithPath: checkpointPath),
                                          videoVAE: URL(fileURLWithPath: videoVAEPath))
    }

    // MARK: - Synthetic checkpoints

    /// Build a checkpoint from a config, so that a test can state the arithmetic it
    /// expects instead of trusting a 1 GB file to be small.
    ///
    /// Every tensor comes from ``LatentSpatialUpsampler/expectedShapes(_:)`` — the same
    /// function the loader validates against — filled with zeros and then overridden.
    /// That makes the *loader* and the *fixture* share one definition of the topology,
    /// which is what lets ``aDisagreeingCheckpointIsRefused()`` perturb exactly one
    /// tensor and know that nothing else moved.
    static func writeSynthetic(config: LatentSpatialUpsampler.Configuration,
                               overrides: [String: MLXArray] = [:],
                               drop: Set<String> = [],
                               extra: [String: MLXArray] = [:],
                               metadata: [String: Any]? = nil,
                               to url: URL) throws {
        var arrays: [String: MLXArray] = [:]
        for (name, shape) in LatentSpatialUpsampler.expectedShapes(config) where !drop.contains(name) {
            arrays[name] = MLX.zeros(shape, type: Float32.self)
        }
        for (name, value) in overrides { arrays[name] = value }
        for (name, value) in extra { arrays[name] = value }

        let json = metadata ?? [
            "_class_name": "LatentUpsampler",
            "in_channels": config.inChannels,
            "mid_channels": config.midChannels,
            "num_blocks_per_stage": config.numBlocksPerStage,
            "dims": config.dims,
            "spatial_upsample": true,
            "temporal_upsample": false,
            "spatial_scale": 2.0,
            "rational_resampler": false,
        ]
        let blob = try JSONSerialization.data(withJSONObject: json)
        try MLX.save(arrays: arrays,
                     metadata: ["config": String(data: blob, encoding: .utf8)!],
                     url: url)
    }

    static func scratch(_ name: String) throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltx-latent-upsampler-tests", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(name)
    }

    /// Identity statistics: `std = 1`, `mean = 0`, so ``LatentSpatialUpsampler/upsample(_:)``
    /// and ``LatentSpatialUpsampler/forward(_:)`` coincide and a test can isolate the
    /// network.
    static func identityStatistics(_ channels: Int) -> (std: MLXArray, mean: MLXArray) {
        (MLX.ones([channels], type: Float32.self), MLX.zeros([channels], type: Float32.self))
    }

    // MARK: - Topology, read from the checkpoint

    @Test("the architecture is read from the checkpoint's own config")
    func configurationIsWhatTheCheckpointSays() throws {
        guard FileManager.default.fileExists(atPath: Self.checkpointPath) else {
            return Self.skip("configurationIsWhatTheCheckpointSays", Self.checkpointPath)
        }
        let header = try SafetensorsHeader.read(from: URL(fileURLWithPath: Self.checkpointPath))
        let config = try LatentSpatialUpsampler.readConfiguration(header)

        #expect(config.inChannels == 128)
        // 1024, and read from the file rather than assumed: an implementation that
        // defaulted to 512 would demand half-width tensors and fail loudly. This is here
        // so the number has a source.
        #expect(config.midChannels == 1024)
        #expect(config.numBlocksPerStage == 4)
        #expect(config.dims == 3)
        #expect(config.spatialScale == 2)
        #expect(config.shuffledChannels == 4096)
        // Not from the checkpoint. See ``Configuration/normGroups``.
        #expect(config.normGroups == 32)

        // The raw metadata, so the assertions above are traceable to text in the file.
        let raw = try #require(header.metadataJSON("config"))
        #expect(raw["_class_name"] as? String == "LatentUpsampler")
        #expect(raw["spatial_upsample"] as? Bool == true)
        #expect(raw["temporal_upsample"] as? Bool == false)
        #expect(raw["rational_resampler"] as? Bool == false)
        #expect((raw["spatial_scale"] as? NSNumber)?.doubleValue == 2.0)
    }

    /// 72 tensors in the file, 72 accounted for, and the rank-4 outlier is the one that
    /// identifies the branch.
    @Test("every tensor in the checkpoint is accounted for, and no other")
    func keyCoverageIsExact() throws {
        guard let module = try Self.module() else {
            return Self.skip("keyCoverageIsExact", Self.checkpointPath)
        }
        let expected = LatentSpatialUpsampler.expectedShapes(module.config)
        let present = Set(module.weights.keys)

        #expect(present.count == 72,
                Comment(rawValue: "checkpoint carries \(present.count) tensors"))
        #expect(Set(expected.keys) == present,
                Comment(rawValue: "unaccounted: \(present.subtracting(expected.keys).sorted()); "
                    + "missing: \(Set(expected.keys).subtracting(present).sorted())"))

        // The rank-4 weight is the checkpoint's own statement that the upsampler is a
        // per-frame Conv2d — the spatial-only, non-rational branch — while every other
        // conv is a Conv3d.
        #expect(module.weights["upsampler.0.weight"]!.shape == [4096, 1024, 3, 3])
        #expect(module.weights["initial_conv.weight"]!.shape == [1024, 128, 3, 3, 3])
        #expect(module.weights["final_conv.weight"]!.shape == [128, 1024, 3, 3, 3])
        let rank5 = present.filter { $0.hasSuffix(".weight") && module.weights[$0]!.ndim == 5 }
        #expect(rank5.count == 18)      // initial + final + 8 blocks x 2

        // The rational-resampler variant would name its tensors `upsampler.conv.*` and
        // carry a `blur_down.kernel` buffer. Neither is here.
        #expect(!present.contains { $0.contains("blur_down") })
        #expect(!present.contains { $0.contains("upsampler.conv") })
        // No `CausalConv3d` wrapper: the VAEs' doubled `.conv.` segment is absent, which
        // is the key-level evidence that padding here is plain zeros on all three axes.
        #expect(!present.contains { $0.contains(".conv.weight") })
        // No block 4: `num_blocks_per_stage` is 4, not 5.
        #expect(!present.contains { $0.contains("res_blocks.4.") })
        // GroupNorm is affine, unlike every PixelNorm in this codebase — the norm tensors
        // exist and are consumed.
        #expect(module.weights["initial_norm.weight"]!.shape == [1024])
        #expect(module.weights["res_blocks.0.norm1.bias"]!.shape == [1024])

        // The statistics came from the *video VAE*, not from this file.
        #expect(!present.contains { $0.contains("per_channel_statistics") })
        #expect(module.stdOfMeans.size == 128)
        #expect(module.meanOfMeans.size == 128)
    }

    @Test("a mis-derived block count demands tensors the checkpoint does not carry")
    func misDerivedConfigDoesNotFitTheCheckpoint() throws {
        guard let module = try Self.module() else {
            return Self.skip("misDerivedConfigDoesNotFitTheCheckpoint", Self.checkpointPath)
        }
        let present = Set(module.weights.keys)
        let c = module.config

        let tooFew = LatentSpatialUpsampler.Configuration(
            inChannels: c.inChannels, midChannels: c.midChannels, numBlocksPerStage: 3,
            dims: c.dims, spatialScale: c.spatialScale)
        let tooMany = LatentSpatialUpsampler.Configuration(
            inChannels: c.inChannels, midChannels: c.midChannels, numBlocksPerStage: 5,
            dims: c.dims, spatialScale: c.spatialScale)
        // 3 blocks: nothing missing, but 16 tensors left unaccounted — the surplus
        // direction, which is the one a "loads fine, runs fine" bug hides in.
        #expect(present.subtracting(LatentSpatialUpsampler.expectedShapes(tooFew).keys).count == 16)
        // 5 blocks: 16 demanded tensors do not exist.
        #expect(Set(LatentSpatialUpsampler.expectedShapes(tooMany).keys)
            .subtracting(present).count == 16)

        // And the width: assuming 512 would ask for half-size tensors.
        let narrow = LatentSpatialUpsampler.Configuration(
            inChannels: 128, midChannels: 512, numBlocksPerStage: 4, dims: 3, spatialScale: 2)
        #expect(LatentSpatialUpsampler.expectedShapes(narrow)["upsampler.0.weight"]
            == [2048, 512, 3, 3])
    }

    @Test("a checkpoint that disagrees with its own config is refused, both directions")
    func aDisagreeingCheckpointIsRefused() throws {
        let config = LatentSpatialUpsampler.Configuration(
            inChannels: 4, midChannels: 64, numBlocksPerStage: 1, dims: 3, spatialScale: 2)
        let stats = Self.identityStatistics(4)

        // Control: the unperturbed file loads.
        let good = try Self.scratch("good.safetensors")
        try Self.writeSynthetic(config: config, to: good)
        let loaded = try LatentSpatialUpsampler(checkpoint: good, statistics: stats)
        #expect(loaded.weights.count == 24)     // 4 + 8 + 2 + 8 + 2

        // A tensor with the wrong shape — the case a rank or channel-count error takes.
        // `upsampler.0.weight` is `[4 x mid, mid, 3, 3]`, so 128 out-channels is a
        // 2x-too-narrow shuffle: a file that would load under a `spatial_scale` of ~1.4
        // if the shuffle factor were inferred from the tensor instead of the config.
        let wrongShape = try Self.scratch("wrong-shape.safetensors")
        try Self.writeSynthetic(config: config,
                                overrides: ["upsampler.0.weight":
                                    MLX.zeros([128, 64, 3, 3], type: Float32.self)],
                                to: wrongShape)
        #expect(throws: (any Error).self) {
            _ = try LatentSpatialUpsampler(checkpoint: wrongShape, statistics: stats)
        }

        // A missing tensor.
        let missing = try Self.scratch("missing.safetensors")
        try Self.writeSynthetic(config: config, drop: ["res_blocks.0.norm2.bias"], to: missing)
        #expect(throws: (any Error).self) {
            _ = try LatentSpatialUpsampler(checkpoint: missing, statistics: stats)
        }

        // A surplus tensor: the file is a variant this branch does not implement, and
        // loading it while ignoring the extra would run a network missing a term.
        let surplus = try Self.scratch("surplus.safetensors")
        try Self.writeSynthetic(config: config,
                                extra: ["res_blocks.0.conv_shortcut.weight":
                                    MLX.zeros([64, 64, 1, 1, 1], type: Float32.self)],
                                to: surplus)
        #expect(throws: (any Error).self) {
            _ = try LatentSpatialUpsampler(checkpoint: surplus, statistics: stats)
        }

        // Statistics of the wrong width — 128 VAE channels against a 4-channel latent.
        #expect(throws: (any Error).self) {
            _ = try LatentSpatialUpsampler(checkpoint: good,
                                           statistics: Self.identityStatistics(128))
        }
    }

    @Test("every config branch this port does not implement is refused by name")
    func unsupportedBranchesAreRefused() throws {
        let config = LatentSpatialUpsampler.Configuration(
            inChannels: 4, midChannels: 64, numBlocksPerStage: 1, dims: 3, spatialScale: 2)
        let base: [String: Any] = [
            "_class_name": "LatentUpsampler", "in_channels": 4, "mid_channels": 64,
            "num_blocks_per_stage": 1, "dims": 3, "spatial_upsample": true,
            "temporal_upsample": false, "spatial_scale": 2.0, "rational_resampler": false,
        ]
        let variants: [String: [String: Any]] = [
            "dims-2": ["dims": 2],
            "temporal": ["temporal_upsample": true],
            "rational": ["rational_resampler": true],
            "scale-1.5": ["spatial_scale": 1.5],
            "no-spatial": ["spatial_upsample": false],
            "wrong-class": ["_class_name": "TemporalUpsampler"],
        ]
        for (name, patch) in variants {
            var json = base
            for (k, v) in patch { json[k] = v }
            let url = try Self.scratch("variant-\(name).safetensors")
            try Self.writeSynthetic(config: config, metadata: json, to: url)
            #expect(throws: (any Error).self,
                    Comment(rawValue: "\(name) must be refused, not silently run as the "
                        + "3D spatial branch")) {
                _ = try LatentSpatialUpsampler(
                    checkpoint: url, statistics: Self.identityStatistics(4))
            }
        }

        // And a file with no config at all: falling back to every default would build a
        // plausible-looking network nobody chose.
        let bare = try Self.scratch("bare.safetensors")
        try MLX.save(arrays: ["initial_conv.weight": MLX.zeros([1], type: Float32.self)],
                     url: bare)
        #expect(throws: (any Error).self) {
            _ = try LatentSpatialUpsampler(checkpoint: bare,
                                           statistics: Self.identityStatistics(4))
        }
    }

    // MARK: - The three things that keep every shape and change every value

    /// `PixelShuffleND(2)`: flat channel `(c·2 + p1)·2 + p2`, `p1` on height, `p2` on
    /// width.
    ///
    /// Built so the answer is countable. `upsampler.0`'s centre tap maps input channel 0
    /// to output channel `j` with weight `j`, and the input is a single voxel with
    /// channel 0 set to 1. So the shuffled 2×2 block for output channel 0 must read
    /// `[[0, 1], [2, 3]]` — and every other sub-pixel ordering is a different pair of
    /// those four numbers, with identical shape and identical channel statistics.
    @Test("the pixel shuffle is c-major with height before width")
    func pixelShuffleOrdering() throws {
        let config = LatentSpatialUpsampler.Configuration(
            inChannels: 4, midChannels: 64, numBlocksPerStage: 1, dims: 3, spatialScale: 2)
        var kernel = [Float](repeating: 0, count: 256 * 64 * 3 * 3)
        for j in 0 ..< 256 {
            // [out, in, kh, kw] with in = 0, kh = kw = 1 (the centre of a 3x3).
            kernel[((j * 64 + 0) * 3 + 1) * 3 + 1] = Float(j)
        }
        let url = try Self.scratch("shuffle.safetensors")
        try Self.writeSynthetic(
            config: config,
            overrides: ["upsampler.0.weight": MLXArray(kernel, [256, 64, 3, 3])],
            to: url)
        let module = try LatentSpatialUpsampler(checkpoint: url,
                                                statistics: Self.identityStatistics(4))

        // NDHWC, one batch, one frame, one voxel, channel 0 hot.
        var input = [Float](repeating: 0, count: 64)
        input[0] = 1
        let x = MLXArray(input, [1, 1, 1, 1, 64])
        let y = try module.spatialUpsample(x)

        #expect(y.shape == [1, 1, 2, 2, 64])
        let values = y.asType(.float32).flattened().asArray(Float.self)
        func at(_ p1: Int, _ p2: Int, _ channel: Int) -> Float {
            values[((p1 * 2) + p2) * 64 + channel]
        }
        #expect(at(0, 0, 0) == 0)
        #expect(at(0, 1, 0) == 1)       // p2 is width and is the least significant
        #expect(at(1, 0, 0) == 2)       // p1 is height
        #expect(at(1, 1, 0) == 3)
        // Channel 1 reads the next four, which pins the decomposition as c-major.
        #expect(at(0, 0, 1) == 4)
        #expect(at(1, 1, 1) == 7)
    }

    /// GroupNorm reduces over `(D, H, W)` **and** a contiguous channel sub-range.
    ///
    /// Checked against an independent scalar implementation written from the definition,
    /// with a fixture whose frames deliberately differ — a per-frame reduction, the
    /// `dims == 2` branch's behaviour, gives visibly different numbers on it.
    @Test("group norm reduces over the whole volume and a contiguous channel group")
    func groupNormReductionSet() throws {
        let groups = 32, channels = 64, frames = 2, height = 2, width = 2
        let config = LatentSpatialUpsampler.Configuration(
            inChannels: 4, midChannels: channels, numBlocksPerStage: 1, dims: 3,
            spatialScale: 2)

        var gamma = [Float](repeating: 0, count: channels)
        var beta = [Float](repeating: 0, count: channels)
        for c in 0 ..< channels {
            gamma[c] = 0.5 + Float(c) * 0.01
            beta[c] = -0.25 + Float(c) * 0.003
        }
        let url = try Self.scratch("groupnorm.safetensors")
        try Self.writeSynthetic(config: config, overrides: [
            "initial_norm.weight": MLXArray(gamma, [channels]),
            "initial_norm.bias": MLXArray(beta, [channels]),
        ], to: url)
        let module = try LatentSpatialUpsampler(checkpoint: url,
                                                statistics: Self.identityStatistics(4))

        let count = frames * height * width * channels
        var raw = [Float](repeating: 0, count: count)
        var seed: UInt64 = 0x2545_F491_4F6C_DD1D
        for i in 0 ..< count {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            raw[i] = Float(Int32(truncatingIfNeeded: seed >> 33)) / Float(1 << 30)
        }
        // Make frame 1 an offset copy of frame 0: identical per-frame statistics, so a
        // per-frame reduction and a per-clip reduction differ by a constant shift.
        let perFrame = height * width * channels
        for i in 0 ..< perFrame { raw[perFrame + i] = raw[i] + 2.0 }
        let x = MLXArray(raw, [1, frames, height, width, channels])

        let observed = try module.groupNorm(x, "initial_norm")
            .asType(.float32).flattened().asArray(Float.self)

        // Independent scalar reference, from the definition.
        let perGroup = channels / groups
        let positions = frames * height * width
        var expected = [Float](repeating: 0, count: count)
        for g in 0 ..< groups {
            var sum = 0.0, sumSquares = 0.0
            let n = Double(positions * perGroup)
            for p in 0 ..< positions {
                for k in 0 ..< perGroup {
                    let v = Double(raw[p * channels + g * perGroup + k])
                    sum += v
                    sumSquares += v * v
                }
            }
            let mean = sum / n
            let variance = sumSquares / n - mean * mean
            let rstd = 1.0 / (variance + 1e-5).squareRoot()
            for p in 0 ..< positions {
                for k in 0 ..< perGroup {
                    let c = g * perGroup + k
                    let v = Double(raw[p * channels + c])
                    expected[p * channels + c] =
                        Float((v - mean) * rstd * Double(gamma[c]) + Double(beta[c]))
                }
            }
        }

        var worst = 0.0
        for i in 0 ..< count { worst = max(worst, Double(abs(observed[i] - expected[i]))) }
        #expect(worst < 2e-5, Comment(rawValue: "group norm max abs \(worst)"))

        // The negative control: a per-frame reduction is a genuinely different answer on
        // this fixture, so the check above has something to fail.
        var perFrameWorst = 0.0
        for p in 0 ..< perFrame {
            perFrameWorst = max(perFrameWorst,
                                Double(abs(observed[perFrame + p] - observed[p])))
        }
        #expect(perFrameWorst > 0.1,
                Comment(rawValue: "frames 0 and 1 are identical up to \(perFrameWorst); "
                    + "the fixture no longer discriminates a per-frame reduction"))
    }

    /// `self.activation(x + residual)`: the sum goes through the second SiLU.
    ///
    /// With every conv and every norm zeroed, the block body contributes exactly 0, so
    /// the block reduces to `silu(x)` if the residual is inside the activation and to
    /// `x` if it is outside. The two differ everywhere `x < 0`.
    @Test("the residual is added before the second activation, not after")
    func residualPassesThroughTheActivation() throws {
        let config = LatentSpatialUpsampler.Configuration(
            inChannels: 4, midChannels: 64, numBlocksPerStage: 1, dims: 3, spatialScale: 2)
        let url = try Self.scratch("residual.safetensors")
        try Self.writeSynthetic(config: config, to: url)   // all zeros
        let module = try LatentSpatialUpsampler(checkpoint: url,
                                                statistics: Self.identityStatistics(4))

        let values: [Float] = (0 ..< 64).map { Float($0) * 0.25 - 8.0 }
        let x = MLXArray(values, [1, 1, 1, 1, 64])
        let y = try module.resBlock(x, "res_blocks.0")
        let observed = y.asType(.float32).flattened().asArray(Float.self)

        for (i, v) in values.enumerated() {
            let silu = v / (1 + Foundation.expf(-v))
            #expect(abs(observed[i] - silu) < 1e-5,
                    Comment(rawValue: "resBlock(\(v)) = \(observed[i]), silu = \(silu)"))
        }
        // And it is not the identity: `silu(h) + x` would return `x` here.
        let negative = observed[0]
        #expect(abs(negative - values[0]) > 1.0,
                Comment(rawValue: "block returned \(negative) for input \(values[0]); the "
                    + "residual appears to be outside the activation"))
    }

    // MARK: - Shape and dtype, on the real checkpoint

    /// 2× in height and width, unchanged in frames and channels. Stated as four separate
    /// facts because only one of them is the feature's headline.
    @Test("the output is exactly 2x spatially and unchanged in frames and channels")
    func outputGeometry() throws {
        guard let module = try Self.module() else {
            return Self.skip("outputGeometry", Self.checkpointPath)
        }
        for shape in [[1, 128, 1, 2, 3], [1, 128, 3, 4, 2], [2, 128, 2, 3, 3]] {
            let latent = MLX.zeros(shape, type: Float32.self)
            let out = try module.upsample(latent)
            #expect(out.shape == [shape[0], 128, shape[2], shape[3] * 2, shape[4] * 2],
                    Comment(rawValue: "\(shape) -> \(out.shape)"))
        }

        // A non-square, odd-framed shape, because `13` frames and a `6x10` grid is what
        // production actually hands it and a temporal drop would show up here.
        let production = MLX.zeros([1, 128, 13, 6, 10], type: Float32.self)
        let upsampled = try module.upsample(production)
        #expect(upsampled.shape == [1, 128, 13, 12, 20])

        // Rank and channel count are refused rather than reinterpreted.
        #expect(throws: (any Error).self) {
            _ = try module.forward(MLX.zeros([1, 128, 4, 4], type: Float32.self))
        }
        #expect(throws: (any Error).self) {
            _ = try module.forward(MLX.zeros([1, 64, 2, 4, 4], type: Float32.self))
        }
    }

    /// The dtype contract: bf16 weights, bf16 output, whatever the input was.
    @Test("the latent is narrowed to the weight dtype and the output is bf16")
    func dtypeContract() throws {
        guard let module = try Self.module() else {
            return Self.skip("dtypeContract", Self.checkpointPath)
        }
        #expect(module.weights["initial_conv.weight"]!.dtype == .bfloat16)
        #expect(module.weights["initial_norm.weight"]!.dtype == .bfloat16)

        // An fp32 latent must not promote the network to fp32. Same reason
        // ``VideoVAEDecoder/decode(_:)`` casts.
        let float = MLX.zeros([1, 128, 2, 2, 3], type: Float32.self)
        #expect(try module.forward(float).dtype == .bfloat16)
        #expect(try module.upsample(float).dtype == .bfloat16)
        let bf16 = float.asType(.bfloat16)
        #expect(try module.forward(bf16).dtype == .bfloat16)

        // And the two agree: casting on the way in is what makes them agree, so this
        // fails if the fp32 arm silently ran a more precise network.
        let a = try module.forward(float).asType(.float32).flattened().asArray(Float.self)
        let b = try module.forward(bf16).asType(.float32).flattened().asArray(Float.self)
        var worst = 0.0
        for i in 0 ..< a.count { worst = max(worst, Double(abs(a[i] - b[i]))) }
        #expect(worst == 0, Comment(rawValue: "fp32 and bf16 inputs differ by \(worst)"))
    }

    /// The normalisation round trip is *present*, and it is not a no-op.
    ///
    /// ``LatentSpatialUpsampler/upsample(_:)`` un-normalises before the network and
    /// re-normalises after, using statistics that live in the **video VAE**. An
    /// implementation that stopped at ``LatentSpatialUpsampler/forward(_:)`` would produce
    /// a right-shaped, finite, plausible latent from the wrong distribution, and nothing
    /// downstream would raise.
    @Test("the per-channel normalisation round trip is applied and is not the identity")
    func normalisationRoundTripIsApplied() throws {
        guard let module = try Self.module() else {
            return Self.skip("normalisationRoundTripIsApplied", Self.checkpointPath)
        }
        // The statistics are genuinely non-trivial: if they were 1/0 the round trip
        // would be undetectable and every claim below would be vacuous.
        let std = module.stdOfMeans.asType(.float32).asArray(Float.self)
        let mean = module.meanOfMeans.asType(.float32).asArray(Float.self)
        #expect(std.count == 128 && mean.count == 128)
        #expect(std.contains { abs($0 - 1) > 0.05 })
        #expect(mean.contains { abs($0) > 0.01 })

        // denormalize/renormalize invert each other.
        let latent = MLX.zeros([1, 128, 1, 1, 1], type: Float32.self) + MLXArray(Float(0.5))
        let round = module.renormalize(module.denormalize(latent))
        let residual = MLX.abs(round - latent).max().item(Float.self)
        #expect(residual < 1e-5, Comment(rawValue: "round trip residual \(residual)"))

        // And the composite is not the bare forward pass.
        let input = MLX.zeros([1, 128, 2, 2, 3], type: Float32.self)
        let bare = try module.forward(input).asType(.float32)
        let full = try module.upsample(input).asType(.float32)
        let gap = MLX.abs(full - bare).max().item(Float.self)
        #expect(gap > 1e-3,
                Comment(rawValue: "upsample and forward differ by only \(gap); the "
                    + "normalisation round trip is missing or inert"))
    }
}

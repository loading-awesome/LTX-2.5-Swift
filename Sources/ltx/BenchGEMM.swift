// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTX25
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX
import MLXFast
import Metal
import MetalPerformanceShadersGraph

/// `ltx bench gemm` — Step 0 of `docs/PERFORMANCE.md`, the roofline.
///
/// ## Three rows, not two
///
/// The doc's Step 0 as written measures one vendor at one set of shapes and calls the
/// result the machine's ceiling. That method has already produced one retracted ceiling
/// claim, and this command deliberately does not reproduce it. What runs here is:
///
/// 1. **MLX, at the model's own shapes, in the checkpoint's own layout.** The checkpoint
///    stores every projection as `[N, K]` and production computes
///    `matmul(x, w.transposed(1, 0))` — see `DiTModules.linear` and `DiTAttention.linear`,
///    which are the only two places a DiT GEMM happens. Measuring a contiguous `[K, N]`
///    operand instead measures a kernel the model never calls: that error has
///    misattributed 1.87 s of a 2.3 s envelope and invented 9% of reclaimable slack that
///    did not exist.
/// 2. **A second vendor at the same shapes, fed the same bytes.** MPSGraph bf16, from an
///    `MTLBuffer` filled with the identical bit patterns handed to MLX. The second vendor
///    has been measured beating the first by 1.12–1.22x while staying bit-identical, which
///    is the entire reason "we are at the ceiling" was retracted. A single-vendor roofline
///    measures a library, not a machine.
/// 3. **Attention separately**, through `MLXFast.scaledDotProductAttention` — the call
///    `DiTAttention` actually makes — at the real head geometry.
///
/// ## The one-ULP control
///
/// A comparison column that reads `0.00e+00` is either a real zero or a dead instrument,
/// and nothing in the number distinguishes them. So every comparison ships with a control:
/// one bf16 ulp is flipped in one element of `x`, MLX is re-run, and the same comparison is
/// taken against the *unmodified* second-vendor output. If the control moves, the zero is
/// real. If the control is also zero, the comparison is not measuring anything and the
/// bit-identical claim must not be made.
///
/// ## Never the square
///
/// A synthetic square GEMM is not the ceiling and is not reported as one. A 4096³ square
/// read 17.6 TFLOP/s against the model's own shapes at 16.0 and was the basis of a
/// retracted claim. `--include-square` adds it, labelled a best case, and it is excluded
/// from the ceiling summary.
struct BenchGEMM: ParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "gemm",
        abstract: "The machine's GEMM and attention ceiling at the model's own shapes.",
        discussion: """
            Shapes come from a ranged header read of the transformer checkpoint — no \
            payload is touched and no 42 GB load happens. Operands are synthetic; this \
            measures the machine, not a render.
            """)

    // MARK: Shape inputs

    @Option(help: "Config file. Defaults to ~/.config/ltx/config.json.")
    var config: String?

    @Option(help: "OVERRIDE the DiT checkpoint. Read HEADER ONLY, for shapes.")
    var checkpoint: String?

    @Option(help: "Pixel frames, on the 8k+1 lattice. Sets the video token count.")
    var frames: Int = 97

    @Option(help: "Width in pixels. Overrides --megapixels. Multiple of 64.")
    var width: Int?

    @Option(help: "Height in pixels. Overrides --megapixels. Multiple of 64.")
    var height: Int?

    @Option(help: "Megapixels (e.g. 0.2, 0.7, 1.0, 2.0).")
    var megapixels: Double?

    @Option(help: "Aspect ratio (16:9, 9:16, 1:1, 21:9, 4:3, 3:4).")
    var aspectRatio: String = "16:9"

    @Option(help: "Frame rate. Sets the audio token count.")
    var fps: Double = 24

    @Option(help: ArgumentHelp("Text conditioning length. Contract 9 forces min_length = "
                + "1024, so this is 1024 on every real render."))
    var textTokens: Int = 1024

    // MARK: Protocol

    @Option(help: "Warm-up evaluations, discarded. Compilation and first-touch land here.")
    var warmup: Int = 3

    @Option(help: "Timed repeats after the warm-up. Median, min and max are all reported.")
    var repeats: Int = 10

    @Option(help: ArgumentHelp("Only measure cases whose name contains this substring. "
                + "Use for a quick smoke, e.g. --only patchify."))
    var only: String?

    @Flag(help: ArgumentHelp("Add a synthetic square GEMM, LABELLED A BEST CASE. It is "
                + "excluded from the ceiling summary — a square is not the model's shape "
                + "and reporting it as the ceiling is the error this flag exists to "
                + "keep visible rather than repeat."))
    var includeSquare: Bool = false

    @Flag(help: "Skip the MPSGraph vendor. Leaves the roofline single-vendor and weaker.")
    var skipMps: Bool = false

    @Flag(help: "Skip the attention cases.")
    var skipAttention: Bool = false

    @Option(help: "Seed for the synthetic operands. Deterministic.")
    var seed: UInt64 = 0x5EED

    @Option(help: "Where to write the machine-readable record.")
    var out: String?

    @Option(help: "Optional flat CSV sidecar.")
    var csv: String?

    // MARK: - A case

    /// One production GEMM: `y = x @ w.T`, `x` is `[m, k]` and `w` is `[n, k]` — the
    /// checkpoint's layout, not a transposed copy of it.
    private struct GEMMCase {
        let name: String
        let tensor: String?
        let m: Int
        let n: Int
        let k: Int
        let rowSource: String
        /// Whether this case counts toward the reported ceiling. False for the square.
        let isProduction: Bool

        var flops: Double { 2 * Double(m) * Double(n) * Double(k) }
        /// bf16 operands and result, once each. The floor a memory-bound shape sits on.
        var operandBytes: Double { 2 * (Double(m) * Double(k) + Double(n) * Double(k) + Double(m) * Double(n)) }
    }

    /// One attention case, through the production SDPA call.
    private struct AttentionCase {
        let name: String
        let heads: Int
        let queryTokens: Int
        let keyTokens: Int
        let headDim: Int

        /// `4 * S * S * H * D` in the square case — two GEMMs (`q·kᵀ` and `p·v`), each
        /// `2 * S * S * D` per head. Generalised to a rectangular `q`/`k` here because
        /// the cross-modal and text-cross modules are not square.
        var flops: Double {
            4 * Double(queryTokens) * Double(keyTokens) * Double(heads) * Double(headDim)
        }
    }

    // MARK: - Run

    func run() throws {
        guard warmup >= 0, repeats >= 1 else {
            throw ValidationError("--warmup must be >= 0 and --repeats >= 1")
        }
        guard let aspect = LTX25.Aspect(rawValue: aspectRatio) else {
            throw ValidationError("--aspect-ratio must be one of "
                + LTX25.Aspect.allCases.map(\.rawValue).joined(separator: ", "))
        }

        // Geometry, resolved exactly the way `render` resolves it.
        let request = LTX25.Request(
            prompt: "bench", videoOutput: URL(fileURLWithPath: "/dev/null"),
            seconds: Double(frames) / fps, frames: frames, frameRate: fps,
            megapixels: megapixels, aspectRatio: aspect, width: width, height: height)
        let shape = try request.resolvedShape()
        let resolvedFrames = try request.resolvedFrames()
        let geometry = DiTForward.Geometry(frames: resolvedFrames, height: shape.height,
                                           width: shape.width, frameRate: fps)
        try geometry.validate()

        // A ranged header read. No payload, no 42 GB load.
        var paths = try Paths(configPath: config)
        let ditURL = try paths.url(.ditDev, override: checkpoint)
        paths.announce()
        // The header is all this reads, but the measurement itself is GPU work, so the
        // kernels have to be there.
        try Preflight.check(paths: paths, slots: [
            CheckpointInventory.Slot(role: "dit", url: ditURL, expected: .transformer,
                                     nameMustContain: nil),
        ])
        let header = try SafetensorsHeader.read(from: ditURL)
        let topology = try TransformerTopology.read(header)

        let videoTokens = geometry.videoTokens
        let audioTokens = geometry.audioTokens

        var caveats: [String] = []
        let cases = productionCases(header: header, topology: topology,
                                    videoTokens: videoTokens, audioTokens: audioTokens,
                                    caveats: &caveats)
        guard !cases.isEmpty else {
            throw ValidationError("no GEMM cases matched --only '\(only ?? "")'")
        }

        let device = MTLCreateSystemDefaultDevice()
        var mpsAvailable = !skipMps && device != nil
        if skipMps {
            caveats.append("MPSGraph vendor skipped by --skip-mps: the roofline is "
                + "single-vendor and cannot support a 'we are at the ceiling' claim.")
        } else if device == nil {
            caveats.append("No Metal device: the MPSGraph vendor could not run.")
            mpsAvailable = false
        }

        BenchPeakMemory.reset()
        var results: [BenchJSON] = []
        var csvRows = BenchCSV(header: [
            "case", "kind", "vendor", "m", "n", "k", "flops",
            "median_s", "min_s", "max_s", "spread", "tflops_median", "tflops_best",
            "implied_operand_gb_s", "is_production", "rel_rms_vs_mps", "max_abs_vs_mps",
            "bit_identical_fraction", "one_ulp_control_rel_rms",
        ])

        var bestMLX: (name: String, tflops: Double)?
        var bestMPS: (name: String, tflops: Double)?

        for gemm in cases {
            FileHandle.standardError.write(Data(
                "gemm \(gemm.name)  [\(gemm.m)x\(gemm.k)] @ [\(gemm.n)x\(gemm.k)]^T\n".utf8))
            let measured = try measure(gemm, device: mpsAvailable ? device : nil)
            results.append(measured.json)

            if gemm.isProduction {
                if bestMLX == nil || measured.mlxTFLOPSMedian > bestMLX!.tflops {
                    bestMLX = (gemm.name, measured.mlxTFLOPSMedian)
                }
                if let mpsTFLOPS = measured.mpsTFLOPSMedian,
                   bestMPS == nil || mpsTFLOPS > bestMPS!.tflops {
                    bestMPS = (gemm.name, mpsTFLOPS)
                }
            }

            csvRows.append([
                gemm.name, "gemm", "mlx", String(gemm.m), String(gemm.n), String(gemm.k),
                BenchCSV.cell(gemm.flops),
                BenchCSV.cell(measured.mlx.median), BenchCSV.cell(measured.mlx.minimum),
                BenchCSV.cell(measured.mlx.maximum), BenchCSV.cell(measured.mlx.spread),
                BenchCSV.cell(measured.mlxTFLOPSMedian),
                BenchCSV.cell(gemm.flops / measured.mlx.minimum / 1e12),
                BenchCSV.cell(gemm.operandBytes / measured.mlx.median / 1e9),
                gemm.isProduction ? "true" : "false",
                BenchCSV.cell(measured.comparison?.relativeRMS),
                BenchCSV.cell(measured.comparison?.maxAbsoluteDifference),
                BenchCSV.cell(measured.comparison?.bitIdenticalFraction),
                BenchCSV.cell(measured.oneULPControl?.relativeRMS),
            ])
            if let mps = measured.mps {
                csvRows.append([
                    gemm.name, "gemm", "mpsgraph", String(gemm.m), String(gemm.n), String(gemm.k),
                    BenchCSV.cell(gemm.flops),
                    BenchCSV.cell(mps.median), BenchCSV.cell(mps.minimum),
                    BenchCSV.cell(mps.maximum), BenchCSV.cell(mps.spread),
                    BenchCSV.cell(measured.mpsTFLOPSMedian),
                    BenchCSV.cell(gemm.flops / mps.minimum / 1e12),
                    BenchCSV.cell(gemm.operandBytes / mps.median / 1e9),
                    gemm.isProduction ? "true" : "false", "", "", "", "",
                ])
            }
        }

        // Attention, through the production call.
        var attentionResults: [BenchJSON] = []
        if !skipAttention {
            for attention in attentionCases(topology: topology, videoTokens: videoTokens,
                                            audioTokens: audioTokens) {
                FileHandle.standardError.write(Data("sdpa \(attention.name)\n".utf8))
                let stats = try measure(attention)
                let tflops = attention.flops / stats.median / 1e12
                attentionResults.append(.object([
                    "name": .string(attention.name),
                    "vendor": .string("mlxfast.scaledDotProductAttention"),
                    "heads": .int(attention.heads),
                    "queryTokens": .int(attention.queryTokens),
                    "keyTokens": .int(attention.keyTokens),
                    "headDim": .int(attention.headDim),
                    "flopConvention": .string("4 * Sq * Skv * H * D"),
                    "flops": .double(attention.flops),
                    "timing": stats.json,
                    "tflopsFromMedian": .double(tflops),
                    "tflopsFromMin": .double(attention.flops / stats.minimum / 1e12),
                ]))
                csvRows.append([
                    attention.name, "attention", "mlxfast_sdpa",
                    String(attention.queryTokens), String(attention.keyTokens),
                    String(attention.headDim), BenchCSV.cell(attention.flops),
                    BenchCSV.cell(stats.median), BenchCSV.cell(stats.minimum),
                    BenchCSV.cell(stats.maximum), BenchCSV.cell(stats.spread),
                    BenchCSV.cell(tflops), BenchCSV.cell(attention.flops / stats.minimum / 1e12),
                    "", "true", "", "", "", "",
                ])
            }
            caveats.append("Attention has one vendor only. A second implementation would "
                + "be a reimplementation of the kernel rather than a second vendor over "
                + "the same call, so no MPSGraph column is offered and no "
                + "'attention is at the ceiling' claim is supported by this run.")
        } else {
            caveats.append("Attention skipped by --skip-attention.")
        }

        let record = BenchRecord(
            kind: "gemm",
            invocation: CommandLine.arguments,
            parameters: .object([
                "checkpoint": .string(ditURL.path),
                "checkpointHeaderOnly": .bool(true),
                "geometry": .object([
                    "frames": .int(geometry.frames),
                    "width": .int(geometry.width),
                    "height": .int(geometry.height),
                    "frameRate": .double(geometry.frameRate),
                    "latentFrames": .int(geometry.latentFrames),
                    "videoTokens": .int(videoTokens),
                    "audioTokens": .int(audioTokens),
                    "textTokens": .int(textTokens),
                ]),
                "topology": .object([
                    "blockCount": .int(topology.blockCount),
                    "videoWidth": .int(topology.videoWidth),
                    "audioWidth": .int(topology.audioWidth),
                    "videoFeedForwardWidth": .int(topology.videoFeedForwardWidth),
                    "audioFeedForwardWidth": .int(topology.audioFeedForwardWidth),
                    "numAttentionHeads": .int(topology.numAttentionHeads),
                    "latentChannels": .int(topology.latentChannels),
                ]),
                "protocol": .object([
                    "warmupEvaluations": .int(warmup),
                    "timedRepeats": .int(repeats),
                    "evalInsideTimingLoop": .bool(true),
                    "dtype": .string("bfloat16"),
                    "operandLayout": .string("x [m, k] and w [n, k]; y = matmul(x, w.T) — "
                        + "the checkpoint's own layout, as DiTModules.linear computes it"),
                    "flopConventionGEMM": .string("2 * m * n * k"),
                    "flopConventionAttention": .string("4 * Sq * Skv * H * D"),
                    "operandSeed": .int(Int(bitPattern: UInt(seed))),
                ]),
            ]),
            measurements: .object([
                "gemm": .array(results),
                "attention": .array(attentionResults),
                "ceiling": .object([
                    "note": .string("Best achieved TFLOP/s over the model's OWN production "
                        + "shapes, per vendor. A synthetic square is never eligible."),
                    "mlxBestCase": .opt(bestMLX?.name),
                    "mlxBestTFLOPS": .opt(bestMLX?.tflops),
                    "mpsGraphBestCase": .opt(bestMPS?.name),
                    "mpsGraphBestTFLOPS": .opt(bestMPS?.tflops),
                    "mpsGraphOverMLX": .opt(bestMPS.flatMap { mps in
                        bestMLX.map { mps.tflops / $0.tflops }
                    }),
                ]),
                "plausibility": .object([
                    "note": .string("Reported, never asserted. Apple GPUs before M5 have no "
                        + "dedicated matrix units; a bf16 GEMM figure far above the fp32 "
                        + "ALU peak, or an implied operand bandwidth above the machine's "
                        + "memory bandwidth, means the timing is measuring lazy graph "
                        + "construction rather than completed work."),
                    "systemMemoryBytes": .opt(BenchRuntime.sysctlInt("hw.memsize")),
                ]),
            ]),
            caveats: caveats,
            peakMemoryAllocatorBytes: BenchPeakMemory.allocatorPeakBytes)

        try record.write(to: out)
        try csvRows.write(to: csv)

        FileHandle.standardError.write(Data(summary(bestMLX: bestMLX, bestMPS: bestMPS).utf8))
    }

    private func summary(bestMLX: (name: String, tflops: Double)?,
                         bestMPS: (name: String, tflops: Double)?) -> String {
        var text = "\nceiling at the model's own shapes (bf16, x @ w.T):\n"
        if let bestMLX {
            text += String(format: "  MLX       %.2f TFLOP/s  (%@)\n", bestMLX.tflops, bestMLX.name)
        }
        if let bestMPS {
            text += String(format: "  MPSGraph  %.2f TFLOP/s  (%@)\n", bestMPS.tflops, bestMPS.name)
        }
        if let bestMLX, let bestMPS {
            text += String(format: "  MPSGraph / MLX = %.3fx\n", bestMPS.tflops / bestMLX.tflops)
        }
        if bestMLX == nil {
            text += "  no production shape was measured — every eligible case was filtered "
                + "out, so there is no ceiling here. A synthetic square is never eligible.\n"
        } else if bestMPS == nil {
            text += "  second vendor absent: this is a library measurement, not a machine ceiling\n"
        }
        return text
    }

    // MARK: - Case construction

    /// The production GEMMs, with `n` and `k` read from the checkpoint's actual weight
    /// shapes rather than declared here.
    ///
    /// A name that is not in the header is *skipped and recorded*, never guessed at: a
    /// wrong tensor name would otherwise silently reduce the sweep.
    private func productionCases(header: SafetensorsHeader, topology: TransformerTopology,
                                 videoTokens: Int, audioTokens: Int,
                                 caveats: inout [String]) -> [GEMMCase] {
        // (tensor suffix, rows of x, what those rows are)
        let wanted: [(String, Int, String)] = [
            ("patchify_proj.weight", videoTokens, "videoTokens"),
            ("audio_patchify_proj.weight", audioTokens, "audioTokens"),
            ("transformer_blocks.0.attn1.to_q.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.attn1.to_out.0.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.attn2.to_q.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.attn2.to_k.weight", textTokens, "textTokens"),
            ("transformer_blocks.0.ff.net.0.proj.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.ff.net.2.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.audio_attn1.to_q.weight", audioTokens, "audioTokens"),
            ("transformer_blocks.0.audio_attn1.to_out.0.weight", audioTokens, "audioTokens"),
            ("transformer_blocks.0.audio_attn2.to_k.weight", textTokens, "textTokens"),
            ("transformer_blocks.0.audio_ff.net.0.proj.weight", audioTokens, "audioTokens"),
            ("transformer_blocks.0.audio_ff.net.2.weight", audioTokens, "audioTokens"),
            // Both cross-modal modules are named after the stream they READ and write the
            // other one; see TransformerTopology.verifyCrossModalWiring. The row counts
            // below follow the stream the input actually belongs to, not the name.
            ("transformer_blocks.0.audio_to_video_attn.to_q.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.audio_to_video_attn.to_out.0.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.video_to_audio_attn.to_k.weight", videoTokens, "videoTokens"),
            ("transformer_blocks.0.video_to_audio_attn.to_out.0.weight", audioTokens, "audioTokens"),
        ]

        var cases: [GEMMCase] = []
        for (suffix, rows, rowSource) in wanted {
            let full = topology.prefix + suffix
            guard let entry = header.tensors[full], entry.shape.count == 2 else {
                caveats.append("tensor \(full) absent from the header (or not rank 2); its "
                    + "GEMM case was skipped rather than assumed.")
                continue
            }
            let name = suffix
                .replacingOccurrences(of: "transformer_blocks.0.", with: "block0.")
            cases.append(GEMMCase(name: name, tensor: full, m: rows,
                                  n: entry.shape[0], k: entry.shape[1],
                                  rowSource: rowSource, isProduction: true))
        }

        if includeSquare {
            let side = topology.videoWidth
            cases.append(GEMMCase(
                name: "SYNTHETIC_SQUARE_\(side)_BEST_CASE_NOT_THE_CEILING",
                tensor: nil, m: side, n: side, k: side,
                rowSource: "synthetic", isProduction: false))
        }

        if let only, !only.isEmpty {
            cases = cases.filter { $0.name.contains(only) }
        }
        return cases
    }

    private func attentionCases(topology: TransformerTopology,
                                videoTokens: Int, audioTokens: Int) -> [AttentionCase] {
        let heads = topology.numAttentionHeads
        let videoHeadDim = topology.videoWidth / heads
        let audioHeadDim = topology.audioWidth / heads
        var cases: [AttentionCase] = [
            AttentionCase(name: "attn1.video_self", heads: heads,
                          queryTokens: videoTokens, keyTokens: videoTokens,
                          headDim: videoHeadDim),
            AttentionCase(name: "audio_attn1.audio_self", heads: heads,
                          queryTokens: audioTokens, keyTokens: audioTokens,
                          headDim: audioHeadDim),
            AttentionCase(name: "attn2.video_text_cross", heads: heads,
                          queryTokens: videoTokens, keyTokens: textTokens,
                          headDim: videoHeadDim),
            AttentionCase(name: "audio_attn2.audio_text_cross", heads: heads,
                          queryTokens: audioTokens, keyTokens: textTokens,
                          headDim: audioHeadDim),
            // Query from video, key from audio — the direction the checkpoint's shapes
            // prove, not the one the module's name suggests.
            AttentionCase(name: "audio_to_video_attn.cross", heads: heads,
                          queryTokens: videoTokens, keyTokens: audioTokens,
                          headDim: audioHeadDim),
            AttentionCase(name: "video_to_audio_attn.cross", heads: heads,
                          queryTokens: audioTokens, keyTokens: videoTokens,
                          headDim: audioHeadDim),
        ]
        if let only, !only.isEmpty {
            cases = cases.filter { $0.name.contains(only) }
        }
        return cases
    }

    // MARK: - Measurement

    private struct GEMMResult {
        let mlx: BenchStats
        let mps: BenchStats?
        let mlxTFLOPSMedian: Double
        let mpsTFLOPSMedian: Double?
        let comparison: BenchComparison?
        let oneULPControl: BenchComparison?
        let json: BenchJSON
    }

    /// One GEMM case, both vendors, plus the comparison and its control.
    private func measure(_ gemm: GEMMCase, device: MTLDevice?) throws -> GEMMResult {
        // The bytes. Generated once and shared: this is what makes the two vendors
        // comparable at all.
        let xBits = BenchOperand.bits(count: gemm.m * gemm.k, seed: seed)
        let wBits = BenchOperand.bits(count: gemm.n * gemm.k, seed: seed &+ 1)

        let x = BenchOperand.mlx(xBits, shape: [gemm.m, gemm.k])
        let w = BenchOperand.mlx(wBits, shape: [gemm.n, gemm.k])
        MLX.eval(x, w)

        // `w.transposed(1, 0)` and NOT a materialised [k, n] copy: this is character for
        // character what `DiTModules.linear` computes.
        func mlxProduct(_ lhs: MLXArray) -> MLXArray { MLX.matmul(lhs, w.transposed(1, 0)) }

        for _ in 0..<warmup { MLX.eval(mlxProduct(x)) }
        var mlxSamples: [Double] = []
        mlxSamples.reserveCapacity(repeats)
        for _ in 0..<repeats {
            let start = benchNow()
            let y = mlxProduct(x)
            // Inside the loop. Without it this measures lazy graph construction, which on
            // MLX is microseconds regardless of the shape.
            MLX.eval(y)
            mlxSamples.append(benchNow() - start)
        }
        let mlxStats = BenchStats(mlxSamples)
        let mlxReference = BenchOperand.bits(of: mlxProduct(x))

        var mpsStats: BenchStats?
        var comparison: BenchComparison?
        var control: BenchComparison?
        var mpsNote: String?

        if let device {
            do {
                let vendor = try BenchMPSGraphGEMM(device: device, m: gemm.m, n: gemm.n, k: gemm.k)
                vendor.load(xBits: xBits, wBits: wBits)
                for _ in 0..<warmup { vendor.run() }
                var samples: [Double] = []
                samples.reserveCapacity(repeats)
                for _ in 0..<repeats {
                    let start = benchNow()
                    vendor.run()   // synchronous: submit and wait, completed work
                    samples.append(benchNow() - start)
                }
                mpsStats = BenchStats(samples)
                let mpsResult = vendor.outputBits()
                comparison = BenchComparison(candidate: mlxReference, reference: mpsResult)

                // The control. One bf16 ulp in one element of x, MLX re-run, compared
                // against the UNMODIFIED MPSGraph output. A zero here means the comparison
                // above is dead and its 0.00e+00 proves nothing.
                var perturbed = xBits
                perturbed[0] ^= 1
                let xPerturbed = BenchOperand.mlx(perturbed, shape: [gemm.m, gemm.k])
                let yPerturbed = BenchOperand.bits(of: MLX.matmul(xPerturbed, w.transposed(1, 0)))
                control = BenchComparison(candidate: yPerturbed, reference: mpsResult)
            } catch {
                mpsNote = "MPSGraph vendor failed: \(error)"
            }
        }

        let mlxTFLOPS = gemm.flops / mlxStats.median / 1e12
        let mpsTFLOPS = mpsStats.map { gemm.flops / $0.median / 1e12 }

        var json: [String: BenchJSON] = [
            "name": .string(gemm.name),
            "tensor": .opt(gemm.tensor),
            "isProductionShape": .bool(gemm.isProduction),
            "rowSource": .string(gemm.rowSource),
            "m": .int(gemm.m), "n": .int(gemm.n), "k": .int(gemm.k),
            "flopConvention": .string("2 * m * n * k"),
            "flops": .double(gemm.flops),
            "operandBytes": .double(gemm.operandBytes),
            "mlx": .object([
                "timing": mlxStats.json,
                "tflopsFromMedian": .double(mlxTFLOPS),
                "tflopsFromMin": .double(gemm.flops / mlxStats.minimum / 1e12),
                "impliedOperandBandwidthGBs": .double(gemm.operandBytes / mlxStats.median / 1e9),
            ]),
        ]
        if let mpsStats, let mpsTFLOPS {
            json["mpsGraph"] = .object([
                "timing": mpsStats.json,
                "tflopsFromMedian": .double(mpsTFLOPS),
                "tflopsFromMin": .double(gemm.flops / mpsStats.minimum / 1e12),
                "impliedOperandBandwidthGBs": .double(gemm.operandBytes / mpsStats.median / 1e9),
                "speedupOverMLX": .double(mpsTFLOPS / mlxTFLOPS),
            ])
        } else {
            json["mpsGraph"] = .null
        }
        json["comparison"] = .object([
            "note": .string("MLX against MPSGraph over identical bf16 bytes."),
            "mlxVsMpsGraph": comparison?.json ?? .null,
            "oneULPControl": .object([
                "note": .string("One bf16 ulp flipped in x[0], MLX re-run, compared against "
                    + "the UNMODIFIED MPSGraph output. If this does not move, the row above "
                    + "is a dead instrument and its zero means nothing."),
                "result": control?.json ?? .null,
                "moved": control.map { BenchJSON.bool($0.relativeRMS > 0) } ?? .null,
            ]),
            "vendorFailure": .opt(mpsNote),
        ])

        return GEMMResult(mlx: mlxStats, mps: mpsStats, mlxTFLOPSMedian: mlxTFLOPS,
                          mpsTFLOPSMedian: mpsTFLOPS, comparison: comparison,
                          oneULPControl: control, json: .object(json))
    }

    /// One attention case through `MLXFast.scaledDotProductAttention` — the exact call
    /// `DiTAttention` makes, with the same `1/sqrt(dimHead)` scale and no mask.
    private func measure(_ attention: AttentionCase) throws -> BenchStats {
        let qShape = [1, attention.heads, attention.queryTokens, attention.headDim]
        let kvShape = [1, attention.heads, attention.keyTokens, attention.headDim]
        let q = BenchOperand.mlx(BenchOperand.bits(count: qShape.reduce(1, *), seed: seed &+ 2),
                                 shape: qShape)
        let k = BenchOperand.mlx(BenchOperand.bits(count: kvShape.reduce(1, *), seed: seed &+ 3),
                                 shape: kvShape)
        let v = BenchOperand.mlx(BenchOperand.bits(count: kvShape.reduce(1, *), seed: seed &+ 4),
                                 shape: kvShape)
        MLX.eval(q, k, v)
        let scale = Float(1.0 / Double(attention.headDim).squareRoot())

        func sdpa() -> MLXArray {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                              scale: scale, mask: .none)
        }
        for _ in 0..<warmup { MLX.eval(sdpa()) }
        var samples: [Double] = []
        samples.reserveCapacity(repeats)
        for _ in 0..<repeats {
            let start = benchNow()
            let out = sdpa()
            MLX.eval(out)
            samples.append(benchNow() - start)
        }
        return BenchStats(samples)
    }
}

// MARK: - The second vendor

/// MPSGraph bf16 `x @ w.T`, fed from `MTLBuffer`s the caller fills with the same bit
/// patterns MLX was given.
///
/// `w` is supplied in the checkpoint's `[n, k]` layout and transposed *inside the graph*,
/// so both vendors are asked the same question: multiply by the transpose of a `[n, k]`
/// operand. Handing MPSGraph a pre-transposed `[k, n]` buffer would measure a different
/// kernel and would be the same error as measuring a contiguous `[k, n]` operand in MLX.
final class BenchMPSGraphGEMM {

    enum Failure: Error, CustomStringConvertible {
        case noCommandQueue
        case allocation(String)
        var description: String {
            switch self {
            case .noCommandQueue: return "could not create a Metal command queue"
            case let .allocation(what): return "could not allocate \(what)"
            }
        }
    }

    private let queue: MTLCommandQueue
    private let graph = MPSGraph()
    private let xPlaceholder: MPSGraphTensor
    private let wPlaceholder: MPSGraphTensor
    private let product: MPSGraphTensor
    private let xBuffer: MTLBuffer
    private let wBuffer: MTLBuffer
    private let outBuffer: MTLBuffer
    private let feeds: [MPSGraphTensor: MPSGraphTensorData]
    private let results: [MPSGraphTensor: MPSGraphTensorData]
    private let elements: Int

    init(device: MTLDevice, m: Int, n: Int, k: Int) throws {
        guard let queue = device.makeCommandQueue() else { throw Failure.noCommandQueue }
        self.queue = queue
        self.elements = m * n

        let xShape = [NSNumber(value: m), NSNumber(value: k)]
        let wShape = [NSNumber(value: n), NSNumber(value: k)]
        let yShape = [NSNumber(value: m), NSNumber(value: n)]

        xPlaceholder = graph.placeholder(shape: xShape, dataType: .bFloat16, name: "x")
        wPlaceholder = graph.placeholder(shape: wShape, dataType: .bFloat16, name: "w")
        let transposed = graph.transposeTensor(wPlaceholder, dimension: 0, withDimension: 1,
                                               name: "w_transposed")
        product = graph.matrixMultiplication(primary: xPlaceholder, secondary: transposed,
                                             name: "y")

        guard let xBuffer = device.makeBuffer(length: m * k * 2, options: .storageModeShared)
        else { throw Failure.allocation("the x buffer") }
        guard let wBuffer = device.makeBuffer(length: n * k * 2, options: .storageModeShared)
        else { throw Failure.allocation("the w buffer") }
        guard let outBuffer = device.makeBuffer(length: m * n * 2, options: .storageModeShared)
        else { throw Failure.allocation("the output buffer") }
        self.xBuffer = xBuffer
        self.wBuffer = wBuffer
        self.outBuffer = outBuffer

        feeds = [
            xPlaceholder: MPSGraphTensorData(xBuffer, shape: xShape, dataType: .bFloat16),
            wPlaceholder: MPSGraphTensorData(wBuffer, shape: wShape, dataType: .bFloat16),
        ]
        // A pre-allocated results dictionary, so a timed repeat does not include the
        // allocation of a fresh output tensor and the two vendors are timed over the same
        // kind of work.
        results = [product: MPSGraphTensorData(outBuffer, shape: yShape, dataType: .bFloat16)]
    }

    func load(xBits: [UInt16], wBits: [UInt16]) {
        xBuffer.contents().withMemoryRebound(to: UInt16.self, capacity: xBits.count) {
            $0.update(from: xBits, count: xBits.count)
        }
        wBuffer.contents().withMemoryRebound(to: UInt16.self, capacity: wBits.count) {
            $0.update(from: wBits, count: wBits.count)
        }
    }

    /// Synchronous: submits and waits. That is the point — it measures completed work, the
    /// same property `MLX.eval` inside the MLX loop buys.
    func run() {
        graph.run(with: queue, feeds: feeds, targetOperations: nil, resultsDictionary: results)
    }

    func outputBits() -> [UInt16] {
        var out = [UInt16](repeating: 0, count: elements)
        outBuffer.contents().withMemoryRebound(to: UInt16.self, capacity: elements) { pointer in
            for index in 0..<elements { out[index] = pointer[index] }
        }
        return out
    }
}

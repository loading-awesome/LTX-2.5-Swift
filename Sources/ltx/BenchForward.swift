// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTX25
import LTXCatalog
import LTXFoundation
import LTXModules
import LTXPipeline
import LTXRecipes
import MLX

/// `ltx bench forward` — where the wall clock of a render actually goes.
///
/// ## Why this does not call `Renderer.render`
///
/// `Renderer.render` fuses the DiT load, the sampling loop and all three decodes into one
/// call and hands its observer only `(step, total)`. A profile taken around it can say
/// "the render took N seconds" and nothing else, which is the exact failure this command
/// exists to fix: measured that way the model load belongs to no phase and a minute of
/// every render goes unattributed.
///
/// So the single-stage route here re-composes the same public seams in the same order —
/// `TextConditioningPipeline`, `MLX.loadArrays`, `Pipeline.DiTDenoiser`, `Pipeline.render`,
/// `VideoVAEDecoder`, `AudioVAEDecoder`, `AudioVocoder`, `RenderOutput` — with a timer and
/// an `eval` between each. It is a transcription of `Renderer.render`, not a variant of
/// it, and the one thing it deliberately does *not* offer is conditioning: `bench forward`
/// is text-to-audio-video only, so that the mirror stays short enough to check by eye.
///
/// ## Every phase boundary sits after an `eval`
///
/// Under lazy MLX a phase that returns an unevaluated array hands its whole cost to
/// whoever forces it, and the profile then reports a fast sampler and a slow VAE. Each
/// compute phase below forces its own result before the clock stops. The sampler is the
/// one place this is free: `Sampler.run` already calls `MLX.eval` on both latents *before*
/// invoking the observer, so the per-step timestamps are completed work already.
///
/// The **checkpoint constructors are the exception, and the record says so.**
/// `MLX.loadArrays` is lazy over a mmap and `VideoVAEDecoder`, `AudioVAEDecoder`,
/// `AudioVocoder` and `TextConditioningPipeline` all hold their weights privately, so
/// there is nothing this file can force. Those phases carry
/// `boundaryAfterExplicitEval: false` and measure the header read, the config parse and
/// the structural checks only — their tensor payload is charged to the phase that first
/// touches it. `model_load` is not among them: `MLX.loadArrays` is called here, so every
/// weight can be and is evaluated, which is the whole point of giving the DiT load a phase
/// of its own.
struct BenchForward: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "forward",
        abstract: "Per-phase wall clock and peak memory for a render.",
        discussion: """
            Runs a real render. GPU work must be serialised — nothing else may be building \
            or rendering while this runs, or the numbers are wrong rather than slow.
            """)

    // MARK: Prompt and geometry — render's flags

    @Option(help: "The prompt.")
    var prompt: String = "a slow pan across a quiet room, dust in the light"

    @Option(help: "The negative prompt. Empty means the pipeline's 223-token default.")
    var negativePrompt: String = ""

    @Option(help: "Pixel frames. Must be on the 8k+1 lattice.")
    var frames: Int = 97

    @Option(help: "Width in pixels. Overrides --megapixels. Multiple of 64.")
    var width: Int?

    @Option(help: "Height in pixels. Overrides --megapixels. Multiple of 64.")
    var height: Int?

    @Option(help: "Megapixels (e.g. 0.2, 0.7, 1.0, 2.0).")
    var megapixels: Double?

    @Option(help: "Aspect ratio (16:9, 9:16, 1:1, 21:9, 4:3, 3:4).")
    var aspectRatio: String = "16:9"

    @Option(help: "Frame rate.")
    var fps: Double = 24

    @Option(help: "Seed. This port's own draw; never comparable to a reference render.")
    var seed: UInt64 = 0

    // MARK: Sampler and guidance

    @Option(help: ArgumentHelp("Sampler steps. Defaults to 30. REFUSED with --two-stage, "
                + "whose two sigma schedules are literals (8 then 3)."))
    var steps: Int?

    @Option(help: "Video CFG scale. 1.0 is neutral. Default 3.0.")
    var videoCfg: Double?

    @Option(help: "Audio CFG scale, independent of the video one. Default 7.0.")
    var audioCfg: Double?

    @Option(help: "STG scale. Neutral at 0.0, NOT 1.0. Default 1.0.")
    var stgScale: Double?

    @Option(help: "Comma-separated STG blocks. [28] by default.")
    var stgBlocks: String?

    @Option(help: "Modality guidance. Neutral at 1.0. Default 3.0.")
    var modalityScale: Double?

    @Option(help: "Guidance rescale. Neutral at 0.0; costs no pass. Default 0.7.")
    var rescaleScale: Double?

    @Flag(help: ArgumentHelp("Profile the two-stage distilled pipeline instead. NOTE the "
                + "phase breakdown is coarser on this route — see the record's caveats."))
    var twoStage: Bool = false

    @Option(help: ArgumentHelp("Where attention accumulates: 'fused' (MLX steel SDPA, the "
                + "render default) or 'explicit' (fp32 softmax, the gate default). This is "
                + "a memory *class* choice, not a constant factor: explicit materialises a "
                + "[1, 32, T, T] fp32 score matrix, so its peak is quadratic in the token "
                + "count and it will SIGKILL at long or large shapes. It is a bench option "
                + "and deliberately not a render one — see Renderer.Spec.attentionPath."))
    var attention: String = "fused"

    // MARK: Protocol

    @Option(help: ArgumentHelp("Repeat the SAMPLING phase this many times. The model load "
                + "and the decodes run once. Repeats are what the spread statistic is "
                + "computed over; at 1 the spread is null, not zero."))
    var repeats: Int = 1

    @Flag(help: "Skip the mux phase (and the WAV write) entirely.")
    var skipMux: Bool = false

    @Option(help: ArgumentHelp("Where the mux writes. Defaults to a temporary file that is "
                + "deleted afterwards — the mp4 is a by-product here, not the point."))
    var videoOut: String?

    @Flag(help: "Keep whatever the mux wrote.")
    var keepOutput: Bool = false

    @Option(help: "Where to write the machine-readable record.")
    var out: String?

    @Option(help: "Optional flat CSV sidecar, one row per phase.")
    var csv: String?

    // MARK: Checkpoints

    @Option(help: "The DiT checkpoint. Defaults to dev, or distilled under --two-stage.")
    var checkpoint: String?

    @Option(help: "Config file. Defaults to ~/.config/ltx/config.json.")
    var config: String?

    @Option(help: "OVERRIDE the x2 latent spatial upsampler (--two-stage only).")
    var upsampler: String?

    @Option(help: "OVERRIDE the text encoder.")
    var textEncoder: String?

    @Option(help: "OVERRIDE the video VAE.")
    var videoVae: String?

    @Option(help: "OVERRIDE the audio VAE.")
    var audioVae: String?

    // MARK: - What comes out

    /// The decoded streams, held locally rather than in a `Renderer.Result`.
    ///
    /// `Renderer.Result`'s memberwise initialiser is internal to `LTXPipeline`, and that is
    /// the right access level for it — it is the shape a *render* returns, and this is not
    /// one. The mux phase takes the three tensors it needs and nothing here pretends to be
    /// a render result.
    private struct Decoded {
        var videoRGB: MLXArray
        var audioMel: MLXArray
        var audioWaveform: MLXArray
        var audioSampleRate: Int
    }

    /// Both text branches, for the single-stage route.
    ///
    /// `Renderer.TextBranches` would be the natural type and its initialiser is internal
    /// to `LTXPipeline`. Rather than reach around that, this route carries its own pair and
    /// hands the two `Pipeline.Conditioning` values straight to `Pipeline.DiTDenoiser`,
    /// which is what `Renderer.render` does with them anyway.
    private struct TextBundle {
        var positive: Pipeline.Conditioning
        var negative: Pipeline.Conditioning
        var validTokenCounts: [Int]
    }

    /// `Renderer.require`'s job, which is internal to `LTXPipeline`.
    private static func require(_ array: MLXArray?, _ name: String) throws -> MLXArray {
        guard let array else {
            throw ValidationError("the text encoder produced no \(name)")
        }
        return array
    }

    // MARK: - Phases

    /// One timed phase: what it was, how long it took, what it peaked at, and — the field
    /// that keeps the profile honest — whether its result was forced before the clock
    /// stopped.
    private struct Phase {
        let name: String
        let seconds: Double
        let allocatorPeakBytes: Int
        let evaluated: Bool
        let note: String?

        var json: BenchJSON {
            .object([
                "name": .string(name),
                "seconds": .double(seconds),
                "allocatorPeakBytes": .int(allocatorPeakBytes),
                "boundaryAfterExplicitEval": .bool(evaluated),
                "note": .opt(note),
            ])
        }
    }

    /// Collects phases and keeps the running allocator high-water mark across resets.
    private final class PhaseLog {
        private(set) var phases: [Phase] = []
        private(set) var overallPeak = 0

        /// Time `body`, which **must** force its own result before returning.
        ///
        /// `evaluated: false` is the escape hatch for the phases where that is not
        /// possible — a checkpoint constructor holds its weights privately and
        /// `MLX.loadArrays` is lazy over a mmap, so there is nothing this file can force.
        /// Passing `false` does not make such a phase respectable; it makes the record say
        /// so, which is the difference between a caveat and a lie.
        func measure<T>(_ name: String, note: String? = nil, evaluated: Bool = true,
                        _ body: () throws -> T) rethrows -> T {
            FileHandle.standardError.write(Data("phase \(name)...\n".utf8))
            BenchPeakMemory.reset()
            let start = benchNow()
            let value = try body()
            let seconds = benchNow() - start
            let peak = BenchPeakMemory.allocatorPeakBytes
            overallPeak = max(overallPeak, peak)
            phases.append(Phase(name: name, seconds: seconds, allocatorPeakBytes: peak,
                                evaluated: evaluated, note: note))
            FileHandle.standardError.write(Data(
                String(format: "  %@ %.3f s  peak %.2f GB\n",
                       name, seconds, Double(peak) / 1e9).utf8))
            return value
        }

        var totalSeconds: Double { phases.reduce(0) { $0 + $1.seconds } }
        var json: BenchJSON { .array(phases.map(\.json)) }
    }

    // MARK: - Run

    func attentionPath() throws -> DiTAttention.AttentionPath {
        switch attention {
        case "fused": return .fused
        case "explicit": return .explicit
        default:
            throw ValidationError(
                "--attention must be 'fused' or 'explicit', got '\(attention)'")
        }
    }

    func run() async throws {
        guard repeats >= 1 else { throw ValidationError("--repeats must be >= 1") }
        if twoStage, steps != nil {
            throw ValidationError("--steps is refused with --two-stage: both stages' sigma "
                + "schedules are fixed tables, 8 steps then 3")
        }
        guard let aspect = LTX25.Aspect(rawValue: aspectRatio) else {
            throw ValidationError("--aspect-ratio must be one of "
                + LTX25.Aspect.allCases.map(\.rawValue).joined(separator: ", "))
        }
        _ = try attentionPath()      // reject a bad --attention before 42 GB is loaded
        var parsedSTGBlocks: [Int]?
        if let stgBlocks {
            let parts = stgBlocks.split(separator: ",")
            let parsed = parts.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard parsed.count == parts.count else {
                throw ValidationError("--stg-blocks must be comma-separated integers")
            }
            parsedSTGBlocks = parsed
        }

        let request = LTX25.Request(
            prompt: prompt, videoOutput: URL(fileURLWithPath: "/dev/null"),
            seconds: Double(frames) / fps, frames: frames, frameRate: fps,
            megapixels: megapixels, aspectRatio: aspect, width: width, height: height)
        let shape = try request.resolvedShape()
        let geometry = DiTForward.Geometry(frames: try request.resolvedFrames(),
                                           height: shape.height, width: shape.width,
                                           frameRate: fps)
        try geometry.validate()

        var paths = try Paths(configPath: config)
        let ditPath = try paths.url(twoStage ? .ditDistilled : .ditDev,
                                    override: checkpoint).path
        let upsamplerURL = try paths.url(.upsampler, override: upsampler)
        let resolvedPaths: [LTXConfiguration.Role: URL] = [
            .textEncoder: try paths.url(.textEncoder, override: textEncoder),
            .videoVAE: try paths.url(.videoVAE, override: videoVae),
            .audioVAE: try paths.url(.audioVAE, override: audioVae),
            .upsampler: upsamplerURL,
        ]
        let checkpoints = Renderer.Checkpoints(
            textEncoder: resolvedPaths[.textEncoder]!,
            dit: URL(fileURLWithPath: ditPath),
            videoVAE: resolvedPaths[.videoVAE]!,
            audioVAE: resolvedPaths[.audioVAE]!)
        paths.announce()
        try Preflight.check(paths: paths, slots: [
            CheckpointInventory.Slot(role: "dit", url: checkpoints.dit,
                                     expected: .transformer,
                                     nameMustContain: twoStage ? "distilled" : "dev"),
            CheckpointInventory.Slot(role: "text encoder", url: checkpoints.textEncoder,
                                     expected: .textEncoder),
            CheckpointInventory.Slot(role: "video vae", url: checkpoints.videoVAE,
                                     expected: .videoVAECausal),
            CheckpointInventory.Slot(role: "audio vae", url: checkpoints.audioVAE,
                                     expected: .audioVAE),
        ])

        let muxURL = URL(fileURLWithPath: videoOut
            ?? NSTemporaryDirectory() + "ltx-bench-\(ProcessInfo.processInfo.processIdentifier).mp4")

        var caveats: [String] = [
            "bench forward is text-to-audio-video only. The conditioning seams (--image, "
                + "--video, --audio, --audio-mel) are not exercised and are not profiled.",
            "This record carries no check that the render is correct. docs/PERFORMANCE.md: "
                + "a bench record without one is not admissible as evidence about a build.",
        ]

        let log = PhaseLog()
        let overallStart = benchNow()

        var measurements: [String: BenchJSON]
        var stepSecondsPerRepeat: [[Double]] = []
        var validTokenCounts: [Int] = []

        if twoStage {
            // `DistilledRenderer.render` requires a `Renderer.TextBranches`, whose
            // initialiser is internal to `LTXPipeline` — correctly so, it is the shape a
            // renderer's text phase returns. The only public way to obtain one on this
            // route is `DistilledRenderer.conditioning`, which loads the encoder AND
            // encodes in one call, so the two cannot be separated here.
            caveats.append(
                "TWO-STAGE: the encoder load and the text forward are fused into one "
                    + "`text_encode` phase. DistilledRenderer.conditioning does both and "
                    + "Renderer.TextBranches cannot be constructed from outside "
                    + "LTXPipeline, so the split the single-stage route reports is not "
                    + "available here without a public initialiser or a second entry point.")
            let spec = DistilledRenderer.Spec(prompt: prompt, geometry: geometry, seed: seed,
                                              attentionPath: try attentionPath())
            let distilled = DistilledRenderer.Checkpoints(
                base: checkpoints, upsampler: upsamplerURL)
            let text = try log.measure("text_encode") { () -> Renderer.TextBranches in
                let branches = try DistilledRenderer.conditioning(spec: spec,
                                                                  checkpoints: distilled)
                MLX.eval(branches.positive.video, branches.positive.audio)
                return branches
            }
            validTokenCounts = text.validTokenCounts
            MLX.Memory.clearCache()

            let (result, twoStageJSON) = try runTwoStage(
                spec: spec, distilled: distilled, text: text, log: log, caveats: &caveats)
            measurements = ["twoStage": twoStageJSON, "sampling": .null]
            try mux(result, geometry: geometry, to: muxURL, log: log)
        } else {
            // ---- Text. The 26 GB encoder is loaded, used, dropped before the DiT. ----
            //
            // Split into two phases on purpose: the load and the two forwards are both
            // large and a fused "text" phase hides which one moved.
            let text: TextBundle = try {
                let pipeline = try log.measure(
                    "text_encoder_load",
                    note: "MLX.loadArrays is LAZY over a mmap and this type holds its weights privately, so there is nothing here to force. This phase is the header read, the config parse and the structural checks; the tensor payload is charged to the phase that first touches it. That is why boundaryAfterExplicitEval is false.",
                    evaluated: false
                ) {
                    try TextConditioningPipeline(textEncoder: checkpoints.textEncoder,
                                                 dit: checkpoints.dit)
                }
                return try log.measure("text_encode") { () -> TextBundle in
                    let branches = try pipeline(prompt: prompt, negativePrompt: negativePrompt)
                    func conditioning(_ e: TextConditioningPipeline.Encoding) throws
                        -> Pipeline.Conditioning {
                        Pipeline.Conditioning(
                            video: try Self.require(e.features[.video], "features.video"),
                            audio: try Self.require(e.features[.audio], "features.audio"))
                    }
                    // Branch order is `callNNN` order: 0 positive, 1 negative — the same
                    // convention `Renderer.conditioning` relies on.
                    let positive = try conditioning(branches[0])
                    let negative = try conditioning(branches[1])
                    MLX.eval(positive.video, positive.audio, negative.video, negative.audio)
                    return TextBundle(positive: positive, negative: negative,
                                      validTokenCounts: branches.map(\.validTokenCount))
                }
            }()
            validTokenCounts = text.validTokenCounts
            // The encoder is out of scope now; hand its buffers back so the DiT load's
            // peak is the DiT's and not the encoder's residue.
            MLX.Memory.clearCache()

            let sampler = Sampler(
                video: guidance(default: RecipeGuidance.productionVideo, blocks: parsedSTGBlocks,
                                cfg: videoCfg),
                audio: guidance(default: RecipeGuidance.productionAudio, blocks: parsedSTGBlocks,
                                cfg: audioCfg))
            let stepCount = steps ?? 30
            let (result, samplingJSON, perRepeat) = try runSingleStage(
                geometry: geometry, checkpoints: checkpoints, text: text,
                sampler: sampler, stepCount: stepCount, log: log)
            stepSecondsPerRepeat = perRepeat
            measurements = ["sampling": samplingJSON]
            try mux(result, geometry: geometry, to: muxURL, log: log)
        }

        if !keepOutput, videoOut == nil {
            try? FileManager.default.removeItem(at: muxURL)
        }

        let wallClock = benchNow() - overallStart
        measurements["phases"] = log.json
        measurements["wallClockSeconds"] = .double(wallClock)
        measurements["phaseSumSeconds"] = .double(log.totalSeconds)
        measurements["unattributedSeconds"] = .double(wallClock - log.totalSeconds)

        // The repeat spread, over the comparison statistic and nothing else.
        let meanPerRepeat = stepSecondsPerRepeat.map { $0.reduce(0, +) / Double(max($0.count, 1)) }
        let repeatStats = BenchStats(meanPerRepeat)
        measurements["repeats"] = .object([
            "count": .int(stepSecondsPerRepeat.count),
            "meanFullStepSecondsPerRepeat": .doubles(meanPerRepeat),
            "spreadOfRangeOverMedian": .opt(stepSecondsPerRepeat.count >= 2
                                            ? repeatStats.spread : nil),
            "note": .string("Spread is the full range (max - min) / median, not a standard "
                + "deviation: with two or three repeats a stddev is a statement about the "
                + "sample size. The measured floor on this statistic is 0.9-1.4%, which "
                + "is why any claimed gain below about 1.5% is not a gain."),
        ])

        let record = BenchRecord(
            kind: "forward",
            invocation: CommandLine.arguments,
            parameters: parameters(geometry: geometry, ditPath: ditPath,
                                   resolved: resolvedPaths,
                                   stgBlocks: parsedSTGBlocks,
                                   validTokenCounts: validTokenCounts),
            measurements: .object(measurements),
            caveats: caveats,
            peakMemoryAllocatorBytes: log.overallPeak)

        try record.write(to: out)
        try writeCSV(log: log)
    }

    // MARK: - Single stage

    /// The `Renderer.render` sequence, phase by phase.
    private func runSingleStage(geometry: DiTForward.Geometry,
                                checkpoints: Renderer.Checkpoints,
                                text: TextBundle,
                                sampler: Sampler, stepCount: Int,
                                log: PhaseLog) throws
        -> (Decoded, BenchJSON, [[Double]]) {

        // ---- model_load. Its own phase, by name. -----------------------------------
        //
        // `MLX.loadArrays` is lazy over a mmap, so without the eval below the 42 GB read
        // would be charged to whichever sampler step first touched each tensor — which is
        // precisely the failure this command exists to avoid: a minute of every render
        // belonging to no phase. The eval moves that cost here, where it can be seen.
        // It also means the allocator peak for this phase is the whole checkpoint rather
        // than nothing, which is the honest figure for "the model is resident".
        let denoiser = try log.measure(
            "model_load",
            note: "DiT header + topology + cross-modal wiring check + MLX.loadArrays, then "
                + "an explicit eval of every weight. Production leaves the load lazy and "
                + "pays for it inside step 1; forcing it here is what makes the phase "
                + "attributable at all."
        ) { () -> Pipeline.DiTDenoiser in
            let header = try SafetensorsHeader.read(from: checkpoints.dit)
            let topology = try TransformerTopology.read(header)
            try topology.verifyCrossModalWiring(header)
            let weights = try MLX.loadArrays(url: checkpoints.dit)
            MLX.eval(Array(weights.values))
            return Pipeline.DiTDenoiser(
                forward: DiTForward(weights: weights, topology: topology,
                                    attentionPath: try attentionPath()),
                head: DiTOutputHead(weights: weights, topology: topology),
                geometry: geometry,
                positive: text.positive, negative: text.negative,
                videoSTGBlocks: sampler.video.stgBlocks,
                audioSTGBlocks: sampler.audio.stgBlocks)
        }

        // ---- sampling, once per repeat ---------------------------------------------
        var stepSecondsPerRepeat: [[Double]] = []
        var rendered: Pipeline.Rendered?
        for repeatIndex in 0..<repeats {
            var stepSeconds: [Double] = []
            var lastMark = benchNow()
            let name = repeats == 1 ? "sampling" : "sampling_repeat_\(repeatIndex)"
            let output = try log.measure(name) { () -> Pipeline.Rendered in
                lastMark = benchNow()
                let noise = try RenderNoise(seed: seed, geometry: geometry)
                let out = try Pipeline.render(
                    sampler: sampler, steps: stepCount,
                    initial: noise.streamsForRenderOnly(),
                    denoiser: denoiser, geometry: geometry,
                    observer: { _, streams in
                        // `Sampler.run` already evaluated both latents before calling
                        // this, so the timestamp is completed work. The eval is kept as
                        // the invariant rather than as a cost: it is a no-op on an already
                        // forced array, and if the sampler ever stops forcing, this keeps
                        // the per-step numbers meaning what they say.
                        MLX.eval(streams.video, streams.audio)
                        let now = benchNow()
                        stepSeconds.append(now - lastMark)
                        lastMark = now
                    })
                MLX.eval(out.videoLatent, out.audioLatent)
                return out
            }
            stepSecondsPerRepeat.append(stepSeconds)
            rendered = output
        }
        guard let rendered else { throw ValidationError("no sampling repeat ran") }

        // ---- the decodes ------------------------------------------------------------
        let videoDecoder = try log.measure("video_vae_load", note: "MLX.loadArrays is LAZY over a mmap and this type holds its weights privately, so there is nothing here to force. This phase is the header read, the config parse and the structural checks; the tensor payload is charged to the phase that first touches it. That is why boundaryAfterExplicitEval is false.", evaluated: false) {
            try VideoVAEDecoder(checkpoint: checkpoints.videoVAE)
        }
        let videoRGB = try log.measure(
            "video_decode",
            note: "TiledDecode on the recorded tile-size policy, then rgb(). This is "
                + "the tiled path; the untiled path is different code and is not profiled "
                + "here (docs/PERFORMANCE.md Step 3)."
        ) { () -> MLXArray in
            let factors = videoDecoder.config.scaleFactors
            let layout = try Renderer.videoTileLayout(
                geometry: geometry,
                scale: TiledDecode.Scale(time: factors.time, height: factors.height,
                                         width: factors.width))
            let raw = try TiledDecode.decode(rendered.videoLatent, layout: layout) {
                try videoDecoder.decode($0)
            }
            let rgb = videoDecoder.rgb(raw)
            MLX.eval(rgb)
            return rgb
        }

        let audioDecoder = try log.measure("audio_vae_load", note: "MLX.loadArrays is LAZY over a mmap and this type holds its weights privately, so there is nothing here to force. This phase is the header read, the config parse and the structural checks; the tensor payload is charged to the phase that first touches it. That is why boundaryAfterExplicitEval is false.", evaluated: false) {
            try AudioVAEDecoder(checkpoint: checkpoints.audioVAE)
        }
        let audioMel = try log.measure("audio_decode") { () -> MLXArray in
            let mel = try audioDecoder.decode(rendered.audioLatent)
            MLX.eval(mel)
            return mel
        }

        let (vocoder, vocoderConfig) = try log.measure(
            "vocoder_load", note: "MLX.loadArrays is LAZY over a mmap and this type holds its weights privately, so there is nothing here to force. This phase is the header read, the config parse and the structural checks; the tensor payload is charged to the phase that first touches it. That is why boundaryAfterExplicitEval is false.", evaluated: false
        ) {
            () -> (AudioVocoder, AudioVocoder.Configuration) in
            let header = try SafetensorsHeader.read(from: checkpoints.audioVAE)
            return (try AudioVocoder(checkpoint: checkpoints.audioVAE),
                    try AudioVocoder.configuration(from: header))
        }
        let waveform = try log.measure("vocoder") { () -> MLXArray in
            let audio = try vocoder.waveform(fromMel: audioMel)
            MLX.eval(audio)
            return audio
        }

        let result = Decoded(videoRGB: videoRGB, audioMel: audioMel, audioWaveform: waveform,
                             audioSampleRate: vocoderConfig.outputSamplingRate)

        // ---- the statistics ---------------------------------------------------------
        let allSteps = stepSecondsPerRepeat.last ?? []
        let passesPerStep = (0..<stepCount).map { sampler.plan(step: $0).passes.count }
        let totalPasses = sampler.passCount(steps: stepCount)
        let meanStep = allSteps.isEmpty ? Double.nan : allSteps.reduce(0, +) / Double(allSteps.count)

        let json = BenchJSON.object([
            "steps": .int(stepCount),
            "stepSeconds": .doubles(allSteps),
            "stepSecondsPerRepeat": .array(stepSecondsPerRepeat.map(BenchJSON.doubles)),
            // The comparison statistic. Never the median.
            "meanFullStepSeconds": .double(meanStep),
            "meanFullStepSecondsNote": .string(
                "THE comparison statistic. Step times form multiple populations — LTX runs "
                    + "1 to 4 forwards per step depending on the guidance schedule and the "
                    + "skip pattern — so a median reports the modal population and ignores "
                    + "the rest. A median has made a cache that halved sampling time "
                    + "report 1.00x."),
            "medianFullStepSeconds": .double(benchMedian(allSteps)),
            "medianFullStepSecondsNote": .string(
                "Reported separately and never used for the headline ratio. A kernel win "
                    + "moves the per-step cost and shows up here; a schedule change moves "
                    + "the pass COUNT and does not. Only the first composes with another "
                    + "kernel win."),
            "totalSamplingSeconds": .double(allSteps.reduce(0, +)),
            "passesPerStep": .ints(passesPerStep),
            "totalTransformerPasses": .int(totalPasses),
            "meanSecondsPerTransformerPass": .double(
                totalPasses > 0 ? allSteps.reduce(0, +) / Double(totalPasses) : .nan),
            "guidanceSchedule": .array((0..<stepCount).map { step in
                let plan = sampler.plan(step: step)
                return .object([
                    "step": .int(step),
                    "passes": .strings(plan.passes.map(\.description)),
                    "videoEnabled": .bool(plan.videoEnabled),
                    "audioEnabled": .bool(plan.audioEnabled),
                ])
            }),
        ])
        return (result, json, stepSecondsPerRepeat)
    }

    // MARK: - Two stage

    /// `DistilledRenderer.render`, which is one call and cannot be taken apart from here.
    ///
    /// What this route can and cannot say is recorded in the caveats rather than papered
    /// over. The observer is `(stage, step, total)` — it carries no tensors and there is no
    /// callback at all around the x2 upsample — so the derived intervals below are the most
    /// this harness can extract without adding a hook to `DistilledRenderer`.
    private func runTwoStage(spec: DistilledRenderer.Spec,
                             distilled: DistilledRenderer.Checkpoints,
                             text: Renderer.TextBranches,
                             log: PhaseLog,
                             caveats: inout [String]) throws -> (Decoded, BenchJSON) {
        try spec.validate()
        try distilled.verifyPresent()

        caveats.append(
            "TWO-STAGE: model_load, video_decode, audio_decode and vocoder are NOT "
                + "separable on this route. DistilledRenderer.render performs all of them "
                + "inside one call and its observer is (stage, step, total) with no "
                + "tensors. The hook that would fix this is a phase observer on "
                + "DistilledRenderer.render — one callback after the DiT load, one on "
                + "either side of the upsample, and one per decode. It was NOT added: "
                + "Renderer.swift and DistilledRenderer.swift are out of bounds for this "
                + "harness.")
        caveats.append(
            "TWO-STAGE: the x2 latent spatial upsample has no observer and is not timed. "
                + "`stageBoundaryGapSeconds` below is the interval between the last stage-1 "
                + "step callback and the first stage-2 one, which contains the unpatchify, "
                + "the upsampler load and forward, the re-patchify and the stage-2 seeding "
                + "TOGETHER. It is an upper bound on the upsample, not the upsample.")

        var stage1Steps: [Double] = []
        var stage2Steps: [Double] = []
        var lastMark = benchNow()
        var phaseStart = benchNow()
        var firstCallbackAt: Double?
        var stage1EndedAt: Double?
        var stage2StartedAt: Double?

        let result = try log.measure(
            "two_stage_render",
            note: "DiT load + stage 1 + x2 upsample + stage 2 + both VAE decodes + the "
                + "vocoder, all inside DistilledRenderer.render. Not separable from here."
        ) { () -> Decoded in
            phaseStart = benchNow()
            lastMark = phaseStart
            let output = try DistilledRenderer.render(
                spec: spec, checkpoints: distilled, text: text,
                observer: { stage, _, _ in
                    let now = benchNow()
                    if firstCallbackAt == nil { firstCallbackAt = now }
                    if stage == 1 {
                        stage1Steps.append(now - lastMark)
                        stage1EndedAt = now
                    } else {
                        if stage2StartedAt == nil { stage2StartedAt = now }
                        stage2Steps.append(now - lastMark)
                    }
                    lastMark = now
                })
            MLX.eval(output.videoRGB, output.audioMel, output.audioWaveform)
            return Decoded(videoRGB: output.videoRGB, audioMel: output.audioMel,
                           audioWaveform: output.audioWaveform,
                           audioSampleRate: output.audioSampleRate)
        }

        // Stage 2's first interval spans the boundary; report it separately rather than
        // letting it contaminate the stage-2 step population.
        let boundaryGap: Double? = {
            guard let ended = stage1EndedAt, let started = stage2StartedAt else { return nil }
            return started - ended
        }()
        let stage2Body = Array(stage2Steps.dropFirst())
        let allSteps = stage1Steps + stage2Body
        let meanStep = allSteps.isEmpty ? Double.nan : allSteps.reduce(0, +) / Double(allSteps.count)

        let json = BenchJSON.object([
            "stage1Steps": .int(spec.stage1Steps),
            "stage2Steps": .int(spec.stage2Steps),
            "stage1Geometry": .string("\(spec.stage1Geometry.width)x\(spec.stage1Geometry.height)"),
            "stage2Geometry": .string("\(spec.geometry.width)x\(spec.geometry.height)"),
            "stage1SigmaSchedule": .doubles(spec.stage1Sigmas.map(Double.init)),
            "stage2SigmaSchedule": .doubles(spec.stage2Sigmas.map(Double.init)),
            "stage1StepSeconds": .doubles(stage1Steps),
            "stage2StepSeconds": .doubles(stage2Body),
            "preFirstStepSeconds": .opt(firstCallbackAt.map { $0 - phaseStart }),
            "preFirstStepNote": .string(
                "DiT header + topology check + 42 GB MLX.loadArrays + noise + the whole of "
                    + "stage-1 step 0. This is where the unattributed model load lives on "
                    + "this route; it is a bound, not the load."),
            "stageBoundaryGapSeconds": .opt(boundaryGap),
            "meanFullStepSeconds": .double(meanStep),
            "meanFullStepSecondsNote": .string(
                "Mean over both stages' steps with the boundary interval excluded. The two "
                    + "stages run at different resolutions, so this mean is over two "
                    + "populations by construction and is only comparable between runs of "
                    + "the SAME geometry."),
            "medianFullStepSeconds": .double(benchMedian(allSteps)),
            "totalTransformerPasses": .int(spec.stage1Steps + spec.stage2Steps),
            "passesPerStep": .int(1),
        ])
        return (result, json)
    }

    // MARK: - Mux

    private func mux(_ result: Decoded, geometry: DiTForward.Geometry,
                     to url: URL, log: PhaseLog) throws {
        guard !skipMux else { return }
        _ = try log.measure(
            "mux",
            note: "RenderOutput.writeWAV then RenderOutput.write (H.264 + AAC via "
                + "AVFoundation). File I/O and a video encoder, not MLX — the allocator "
                + "peak for this phase is not the memory it uses."
        ) { () -> Bool in
            let wav = url.deletingPathExtension().appendingPathExtension("wav")
            try RenderOutput.writeWAV(audio: result.audioWaveform, to: wav,
                                      sampleRate: result.audioSampleRate)
            try RenderOutput.write(
                rgb: result.videoRGB, audio: result.audioWaveform, to: url,
                spec: RenderOutput.VideoSpec(fps: geometry.frameRate),
                audioSpec: RenderOutput.AudioSpec(sampleRate: result.audioSampleRate))
            if !keepOutput, videoOut == nil { try? FileManager.default.removeItem(at: wav) }
            return true
        }
    }

    // MARK: - Parameters and CSV

    private func guidance(default base: RecipeGuidance, blocks: [Int]?,
                          cfg: Double?) -> GuidanceParams {
        GuidanceParams(
            cfgScale: cfg ?? base.cfgScale,
            stgScale: stgScale ?? base.stgScale,
            stgBlocks: blocks ?? base.stgBlocks,
            rescaleScale: rescaleScale ?? base.rescaleScale,
            modalityScale: modalityScale ?? base.modalityScale,
            skipStep: 0)
    }

    private func parameters(geometry: DiTForward.Geometry, ditPath: String,
                            resolved: [LTXConfiguration.Role: URL],
                            stgBlocks blocks: [Int]?,
                            validTokenCounts: [Int]) -> BenchJSON {
        // Hoisted out of the literal below. Inlining optionals into an expression this
        // large defeats the type checker outright ("unable to type-check in reasonable
        // time"), and the record must name the files that were actually opened.
        let checkpointsJSON = BenchJSON.object([
            "dit": .string(ditPath),
            "textEncoder": .string(resolved[.textEncoder]?.path ?? ""),
            "videoVAE": .string(resolved[.videoVAE]?.path ?? ""),
            "audioVAE": .string(resolved[.audioVAE]?.path ?? ""),
            "upsampler": twoStage ? .string(resolved[.upsampler]?.path ?? "") : .null,
        ])
        let video = guidance(default: RecipeGuidance.productionVideo, blocks: blocks, cfg: videoCfg)
        let audio = guidance(default: RecipeGuidance.productionAudio, blocks: blocks, cfg: audioCfg)
        func guidanceJSON(_ g: GuidanceParams) -> BenchJSON {
            .object([
                "cfgScale": .double(g.cfgScale),
                "stgScale": .double(g.stgScale),
                "stgBlocks": .ints(g.stgBlocks),
                "rescaleScale": .double(g.rescaleScale),
                "modalityScale": .double(g.modalityScale),
                "skipStep": .int(g.skipStep),
                "stgStartPercent": .double(g.stgStartPercent),
                "stgEndPercent": .double(g.stgEndPercent),
                "modalityStartPercent": .double(g.modalityStartPercent),
                "modalityEndPercent": .double(g.modalityEndPercent),
            ])
        }
        return .object([
            "route": .string(twoStage ? "two-stage distilled" : "single-stage guided"),
            "prompt": .string(prompt),
            "negativePrompt": .string(negativePrompt),
            "validTokenCounts": .ints(validTokenCounts),
            "seed": .int(Int(bitPattern: UInt(seed))),
            "steps": twoStage ? .null : .int(steps ?? 30),
            "geometry": .object([
                "frames": .int(geometry.frames),
                "width": .int(geometry.width),
                "height": .int(geometry.height),
                "frameRate": .double(geometry.frameRate),
                "latentFrames": .int(geometry.latentFrames),
                "videoTokens": .int(geometry.videoTokens),
                "audioTokens": .int(geometry.audioTokens),
            ]),
            "videoGuidance": twoStage ? .null : guidanceJSON(video),
            "audioGuidance": twoStage ? .null : guidanceJSON(audio),
            "repeats": .int(repeats),
            "attention": .object([
                "path": .string(attention),
                "note": .string(
                    "Where the softmax accumulates. 'explicit' materialises a "
                        + "[1, 32, T, T] fp32 score matrix per self-attention — quadratic "
                        + "in videoTokens above — and 'fused' streams it through MLX's "
                        + "steel SDPA. A memory-peak comparison between two records is "
                        + "meaningless unless this field matches."),
            ]),
            "checkpoints": checkpointsJSON,
            "conditioning": .string("none (text to audio-video)"),
        ])
    }

    private func writeCSV(log: PhaseLog) throws {
        guard csv != nil else { return }
        var rows = BenchCSV(header: ["phase", "seconds", "allocator_peak_bytes",
                                     "boundary_after_eval", "note"])
        for phase in log.phases {
            rows.append([phase.name, BenchCSV.cell(phase.seconds),
                         BenchCSV.cell(phase.allocatorPeakBytes),
                         phase.evaluated ? "true" : "false", phase.note ?? ""])
        }
        try rows.write(to: csv)
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

import LTXFoundation
import LTXModules
@testable import LTXPipeline

/// The two-stage distilled pipeline's **structure**, checked without a 22 B checkpoint.
///
/// The two-stage feature is arithmetic and bookkeeping wrapped around a transformer, and
/// nearly all of it is decidable with no transformer in the room. That is what this suite
/// holds:
///
/// * both sigma schedules against literals written out here, so each is compared against an
///   independent transcription rather than against itself;
/// * that `explicitSigmas` derives 8 and 3 steps, since `count - 1` is the whole
///   relationship between a schedule and a step count;
/// * that the distilled guidance costs exactly one transformer forward per step, and that
///   the combination is the conditional prediction unchanged, bit for bit;
/// * the stage-2 handoff arithmetic, against a `lerp` computed by hand;
/// * that the **audio** latent never meets the upsampler;
/// * the geometry, the conditioning seam, and the refusals either side of them.
///
/// Two tests read a real checkpoint, header only — which sampler the distilled 2.5
/// transformer selects, and that the temporal upscaler is refused on its own config. Both
/// return early when the model tree is not mounted, because the rest of the suite is worth
/// running on a machine that does not carry a gigabyte of weights.
///
/// The upsampler's own arithmetic is checked in `LatentSpatialUpsamplerTests`.
@Suite("Distilled two-stage")
struct DistilledTwoStageTests {

    private static let root = LTXConfiguration.resolved.checkpoints.root!

    // MARK: - The schedules

    /// The distilled stage-1 sigma values, spelled out independently.
    ///
    /// Written out here rather than referenced from `FlowSchedule` — a test that compares a
    /// constant to itself asserts nothing. Nine values, eight steps, ending at 0. Note the
    /// flat head: the first five are within 6.3e-03 of each other, so five of the eight
    /// steps barely move sigma and the trajectory does its work in the last three.
    static let referenceStage1: [Float] =
        [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    /// The distilled stage-2 sigma values.
    static let referenceStage2: [Float] = [0.909375, 0.725, 0.421875, 0.0]

    @Test("Stage 1's schedule is nine literal sigmas")
    func stage1SchedulesMatchTheReference() {
        #expect(FlowSchedule.distilledStage1 == Self.referenceStage1)
        #expect(FlowSchedule.distilledStage1.count == 9)
    }

    @Test("Stage 2's schedule is four literal sigmas, and is stage 1's tail")
    func stage2SchedulesMatchTheReference() {
        #expect(FlowSchedule.distilledStage2 == Self.referenceStage2)
        // Not a coincidence worth leaving unstated: stage 2 re-runs exactly the part of the
        // schedule where the trajectory moves, at full resolution. If someone "tidied" one
        // list and not the other, this is what would notice.
        #expect(Array(FlowSchedule.distilledStage1.suffix(4)) == FlowSchedule.distilledStage2)
    }

    // MARK: - Step counts

    /// A denoiser that returns zero velocity and counts what it was asked for.
    ///
    /// Zero velocity makes `Sampler.denoised` the identity on the latent, so the loop runs
    /// its full arithmetic on a tensor whose value never becomes interesting. This suite is
    /// counting passes and steps, not measuring a trajectory — and a fixture that produced
    /// plausible-looking numbers would invite exactly the numerical reading the recorded
    /// data cannot support.
    final class CountingDenoiser: Sampler.Denoiser, @unchecked Sendable {
        private(set) var passesByStep: [Int: [Sampler.Pass]] = [:]
        private(set) var sigmasSeen: [Float] = []

        func velocity(pass: Sampler.Pass, step: Int, sigma: Float,
                      latents: Sampler.Streams,
                      denoiseMask: Sampler.Streams?) throws -> Sampler.Streams {
            passesByStep[step, default: []].append(pass)
            sigmasSeen.append(sigma)
            return Sampler.Streams(
                video: MLXArray.zeros(latents.video.shape, dtype: latents.video.dtype),
                audio: MLXArray.zeros(latents.audio.shape, dtype: latents.audio.dtype))
        }
    }

    static func tinyStreams() -> Sampler.Streams {
        Sampler.Streams(video: MLXArray.zeros([1, 6, 4], dtype: .bfloat16),
                        audio: MLXArray.zeros([1, 3, 4], dtype: .bfloat16))
    }

    /// `explicitSigmas` makes `steps:` irrelevant — the list decides, `count - 1`.
    ///
    /// The deliberately wrong `steps:` argument is the assertion. A loop that honoured it
    /// would run 99 steps off the end of a nine-element schedule, or three of them and stop.
    @Test("explicitSigmas gives 8 stage-1 steps and 3 stage-2 steps, whatever steps: says")
    func explicitSigmasDecideTheStepCount() throws {
        let sampler = Sampler(video: DistilledRenderer.neutralGuidance,
                              audio: DistilledRenderer.neutralGuidance)

        let stage1 = CountingDenoiser()
        let t1 = try sampler.run(steps: 99, initial: Self.tinyStreams(), denoiser: stage1,
                                 explicitSigmas: FlowSchedule.distilledStage1,
                                 stepper: .ancestral(eta: 1.0, sNoise: 1.0, seed: 10_000))
        #expect(stage1.passesByStep.count == 8)
        #expect(t1.states.count == 9)            // `steps + 1` starting latents
        #expect(t1.sigmas == FlowSchedule.distilledStage1)
        #expect(stage1.sigmasSeen == Array(FlowSchedule.distilledStage1.dropLast()))

        let stage2 = CountingDenoiser()
        let t2 = try sampler.run(steps: 99, initial: Self.tinyStreams(), denoiser: stage2,
                                 explicitSigmas: FlowSchedule.distilledStage2,
                                 stepper: .euler)
        #expect(stage2.passesByStep.count == 3)
        #expect(t2.states.count == 4)
        #expect(t2.sigmas == FlowSchedule.distilledStage2)
    }

    // MARK: - No guider

    /// The claim `DistilledRenderer` makes about needing no new code for the no-guider path.
    ///
    /// The distilled path denoises without a guider at all, which here is `neutralGuidance`,
    /// and the whole of the evidence that the two are the same thing is here: every enable
    /// predicate false, so `plan` asks for one pass; and one pass per step for both stages
    /// is the recorded `observed_transformer_forwards_total: 11`.
    @Test("The distilled guidance plans exactly one conditional pass per step")
    func neutralGuidanceIsOnePassPerStep() {
        let g = DistilledRenderer.neutralGuidance
        #expect(g.runsUnconditional == false)
        #expect(g.runsPerturbed == false)
        #expect(g.runsIsolatedModality == false)
        #expect(g.rescaleScale == 0.0)

        let sampler = Sampler(video: g, audio: g)
        for step in 0 ..< 8 {
            #expect(sampler.plan(step: step).passes == [.conditional])
            #expect(sampler.plan(step: step).videoEnabled)
            #expect(sampler.plan(step: step).audioEnabled)
        }
        #expect(sampler.passCount(steps: 8) == 8)
        #expect(sampler.passCount(steps: 3) == 3)
        #expect(sampler.passCount(steps: 8) + sampler.passCount(steps: 3) == 11)
    }

    /// The distilled denoiser returns the prediction; it does not combine anything. With
    /// every coefficient at its neutral value and no rescale, `Sampler.combine` must be the
    /// identity **bit for bit** — not merely close. A stray `rescaleScale` here would be
    /// 2.47e-01 of quiet difference.
    @Test("With no guider the combination is the conditional prediction, unchanged")
    func combineIsTheIdentityUnderNeutralGuidance() throws {
        let sampler = Sampler(video: DistilledRenderer.neutralGuidance,
                              audio: DistilledRenderer.neutralGuidance)
        let cond = MLXArray(converting: [0.5, -1.25, 3.0, 0.0, -0.125, 7.5]).reshaped([1, 2, 3])
            .asType(.bfloat16)
        let out = try sampler.combine([.conditional: cond],
                                      params: DistilledRenderer.neutralGuidance,
                                      stream: "video")
        #expect(out.dtype == cond.dtype)
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(cond, out))
    }

    // MARK: - The stage-2 handoff

    /// Seeding stage 2 at `noiseScale = stage2Sigmas[0]`, spelled out by hand.
    ///
    /// `lerp(a, b, w)` is `a + w * (b - a)`, in fp32. This is the entire mechanism by
    /// which stage 2 continues stage 1's draft rather than starting over, and it is a move
    /// *toward* noise, not an addition of noise: at 0.909375 the upscaled latent keeps
    /// 9.0625% of its weight. Getting it backwards (`noise + 0.909375 * initial`) or
    /// additive (`initial + 0.909375 * noise`) produces a finite, correctly shaped tensor
    /// with a plausible histogram.
    @Test("The stage-2 handoff is lerp(initial, noise, 0.909375) to the bit")
    func stage2HandoffIsALerpTowardNoise() {
        let scale = FlowSchedule.distilledStage2[0]
        #expect(scale == 0.909375)

        let initial = MLXArray(converting: [1.0, -2.0, 0.5, 0.0]).reshaped([1, 2, 2])
            .asType(.bfloat16)
        let noise = MLXArray(converting: [-0.5, 0.25, 4.0, -1.0]).reshaped([1, 2, 2])
            .asType(.bfloat16)

        let seeded = Sampler.seedConditioned(noise: noise, denoiseMask: nil, cleanLatent: nil,
                                             initial: initial, noiseScale: scale)
        let byHand = (initial.asType(.float32)
            + scale * (noise.asType(.float32) - initial.asType(.float32))).asType(.bfloat16)
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(byHand, seeded))

        // And it is genuinely a blend, not a pass-through of either operand: at 0.909375 the
        // result must differ from both. A `noiseScale` accidentally read as 1.0 (a fresh
        // render) would make the first of these fail; as 0.0 (keep the draft), the second.
        #expect(!Pipeline.noiseExactlyMatchesInitialLatent(seeded, noise))
        #expect(!Pipeline.noiseExactlyMatchesInitialLatent(seeded, initial))
    }

    /// A fresh render's seeding is unchanged by the two-stage code existing: with no
    /// `initial` and scale 1.0, the initial state starts from zeros and the first `lerp`
    /// returns the noise. That is stage 1's path.
    @Test("Stage 1 seeds from pure noise, as create_initial_state does with no initial latent")
    func stage1SeedsFromPureNoise() {
        let noise = MLXArray(converting: [0.75, -3.5, 1.0, 0.0]).reshaped([1, 2, 2])
            .asType(.bfloat16)
        let seeded = Sampler.seedConditioned(noise: noise, denoiseMask: nil, cleanLatent: nil)
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(noise, seeded))
    }

    // MARK: - The audio latent does not go through the upsampler

    /// **The single easiest mistake in this feature, held by a test.**
    ///
    /// Only the video latent is upsampled; stage 2 receives the stage-1 audio tensor
    /// untouched. There is no audio upsampler in this codebase or in any checkpoint.
    /// `DistilledRenderer.stage2Initial` is the seam where the two could be confused,
    /// and this drives it with marker tensors: the audio result must move when
    /// `stage1Audio` moves and must **not** move when the upscaled video tokens move.
    @Test("The audio latent is stage 1's, not the upsampler's")
    func theAudioLatentIsNotUpsampled() {
        let scale = FlowSchedule.distilledStage2[0]
        let noise = Sampler.Streams(
            video: MLXArray.zeros([1, 3120, 128], dtype: .bfloat16),
            audio: MLXArray.zeros([1, 101, 128], dtype: .bfloat16))
        let stage1Audio = MLXArray.full([1, 101, 128], values: MLXArray(Float(2)))
            .asType(.bfloat16)

        func handoff(video marker: Float) -> Sampler.Streams {
            DistilledRenderer.stage2Initial(
                upscaledVideoTokens: MLXArray.full([1, 3120, 128],
                                                   values: MLXArray(marker)).asType(.bfloat16),
                stage1Audio: stage1Audio, noise: noise, noiseScale: scale,
                conditioning: nil)
        }

        let a = handoff(video: 5)
        let b = handoff(video: -9)

        // The audio stream ignores the video stream entirely.
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(a.audio, b.audio))
        // ...and the video stream does not, so the fixture is discriminating.
        #expect(!Pipeline.noiseExactlyMatchesInitialLatent(a.video, b.video))

        // The audio value is stage 1's, lerped toward noise — the same arithmetic the video
        // gets, from a different tensor.
        let expected = (stage1Audio.asType(.float32) * (1 - scale)).asType(.bfloat16)
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(expected, a.audio))

        // Shape, stated separately: the video stream grew 4x in tokens across the upsample
        // and the audio stream did not move at all.
        #expect(a.audio.shape == stage1Audio.shape)
        #expect(a.video.shape == [1, 3120, 128])
    }

    /// The other half of the same claim, at the geometry level: halving the picture leaves
    /// the audio stream identical, because the audio latent shape is derived from the frame
    /// count and the frame rate and never from the picture size. So there is no "audio at
    /// half resolution" for an upsampler to undo.
    @Test("Stage 1 halves the picture and leaves the audio stream alone")
    func stage1GeometryHalvesOnlyThePicture() throws {
        let spec = DistilledRenderer.Spec(prompt: "x",
                                          geometry: Renderer.Spec.productionGeometry)
        let full = spec.geometry, half = spec.stage1Geometry
        #expect((full.width, full.height) == (640, 384))
        #expect((half.width, half.height) == (320, 192))
        #expect(half.frames == full.frames)
        #expect(half.frameRate == full.frameRate)

        // 780 tokens at stage 1, 3120 at stage 2 — exactly the recorded
        // `noise.video` [1, 780, 128] and `noise.video.001` [1, 3120, 128].
        #expect(half.videoTokens == 780)
        #expect(full.videoTokens == 3120)
        #expect(full.videoTokens == half.videoTokens * 4)

        // 101 in both, which is `noise.audio` and `noise.audio.001`.
        #expect(half.audioTokens == 101)
        #expect(half.audioTokens == full.audioTokens)

        try spec.validate()
    }

    /// The video latent's NCDHW shape at each stage — the upsampler's input and output.
    @Test("The upsample runs [1,128,13,6,10] -> [1,128,13,12,20] at production shape")
    func theUpsampleBoundaryShapes() throws {
        let spec = DistilledRenderer.Spec(prompt: "x",
                                          geometry: Renderer.Spec.productionGeometry)
        func latentShape(_ g: DiTForward.Geometry) -> [Int] {
            [1, g.latent.videoChannels, g.latentFrames,
             g.height / g.latent.spatialScale, g.width / g.latent.spatialScale]
        }
        #expect(latentShape(spec.stage1Geometry) == [1, 128, 13, 6, 10])
        #expect(latentShape(spec.geometry) == [1, 128, 13, 12, 20])

        // And the round trip the orchestration performs — unpatchify, upsample, patchify —
        // is exact in both directions, so the detour into VAE layout costs nothing.
        var values: [Float32] = []
        values.reserveCapacity(780 * 128)
        for i in 0 ..< (780 * 128) { values.append(Float32(i % 17) - 8) }
        let tokens = MLXArray(values, [1, 780, 128]).asType(.bfloat16)
        let unpatched = try Sampler.unpatchifyVideo(tokens, latentFrames: 13, height: 6, width: 10)
        #expect(unpatched.shape == [1, 128, 13, 6, 10])
        #expect(Pipeline.noiseExactlyMatchesInitialLatent(
            tokens, try Sampler.patchifyVideo(unpatched)))
    }

    /// Routing the audio latent into the video path is a throw, not a silent success: it is
    /// rank 4 where the upsampler and `patchifyVideo` both demand rank 5.
    @Test("An audio latent cannot be fed to the video patchifier")
    func theVideoPatchifierRefusesAnAudioLatent() {
        let audio = MLXArray.zeros([1, 8, 101, 16], dtype: .bfloat16)
        #expect(throws: Sampler.Failure.self) { _ = try Sampler.patchifyVideo(audio) }
    }

    // MARK: - The checkpoint gate on the ancestral sampler

    /// The ancestral sampler is selected when the model version is at least 2.5, compared
    /// component by component so that a shorter prefix sorts *below* a longer one sharing
    /// it. An unset or unparseable version is empty, which selects the deterministic
    /// sampler.
    @Test("The ancestral sampler is gated on model_version >= 2.5, Python's ordering")
    func ancestralSamplerVersionGate() {
        func at(_ v: String?) -> Bool {
            DistilledRenderer.isAtLeastVersion(
                DistilledRenderer.parseModelVersion(v),
                DistilledRenderer.ancestralSamplerSinceVersion)
        }
        #expect(at("2.5"))
        #expect(at("2.5.1"))
        #expect(at("2.6"))
        #expect(at("3"))
        // Pre-release tags, dot- and hyphen-separated. The separator is normalised before
        // parsing so a pre-release maps onto the generation it is for.
        #expect(at("2.5.rc1"))
        #expect(at("2.5-rc2"))

        #expect(!at("2.4"))
        #expect(!at("2.3.rc1"))
        #expect(!at("2"))          // (2,) < (2, 5)
        #expect(!at(nil))          // () < everything
        #expect(!at(""))
        #expect(!at("dev"))

        #expect(DistilledRenderer.parseModelVersion("2.5-rc2") == [2, 5])
        #expect(DistilledRenderer.parseModelVersion("2.4") == [2, 4])
        #expect(DistilledRenderer.parseModelVersion("dev").isEmpty)
    }

    /// The distilled transformer actually on this box, if it is mounted: the whole point of
    /// the gate is which sampler stage 1 gets, and reading it off the real file is the only
    /// way to know the answer is not "deterministic, because the metadata key moved".
    @Test("The distilled 2.5 checkpoint selects the ancestral stage-1 sampler")
    func theRealDistilledCheckpointIsAncestral() {
        let path = Self.root + "/diffusion_models/"
            + "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        guard FileManager.default.fileExists(atPath: path) else { return }
        #expect(DistilledRenderer.shouldUseAncestralSampler(
            transformer: URL(fileURLWithPath: path)))
    }

    /// `DistilledRenderer.Checkpoints.verifyPresent` parses the upsampler's
    /// `__metadata__.config` rather than only stat-ing the file, and this is why.
    ///
    /// Two upscalers ship in `latent_upscale_models/` under names that differ by one word.
    /// The temporal one declares `spatial_upsample: false, temporal_upsample: true,
    /// rational_resampler: true, mid_channels: 512` — a different network entirely — so
    /// naming it by mistake must fail at admission, not after a stage-1 render, and not as
    /// a rank error deep inside `spatialUpsample`.
    ///
    /// Header-only: both reads are a few kilobytes, not the 995 MB and 262 MB files.
    @Test("The temporal upscaler is rejected on its own config; the spatial one is accepted")
    func theTemporalUpscalerIsRejected() throws {
        let dir = Self.root + "/latent_upscale_models/"
        let spatial = dir + "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
        let temporal = dir + "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
        guard FileManager.default.fileExists(atPath: spatial),
              FileManager.default.fileExists(atPath: temporal) else { return }

        let good = try LatentSpatialUpsampler.readConfiguration(
            try SafetensorsHeader.read(from: URL(fileURLWithPath: spatial)))
        #expect(good.inChannels == 128)
        #expect(good.midChannels == 1024)
        #expect(good.spatialScale == 2)

        #expect(throws: LatentSpatialUpsampler.Failure.self) {
            _ = try LatentSpatialUpsampler.readConfiguration(
                try SafetensorsHeader.read(from: URL(fileURLWithPath: temporal)))
        }
    }

    // MARK: - Constants and refusals

    @Test("The ancestral constants, as named")
    func ancestralConstants() {
        #expect(DistilledRenderer.ancestralEta == 1.0)
        #expect(DistilledRenderer.ancestralSNoise == 1.0)
        #expect(DistilledRenderer.ancestralNoiseSeedOffset == 10_000)
        // The two offsets must differ, or the loop's first draw and stage 2's initial latent
        // would come off the same key — the collision the offsets exist to prevent.
        #expect(DistilledRenderer.stage2NoiseSeedOffset
            != DistilledRenderer.ancestralNoiseSeedOffset)
        #expect(DistilledRenderer.stage2NoiseSeedOffset != 0)
    }

    /// A two-stage resolution must be divisible by 64, not 32, because stage 1 samples at
    /// half and that half must still sit on the VAE's 32-pixel latent grid.
    @Test("A two-stage render refuses a resolution that is only 32-divisible")
    func resolutionMustBeSixtyFourDivisible() {
        func spec(_ w: Int, _ h: Int) -> DistilledRenderer.Spec {
            DistilledRenderer.Spec(
                prompt: "x",
                geometry: DiTForward.Geometry(frames: 97, height: h, width: w, frameRate: 24))
        }
        #expect(throws: DistilledRenderer.Failure.self) { try spec(672, 384).validate() }
        #expect(throws: DistilledRenderer.Failure.self) { try spec(640, 416).validate() }
        #expect(throws: Never.self) { try spec(640, 384).validate() }
    }

    /// A conditioning pair built for one stage handed to the other is refused at the seam.
    /// The audio half is why this is not redundant with a shape check inside the sampler:
    /// the audio mask is `[1, 101, 1]` in *both* stages, so only the video half would ever
    /// throw on its own.
    @Test("A conditioning pair built for the wrong stage is refused")
    func conditioningMustMatchItsStage() throws {
        let spec = DistilledRenderer.Spec(prompt: "x",
                                          geometry: Renderer.Spec.productionGeometry)
        let built = try #require(try RenderConditioning.build(
            videoLatent: MLXArray.zeros([1, 128, 13, 12, 20], dtype: .float32),
            strength: 1.0, geometry: spec.geometry))

        // Right stage: accepted.
        #expect(throws: Never.self) {
            try DistilledRenderer.checkConditioning(built, geometry: spec.geometry, stage: 2)
        }
        // Stage 1's geometry: 780 video tokens against the pair's 3120.
        #expect(throws: DistilledRenderer.Failure.self) {
            try DistilledRenderer.checkConditioning(built, geometry: spec.stage1Geometry,
                                                    stage: 1)
        }
        // No conditioning at all is the ordinary case and must pass.
        #expect(throws: Never.self) {
            try DistilledRenderer.checkConditioning(nil, geometry: spec.geometry, stage: 2)
        }
    }
}

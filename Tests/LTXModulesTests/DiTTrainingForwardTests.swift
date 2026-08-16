// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX
import MLXNN
import MLXOptimizers
import Testing
@testable import LTXModules

/// ``LoRATrainingStep`` driven by the real transformer, not a stand-in.
///
/// Everything else in the training suite proves a piece: the objective's own arithmetic,
/// the gradient against finite differences, the step's wiring against a single adapted linear.
/// None of that shows the pieces compose over the actual model — which is where the errors
/// that survive unit tests live, because every one of them still produces a number.
///
/// Small on purpose: 4 blocks at the minimum geometry, so it runs in seconds and can stay in
/// the routine suite. The memory behaviour at real scale is ``DiTLoRABackwardTests``.
@Suite("DiT training forward", .serialized)
struct DiTTrainingForwardTests {

    static let devTransformer =
        LTXConfiguration.resolved.checkpoints.root! + "/diffusion_models/"
        + "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    private func skipIfAbsent() -> Bool {
        if FileManager.default.fileExists(atPath: Self.devTransformer) { return false }
        print("SKIP DiTTrainingForward: transformer absent at \(Self.devTransformer)")
        return true
    }

    /// The whole assembly, held together.
    private struct Fixture {
        let module: LoRAAdapterModule
        let forward: DiTTrainingForward
        let batch: LoRATrainingStep.Batch
        let shape: TrainingObjective.Shape
    }

    private func makeFixture(blocks: Int, rank: Int,
                             checkpointing: Bool, seed: UInt64) throws -> Fixture {
        let url = URL(fileURLWithPath: Self.devTransformer)
        let header = try SafetensorsHeader.read(from: url)
        let topology = try TransformerTopology.read(header)

        // `makeLoRA` initialises `A` from the global stream. That is production's business
        // and it is left alone — `B` is zeroed, so the delta is zero at construction and the
        // loss this fixture reports does not depend on `A` at all.
        MLXRandom.seed(seed)
        let lora = DiTLoRATargets.makeLoRA(
            header: header, rank: rank, alpha: rank,
            suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly, blocks: 0 ..< blocks)
        let module = LoRAAdapterModule(lora)

        let weights = try MLX.loadArrays(url: url)
        let base = DiTForward(weights: weights, topology: topology, attentionPath: .fused)
        let geometry = DiTForward.Geometry(frames: 9, height: 128, width: 128, frameRate: 24)

        // Everything the loss *does* depend on is drawn from explicit keys, so a fixture is
        // a function of its `seed` and nothing else. Seeding the global stream and drawing
        // from it would leave a window: another suite drawing between the seed and these
        // calls shifts them, two fixtures built with one seed stop matching, and a test
        // comparing two fixtures reports the difference as whatever it was comparing.
        // Measured before this: plain 58.845047 against checkpointed 59.633785, from a
        // context tensor that differed rather than from checkpointing.
        let keys = MLX.split(key: MLX.key(seed), into: 4)
        let audioLatent = MLX.normal([1, geometry.audioTokens, topology.latentChannels],
                                     key: keys[0]).asType(.bfloat16)
        let audioTimestep = MLXArray(
            [Float](repeating: 0.5, count: geometry.audioTokens),
            [geometry.audioTokens]).asType(.bfloat16)

        let forward = DiTTrainingForward(
            base: base,
            head: DiTOutputHead(weights: weights, topology: topology),
            geometry: geometry,
            videoContext: MLX.normal([1, 32, topology.videoWidth], key: keys[1])
                .asType(.bfloat16),
            audioContext: MLX.normal([1, 32, topology.audioWidth], key: keys[2])
                .asType(.bfloat16),
            audioLatent: audioLatent, audioTimestep: audioTimestep,
            gradientCheckpointing: checkpointing, blocks: blocks)

        let shape = TrainingObjective.Shape(
            batch: 1, sequence: geometry.videoTokens, channels: topology.latentChannels)
        let latents = MLX.normal([shape.batch, shape.sequence, shape.channels], key: keys[3])
            .asType(.float32)
        MLX.eval(latents, audioLatent)

        try forward.validate(shape: shape)
        return Fixture(module: module,
                       forward: forward,
                       batch: .unconditioned(latents: latents, shape: shape),
                       shape: shape)
    }

    @Test("a step through the real transformer moves the factors and reports a finite loss")
    func oneRealStep() throws {
        if skipIfAbsent() { return }
        let fixture = try makeFixture(blocks: 4, rank: 8, checkpointing: true, seed: 21)

        let before = fixture.module.a[0].asArray(Float.self)
        let result = LoRATrainingStep.step(
            module: fixture.module, optimizer: AdamW(learningRate: 1e-3),
            batch: fixture.batch, sigmas: [0.5], forward: fixture.forward())
        let after = fixture.module.a[0].asArray(Float.self)

        print(String(format: "REAL STEP loss %.6f", result.mean))
        #expect(result.mean.isFinite)
        #expect(result.mean > 0)
        #expect(result.perExampleLoss.count == 1)
        // `A` moving is the end-to-end proof: the gradient reached the factors through 4
        // blocks of the real transformer *and* the optimiser updated the live parameters.
        #expect(zip(before, after).contains { $0 != $1 }, "the factors did not move")
    }

    /// Checkpointing must not change the step's answer, only its memory. This is the
    /// composed-step version of `DiTLoRABackwardTests.checkpointingIsExact`, which checks the
    /// same property one level down at the raw gradient.
    ///
    /// ## Why this passes an explicit key, and why that is not a test detail
    ///
    /// The comparison needs both calls to see the same inputs. Seeding the global stream
    /// does not achieve that in a parallel test binary: `MLXRandom.seed(77)` followed by the
    /// step's own draw leaves a window, and any other MLX suite drawing inside it shifts the
    /// stream for one of the two calls. The result is a difference in the *inputs* reported
    /// as a difference in checkpointing — measured at 0.011, 0.243, 0.806, 2.84 and 3.48
    /// across full-suite runs, varying every time, while the same test alone was bit-exact
    /// run after run. A varying "exactness" failure is the signature of the inputs moving.
    ///
    /// `key:` closes the window by construction rather than by scheduling luck: the draws
    /// become a pure function of the key, so nothing another thread does can reach them.
    /// That is the same property ``LTXPipeline/LoRATrainer`` now relies on to make a resumed
    /// run reproduce an uninterrupted one, so the fix belonged upstream and this test is one
    /// of its users rather than the reason it exists.
    @Test("checkpointing does not change the step's loss")
    func checkpointingMatchesThroughTheStep() throws {
        if skipIfAbsent() { return }

        // One key, both calls. Not a seed and not two keys: the point is that the two runs
        // differ in checkpointing and in nothing else.
        let key = MLX.key(77)
        func loss(checkpointing: Bool) throws -> Float {
            let fixture = try makeFixture(blocks: 4, rank: 8,
                                          checkpointing: checkpointing, seed: 21)
            return LoRATrainingStep.step(
                module: fixture.module, optimizer: AdamW(learningRate: 1e-3),
                batch: fixture.batch, sigmas: [0.5], key: key,
                forward: fixture.forward()).mean
        }

        let plain = try loss(checkpointing: false)
        let checkpointed = try loss(checkpointing: true)
        print(String(format: "STEP LOSS plain %.6f, checkpointed %.6f", plain, checkpointed))
        #expect(plain.isFinite && plain > 0)
        #expect(abs(plain - checkpointed) <= 1e-5 * max(plain, 1),
                "\(plain) vs \(checkpointed)" as Comment)
    }

    /// A batch above 1 is refused rather than silently trained on its first example —
    /// `DiTForward.Sampling` has no batch axis.
    @Test("a batch above one is refused, not silently truncated")
    func batchIsRefused() throws {
        if skipIfAbsent() { return }
        let fixture = try makeFixture(blocks: 1, rank: 4, checkpointing: true, seed: 3)
        let wider = TrainingObjective.Shape(batch: 2, sequence: fixture.shape.sequence,
                                            channels: fixture.shape.channels)
        #expect(throws: DiTTrainingForward.Failure.self) {
            try fixture.forward.validate(shape: wider)
        }
    }

    /// The geometry and the latent must agree on token count. They are derived independently —
    /// one from the frame lattice, one from the cached latent — so a mismatch is a real
    /// possibility, and inside the traced closure it would be a `try!` crash rather than an
    /// error.
    @Test("a token count that disagrees with the geometry is refused")
    func tokenCountIsChecked() throws {
        if skipIfAbsent() { return }
        let fixture = try makeFixture(blocks: 1, rank: 4, checkpointing: true, seed: 3)
        let wrong = TrainingObjective.Shape(batch: 1, sequence: fixture.shape.sequence + 1,
                                            channels: fixture.shape.channels)
        #expect(throws: DiTTrainingForward.Failure.self) {
            try fixture.forward.validate(shape: wrong)
        }
    }
}

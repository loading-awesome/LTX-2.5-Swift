// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXFoundation

/// The training objective, pinned to a recorded table of inputs and the losses they
/// produce.
///
/// `Tests/Fixtures/reference/training_objective.json` holds the tensors alongside the
/// expected outputs, so what is checked is the objective's result rather than a
/// restatement of its formula.
///
/// The negative tests matter as much as the positive one. Each of `flippedTargetDiffers`,
/// `swappedInterpolationDiffers` and `unmaskedVideoLossDiffers` asserts that a plausible
/// wrong implementation produces a *different* number — without them, a test suite can pass
/// against an objective that merely resembles the intended one.
@Suite("Training objective")
struct TrainingObjectiveTests {

    struct Table: Decodable {
        struct Shape: Decodable {
            let batch: Int, seq: Int, channels: Int
            let audio_seq: Int, audio_channels: Int
        }
        struct Loss: Decodable { let with_audio: Bool; let loss: [Float] }
        let shape: Shape
        let sigmas: [Float]
        let clean: [Float]
        let noise: [Float]
        let pred: [Float]
        let audio_pred: [Float]
        let audio_target: [Float]
        let conditioning_mask: [Bool]
        let noisy: [Float]
        let target: [Float]
        let losses: [Loss]
        let fully_masked_loss: [Float]
    }

    static func table() throws -> Table? {
        var dir = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 6 {
            dir = dir.deletingLastPathComponent()
            let candidate = dir.appendingPathComponent("Tests/Fixtures/reference/training_objective.json")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try JSONDecoder().decode(Table.self,
                                                from: try Data(contentsOf: candidate))
            }
        }
        print("SKIP TrainingObjective: Tests/Fixtures/reference/training_objective.json absent")
        return nil
    }

    private static func videoShape(_ t: Table) -> TrainingObjective.Shape {
        .init(batch: t.shape.batch, sequence: t.shape.seq, channels: t.shape.channels)
    }

    private static func audioShape(_ t: Table) -> TrainingObjective.Shape {
        .init(batch: t.shape.batch, sequence: t.shape.audio_seq,
              channels: t.shape.audio_channels)
    }

    // MARK: - The rules

    @Test("the noising interpolation matches the recorded table exactly")
    func noisingMatches() throws {
        guard let t = try Self.table() else { return }
        let ours = TrainingObjective.noised(clean: t.clean, noise: t.noise, sigmas: t.sigmas,
                                            conditioningMask: t.conditioning_mask,
                                            shape: Self.videoShape(t))
        #expect(ours == t.noisy)
    }

    @Test("conditioning tokens keep the clean latent")
    func conditioningTokensStayClean() throws {
        guard let t = try Self.table() else { return }
        let shape = Self.videoShape(t)
        let ours = TrainingObjective.noised(clean: t.clean, noise: t.noise, sigmas: t.sigmas,
                                            conditioningMask: t.conditioning_mask, shape: shape)
        var checked = 0
        for b in 0 ..< shape.batch {
            for s in 0 ..< shape.sequence where t.conditioning_mask[b * shape.sequence + s] {
                for c in 0 ..< shape.channels {
                    let i = shape.index(b, s, c)
                    #expect(ours[i] == t.clean[i])
                    checked += 1
                }
            }
        }
        #expect(checked > 0, "the fixture has no conditioning tokens to check")
    }

    @Test("the target is noise minus clean")
    func targetMatches() throws {
        guard let t = try Self.table() else { return }
        #expect(TrainingObjective.velocityTarget(clean: t.clean, noise: t.noise) == t.target)
    }

    @Test("the video loss matches the recorded table")
    func videoLossMatches() throws {
        guard let t = try Self.table() else { return }
        guard let expected = t.losses.first(where: { !$0.with_audio })?.loss else { return }
        let mask = t.conditioning_mask.map { !$0 }
        let ours = TrainingObjective.videoLoss(prediction: t.pred, target: t.target,
                                               lossMask: mask, shape: Self.videoShape(t))
        for (index, value) in expected.enumerated() {
            #expect(abs(ours[index] - value) <= 1e-6,
                    "batch \(index): got \(ours[index]), reference \(value)" as Comment)
        }
    }

    @Test("video plus audio matches the recorded table, and audio is unmasked")
    func combinedLossMatches() throws {
        guard let t = try Self.table() else { return }
        guard let expected = t.losses.first(where: { $0.with_audio })?.loss else { return }
        let mask = t.conditioning_mask.map { !$0 }
        let video = TrainingObjective.videoLoss(prediction: t.pred, target: t.target,
                                                lossMask: mask, shape: Self.videoShape(t))
        let audio = TrainingObjective.audioLoss(prediction: t.audio_pred,
                                                target: t.audio_target,
                                                shape: Self.audioShape(t))
        let ours = TrainingObjective.combinedLoss(video: video, audio: audio)
        for (index, value) in expected.enumerated() {
            #expect(abs(ours[index] - value) <= 1e-6,
                    "batch \(index): got \(ours[index]), reference \(value)" as Comment)
        }
    }

    /// The clamp, at the only input that reveals it. Dividing by an unmasked-token count
    /// returns NaN here; the objective must return 0.
    @Test("an entirely conditioning example gives 0, not NaN")
    func fullyMaskedGivesZero() throws {
        guard let t = try Self.table() else { return }
        let shape = TrainingObjective.Shape(batch: 1, sequence: t.shape.seq,
                                            channels: t.shape.channels)
        let ours = TrainingObjective.videoLoss(
            prediction: Array(t.pred.prefix(shape.count)),
            target: Array(t.target.prefix(shape.count)),
            lossMask: [Bool](repeating: false, count: shape.sequence),
            shape: shape)
        #expect(ours[0] == t.fully_masked_loss[0])
        #expect(!ours[0].isNaN)
    }

    @Test("per-token timesteps are zero on conditioning tokens and sigma elsewhere")
    func perTokenTimesteps() throws {
        guard let t = try Self.table() else { return }
        let ours = TrainingObjective.perTokenTimesteps(
            sigmas: t.sigmas, conditioningMask: t.conditioning_mask,
            batch: t.shape.batch, sequence: t.shape.seq)
        for b in 0 ..< t.shape.batch {
            for s in 0 ..< t.shape.seq {
                let i = b * t.shape.seq + s
                #expect(ours[i] == (t.conditioning_mask[i] ? 0 : t.sigmas[b]))
            }
        }
    }

    // MARK: - The wrong implementations, proven wrong

    /// `clean - noise` instead of `noise - clean`. Squared error is symmetric in the sign of
    /// the *error*, not of the target, so this is a real divergence that produces a
    /// perfectly healthy-looking loss curve.
    @Test("a flipped target sign does not match")
    func flippedTargetDiffers() throws {
        guard let t = try Self.table() else { return }
        let flipped = TrainingObjective.velocityTarget(clean: t.noise, noise: t.clean)
        #expect(flipped != t.target)
    }

    @Test("swapping the interpolation coefficients does not match")
    func swappedInterpolationDiffers() throws {
        guard let t = try Self.table() else { return }
        // σ·clean + (1-σ)·noise — the time-reversed schedule.
        let swapped = TrainingObjective.noised(clean: t.noise, noise: t.clean,
                                               sigmas: t.sigmas,
                                               conditioningMask: t.conditioning_mask,
                                               shape: Self.videoShape(t))
        #expect(swapped != t.noisy)
    }

    /// If the mask were ignored, the fixture's conditioned example would change. The other
    /// example has no conditioning tokens, so it must be unaffected — which is what makes
    /// this a check on the masking rather than on the arithmetic.
    @Test("ignoring the loss mask changes the conditioned example only")
    func unmaskedVideoLossDiffers() throws {
        guard let t = try Self.table() else { return }
        let shape = Self.videoShape(t)
        let masked = TrainingObjective.videoLoss(
            prediction: t.pred, target: t.target,
            lossMask: t.conditioning_mask.map { !$0 }, shape: shape)
        let unmasked = TrainingObjective.videoLoss(
            prediction: t.pred, target: t.target,
            lossMask: [Bool](repeating: true, count: shape.tokens), shape: shape)
        #expect(masked[0] != unmasked[0], "example 0 has conditioning tokens")
        #expect(masked[1] == unmasked[1], "example 1 has none, so it cannot move")
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXFoundation

/// The timestep sampler, pinned to a recorded table.
///
/// `Tests/Fixtures/reference/training_timestep_sampler.json` holds injected normal and
/// uniform draws alongside the timesteps they must produce. Because the draws are injected,
/// what is checked is the transform rather than a distribution.
@Suite("Training timestep sampler")
struct TrainingTimestepSamplerTests {

    struct Table: Decodable {
        struct Row: Decodable {
            let seq_length: Int
            let shift: Double
            let normal: [Float]
            let uniform: [Float]
            let prob: [Float]
            let timesteps: [Float]
        }
        struct Defaults: Decodable {
            let std: Float
            let eps: Float
            let uniform_prob: Float
            let normal_999_percentile: Double
            let normal_005_percentile: Double
        }
        let defaults: Defaults
        let rows: [Row]
    }

    /// Walk up to the package root: tests run with an unrelated working directory.
    static func referenceURL() -> URL? {
        var dir = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 6 {
            dir = dir.deletingLastPathComponent()
            let candidate = dir.appendingPathComponent(
                "Tests/Fixtures/reference/training_timestep_sampler.json")
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    static func table() throws -> Table? {
        guard let url = referenceURL() else {
            print("SKIP TrainingTimestepSampler: "
                + "Tests/Fixtures/reference/training_timestep_sampler.json absent")
            return nil
        }
        return try JSONDecoder().decode(Table.self, from: try Data(contentsOf: url))
    }

    /// Every draw, at every sequence length.
    ///
    /// **Not bit-exact, and the reason is measured rather than assumed.** 40 of 48 values
    /// match exactly; the other 8 differ by at most **2.98e-07**, and every one of them is
    /// on the stretched logit-normal branch. The difference is in the last ulps of the
    /// sigmoid, and the division by the percentile span `(p999 - p005)` amplifies it by a
    /// few ulps more. Evaluating the sigmoid in `Double` and rounding once does not close
    /// it — it makes it marginally worse.
    ///
    /// A tolerance is the right answer here. This value is the noise level assigned to one
    /// training example, drawn from a deliberately stochastic distribution; 3e-07 of sigma
    /// is not a difference that exists. What must be exact is the **branch** — see
    /// `uniformBranchIsExact`, because choosing the wrong branch is categorical, not
    /// numerical.
    @Test("every reference row reproduces within a measured tolerance")
    func matchesReference() throws {
        guard let table = try Self.table() else { return }
        let sampler = TrainingTimestepSampler.ShiftedLogitNormal(
            std: table.defaults.std, eps: table.defaults.eps,
            uniformProb: table.defaults.uniform_prob)

        var compared = 0, exact = 0
        var worst: Float = 0
        for row in table.rows {
            let ours = sampler.timesteps(normal: row.normal, uniform: row.uniform,
                                         prob: row.prob, sequenceLength: row.seq_length)
            #expect(ours.count == row.timesteps.count)
            for (index, expected) in row.timesteps.enumerated() {
                let delta = abs(ours[index] - expected)
                worst = max(worst, delta)
                if ours[index] == expected { exact += 1 }
                let detail = "seq \(row.seq_length) draw \(index): got \(ours[index]), "
                    + "reference \(expected), delta \(delta)"
                #expect(delta <= 1e-6, "\(detail)")
                compared += 1
            }
        }
        #expect(compared > 0, "the table decoded but contained no rows")
        // Pinned so an implementation change that made agreement *worse* fails here rather
        // than passing under a loose bound.
        #expect(worst <= 3.0e-07, "worst deviation \(worst) — was 2.98e-07 when measured")
        #expect(exact >= 40, "\(exact)/\(compared) exact — was 40 when measured")
    }

    /// The uniform fallback is pure arithmetic with no sigmoid in it, so it must be exact —
    /// and its exactness is what proves the branch selection itself agrees.
    @Test("the uniform branch is bit-exact, so branch selection is proven")
    func uniformBranchIsExact() throws {
        guard let table = try Self.table() else { return }
        let sampler = TrainingTimestepSampler.ShiftedLogitNormal(
            std: table.defaults.std, eps: table.defaults.eps,
            uniformProb: table.defaults.uniform_prob)
        var checked = 0
        for row in table.rows {
            for (index, prob) in row.prob.enumerated() where prob <= table.defaults.uniform_prob {
                let ours = sampler.timestep(normal: row.normal[index],
                                            uniform: row.uniform[index], prob: prob,
                                            sequenceLength: row.seq_length)
                #expect(ours == row.timesteps[index],
                        "\(row.seq_length)/\(index) took the wrong branch or rounded")
                checked += 1
            }
        }
        #expect(checked > 0)
    }

    @Test("the shift matches the recorded table at every sequence length")
    func shiftMatchesReference() throws {
        guard let table = try Self.table() else { return }
        for row in table.rows {
            let ours = TrainingTimestepSampler.ShiftedLogitNormal
                .shift(forSequenceLength: row.seq_length)
            #expect(abs(ours - row.shift) < 1e-12, "seq \(row.seq_length)" as Comment)
        }
    }

    /// The interpolation has no clamp, and adding one would look more careful while
    /// training long sequences against a different distribution.
    @Test("the shift extrapolates past the interpolation range rather than saturating")
    func shiftDoesNotClamp() {
        let s = TrainingTimestepSampler.ShiftedLogitNormal.self
        #expect(s.shift(forSequenceLength: 1024) == 0.95)
        #expect(abs(s.shift(forSequenceLength: 4096) - 2.05) < 1e-12)
        #expect(s.shift(forSequenceLength: 8192) > 2.05, "must extrapolate, not clamp")
        #expect(s.shift(forSequenceLength: 512) < 0.95, "and below the floor too")
    }

    /// The production shape's own value, written out so a change to the interpolation is
    /// visible as a diff rather than as a distribution that moved.
    @Test("the production shape's shift is 1.7005")
    func productionShift() {
        let shift = TrainingTimestepSampler.ShiftedLogitNormal
            .shift(forSequenceLength: 3120)
        #expect(abs(shift - 1.700520833) < 1e-9)
    }

    @Test("the mixing probability selects the uniform branch below uniform_prob")
    func uniformBranch() {
        let sampler = TrainingTimestepSampler.ShiftedLogitNormal()
        // prob <= uniformProb takes the uniform fallback, offset by eps.
        let uniform = sampler.timestep(normal: 0, uniform: 0.5, prob: 0.05,
                                       sequenceLength: 3120)
        #expect(abs(uniform - 0.5005) < 1e-6)
        // prob > uniformProb takes the stretched logit-normal, which at these draws is not
        // 0.5005 — i.e. the branch genuinely changed the answer.
        let stretched = sampler.timestep(normal: 0, uniform: 0.5, prob: 0.5,
                                         sequenceLength: 3120)
        #expect(stretched != uniform)
    }

    @Test("the uniform sampler is the identity over its range")
    func uniformSampler() {
        let sampler = TrainingTimestepSampler.Uniform()
        #expect(sampler.timestep(uniform: 0.0) == 0.0)
        #expect(sampler.timestep(uniform: 1.0) == 1.0)
        #expect(sampler.timestep(uniform: 0.25) == 0.25)
        let narrowed = TrainingTimestepSampler.Uniform(minValue: 0.2, maxValue: 0.8)
        #expect(abs(narrowed.timestep(uniform: 0.5) - 0.5) < 1e-6)
    }
}

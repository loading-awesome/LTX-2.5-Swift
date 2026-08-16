// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import MLXNN
import MLXOptimizers
import Testing
@testable import LTXModules

/// The training step's **gradient and optimiser**, against a recorded trajectory.
///
/// `TrainingObjectiveTests` pins the loss against `Tests/Fixtures/reference/training_objective.json`.
/// This pins what happens after it, which is where the failures that survive a loss check
/// live:
///
/// - `dL/dA` and `dL/dB` are not symmetric. A transposed factor, or `scaling` applied to the
///   wrong one, still yields a falling loss — differently conditioned, not obviously wrong.
///   ``TrainableLoRATests`` checks the gradient by central finite differences, which
///   differentiates *this implementation's own forward* and therefore cannot catch a
///   gradient that is correct for the wrong parameterisation.
/// - The video loss divides by the mask's own mean, and that divisor varies per example, so
///   it survives into `dL/dpred`. An implementation that treated it as a constant scale gets
///   the loss exactly right and the gradient wrong.
/// - AdamW's decoupled weight decay and bias correction are invisible in a loss curve.
///
/// The forward is a single adapted linear, so a failure here can only be the gradient or the
/// optimiser; putting the transformer in the middle would mean a failure could have come
/// from anywhere.
///
/// `Tests/Fixtures/reference/training_backward.json` records an actual run — losses,
/// gradients, and the factors after each optimiser step — so the trajectory is an
/// observation rather than a restatement of the formulas.
@Suite("Training backward gate", .serialized)
struct TrainingBackwardGateTests {

    struct Reference: Decodable {
        struct Shape: Decodable {
            let batch: Int, seq: Int
            let in_features: Int, out_features: Int
            let rank: Int, alpha: Int
        }
        struct Optimizer: Decodable {
            let lr: Float, beta1: Float, beta2: Float
            let eps: Float, weight_decay: Float, steps: Int
        }
        struct Step: Decodable {
            let loss: Float
            let a: [Float]
            let b: [Float]
        }
        let shape: Shape
        let scaling: Float
        let optimizer: Optimizer
        let weight: [Float]
        let a: [Float]
        let b: [Float]
        let noisy: [Float]
        let target_out: [Float]
        let conditioning_mask: [Bool]
        let loss: [Float]
        let grad_a: [Float]
        let grad_b: [Float]
        let trajectory: [Step]
    }

    static func load() throws -> Reference? {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Tests/Fixtures/reference/training_backward.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("SKIP TrainingBackwardGate: \(url.path) absent — run the exporter")
            return nil
        }
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    /// The forward, built from the same pieces the trainer composes: the base linear plus
    /// ``TrainableLoRA/delta(input:a:b:scaling:)``.
    private static func predict(noisy: MLXArray, weight: MLXArray,
                                a: MLXArray, b: MLXArray, scaling: Float) -> MLXArray {
        MLX.matmul(noisy, weight.transposed())
            + TrainableLoRA.delta(input: noisy, a: a, b: b, scaling: scaling)
    }

    /// The video loss, per example, as `TrainingObjective.videoLoss` defines it — in tensor
    /// space so it can be differentiated.
    private static func loss(pred: MLXArray, target: MLXArray, mask: MLXArray,
                             sequence: Int, channels: Int) -> MLXArray {
        let error = pred - target
        let masked = (error * error) * mask
        let numerator = masked.sum(axes: [1, 2]) / Float(sequence * channels)
        let denominator = MLX.maximum(mask.sum(axes: [1, 2]) / Float(sequence),
                                      MLXArray(Float(1e-8)))
        return numerator / denominator
    }

    private struct Fixture {
        let noisy: MLXArray, weight: MLXArray, target: MLXArray, mask: MLXArray
        let a: MLXArray, b: MLXArray
        let sequence: Int, channels: Int, scaling: Float
    }

    private static func fixture(_ reference: Reference) -> Fixture {
        let s = reference.shape
        // The loss mask is the conditioning mask's complement, shaped `[B, T, 1]` to
        // broadcast over channels — the same layout `LoRATrainingStep` builds.
        let lossMask = reference.conditioning_mask.map { $0 ? Float(0) : Float(1) }
        return Fixture(
            noisy: MLXArray(reference.noisy, [s.batch, s.seq, s.in_features]),
            weight: MLXArray(reference.weight, [s.out_features, s.in_features]),
            target: MLXArray(reference.target_out, [s.batch, s.seq, s.out_features]),
            mask: MLXArray(lossMask, [s.batch, s.seq, 1]),
            a: MLXArray(reference.a, [s.rank, s.in_features]),
            b: MLXArray(reference.b, [s.out_features, s.rank]),
            sequence: s.seq, channels: s.out_features, scaling: reference.scaling)
    }

    private static func compare(_ got: [Float], _ want: [Float], _ label: String,
                                tolerance: Float = 2e-6) {
        #expect(got.count == want.count, "\(label): \(got.count) vs \(want.count)" as Comment)
        guard got.count == want.count else { return }
        let magnitude = want.map(abs).max() ?? 0
        var worst: Float = 0
        for (g, w) in zip(got, want) { worst = max(worst, abs(g - w)) }
        // Non-trivial: a table of zeros would satisfy any comparison.
        #expect(magnitude > 1e-6, "\(label): the expected values are ~0" as Comment)
        #expect(worst <= tolerance * max(magnitude, 1),
                "\(label): max |Δ| \(worst) against magnitude \(magnitude)" as Comment)
    }

    // MARK: - The checks

    @Test("the loss on the composed forward matches the recorded table")
    func lossMatches() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)
        let value = Self.loss(pred: Self.predict(noisy: f.noisy, weight: f.weight,
                                                 a: f.a, b: f.b, scaling: f.scaling),
                              target: f.target, mask: f.mask,
                              sequence: f.sequence, channels: f.channels)
        MLX.eval(value)
        Self.compare(value.asArray(Float.self), reference.loss, "loss")
    }

    /// The check that matters. Finite differences compare this implementation against
    /// itself; this compares it against recorded gradients of the same loss.
    @Test("dL/dA and dL/dB match the recorded autograd values")
    func gradientsMatch() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)

        let valueAndGrad = MLX.valueAndGrad({ (arrays: [MLXArray]) -> [MLXArray] in
            let pred = Self.predict(noisy: f.noisy, weight: f.weight,
                                    a: arrays[0], b: arrays[1], scaling: f.scaling)
            // Summed over the batch, matching how the recorded gradients were reduced.
            return [Self.loss(pred: pred, target: f.target, mask: f.mask,
                              sequence: f.sequence, channels: f.channels).sum()]
        }, argumentNumbers: [0, 1])

        let (_, gradients) = valueAndGrad([f.a, f.b])
        MLX.eval(gradients)
        #expect(gradients.count == 2, "argumentNumbers defaults to [0] — a short list is that")

        Self.compare(gradients[0].asArray(Float.self), reference.grad_a, "dL/dA")
        Self.compare(gradients[1].asArray(Float.self), reference.grad_b, "dL/dB")
    }

    /// A and B after each of three AdamW steps.
    ///
    /// Three is the smallest number that separates decoupled weight decay from L2: at step 1
    /// the bias correction dominates and the two forms are close, and they diverge as the
    /// moment estimates warm up.
    @Test("three AdamW steps follow the recorded trajectory")
    func optimiserTrajectoryMatches() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)

        // Hyperparameters from the fixture's own table, never defaulted: a mismatched weight
        // decay reads as a tuning difference rather than a bug.
        //
        // **`biasCorrection: true` is not optional here.** mlx-swift's Adam follows the
        // original paper and omits bias correction — its default is `false`. Left at that
        // default this trajectory diverges from the recorded one by 0.108 on the very first
        // step, and nothing else about the run looks wrong: the loss still falls, just along
        // a different path with a much larger effective early step. This test is what found
        // it.
        let optimizer = AdamW(learningRate: reference.optimizer.lr,
                              betas: (reference.optimizer.beta1, reference.optimizer.beta2),
                              eps: reference.optimizer.eps,
                              weightDecay: reference.optimizer.weight_decay,
                              biasCorrection: true)

        let lora = TrainableLoRA(keys: ["w"],
                                 factors: [TrainableLoRA.Factors(a: f.a, b: f.b)],
                                 rank: reference.shape.rank, alpha: reference.shape.alpha)
        let module = LoRAAdapterModule(lora)

        for (index, expected) in reference.trajectory.enumerated() {
            let valueAndGrad = MLXNN.valueAndGrad(model: module) {
                (model: LoRAAdapterModule, _: [MLXArray]) -> [MLXArray] in
                let pred = Self.predict(noisy: f.noisy, weight: f.weight,
                                        a: model.a[0], b: model.b[0], scaling: f.scaling)
                return [Self.loss(pred: pred, target: f.target, mask: f.mask,
                                  sequence: f.sequence, channels: f.channels).sum()]
            }
            let (losses, gradients) = valueAndGrad(module, [])
            MLX.eval(losses, gradients)

            // The loss reported is the pre-update one, which is what the fixture records.
            let value = losses[0].item(Float.self)
            #expect(abs(value - expected.loss) <= 2e-6 * max(abs(expected.loss), 1),
                    "step \(index) loss \(value) vs \(expected.loss)" as Comment)

            optimizer.update(model: module, gradients: gradients)
            MLX.eval(module, optimizer)

            // Tolerance loosens slightly across steps: each one compounds the previous one's
            // float32 error, and the trajectory is what is under test, not bit-equality.
            let tolerance = Float(2e-6) * Float(index + 1)
            Self.compare(module.a[0].asArray(Float.self), expected.a,
                         "step \(index) A", tolerance: tolerance)
            Self.compare(module.b[0].asArray(Float.self), expected.b,
                         "step \(index) B", tolerance: tolerance)
        }
    }

    /// The same three steps through ``ResumableAdamW``, against the same recorded table.
    ///
    /// The trainer runs on that optimiser rather than the library's, because the library's
    /// moments cannot be restored — its state storage and `AdamState`'s initialisers are
    /// both internal to `MLXOptimizers`, so a resumed run can read the old moments and not
    /// put them back. Owning the state means owning the arithmetic, and arithmetic that is
    /// owned has to be checked: this is the same trajectory the test above uses, so an error
    /// in the decoupled decay or in where `eps` lands fails here exactly as it would there.
    @Test("three ResumableAdamW steps follow the recorded trajectory")
    func resumableOptimiserTrajectoryMatches() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)

        let optimizer = ResumableAdamW(learningRate: reference.optimizer.lr,
                                       betas: (reference.optimizer.beta1,
                                               reference.optimizer.beta2),
                                       eps: reference.optimizer.eps,
                                       weightDecay: reference.optimizer.weight_decay,
                                       biasCorrection: true)

        let lora = TrainableLoRA(keys: ["w"],
                                 factors: [TrainableLoRA.Factors(a: f.a, b: f.b)],
                                 rank: reference.shape.rank, alpha: reference.shape.alpha)
        let module = LoRAAdapterModule(lora)

        for (index, expected) in reference.trajectory.enumerated() {
            let valueAndGrad = MLXNN.valueAndGrad(model: module) {
                (model: LoRAAdapterModule, _: [MLXArray]) -> [MLXArray] in
                let pred = Self.predict(noisy: f.noisy, weight: f.weight,
                                        a: model.a[0], b: model.b[0], scaling: f.scaling)
                return [Self.loss(pred: pred, target: f.target, mask: f.mask,
                                  sequence: f.sequence, channels: f.channels).sum()]
            }
            let (losses, gradients) = valueAndGrad(module, [])
            MLX.eval(losses, gradients)

            let value = losses[0].item(Float.self)
            #expect(abs(value - expected.loss) <= 2e-6 * max(abs(expected.loss), 1),
                    "step \(index) loss \(value) vs \(expected.loss)" as Comment)

            try optimizer.update(module: module, gradients: gradients)
            MLX.eval(module.parameters().flattenedValues() + optimizer.innerState())

            #expect(optimizer.step == index + 1)
            let tolerance = Float(2e-6) * Float(index + 1)
            Self.compare(module.a[0].asArray(Float.self), expected.a,
                         "step \(index) A", tolerance: tolerance)
            Self.compare(module.b[0].asArray(Float.self), expected.b,
                         "step \(index) B", tolerance: tolerance)
        }
    }

    /// And the direct equivalence: the two optimisers agree step for step on one problem.
    ///
    /// Stronger than the recorded table for the purpose it serves. That table has three steps
    /// at one set of hyperparameters; this runs both implementations side by side for longer
    /// and with a non-zero weight decay, so a divergence that only appears once the moments
    /// have warmed up — the kind a three-step table cannot see — shows up here.
    @Test("ResumableAdamW and the library's AdamW agree step for step")
    func agreesWithTheLibrary() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)

        // A fresh fixture per run, not a shared one. `Module.update(parameters:)` writes
        // through to the arrays the module was built from, so a second run over the same
        // `f.a` starts where the first finished — which presents as a constant offset from
        // step 0 (measured at 0.0807, exactly one run's worth of travel) and reads like an
        // arithmetic disagreement rather than a fixture that was reused.
        func run(_ update: (LoRAAdapterModule, ModuleParameters) throws -> Void) -> [[Float]] {
            let f = Self.fixture(reference)
            let lora = TrainableLoRA(keys: ["w"],
                                     factors: [TrainableLoRA.Factors(a: f.a, b: f.b)],
                                     rank: reference.shape.rank, alpha: reference.shape.alpha)
            let module = LoRAAdapterModule(lora)
            var out: [[Float]] = []
            for _ in 0 ..< 8 {
                let valueAndGrad = MLXNN.valueAndGrad(model: module) {
                    (model: LoRAAdapterModule, _: [MLXArray]) -> [MLXArray] in
                    let pred = Self.predict(noisy: f.noisy, weight: f.weight,
                                            a: model.a[0], b: model.b[0], scaling: f.scaling)
                    return [Self.loss(pred: pred, target: f.target, mask: f.mask,
                                      sequence: f.sequence, channels: f.channels).sum()]
                }
                let (losses, gradients) = valueAndGrad(module, [])
                MLX.eval(losses, gradients)
                try! update(module, gradients)
                MLX.eval(module.parameters().flattenedValues())
                out.append(module.a[0].asArray(Float.self))
            }
            return out
        }

        let library = AdamW(learningRate: 1e-2, weightDecay: 0.05, biasCorrection: true)
        let mine = ResumableAdamW(learningRate: 1e-2, weightDecay: 0.05, biasCorrection: true)
        let fromLibrary = run { module, gradients in
            library.update(model: module, gradients: gradients)
        }
        let fromMine = run { module, gradients in
            try mine.update(module: module, gradients: gradients)
        }

        #expect(fromLibrary.count == fromMine.count)
        for step in 0 ..< fromLibrary.count {
            Self.compare(fromMine[step], fromLibrary[step],
                         "step \(step) against the library", tolerance: 1e-6)
        }
    }

    /// The negative test. `scaling` is `alpha / rank = 2`, not 1, and the gradient scales with
    /// it — so an implementation that hardcoded 1 must not pass the test above.
    @Test("a hardcoded scaling of 1 does not reproduce the recorded gradients")
    func wrongScalingIsCaught() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)
        #expect(f.scaling != 1, "the fixture must not use a scaling of 1")

        let valueAndGrad = MLX.valueAndGrad({ (arrays: [MLXArray]) -> [MLXArray] in
            let pred = Self.predict(noisy: f.noisy, weight: f.weight,
                                    a: arrays[0], b: arrays[1], scaling: 1)
            return [Self.loss(pred: pred, target: f.target, mask: f.mask,
                              sequence: f.sequence, channels: f.channels).sum()]
        }, argumentNumbers: [0, 1])
        let (_, gradients) = valueAndGrad([f.a, f.b])
        MLX.eval(gradients)

        let got = gradients[0].asArray(Float.self)
        let worst = zip(got, reference.grad_a).map { abs($0 - $1) }.max() ?? 0
        let magnitude = reference.grad_a.map(abs).max() ?? 0
        #expect(worst > 1e-4 * magnitude,
                "scaling=1 reproduced the recorded gradient, so this check proves nothing"
                    as Comment)
    }

    /// The other negative: the mask's divisor varies per example, so dropping it changes the
    /// gradient. An implementation that divided by a constant token count would match
    /// neither.
    @Test("a constant divisor instead of the mask's mean does not reproduce the gradients")
    func constantDivisorIsCaught() throws {
        guard let reference = try Self.load() else { return }
        let f = Self.fixture(reference)

        let valueAndGrad = MLX.valueAndGrad({ (arrays: [MLXArray]) -> [MLXArray] in
            let pred = Self.predict(noisy: f.noisy, weight: f.weight,
                                    a: arrays[0], b: arrays[1], scaling: f.scaling)
            let error = pred - f.target
            let masked = (error * error) * f.mask
            // The plausible wrong implementation: divide by the token count.
            let wrong = masked.sum(axes: [1, 2]) / Float(f.sequence * f.channels)
            return [wrong.sum()]
        }, argumentNumbers: [0, 1])
        let (_, gradients) = valueAndGrad([f.a, f.b])
        MLX.eval(gradients)

        let got = gradients[1].asArray(Float.self)
        let worst = zip(got, reference.grad_b).map { abs($0 - $1) }.max() ?? 0
        let magnitude = reference.grad_b.map(abs).max() ?? 0
        #expect(worst > 1e-4 * magnitude,
                "a constant divisor reproduced the recorded gradient" as Comment)
    }
}

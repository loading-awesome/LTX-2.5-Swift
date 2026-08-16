// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import MLXNN
import Testing
@testable import LTXModules

/// The optimiser's own contract: it must survive a round trip through a file.
///
/// The step arithmetic itself, including agreement with `MLXOptimizers.AdamW`, belongs to
/// ``TrainingBackwardGateTests``. What is left here is the reason this type exists at all —
/// that its state can be written out and read back and produce the run that would have
/// happened anyway. A state that serialises *almost* exactly is the worst outcome available:
/// the resumed run trains, converges, and follows a path neither the interrupted run nor a
/// fresh one would have taken, and nothing reports it.
@Suite("Resumable AdamW", .serialized)
struct ResumableAdamWTests {

    /// Two modules built with the same arguments must be **identical**, which the shapes
    /// initialiser does not give: `TrainableLoRA(shapes:rank:alpha:)` draws `A` from the
    /// global stream, so two calls produce two different starting points and a comparison
    /// between them measures the initialisation rather than the optimiser.
    private func makeModule(targets: Int = 3, inFeatures: Int = 8, outFeatures: Int = 8,
                            rank: Int = 4) -> LoRAAdapterModule {
        let keys = (0 ..< targets).map { "blocks.\($0).attn1.to_q.weight" }
        let factors = (0 ..< targets).map { index -> TrainableLoRA.Factors in
            let parts = MLX.split(key: MLX.key(UInt64(index)), into: 2)
            return TrainableLoRA.Factors(
                a: MLX.normal([rank, inFeatures], key: parts[0]).asType(.float32),
                b: MLX.normal([outFeatures, rank], key: parts[1]).asType(.float32))
        }
        return LoRAAdapterModule(TrainableLoRA(keys: keys, factors: factors,
                                               rank: rank, alpha: rank))
    }

    /// A fixed, seeded gradient for a step, so two runs see identical inputs.
    private func gradients(for module: LoRAAdapterModule, step: Int) -> ModuleParameters {
        let pairs = module.parameters().flattened().enumerated().map {
            index, entry -> (String, MLXArray) in
            let key = MLX.key(UInt64(1000 * step + index))
            return (entry.0, MLX.normal(entry.1.shape, key: key).asType(entry.1.dtype))
        }
        return ModuleParameters.unflattened(pairs)
    }

    private func parameters(_ module: LoRAAdapterModule) -> [Float] {
        module.parameters().flattenedValues().flatMap { $0.asArray(Float.self) }
    }

    /// The property the whole type exists for.
    ///
    /// Eight steps uninterrupted against four, a save, a load, and four more. Bit equality,
    /// not a tolerance: every input is identical by construction, so anything other than
    /// equality means the state did not survive.
    @Test("a run resumed through a state dict continues the uninterrupted trajectory")
    func roundTripContinuesTheTrajectory() throws {
        func optimizer() -> ResumableAdamW {
            ResumableAdamW(learningRate: 1e-2, weightDecay: 0.05, biasCorrection: true)
        }

        // Uninterrupted.
        let whole = makeModule()
        let wholeOptimizer = optimizer()
        for step in 0 ..< 8 {
            try wholeOptimizer.update(module: whole, gradients: gradients(for: whole, step: step))
            MLX.eval(whole, wholeOptimizer)
        }

        // Interrupted after four, through a state dict.
        let part = makeModule()
        let first = optimizer()
        for step in 0 ..< 4 {
            try first.update(module: part, gradients: gradients(for: part, step: step))
            MLX.eval(part, first)
        }
        let saved = try first.stateDict(for: part)
        #expect(saved.count == part.parameters().flattened().count * 2)

        let second = ResumableAdamW(learningRate: 1e-2, weightDecay: 0.05,
                                    biasCorrection: true,
                                    step: first.step,
                                    moments: try ResumableAdamW.load(saved, for: part))
        for step in 4 ..< 8 {
            try second.update(module: part, gradients: gradients(for: part, step: step))
            MLX.eval(part, second)
        }

        #expect(second.step == wholeOptimizer.step)
        let got = parameters(part), want = parameters(whole)
        #expect(got.count == want.count)
        for i in 0 ..< min(got.count, want.count) {
            #expect(got[i].bitPattern == want[i].bitPattern,
                    "parameter \(i): \(got[i]) vs \(want[i])" as Comment)
        }
    }

    /// The step counter is state too, and the one most easily left behind.
    ///
    /// Restoring `m` and `v` with the counter at zero divides them by `1 - beta^1` — the
    /// largest bias correction there is — so the first resumed step is *bigger* than a fresh
    /// optimiser's would be, not smaller. This asserts the counter matters rather than
    /// assuming it, because a resume that restored only the moments would otherwise look
    /// like it had restored everything.
    @Test("restoring the moments without the step counter is not the same run")
    func stepCounterIsState() throws {
        let module = makeModule()
        let first = ResumableAdamW(learningRate: 1e-2, biasCorrection: true)
        for step in 0 ..< 4 {
            try first.update(module: module, gradients: gradients(for: module, step: step))
            MLX.eval(module, first)
        }
        let saved = try first.stateDict(for: module)
        let moments = try ResumableAdamW.load(saved, for: module)

        func continued(step: Int) -> [Float] {
            let clone = makeModule()
            let optimizer = ResumableAdamW(learningRate: 1e-2, biasCorrection: true,
                                           step: step, moments: moments)
            try! optimizer.update(module: clone, gradients: gradients(for: clone, step: 4))
            MLX.eval(clone, optimizer)
            return parameters(clone)
        }
        let correct = continued(step: first.step)
        let dropped = continued(step: 0)
        #expect(zip(correct, dropped).contains { $0 != $1 },
                "the step counter made no difference, so bias correction is not being applied")
    }

    /// The state is keyed by the adapter's own weight names, so a mismatch is refusable.
    @Test("a state dict from a different target set is refused, not partly applied")
    func mismatchedStateIsRefused() throws {
        let module = makeModule(targets: 3)
        let optimizer = ResumableAdamW(learningRate: 1e-2)
        try optimizer.update(module: module, gradients: gradients(for: module, step: 0))
        MLX.eval(module, optimizer)
        let saved = try optimizer.stateDict(for: module)

        // More targets than the state covers.
        #expect(throws: ResumableAdamW.Failure.self) {
            _ = try ResumableAdamW.load(saved, for: makeModule(targets: 4))
        }
        // Same targets, different shapes.
        #expect(throws: ResumableAdamW.Failure.self) {
            _ = try ResumableAdamW.load(saved, for: makeModule(targets: 3, inFeatures: 16))
        }
        // A state missing one tensor is refused rather than zero-filled.
        var partial = saved
        partial.removeValue(forKey: partial.keys.sorted()[0])
        #expect(throws: ResumableAdamW.Failure.self) {
            _ = try ResumableAdamW.load(partial, for: module)
        }
    }

    /// One counter for the adapter is only equivalent to the library's per-parameter counter
    /// while every parameter is stepped every time, so a gradient set that misses one is
    /// refused rather than silently bias-corrected against the wrong step number.
    @Test("a partial gradient set is refused")
    func partialGradientsRefused() throws {
        let module = makeModule(targets: 3)
        let optimizer = ResumableAdamW(learningRate: 1e-2)
        var pairs = module.parameters().flattened().map {
            ($0.0, MLXArray.zeros(like: $0.1))
        }
        pairs.removeLast()
        #expect(throws: ResumableAdamW.Failure.self) {
            try optimizer.update(module: module,
                                 gradients: ModuleParameters.unflattened(pairs))
        }
    }
}

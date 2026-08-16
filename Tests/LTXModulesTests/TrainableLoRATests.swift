// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing
@testable import LTXModules

/// The trainable LoRA factors, and that gradients actually reach them.
///
/// No checkpoint and no recorded fixture here — this is a property of the factors alone.
/// The analytic gradient MLX produces is compared to a central finite difference of the same
/// loss, which is the standard way to catch a transposed factor, a missing scaling, or a
/// parameter the graph does not actually depend on. A LoRA whose `B` gradient is silently
/// zero trains nothing and reports a falling loss from the `A` side alone.
@Suite("Trainable LoRA")
struct TrainableLoRATests {

    /// A small adapted linear: `y = x @ Wᵀ + delta(x)`, loss = mean(y²).
    ///
    /// Deliberately tiny so the finite difference is well conditioned and the suite stays
    /// fast; the property being checked does not depend on size.
    private func fixture() -> (x: MLXArray, w: MLXArray, lora: TrainableLoRA) {
        MLXRandom.seed(7)
        let inFeatures = 6, outFeatures = 4
        let x = MLXRandom.normal([2, 5, inFeatures]).asType(.float32)
        let w = MLXRandom.normal([outFeatures, inFeatures]).asType(.float32)
        let lora = TrainableLoRA(
            shapes: [(key: "blocks.0.attn1.to_q.weight",
                      inFeatures: inFeatures, outFeatures: outFeatures)],
            rank: 2, alpha: 4)
        return (x, w, lora)
    }

    private func loss(_ params: [MLXArray], x: MLXArray, w: MLXArray,
                      scaling: Float) -> MLXArray {
        let base = MLX.matmul(x, w.transposed())
        let delta = TrainableLoRA.delta(input: x, a: params[0], b: params[1], scaling: scaling)
        let y = base + delta
        return (y * y).mean()
    }

    // MARK: - Initialisation

    /// `B = 0` is what makes the adapter a no-op at step zero. If it were not, the first
    /// forward would already differ from the base model and every early-training
    /// observation would be measuring the initialisation.
    @Test("B initialises to exact zeros, so the adapter starts as a no-op")
    func bStartsAtZero() {
        let (x, _, lora) = fixture()
        let delta = lora.delta(input: x, at: 0)
        MLX.eval(delta)
        #expect(delta.max().item(Float.self) == 0)
        #expect(delta.min().item(Float.self) == 0)
    }

    /// Distinct storage per parameter. Not because `MLX.zeros` was found to alias — measured
    /// on this mlx-swift, two separately created zeros arrays receive distinct gradients —
    /// but because the property is cheap to hold and cheaper to rely on than to re-verify
    /// after a dependency bump.
    @Test("each B has its own storage")
    func parametersAreDistinctBuffers() {
        let lora = TrainableLoRA(
            shapes: (0 ..< 4).map { (key: "m\($0).weight", inFeatures: 6, outFeatures: 4) },
            rank: 2, alpha: 2)
        let params = lora.parameters()
        #expect(params.count == 8, "four pairs, interleaved")

        // Distinctness has to be observed, not asserted about pointers: mutate one B and
        // check the others did not move with it.
        var bs = [params[1], params[3], params[5], params[7]]
        bs[0] = bs[0] + 1
        MLX.eval(bs)
        #expect(bs[0].max().item(Float.self) == 1)
        for index in 1 ..< bs.count {
            #expect(bs[index].max().item(Float.self) == 0,
                    "B\(index) moved with B0 — the buffers alias")
        }
    }

    @Test("A initialises inside the expected Kaiming-uniform bound")
    func aInitBound() {
        let inFeatures = 6
        let lora = TrainableLoRA(
            shapes: [(key: "m.weight", inFeatures: inFeatures, outFeatures: 4)],
            rank: 8, alpha: 8)
        let a = lora.factors[0].a
        MLX.eval(a)
        // Kaiming-uniform(a = √5) on [rank, in] reduces to U(±1/√in).
        let bound = 1.0 / Float(inFeatures).squareRoot()
        #expect(a.max().item(Float.self) <= bound)
        #expect(a.min().item(Float.self) >= -bound)
        // And it is not degenerate — a constant would also satisfy the bound.
        #expect(a.max().item(Float.self) > a.min().item(Float.self))
    }

    @Test("scaling is alpha over rank, not a hardcoded 1")
    func scalingFollowsConfig() {
        #expect(TrainableLoRA(shapes: [], rank: 64, alpha: 64).scaling == 1.0)
        #expect(TrainableLoRA(shapes: [], rank: 32, alpha: 64).scaling == 2.0)
        #expect(TrainableLoRA(shapes: [], rank: 64, alpha: 32).scaling == 0.5)
    }

    // MARK: - Gradients

    /// The check that matters: analytic gradients against central finite differences.
    ///
    /// `B` starts at zero, which is the one point where a bug is easiest to hide — the delta
    /// is zero there, so an implementation that forgot to differentiate through `A` would
    /// still produce a plausible `B` gradient. So the factors are perturbed off their
    /// initialisation first.
    @Test("analytic gradients match central finite differences")
    func gradientsMatchFiniteDifference() {
        let (x, w, lora) = fixture()
        let scaling = lora.scaling

        // Move off the zero initialisation so both factors carry signal.
        var params = lora.parameters()
        params[1] = MLXRandom.normal(params[1].shape).asType(.float32) * 0.3
        MLX.eval(params)

        // `argumentNumbers` defaults to `[0]` — one gradient for the first array only.
        // Every parameter has to be named or the rest are silently not differentiated.
        let valueAndGrad = MLX.valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            [self.loss(p, x: x, w: w, scaling: scaling)]
        }, argumentNumbers: Array(params.indices))
        let (_, grads) = valueAndGrad(params)
        MLX.eval(grads)
        #expect(grads.count == params.count,
                "short gradient list — check argumentNumbers, which defaults to [0]")

        // Central differences on a sample of coordinates: every coordinate would be slow
        // and adds nothing, but both A and B must be represented.
        let eps: Float = 1e-3
        for (which, index) in [(0, 3), (0, 7), (1, 1), (1, 5)] {
            var flat = params.map { $0 }
            let shape = flat[which].shape
            var values = flat[which].asArray(Float.self)

            let original = values[index]
            values[index] = original + eps
            flat[which] = MLXArray(values, shape)
            let plus = loss(flat, x: x, w: w, scaling: scaling).item(Float.self)

            values[index] = original - eps
            flat[which] = MLXArray(values, shape)
            let minus = loss(flat, x: x, w: w, scaling: scaling).item(Float.self)

            let numeric = (plus - minus) / (2 * eps)
            let analytic = grads[which].asArray(Float.self)[index]
            let tolerance = max(2e-2, abs(numeric) * 2e-2)
            #expect(abs(numeric - analytic) <= tolerance,
                    "param \(which)[\(index)]: analytic \(analytic), numeric \(numeric)"
                        as Comment)
        }
    }

    /// A gradient that is exactly zero everywhere means the parameter is not in the graph —
    /// the failure mode that looks like "training is just slow".
    @Test("both factors receive non-zero gradients")
    func bothFactorsAreLive() {
        let (x, w, lora) = fixture()
        var params = lora.parameters()
        params[1] = MLXRandom.normal(params[1].shape).asType(.float32) * 0.3
        MLX.eval(params)

        let valueAndGrad = MLX.valueAndGrad({ (p: [MLXArray]) -> [MLXArray] in
            [self.loss(p, x: x, w: w, scaling: lora.scaling)]
        }, argumentNumbers: Array(params.indices))
        let (_, grads) = valueAndGrad(params)
        MLX.eval(grads)
        for (index, grad) in grads.enumerated() {
            let magnitude = MLX.abs(grad).max().item(Float.self)
            #expect(magnitude > 0, "parameter \(index) has an all-zero gradient")
        }
    }

    // MARK: - Round trip

    /// An adapter trained here has to load back through the inference path, or the two
    /// halves of the port are describing different files.
    @Test("the state dict uses the key form LoRAOverlay resolves")
    func stateDictKeys() {
        let lora = TrainableLoRA(
            shapes: [(key: "transformer_blocks.0.attn1.to_q.weight",
                      inFeatures: 6, outFeatures: 4)],
            rank: 2, alpha: 2)
        let state = lora.stateDict()
        #expect(state.keys.sorted() == [
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
        ])
    }

    @Test("an already-prefixed key is not prefixed twice")
    func stateDictDoesNotDoublePrefix() {
        let lora = TrainableLoRA(
            shapes: [(key: "diffusion_model.blocks.0.attn1.to_q.weight",
                      inFeatures: 6, outFeatures: 4)],
            rank: 2, alpha: 2)
        #expect(lora.stateDict().keys.allSatisfy {
            !$0.contains("diffusion_model.diffusion_model.")
        })
    }

    /// The form `DiTLoRATargets.discover` actually produces.
    ///
    /// The two tests above use a bare key and a `diffusion_model.`-prefixed one — neither of
    /// which a real run ever sees. Every key from the checkpoint carries the full prefix
    /// `model.diffusion_model.`, and against that the old implementation prepended rather than
    /// normalised, producing `diffusion_model.model.diffusion_model.…`. `LoRAOverlay.load`
    /// then added the base prefix back and resolved to a weight no checkpoint has.
    ///
    /// Nothing failed at save time. The file was well formed and the shapes were right; the
    /// adapter was simply unloadable, and the tests above passed throughout — they compared
    /// the implementation against itself on inputs it never receives.
    @Test("a checkpoint key is normalised to the released adapters' convention")
    func stateDictNormalisesTheComfyPrefix() {
        let lora = TrainableLoRA(
            shapes: [(key: "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
                      inFeatures: 6, outFeatures: 4)],
            rank: 2, alpha: 2)
        // Exactly what `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` names this module.
        #expect(lora.stateDict().keys.sorted() == [
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
        ])
    }

    /// Whichever form goes in, one `diffusion_model.` comes out and no `model.` survives.
    @Test("every key form lands on one prefix")
    func everyKeyFormNormalises() {
        for key in ["model.diffusion_model.blocks.0.attn1.to_q.weight",
                    "diffusion_model.blocks.0.attn1.to_q.weight",
                    "blocks.0.attn1.to_q.weight"] {
            let state = TrainableLoRA(
                shapes: [(key: key, inFeatures: 4, outFeatures: 4)],
                rank: 2, alpha: 2).stateDict()
            for name in state.keys {
                #expect(name.hasPrefix("diffusion_model."), "\(key) -> \(name)" as Comment)
                #expect(!name.contains("model.diffusion_model."),
                        "\(key) -> \(name)" as Comment)
                #expect(name.hasSuffix("blocks.0.attn1.to_q.lora_A.weight")
                    || name.hasSuffix("blocks.0.attn1.to_q.lora_B.weight"),
                        "\(key) -> \(name)" as Comment)
            }
        }
    }
}

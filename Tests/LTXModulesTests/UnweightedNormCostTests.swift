// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
import MLX

/// D7: is `MLXArray.ones([dim])` on every unweighted RMSNorm worth caching?
///
/// `MLX.rmsNorm` requires a weight. Isolation at production AdaLN shape
/// (48 blocks × 10 norms, `[1, 3120, 4096]` bf16, eval per block) measured
/// **5.6% of the rmsNorm stack** for a cached ones vector — 0.043 s vs 0.041 s.
/// That stack is tens of milliseconds against seconds per DiT pass, so the
/// render-level gain is far below the 1.5% claim floor. Do not cache ones
/// without a new measurement that moves `meanFullStepSeconds`.
@Suite("unweighted rmsNorm ones() cost")
struct UnweightedNormCostTests {

    @Test("cached ones is bit-identical to ones-per-call")
    func cachedOnesBitIdentical() {
        let x = MLXArray.ones([1, 32, 64], dtype: .bfloat16)
        let fresh = MLX.rmsNorm(x, weight: MLXArray.ones([64], dtype: .bfloat16), eps: 1e-6)
        let cached = MLXArray.ones([64], dtype: .bfloat16)
        let reused = MLX.rmsNorm(x, weight: cached, eps: 1e-6)
        MLX.eval(fresh, reused)
        let delta = MLX.max(MLX.abs(fresh.asType(.float32) - reused.asType(.float32)))
        #expect(delta.item(Float.self) == 0)
    }

    @Test("ones-per-call vs cached weight at production AdaLN shape")
    func measureOnesCost() {
        // One video AdaLN: 640×384×97 → 13 latent frames × 12 × 20 = 3120 tokens, dim 4096.
        // 10 unweighted norms per block × 48 blocks = 480 per forward. Time that many.
        let tokens = 3120
        let dim = 4096
        let x = MLXArray.ones([1, tokens, dim], dtype: .bfloat16)
        MLX.eval(x)

        // Live path evals once per block, and a block has 10 unweighted norms.
        // Time 48 such graphs — one forward — with and without a cached weight.
        func timeBlockGraphs(cached: Bool) -> Double {
            let w = cached ? { let u = MLXArray.ones([dim], dtype: .bfloat16); MLX.eval(u); return u }() : nil
            let start = Date()
            for _ in 0..<48 {
                var h = x
                for _ in 0..<10 {
                    let weight = w ?? MLXArray.ones([dim], dtype: .bfloat16)
                    h = MLX.rmsNorm(h, weight: weight, eps: 1e-6)
                }
                MLX.eval(h)
            }
            return Date().timeIntervalSince(start)
        }

        _ = timeBlockGraphs(cached: false)
        _ = timeBlockGraphs(cached: true)
        let fresh = timeBlockGraphs(cached: false)
        let cached = timeBlockGraphs(cached: true)
        let gain = (fresh - cached) / fresh
        print(String(
            format: "D7 48×(10 rmsNorm) at [%d,%d] bf16: fresh %.3fs cached %.3fs gain %.2f%% (1.5%% floor)",
            tokens, dim, fresh, cached, gain * 100))
        #expect(fresh > 0 && cached > 0)
    }
}

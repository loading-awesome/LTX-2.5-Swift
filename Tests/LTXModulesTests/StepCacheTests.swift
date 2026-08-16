// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
import MLX
@testable import LTXModules

@Suite("step cache measurement")
struct StepCacheTests {

    @Test("relative L1 is mean-abs, not RMS")
    func relativeL1() {
        // One loud element in a quiet tensor: RMS would hide it, mean-abs would not.
        let prev = MLXArray([Float](repeating: 1, count: 8), [1, 8, 1])
        var cur = [Float](repeating: 1, count: 8)
        cur[0] = 3
        let now = MLXArray(cur, [1, 8, 1])
        // mean(|delta|) = 2/8 = 0.25; mean(|prev|) = 1; ratio = 0.25
        let got = StepCache.relativeChange(now, prev)
        #expect(abs(got - 0.25) < 1e-6)
    }

    @Test("whole-sequence change weights streams by element count")
    func wholeSequenceWeightsByCount() {
        let pv = MLXArray([Float](repeating: 1, count: 8), [1, 8, 1])
        let pa = MLXArray([Float](repeating: 1, count: 2), [1, 2, 1])
        var cv = [Float](repeating: 1, count: 8)
        cv[0] = 2
        let ca = MLXArray([Float](repeating: 1, count: 2), [1, 2, 1])
        // num = 8*(1/8) + 2*0 = 1; den = 8*1 + 2*1 = 10; ratio = 0.1
        let got = StepCache.wholeSequenceChange(
            currentVideo: MLXArray(cv, [1, 8, 1]), currentAudio: ca,
            previousVideo: pv, previousAudio: pa)
        #expect(abs(got - 0.1) < 1e-6)
    }

    @Test("record evals the residual; reuse is h_in + buffer")
    func recordEvalsResidual() {
        let cache = StepCache(pass: "cond", threshold: 0.10)
        let hIn = MLXArray([Float](repeating: 2, count: 8), [1, 8, 1])
        let hOut = MLXArray([Float](repeating: 5, count: 8), [1, 8, 1])
        MLX.eval(hIn, hOut)
        cache.record(videoResidual: hOut - hIn, audioResidual: hOut - hIn)

        let nextIn = MLXArray([Float](repeating: 7, count: 8), [1, 8, 1])
        let reused = nextIn + cache.cachedVideo!
        MLX.eval(reused)
        #expect(reused[0, 0, 0].item(Float.self) == 10)
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import MLX
import Testing

/// A smoke test, not a formality. MLX aborts the whole test process when it cannot
/// load its Metal library, so "does an MLX op evaluate at all" has to be asserted
/// before any MLX-based test is worth writing.
@Suite("LTXModules")
struct LTXModulesSmokeTests {

    @Test("MLX evaluates a trivial op on the default device")
    func mlxRuns() {
        let a = MLXArray([1.0, 2.0, 3.0] as [Float])
        let b = MLXArray([4.0, 5.0, 6.0] as [Float])
        let dot = MLX.sum(a * b)
        #expect(dot.item(Float.self) == 32.0)
    }

    @Test("a matmul the size of one projection row evaluates")
    func mlxMatmul() {
        // Small but not trivial: exercises the GPU kernel path rather than a
        // scalar fast path.
        let x = MLXArray.ones([8, 256])
        let w = MLXArray.ones([256, 64])
        let y = MLX.matmul(x, w)
        #expect(y.shape == [8, 64])
        #expect(y[0, 0].item(Float.self) == 256.0)
    }
}

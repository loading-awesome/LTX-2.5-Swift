// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXFast
import Testing

/// Does MLX actually work here, for the operations this port needs?
///
/// Not a formality. `swift build` does not compile `.metal` files — see
/// `tools/build_mlx_metallib.sh` for the full diagnosis — so without that script MLX
/// aborts the entire test process at device resolution, taking every other suite with
/// it. These tests are the tripwire for that, and they are deliberately more than a
/// smoke test:
///
/// Only nine kernels are precompiled into `mlx.metallib`. Everything else — binary
/// and unary ops, reductions, softmax, the gemm kernels — is **JIT-compiled at
/// runtime** from source strings embedded in the library. So "a trivial op evaluated"
/// proves the metallib loaded; it does not prove the JIT path works. Each test below
/// exercises a different kernel family so a partial failure is visible as a partial
/// failure.
@Suite("MLX capability")
struct MLXCapabilityTests {

    @Test("the metallib loads and a precompiled kernel runs")
    func metallibLoads() {
        // RMS norm is one of the nine precompiled kernels.
        let x = MLXArray.ones([4, 128])
        let y = MLXFast.rmsNorm(x, weight: MLXArray.ones([128]), eps: 1e-6)
        #expect(y.shape == [4, 128])
        #expect(abs(y[0, 0].item(Float.self) - 1.0) < 1e-5)
    }

    @Test("JIT-compiled kernels work: binary, unary, reduction")
    func jitKernels() {
        let a = MLXArray(0..<1024).asType(.float32).reshaped([32, 32])
        #expect(MLX.sum(a).item(Float.self) == 523776.0)          // reduction
        #expect(MLX.max(a).item(Float.self) == 1023.0)
        let b = MLX.sqrt(a + 1.0)                                  // unary + binary
        #expect(abs(b[0, 0].item(Float.self) - 1.0) < 1e-6)
        #expect(abs(b[31, 31].item(Float.self) - 32.0) < 1e-4)
    }

    @Test("softmax and matmul, the two the DiT leans on hardest")
    func softmaxAndMatmul() {
        let logits = MLXArray([1.0, 2.0, 3.0] as [Float]).reshaped([1, 3])
        let p = MLX.softmax(logits, axis: -1)
        #expect(abs(MLX.sum(p).item(Float.self) - 1.0) < 1e-6)
        #expect(p[0, 2].item(Float.self) > p[0, 0].item(Float.self))

        // 4096-wide, the video stream's inner dimension.
        let x = MLXArray.ones([64, 4096])
        let w = MLXArray.ones([4096, 128]) * MLXArray(Float(0.5))
        let y = MLX.matmul(x, w)
        #expect(y.shape == [64, 128])
        #expect(abs(y[0, 0].item(Float.self) - 2048.0) < 1e-2)
    }

    /// The checkpoints are bf16, so the arithmetic has to be. A dtype that silently
    /// upcasts would hide precision loss the real weights carry.
    @Test("bfloat16 arithmetic runs and rounds like bf16")
    func bfloat16() {
        let x = MLXArray([1.0, 1.0009765625, 3.14159265] as [Float]).asType(.bfloat16)
        let back = x.asType(.float32)
        // bf16 has 8 mantissa bits: 1.0009765625 is not representable and rounds to 1.
        #expect(back[0].item(Float.self) == 1.0)
        #expect(back[1].item(Float.self) == 1.0)
        // pi rounds to 3.140625 in bf16.
        #expect(abs(back[2].item(Float.self) - 3.140625) < 1e-6)

        let y = MLX.matmul(MLXArray.ones([8, 64]).asType(.bfloat16),
                           MLXArray.ones([64, 8]).asType(.bfloat16))
        #expect(y.dtype == .bfloat16)
        #expect(y[0, 0].item(Float.self) == 64.0)
    }

    /// Scaled dot-product attention, via the fused path the DiT will use.
    @Test("fused scaled dot-product attention evaluates")
    func fusedAttention() {
        let heads = 4, tokens = 16, headDim = 64
        let q = MLXArray.ones([1, heads, tokens, headDim])
        let k = MLXArray.ones([1, heads, tokens, headDim])
        let v = MLXArray.ones([1, heads, tokens, headDim]) * MLXArray(Float(2.0))
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / Float(headDim).squareRoot(),
            mask: .none)
        #expect(out.shape == [1, heads, tokens, headDim])
        // All values identical, so attention is uniform and the output is just v.
        #expect(abs(out[0, 0, 0, 0].item(Float.self) - 2.0) < 1e-4)
    }

    @Test("safetensors round-trips through MLX")
    func safetensorsRoundTrip() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlx-capability-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("t.safetensors")
        let original = MLXArray(0..<24).asType(.float32).reshaped([2, 3, 4])
        try MLX.save(arrays: ["t": original], url: url)

        let loaded = try MLX.loadArrays(url: url)
        let t = try #require(loaded["t"])
        #expect(t.shape == [2, 3, 4])
        #expect(MLX.sum(MLX.abs(t - original)).item(Float.self) == 0.0)
    }
}

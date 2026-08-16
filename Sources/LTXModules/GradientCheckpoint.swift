// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Cmlx
import Foundation
import MLX

/// Gradient checkpointing: trade recomputation for activation memory.
///
/// Without this, every intermediate a traced function produces stays resident until the
/// backward pass consumes it. Through 48 transformer blocks that is the dominant cost of a
/// training step — measured at ~102 MB per video token, against ~39 GB of weights that do not
/// grow with sequence length (`docs/TRAINING.md`). `checkpoint` drops the intermediates at the
/// forward's end and recomputes them per-segment during the backward, so peak memory follows
/// the largest single segment rather than their sum.
///
/// ## Why this exists rather than `MLX.checkpoint`
///
/// mlx-swift does not surface it. `mlx_checkpoint` is nonetheless present in the vendored
/// mlx-c (`Source/Cmlx/include/mlx/c/transforms.h`), so this is a binding, not a
/// reimplementation — the transform is mlx's own, and its VJP is the one mlx tests.
///
/// ## The rule that makes it silent when wrong
///
/// **Only arrays passed in `inputs` receive gradients.** Anything `body` captures is a
/// constant to the transform. MLX Python's `nn.checkpoint` is built around this: it passes
/// `module.trainable_parameters()` as the *first argument* and calls `module.update(params)`
/// inside, precisely so the parameters are inputs rather than captures.
///
/// Capture a traced parameter instead of passing it and the gradient comes back structurally
/// zero, with no error anywhere. ``checkpointTests`` pins this with a negative case, because
/// a zero gradient is indistinguishable from a converged one at a glance.
public enum GradientCheckpoint {

    /// Run `body` on `inputs` under checkpointing, recomputing its intermediates during the
    /// backward pass instead of holding them.
    ///
    /// - Parameters:
    ///   - inputs: every array the result should be differentiable with respect to. Arrays
    ///     `body` captures instead are constants — see the type's discussion.
    ///   - body: the segment to checkpoint.
    ///
    /// Both closure handles are released before returning. The graph keeps its own copy of the
    /// function for the backward pass (mlx's `checkpoint` captures it by value into the
    /// `custom_vjp` node), which is why freeing here is safe — and necessary. Holding them
    /// instead would accumulate one retained closure, with its captured arrays, per call: over
    /// a training run that is precisely the leak this transform exists to avoid.
    public static func apply(
        _ inputs: [MLXArray],
        _ body: @escaping ([MLXArray]) -> [MLXArray]
    ) -> [MLXArray] {
        let inner = newClosure(body)
        defer { mlx_closure_free(inner) }

        var checkpointed = mlx_closure_new()
        defer { mlx_closure_free(checkpointed) }
        mlx_checkpoint(&checkpointed, inner)

        let arguments = newVector(inputs)
        defer { mlx_vector_array_free(arguments) }

        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }

        mlx_closure_apply(&result, checkpointed, arguments)
        return values(result)
    }

    /// ``apply(_:_:)`` in function form, for composing into a transform.
    public static func wrap(
        _ body: @escaping ([MLXArray]) -> [MLXArray]
    ) -> ([MLXArray]) -> [MLXArray] {
        { inputs in apply(inputs, body) }
    }
}

// MARK: - Cmlx glue
//
// mlx-swift keeps its equivalents (`new_mlx_closure`, `new_mlx_vector_array`,
// `mlx_vector_array_values`) internal to the MLX module, so they are mirrored here. These
// follow the originals line for line — the ownership conventions are the part that must not
// be improvised, and matching a working implementation is the cheapest way to be sure.

/// Holds the Swift closure alive as the mlx_closure's capture state.
private final class ClosureCaptureState {
    let body: ([MLXArray]) -> [MLXArray]
    init(_ body: @escaping ([MLXArray]) -> [MLXArray]) { self.body = body }
}

private func releaseState(_ pointer: UnsafeMutableRawPointer?) {
    Unmanaged<ClosureCaptureState>.fromOpaque(pointer!).release()
}

private func trampoline(
    resultOut: UnsafeMutablePointer<mlx_vector_array>?,
    vector: mlx_vector_array,
    payload: UnsafeMutableRawPointer?
) -> Int32 {
    let state = Unmanaged<ClosureCaptureState>.fromOpaque(payload!).takeUnretainedValue()
    let result = state.body(values(vector))
    guard let resultOut else { fatalError("mlx_closure called without a result pointer") }
    resultOut.pointee = newVector(result)
    return 0
}

private func newClosure(_ body: @escaping ([MLXArray]) -> [MLXArray]) -> mlx_closure {
    let payload = Unmanaged.passRetained(ClosureCaptureState(body)).toOpaque()
    return mlx_closure_new_func_payload(trampoline, payload, releaseState)
}

/// A +1 `mlx_vector_array` holding `arrays`.
private func newVector(_ arrays: some Collection<MLXArray>) -> mlx_vector_array {
    withExtendedLifetime(Array(arrays)) { held in
        mlx_vector_array_new_data(held.map { $0.ctx }, held.count)
    }
}

private func values(_ vector: mlx_vector_array) -> [MLXArray] {
    (0 ..< mlx_vector_array_size(vector)).map { index in
        // +1 object; MLXArray takes ownership.
        var ctx = mlx_array_new()
        mlx_vector_array_get(&ctx, vector, index)
        return MLXArray(ctx)
    }
}

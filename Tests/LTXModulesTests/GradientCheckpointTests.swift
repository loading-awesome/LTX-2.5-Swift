// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXRandom
import Testing
@testable import LTXModules

/// ``GradientCheckpoint`` is a binding onto mlx's own `checkpoint`, so what needs proving is
/// not the transform's calculus but the plumbing around it: that the closure trampoline hands
/// back the right arrays, that gradients survive the round trip unchanged, and that the memory
/// actually drops. All of it runs on small synthetic tensors in under a second — no checkpoint,
/// no GPU-hours.
///
/// The negative test is the important one. Checkpointing has a failure mode that raises no
/// error and produces no NaN: capture a parameter instead of passing it and its gradient is
/// silently zero.
@Suite("Gradient checkpoint", .serialized)
struct GradientCheckpointTests {

    /// A chain deep enough that intermediates dominate, which is the regime checkpointing is
    /// for. `tanh` keeps the values bounded so a 32-deep chain stays finite.
    private static func chain(depth: Int) -> ([MLXArray]) -> [MLXArray] {
        { inputs in
            let x = inputs[0], w = inputs[1]
            var h = x
            for _ in 0 ..< depth { h = MLX.tanh(h.matmul(w)) }
            return [h.square().sum()]
        }
    }

    private static func inputs(width: Int, seed: UInt64) -> [MLXArray] {
        MLXRandom.seed(seed)
        // Scaled so the chain neither saturates tanh nor collapses to zero — a chain that
        // saturates would compare two gradients that are both ~0 and pass regardless.
        let x = MLXRandom.normal([width, width]) * 0.1
        let w = MLXRandom.normal([width, width]) * (1.0 / Float(width).squareRoot())
        return [x, w]
    }

    @Test("the checkpointed value equals the plain value")
    func valueIsUnchanged() {
        let f = Self.chain(depth: 8)
        let arguments = Self.inputs(width: 64, seed: 7)

        let plain = f(arguments)[0]
        let wrapped = GradientCheckpoint.wrap(f)(arguments)[0]

        let difference = MLX.abs(plain - wrapped).item(Float.self)
        #expect(plain.item(Float.self).isFinite)
        // Non-trivial: a chain that collapsed to zero would satisfy any comparison below.
        #expect(MLX.abs(plain).item(Float.self) > 1e-3, "the probe collapsed; it proves nothing")
        #expect(difference == 0, "\(difference) — recomputation runs the identical graph")
    }

    @Test("gradients through the checkpoint match the uncheckpointed gradients")
    func gradientsMatch() {
        let f = Self.chain(depth: 8)
        let arguments = Self.inputs(width: 64, seed: 11)

        let plain = MLX.valueAndGrad(f, argumentNumbers: [0, 1])(arguments)
        let wrapped = MLX.valueAndGrad(
            GradientCheckpoint.wrap(f), argumentNumbers: [0, 1])(arguments)

        #expect(plain.1.count == 2)
        #expect(wrapped.1.count == 2)

        for (index, (a, b)) in zip(plain.1, wrapped.1).enumerated() {
            let magnitude = MLX.abs(a).max().item(Float.self)
            // Each gradient must actually carry signal, or "they match" is vacuous.
            #expect(magnitude > 1e-6, "argument \(index) gradient is ~0 before comparison")

            let error = MLX.abs(a - b).max().item(Float.self)
            #expect(error <= 1e-6 * max(magnitude, 1),
                    "argument \(index): max |Δ| \(error) against magnitude \(magnitude)"
                        as Comment)
        }
    }

    /// The silent failure, pinned. `body` ignores its second argument and uses a captured `w`
    /// instead — exactly the mistake that is easy to make when refactoring a forward pass into
    /// a checkpointed segment. Nothing throws; the gradient is simply zero.
    @Test("a captured parameter gets no gradient, and that failure is silent")
    func capturedParameterIsNotDifferentiated() {
        let arguments = Self.inputs(width: 64, seed: 13)
        let captured = arguments[1]

        let capturing: ([MLXArray]) -> [MLXArray] = { inputs in
            var h = inputs[0]
            for _ in 0 ..< 8 { h = MLX.tanh(h.matmul(captured)) }  // <- capture, not input
            return [h.square().sum()]
        }

        let (value, gradients) = MLX.valueAndGrad(
            GradientCheckpoint.wrap(capturing), argumentNumbers: [0, 1])(arguments)

        // The forward is perfectly healthy — this is why the failure is hard to spot.
        #expect(value[0].item(Float.self).isFinite)
        #expect(MLX.abs(gradients[0]).max().item(Float.self) > 1e-6, "the passed input is fine")

        #expect(MLX.abs(gradients[1]).max().item(Float.self) == 0,
                "captured arrays are constants to the transform")

        // And passing the same array instead of capturing it recovers the gradient.
        let passing = Self.chain(depth: 8)
        let recovered = MLX.valueAndGrad(
            GradientCheckpoint.wrap(passing), argumentNumbers: [0, 1])(arguments).1
        #expect(MLX.abs(recovered[1]).max().item(Float.self) > 1e-6)
    }

    /// A stack of `depth` steps, each optionally checkpointed on its own.
    ///
    /// The weight is **threaded through** the step's arguments rather than captured, which is
    /// the whole discipline this transform imposes — see ``capturedParameterIsNotDifferentiated``.
    private static func stack(depth: Int, checkpointed: Bool) -> ([MLXArray]) -> [MLXArray] {
        let step: ([MLXArray]) -> [MLXArray] = { io in
            [MLX.tanh(io[0].matmul(io[1])), io[1]]
        }
        let applied = checkpointed ? GradientCheckpoint.wrap(step) : step
        return { inputs in
            var carried = inputs
            for _ in 0 ..< depth { carried = applied(carried) }
            return [carried[0].square().sum()]
        }
    }

    /// The point of the exercise — and the shape it has to take.
    ///
    /// Checkpointing must be applied **per segment**, not once around the whole computation.
    /// One wrap around everything saves nothing: the backward recomputes the entire forward in
    /// a single trace, so peak is the same or slightly worse. Wrapping each step keeps only
    /// one step's intermediates live at a time, which is the saving. mlx's own
    /// `nn.TransformerEncoder` does exactly this (`l = checkpoint(l)` per layer), and this test
    /// exists so that fact is discovered here rather than during a training run.
    @Test("per-segment checkpointing drops peak memory through the backward")
    func peakMemoryDrops() {
        let depth = 32, width = 1024
        let arguments = Self.inputs(width: width, seed: 17)
        MLX.eval(arguments)

        func peak(_ g: @escaping ([MLXArray]) -> [MLXArray]) -> Double {
            MLX.GPU.resetPeakMemory()
            let (value, gradients) = MLX.valueAndGrad(g, argumentNumbers: [0, 1])(arguments)
            MLX.eval(value, gradients)
            return Double(MLX.GPU.peakMemory) / 1e6
        }

        let plain = peak(Self.stack(depth: depth, checkpointed: false))
        let segmented = peak(Self.stack(depth: depth, checkpointed: true))
        // Recorded for contrast: this is the arrangement that does *not* help.
        let wholeChain = peak(GradientCheckpoint.wrap(Self.chain(depth: depth)))

        print(String(format:
            "PEAK plain %.1f MB, per-segment %.1f MB (%.0f%% saved), whole-chain %.1f MB",
            plain, segmented, 100 * (1 - segmented / plain), wholeChain))

        // `MLX.GPU.peakMemory` is a **process** counter, not this computation's.
        //
        // `resetPeakMemory()` zeroes it globally, and every other suite allocating on the
        // GPU at the same moment raises it. Run alone this stack peaks at ~690 MB and the
        // saving is ~19%; run inside the full suite the same three readings come back at
        // 85874 / 86268 / 86925 MB — the 42 GB transformer another suite is holding, with
        // this test's 130 MB of signal buried three orders of magnitude beneath it. The
        // comparison then decides on which reading happened to catch a larger allocation,
        // which is a coin flip: measured across three full-suite runs it came out
        // 77160 > 77030 (pass), 85874 < 86268 (fail), 103431 < 104586 (fail). Contamination
        // is not all-or-nothing either — one reading can be clean and the next not.
        //
        // So the ceiling is a validity condition on the instrument, not a memory budget.
        // Above it the number is somebody else's and nothing can be concluded from it.
        //
        // Returning early rather than failing, which is this repository's existing treatment
        // of an unavailable instrument — `guard let l = try Self.load() else { return }` is
        // the same shape wherever a fixture is absent, and `docs/SWIFT_ARCHITECTURE.md`
        // records it as a known sharp edge: the framework counts it as a pass, so a test
        // count only means something alongside where it ran. Failing instead would be
        // *permanently* red in every full-suite run — this reading has never been valid
        // there, and asserting on it would report a checkpointing regression that is really
        // another suite's transformer. There is no trait that fixes it either:
        // `.serialized` orders a suite's own tests and the contamination is cross-suite.
        //
        // The consequence is stated plainly because it is a real loss of coverage: **the
        // central claim of `GradientCheckpoint` is only measured when this suite is run on
        // its own.** Do that before trusting a change to it.
        // The ceiling applies to **every** reading, and it is set just above what this
        // stack actually costs rather than at some round large number.
        //
        // Both halves of that matter. Checking `plain` alone lets a run through where
        // `plain` was taken in a quiet moment and `segmented` caught another suite's
        // allocation; and a loose ceiling lets through readings that are contaminated but
        // not yet enormous — 4753 / 5461 MB slipped under an 8 GB ceiling and failed, on
        // a stack whose honest readings are 688 / 558 / 814. Contamination is not
        // all-or-nothing, and it does not have to be large to be fatal: it only has to be
        // larger than the 130 MB of signal being measured.
        //
        // 2 GB is therefore the bound — more than twice the honest worst reading, and far
        // below anything another suite holding a model produces.
        let ceiling = 2_000.0
        let worst = max(plain, max(segmented, wholeChain))
        guard worst < ceiling else {
            print("SKIP peak memory: readings were \(Int(plain)) / \(Int(segmented)) / "
                + "\(Int(wholeChain)) MB for a \(depth)x\(width) stack that needs ~690 MB "
                + "alone. `MLX.GPU.peakMemory` is process-global, so this is another "
                + "suite's allocation and the plain-vs-segmented comparison would be "
                + "meaningless. Run `swift test --filter GradientCheckpointTests` on its "
                + "own to measure it — that is the only configuration in which this test "
                + "tests anything.")
            return
        }

        #expect(segmented < plain,
                "per-segment \(segmented) MB vs plain \(plain) MB" as Comment)
    }
}

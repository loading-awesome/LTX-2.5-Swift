// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Runs work on a thread with a stack big enough for mlx's recursive graph traversal.
///
/// `mlx::core::vjp` walks the graph by recursion, one frame per node. That is fine for a
/// single backward pass. It stops being fine under gradient checkpointing, because
/// `checkpoint`'s VJP is *itself* a call to `vjp` — so a 48-block forward with a checkpoint
/// per block produces 48 nested traversals rather than one.
///
/// The stack that has to hold all of it depends on where the work was scheduled, and the
/// unlucky cases are the common ones:
///
/// | thread | stack |
/// |---|---|
/// | main | 8 MB |
/// | `Thread` (default) | 512 KB |
/// | Swift concurrency cooperative pool | 512 KB |
///
/// A `swift-testing` test body, and anything inside a `Task`, runs on the cooperative pool.
/// Past roughly 16 checkpointed blocks that overflows, and the failure is
/// `EXC_BAD_ACCESS (SIGBUS)` in the guard page with a backtrace that is a thousand frames of
/// `std::function` templates and no mention of anything in this package — measured here, not
/// inferred. Nothing reports it as a stack overflow, so it is worth knowing on sight.
public enum DeepStack {

    /// 256 MB. Empirically 48 nested traversals need far less, but the cost of a too-large
    /// stack is address space rather than memory — the pages are never touched — while the
    /// cost of a too-small one is an unattributable crash.
    public static let defaultStackSize = 256 << 20

    /// Runs `body` on a dedicated thread with `stackSize` bytes of stack and returns its
    /// result, blocking the caller until it finishes.
    ///
    /// The thread exists only for its stack — there is no concurrency here. The caller is
    /// blocked from start to finish, and the semaphore orders the handoff, so only one thread
    /// ever touches the box: that is what makes the unchecked-`Sendable` sound.
    ///
    /// `body` must be `@escaping`, and not merely as a formality. `Thread` retains its block
    /// beyond the block's own execution, so a `withoutActuallyEscaping` wrapper here trips
    /// "non-escaping closure has escaped" at runtime even though the work has demonstrably
    /// finished — measured, not assumed.
    ///
    /// Deliberately non-throwing. `rethrows` cannot express this: Swift attributes a
    /// function's throwing to calls on its own parameters, and the error would surface from
    /// the box instead. A throwing body should catch inside and return its own `Result`.
    public static func run<T>(stackSize: Int = defaultStackSize,
                              _ body: @escaping () -> T) -> T {
        let work = Work(body)
        let finished = DispatchSemaphore(value: 0)

        let thread = Thread {
            work.run()
            finished.signal()
        }
        thread.stackSize = stackSize
        thread.start()
        finished.wait()

        // `run` blocks until the thread signals, so the value is always set by here.
        return work.value!
    }

    /// Carries the closure over to the thread and the value back. The semaphore is the
    /// synchronisation; this is only a box.
    private final class Work<T>: @unchecked Sendable {
        private let body: () -> T
        var value: T?

        init(_ body: @escaping () -> T) { self.body = body }
        func run() { value = body() }
    }
}

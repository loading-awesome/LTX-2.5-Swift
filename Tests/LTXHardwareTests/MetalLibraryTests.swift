// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXHardware

/// The environment probes `doctor` leads with.
///
/// The GEMM patch cases matter most. Its absence has **no other symptom** — the build
/// succeeds, the tests pass, the numbers stay bit-identical — so a probe that quietly
/// answered "fine" when it could not find the checkout would be worse than no probe.
@Suite("Metal library")
struct MetalLibraryTests {

    private func withTree(_ body: (URL) throws -> Void) throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("metallib-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    private func writeMatmul(_ contents: String, in root: URL) throws {
        let target = MetalLibrary.gemmPatchTarget(packageRoot: root)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try contents.write(to: target, atomically: true, encoding: .utf8)
    }

    @Test("a patched checkout reads as applied")
    func patchApplied() throws {
        try withTree { root in
            try writeMatmul("if (M >= 8192) { /* \(MetalLibrary.patchMarker) */ }", in: root)
            #expect(MetalLibrary.gemmPatch(packageRoot: root) == .applied)
        }
    }

    @Test("an unpatched checkout reads as missing and names the file")
    func patchMissing() throws {
        try withTree { root in
            try writeMatmul("bm = 64; bn = 64; bk = 16;", in: root)
            guard case let .missing(target) = MetalLibrary.gemmPatch(packageRoot: root) else {
                Issue.record("expected missing"); return
            }
            #expect(target.hasSuffix("matmul.cpp"))
        }
    }

    /// A check that passes because it could not find what it was checking is worse than no
    /// check: it reports the state it was asked about without ever having observed it.
    @Test("no checkout reads as unknown, never as applied or missing")
    func patchNotCheckedOut() throws {
        try withTree { root in
            guard case let .notCheckedOut(looked) = MetalLibrary.gemmPatch(packageRoot: root)
            else {
                Issue.record("an absent checkout must not resolve to a verdict"); return
            }
            #expect(looked.contains("mlx-swift"))
        }
    }

    @Test("Ultra and Max chips are the ones the patch tunes for")
    func ultraDetection() {
        func machine(_ chip: String) -> Machine {
            Machine(model: "Mac15,14", chip: chip, memoryBytes: 0, cores: 0)
        }
        #expect(MetalLibrary.isUltraClass(machine("Apple M3 Ultra")))
        #expect(MetalLibrary.isUltraClass(machine("Apple M4 Max")))
        #expect(!MetalLibrary.isUltraClass(machine("Apple M3 Pro")))
        #expect(!MetalLibrary.isUltraClass(machine("Apple M2")))
    }

    /// `dladdr`, not `Bundle.main`: under `swift test` the main bundle is the toolchain's
    /// test helper, and a probe rooted there would report a directory MLX never consults.
    @Test("the search order starts beside the running binary")
    func searchOrder() throws {
        let paths = MetalLibrary.searchPaths
        #expect(paths.count >= 1)
        #expect(paths.last?.lastPathComponent == "default.metallib",
                "the cwd fallback must be last, since finding only it is a warning")
        if let dir = MetalLibrary.binaryDirectory {
            #expect(paths.first?.path == dir.appendingPathComponent("mlx.metallib").path)
            #expect(!dir.path.contains("swiftpm-testing-helper"),
                    "binaryDirectory must be this image, not the test host")
        }
    }
}

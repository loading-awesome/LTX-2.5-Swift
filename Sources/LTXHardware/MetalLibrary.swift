// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Finding MLX's compiled Metal kernels, and the tuning patch, before anything needs them.
///
/// **`swift build` does not compile `.metal` files at all** — not for mlx-swift, not for
/// anything. Metal compilation is a feature of Xcode's build system, and mlx-swift relies on
/// it anyway, so a command-line build links cleanly and then dies on its first GPU op with
/// an untyped C++ `runtime_error` that contains no path.
/// `tools/build_mlx_metallib.sh` is what puts `mlx.metallib` beside the built binary.
///
/// MLX looks in four places, in order (`mlx/backend/metal/device.cpp`):
///
///  1. `mlx.metallib` **beside the running binary** — what the script targets;
///  2. `Resources/mlx.metallib` beside the binary;
///  3. `default.metallib` in a loaded `mlx-swift_Cmlx.bundle` — the Xcode path, empty here;
///  4. `default.metallib` **relative to the current working directory**.
///
/// Step 4 is the trap. A tree can run on it for months because every command happens to be
/// issued from the repository root, and then the first person to run an installed binary
/// from their home directory gets the untyped error. So a library found only that way is
/// reported as a warning rather than a pass.
public enum MetalLibrary {

    /// The directory holding the binary this code is compiled into.
    ///
    /// **`dladdr`, not `Bundle.main`.** MLX finds itself with `dladdr`; under `swift test`
    /// the process's main bundle is the toolchain's test helper while the image containing
    /// this code is the `.xctest` bundle. Asking `Bundle.main` would report a directory MLX
    /// never looks in, producing a check that disagrees with the thing it is checking.
    public static var binaryDirectory: URL? {
        var info = Dl_info()
        guard dladdr(#dsohandle, &info) != 0, let name = info.dli_fname else { return nil }
        return URL(fileURLWithPath: String(cString: name))
            .resolvingSymlinksInPath().deletingLastPathComponent()
    }

    /// Every path MLX will try, in the order it tries them. Reported verbatim by `doctor`:
    /// a list of places looked is worth more to somebody stuck than "not found".
    public static var searchPaths: [URL] {
        var paths: [URL] = []
        if let dir = binaryDirectory {
            paths.append(dir.appendingPathComponent("mlx.metallib"))
            paths.append(dir.appendingPathComponent("Resources/mlx.metallib"))
        }
        paths.append(URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("default.metallib"))
        return paths
    }

    public static func locate() -> URL? {
        searchPaths.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Whether the only copy found is the working-directory fallback, which makes the
    /// process work from here and nowhere else.
    public static func locatedOnlyViaWorkingDirectory() -> Bool {
        guard let found = locate() else { return false }
        return found.lastPathComponent == "default.metallib"
    }

    // MARK: - The large-M GEMM patch

    /// Whether the M3-Ultra large-M GEMM tuning is in the dependency checkout.
    ///
    /// The patch routes bf16/fp16 matmuls with `M >= 8192` on Ultra-class devices to a
    /// `32x64x16, 1x2sg` Steel kernel MLX already ships but does not choose. It is
    /// bit-identical and worth 8.7–14.6% on the feed-forward GEMMs above M = 8192.
    ///
    /// **It lives in `.build/checkouts`, which is gitignored, so nothing in this repository
    /// can hold it.** `swift package reset`, `swift package update`, a clean checkout and a
    /// fresh clone all drop it, and every one of those is routine. Nothing then fails: the
    /// build succeeds, the tests pass, the numbers stay bit-identical, and the binary is
    /// measurably slower on long clips for the rest of its life. **The failure has no other
    /// symptom**, which is the entire reason it is worth a line in a diagnostic.
    public enum GEMMPatch: Sendable, Equatable {
        case applied
        case missing(target: String)
        /// The checkout is not there — an installed binary, or a tree before
        /// `swift package resolve`. Reported as unknown rather than as either answer,
        /// because a check that passes for want of anything to look at is worse than none.
        case notCheckedOut(looked: String)

        public var label: String {
            switch self {
            case .applied: return "applied"
            case .missing: return "MISSING"
            case .notCheckedOut: return "unknown"
            }
        }
    }

    /// A marker unique to the patched branch, from the patch's own added lines.
    ///
    /// Matched as a string rather than by `git apply --reverse --check` so this needs no
    /// subprocess and no git; the trade is that a hand-edit reproducing the comment but not
    /// the tuning would read as applied.
    static let patchMarker = "Large-M Max/Ultra tuning"

    /// Where the patch lands, relative to a package root. The patch's paths are
    /// `a/mlx/backend/metal/matmul.cpp`, relative to the vendored mlx submodule rather than
    /// to the mlx-swift checkout above it.
    public static func gemmPatchTarget(packageRoot: URL) -> URL {
        packageRoot
            .appendingPathComponent(".build/checkouts/mlx-swift/Source/Cmlx/mlx")
            .appendingPathComponent("mlx/backend/metal/matmul.cpp")
    }

    public static func gemmPatch(packageRoot: URL) -> GEMMPatch {
        let target = gemmPatchTarget(packageRoot: packageRoot)
        guard let source = try? String(contentsOf: target, encoding: .utf8) else {
            return .notCheckedOut(looked: target.path)
        }
        return source.contains(patchMarker) ? .applied : .missing(target: target.path)
    }

    /// Whether this machine is one the patch is tuning for.
    ///
    /// The patch is Ultra-class tuning and a machine that will never take that branch does
    /// not need it, so a diagnostic should not report its absence as a problem on, say, an
    /// M3 Pro. Read from the chip string rather than assumed from core count.
    public static func isUltraClass(_ machine: Machine) -> Bool {
        machine.chip.localizedCaseInsensitiveContains("ultra")
            || machine.chip.localizedCaseInsensitiveContains("max")
    }
}

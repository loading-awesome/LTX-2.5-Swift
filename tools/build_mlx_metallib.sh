#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
#
# Build MLX's Metal library and put it where MLX looks for it.
#
#   tools/build_mlx_metallib.sh [--release] [--clean]
#
# ## Why this script has to exist
#
# **SwiftPM's command-line build does not compile `.metal` files at all.** Not for
# mlx-swift, not for anything: a package containing a single trivial kernel builds
# clean, produces no `.metallib`, creates no resource bundle, and emits no warning
# about an unhandled file. Metal compilation is a feature of Xcode's build system,
# not of `swift build`.
#
# mlx-swift relies on it anyway. Its `Package.swift` defines
# `METAL_PATH="default.metallib"` and `SWIFTPM_BUNDLE="mlx-swift_Cmlx"`, and its
# excludes carry a comment referring to a `PrepareMetalShaders` step that no longer
# exists in the tree (0.31.6 ships only a `CudaBuild` plugin). So under `swift build`
# / `swift test` every MLX entry point dies at device resolution:
#
#     MLX error: Failed to load the default metallib. library not found ...
#
# and it dies even when the caller asks for `Device(.cpu)`, because the default
# device is resolved during initialisation. That aborts the whole test process, which
# is why the MLX-free suites must never gain an MLX dependency.
#
# ## What it builds, and why that is the whole set
#
# `Source/Cmlx/mlx-generated/metal` is mlx's prepared shader tree: the handful of
# kernels that are precompiled, with every header they include sitting alongside so
# `#include "utils.h"` resolves. Everything else — binary, unary, copy, reduce, sort,
# softmax, the gemm kernels — is JIT-compiled at runtime from source strings embedded
# in `mlx-generated/*.cpp`, which is why those appear here as `.h` and not `.metal`.
# Compiling the nine `.metal` files in that tree is therefore sufficient; verified by
# a matmul (a JIT kernel) evaluating correctly afterwards.
#
# ## Where it puts the result, and why the name matters
#
# From `load_default_library` in `mlx/backend/metal/device.cpp`, in order:
#
#   1. <binary dir>/mlx.metallib            <- this is the one we use
#   2. <binary dir>/Resources/mlx.metallib
#   3. mlx-swift_Cmlx.bundle/<resources>/default.metallib
#   4. <binary dir>/Resources/default.metallib
#   5. default.metallib, relative to the working directory
#
# Note that the colocated name is **mlx**.metallib while the bundle name is
# **default**.metallib. Dropping a correctly built `default.metallib` next to the
# binary does nothing, which cost a debugging round here.
#
# `binary dir` is the directory of the loaded binary, so it differs per product: for
# `swift test` it is inside `LTX25PackageTests.xctest/Contents/MacOS`, for the CLI it
# is `.build/<triple>/<config>`. The script copies to every such directory it finds,
# which is cheap (3 MB) and avoids encoding a guess about which product is running.

set -euo pipefail

CONFIG="debug"
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --release) CONFIG="release" ;;
        --clean) CLEAN=1 ;;
        -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
        *) echo "error: unknown argument $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CMLX="$(find .build/checkouts/mlx-swift/Source/Cmlx -maxdepth 0 -type d 2>/dev/null || true)"
if [[ -z "$CMLX" ]]; then
    echo "error: mlx-swift is not checked out. Run 'swift package resolve' first." >&2
    exit 1
fi

SHADERS="$CMLX/mlx-generated"
if [[ ! -d "$SHADERS/metal" ]]; then
    echo "error: $SHADERS/metal is missing. The vendored shader tree moved, which" >&2
    echo "  means this script's assumption about what needs compiling is stale." >&2
    exit 1
fi

WORK="$ROOT/.build/mlx-metallib"
OUT="$WORK/mlx.metallib"
[[ "$CLEAN" == "1" ]] && rm -rf "$WORK"
mkdir -p "$WORK"

# Read into an array without `mapfile`: macOS ships bash 3.2, where it does not
# exist. A script that only runs under a Homebrew bash is a script that fails on a
# clean machine.
SOURCES=()
while IFS= read -r line; do
    SOURCES+=("$line")
done < <(find "$SHADERS" -name '*.metal' | sort)
if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "error: no .metal sources under $SHADERS" >&2
    exit 1
fi

# Rebuild only when a source or header is newer than the library. Headers count:
# most of the kernel bodies live in .h files included by these nine.
NEEDS_BUILD=0
if [[ ! -f "$OUT" ]]; then
    NEEDS_BUILD=1
elif [[ -n "$(find "$SHADERS" -newer "$OUT" \( -name '*.metal' -o -name '*.h' \) -print -quit)" ]]; then
    NEEDS_BUILD=1
fi

if [[ "$NEEDS_BUILD" == "1" ]]; then
    echo "building mlx.metallib from ${#SOURCES[@]} kernel(s)"
    AIR=()
    for src in "${SOURCES[@]}"; do
        # Flatten the path into the object name: two kernels share a basename
        # across subdirectories, and colliding .air files would silently drop one.
        obj="$WORK/$(echo "${src#$SHADERS/}" | tr '/' '_' | sed 's/\.metal$/.air/')"
        xcrun -sdk macosx metal -c "$src" \
            -I "$SHADERS/metal" -I "$SHADERS" -I "$CMLX/mlx" -I "$CMLX" \
            -o "$obj"
        AIR+=("$obj")
    done
    xcrun -sdk macosx metallib "${AIR[@]}" -o "$OUT"
    echo "  wrote $OUT ($(du -h "$OUT" | cut -f1))"
else
    echo "mlx.metallib is up to date"
fi

# Place it beside every built binary. `swift build`/`swift test` must have run at
# least once for these to exist.
BUILD_DIR="$ROOT/.build/arm64-apple-macosx/$CONFIG"
if [[ ! -d "$BUILD_DIR" ]]; then
    BUILD_DIR="$(swift build --configuration "$CONFIG" --show-bin-path 2>/dev/null || true)"
fi
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "note: no $CONFIG build directory yet. Run 'swift build' then re-run this." >&2
    exit 0
fi

PLACED=0
cp "$OUT" "$BUILD_DIR/mlx.metallib" && PLACED=$((PLACED + 1))
while IFS= read -r bundle; do
    macos="$bundle/Contents/MacOS"
    if [[ -d "$macos" ]]; then
        cp "$OUT" "$macos/mlx.metallib"
        PLACED=$((PLACED + 1))
    fi
done < <(find "$BUILD_DIR" -maxdepth 1 -name '*.xctest' -o -maxdepth 1 -name '*.app' 2>/dev/null)

echo "placed mlx.metallib in $PLACED location(s) under $BUILD_DIR"
echo

# The GEMM tile-routing patch rides along here because this is the step that
# turns a bare `swift build` into a working GPU binary, so it is the only place
# a development build reliably passes through. It runs last — the metallib copy above is this script's
# actual job and should not be held up by a performance concern.
"$ROOT/tools/check-mlx-patch.sh"

echo "Verify with:  swift test --filter MLXCapabilityTests"

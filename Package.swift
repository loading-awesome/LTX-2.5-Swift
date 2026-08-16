// swift-tools-version: 6.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich
import PackageDescription

// The target graph is the architecture, so it is worth reading as one.
//
// Dependencies only ever point downward, and the four lowest layers do not
// link MLX at all. That is not tidiness: it means `swift test` for the frame
// lattice, latent geometry, safetensors headers, checkpoint identification,
// the flow schedule and the memory planner runs in seconds on any machine,
// with no GPU and no 40+ GB checkpoint. Those are exactly the parts where a
// silent error stays silent, so they are the parts that must stay cheap to
// check.
//
//   LTXFoundation   errors, geometry, frame lattice, config, safetensors  (no MLX)
//   LTXHardware     chip + memory detection, the memory planner           (no MLX)
//   LTXCatalog      checkpoint discovery and identification               (no MLX)
//   LTXRecipes      capability-aware recipe resolution                    (no MLX)
//   LTXAttention    the attention backend seam, incl. the STG passthrough (MLX)
//   LTXModules      DiT, VAEs, vocoder, text encoder                      (MLX)
//   LTXPipeline     conditioning, layout, sampler, guidance, decode, mux  (MLX)
//   LTX25           the public API and actor-owned runtime facade
//   ltx             a thin CLI over the public API
//
// See docs/SWIFT_ARCHITECTURE.md for why the boundary sits where it does.
let package = Package(
    name: "LTX25",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "LTX25", targets: ["LTX25"]),
        .executable(name: "ltx", targets: ["ltx"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.2"),
    ],
    targets: [
        .target(name: "LTXFoundation"),
        .target(name: "LTXHardware", dependencies: ["LTXFoundation"]),
        .target(name: "LTXCatalog", dependencies: ["LTXFoundation"]),
        .target(name: "LTXRecipes", dependencies: ["LTXFoundation", "LTXHardware", "LTXCatalog"]),
        .target(
            name: "LTXAttention",
            dependencies: [
                "LTXFoundation", "LTXHardware",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "LTXModules",
            dependencies: [
                "LTXFoundation", "LTXCatalog", "LTXAttention",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Training only. AdamW's bias correction and decoupled weight decay are
                // exactly the arithmetic that is easy to get subtly wrong and impossible to
                // notice from a loss curve, so the library's optimiser is used rather than
                // a local one.
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                // No MLXFFT. The vocoder's bandwidth-extension stage re-analyses its own
                // 16 kHz output through a mel STFT, which looks like it needs an FFT and
                // does not: `vocoder.mel_stft.stft_fn.forward_basis` is a *stored* buffer
                // and the reference applies it as a convolution. Using a real FFT here
                // would be a second implementation of the reference's own transform,
                // agreeing with it only to the extent the two happen to round alike.
            ]
        ),
        .target(
            name: "LTXPipeline",
            dependencies: [
                "LTXFoundation", "LTXHardware", "LTXCatalog", "LTXRecipes", "LTXModules",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "LTX25",
            dependencies: [
                "LTXFoundation", "LTXHardware", "LTXCatalog",
                "LTXRecipes", "LTXModules", "LTXPipeline",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "ltx",
            dependencies: [
                "LTX25",
                "LTXFoundation", "LTXHardware", "LTXCatalog", "LTXRecipes",
                // Already transitive through LTX25 -> LTXPipeline; direct so that
                // subcommands reaching past the facade can import them.
                "LTXModules", "LTXPipeline",
                .product(name: "MLX", package: "mlx-swift"),
                // `bench gemm` measures attention through the same
                // `MLXFast.scaledDotProductAttention` call `DiTAttention` makes. A
                // hand-rolled equivalent here would measure a different kernel.
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),

        // The MLX-free suites. These must never gain an MLX dependency: their
        // value is that they run on a bare checkout in seconds.
        .testTarget(name: "LTXFoundationTests", dependencies: ["LTXFoundation"]),
        .testTarget(name: "LTXHardwareTests", dependencies: ["LTXHardware"]),
        .testTarget(name: "LTXCatalogTests", dependencies: ["LTXCatalog"]),
        .testTarget(
            name: "LTXModulesTests",
            dependencies: [
                "LTXModules", "LTXCatalog", "LTXFoundation",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        // The sampler. Needs MLX for the arithmetic and `LTXModules` only for
        // `DiTForward.Geometry` — the suite drives the step loop with a stub denoiser, so
        // it never loads the 42 GB transformer and runs in seconds.
        .testTarget(
            name: "LTXPipelineTests",
            dependencies: [
                "LTXPipeline", "LTXModules", "LTXFoundation", "LTXCatalog",
                .product(name: "MLX", package: "mlx-swift"),
                // `TrainingDatasetTests` takes a real optimiser step on a cached sample.
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "LTXRecipesTests", dependencies: ["LTXRecipes"]),
    ]
)

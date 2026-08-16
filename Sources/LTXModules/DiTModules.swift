// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// The pieces a DiT block is assembled from: the patchify projections, the feed-forwards,
/// the AdaLN timestep path, and the activations they share.
///
/// The DiT is 48 blocks of two bridged streams. Each module here stands on its own and
/// takes its input directly, so a block is composed of parts that can each be read — and
/// exercised — without the rest of the stack around them.
public struct DiTModules {

    public let weights: [String: MLXArray]
    /// `model.diffusion_model.` on a ComfyUI export, `""` on a stripped one. Both occur.
    public let prefix: String
    public let topology: TransformerTopology

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case shape(String, got: [Int], expected: String)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            }
        }
    }

    public init(checkpoint url: URL, topology: TransformerTopology) throws {
        self.topology = topology
        self.prefix = topology.prefix
        self.weights = try MLX.loadArrays(url: url)
    }

    /// Share an already-loaded weight map, so a caller that also builds `DiTAttention`
    /// does not pay for a second 21 B-parameter dictionary.
    public init(weights: [String: MLXArray], topology: TransformerTopology) {
        self.topology = topology
        self.prefix = topology.prefix
        self.weights = weights
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[prefix + name] else { throw Failure.missing(prefix + name) }
        return w
    }

    private func bias(_ name: String) -> MLXArray? { weights[prefix + name] }

    /// Adapters applied on top of the base weights, or `nil` for an unmodified model.
    ///
    /// **Overlay, not bake.** The alternative merges `W += strength · B·A` into
    /// `weights`, which for a 22 B model is a permanent multi-gigabyte expansion per
    /// adapter and has to be undone exactly to swap one. This adds the low-rank product
    /// at use, so `weights` stays the checkpoint and a recipe can change adapters
    /// between stages for the cost of a dictionary swap. `LoRAOverlay` has the
    /// measurement.
    public var lora: LoRAOverlay?

    /// A Linear, `[out, in]` in the checkpoint's layout.
    ///
    /// **The input is cast to the weight's dtype, and that is load-bearing.** The model
    /// runs bf16 end to end; hand one of these an fp32 input and MLX promotes the matmul,
    /// so the layer computes *more precisely* than every other layer around it — closer
    /// to the exact arithmetic and further from what the rest of the model produces.
    ///
    /// This is the na3d warning in `FRAGILE_CONTRACTS.md` arriving somewhere nobody was
    /// watching for it: higher internal precision can be arithmetically more accurate
    /// while agreeing with the bf16 path less. The cast is lossless whenever the values
    /// really are bf16, so it restores the intended precision rather than discarding
    /// anything.
    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let w = try weight(name + ".weight")
        var y = MLX.matmul(x.asType(w.dtype), w.transposed(1, 0))
        // Before the bias, because a LoRA is a delta on the *linear map* and not on the
        // affine one. `B·A·x + b` and `B·A·(x) + b` agree here only because the delta is
        // added to `y` rather than folded into `w` — but adding it after the bias would
        // still be the same arithmetic, and adding it to `x` would not be. The ordering
        // is written down because the wrong one is easy and silent.
        if let lora, let delta = lora.delta(forWeightKey: prefix + name + ".weight", input: x) {
            y = y + delta.asType(y.dtype)
        }
        if let b = bias(name + ".bias") { y = y + b.asType(y.dtype) }
        return y
    }

    // MARK: - Patchify

    /// `patchify_proj` / `audio_patchify_proj`: latent channels → stream width.
    ///
    /// The two streams have **different widths** — 128 → 4096 for video and 128 → 2048
    /// for audio — which is the first place the "no packed sequence" structure becomes
    /// visible in code rather than in a document. Both carry biases.
    public func patchify(_ x: MLXArray, stream: TextConditioningLayout.Stream) throws
        -> MLXArray {
        try linear(x, stream == .video ? "patchify_proj" : "audio_patchify_proj")
    }

    // MARK: - Feed-forward

    /// A DiT block's feed-forward.
    ///
    /// The checkpoint names the two projections `net.0.proj` and `net.2`, with an
    /// identity in the slot between them, so the arithmetic is
    /// `Linear -> gelu(tanh approximation) -> Linear` with a plain 4× expansion.
    /// **Not** a GEGLU: a gated variant would need `2 x inner` from the first projection,
    /// and the checkpoint's shape is `[16384, 4096]` for a 4096-wide stream — exactly 4×,
    /// with nothing to split.
    ///
    /// **The bias asymmetry is real and is read from the checkpoint, never assumed.**
    /// The video `ff` has no biases; `audio_ff` has them on both projections. One
    /// shared code path with `bias: true` fails to load the video weights, and with
    /// `bias: false` silently drops the audio biases — a numerical divergence at every
    /// block with no shape error to catch it. `TransformerTopology` measures which is
    /// which; this reads the tensors and simply adds a bias when one exists.
    public func feedForward(_ x: MLXArray, block: Int,
                           stream: TextConditioningLayout.Stream) throws -> MLXArray {
        let stem = "transformer_blocks.\(block)."
            + (stream == .video ? "ff." : "audio_ff.")
        let projected = try linear(x, stem + "net.0.proj")
        let activated = geluTanh(projected)
        return try linear(activated, stem + "net.2")
    }

    // MARK: - AdaLN

    /// Sinusoidal timestep embedding, `256` channels.
    ///
    /// Three choices decide the values, and each has a plausible alternative that
    /// produces a same-shaped, wrong tensor:
    ///
    /// - **cos before sin** — the output is `[cos, sin]`, **not** `[sin, cos]`. Getting
    ///   the halves the wrong way round is a permutation the next Linear reads as
    ///   garbage, and `[sin, cos]` is the more common convention.
    /// - **no frequency shift** — the exponent is divided by `half_dim`, not by
    ///   `half_dim - 1`. With 128 half-dims the two differ by 0.8% in every frequency.
    /// - **a period of 10000** and unit scale.
    ///
    /// Computed in fp32: the frequencies span `1` to `1e-4` and a bf16 exponent would
    /// quantise the small end badly.
    public func timestepProjection(_ timesteps: MLXArray, channels: Int = 256)
        -> MLXArray {
        let halfDim = channels / 2
        let logPeriod = Float(log(10_000.0))
        // -ln(max_period) * i / half_dim — no frequency shift.
        let i = MLXArray(0..<halfDim).asType(.float32)
        let exponent = (-logPeriod * i) / Float(halfDim)
        let frequencies = MLX.exp(exponent)                       // [halfDim]

        let t = timesteps.asType(.float32).reshaped([timesteps.size, 1])
        let angles = t * frequencies.reshaped([1, halfDim])        // [N, halfDim]
        // cos first, then sin.
        return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
    }

    /// `AdaLayerNormSingle`: timesteps → per-token modulation parameters.
    ///
    /// ```
    /// proj      = timestepProjection(t)                256 channels, cos before sin
    /// embedded  = linear_2(silu(linear_1(proj)))       <- TimestepEmbedding
    /// out       = linear(silu(embedded))               <- AdaLayerNormSingle
    /// ```
    ///
    /// **SiLU appears twice**, once inside `TimestepEmbedding` between its two linears
    /// and once in `AdaLayerNormSingle` before the final projection. Dropping either
    /// leaves the shapes intact.
    ///
    /// The output width is `coefficient × streamWidth`, and the coefficient is a
    /// **structural fact read from the checkpoint**: 6 base parameters (shift and scale
    /// for the self-attention, feed-forward and output norms) plus 3 more when
    /// cross-attention AdaLN is enabled. It is `36864 = 9 × 4096` for video and
    /// `18432 = 9 × 2048` for audio, so it *is* enabled here — while
    /// `prompt_adaln_single` is `8192 = 2 × 4096`.
    ///
    /// Returns both halves: the blocks consume the modulation, and the output head
    /// consumes the embedded timestep.
    public func adaLayerNormSingle(_ timesteps: MLXArray, module: String) throws
        -> (modulation: MLXArray, embedded: MLXArray) {
        let projection = timestepProjection(timesteps)
        let stem = module + ".emb.timestep_embedder."

        var embedded = try linear(projection.asType(
            try weight(stem + "linear_1.weight").dtype), stem + "linear_1")
        embedded = silu(embedded)
        embedded = try linear(embedded, stem + "linear_2")

        let modulation = try linear(silu(embedded), module + ".linear")
        return (modulation, embedded)
    }

    /// `x * sigmoid(x)`, evaluated in fp32 and cast back.
    ///
    /// Same reasoning as `geluTanh`: an elementwise nonlinearity on a bf16 tensor is
    /// evaluated with an fp32 accumulate type.
    public func silu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * MLX.sigmoid(f)).asType(x.dtype)
    }

    /// How many modulation parameters a given AdaLN module emits, from its own weights.
    ///
    /// Derived, never assumed: 9 means cross-attention AdaLN is enabled, 6 means it is
    /// not, and `prompt_adaln_single` emits 2. Hardcoding 6 would slice the modulation
    /// tensor into the wrong pieces.
    public func modulationCoefficient(module: String, streamWidth: Int) throws -> Int {
        let w = try weight(module + ".linear.weight")
        guard w.shape.count == 2, w.shape[0] % streamWidth == 0 else {
            throw Failure.shape(module + ".linear.weight", got: w.shape,
                                expected: "[k * \(streamWidth), \(streamWidth)]")
        }
        return w.shape[0] / streamWidth
    }

    /// GELU in its tanh approximation, not the erf form.
    ///
    /// The checkpoint's config names `gelu-approximate`, and the two forms are far enough
    /// apart that substituting one visibly moves a block's feed-forward output.
    ///
    /// **Evaluated in fp32 and cast back.** An elementwise nonlinearity on a
    /// reduced-precision tensor uses fp32 as its accumulate type, and here it matters far
    /// more than it looks — the cubic term is the problem. Computing
    /// `x + 0.044715 * x^3` in bf16 loses most of the cube's precision in the mid-range
    /// where the tanh has not yet saturated, and that lands directly on the block's
    /// feed-forward output.
    ///
    /// So "compute in the checkpoint's dtype" is not a blanket rule: matmuls take the
    /// weight dtype, nonlinearities take fp32. Getting either backwards moves the output
    /// by an order of magnitude while leaving its shape alone.
    public func geluTanh(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let c = Float(0.7978845608028654)          // sqrt(2/pi)
        let inner = c * (f + Float(0.044715) * f * f * f)
        return (Float(0.5) * f * (1.0 + MLX.tanh(inner))).asType(x.dtype)
    }
}

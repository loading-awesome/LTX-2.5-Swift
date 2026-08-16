// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// The last block's residual stream back to latent channels.
///
/// Four lines, and every one of them is a decision that can go wrong while every shape
/// stays intact:
///
/// ```
/// shift, scale = scale_shift_table.to(x.dtype) + embedded_timestep
/// x = norm_out(x)                  LayerNorm(dim), no affine, eps 1e-6
/// x = x * (1 + scale) + shift
/// x = proj_out(x)                  Linear(dim, out_channels), with bias
/// ```
///
/// ## The norm is a LayerNorm, and nothing else in the DiT is
///
/// `DiTBlock.rmsNorm` is the norm used by every sub-layer of every block. The head's
/// **subtracts the mean** before dividing by the standard deviation. Both are
/// unparameterised — there is correspondingly no `norm_out.weight` in the checkpoint — so
/// there is no tensor whose presence or absence distinguishes them, and both accept and
/// return `[B, T, dim]`. Substituting one for the other changes nothing structural and
/// everything numerical: an RMS norm preserves the mean-to-RMS ratio, and the stream
/// arriving here has a mean well away from zero.
///
/// ## The table is the top-level one, not a block's
///
/// `scale_shift_table` and `audio_scale_shift_table` exist twice in this checkpoint: once
/// per block as `[9, dim]` and `[5, dim]` modulation tables, and once at the top level as
/// `[2, dim]`. The head uses the **top-level** pair. Slicing rows 0 and 1 out of
/// `transformer_blocks.47.scale_shift_table` produces the same shape and the wrong values.
///
/// ## The timestep is the *embedded* one, not the projected one
///
/// `AdaLayerNormSingle` returns `(linear(silu(embedded)), embedded)`. The blocks consume
/// the first — `[T, 9 * dim]`; the head consumes the second — `[T, dim]`. They are one
/// SiLU and one `[9*dim, dim]` matmul apart, so reaching for "the AdaLN output" and
/// slicing `dim` columns off the wide one gives a well-shaped, wrong modulation.
///
/// ## The table is downcast
///
/// `scale_shift_table` and `audio_scale_shift_table` are among the checkpoint's **290
/// fp32 tensors** (of 4349; all 290 are scale/shift tables). They are cast down to the
/// stream's bf16 before the add, so the modulation is computed at bf16 like everything
/// else. Skip the downcast and the whole affine promotes, computing the head at a
/// precision nothing else in the model uses.
public struct DiTOutputHead {

    public let weights: [String: MLXArray]
    /// `model.diffusion_model.` on a ComfyUI export, `""` on a stripped one.
    public let prefix: String
    public let topology: TransformerTopology

    /// `norm_eps` from the model config — the same 1e-6 the blocks and both output
    /// LayerNorms use.
    public static let normEpsilon = Float(1e-6)

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

    /// The three tensors the head produces, in the order it produces them.
    public struct Output {
        /// The LayerNorm alone, **before** the affine.
        public let normed: MLXArray
        /// After `x * (1 + scale) + shift` — what `proj_out` is given.
        public let modulated: MLXArray
        /// `proj_out`'s output: latent-shaped, `[B, T, latentChannels]`.
        public let latent: MLXArray
    }

    public init(checkpoint url: URL, topology: TransformerTopology) throws {
        self.weights = try MLX.loadArrays(url: url)
        self.topology = topology
        self.prefix = topology.prefix
    }

    /// Share an already-loaded weight map rather than paying for a second 21 B-parameter
    /// dictionary.
    public init(weights: [String: MLXArray], topology: TransformerTopology) {
        self.weights = weights
        self.topology = topology
        self.prefix = topology.prefix
    }

    public func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[prefix + name] else { throw Failure.missing(prefix + name) }
        return w
    }

    // MARK: - Names

    /// The four tensors one stream's head needs, named as the checkpoint names them.
    ///
    /// The audio names are not the video names under a different prefix in the way a
    /// regex would produce: it is `audio_scale_shift_table` and `audio_proj_out`, both
    /// top-level, while the block-level tables live under `transformer_blocks.<n>.` with
    /// the *same* leaf names. Resolving `scale_shift_table` without knowing which of the
    /// two namespaces is meant finds a `[9, dim]` tensor where a `[2, dim]` one belongs.
    public static func tableName(_ stream: TextConditioningLayout.Stream) -> String {
        stream == .video ? "scale_shift_table" : "audio_scale_shift_table"
    }

    public static func projectionName(_ stream: TextConditioningLayout.Stream) -> String {
        stream == .video ? "proj_out" : "audio_proj_out"
    }

    /// The dtype the head computes in, read from the **projection** weight.
    ///
    /// Deliberately not read from `scale_shift_table`: that is one of the checkpoint's
    /// fp32 tensors, so it would report fp32 and every caller that trusted it would run
    /// the head at a precision nothing else in the model uses. `DiTBlock.computeDType`
    /// avoids the same trap by reading a feed-forward projection.
    public func computeDType(stream: TextConditioningLayout.Stream) throws -> DType {
        try weight(Self.projectionName(stream) + ".weight").dtype
    }

    // MARK: - Primitives

    /// LayerNorm over the last axis: no learned affine, `eps = 1e-6`.
    ///
    /// Mean-subtracting, with the **biased** variance — divided by `N`, not `N - 1`. At
    /// `dim = 4096` the two differ by 1.2e-04 in the scale, small enough to pass
    /// unnoticed, so this is written from the definition rather than fitted.
    ///
    /// The statistics are accumulated in fp32 inside `layer_norm.metal` (one of
    /// the nine kernels `tools/build_mlx_metallib.sh` compiles). `eps` is added to
    /// the variance, inside the square root — not to the standard deviation.
    public func layerNorm(_ x: MLXArray) -> MLXArray {
        MLX.layerNorm(x, weight: nil, bias: nil, eps: Self.normEpsilon)
    }

    /// `scale_shift_table.to(dtype) + embeddedTimestep`, unpacked as `(shift, scale)` —
    /// **shift is row 0**.
    ///
    /// The same order the blocks' modulation triple uses, and the opposite of the
    /// cross-modal pair.
    ///
    /// - Parameters:
    ///   - table: `[2, dim]`. Passed in rather than resolved, so the caller decides which
    ///     of the two namespaces below it means.
    ///   - embeddedTimestep: `[B, T, dim]`. Reshaping the `[T, dim]` embedded timestep is
    ///     the caller's job, so that a rank-2 tensor arriving here fails instead of being
    ///     read as one batch row.
    ///   - dtype: the stream's compute dtype. The table is cast to it — see the type's
    ///     documentation for why that is not a formality on this checkpoint.
    public func modulation(table: MLXArray, embeddedTimestep: MLXArray, dtype: DType)
        throws -> (shift: MLXArray, scale: MLXArray) {
        guard table.shape.count == 2, table.dim(0) == 2 else {
            throw Failure.shape("scale_shift_table", got: table.shape, expected: "[2, dim]")
        }
        let dim = table.dim(1)
        guard embeddedTimestep.shape.count == 3, embeddedTimestep.dim(2) == dim else {
            throw Failure.shape("embedded_timestep", got: embeddedTimestep.shape,
                                expected: "[B, T, \(dim)]")
        }
        let t = embeddedTimestep.asType(dtype)
        let values = table.asType(dtype).reshaped([1, 1, 2, dim])
            + t.reshaped([t.dim(0), t.dim(1), 1, dim])
        return (values[0..., 0..., 0, 0...], values[0..., 0..., 1, 0...])
    }

    /// `proj_out` / `audio_proj_out`: stream width → latent channels, with a bias.
    ///
    /// Both streams project to the **same** 128 latent channels from different widths
    /// (4096 and 2048), which is the last place the two-stream structure is visible.
    ///
    /// The input is cast to the weight's dtype for the reason `DiTModules.linear`
    /// documents: an fp32 input promotes the matmul out of the model's working precision.
    public func project(_ x: MLXArray, stream: TextConditioningLayout.Stream) throws
        -> MLXArray {
        let name = Self.projectionName(stream)
        let w = try weight(name + ".weight")
        let b = try weight(name + ".bias")
        return MLX.matmul(x.asType(w.dtype), w.transposed(1, 0)) + b.asType(w.dtype)
    }

    // MARK: - Forward

    /// One stream's head, from the last block's output to latent-shaped velocity.
    ///
    /// - Parameters:
    ///   - x: the final block's residual stream, `[B, T, dim]`, unchanged — the head adds
    ///     nothing between the block loop and itself.
    ///   - embeddedTimestep: the AdaLN module's embedded timestep, `[B, T, dim]`.
    public func callAsFunction(_ x: MLXArray, embeddedTimestep: MLXArray,
                               stream: TextConditioningLayout.Stream) throws -> Output {
        let dtype = try computeDType(stream: stream)
        // Cast at entry, exactly as `DiTBlock.callAsFunction` does. The head is almost
        // entirely elementwise — a norm, an affine — and none of that goes near a weight
        // to pick up its dtype on the way, so an fp32 input left alone would run the
        // whole head in fp32.
        let x = x.asType(dtype)
        let table = try weight(Self.tableName(stream))

        let (shift, scale) = try modulation(table: table,
                                            embeddedTimestep: embeddedTimestep,
                                            dtype: dtype)

        let normed = layerNorm(x)
        let modulated = (normed.asType(scale.dtype) * (1.0 + scale) + shift).asType(dtype)

        return Output(normed: normed, modulated: modulated,
                      latent: try project(modulated, stream: stream))
    }

    /// Both streams, video first.
    ///
    /// The two heads share no state, so this runs them in sequence rather than fusing
    /// them into one path.
    public func callAsFunction(video: (x: MLXArray, embeddedTimestep: MLXArray),
                               audio: (x: MLXArray, embeddedTimestep: MLXArray)) throws
        -> (video: Output, audio: Output) {
        (try callAsFunction(video.x, embeddedTimestep: video.embeddedTimestep,
                            stream: .video),
         try callAsFunction(audio.x, embeddedTimestep: audio.embeddedTimestep,
                            stream: .audio))
    }
}

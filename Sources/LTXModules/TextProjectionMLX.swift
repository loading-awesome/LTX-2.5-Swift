// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// The V2 feature extractor in MLX: Gemma's 49 hidden states → the two conditioning
/// streams.
///
/// `LTXFoundation.TextFeatureProjection` is the executable specification — pure Swift,
/// exercised on a hand-checkable fixture — and this is the MLX implementation of the
/// same math at full 188 160-wide scale (188 160 × 4096 ≈ 771 M parameters for the video
/// stream, and again at × 2048 for audio).
///
/// Both exist deliberately. The spec catches a wrong formula in microseconds; this
/// catches a wrong formula *and* a wrong tensor layout, dtype or reduction axis. Neither
/// subsumes the other.
public struct TextProjectionMLX {

    /// `[flat, videoWidth]` — transposed from the checkpoint's `[out, in]` Linear
    /// layout at load time, once, so the hot path is a plain matmul.
    public let videoWeight: MLXArray
    public let audioWeight: MLXArray
    public let videoBias: MLXArray?
    public let audioBias: MLXArray?
    public let spec: TextFeatureProjection

    public enum Failure: Error, CustomStringConvertible {
        case missingWeight(String)
        case shapeMismatch(String, got: [Int], expected: [Int])
        case layerCountMismatch(got: Int, expected: Int)
        case hiddenShapeMismatch(layer: Int, got: [Int], expected: [Int])

        public var description: String {
            switch self {
            case let .missingWeight(name):
                return "projection weight \(name) is absent from the checkpoint"
            case let .shapeMismatch(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .layerCountMismatch(got, expected):
                return "got \(got) hidden states, expected \(expected). Contract 8: the "
                    + "projection consumes every layer PLUS the embedding output, so this "
                    + "count is num_hidden_layers + 1 and a port that passes only the "
                    + "transformer layers is one tensor short"
            case let .hiddenShapeMismatch(layer, got, expected):
                return "hidden state \(layer) is \(got), expected \(expected)"
            }
        }
    }

    /// Load the two projections from a text-encoder checkpoint.
    ///
    /// Weights are transposed on load. MLX's `matmul` wants `[..., in] x [in, out]`
    /// while the checkpoint stores `[out, in]`, and doing that per call
    /// would transpose a 771 M-element array on every forward.
    public init(checkpoint url: URL, topology: TextEncoderTopology,
                header: SafetensorsHeader) throws {
        try self.init(weights: try MLX.loadArrays(url: url), topology: topology, header: header)
    }

    /// Share the encoder's already-loaded map. The projection is four tensors in the
    /// same 26 GB file; a second `MLX.loadArrays` would map it again while Gemma is
    /// still holding the first.
    public init(weights loaded: [String: MLXArray], topology: TextEncoderTopology,
                header: SafetensorsHeader) throws {
        let widths = try topology.projectionWidths(header)
        self.spec = TextFeatureProjection(
            hiddenSize: topology.hiddenSize,
            layerCount: topology.layerCount + 1,      // contract 8: layers + embeddings
            videoWidth: widths.video,
            audioWidth: widths.audio,
            embeddingDim: topology.hiddenSize)

        func weight(_ name: String, expected: [Int]) throws -> MLXArray {
            guard let w = loaded[name] else { throw Failure.missingWeight(name) }
            guard w.shape == expected else {
                throw Failure.shapeMismatch(name, got: w.shape, expected: expected)
            }
            return w.transposed(1, 0)
        }
        let base = "text_embedding_projection."
        self.videoWeight = try weight(base + "video_aggregate_embed.weight",
                                      expected: [widths.video, widths.flat])
        self.audioWeight = try weight(base + "audio_aggregate_embed.weight",
                                      expected: [widths.audio, widths.flat])
        self.videoBias = loaded[base + "video_aggregate_embed.bias"]
        self.audioBias = loaded[base + "audio_aggregate_embed.bias"]
    }

    /// Normalise, flatten and project. Returns `(video, audio)`, each `[B, T, width]`.
    ///
    /// - Parameters:
    ///   - hiddenStates: `layerCount` arrays of `[B, T, hiddenSize]`, in layer order
    ///     with the embedding output first.
    ///   - mask: `[B, T]`, non-zero for valid tokens.
    public func callAsFunction(hiddenStates: [MLXArray], mask: MLXArray) throws
        -> (video: MLXArray, audio: MLXArray) {
        guard hiddenStates.count == spec.layerCount else {
            throw Failure.layerCountMismatch(got: hiddenStates.count,
                                             expected: spec.layerCount)
        }
        let reference = hiddenStates[0].shape
        for (i, h) in hiddenStates.enumerated() where h.shape != reference {
            throw Failure.hiddenShapeMismatch(layer: i, got: h.shape, expected: reference)
        }

        // `[B, T, D, L]` — the layers become a new trailing axis, not a concatenation.
        let stacked = MLX.stacked(hiddenStates, axis: -1)

        // Mean of squares over the HIDDEN axis (-2), per (batch, token, layer) — not
        // over the flattened width.
        //
        // ## The statistics run in fp32, and are not narrowed back
        //
        // The hidden states arrive bf16, so computing the norm in their own dtype is the
        // obvious move. It is the wrong one. The scale is coherent across the 3840
        // elements of a (token, layer) block, so one wrong ulp in it reaches the output
        // at full strength instead of averaging away — and in bf16 the scale is barely
        // well defined. The same formula spelled three defensible ways — `mean(x**2)`,
        // an fp32 accumulation narrowed at the end, `sum / 3840` — disagrees with
        // *itself* by 1.8e-03 relative, and `rsqrt` against `pow(-0.5)` on the same bf16
        // variance by 3.0e-03. In fp32 those spellings collapse onto each other, 5.6e-08
        // apart.
        //
        // So the whole statistic is fp32 and the normalised tensor stays fp32 into the
        // projection below.
        let x = stacked.asType(.float32)
        let variance = MLX.mean(x * x, axis: -2, keepDims: true)
        // `rsqrt`, not `pow(v, -0.5)`. `GemmaAttentionSpec.rmsNormUsesPowNotRsqrt` is
        // about Gemma's own RMS norm, a different module, and does not transfer here. In
        // fp32 the two are bit-identical, so the choice costs nothing; in bf16 they are
        // 3.0e-03 apart, which is the second reason the statistics do not belong in bf16.
        let normed = x * MLX.rsqrt(variance + TextFeatureProjection.varianceEpsilon)

        // Flatten hidden-major, layer-minor: `(d, l) -> d * L + l`. This is what
        // `reshape` on `[B, T, D, L]` does, and it is NOT "concatenate the layers"
        // (which would be `l * D + d`, a permutation of the same values that makes
        // every weight read the wrong input element).
        let b = reference[0], t = reference[1]
        let flat = normed.reshaped([b, t, spec.flatDimension])

        // Zero the padded positions. Done after normalising, and it is only safe to
        // do afterwards because the V2 statistic is per (token, layer) and therefore
        // independent of which positions are padding. V1's masked mean is not.
        let keep = (mask .!= 0).asType(.float32).reshaped([b, t, 1])
        let masked = flat * keep

        // Per-stream rescale, BEFORE each stream's own matmul, with its own factor.
        // One factor reused for both streams leaves one of them wrong by a constant,
        // which nothing downstream is shaped to notice.
        //
        // The **weights are widened rather than the input narrowed**, which is the
        // opposite of `DiTModules.linear`, and deliberately so. Narrowing there keeps a
        // block's arithmetic in step with the bf16 stack around it. Here the input is the
        // fp32 normalisation above, and rounding it back to bf16 would throw away exactly
        // the precision this function goes out of its way to compute.
        let video = MLX.matmul(masked * MLXArray(spec.rescale(for: .video)),
                               videoWeight.asType(.float32))
        let audio = MLX.matmul(masked * MLXArray(spec.rescale(for: .audio)),
                               audioWeight.asType(.float32))

        return (video: videoBias.map { video + $0.asType(.float32) } ?? video,
                audio: audioBias.map { audio + $0.asType(.float32) } ?? audio)
    }
}

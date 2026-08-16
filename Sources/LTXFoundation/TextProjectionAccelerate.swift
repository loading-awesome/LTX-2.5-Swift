// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Accelerate
import Foundation

/// The text feature projection on Accelerate: Gemma's 49 hidden states → two
/// conditioning streams.
///
/// The math is `TextFeatureProjection`'s, checked there on a fixture small enough to
/// verify by hand. This runs the same rules at the full 188 160-wide scale.
///
/// ## Why Accelerate and not MLX
///
/// MLX is the port's intended runtime, and it is currently unusable in this
/// checkout: mlx-swift 0.31.2 through SwiftPM builds no `default.metallib`, so every
/// MLX entry point fails at device resolution — including with `Device(.cpu)`, since
/// the default device is resolved during initialisation. See the task filed for the
/// packaging diagnosis.
///
/// Rather than let that block the first real numerical work, the projection is
/// implemented here on vDSP. This is not a throwaway: Accelerate is a legitimate path
/// for a single large matmul on Apple silicon, and having the projection in a
/// dependency-free target means it stays testable no matter what happens to the MLX
/// toolchain. When MLX works, it is checked against the same fixture, and any
/// disagreement between the two implementations is itself a finding.
public struct TextProjectionAccelerate {

    public let spec: TextFeatureProjection
    /// `[flat, width]`, transposed from the checkpoint's `[width, flat]` on load.
    private let videoWeight: [Float]
    private let audioWeight: [Float]
    private let videoBias: [Float]?
    private let audioBias: [Float]?

    public enum Failure: Error, CustomStringConvertible {
        case layerCountMismatch(got: Int, expected: Int)
        case hiddenSizeMismatch(got: Int, expected: Int)
        case weightShape(String, got: [Int], expected: [Int])

        public var description: String {
            switch self {
            case let .layerCountMismatch(got, expected):
                return "got \(got) hidden states, expected \(expected). Contract 8: the "
                    + "projection consumes every layer PLUS the embedding output, so a "
                    + "port passing only the transformer layers is one tensor short"
            case let .hiddenSizeMismatch(got, expected):
                return "hidden states are \(got) wide, expected \(expected)"
            case let .weightShape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            }
        }
    }

    /// Load and transpose both projections.
    ///
    /// The transpose happens once here rather than per call: the operand is 771 M
    /// elements and `vDSP_mmul` wants both matrices row-major, so paying once at load
    /// is the difference between a warm cache and a transposed read per call.
    public init(reader: SafetensorsReader, spec: TextFeatureProjection) throws {
        self.spec = spec
        let base = "text_embedding_projection."

        func load(_ name: String, width: Int) throws -> [Float] {
            let full = base + name + ".weight"
            let shape = try reader.shape(full)
            guard shape == [width, spec.flatDimension] else {
                throw Failure.weightShape(full, got: shape,
                                          expected: [width, spec.flatDimension])
            }
            let rowMajor = try reader.floats(full)      // [width][flat]
            var transposed = [Float](repeating: 0, count: rowMajor.count)
            // vDSP's transpose is the readable and fast way to say this.
            vDSP_mtrans(rowMajor, 1, &transposed, 1,
                        vDSP_Length(spec.flatDimension), vDSP_Length(width))
            return transposed                            // [flat][width]
        }

        self.videoWeight = try load("video_aggregate_embed", width: spec.videoWidth)
        self.audioWeight = try load("audio_aggregate_embed", width: spec.audioWidth)
        self.videoBias = try? reader.floats(base + "video_aggregate_embed.bias")
        self.audioBias = try? reader.floats(base + "audio_aggregate_embed.bias")
    }

    /// Project one sequence.
    ///
    /// - Parameters:
    ///   - hiddenStates: `layerCount` arrays, each `tokens * hiddenSize` row-major —
    ///     the embedding output first, then each transformer layer.
    ///   - mask: `tokens` values, non-zero for valid tokens.
    ///   - tokens: sequence length.
    /// - Returns: row-major `[tokens][width]` for each stream.
    public func project(hiddenStates: [[Float]], mask: [Float], tokens: Int) throws
        -> (video: [Float], audio: [Float]) {
        guard hiddenStates.count == spec.layerCount else {
            throw Failure.layerCountMismatch(got: hiddenStates.count,
                                             expected: spec.layerCount)
        }
        let d = spec.hiddenSize, l = spec.layerCount, flat = spec.flatDimension
        for h in hiddenStates where h.count != tokens * d {
            throw Failure.hiddenSizeMismatch(got: h.count / max(tokens, 1), expected: d)
        }

        // Normalise and flatten, one token at a time. `[tokens][flat]` row-major.
        var normalized = [Float](repeating: 0, count: tokens * flat)
        var inverseRMS = [Float](repeating: 0, count: l)

        for t in 0..<tokens {
            if mask[t] == 0 { continue }        // padded rows stay zero

            // Mean of squares over the HIDDEN axis, separately per layer. Not over the
            // flattened 188160 vector, and not shared across layers -- each layer of
            // each token carries its own scale, which is what makes the 49 states
            // comparable at all after `layer_scalar` rescaled each one differently
            // (contract 13).
            //
            // Float, not bf16, and that is a measured choice rather than an accident of
            // this target having no bf16 type. Computing the statistic in bf16 lands
            // further from the right answer, not closer: a bf16 statistic is not
            // reproducible to the ulp, while the per-layer scale is coherent across 3840
            // elements. The measurements are in `TextProjectionMLX`. This implementation
            // and the MLX one must make the SAME choice -- they agree to 2.3e-07, and one
            // narrowing while the other does not would surface as a 2e-03 disagreement
            // between two supposedly identical formulas.
            for layer in 0..<l {
                let h = hiddenStates[layer]
                var sumSquares: Float = 0
                h.withUnsafeBufferPointer { p in
                    let row = p.baseAddress! + t * d
                    // Sum of squares. vDSP rather than `cblas_sdot`: the legacy CBLAS
                    // declarations are deprecated as of macOS 13.3 in favour of an
                    // ILP64 interface that SwiftPM cannot select without `unsafeFlags`,
                    // and `unsafeFlags` would make this package unusable as a
                    // dependency. vDSP has no such problem and is the same arithmetic.
                    vDSP_svesq(row, 1, &sumSquares, vDSP_Length(d))
                }
                let variance = sumSquares / Float(d)
                inverseRMS[layer] = 1.0
                    / (variance + TextFeatureProjection.varianceEpsilon).squareRoot()
            }

            // Hidden-major, layer-minor: (hidden, layer) -> hidden * L + layer. This is
            // what `reshape` on [B, T, D, L] produces. "Concatenate the layers" gives
            // layer * D + hidden, a permutation of the same numbers that makes every
            // weight read the wrong input element -- and no shape check sees it.
            normalized.withUnsafeMutableBufferPointer { out in
                let rowBase = out.baseAddress! + t * flat
                for layer in 0..<l {
                    let scale = inverseRMS[layer]
                    hiddenStates[layer].withUnsafeBufferPointer { p in
                        let row = p.baseAddress! + t * d
                        for hidden in 0..<d {
                            rowBase[hidden * l + layer] = row[hidden] * scale
                        }
                    }
                }
            }
        }

        // Per-stream rescale then matmul, each with its OWN factor. Reusing one leaves
        // a stream wrong by a constant, which `cos` cannot see.
        func stream(_ weight: [Float], width: Int, rescale: Float, bias: [Float]?)
            -> [Float] {
            var scaled = normalized
            var factor = rescale
            vDSP_vsmul(normalized, 1, &factor, &scaled, 1, vDSP_Length(normalized.count))

            var out = [Float](repeating: 0, count: tokens * width)
            // [tokens, flat] x [flat, width] -> [tokens, width], row-major.
            vDSP_mmul(scaled, 1, weight, 1, &out, 1,
                      vDSP_Length(tokens), vDSP_Length(width), vDSP_Length(flat))
            if let bias {
                out.withUnsafeMutableBufferPointer { o in
                    for t in 0..<tokens {
                        let row = o.baseAddress! + t * width
                        vDSP_vadd(row, 1, bias, 1, row, 1, vDSP_Length(width))
                    }
                }
            }
            return out
        }

        return (video: stream(videoWeight, width: spec.videoWidth,
                              rescale: spec.rescale(for: .video), bias: videoBias),
                audio: stream(audioWeight, width: spec.audioWidth,
                              rescale: spec.rescale(for: .audio), bias: audioBias))
    }
}

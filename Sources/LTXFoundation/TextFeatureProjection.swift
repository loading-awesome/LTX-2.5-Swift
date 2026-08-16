// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The text-encoder-side projection: per-token RMS norm over Gemma's layer stack,
/// then a per-stream rescale, then two Linears.
///
/// This is contract 8 in executable form. It carries no MLX and no weights, so it
/// runs in microseconds and its errors cannot hide behind a 22 B checkpoint.
///
/// Over hidden states stacked as `[B, T, D, L]` — batch, token, hidden, layer:
///
/// * the variance is a mean of squares over `D`, taken per `(token, layer)`;
/// * `normed = encoded * rsqrt(variance + 1e-6)`, then flattened to `[B, T, D * L]`;
/// * padded positions are zeroed;
/// * each stream rescales that same vector by its own factor — 4096/3840 for video,
///   2048/3840 for audio — before its own Linear.
///
/// ## The three ways to reimplement this wrongly
///
/// 1. **Normalising over the wrong axis.** The variance is a mean of squares over
///    the *hidden* dimension, computed separately for every (token, layer) pair —
///    not over the flattened 188 160-vector, and not per token across all layers.
///    Each layer of each token is normalised on its own scale.
/// 2. **Flattening layer-major.** `reshape` on `[B, T, D, L]` is hidden-major and
///    layer-minor: element `(d, l)` lands at `d * L + l`. "Concatenate the layers"
///    is the intuitive reading and gives `l * D + d`, a permutation of the same
///    188 160 numbers. The Linear then reads every weight against the wrong input
///    element, and nothing about the shapes complains.
/// 3. **Rescaling once.** The same normed vector is scaled by
///    `sqrt(4096/3840) ≈ 1.0328` for video and `sqrt(2048/3840) ≈ 0.7303` for
///    audio, *before* each stream's own Linear. Computing it once and reusing it
///    makes one stream wrong by a constant factor — which a cosine similarity check
///    cannot see at all.
public struct TextFeatureProjection: Sendable, Equatable {

    /// Gemma's hidden size — the axis the variance is taken over. 3840 here.
    public let hiddenSize: Int
    /// Layers consumed: transformer layers **+ 1** for the embedding output.
    /// Measured as 49 on this checkpoint (48 + 1).
    public let layerCount: Int
    /// Video stream width, `num_attention_heads * attention_head_dim`.
    public let videoWidth: Int
    /// Audio stream width, `audio_num_attention_heads * audio_attention_head_dim`.
    public let audioWidth: Int

    /// Denominator of the per-stream rescale.
    ///
    /// Equal to `hiddenSize` on every real checkpoint, and kept separate anyway
    /// because **the two have independent sources**: the variance axis comes from the
    /// hidden-state tensor's shape, while the rescale's denominator comes from the
    /// Gemma text config's `hidden_size`. Folding them into one field means a
    /// checkpoint where they disagreed would be silently rescaled by the wrong
    /// factor — and a wrong constant factor is invisible to a cosine check.
    ///
    /// It also stops a fixture from lying: a fixture that normalises over a synthetic
    /// `D = 4` while rescaling by the real 3840 is not expressible with one field.
    public let embeddingDim: Int

    /// Epsilon added to the variance, not to its root.
    public static let varianceEpsilon: Float = 1e-6

    public init(hiddenSize: Int = 3840, layerCount: Int = 49,
                videoWidth: Int = 4096, audioWidth: Int = 2048,
                embeddingDim: Int? = nil) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.videoWidth = videoWidth
        self.audioWidth = audioWidth
        self.embeddingDim = embeddingDim ?? hiddenSize
    }

    /// `hidden_size * (num_hidden_layers + 1)` — contract 8's derived input width.
    ///
    /// Never hardcode this. The literal `188160` is only right for one checkpoint's
    /// layer count; on a checkpoint of a different depth it reads every weight
    /// against the wrong input width.
    public var flatDimension: Int { hiddenSize * layerCount }

    /// Where element `(hidden, layer)` lands in the flattened vector.
    ///
    /// `hidden * layerCount + layer` — hidden-major, layer-minor. See trap 2 in the
    /// type documentation: the plausible alternative is a permutation that no shape
    /// check rejects.
    public func flatIndex(hidden: Int, layer: Int) -> Int {
        hidden * layerCount + layer
    }

    /// Per-stream rescale applied *before* that stream's Linear.
    ///
    /// `sqrt(streamWidth / embeddingDim)` — note the denominator, see `embeddingDim`.
    public func rescale(for stream: TextConditioningLayout.Stream) -> Float {
        let target = stream == .video ? videoWidth : audioWidth
        return (Float(target) / Float(embeddingDim)).squareRoot()
    }

    /// Normalise and flatten one token's layer stack.
    ///
    /// - Parameter tokenLayers: `[hidden][layer]`, i.e. `layers[h][l]` is Gemma
    ///   layer `l`'s activation for hidden unit `h` at this token.
    /// - Returns: the flattened `hidden * layerCount + layer` vector.
    public func normalizeToken(_ tokenLayers: [[Float]]) -> [Float] {
        precondition(tokenLayers.count == hiddenSize,
                     "expected \(hiddenSize) hidden units, got \(tokenLayers.count)")

        // Mean of squares over the HIDDEN dimension, separately per layer. Computed
        // in Float even though the surrounding stack runs bf16, and for a measured
        // reason: a bf16 norm statistic is not a fixed target. Three defensible
        // spellings of the same bf16 formula disagree with each other by 1.8e-03 to
        // 2.5e-03, and `rsqrt` against `pow(-0.5)` on the same bf16 variance by
        // 3.0e-03 — as large as the whole bf16-versus-fp32 gap — while in fp32 all of
        // them are bit-identical. The numbers, and what narrowing costs, are in
        // `TextProjectionMLX`.
        var inverseRMS = [Float](repeating: 0, count: layerCount)
        for l in 0..<layerCount {
            var sumSquares: Float = 0
            for h in 0..<hiddenSize {
                let v = tokenLayers[h][l]
                sumSquares += v * v
            }
            let variance = sumSquares / Float(hiddenSize)
            inverseRMS[l] = 1.0 / (variance + Self.varianceEpsilon).squareRoot()
        }

        var flat = [Float](repeating: 0, count: flatDimension)
        for h in 0..<hiddenSize {
            for l in 0..<layerCount {
                flat[flatIndex(hidden: h, layer: l)] = tokenLayers[h][l] * inverseRMS[l]
            }
        }
        return flat
    }

    /// Normalise a whole sequence, zeroing padded positions.
    ///
    /// Padding is zeroed **after** normalising, and the padded positions do not
    /// participate in any statistic — the per-token, per-layer variance makes them
    /// independent by construction. That is a genuine difference from the V1
    /// extractor, whose normalisation is a masked mean and range over the whole
    /// batch, so V1 *is* sensitive to which positions are padding. Do not port one
    /// as the other.
    public func normalize(sequence: [[[Float]]], mask: [Bool]) -> [[Float]] {
        precondition(sequence.count == mask.count,
                     "sequence has \(sequence.count) tokens but the mask has \(mask.count)")
        return zip(sequence, mask).map { token, isValid in
            isValid ? normalizeToken(token) : [Float](repeating: 0, count: flatDimension)
        }
    }

    /// The rescaled input to one stream's Linear.
    public func projectionInput(normalized: [Float],
                                stream: TextConditioningLayout.Stream) -> [Float] {
        let s = rescale(for: stream)
        return normalized.map { $0 * s }
    }
}

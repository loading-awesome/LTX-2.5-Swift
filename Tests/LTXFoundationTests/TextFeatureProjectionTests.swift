// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// The per-token RMS normalisation and the per-stream rescale, pinned to concrete
/// values.
///
/// The fixture is deliberately tiny (D=4, L=2, 3 tokens) and its inputs are a
/// linear ramp rather than random draws, so every expected number can be recomputed
/// by hand from the formula — and `varianceAxisByHand` does exactly that. A fixture
/// nobody can check by hand only proves that two implementations agree, not that
/// either is right.
@Suite("Text feature projection")
struct TextFeatureProjectionTests {

    /// Exactly the fixture's geometry: variance over a synthetic `D = 4` across 2
    /// layers, but rescaled by the real `embedding_dim` of 3840.
    let fixture = TextFeatureProjection(hiddenSize: 4, layerCount: 2,
                                        videoWidth: 4096, audioWidth: 2048,
                                        embeddingDim: 3840)

    /// `encoded[d][l]` for the fixture's token 0 and token 1. Values are the ramp
    /// `i/10 - 1` for `i` in `0..<24`, shaped `[1, 3, 4, 2]`.
    let token0: [[Float]] = [[-1.0, -0.9], [-0.8, -0.7], [-0.6, -0.5], [-0.4, -0.3]]
    let token1: [[Float]] = [[-0.2, -0.1], [0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]
    let token2: [[Float]] = [[0.6, 0.7], [0.8, 0.9], [1.0, 1.1], [1.2, 1.3]]

    /// Tolerance for the fp32 expected values. This is exact-ish arithmetic on tiny
    /// inputs, so all that is being allowed for is float noise.
    let eps: Float = 2e-6

    @Test("matches the reference's per-token RMS norm, value for value")
    func matchesReferenceNorm() {
        // Expected: normed[0, 0, :] and normed[0, 1, :]
        let expected0: [Float] = [-1.3608263731002808, -1.405562162399292,
                                  -1.0886610746383667, -1.093214988708496,
                                  -0.8164958357810974, -0.780867874622345,
                                  -0.5443305373191833, -0.46852073073387146]
        let expected1: [Float] = [-0.8164897561073303, -0.33333155512809753,
                                  0.0, 0.33333155512809753,
                                  0.8164899945259094, 0.9999943375587463,
                                  1.6329795122146606, 1.6666574478149414]

        let got0 = fixture.normalizeToken(token0)
        let got1 = fixture.normalizeToken(token1)
        #expect(got0.count == 8)
        for (g, e) in zip(got0, expected0) { #expect(abs(g - e) < eps) }
        for (g, e) in zip(got1, expected1) { #expect(abs(g - e) < eps) }
    }

    /// The formula, recomputed by hand, so the fixture is not the only witness.
    /// Token 0, layer 0 spans the hidden axis: [-1.0, -0.8, -0.6, -0.4].
    /// mean of squares = (1.00 + 0.64 + 0.36 + 0.16) / 4 = 0.54
    /// rsqrt(0.54 + 1e-6) = 1.3608263…  and  -1.0 x that = -1.3608263…
    @Test("the normalisation is a mean of squares over HIDDEN, per layer")
    func varianceAxisByHand() {
        let layer0 = token0.map { $0[0] }                      // across hidden
        let meanSquares = layer0.map { $0 * $0 }.reduce(0, +) / Float(layer0.count)
        #expect(abs(meanSquares - 0.54) < 1e-6)

        let inverseRMS = 1.0 / (meanSquares + TextFeatureProjection.varianceEpsilon)
            .squareRoot()
        #expect(abs(inverseRMS - 1.3608263731002808) < eps)

        // And that is exactly what the implementation produces at (d=0, l=0).
        let got = fixture.normalizeToken(token0)
        #expect(abs(got[fixture.flatIndex(hidden: 0, layer: 0)]
                    - (-1.0 * inverseRMS)) < eps)

        // Layer 1 has its OWN scale: [-0.9, -0.7, -0.5, -0.3], mean sq 0.41.
        let layer1 = token0.map { $0[1] }
        let ms1 = layer1.map { $0 * $0 }.reduce(0, +) / Float(layer1.count)
        #expect(abs(ms1 - 0.41) < 1e-6)
        #expect(abs(ms1 - meanSquares) > 0.1)   // the two layers differ, as intended
    }

    /// Trap 2. The flattening puts `(d, l)` at `d * L + l`: `encoded[0,0,1,0] = -0.8`
    /// normalises to -1.0886610746383667 and lands at flat index 2.
    @Test("flattening is hidden-major, layer-minor — not concatenated layers")
    func layoutIsHiddenMajor() {
        #expect(fixture.flatIndex(hidden: 1, layer: 0) == 2)
        let got = fixture.normalizeToken(token0)
        #expect(abs(got[2] - (-1.0886610746383667)) < eps)

        // The plausible wrong reading, spelled out: "concatenate the layers" puts
        // (d=1, l=0) at l * D + d = 1, which holds a DIFFERENT value. Both layouts
        // are 8 elements of the same 8 numbers, so no shape check separates them.
        #expect(abs(got[1] - (-1.0886610746383667)) > 0.1)

        let layerMajorIndex = 0 * 4 + 1     // l * D + d
        #expect(layerMajorIndex != fixture.flatIndex(hidden: 1, layer: 0))
    }

    @Test("per-stream rescale factors, and they differ")
    func rescaleFactors() {
        let g = TextFeatureProjection()      // real geometry: 3840 / 4096 / 2048
        #expect(abs(g.rescale(for: .video) - 1.0327955589886444) < eps)
        #expect(abs(g.rescale(for: .audio) - 0.7302967433402214) < eps)
        // Trap 3: one scale reused for both streams is wrong by a constant factor
        // in one of them, and a cosine similarity cannot see a constant factor.
        #expect(g.rescale(for: .video) != g.rescale(for: .audio))
    }

    @Test("rescaled outputs match the reference for both streams")
    func matchesReferenceRescale() {
        let normed = fixture.normalizeToken(token0)
        let video = fixture.projectionInput(normalized: normed, stream: .video)
        let audio = fixture.projectionInput(normalized: normed, stream: .audio)

        // Expected: rescaled_video[0, 0, :2] and rescaled_audio[0, 0, :2]
        #expect(abs(video[0] - (-1.405455470085144)) < eps)
        #expect(abs(video[1] - (-1.4516583681106567)) < eps)
        #expect(abs(audio[0] - (-0.9938070774078369)) < eps)
        #expect(abs(audio[1] - (-1.0264774560928345)) < eps)
    }

    @Test("padded tokens are zeroed, and do not affect valid ones")
    func padding() {
        // The fixture's mask is [1, 1, 0]: token 2 is padding.
        let out = fixture.normalize(sequence: [token0, token1, token2],
                                   mask: [true, true, false])
        #expect(out.count == 3)
        #expect(out[2].allSatisfy { $0 == 0.0 })

        // And token 0's values are unchanged by the presence of padding, because the
        // statistic is per-token. A masked mean over the batch would be sensitive to
        // padding, and the two are easy to mistake for each other.
        let alone = fixture.normalizeToken(token0)
        for (a, b) in zip(out[0], alone) { #expect(a == b) }
    }

    /// The variance axis and the rescale denominator come from different places — the
    /// tensor's shape and the Gemma config respectively — so the type keeps them
    /// separate even though every real checkpoint has them equal. This test is the
    /// reason: it is what caught the conflation.
    @Test("embeddingDim defaults to hiddenSize but is independently settable")
    func embeddingDimIndependence() {
        let defaulted = TextFeatureProjection(hiddenSize: 3840)
        #expect(defaulted.embeddingDim == 3840)
        #expect(abs(defaulted.rescale(for: .video) - 1.0327955589886444) < eps)

        // Fixture geometry: normalise over 4, rescale by 3840.
        #expect(fixture.hiddenSize == 4)
        #expect(fixture.embeddingDim == 3840)
        #expect(abs(fixture.rescale(for: .video) - 1.0327955589886444) < eps)

        // If they were conflated, the fixture's factor would be sqrt(4096/4) = 32.
        #expect(abs(fixture.rescale(for: .video) - 32.0) > 30.0)
    }

    /// Contract 8: the input width is derived, never written down. 3840 x 49.
    @Test("flat dimension is derived and equals the measured 188160")
    func flatDimension() {
        let g = TextFeatureProjection()
        #expect(g.flatDimension == 188160)
        #expect(g.flatDimension == 3840 * 49)

        // A checkpoint with a different layer count must change this without an edit.
        let other = TextFeatureProjection(hiddenSize: 2048, layerCount: 33)
        #expect(other.flatDimension == 67584)
    }
}

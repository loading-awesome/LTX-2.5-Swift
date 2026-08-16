// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// The RoPE parameters and attention scalars for the two attention kinds. The
/// values that pin each one are also recomputed by hand here, so a transcription
/// error has nowhere to hide.
@Suite("Gemma attention spec")
struct GemmaAttentionSpecTests {

    let spec = GemmaAttentionSpec.ltx25

    /// **Relative**, not absolute, and that is not slack. The expected frequencies are
    /// fp32 while this implementation computes in `Double`, so the two agree to about
    /// 1e-8 absolute near 0.93 and no closer.
    ///
    /// A fixed absolute bound would also be wrong across the range: these frequencies
    /// span 1.0 down to 1.07e-04, so it is either meaningless at the top or unmeetable
    /// at the bottom.
    let relativeTolerance = 1e-6

    /// Assert `got` matches an expected fp32 value to `relativeTolerance`.
    private func expectClose(_ got: Double, _ reference: Double,
                            _ comment: Comment? = nil,
                            sourceLocation: SourceLocation = #_sourceLocation) {
        let scale = max(abs(reference), Double.leastNormalMagnitude)
        #expect(abs(got - reference) / scale < relativeTolerance, comment,
                sourceLocation: sourceLocation)
    }

    // MARK: - RoPE

    @Test("sliding frequencies: 128 entries, all rotating, base 1e4")
    func slidingFrequencies() {
        let inv = spec.inverseFrequencies(for: .sliding)
        #expect(inv.count == 128)
        #expect(spec.rotatingFrequencyCount(for: .sliding) == 128)

        expectClose(inv[0], 1.0)
        expectClose(inv[1], 0.9305720329284668)
        expectClose(inv[2], 0.8659643530845642)
        // The last two non-zero entries: four orders of magnitude below the
        // first, which is why the bound is relative.
        expectClose(inv[126], 0.00011547819303814322)
        expectClose(inv[127], 0.00010746077896328643)
    }

    /// The denominator, recomputed independently: default RoPE divides by `dim`,
    /// which is 256 for the sliding heads. `10000^(-2/256) = 10000^(-1/128)`.
    @Test("sliding denominator is dim (256), verified by hand")
    func slidingDenominatorByHand() {
        let byHand = 1.0 / pow(10_000.0, 1.0 / 128.0)
        expectClose(byHand, 0.9305720329284668)
        // Against our own value this IS exact -- same formula, same precision.
        #expect(spec.inverseFrequencies(for: .sliding)[1] == byHand)
    }

    @Test("full frequencies: 256 entries, only 64 rotating, 192 zeros")
    func fullFrequencies() {
        let inv = spec.inverseFrequencies(for: .full)
        #expect(inv.count == 256)              // headDim / 2, NOT the rotated width
        #expect(spec.rotatingFrequencyCount(for: .full) == 64)   // 0.25 * 512 / 2

        expectClose(inv[0], 1.0)
        expectClose(inv[1], 0.9474635124206543)
        // The boundary: [62] and [63] non-zero, then zeros all the way out.
        expectClose(inv[62], 0.03522694483399391)
        expectClose(inv[63], 0.03337624669075012)
        #expect(inv[64] == 0.0)
        #expect(inv[255] == 0.0)
        #expect(inv[64...].allSatisfy { $0 == 0.0 })
    }

    /// The trap: `proportional` divides by the **full** head dim (512), not by the
    /// rotated width (128). `1000000^(-2/512) = 1000000^(-1/256)`.
    @Test("full denominator is the FULL head dim (512), not the rotated width")
    func fullDenominatorByHand() {
        let correct = 1.0 / pow(1_000_000.0, 1.0 / 256.0)
        expectClose(correct, 0.9474635124206543)
        #expect(spec.inverseFrequencies(for: .full)[1] == correct)

        // What the `default` formula would give if applied to a partial spec:
        // dim = 512 * 0.25 = 128, so 1000000^(-2/128) -- a completely different value.
        let wrong = 1.0 / pow(1_000_000.0, 2.0 / 128.0)
        #expect(abs(wrong - correct) > 0.1)
    }

    /// Zero frequency means identity rotation, which is how the non-rotated tail is
    /// expressed. `cos(0) = 1`, `sin(0) = 0`, so `x*1 + rotate_half(x)*0 == x`.
    @Test("a zero frequency is an identity rotation")
    func zeroFrequencyIsIdentity() {
        let angle = 0.0 * 7.0                  // any position times a zero frequency
        #expect(cos(angle) == 1.0)
        #expect(sin(angle) == 0.0)

        let x = 0.42
        let rotated = x * cos(angle) + 0.99 * sin(angle)   // rotate_half term ignored
        #expect(abs(rotated - x) < 1e-15)
    }

    @Test("the two kinds do not share RoPE parameters")
    func kindsDiffer() {
        #expect(spec.sliding.theta != spec.full.theta)
        #expect(spec.sliding.headDim != spec.full.headDim)
        #expect(spec.sliding.ropeType != spec.full.ropeType)
        #expect(spec.inverseFrequencies(for: .sliding).count
                != spec.inverseFrequencies(for: .full).count)
    }

    // MARK: - The scalars that are easy to get wrong

    /// The attention scale is a fixed 1.0, not derived from the head dimension.
    @Test("attention scaling is 1.0 — no 1/sqrt(head_dim)")
    func noSoftmaxScale() {
        #expect(GemmaAttentionSpec.attentionScaling == 1.0)

        // What the conventional scale would be, and how far off it is. The measured
        // k_norm weights (~0.061 for the 512-dim heads) carry this instead.
        let conventional512 = 1.0 / Double(512).squareRoot()
        #expect(abs(conventional512 - 0.044194173824159216) < 1e-12)
        #expect(Double(GemmaAttentionSpec.attentionScaling) / conventional512 > 20.0)
    }

    @Test("RMSNorm multiplies by weight, without adding one")
    func rmsNormConvention() {
        #expect(GemmaAttentionSpec.rmsNormAddsOneToWeight == false)
        #expect(GemmaAttentionSpec.rmsNormAccumulatesInFloat32)
        #expect(GemmaAttentionSpec.rmsNormUsesPowNotRsqrt)
    }

    // MARK: - Projection order

    /// The operation with no weights, and therefore no trace in the checkpoint.
    @Test("V is normalised by a weightless norm, and never rotated")
    func valueNorm() {
        let order = GemmaAttentionSpec.projectionOrder
        #expect(order.valueIsNormalized)
        #expect(order.valueNormHasWeight == false)
        #expect(order.valueIsRotated == false)
        #expect(order.queryIsNormalizedThenRotated)
        #expect(order.keyIsNormalizedThenRotated)
    }

    /// When K is reused as V, the alias is of the *pre-norm* projection.
    @Test("shared V aliases the pre-norm key, not the finished key")
    func sharedValueAliasing() {
        #expect(GemmaAttentionSpec.projectionOrder.sharedValueAliasesPreNormKey)

        // Modelled numerically: with k_norm's scale ~0.061 folded in, taking V from
        // the finished K instead of the projection scales V by that factor and adds a
        // rotation. Neither is visible in the shape.
        let projection = 1.0
        let kNormScale = 0.0605
        let vFromProjection = projection            // then v_norm, scale-free
        let vFromFinishedKey = projection * kNormScale
        #expect(abs(vFromProjection - vFromFinishedKey) > 0.9)
    }

    @Test("only the global layers share K as V")
    func sharingByKind() {
        #expect(!spec.projectsValueSeparately(for: .full, keyEqualsValue: true))
        #expect(spec.projectsValueSeparately(for: .sliding, keyEqualsValue: true))
        // With the flag off, both project separately.
        #expect(spec.projectsValueSeparately(for: .full, keyEqualsValue: false))
    }
}

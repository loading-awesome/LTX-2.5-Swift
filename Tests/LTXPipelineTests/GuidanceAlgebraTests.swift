// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXPipeline

/// `Sampler.combine` checked term by term, with no tolerance anywhere.
///
/// A render exercises the guidance stack at exactly one point in its parameter space, and
/// several of the terms it combines are large enough to hide each other. This suite drives
/// `combine` directly at values chosen so that **every intermediate is exactly
/// representable**, which makes a wrong sign or a swapped operand a wrong *integer* rather
/// than a small difference.
///
/// The recipe: the four passes are `cond`, `cond - 1`, `cond - 10`, `cond - 100`, so the
/// three deltas are 1, 10 and 100 and each term's contribution is legible in its own
/// decimal place. Every coefficient is a small integer, `cond` is small integers, and fp32
/// carries all of it without rounding. There is no tolerance here and there should not be:
/// an identity that holds exactly is stronger than any bound, and no bound can be tighter
/// than the storage floor anyway.
///
/// What this catches that the replay cannot:
///
/// * **STG spelled like the other two.** `stgScale * (cond - ptb)` against
///   `(scale - 1) * (...)`. At the replay's `stgScale = 1.0` the wrong spelling gives a
///   coefficient of 0 and merely drops the term; at `stgScale = 5` it is off by a factor.
/// * **A swapped operand.** `(uncond - cond)` reads identically and negates one term.
/// * **The rescale inverted.** `std(pred) / std(cond)` is the same expression backwards
///   and, at the replay's numbers, moves the result in the same direction.
/// * **The rescale applied before the sum**, or to `cond` rather than to `pred`.
@Suite("Guidance algebra")
struct GuidanceAlgebraTests {

    /// Eight small integers. Everything downstream stays on the integer grid, so fp32
    /// arithmetic is exact and equality is the assertion rather than a stand-in for one.
    static let base: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]

    static func array(_ values: [Float]) -> MLXArray { MLXArray(values, [1, 8, 1]) }

    static func values(_ x: MLXArray) -> [Float] {
        x.asType(.float32).flattened().asArray(Float.self)
    }

    /// The four passes, offset so each delta lands in its own decimal place.
    static func passes(cond: [Float] = base) -> [Sampler.Pass: MLXArray] {
        [.conditional: array(cond),
         .unconditional: array(cond.map { $0 - 1 }),
         .perturbed: array(cond.map { $0 - 10 }),
         .modality: array(cond.map { $0 - 100 })]
    }

    static func combine(_ params: GuidanceParams,
                        _ passes: [Sampler.Pass: MLXArray]) throws -> [Float] {
        let sampler = Sampler(video: params, audio: params)
        return values(try sampler.combine(passes, params: params, stream: "video"))
    }

    // MARK: - The four terms

    /// Every term, at a distinct non-neutral scale, to the exact integer.
    ///
    /// ```
    /// pred = cond
    ///      + (cfg - 1)      * (cond - uncond)     = 2 * 1   =   2
    ///      + stg            * (cond - perturbed)  = 5 * 10  =  50
    ///      + (modality - 1) * (cond - modality)   = 3 * 100 = 300
    /// ```
    ///
    /// so `pred == cond + 352`, and the three coefficients are `2`, `5`, `3` — pairwise
    /// distinct, and distinct from the scales that produced them, so a term that read the
    /// wrong scale is a different integer too.
    @Test("each guidance term contributes exactly its coefficient times its delta")
    func eachTermContributesExactly() throws {
        let params = GuidanceParams(cfgScale: 3, stgScale: 5, stgBlocks: [28],
                                    rescaleScale: 0, modalityScale: 4)
        let got = try Self.combine(params, Self.passes())
        let want = Self.base.map { $0 + 352 }
        print("cfg 3 / stg 5 / modality 4: \(got.prefix(4)) against \(want.prefix(4))")
        #expect(got == want,
                Comment(rawValue: "combine gave \(got), expected \(want). The deltas are "
                        + "1, 10 and 100, so the digit that moved names the wrong term: "
                        + "units = CFG, tens = STG, hundreds = modality."))
    }

    /// STG is `scale`; CFG and modality are `scale - 1`. The asymmetry, isolated.
    ///
    /// Three runs that differ only in *which* term is switched on, each at scale 2. CFG and
    /// modality contribute `1 x delta`; STG contributes `2 x delta`. Writing all three
    /// alike makes the STG row read 10 instead of 20 — and at the replay's `stgScale` of
    /// 1.0 it would read 0, which is why the replay cannot see this.
    @Test("STG uses the scale and the other two use scale minus one")
    func stgIsTheOddOneOut() throws {
        let neutral = GuidanceParams(cfgScale: 1, stgScale: 0, modalityScale: 1)
        var cfg = neutral; cfg.cfgScale = 2
        var stg = neutral; stg.stgScale = 2
        var mod = neutral; mod.modalityScale = 2

        // delta 1 at coefficient (2 - 1) = 1
        #expect(try Self.combine(cfg, Self.passes()) == Self.base.map { $0 + 1 })
        // delta 10 at coefficient 2 -- NOT (2 - 1)
        #expect(try Self.combine(stg, Self.passes()) == Self.base.map { $0 + 20 },
                Comment(rawValue: "STG at scale 2 over a delta of 10 must contribute 20. "
                        + "10 means it was written as (scale - 1) like the other two; "
                        + "the STG delta is `scale * (pos - ptb)` "
                        + "and its neutral value is 0, not 1."))
        // delta 100 at coefficient (2 - 1) = 1
        #expect(try Self.combine(mod, Self.passes()) == Self.base.map { $0 + 100 })

        // And each neutral value really is neutral: the terms vanish exactly.
        #expect(try Self.combine(neutral, Self.passes()) == Self.base,
                Comment(rawValue: "cfg 1 / stg 0 / modality 1 must return cond untouched"))
    }

    /// The operands are `(cond - other)`, never the reverse.
    ///
    /// A negated term is the single easiest mistake to make and the hardest to see: it
    /// produces a plausible render that is steered away from the prompt instead of toward
    /// it. Here it is the difference between `+352` and `-348`.
    @Test("the deltas are cond minus the other pass, not the other way round")
    func deltasPointFromCond() throws {
        let params = GuidanceParams(cfgScale: 3, stgScale: 5, rescaleScale: 0,
                                    modalityScale: 4)
        let forward = try Self.combine(params, Self.passes())
        // The same passes with cond and the others exchanged: cond becomes the smallest
        // rather than the largest, so every delta flips sign.
        let mirrored: [Sampler.Pass: MLXArray] = [
            .conditional: Self.array(Self.base),
            .unconditional: Self.array(Self.base.map { $0 + 1 }),
            .perturbed: Self.array(Self.base.map { $0 + 10 }),
            .modality: Self.array(Self.base.map { $0 + 100 }),
        ]
        let backward = try Self.combine(params, mirrored)
        print("deltas +: \(forward.prefix(2)), deltas -: \(backward.prefix(2))")
        #expect(forward == Self.base.map { $0 + 352 })
        #expect(backward == Self.base.map { $0 - 352 },
                Comment(rawValue: "reversing every delta must reverse the correction "
                        + "exactly; got \(backward)"))
    }

    // MARK: - The rescale

    /// `factor = std(cond) / std(pred)`, and it multiplies `pred`.
    ///
    /// Constructed so the answer is an exact power of two. With all three unconditioned
    /// passes zero, every delta is `cond` itself and
    ///
    /// ```
    /// pred = cond * (1 + (cfg - 1) + stg + (modality - 1)) = cond * 8
    /// ```
    ///
    /// so `std(cond) / std(pred)` is exactly `1/8`, and at `rescale_scale = 1` the whole
    /// combination collapses back to `cond` **bit for bit**. Inverted, the factor would be
    /// 8 and the result `64 * cond` — not a subtle difference, but one that only shows up
    /// once the ratio is not 1, which is never at neutral settings.
    @Test("the rescale factor is cond over pred and multiplies pred")
    func rescaleFactorIsCondOverPred() throws {
        let zeros = Self.array([Float](repeating: 0, count: 8))
        let passes: [Sampler.Pass: MLXArray] = [
            .conditional: Self.array(Self.base),
            .unconditional: zeros, .perturbed: zeros, .modality: zeros,
        ]
        // 1 + 2 + 2 + 3 = 8
        var params = GuidanceParams(cfgScale: 3, stgScale: 2, rescaleScale: 1,
                                    modalityScale: 4)

        let full = try Self.combine(params, passes)
        print("rescale 1.0: \(full.prefix(4)) against cond \(Self.base.prefix(4))")
        #expect(full == Self.base,
                Comment(rawValue: "a full rescale of an 8x-amplified prediction must "
                        + "return cond exactly; got \(full). 64x means the ratio is "
                        + "inverted, 8x means the factor was not applied."))

        // Half rescale: factor = 0.5 * (1/8) + 0.5 = 0.5625, and 8 * 0.5625 = 4.5.
        // Every value here is exact in fp32, including the halves.
        params.rescaleScale = 0.5
        let half = try Self.combine(params, passes)
        #expect(half == Self.base.map { $0 * 4.5 },
                Comment(rawValue: "rescale 0.5 blends the factor toward 1: "
                        + "0.5 * 0.125 + 0.5 = 0.5625, so 8x becomes 4.5x. Got \(half)"))

        // And zero rescale leaves the sum alone.
        params.rescaleScale = 0
        #expect(try Self.combine(params, passes) == Self.base.map { $0 * 8 })
    }

    /// The `ddof` choice **cancels**, exactly.
    ///
    /// `Sampler.combine` uses `ddof: 1`, the unbiased convention. But the correction
    /// appears in both the numerator and the denominator of `std(cond) / std(pred)` over
    /// the same element count, so the `sqrt(n / (n - 1))` divides out **algebraically** —
    /// not "too small to survive the bf16 round", which is what a comment here used to say.
    /// The two claims are very different: one says the difference is real and invisible,
    /// the other says there is no difference to see.
    ///
    /// Both halves are checked, because "it cancels" is only reassuring if the two
    /// conventions really do differ before they cancel.
    @Test("the std correction cancels between numerator and denominator")
    func ddofCancels() throws {
        // The two conventions differ by sqrt(8/7) = 1.069 at this size -- 7%, which is
        // nowhere near any floor.
        let x = Self.array(Self.base)
        let unbiased = Double(MLX.std(x, ddof: 1).item(Float.self))
        let population = Double(MLX.std(x, ddof: 0).item(Float.self))
        print("std ddof:1 \(unbiased), ddof:0 \(population), "
            + "ratio \(unbiased / population)")
        #expect(unbiased != population,
                Comment(rawValue: "the two conventions must differ before the "
                        + "cancellation is worth demonstrating"))

        // The cancellation, at the shape the rescale actually uses it: `std(cond)` over
        // `std(pred)`, two tensors of the same element count. The `sqrt(n / (n - 1))`
        // factor multiplies both, so the ratio — which is all the rescale consumes — is
        // the same under either convention.
        let pred = Self.array(Self.base.map { $0 * 3 - 1 })
        func ratio(ddof: Int) -> Double {
            Double(MLX.std(x, ddof: ddof).item(Float.self))
                / Double(MLX.std(pred, ddof: ddof).item(Float.self))
        }
        let unbiasedRatio = ratio(ddof: 1), populationRatio = ratio(ddof: 0)
        print("rescale ratio ddof:1 \(unbiasedRatio), ddof:0 \(populationRatio)")
        // Not asserted as an equality: the identity is exact in the reals, but each `std`
        // is rounded to fp32 on its own before the division, so the two ratios may differ
        // in the last place. A bound at 1e-6 is four orders below the 7% the conventions
        // differ by, which is the gap this is distinguishing.
        #expect(abs(unbiasedRatio / populationRatio - 1) < 1e-6,
                Comment(rawValue: "the correction must cancel in the ratio even at 8 "
                        + "elements, where the stds themselves are 7% apart. "
                        + "\(unbiasedRatio) vs \(populationRatio)"))
    }

    // MARK: - The invariants

    /// Identical prompts make guidance a no-op, as arithmetic rather than as a render.
    ///
    /// End to end the property is: with the same text as positive and negative prompt,
    /// cond and uncond are the same tensor, so `(scale - 1) * (cond - uncond)` is a **true
    /// zero** and the output cannot depend on the scale at all. This is the part of that
    /// which can be checked without a GPU.
    ///
    /// Checked in the stronger form the combination actually supports: with all three
    /// unconditioned passes equal to cond, every delta vanishes and the result is cond for
    /// **any** combination of the three scales — and with the rescale on as well, since
    /// `std(cond) / std(cond)` is 1 and `r * 1 + (1 - r)` is 1 for any `r`.
    @Test("identical passes make every guidance term an exact zero")
    func identicalPassesAreANoOp() throws {
        let cond = Self.array(Self.base)
        let passes: [Sampler.Pass: MLXArray] = [
            .conditional: cond, .unconditional: cond, .perturbed: cond, .modality: cond,
        ]
        for (cfg, stg, mod, rescale) in [(1.0, 0.0, 1.0, 0.0),
                                         (3.0, 1.0, 3.0, 0.7),
                                         (9.0, 5.0, 7.0, 1.0),
                                         (-2.0, -3.0, 0.5, 0.25)] {
            let params = GuidanceParams(cfgScale: cfg, stgScale: stg,
                                        rescaleScale: rescale, modalityScale: mod)
            let got = try Self.combine(params, passes)
            #expect(got == Self.base,
                    Comment(rawValue: "cfg \(cfg) / stg \(stg) / modality \(mod) / rescale "
                            + "\(rescale) over identical passes gave \(got), not cond. "
                            + "The deltas are exact zeros, so the scales cannot matter."))
        }
    }

    /// A live coefficient with no pass behind it is refused, not silently treated as zero.
    ///
    /// Substituting a zero for a pass that was never run turns the term into
    /// `(scale - 1) * (cond - 0.0)` — harmless only while the coefficient is neutral.
    /// `plan` takes the union of both streams' requirements precisely so that combination
    /// cannot arise; this is the assertion that it gets noticed if it ever does, rather
    /// than computing `cond * scale` and rendering something plausible.
    @Test("a live term with no pass is an error, not a silent zero")
    func liveTermWithoutAPassThrows() throws {
        let cond = Self.array(Self.base)
        let params = GuidanceParams(cfgScale: 3, stgScale: 1, rescaleScale: 0,
                                    modalityScale: 3)
        let sampler = Sampler(video: params, audio: params)
        for missing in [Sampler.Pass.unconditional, .perturbed, .modality] {
            var passes = Self.passes()
            passes[missing] = nil
            #expect(throws: Sampler.Failure.self,
                    Comment(rawValue: "dropping the \(missing) pass while its scale is "
                            + "live must throw")) {
                _ = try sampler.combine(passes, params: params, stream: "video")
            }
        }
        // The conditional pass is never optional.
        #expect(throws: Sampler.Failure.self) {
            _ = try sampler.combine([:], params: params, stream: "video")
        }
        // But a *neutral* coefficient does not need its pass -- that is the union rule's
        // whole point.
        let neutral = GuidanceParams(cfgScale: 1, stgScale: 0, modalityScale: 1)
        #expect(Self.values(try sampler.combine([.conditional: cond], params: neutral,
                                                stream: "video")) == Self.base)
    }
}

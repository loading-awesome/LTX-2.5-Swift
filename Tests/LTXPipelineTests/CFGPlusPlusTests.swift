// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

import MLX
import LTXFoundation
import LTXRecipes
@testable import LTXPipeline

/// CFG++ — stepping the unconditional direction while keeping the guided destination.
///
/// The guided x0 is an extrapolation away from the unconditional one, and at a high CFG
/// scale a plain Euler step inherits the whole of that extrapolation as its *direction*.
/// `euler_cfg_pp` keeps the guided prediction as the destination and takes the direction
/// from the unconditional one, which is the documented fix for the crushed, over-saturated
/// look at high guidance.
///
/// The sampler that ships with the Ingredients workflow is `euler_ancestral_cfg_pp`, and
/// this port ran plain deterministic Euler on every guided reference render before this.
@Suite("CFG++")
struct CFGPlusPlusTests {

    private static let sampler = Sampler(
        video: GuidanceParams(cfgScale: 3.0), audio: GuidanceParams(cfgScale: 3.0))

    private static func arrays(_ tokens: Int = 32)
        -> (sample: MLXArray, guided: MLXArray, uncond: MLXArray) {
        (MLXArray(Array(0 ..< tokens).map { Float($0) * 0.01 }).reshaped([1, tokens, 1]),
         MLXArray(Array(0 ..< tokens).map { Float($0) * 0.02 - 0.1 }).reshaped([1, tokens, 1]),
         MLXArray(Array(0 ..< tokens).map { Float($0) * 0.015 + 0.05 }).reshaped([1, tokens, 1]))
    }

    // MARK: - The standard path must not move

    /// The whole safety argument for adding this. Every existing render, fixture and gate
    /// went through `eulerStep`, so `nil` must reach byte-for-byte the same code it always
    /// did — which is why the steppers branch on nil rather than generalising the formula
    /// with a factor that happens to be 0.
    @Test("Passing no direction is bit-identical to the step that has always run")
    func standardPathIsUntouched() throws {
        let (sample, guided, _) = Self.arrays()
        let sigmas: [Float] = [1.0, 0.7, 0.4, 0.0]

        for index in 0 ..< sigmas.count - 1 {
            let plain = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                                   sigmas: sigmas, index: index)
            let withNil = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                                     sigmas: sigmas, index: index,
                                                     directionDenoised: nil)
            #expect(MLX.allClose(plain, withNil, rtol: 0, atol: 0).item(Bool.self),
                    "step \(index) moved with no direction supplied")
        }
    }

    @Test("The ancestral standard path is untouched too")
    func ancestralStandardPathIsUntouched() throws {
        let (sample, guided, _) = Self.arrays()
        let sigmas: [Float] = [1.0, 0.7, 0.4, 0.0]
        let noise = MLXArray.zeros(sample.shape, dtype: .float32)

        for index in 0 ..< sigmas.count - 1 {
            let plain = try Self.sampler.ancestralStep(sample: sample, denoised: guided,
                                                       sigmas: sigmas, index: index,
                                                       noise: noise)
            let withNil = try Self.sampler.ancestralStep(sample: sample, denoised: guided,
                                                         sigmas: sigmas, index: index,
                                                         noise: noise,
                                                         directionDenoised: nil)
            #expect(MLX.allClose(plain, withNil, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    // MARK: - What CFG++ computes

    /// The **rectified-flow** definition, written out: recover the unconditional noise
    /// estimate and step with this schedule's own coefficients.
    ///
    /// Deliberately not k-diffusion's `x0_guided + r(sample - x0_uncond)`. That form belongs
    /// to `x = x0 + sigma·eps`, and transcribing it here leaves x0 at coefficient 1 where
    /// this schedule wants `1 - sigma_next/sigma` — about 0.005 on the first step. It was
    /// implemented that way first and rendered uniform mush at guidance 3.0 and 1.5 alike.
    @Test("The CFG++ step is the rectified-flow step with the unconditional noise estimate")
    func cfgPPMatchesItsDefinition() throws {
        let (sample, guided, uncond) = Self.arrays()
        let sigmas: [Float] = [0.8, 0.6, 0.0]
        let sigma = sigmas[0], sigmaNext = sigmas[1]

        let got = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                             sigmas: sigmas, index: 0,
                                             directionDenoised: uncond)
        let eps = (sample - (1 - sigma) * uncond) / sigma
        let want = (1 - sigmaNext) * guided + sigmaNext * eps
        #expect(MLX.allClose(got, want, atol: 1e-6).item(Bool.self))
    }

    /// At sigma 1 there is no signal in the latent yet, so `(1 - sigma)` zeroes the
    /// unconditional term and CFG++ *must* collapse to the ordinary step. Worth pinning:
    /// it is the boundary that the VE transcription got wrong, where it instead applied the
    /// largest correction of the whole trajectory.
    @Test("At pure noise CFG++ is the ordinary step")
    func cfgPPIsInertAtSigmaOne() throws {
        let (sample, guided, uncond) = Self.arrays()
        let sigmas: [Float] = [1.0, 0.995, 0.0]
        let plain = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                               sigmas: sigmas, index: 0)
        let cfgpp = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                               sigmas: sigmas, index: 0,
                                               directionDenoised: uncond)
        #expect(MLX.allClose(plain, cfgpp, atol: 1e-5).item(Bool.self))
    }

    /// When the two predictions coincide there is no extrapolation to over-shoot, so CFG++
    /// must reduce to the ordinary step. This is what makes it a *correction* rather than a
    /// different sampler: it does nothing at CFG scale 1.
    @Test("With no guidance to over-shoot, CFG++ reduces to the standard step")
    func cfgPPIsTheIdentityWithoutGuidance() throws {
        let (sample, guided, _) = Self.arrays()
        let sigmas: [Float] = [1.0, 0.6, 0.3, 0.0]

        for index in 0 ..< sigmas.count - 1 {
            let plain = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                                   sigmas: sigmas, index: index)
            let cfgpp = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                                   sigmas: sigmas, index: index,
                                                   directionDenoised: guided)
            #expect(MLX.allClose(plain, cfgpp, atol: 1e-5).item(Bool.self),
                    "step \(index): identical predictions should give identical steps")
        }
    }

    /// The over-shoot it removes, made visible: with the guided prediction extrapolated away
    /// from the unconditional one, the standard step travels further from the destination
    /// than the CFG++ step does.
    @Test("CFG++ lands closer to the guided prediction than the plain step does")
    func cfgPPReducesOvershoot() throws {
        let sample = MLXArray([0.0 as Float, 0, 0]).reshaped([1, 3, 1])
        let uncond = MLXArray([1.0 as Float, 1, 1]).reshaped([1, 3, 1])
        // Guided sits well beyond uncond — the shape a high CFG scale produces.
        let guided = MLXArray([3.0 as Float, 3, 3]).reshaped([1, 3, 1])
        // Mid-trajectory, where `(1 - sigma)` is non-zero and the correction is live.
        let sigmas: [Float] = [0.6, 0.3, 0.0]

        let plain = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                               sigmas: sigmas, index: 0)
        let cfgpp = try Self.sampler.eulerStep(sample: sample, denoised: guided,
                                               sigmas: sigmas, index: 0,
                                               directionDenoised: uncond)
        let plainGap = MLX.abs(plain - guided).mean().item(Float.self)
        let cfgppGap = MLX.abs(cfgpp - guided).mean().item(Float.self)
        #expect(cfgppGap < plainGap,
                "cfg++ \(cfgppGap) should sit nearer the destination than plain \(plainGap)")
    }

    /// The terminal step is the prediction itself on the ancestral path, and the destination
    /// is the guided one either way — so CFG++ must not move it.
    @Test("The terminal ancestral step is the guided prediction, with or without CFG++")
    func terminalStepIgnoresCFGPP() throws {
        let (sample, guided, uncond) = Self.arrays()
        let sigmas: [Float] = [0.4, 0.0]
        let stepped = try Self.sampler.ancestralStep(sample: sample, denoised: guided,
                                                     sigmas: sigmas, index: 0, noise: nil,
                                                     directionDenoised: uncond)
        #expect(MLX.allClose(stepped, guided, rtol: 0, atol: 0).item(Bool.self))
    }

    // MARK: - The recipe seam

    @Test("The sampler kinds classify themselves")
    func samplerKindsClassify() {
        #expect(!SamplerKind.euler.usesCFGPP)
        #expect(!SamplerKind.euler.isAncestral)
        #expect(SamplerKind.eulerAncestral.isAncestral)
        #expect(!SamplerKind.eulerAncestral.usesCFGPP)
        #expect(SamplerKind.eulerCFGPP.usesCFGPP)
        #expect(!SamplerKind.eulerCFGPP.isAncestral)
        #expect(SamplerKind.eulerAncestralCFGPP.usesCFGPP)
        #expect(SamplerKind.eulerAncestralCFGPP.isAncestral)
    }


}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

import LTXModules
@testable import LTXPipeline

/// The conditioning blend on the **ancestral** path, where it is applied twice.
///
/// ## The asymmetry between the two loops
///
/// The two denoising loops do not treat the blend the same way:
///
/// * Euler: blend the x0 prediction, then step. **Once.**
/// * ancestral: blend the x0 prediction, step, and then — gated on whether the step draws
///   noise, which is `eta > 0` — blend the **stepped** latent again. **Twice.**
///
/// Two details of the second application are load-bearing:
///
/// * it happens only when the step draws noise. At `eta == 0` the ancestral step degenerates
///   to a plain Euler step and takes no second blend.
/// * the terminal step (`sigmas[i + 1] == 0`) returns **before** stepping, so the
///   once-blended prediction becomes the final latent and is *not* blended again. Applying
///   the blend twice there would be wrong at any fractional strength, and invisible at
///   strength 1.
///
/// ## Why it only becomes visible under conditioning
///
/// With an all-ones mask the blend is the identity, so applying it once or twice is the same
/// tensor to the bit. Conditioning makes it live: `ancestralStep` renoises *every* token,
/// frozen ones included, so without the second application a frozen token carries a fresh
/// draw of noise into the next forward while its timestep still tells the model that token
/// is clean at 0. That is the "conditioning looks applied but isn't" failure, and it is why
/// this suite exists at all: `--two-stage --image` is the first path here that runs an
/// ancestral loop over a non-uniform mask.
@Suite("Ancestral conditioning")
struct AncestralConditioningTests {

    /// One conditional pass and no guidance, so `combine` returns the prediction unchanged
    /// and the arithmetic under test is not hiding behind a guider.
    static let sampler = Sampler(video: DistilledRenderer.neutralGuidance,
                                 audio: DistilledRenderer.neutralGuidance)

    static let videoTokens = 4
    static let audioTokens = 2

    /// Every marker below is exact in bf16, so an equality assertion is an equality
    /// assertion and not a tolerance with the tolerance left off.
    static func filled(_ value: Float, tokens: Int, dtype: DType = .bfloat16) -> MLXArray {
        MLXArray([Float](repeating: value, count: tokens * 2), [1, tokens, 2]).asType(dtype)
    }

    static func streams(_ value: Float, dtype: DType = .bfloat16) -> Sampler.Streams {
        Sampler.Streams(video: filled(value, tokens: videoTokens, dtype: dtype),
                        audio: filled(value, tokens: audioTokens, dtype: dtype))
    }

    /// A mask that freezes token 0 of each stream at `1 - strength` and leaves the rest free,
    /// which is exactly what conditioning by latent index writes.
    static func conditioning(strength: Float, clean: Float)
        -> (mask: Sampler.Streams, clean: Sampler.Streams) {
        func mask(_ tokens: Int) -> MLXArray {
            let m = MLXArray.ones([1, tokens, 1], dtype: .float32)
            m[0..., 0 ..< 1, 0...] = MLXArray(Float(1) - strength)
            return m
        }
        func cleanLatent(_ tokens: Int) -> MLXArray {
            let c = MLXArray.zeros([1, tokens, 2], dtype: .float32)
            c[0..., 0 ..< 1, 0...] = MLXArray(clean)
            return c
        }
        return (mask: Sampler.Streams(video: mask(videoTokens), audio: mask(audioTokens)),
                clean: Sampler.Streams(video: cleanLatent(videoTokens),
                                       audio: cleanLatent(audioTokens)))
    }

    /// One velocity of 1.0 per stream, for the single conditional pass.
    static func velocities() -> (video: [Sampler.Pass: MLXArray],
                                 audio: [Sampler.Pass: MLXArray]) {
        ([.conditional: filled(1.0, tokens: videoTokens)],
         [.conditional: filled(1.0, tokens: audioTokens)])
    }

    static func token(_ x: MLXArray, _ index: Int) -> Float {
        x[0, index, 0].asType(.float32).item(Float.self)
    }

    // MARK: - The live case

    /// The whole point, as one number.
    ///
    /// Sample 3.0, velocity 1.0, sigma 1.0 → the free tokens predict 2.0 and the frozen one
    /// predicts 3.0 (its timestep is 0). The blend pins the frozen token to the clean value
    /// 7.0. `ancestralStep` then steps *and renoises* it to
    /// `(0.5/0.75) * (0.25*3 + 0.75*7) + 5.0 * sqrt(0.25 - 0.0625 * (0.5/0.75)^2)` ≈ 6.36,
    /// which is not 7.0 and is not anything in particular — it is the clean value with a
    /// step and a noise draw on top.
    ///
    /// The second blend brings it back to exactly 7.0. Without it, the model's next forward
    /// sees noise on a token whose timestep says clean.
    @Test("A frozen token comes out exactly equal to the clean value after an ancestral step")
    func ancestralStepRestoresTheFrozenToken() throws {
        let (mask, clean) = Self.conditioning(strength: 1.0, clean: 7.0)
        let result = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.5], latents: Self.streams(3.0),
            velocities: Self.velocities(), denoiseMask: mask, cleanLatent: clean,
            stepper: .ancestral(eta: 1.0, sNoise: 1.0, seed: 0),
            stepNoise: Self.streams(5.0))

        #expect(Self.token(result.latents.video, 0) == 7.0)
        #expect(Self.token(result.latents.audio, 0) == 7.0)

        // The free tokens moved, which is what says the step ran at all rather than the
        // whole tensor having been pinned.
        #expect(Self.token(result.latents.video, 1) != 2.0)
        #expect(Self.token(result.latents.video, 1) != 3.0)
    }

    /// The same inputs on the deterministic path, which blends **once**.
    ///
    /// `eulerStep(3.0, denoised: 7.0)` at `dt = -0.5` is `3 + (-4) * (-0.5) = 5.0`. A frozen
    /// token must land there and *not* on 7.0 — if it landed on 7.0 the second blend had
    /// leaked onto the Euler path, which would change every deterministic render.
    @Test("The Euler path blends once: a frozen token lands on the step, not on the clean value")
    func eulerStepDoesNotRestoreTheFrozenToken() throws {
        let (mask, clean) = Self.conditioning(strength: 1.0, clean: 7.0)
        let result = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.5], latents: Self.streams(3.0),
            velocities: Self.velocities(), denoiseMask: mask, cleanLatent: clean,
            stepper: .euler)

        #expect(Self.token(result.latents.video, 0) == 5.0)
        #expect(Self.token(result.latents.audio, 0) == 5.0)
    }

    /// The Euler path, whole, against the definition it had before the ancestral fix:
    /// `eulerStep(sample, postProcess(combined))`, one blend and nothing after it.
    ///
    /// Bit-equality over every token of both streams, so this is a regression lock rather
    /// than a spot check.
    @Test("The Euler path is bit-identical to blend-once-then-step")
    func eulerPathIsBitIdenticalToTheOldDefinition() throws {
        let (mask, clean) = Self.conditioning(strength: 0.6, clean: 8.0)
        let latents = Self.streams(3.0)
        let result = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.5], latents: latents,
            velocities: Self.velocities(), denoiseMask: mask, cleanLatent: clean,
            stepper: .euler)

        for (stream, tokens, got, latent, m, c) in [
            ("video", Self.videoTokens, result.latents.video, latents.video,
             mask.video, clean.video),
            ("audio", Self.audioTokens, result.latents.audio, latents.audio,
             mask.audio, clean.audio),
        ] {
            let timesteps = Sampler.timesteps(sigma: 1.0, mask: m, tokens: tokens)
            let combined = Sampler.denoised(latent: latent,
                                            velocity: Self.filled(1.0, tokens: tokens),
                                            timesteps: timesteps)
            let processed = Sampler.postProcess(denoised: combined, denoiseMask: m,
                                                cleanLatent: c)
            let want = try Self.sampler.eulerStep(sample: latent, denoised: processed,
                                                  sigmas: [1.0, 0.5], index: 0)
            let difference = MLX.max(MLX.abs(got.asType(.float32) - want.asType(.float32)))
            #expect(difference.item(Float.self) == 0.0, "\(stream)")
        }
    }

    // MARK: - The two gates on the second application

    /// The terminal step takes the once-blended prediction and is **not** blended again.
    ///
    /// Driven at strength 0.5, where the two differ: the prediction is
    /// `3 - 1 * (0.5 * 1.0) = 2.5`, one blend gives `0.5 * 2.5 + 0.5 * 8 = 5.25`, and a
    /// second would give `0.5 * 5.25 + 0.5 * 8 = 6.625`. At strength 1.0 both are 8.0 and
    /// the mistake is invisible, which is why this is the fractional case.
    @Test("The terminal ancestral step blends once, not twice")
    func terminalAncestralStepBlendsOnce() throws {
        let (mask, clean) = Self.conditioning(strength: 0.5, clean: 8.0)
        let result = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.0], latents: Self.streams(3.0),
            velocities: Self.velocities(), denoiseMask: mask, cleanLatent: clean,
            stepper: .ancestral(eta: 1.0, sNoise: 1.0, seed: 0),
            stepNoise: Self.streams(5.0))

        #expect(Self.token(result.latents.video, 0) == 5.25)
        #expect(Self.token(result.latents.audio, 0) == 5.25)
    }

    /// The second blend is gated on the step drawing noise, which is `eta > 0`. At
    /// `eta == 0` the ancestral step is a plain Euler step and blends only once — so the
    /// frozen token must land on 5.0, exactly as it does under `.euler`.
    @Test("At eta 0 the ancestral path takes no second blend")
    func etaZeroTakesNoSecondBlend() throws {
        let (mask, clean) = Self.conditioning(strength: 1.0, clean: 7.0)
        let result = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.5], latents: Self.streams(3.0),
            velocities: Self.velocities(), denoiseMask: mask, cleanLatent: clean,
            stepper: .ancestral(eta: 0.0, sNoise: 1.0, seed: 0))

        #expect(Self.token(result.latents.video, 0) == 5.0)
        #expect(Self.token(result.latents.audio, 0) == 5.0)
    }

    // MARK: - Invisibility without conditioning

    /// The unconditioned ancestral path is unchanged, which is why the missing term went
    /// unnoticed for so long and why adding it alters no unconditioned render.
    ///
    /// Both arms: no mask at all (`postProcess` short-circuits), and an all-ones mask over a
    /// zero clean latent (`x * 1 + 0 * 0`, exact in fp32 and exact on the way back to bf16).
    @Test("With nothing frozen the ancestral step is bit-identical either way")
    func theSecondBlendIsInvisibleWhenNothingIsFrozen() throws {
        let latents = Self.streams(3.0)
        let noise = Self.streams(5.0)

        let unconditioned = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.5], latents: latents,
            velocities: Self.velocities(),
            stepper: .ancestral(eta: 1.0, sNoise: 1.0, seed: 0), stepNoise: noise)

        let allOnes = Sampler.Streams(
            video: MLXArray.ones([1, Self.videoTokens, 1], dtype: .float32),
            audio: MLXArray.ones([1, Self.audioTokens, 1], dtype: .float32))
        let zeros = Sampler.Streams(
            video: MLXArray.zeros([1, Self.videoTokens, 2], dtype: .float32),
            audio: MLXArray.zeros([1, Self.audioTokens, 2], dtype: .float32))
        let neutral = try Self.sampler.step(
            index: 0, sigmas: [1.0, 0.5], latents: latents,
            velocities: Self.velocities(), denoiseMask: allOnes, cleanLatent: zeros,
            stepper: .ancestral(eta: 1.0, sNoise: 1.0, seed: 0), stepNoise: noise)

        for (stream, a, b) in [("video", unconditioned.latents.video, neutral.latents.video),
                               ("audio", unconditioned.latents.audio, neutral.latents.audio)] {
            let difference = MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32)))
            #expect(difference.item(Float.self) == 0.0, "\(stream)")
        }
    }
}

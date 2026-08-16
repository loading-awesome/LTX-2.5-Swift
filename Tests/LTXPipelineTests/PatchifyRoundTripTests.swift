// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import Testing

@testable import LTXPipeline

/// The seam between VAE space and token space, in both directions.
///
/// This suite is exact — every assertion is bit equality, none has a tolerance — and that
/// is a property of the thing under test rather than a choice. `patchify` and `unpatchify`
/// are a transpose and a reshape; they move values without arithmetic, so anything less
/// than exact agreement means the *order* is wrong, and an order error is not a small
/// error. `unpatchifyVideo`'s own doc records the consequence on the way out: a correctly
/// shaped, spatially transposed latent that decodes to something which still looks like a
/// video. Going in, the same mistake writes conditioned content to scrambled token
/// positions at full strength, with no shape error to catch it.
///
/// No checkpoint is needed: the round trip is complete evidence on its own, which makes
/// this one of the few checks in the repository that costs nothing and hides nothing.
@Suite("Patchify round trip")
struct PatchifyRoundTripTests {

    /// The production video latent: 13 latent frames on the `8k+1` lattice at 97 pixel
    /// frames, 12 x 20 spatially at 384 x 640 with `spatialScale` 32, 128 channels.
    private static let videoShape = [1, 128, 13, 12, 20]
    /// The production audio latent: 8 channels x 101 latents x 16 features, which packs to
    /// 128 per token.
    private static let audioShape = [1, 8, 101, 16]

    private static func ramp(_ shape: [Int]) -> MLXArray {
        // A ramp, not random noise: every element is distinct, so a permutation that
        // merely reorders values is caught. With random values a duplicate pair could
        // mask a transposition, and with a constant tensor every wrong order passes.
        let n = shape.reduce(1, *)
        return MLXArray(Array(0 ..< n).map { Float($0) }).reshaped(shape)
    }

    @Test("video survives latent -> tokens -> latent, bit for bit")
    func videoRoundTripFromLatent() throws {
        let latent = Self.ramp(Self.videoShape)
        let tokens = try Sampler.patchifyVideo(latent)
        #expect(tokens.shape == [1, 13 * 12 * 20, 128])

        let back = try Sampler.unpatchifyVideo(tokens, latentFrames: 13, height: 12, width: 20)
        #expect(back.shape == Self.videoShape)
        #expect(MLX.all(back .== latent).item(Bool.self))
    }

    /// The other direction, and not redundant with the one above. A pair of functions that
    /// are inverses *of each other* can still both be wrong in a way that cancels — two
    /// matching transpositions compose to the identity. Starting from token space exercises
    /// the composition in the order a render actually performs it.
    @Test("video survives tokens -> latent -> tokens, bit for bit")
    func videoRoundTripFromTokens() throws {
        let tokens = Self.ramp([1, 13 * 12 * 20, 128])
        let latent = try Sampler.unpatchifyVideo(tokens, latentFrames: 13, height: 12, width: 20)
        let back = try Sampler.patchifyVideo(latent)
        #expect(back.shape == tokens.shape)
        #expect(MLX.all(back .== tokens).item(Bool.self))
    }

    @Test("audio survives latent -> tokens -> latent, bit for bit")
    func audioRoundTripFromLatent() throws {
        let latent = Self.ramp(Self.audioShape)
        let tokens = try Sampler.patchifyAudio(latent)
        #expect(tokens.shape == [1, 101, 128])

        let back = Sampler.unpatchifyAudio(tokens, channels: 8)
        #expect(back.shape == Self.audioShape)
        #expect(MLX.all(back .== latent).item(Bool.self))
    }

    @Test("audio survives tokens -> latent -> tokens, bit for bit")
    func audioRoundTripFromTokens() throws {
        let tokens = Self.ramp([1, 101, 128])
        let latent = Sampler.unpatchifyAudio(tokens, channels: 8)
        let back = try Sampler.patchifyAudio(latent)
        #expect(MLX.all(back .== tokens).item(Bool.self))
    }

    /// The negative control, and the reason the round trips above are evidence rather than
    /// coincidence. Both round trips would pass if `patchifyVideo` and `unpatchifyVideo`
    /// shared one *wrong* token order, so this pins the order itself against the
    /// convention `DiTForward.videoPositions` builds the rotary grid in: frame, then y,
    /// then x, with **x fastest**.
    ///
    /// Read off a `[1, 1, 2, 2, 3]` latent whose values are `0...11`: token `k` must be
    /// element `k` of the flattened spatial grid, so the first three tokens are the first
    /// row's x sweep. If x were slowest the sequence would start `0, 3, 6`.
    @Test("the token order is frame, then y, then x with x fastest")
    func tokenOrderIsXFastest() throws {
        // One channel keeps the assertion readable: each token is a single scalar.
        let latent = Self.ramp([1, 1, 2, 2, 3])
        let tokens = try Sampler.patchifyVideo(latent)
        #expect(tokens.shape == [1, 12, 1])
        let flat = tokens.reshaped([12]).asArray(Float.self)
        #expect(flat == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    }

    /// The audio equivalent: a token is `(channel, feature)` flattened with **feature
    /// fastest**, so a `[1, 2, 2, 3]` latent gives token 0 = channel 0's three features
    /// followed by channel 1's three.
    @Test("an audio token packs channel-major, feature fastest")
    func audioTokenPacking() throws {
        let latent = Self.ramp([1, 2, 2, 3])   // [B, C=2, T=2, F=3]
        let tokens = try Sampler.patchifyAudio(latent)
        #expect(tokens.shape == [1, 2, 6])
        let first = tokens[0, 0].asArray(Float.self)
        // channel 0 contributes t=0's features 0,1,2; channel 1 contributes 6,7,8.
        #expect(first == [0, 1, 2, 6, 7, 8])
    }

    /// Patchifying a stream that is already in token space is the mistake this rank check
    /// exists for, and it is an easy one to make once conditioning has several entry
    /// points. Without the guard a `[1, T, C]` tensor is rank 3 where rank 5 is expected
    /// and MLX would throw something about dimensions rather than about latents.
    @Test("a token-space stream is refused rather than patchified twice")
    func doublePatchifyIsRefused() {
        #expect(throws: Sampler.Failure.self) {
            _ = try Sampler.patchifyVideo(Self.ramp([1, 3120, 128]))
        }
        #expect(throws: Sampler.Failure.self) {
            _ = try Sampler.patchifyAudio(Self.ramp([1, 101, 128]))
        }
    }
}

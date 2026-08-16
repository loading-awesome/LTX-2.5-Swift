// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import LTXModules
import MLX
import Testing

@testable import LTXPipeline

/// The render-only initial latent.
///
/// No checkpoint is needed: this draws numbers and checks their shape, dtype and
/// reproducibility, all of which are properties of `RenderNoise` alone. That matters for
/// this particular suite, because a check that silently skipped on a machine without model
/// files would guard nothing at all.
@Suite("Render noise")
struct RenderNoiseTests {

    /// 640x384x97 at **24** fps, the production render geometry.
    ///
    /// The frame rate is not decoration here and 25 is not a harmless guess. Audio tokens
    /// are `round(frames / frameRate x audioLatentsPerSecond)`, so 24 fps gives
    /// `round(97/24 x 25) = 101` and 25 fps gives exactly 97. Both are plausible counts of
    /// the right rank, the video stream is identical either way, and the error would only
    /// surface much later as an audio shape mismatch. `audioLatentsPerSecond` defaulting
    /// to 25.0 is what makes 25 fps look natural and land wrong.
    private static func productionGeometry() -> DiTForward.Geometry {
        DiTForward.Geometry(frames: 97, height: 384, width: 640, frameRate: 24)
    }

    @Test("the drawn streams match the production capture's token-space shapes")
    func productionShapes() throws {
        let noise = try RenderNoise(seed: 0, geometry: Self.productionGeometry())
        let s = noise.streamsForRenderOnly()

        // 3120 is 13 latent frames x 240 tokens per frame; 101 is the audio latent count
        // at 97 frames / 24 fps.
        #expect(s.video.shape == [1, 3120, 128])
        #expect(s.audio.shape == [1, 101, 128])
    }

    /// The cast is load-bearing rather than cosmetic: an fp32 initial latent changes the
    /// arithmetic of every step that follows it, so a render drawn in fp32 is a different
    /// render from one drawn in bf16.
    @Test("both streams arrive as bf16")
    func dtypeIsBFloat16() throws {
        let s = try RenderNoise(seed: 7, geometry: Self.productionGeometry())
            .streamsForRenderOnly()
        #expect(s.video.dtype == .bfloat16)
        #expect(s.audio.dtype == .bfloat16)
    }

    /// The property that makes a render reproducible. Without it, a render someone flagged
    /// as wrong cannot be re-rendered to look at again.
    ///
    /// Bit equality, not a tolerance. Two draws from one seed are the same computation;
    /// anything less than exact here would mean the PRNG is picking up state from
    /// somewhere else, which is precisely the failure worth catching.
    @Test("the same seed draws the same latent, bit for bit")
    func seedIsDeterministic() throws {
        let g = Self.productionGeometry()
        let a = try RenderNoise(seed: 42, geometry: g).streamsForRenderOnly()
        let b = try RenderNoise(seed: 42, geometry: g).streamsForRenderOnly()

        #expect(MLX.all(a.video .== b.video).item(Bool.self))
        #expect(MLX.all(a.audio .== b.audio).item(Bool.self))
    }

    /// The negative control for the test above. A constant generator would satisfy
    /// `seedIsDeterministic` perfectly, so determinism is only evidence once it is paired
    /// with a demonstration that the seed is actually consulted.
    @Test("a different seed draws a different latent on both streams")
    func seedIsConsulted() throws {
        let g = Self.productionGeometry()
        let a = try RenderNoise(seed: 42, geometry: g).streamsForRenderOnly()
        let b = try RenderNoise(seed: 43, geometry: g).streamsForRenderOnly()

        #expect(!MLX.all(a.video .== b.video).item(Bool.self))
        #expect(!MLX.all(a.audio .== b.audio).item(Bool.self))
    }

    /// Video and audio are drawn from two splits of one key. If they were drawn from the
    /// *same* key they would be identical wherever their shapes overlap — both streams are
    /// 128 channels wide here, so the first 101 audio tokens would duplicate the first 101
    /// video tokens exactly. That is a real and easy mistake, it produces a plausible
    /// tensor of the right shape, and nothing downstream would complain about it.
    @Test("the two streams are not the same draw")
    func streamsAreIndependent() throws {
        let s = try RenderNoise(seed: 1, geometry: Self.productionGeometry())
            .streamsForRenderOnly()
        let overlap = s.audio.dim(1)
        let videoHead = s.video[0..., 0 ..< overlap, 0...]
        #expect(!MLX.all(videoHead .== s.audio).item(Bool.self))
    }

    /// A geometry that yields no video tokens is a configuration error, and the cheapest
    /// place to say so is here. Left unchecked it produces a `[1, 0, 128]` latent that
    /// survives the sampler and only fails inside `unpatchifyVideo`, where the message
    /// names a token count and reads as a sampler bug.
    @Test("a zero-width geometry throws rather than drawing an empty video latent")
    func degenerateVideoGeometryThrows() {
        let g = DiTForward.Geometry(frames: 97, height: 384, width: 0, frameRate: 24)
        #expect(throws: RenderNoise.Failure.self) {
            _ = try RenderNoise(seed: 0, geometry: g)
        }
    }

    /// The audio side has its own way of collapsing, and it is not the same input.
    ///
    /// Worth writing down because it is counterintuitive: `frames: 0` does **not** empty
    /// the video stream. `LatentGeometry.latentFrames` opens with `guard frames > 1 else
    /// { return 1 }`, so zero pixel frames still yields one latent frame and 240 video
    /// tokens. The audio count has no such clamp — `round(0 seconds x latentsPerSecond)`
    /// is 0 — so this geometry produces a full video latent beside an empty audio one.
    /// A check that only looked at the video stream would pass it.
    @Test("a zero-frame geometry throws on the audio stream, which is the one that empties")
    func degenerateAudioGeometryThrows() {
        let g = DiTForward.Geometry(frames: 0, height: 384, width: 640, frameRate: 24)
        #expect(g.videoTokens == 240)   // the clamp, stated rather than assumed
        #expect(g.audioTokens == 0)
        #expect(throws: RenderNoise.Failure.self) {
            _ = try RenderNoise(seed: 0, geometry: g)
        }
    }

    /// The provenance string is what survives next to an mp4 after the process that made
    /// it is gone. It has to carry the seed, so the render can be reproduced, and it has to
    /// say in words how the latent was drawn, so a future reader finding a bare mp4 and a
    /// bare latent does not treat these numbers as comparable to another run's.
    @Test("provenance names the seed and how the latent was drawn")
    func provenanceIsHonest() throws {
        let n = try RenderNoise(seed: 12345, geometry: Self.productionGeometry())
        #expect(n.seed == 12345)
        #expect(n.provenance.contains("12345"))
        #expect(n.provenance.contains("drawn"))
        #expect(n.provenance.lowercased().contains("rng"))
    }
}

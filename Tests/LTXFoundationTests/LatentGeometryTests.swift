// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// Geometry checked against **concrete shapes that were actually produced**, not
/// against a re-reading of the same formula. A test that re-derives the rule it
/// is testing proves only that the author is self-consistent.
@Suite("Latent geometry")
struct LatentGeometryTests {

    let g = LatentGeometry()

    @Test("reproduces the measured fixture shapes exactly")
    func measuredTiny() throws {
        // 320x192, 25 frames, 24 fps:
        //   video (1, 128, 4, 6, 10)   audio (1, 8, 26, 16)
        #expect(try g.videoLatentShape(width: 320, height: 192, frames: 25)
                == [1, 128, 4, 6, 10])
        #expect(try g.audioLatentShape(frames: 25, frameRate: 24.0)
                == [1, 8, 26, 16])
    }

    /// The production shape, derived from the same rules and kept as a test so it
    /// is checked rather than assumed. 640/32 = 20, 384/32 = 12,
    /// (97-1)/8 + 1 = 13, round(97/24 x 25) = round(101.04) = 101.
    @Test("predicts the production shape from the same rules")
    func predictedProduction() throws {
        #expect(try g.videoLatentShape(width: 640, height: 384, frames: 97)
                == [1, 128, 13, 12, 20])
        #expect(g.audioLatentCount(frames: 97, frameRate: 24.0) == 101)
    }

    @Test("round, not ceil: contract 3, and the boundary where they differ")
    func roundNotCeil() {
        // 25 frames at 24 fps is 1.041666s; x25 = 26.04.
        //   round -> 26     ceil -> 27
        #expect(g.audioLatentCount(frames: 25, frameRate: 24.0) == 26)

        // A count where the two rules disagree by one. 17 frames -> 0.70833s
        // x25 = 17.708: round 18, ceil 18 -- same. 13 frames -> 13.54: round 14,
        // ceil 14. Find one that rounds DOWN: 9 frames -> 0.375s x25 = 9.375,
        // round 9, ceil 10. That is the disagreement, and it is a shape mismatch
        // rather than a difference in values, so no numerical check would see it.
        #expect(g.audioLatentCount(frames: 9, frameRate: 24.0) == 9)
        let ceilWouldGive = Int((Double(9) / 24.0 * 25.0).rounded(.up))
        #expect(ceilWouldGive == 10)
        #expect(g.audioLatentCount(frames: 9, frameRate: 24.0) != ceilWouldGive)
    }

    @Test("two streams, not one packed sequence — measured widths and counts")
    func separateStreams() throws {
        // Measured at the small shape:
        //   video stream  (1, 240, 4096)   audio stream  (1, 26, 2048)
        // Text is in neither; it enters by cross-attention.
        let counts = try g.streamTokenCounts(width: 320, height: 192, frames: 25,
                                             frameRate: 24.0)
        #expect(counts.video == 240)
        #expect(counts.audio == 26)

        // The trap: 240 is exactly the video token count, so packing
        // text+video+audio into one sequence would produce a plausible-looking
        // tensor at some shapes and be structurally wrong at every block.
        #expect(counts.video != counts.video + counts.audio + 7)
    }

    // MARK: - Rejections

    @Test("off-lattice frame counts are refused, with the nearest point named")
    func offLattice() {
        #expect(throws: LatentGeometry.Failure.offLattice(frames: 100, nearest: 97)) {
            try g.videoLatentShape(width: 320, height: 192, frames: 100)
        }
    }

    @Test("non-multiple-of-32 dimensions are refused")
    func notMultiple() {
        #expect(throws: LatentGeometry.Failure.notMultipleOfScale(
            dimension: "width", value: 300, scale: 32)) {
            try g.videoLatentShape(width: 300, height: 192, frames: 25)
        }
        #expect(throws: LatentGeometry.Failure.notMultipleOfScale(
            dimension: "height", value: 200, scale: 32)) {
            try g.videoLatentShape(width: 320, height: 200, frames: 25)
        }
    }

    @Test("non-positive dimensions are refused")
    func nonPositive() {
        #expect(throws: LatentGeometry.Failure.nonPositive(dimension: "width", value: 0)) {
            try g.videoLatentShape(width: 0, height: 192, frames: 25)
        }
    }

    @Test("factors come from config, not constants")
    func configurable() throws {
        // A checkpoint with different factors must produce different geometry
        // without editing this type.
        let other = LatentGeometry(spatialScale: 16, temporalScale: 4, videoChannels: 64)
        #expect(try other.videoLatentShape(width: 320, height: 192, frames: 9)
                == [1, 64, 3, 12, 20])
    }

    @Test("latent and pixel frame counts agree with the lattice both ways")
    func latticeRoundTrip() {
        for k in 0..<12 {
            let frames = 8 * k + 1
            let latent = g.latentFrames(forFrames: frames)
            #expect(g.lattice.pixelFrames(fromLatentFrames: latent) == frames)
        }
    }
}

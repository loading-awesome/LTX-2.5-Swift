// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import Testing
@testable import LTXModules

/// Where several in-context references sit relative to each other and to the target.
///
/// This is the whole of what makes Multiple-Subject-Reference work, and none of it has a
/// shape to check it. Every arrangement below produces a `[1, 3, tokens, 2]` grid of
/// plausible floats: the slots stacked on top of each other, the slots at the wrong sign,
/// the offset applied after the frame-rate divide instead of before. A render made with any
/// of them completes and looks like a render made with a mediocre adapter.
///
/// The convention: the offset is added to the time column of a causally-fixed patch grid,
/// in pixel frames, and slot `i` of `n` sits at `-(n - i)` frames — so the first slot is
/// furthest back in time and the last sits one frame ahead of the target.
@Suite("Reference positions", .serialized)
struct ReferencePositionTests {

    static let fps = 24.0
    static let latent = LatentGeometry()

    static func geometry(references: [DiTForward.Geometry.Reference] = [])
        -> DiTForward.Geometry {
        var g = DiTForward.Geometry(frames: 25, height: 128, width: 192,
                                    frameRate: fps, latent: latent)
        g.references = references
        return g
    }

    /// A reference grid, 5 latent frames of 4x6 — an MSR still repeated to 33 frames.
    static func slot(_ index: Int, of count: Int) -> DiTForward.Geometry.Reference {
        DiTForward.Geometry.Reference(
            latentFrames: 5, latentHeight: 4, latentWidth: 6,
            downscaleFactor: 1, temporalScaleFactor: 1,
            frameOffset: -(count - index), slotIndex: index)
    }

    static func times(_ positions: MLXArray) -> [Float] {
        let time = positions[0, 0, 0..., 0...]
        MLX.eval(time)
        return time.asArray(Float.self)
    }

    /// The path every shipped IC-LoRA takes must be untouched by the MSR work.
    ///
    /// The offset is applied in pixel frames and the divide by the frame rate moved out of
    /// `videoPositions` to sit after it. That is a different expression, and "different
    /// expression, same value" is a claim about float32 rather than about algebra — so it
    /// is checked bit for bit rather than within a tolerance.
    @Test("a reference with no offset is bit-identical to its own patch grid")
    func zeroOffsetIsUnchanged() {
        let reference = DiTForward.Geometry.Reference(
            latentFrames: 1, latentHeight: 4, latentWidth: 6)
        let g = Self.geometry(references: [reference])
        let target = DiTForward.videoPositions(
            latentFrames: g.latentFrames, latentHeight: g.height / Self.latent.spatialScale,
            latentWidth: g.width / Self.latent.spatialScale,
            temporalScale: Self.latent.temporalScale,
            spatialScale: Self.latent.spatialScale, frameRate: Self.fps)

        let got = DiTForward.referencePositions(reference, target: target, geometry: g)
        let want = DiTForward.videoPositions(
            latentFrames: 1, latentHeight: 4, latentWidth: 6,
            temporalScale: Self.latent.temporalScale,
            spatialScale: Self.latent.spatialScale, frameRate: Self.fps)
        MLX.eval(got, want)
        #expect(got.shape == want.shape)
        let a = got.asArray(Float.self), b = want.asArray(Float.self)
        for i in 0 ..< a.count {
            #expect(a[i].bitPattern == b[i].bitPattern,
                    "index \(i): \(a[i]) vs \(b[i])" as Comment)
        }
    }

    /// The offset is in pixel frames and lands before the frame-rate divide.
    ///
    /// A port that added `frameOffset` to the *seconds* grid would pass a test written in
    /// seconds and be wrong by a factor of the frame rate. This pins the unit: an offset of
    /// -3 moves time by 3/24 of a second, not by 3 seconds and not by 3 latent frames.
    @Test("frameOffset shifts time by exactly offset/fps")
    func offsetIsInPixelFrames() {
        let plain = DiTForward.Geometry.Reference(latentFrames: 5, latentHeight: 4,
                                                  latentWidth: 6)
        var shifted = plain
        shifted.frameOffset = -3
        let g = Self.geometry(references: [plain])
        let target = MLXArray.zeros([1, 3, 1, 2])

        let base = Self.times(DiTForward.referencePositions(plain, target: target, geometry: g))
        let moved = Self.times(DiTForward.referencePositions(shifted, target: target,
                                                             geometry: g))
        #expect(base.count == moved.count)
        for i in 0 ..< base.count {
            #expect(abs((base[i] - moved[i]) - 3.0 / Float(Self.fps)) < 1e-7,
                    "index \(i): \(base[i]) -> \(moved[i])" as Comment)
        }
        // And it genuinely goes negative — the whole point. The causal first patch of a
        // reference at offset -3 starts at -3/24 s, ahead of the target's own frame 0.
        #expect(moved[0] < 0, "the first patch start is \(moved[0]), not negative" as Comment)
    }

    /// Four slots must occupy four distinct time ranges, ordered pic1 furthest back.
    ///
    /// The failure this catches is the one with no other symptom: every slot at the same
    /// coordinates, which is what the pre-MSR code did unconditionally and what a dropped
    /// `frameOffset` would do again. The render completes and the subjects blend.
    @Test("slots are ordered and separated, pic1 earliest")
    func slotsAreSeparated() {
        let count = 4
        let references = (0 ..< count).map { Self.slot($0, of: count) }
        let g = Self.geometry(references: references)
        let target = MLXArray.zeros([1, 3, 1, 2])

        var starts: [Float] = []
        for reference in references {
            let t = Self.times(DiTForward.referencePositions(reference, target: target,
                                                             geometry: g))
            starts.append(t[0])
        }
        #expect(starts.allSatisfy { $0 < 0 }, "\(starts) are not all negative" as Comment)
        for i in 1 ..< starts.count {
            #expect(starts[i] > starts[i - 1],
                    "slot \(i) starts at \(starts[i]), not after \(starts[i - 1])" as Comment)
        }
        // pic1 sits at -count frames, the last slot at -1 frame.
        #expect(abs(starts[0] + Float(count) / Float(Self.fps)) < 1e-7)
        #expect(abs(starts[count - 1] + 1.0 / Float(Self.fps)) < 1e-7)
    }

    /// The rotary grid must grow by every reference, in order.
    ///
    /// `videoTokensWithReference` is what `prepare` checks the latent against, so a grid
    /// that disagrees with it is caught — but a grid of the right *length* built in the
    /// wrong order is not, and that is what the per-block comparison here pins.
    @Test("the video position grid appends every reference in order")
    func gridAppendsEveryReference() {
        let count = 3
        let references = (0 ..< count).map { Self.slot($0, of: count) }
        let g = Self.geometry(references: references)

        let target = DiTForward.videoPositions(
            latentFrames: g.latentFrames, latentHeight: g.height / Self.latent.spatialScale,
            latentWidth: g.width / Self.latent.spatialScale,
            temporalScale: Self.latent.temporalScale,
            spatialScale: Self.latent.spatialScale, frameRate: Self.fps)

        var blocks = [target]
        for reference in references {
            blocks.append(DiTForward.referencePositions(reference, target: target, geometry: g))
        }
        let want = MLX.concatenated(blocks, axis: 2)
        #expect(want.dim(2) == g.videoTokensWithReference)
        #expect(g.referenceTokens == count * 5 * 4 * 6)

        // Each reference's block sits where the token stream says it does. Comparing the
        // first token's time of each block is enough: the offsets are distinct by
        // construction, so a swapped pair moves them.
        var offset = g.videoTokens
        for (index, reference) in references.enumerated() {
            let first = want[0, 0, offset, 0].item(Float.self)
            #expect(abs(first - Float(reference.frameOffset) / Float(Self.fps)) < 1e-7,
                    "block \(index) starts at \(first)" as Comment)
            offset += reference.tokens
        }
    }

    /// The single-reference accessor must not quietly report one of several.
    @Test("`reference` reads nil once a second reference is added")
    func singularAccessorIsHonest() {
        var g = Self.geometry()
        #expect(g.reference == nil)
        g.reference = Self.slot(0, of: 1)
        #expect(g.reference != nil)
        #expect(g.references.count == 1)
        g.references.append(Self.slot(1, of: 2))
        #expect(g.reference == nil, "the singular view still answers with two references")
        #expect(g.referenceTokens == 2 * 5 * 4 * 6)
    }
}

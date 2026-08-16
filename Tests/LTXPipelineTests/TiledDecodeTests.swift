// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import MLXRandom
import Testing

import LTXFoundation

@testable import LTXModules
@testable import LTXPipeline

/// The general `TileSizeConfig` decode, checked without a checkpoint.
///
/// The one test that carries the whole safety argument is
/// ``generalPathReproducesTheRecordedTwoChunkComposition``: at 13 latent frames the
/// general path must be **bit-identical** to the hard-coded two-chunk composition it
/// replaced, the one `Renderer.decodeVideoOnRecordedPolicy` shipped. That old composition
/// is kept below in full, so the comparison is against the arithmetic itself and not
/// against a remembered number.
///
/// Everything else here is arithmetic and composition: tile coverage, mask partition of
/// unity, and a synthetic ramp that a mask or index error turns into a visible seam. None
/// of that needs the 1.4 GB video VAE, so it runs on any machine; the two tests that do
/// load the VAE are opt-in behind an environment variable.
@Suite("Tiled video decode", .serialized)
struct TiledDecodeTests {

    // MARK: - The composition that has evidence

    /// The exact pre-generalisation body of `Renderer.decodeVideoOnRecordedPolicy`,
    /// verbatim apart from taking the decode as a closure and dropping the 640x384 shape
    /// assertion. Do not "tidy" this: its value is that it is the expression the earlier
    /// measurements were taken through.
    static func recordedTwoChunk(_ latent: MLXArray,
                                 _ decode: (MLXArray) throws -> MLXArray) throws -> MLXArray {
        let first = try decode(latent[0..., 0..., 0..<10, 0..., 0...]).asType(.bfloat16)
        let second = try decode(latent[0..., 0..., 6..<13, 0..., 0...]).asType(.bfloat16)

        let firstMask = MLXArray((0..<25).map { $0 == 0 ? Float(1) : Float(25 - $0) / 25 })
            .reshaped([1, 1, 25, 1, 1])
        let secondMask = MLXArray((0..<25).map { Float($0) / 25 })
            .reshaped([1, 1, 25, 1, 1])
        let overlap = ((first[0..., 0..., 48..<73, 0..., 0...] * firstMask).asType(.bfloat16)
            + second[0..., 0..., 0..<25, 0..., 0...] * secondMask).asType(.bfloat16)
        return MLX.concatenated([
            first[0..., 0..., 0..<48, 0..., 0...], overlap,
            second[0..., 0..., 25..<49, 0..., 0...],
        ], axis: 2)
    }

    /// A stand-in decoder with the real one's shape contract and none of its weights.
    ///
    /// Deterministic in the tile's *content*, so the two compositions being compared feed
    /// on identical bytes; and irregular enough that a wrong mask or a wrong index cannot
    /// hide in a smooth field. It deliberately produces bf16-representable values only,
    /// so that "bit-identical" is a statement about the composition and not about the
    /// stand-in's own rounding.
    static func textureDecoder(scale: TiledDecode.Scale = .video)
        -> (MLXArray) throws -> MLXArray {
        { tile in
            let latentFrames = tile.shape[2]
            let frames = max(scale.time * (latentFrames - 1) + 1, 1)
            let height = tile.shape[3] * scale.height
            let width = tile.shape[4] * scale.width
            // Seed from the tile's own contents: the first latent element, which the
            // callers below make a function of the tile's absolute start.
            let seed = tile[0, 0, 0, 0, 0].item(Float.self)
            let count = 3 * frames * height * width
            var values = [Float](repeating: 0, count: count)
            var state = UInt64(bitPattern: Int64(seed * 1024)) &* 6_364_136_223_846_793_005 &+ 1
            for i in 0 ..< count {
                state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                // 8-bit payload over [-1, 1): exactly representable in bf16.
                values[i] = Float(Int(state >> 56) - 128) / 128
            }
            return MLXArray(values, [1, 3, frames, height, width])
        }
    }

    /// A latent whose first element identifies the tile's absolute start, so the stand-in
    /// decoder above returns *different* content for different tiles — as the real one
    /// does — while remaining a pure function of what it is handed.
    static func positionalLatent(frames: Int, height: Int, width: Int,
                                 channels: Int = 128) -> MLXArray {
        var values = [Float](repeating: 0, count: channels * frames * height * width)
        let plane = height * width
        for c in 0 ..< channels {
            for f in 0 ..< frames {
                for p in 0 ..< plane {
                    values[(c * frames + f) * plane + p] = Float(f) + Float(c) / 512
                }
            }
        }
        return MLXArray(values, [1, channels, frames, height, width]).asType(.bfloat16)
    }

    @Test("the general path reproduces the recorded two-chunk composition bit-exactly")
    func generalPathReproducesTheRecordedTwoChunkComposition() throws {
        // 2x3 latent spatially so the test is fast; the composition is on axis 2 and the
        // spatial extent is inert to it. The production 12x20 frame is planned in
        // `tileCoverage` and `rendererLayouts`, which is where its layout is checked.
        let latent = Self.positionalLatent(frames: 13, height: 2, width: 3)
        let decode = Self.textureDecoder()

        let layout = try TiledDecode.plan(latentFrames: 13, latentHeight: 2, latentWidth: 3)
        #expect(layout.frames.tiles.map(\.latent) == [0 ..< 10, 6 ..< 13])
        #expect(layout.frames.tiles.map(\.output) == [0 ..< 73, 48 ..< 97])
        #expect(layout.frames.overlap(after: 0) == 25)
        #expect(layout.tileCount == 2)

        let general = try TiledDecode.decode(latent, layout: layout, decodeTile: decode)
        let recorded = try Self.recordedTwoChunk(latent, decode)

        #expect(general.shape == recorded.shape)
        #expect(general.dtype == recorded.dtype)
        #expect(general.dtype == .bfloat16)

        // Bitwise, on the bf16 payload — not "close".
        let a = general.asType(.float32).flattened().asArray(Float.self)
        let b = recorded.asType(.float32).flattened().asArray(Float.self)
        var differing = 0
        for i in 0 ..< a.count where a[i].bitPattern != b[i].bitPattern { differing += 1 }
        print("13-frame bit-exactness: \(a.count) values, \(differing) differing")
        #expect(differing == 0)
    }

    // MARK: - Tile boundary arithmetic

    static let seconds: [(label: String, latentFrames: Int)] = [
        ("4s", 13), ("8s", 24), ("12s", 36), ("16s", 48), ("20s", 60),
    ]

    @Test("tiles cover the latent and the output with no gap and no double-count")
    func tileCoverage() throws {
        for (label, frames) in Self.seconds {
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 12, latentWidth: 20,
                                              policy: .conv(height: 384, width: 640))
            let tiles = layout.frames.tiles
            let latents = tiles.map { "[\($0.latent.lowerBound),\($0.latent.upperBound))" }
            let seams = (0 ..< tiles.count - 1).map { layout.frames.overlap(after: $0) }
            print("\(label): F'=\(frames) -> \(tiles.count) temporal x "
                + "\(layout.height.tiles.count)x\(layout.width.tiles.count) spatial, "
                + "output \(layout.frames.outputLength) frames, "
                + "latent tiles \(latents.joined(separator: " ")), seams \(seams)")

            // Latent coverage: starts at 0, ends at F, consecutive tiles touch or overlap.
            #expect(tiles.first?.latent.lowerBound == 0)
            #expect(tiles.last?.latent.upperBound == frames)
            for i in 1 ..< tiles.count {
                #expect(tiles[i].latent.lowerBound < tiles[i - 1].latent.upperBound)
                #expect(tiles[i].latent.lowerBound > tiles[i - 1].latent.lowerBound)
            }

            // Output coverage: the union is exactly [0, 8(F-1)+1), contiguous.
            #expect(tiles.first?.output.lowerBound == 0)
            #expect(tiles.last?.output.upperBound == 8 * (frames - 1) + 1)
            #expect(layout.frames.outputLength == 8 * (frames - 1) + 1)
            for i in 1 ..< tiles.count {
                #expect(tiles[i].output.lowerBound < tiles[i - 1].output.upperBound)
            }

            // Every seam is 25 output frames: 1 + (overlap_latent + 1 - 1) * 8 with the
            // causal +1, for the whole 80/24 family.
            for i in 0 ..< tiles.count - 1 { #expect(layout.frames.overlap(after: i) == 25) }

            // Seams never touch: no output frame is written by three tiles.
            for i in 1 ..< max(1, tiles.count - 1) {
                #expect(tiles[i + 1].output.lowerBound >= tiles[i - 1].output.upperBound)
            }

            // Each tile's own output range is its causal image.
            for tile in tiles {
                #expect(tile.output.lowerBound == 8 * tile.latent.lowerBound)
                #expect(tile.output.upperBound == 8 * (tile.latent.upperBound - 1) + 1)
                #expect(tile.output.count == 8 * (tile.latent.count - 1) + 1)
                #expect(tile.mask.count == tile.output.count)
            }

            // 640x384 resolves to one spatial tile at every length.
            #expect(layout.height.tiles.count == 1)
            #expect(layout.width.tiles.count == 1)
        }
    }

    @Test("the masks partition unity exactly, everywhere")
    func masksSumToOne() throws {
        for (label, frames) in Self.seconds {
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 12, latentWidth: 20)
            let sum = layout.frames.maskSum()
            #expect(sum.count == 8 * (frames - 1) + 1)
            let offBy = sum.filter { $0 != Float(1) }
            print("\(label): mask partition of unity over \(sum.count) frames, "
                + "\(offBy.count) not exactly 1")
            // Exactly 1.0 in fp32, not merely within a tolerance: (25-j)/25 + j/25 is
            // exact for every j, which is what lets the blend skip a denominator.
            #expect(offBy.isEmpty)

            // And the seam itself is a real ramp, not a step.
            for i in 0 ..< layout.frames.tiles.count - 1 {
                let left = layout.frames.tiles[i]
                let right = layout.frames.tiles[i + 1]
                let start = right.output.lowerBound - left.output.lowerBound
                let head = Array(left.mask[start...])
                let tail = Array(right.mask[0 ..< head.count])
                #expect(head.count == 25)
                #expect(head.first == 1)
                #expect(tail.first == 0)
                #expect(abs((head.last ?? 0) - Float(1) / 25) < 1e-7)
                #expect(abs((tail.last ?? 0) - Float(24) / 25) < 1e-7)
                for j in 0 ..< head.count { #expect(head[j] + tail[j] == Float(1)) }
            }
        }
    }

    @Test("spatial masks partition unity when the frame is wide enough to tile")
    func spatialMasksSumToOne() throws {
        // 1920 px long side: above 768, so the policy really does tile.
        let layout = try TiledDecode.plan(latentFrames: 13, latentHeight: 34, latentWidth: 60,
                                          policy: .conv(height: 1088, width: 1920))
        print("1920x1088: \(layout.height.tiles.count)x\(layout.width.tiles.count) spatial "
            + "tiles, width tiles "
            + "\(layout.width.tiles.map { "[\($0.output.lowerBound),\($0.output.upperBound))" })")
        #expect(layout.width.tiles.count > 1)
        #expect(layout.width.maskSum().allSatisfy { $0 == Float(1) })
        #expect(layout.height.maskSum().allSatisfy { $0 == Float(1) })
        #expect(layout.width.outputLength == 1920)
        #expect(layout.height.outputLength == 1088)
    }

    /// The partition of unity on the spatial axes, at every tile count this code can
    /// produce — and *exactly* 1.0 in fp32, not within a tolerance.
    ///
    /// A looser check would accept an approximate partition and normalise by an accumulated
    /// weight buffer whenever it failed. `plan` throws `Failure.notComplementary` instead
    /// of taking a division nothing has measured, so this test is what keeps that throw
    /// unreachable in practice: every
    /// spatial overlap pairs `(L-p)/(L+1)` against `(p+1)/(L+1)`, numerators summing to
    /// `L+1`, and the two fp32 quotients add back to exactly one.
    @Test("both spatial axes partition unity exactly, at 2 through 8 tiles")
    func spatialMasksPartitionUnityAtEveryTileCount() throws {
        // 768/64 px on both axes = 24/2 in latent units, stride 22. Latent widths chosen
        // to land on 2..8 tiles.
        for latentSide in [26, 60, 90, 112, 134, 156, 178] {
            let layout = try TiledDecode.plan(latentFrames: 2,
                                              latentHeight: latentSide, latentWidth: latentSide,
                                              policy: .reference)
            let h = layout.height, w = layout.width
            #expect(h.tiles.count > 1)
            #expect(h.tiles.count == w.tiles.count)
            let hOff = h.maskSum().filter { $0 != Float(1) }
            let wOff = w.maskSum().filter { $0 != Float(1) }
            print("latent side \(latentSide) (\(latentSide * 32) px): \(h.tiles.count) tiles, "
                + "height \(hOff.count) sums != 1, width \(wOff.count) sums != 1")
            #expect(hOff.isEmpty)
            #expect(wOff.isEmpty)
            #expect(h.outputLength == latentSide * 32)

            // Every spatial seam is a real ramp that starts above zero — the spatial
            // fade-in is (i+1)/(l+1), unlike the temporal (i)/l which reaches 0.
            for i in 0 ..< h.tiles.count - 1 {
                let left = h.tiles[i], right = h.tiles[i + 1]
                let start = right.output.lowerBound - left.output.lowerBound
                let head = Array(left.mask[start...])
                let tail = Array(right.mask[0 ..< head.count])
                #expect(head.count == 64)
                #expect(tail.first != 0)
                #expect(abs((tail.first ?? 0) - Float(1) / 65) < 1e-7)
                #expect(abs((head.first ?? 0) - Float(64) / 65) < 1e-7)
                for j in 0 ..< head.count { #expect(head[j] + tail[j] == Float(1)) }
            }
        }
    }

    // MARK: - The spatial blend must not disturb what was already measured

    /// The pre-spatial body of `TiledDecode.decode`, verbatim apart from taking the layout
    /// and the decode as parameters. Do not "tidy" this either: its only value is that it
    /// is the arithmetic that shipped before the spatial blend existed.
    static func temporalOnlyDecode(_ latent: MLXArray, layout: TiledDecode.Layout,
                                   _ decodeTile: (MLXArray) throws -> MLXArray) throws
        -> MLXArray {
        let tiles = layout.frames.tiles
        func masked(_ index: Int) throws -> MLXArray {
            let tile = tiles[index]
            let slice = latent[0..., 0..., tile.latent.lowerBound ..< tile.latent.upperBound,
                               0..., 0...]
            let decoded = try decodeTile(slice).asType(.bfloat16)
            let mask = MLXArray(tile.mask).reshaped([1, 1, tile.mask.count, 1, 1])
            return decoded * mask
        }
        func frames(_ x: MLXArray, _ range: Range<Int>) -> MLXArray {
            x[0..., 0..., range.lowerBound ..< range.upperBound, 0..., 0...]
        }
        var pieces: [MLXArray] = []
        var carry: MLXArray?
        for k in tiles.indices {
            let tile = tiles[k]
            let product = try masked(k)
            let headEnd = k > 0 ? layout.frames.overlap(after: k - 1) : 0
            let tailStart = k + 1 < tiles.count
                ? tiles[k + 1].output.lowerBound - tile.output.lowerBound
                : tile.output.count
            if let previous = carry, headEnd > 0 {
                pieces.append((previous + frames(product, 0 ..< headEnd)).asType(.bfloat16))
            }
            if tailStart > headEnd {
                pieces.append(frames(product, headEnd ..< tailStart).asType(.bfloat16))
            }
            carry = k + 1 < tiles.count
                ? frames(product, tailStart ..< tile.output.count).asType(.bfloat16)
                : nil
        }
        return pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 2)
    }

    /// Adding the spatial blend must be **inert** on every layout that has one spatial
    /// tile: untiled, and temporally tiled at each production length. Those are the layouts
    /// every earlier measurement was taken through, and the whole safety argument for
    /// touching `decode` at all is that they did not move by one bit.
    @Test("the spatial blend leaves the untiled and temporal-only paths bit-identical")
    func spatialBlendLeavesTheOldPathsAlone() throws {
        let decode = Self.textureDecoder()
        for frames in [1, 5, 10, 13, 24, 36] {
            let latent = Self.positionalLatent(frames: frames, height: 2, width: 3)
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 2, latentWidth: 3,
                                              policy: .conv(height: 384, width: 640))
            #expect(layout.height.tiles.count == 1)
            #expect(layout.width.tiles.count == 1)

            let now = try TiledDecode.decode(latent, layout: layout, decodeTile: decode)
            let before = try Self.temporalOnlyDecode(latent, layout: layout, decode)
            #expect(now.shape == before.shape)
            #expect(now.dtype == before.dtype)
            #expect(now.dtype == .bfloat16)

            let a = now.asType(.float32).flattened().asArray(Float.self)
            let b = before.asType(.float32).flattened().asArray(Float.self)
            var differing = 0
            for i in 0 ..< a.count where a[i].bitPattern != b[i].bitPattern { differing += 1 }
            print("F'=\(frames) (\(layout.frames.tiles.count) temporal tiles): \(a.count) "
                + "values, \(differing) differing from the pre-spatial body")
            #expect(differing == 0)
        }
    }

    /// The spatial blend, on a decoder whose tiles agree with the untiled answer.
    ///
    /// `rampDecoder` returns the absolute ramp restricted to the tile, so a correct blend
    /// reassembles the ramp exactly wherever the mask sums to one — which, per
    /// `spatialMasksPartitionUnityAtEveryTileCount`, is everywhere. Any index slip on
    /// height or width, any mask applied on the wrong axis, or any missing tile in the
    /// `product(h, w)` iteration dents this. It cannot detect a *rounding* choice: the
    /// bound below has to admit the bf16 ulps a four-tile corner accumulates, so moving
    /// where the blend rounds passes here.
    @Test("a 2x2 and 3x3 spatial blend reassemble a separable ramp")
    func spatialBlendReassemblesASeparableRamp() throws {
        let scale = TiledDecode.Scale(time: 8, height: 1, width: 1)
        // A "decoder" whose output depends only on the absolute output coordinate, so the
        // tiled and untiled answers must agree wherever the masks partition unity.
        func planarDecoder(latentH: Int, latentW: Int) -> (MLXArray) throws -> MLXArray {
            { tile in
                let f = max(scale.time * (tile.shape[2] - 1) + 1, 1)
                let h = tile.shape[3], w = tile.shape[4]
                // The latent encodes its own absolute (row, col) in two channels.
                let row0 = tile[0, 0, 0, 0, 0].item(Float.self)
                let col0 = tile[0, 1, 0, 0, 0].item(Float.self)
                var values = [Float](repeating: 0, count: 3 * f * h * w)
                for c in 0 ..< 3 {
                    for t in 0 ..< f {
                        for i in 0 ..< h {
                            for j in 0 ..< w {
                                let r = (row0 + Float(i)) / Float(2 * latentH)
                                let s = (col0 + Float(j)) / Float(2 * latentW)
                                values[((c * f + t) * h + i) * w + j] = (r + s) / 4
                            }
                        }
                    }
                }
                return MLXArray(values, [1, 3, f, h, w])
            }
        }
        func planarLatent(h: Int, w: Int) -> MLXArray {
            var values = [Float](repeating: 0, count: 2 * h * w)
            for i in 0 ..< h {
                for j in 0 ..< w {
                    values[i * w + j] = Float(i)               // channel 0: absolute row
                    values[h * w + i * w + j] = Float(j)       // channel 1: absolute column
                }
            }
            return MLXArray(values, [1, 2, 1, h, w]).asType(.bfloat16)
        }

        for (side, tiles) in [(20, 2), (30, 3)] {
            let policy = TiledDecode.Policy(frames: .init(tileSize: 80, overlap: 24),
                                            height: .init(tileSize: 12, overlap: 2),
                                            width: .init(tileSize: 12, overlap: 2))
            let layout = try TiledDecode.plan(latentFrames: 1, latentHeight: side,
                                              latentWidth: side, policy: policy, scale: scale)
            #expect(layout.height.tiles.count == tiles)
            #expect(layout.width.tiles.count == tiles)
            #expect(layout.tileCount == tiles * tiles)

            let latent = planarLatent(h: side, w: side)
            let decoder = planarDecoder(latentH: side, latentW: side)
            let out = try TiledDecode.decode(latent, layout: layout, decodeTile: decoder)
            #expect(out.shape == [1, 3, 1, side, side])
            let untiled = try decoder(latent).asType(.bfloat16)

            let got = out.asType(.float32).flattened().asArray(Float.self)
            let want = untiled.asType(.float32).flattened().asArray(Float.self)
            var worst = Float(0)
            for i in 0 ..< got.count { worst = max(worst, abs(got[i] - want[i])) }
            print("\(tiles)x\(tiles) spatial on a \(side)x\(side) plane: worst |diff| "
                + "\(worst) vs the untiled plane")
            // Values live in [0, 0.25], where one bf16 ulp is 2^-9 ~ 2.0e-03, and a
            // four-tile corner rounds three times. A seam from a wrong axis or a dropped
            // tile is O(0.1) — two orders up from this bound.
            #expect(worst <= Float(1) / 128)
        }
    }

    @Test("640x384 is one spatial tile; the policy is aspect-coupled to the long side")
    func spatialPolicyTrigger() throws {
        let production = TiledDecode.Policy.conv(height: 384, width: 640)
        #expect(production.width == TiledDecode.AxisPolicy(tileSize: 768, overlap: 64))
        #expect(production.height == TiledDecode.AxisPolicy(tileSize: 448, overlap: 64))
        #expect(production.frames == TiledDecode.AxisPolicy(tileSize: 80, overlap: 24))

        // 768 px long side is the last untiled width; 800 tiles.
        for (h, w, tiled) in [(432, 768, false), (432, 800, true), (1088, 1920, true)] {
            let policy = TiledDecode.Policy.conv(height: h, width: w)
            let layout = try TiledDecode.plan(latentFrames: 13,
                                              latentHeight: h / 32, latentWidth: w / 32,
                                              policy: policy)
            #expect(layout.width.isTiled == tiled,
                    Comment(rawValue: "\(w)px width, tile \(policy.width.tileSize)px"))
        }
    }

    // MARK: - Composition, end to end, against a known ramp

    /// A "decoder" that returns the untiled answer restricted to its own tile: output
    /// frame `j` of a tile starting at latent `b` carries the absolute value `8b + j`.
    /// Reassembly must therefore rebuild the whole ramp — any index slip shifts a block
    /// and any mask error dents the seam.
    static func rampDecoder(scale: TiledDecode.Scale)
        -> (MLXArray) throws -> MLXArray {
        { tile in
            let latentFrames = tile.shape[2]
            let frames = max(scale.time * (latentFrames - 1) + 1, 1)
            let height = tile.shape[3] * scale.height
            let width = tile.shape[4] * scale.width
            let base = tile[0, 0, 0, 0, 0].item(Float.self)   // the tile's latent start
            let plane = height * width
            var values = [Float](repeating: 0, count: 3 * frames * plane)
            for c in 0 ..< 3 {
                for f in 0 ..< frames {
                    let v = (Float(scale.time) * base + Float(f)) / 512
                    for p in 0 ..< plane { values[(c * frames + f) * plane + p] = v }
                }
            }
            return MLXArray(values, [1, 3, frames, height, width])
        }
    }

    @Test("a tiled ramp reassembles into the untiled ramp")
    func syntheticRampHasNoSeam() throws {
        let scale = TiledDecode.Scale(time: 8, height: 1, width: 1)
        for (label, frames) in Self.seconds {
            let latent = Self.positionalLatent(frames: frames, height: 1, width: 1, channels: 1)
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 1, latentWidth: 1,
                                              policy: .conv(height: 32, width: 32),
                                              scale: scale)
            let out = try TiledDecode.decode(latent, layout: layout,
                                             decodeTile: Self.rampDecoder(scale: scale))
            #expect(out.shape == [1, 3, 8 * (frames - 1) + 1, 1, 1])

            let got = out.asType(.float32).flattened().asArray(Float.self)
            let n = 8 * (frames - 1) + 1

            // Where exactly one tile writes, the answer must be *bit* identical to the
            // untiled ramp: that is where an index slip lives.
            var seam = Set<Int>()
            for i in 0 ..< layout.frames.tiles.count - 1 {
                let a = layout.frames.tiles[i + 1].output.lowerBound
                let b = layout.frames.tiles[i].output.upperBound
                for k in a ..< b { seam.insert(k) }
            }
            var worstSolo = Float(0), worstSeam = Float(0)
            for c in 0 ..< 3 {
                for f in 0 ..< n {
                    let expected = MLXArray(Float(f) / 512).asType(.bfloat16)
                        .asType(.float32).item(Float.self)
                    let d = abs(got[c * n + f] - expected)
                    if seam.contains(f) { worstSeam = max(worstSeam, d) }
                    else { worstSolo = max(worstSolo, d) }
                }
            }
            print("\(label): ramp reassembly, solo max |diff| \(worstSolo), "
                + "seam max |diff| \(worstSeam) over \(seam.count) blended frames")
            #expect(worstSolo == 0)
            // One bf16 ulp at the top of the ramp (472/512 -> 2^-8). A seam from a mask
            // or index error is orders of magnitude bigger than this.
            #expect(worstSeam <= Float(1) / 256)
        }
    }

    // MARK: - Degenerate lengths

    @Test("one latent frame decodes to one pixel frame, untiled")
    func singleLatentFrame() throws {
        let layout = try TiledDecode.plan(latentFrames: 1, latentHeight: 2, latentWidth: 3)
        #expect(layout.frames.tiles.count == 1)
        #expect(layout.frames.tiles[0].latent == 0 ..< 1)
        #expect(layout.frames.tiles[0].output == 0 ..< 1)
        #expect(layout.frames.outputLength == 1)
        #expect(layout.frames.tiles[0].mask == [1])

        let scale = TiledDecode.Scale(time: 8, height: 1, width: 1)
        let single = try TiledDecode.plan(latentFrames: 1, latentHeight: 1, latentWidth: 1,
                                          policy: .conv(height: 32, width: 32), scale: scale)
        let latent = Self.positionalLatent(frames: 1, height: 1, width: 1, channels: 1)
        let out = try TiledDecode.decode(latent, layout: single,
                                         decodeTile: Self.rampDecoder(scale: scale))
        #expect(out.shape == [1, 3, 1, 1, 1])
    }

    @Test("a latent shorter than one tile is a single tile with an all-ones mask")
    func shorterThanOneTile() throws {
        // The 80/24 policy is a 10/3 latent split, so anything up to 10 latent frames is
        // one tile — 73 pixel frames, three seconds.
        for frames in [2, 5, 9, 10] {
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 12, latentWidth: 20)
            #expect(layout.frames.tiles.count == 1)
            #expect(layout.frames.tiles[0].latent == 0 ..< frames)
            #expect(layout.frames.outputLength == 8 * (frames - 1) + 1)
            #expect(layout.frames.tiles[0].mask.allSatisfy { $0 == 1 })
        }
        // 11 is the first length that tiles.
        let eleven = try TiledDecode.plan(latentFrames: 11, latentHeight: 12, latentWidth: 20)
        #expect(eleven.frames.tiles.map(\.latent) == [0 ..< 10, 6 ..< 11])
        #expect(eleven.frames.overlap(after: 0) == 25)
    }

    @Test("a length that divides exactly into tiles leaves no short remainder")
    func exactDivision() throws {
        // Stride 7, size 10: 17 = 10 + 7 is two whole tiles, 24 = 10 + 7 + 7 is three.
        for (frames, expected) in [(17, [0 ..< 10, 6 ..< 17]),
                                   (24, [0 ..< 10, 6 ..< 17, 13 ..< 24]),
                                   (31, [0 ..< 10, 6 ..< 17, 13 ..< 24, 20 ..< 31])] {
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 12, latentWidth: 20)
            #expect(layout.frames.tiles.map(\.latent) == expected)
            // Every tile after the first is the same length — no runt at the end.
            let lengths = Set(layout.frames.tiles.dropFirst().map(\.latent.count))
            #expect(lengths == [11])
            #expect(layout.frames.outputLength == 8 * (frames - 1) + 1)
        }
    }

    // MARK: - What the spatial seam costs

    static let videoVAEPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-video-vae-conv-bf16.safetensors"

    /// Relative RMS, cosine and max absolute difference between two tensors: the fp32
    /// payloads accumulated in Double, `||c-r|| / ||r||`. Spelled out here so the numbers
    /// printed below are a definite expression and not a library default that could
    /// change under them.
    static func compare(_ candidate: MLXArray, _ reference: MLXArray)
        -> (relRMS: Double, cos: Double, maxAbs: Double) {
        let c = candidate.asType(.float32).flattened().asArray(Float.self)
        let r = reference.asType(.float32).flattened().asArray(Float.self)
        precondition(c.count == r.count, "\(candidate.shape) vs \(reference.shape)")
        var dot = 0.0, cc = 0.0, rr = 0.0, dd = 0.0, maxAbs = 0.0
        for i in 0 ..< c.count {
            let a = Double(c[i]), b = Double(r[i])
            dot += a * b; cc += a * a; rr += b * b
            let d = a - b; dd += d * d
            maxAbs = max(maxAbs, abs(d))
        }
        let relRMS = rr > 0 ? (dd / rr).squareRoot() : 0
        let denom = (cc * rr).squareRoot()
        return (relRMS, denom > 0 ? min(dot / denom, 1.0) : 0, maxAbs)
    }

    /// Does the ceiling actually lift, and what does it cost.
    ///
    /// 576x1024 is the size `Renderer` refused before the spatial blend existed: at a
    /// 1024 px long side `Policy.conv` gives a 768 px width tile and a 448 px height tile,
    /// so the frame resolves to 2x2 spatial tiles and `decode` used to throw. This decodes
    /// a still and a short clip at that size through the production policy.
    ///
    /// It also carries the seam measurement at the **production** tile size, which a
    /// 640x384 frame cannot: a 768 px tile does not split 640 px at all, so measuring a
    /// spatial seam there means forcing 384/256 px tiles — 12 and 8 latent units against
    /// production's 24 and 14 — and a tile that small has proportionally more boundary.
    /// Here the policy is the real long-side one and the frame still decodes untiled in a
    /// single pass, which is what makes tiled and untiled comparable at all.
    ///
    /// The latent is noise. A seam measured on noise measures the noise: there is no
    /// spatial structure for the blend to preserve or break, so the `rel_rms`, `cos` and
    /// `max_abs` printed at the end are an arithmetic check on the composition and a
    /// tripwire for a regression, not a statement about what the seam does to a picture.
    ///
    /// Opt-in (`LTX_SPATIAL_CEILING=1`): it loads the 1.4 GB conv VAE.
    @Test("576x1024 — the size that was refused — decodes through the production policy")
    func theCeilingAt576x1024() throws {
        guard ProcessInfo.processInfo.environment["LTX_SPATIAL_CEILING"] != nil else {
            print("SKIP ceiling: set LTX_SPATIAL_CEILING=1")
            return
        }
        let checkpoint = URL(fileURLWithPath: Self.videoVAEPath)
        guard FileManager.default.fileExists(atPath: checkpoint.path) else {
            print("SKIP ceiling: no video VAE at \(checkpoint.path)")
            return
        }
        let decoder = try VideoVAEDecoder(checkpoint: checkpoint)
        let policy = TiledDecode.Policy.conv(height: 576, width: 1024)
        print("576x1024 policy: height \(policy.height), width \(policy.width)")

        for latentFrames in [1, 5] {
            let layout = try TiledDecode.plan(latentFrames: latentFrames,
                                              latentHeight: 18, latentWidth: 32,
                                              policy: policy)
            #expect(layout.height.tiles.count == 2)
            #expect(layout.width.tiles.count == 2)
            #expect(layout.height.outputLength == 576)
            #expect(layout.width.outputLength == 1024)
            #expect(layout.width.tiles.map(\.latent.count) == [24, 10])
            #expect(layout.height.tiles.map(\.latent.count) == [14, 6])

            let latent = MLXRandom.normal([1, 128, latentFrames, 18, 32]).asType(.bfloat16)
            #expect(latent.shape == [1, 128, latentFrames, 18, 32])

            MLX.Memory.cacheLimit = 0
            MLX.Memory.peakMemory = 0
            var started = Date()
            let out = try TiledDecode.decode(latent, layout: layout) { try decoder.decode($0) }
            out.eval()
            let tiledSeconds = Date().timeIntervalSince(started)
            let tiledPeak = MLX.Memory.snapshot().peakMemory / (1024 * 1024)
            #expect(out.shape == [1, 3, 8 * (latentFrames - 1) + 1, 576, 1024])

            MLX.Memory.peakMemory = 0
            started = Date()
            let untiled = try decoder.decode(latent).asType(.bfloat16)
            untiled.eval()
            let untiledSeconds = Date().timeIntervalSince(started)
            let untiledPeak = MLX.Memory.snapshot().peakMemory / (1024 * 1024)

            let m = Self.compare(decoder.rgb(out), decoder.rgb(untiled))
            print("576x1024 F'=\(latentFrames) (\(out.shape[2]) frames): "
                + "tiled \(layout.tileCount) decodes in "
                + "\(String(format: "%.2f", tiledSeconds)) s, peak \(tiledPeak) MiB | "
                + "untiled 1 decode in \(String(format: "%.2f", untiledSeconds)) s, "
                + "peak \(untiledPeak) MiB")
            print("    production-policy spatial seam: rel_rms \(m.relRMS) cos \(m.cos) "
                + "max_abs \(m.maxAbs)")
        }
    }

    // MARK: - What tiling actually costs

    /// What the tiling buys, in bytes, on the real decoder.
    ///
    /// Opt-in (`LTX_TILED_DECODE_MEMORY=1`) because it loads the 1.4 GB video VAE and
    /// decodes a whole 20-second clip. It exists because "tiling saves memory" is a claim
    /// with a number attached, and a port that tiled into *more* memory than it saved
    /// would be worth knowing about. The latent is noise: this measures footprint, not
    /// pixels.
    @Test("peak memory, tiled versus one shot, at 4 and 20 seconds")
    func peakMemory() throws {
        guard ProcessInfo.processInfo.environment["LTX_TILED_DECODE_MEMORY"] != nil else {
            print("SKIP peak memory: set LTX_TILED_DECODE_MEMORY=1")
            return
        }
        let checkpoint = URL(fileURLWithPath: Self.videoVAEPath)
        guard FileManager.default.fileExists(atPath: checkpoint.path) else {
            print("SKIP peak memory: no video VAE at \(checkpoint.path)")
            return
        }
        let decoder = try VideoVAEDecoder(checkpoint: checkpoint)

        func measure(_ what: String, _ body: () throws -> MLXArray) rethrows {
            MLX.Memory.cacheLimit = 0
            MLX.Memory.peakMemory = 0
            let before = MLX.Memory.snapshot()
            let out = try body()
            out.eval()
            let after = MLX.Memory.snapshot()
            print("\(what): shape \(out.shape), peak "
                + "\(after.peakMemory / (1024 * 1024)) MiB, active delta "
                + "\((after.activeMemory - before.activeMemory) / (1024 * 1024)) MiB")
        }

        for frames in [13, 60] {
            let latent = MLXRandom.normal([1, 128, frames, 12, 20]).asType(.bfloat16)
            let layout = try TiledDecode.plan(latentFrames: frames,
                                              latentHeight: 12, latentWidth: 20,
                                              policy: .conv(height: 384, width: 640))
            try measure("F'=\(frames) tiled (\(layout.tileCount) tiles)") {
                try TiledDecode.decode(latent, layout: layout) { try decoder.decode($0) }
            }
            try measure("F'=\(frames) one shot") { try decoder.decode(latent) }
        }

        // Spatial tiling exists to reduce peak memory. At 640x384x33 both paths fit, so
        // the claim is checkable rather than assumed. The policy is forced: a 768 px
        // production tile does not split a 640 px frame at all.
        let short = MLXRandom.normal([1, 128, 5, 12, 20]).asType(.bfloat16)
        let spatial = TiledDecode.Policy(frames: .init(tileSize: 80, overlap: 24),
                                         height: .init(tileSize: 256, overlap: 64),
                                         width: .init(tileSize: 384, overlap: 64))
        let spatialLayout = try TiledDecode.plan(latentFrames: 5, latentHeight: 12,
                                                 latentWidth: 20, policy: spatial)
        try measure("640x384x33 spatially tiled (\(spatialLayout.tileCount) tiles)") {
            try TiledDecode.decode(short, layout: spatialLayout) { try decoder.decode($0) }
        }
        try measure("640x384x33 one shot") { try decoder.decode(short) }
    }

    @Test("the layout Renderer would use at each production length")
    func rendererLayouts() throws {
        for (label, frames) in Self.seconds {
            let pixels = 8 * (frames - 1) + 1
            let geometry = DiTForward.Geometry(frames: pixels, height: 384, width: 640,
                                               frameRate: 24)
            #expect(geometry.latentFrames == frames)
            let layout = try Renderer.videoTileLayout(geometry: geometry)
            #expect(layout.frames.outputLength == pixels)
            #expect(layout.tileCount == layout.frames.tiles.count)
            print("\(label): Renderer plans \(layout.tileCount) tiles for "
                + "640x384x\(pixels) (F'=\(frames))")
        }
    }
}

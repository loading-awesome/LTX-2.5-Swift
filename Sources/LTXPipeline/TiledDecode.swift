// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX

/// The `TileSizeConfig` video decode, generalised to any latent length.
///
/// ## What is contract here and what is policy
///
/// **Tile count is a memory strategy, not a contract** — 105 tiles at production and 1 at
/// tiny both decode the same latent, and neither is wrong. So the *number* of tiles this
/// file chooses is policy, and an implementation that chose differently would not thereby
/// be wrong.
///
/// What is **contract** is everything the tiles are made of:
///
/// * **Causal frame accounting.** The VAE's first latent frame carries 1 pixel frame and
///   every later one carries 8, so a latent tile `[b, e)` lands at output
///   `[8b, 8(e−1)+1)` and `F` latent frames decode to `8(F−1)+1` pixel frames. That is why
///   13 latent frames give 97 pixel frames and not 104.
/// * **The causal split.** Every tile after the first is pulled back by one latent frame
///   and its left ramp widened by one, so the blend ramp itself carries the causal
///   context. That is what turns a nominal `[7, 17)` second tile into `[6, 17)` with a
///   4-frame left ramp.
/// * **The trapezoidal masks and their exact values.** Complementary *by construction*,
///   not by normalisation: the masks partition unity for these splits, so the weight
///   denominator is skipped entirely.
/// * **The dtype asymmetry.** Masks are fp32, the accumulation buffer is the latent's
///   dtype — bf16 in production. Each tile's masked product is formed in fp32 and lands
///   in a bf16 buffer. That asymmetry is deliberately not "improved": computing the blend
///   in fp32 throughout would be *more* precise than the tensor this is measured against,
///   which is the same trap ``VideoVAEDecoder/decode(_:)`` documents for the convolutions.
///
/// ## The seam expression, and the one it is not
///
/// A decoder that streams one temporal group at a time gives each group its own bf16
/// buffer, so its cross-group seam is `previous[ovl] += buffer[ovl]` — **bf16 + bf16**,
/// because the incoming group's masked product was already rounded on its way into its own
/// buffer. This file instead adds the current tile's masked product to the previous tile's
/// bf16 tail while it is still fp32: `bf16( bf16(P_k · m_k) + P_{k+1} · m_{k+1} )`.
///
/// That is not an accident — it is the composition that was *measured*, and
/// `TiledDecodeTests` asserts bit-for-bit that this general path reproduces it at 13
/// latent frames.
///
/// **It is also, measurably, the worse of the two.** `TiledDecodeTests` runs both forms
/// over the same recorded decode: the fp32 seam lands at `rel_rms 1.3649e-03` and the
/// twice-rounded streamed form at `1.2762e-03` — 6.5% closer, over 25 of 97 frames. That
/// is a good reason to adopt the streamed form, and not a reason to change it here and
/// now: the recorded boundary and every number resting on `1.3649e-03` were measured
/// through the expression below. Switching means re-measuring, the same shape of change as
/// contract 16, not an edit.
///
/// ## Spatial tiling
///
/// The production policy is a 768 px long side and 80 frames, scaled by aspect. The long
/// spatial side always gets
/// a 768 px tile and the short side is scaled by aspect in *latent* units, so **spatial
/// tiling engages only when the long side exceeds 768 px**. At 640x384 the width tile is
/// 768 px and the height tile 448 px, both at or above the frame, so each spatial axis
/// resolves to a single tile — which is why the recorded boundary is purely temporal.
///
/// ``plan(latentFrames:latentHeight:latentWidth:policy:scale:)`` computes the spatial
/// layout, and ``decode(_:layout:decodeTile:)`` now blends it. `decode` used to **refuse**
/// any layout with more than one spatial tile, because the separable two-axis blend had no
/// recorded boundary to stand on. It still has none — no recording of a spatially tiled
/// decode exists — so what stands in its place is a **self-consistency** measurement:
/// `TiledDecodeTests.spatialSeamAgainstTheUntiledDecode` decodes one latent both ways on
/// the real conv VAE, at a shape small enough to decode in a single pass. That bounds the
/// seam against this implementation's own untiled decode; it cannot detect an error the
/// two paths share.
///
/// Measured post-`rgb`, with relative RMS and cosine accumulated in Double — the same
/// expression the temporal boundary uses:
///
/// | layout | rel_rms | cos | max_abs |
/// |---|---|---|---|
/// | temporal, 2 tiles, 640x384x97 | `1.5310e-03` | 0.99999883 | 0.0547 |
/// | spatial 1x2, 384 px tiles, 640x384x33 | `3.8400e-03` | 0.99999266 | 0.3145 |
/// | spatial 2x1, 256 px tiles, 640x384x33 | `3.1912e-03` | 0.99999493 | 0.1328 |
/// | spatial 2x2, 640x384x33 | `4.8255e-03` | 0.99998836 | 0.3223 |
/// | spatial 2x2, production policy, 576x1024x33 | `6.3692e-03` | 0.99997973 | 0.2246 |
///
/// **The spatial seam is 2.1x to 4.2x the temporal seam by rel_rms, and its worst pixel is
/// 2x to 6x the temporal worst pixel on a `[0, 1]` scale.** Read that as a real cost, not
/// a rounding difference.
///
/// Two things localise it, and both say the cost is the *decoder's*, not this file's:
///
/// * Split by region, the 1x2 case is `rel_rms 1.3121e-02` over the 64 px seam band and
///   `1.6150e-03` over the solo interiors. The seam band carries it.
/// * The solo interiors are nonetheless **not** clean, which means a tile's decode differs
///   from the one-pass decode even far from the blend: the conv decoder's spatial
///   receptive field is wider than these tiles, so no interior pixel is uncontaminated.
///   Widening the width tile from 12 to 16 latent units drops the seam band from
///   `1.3121e-02` to `7.0604e-03` — the error is a function of tile size, i.e. of the
///   policy, not of the arithmetic.
/// * `TiledDecodeTests.spatialBlendReassemblesASeparableRamp` runs the same blend over a
///   decoder that *is* translation-consistent and gets `max |diff| 0.0` at 2x2 and 3x3.
///   The blend contributes nothing measurable; all of the above is the VAE's own
///   behaviour at a tile boundary, which any tiled decode of it inherits.
///
/// ### The composition, and where it sits relative to the temporal one
///
/// Tiles are visited in `(t, h, w)` order, and every tile of one temporal group is written
/// into the *same* bf16 buffer. So a spatial seam is `bf16( bf16(A·m_A) + B·m_B )` — the
/// accumulated term already rounded, the incoming term still fp32, the sum rounded on
/// store. That is *the same expression* this file already uses for its temporal seam, so
/// the spatial axes need no new dtype decision: they get the one that was already
/// measured, in that same tile order so the four-tile corner rounds in the same sequence.
///
/// The single-spatial-tile case does not go near any of it. It returns the fp32 masked
/// product exactly as before, and `TiledDecodeTests.spatialBlendLeavesTheOldPathsAlone`
/// asserts bit-identity against a verbatim copy of the pre-spatial `decode` body at
/// F' = 1, 5, 10, 13, 24 and 36 — 0 of 5,179,392 bf16 payloads differing at the longest.
///
/// ### What spatial tiling buys
///
/// Peak memory, tiled versus one shot, measured the same day:
///
/// | shape | tiled | one shot |
/// |---|---|---|
/// | 640x384x33, 2x2 spatial | 2482 MiB | 2771 MiB |
/// | 576x1024x33, 2x2 spatial | 3569 MiB | 5674 MiB |
/// | 640x384x473, 9 temporal | 14006 MiB | 29251 MiB |
///
/// So it does reduce peak memory — 10% at 640x384, 37% at 576x1024 — but nothing like the
/// 2.1x the temporal tiling gives on a long clip, and it costs 17% more wall clock at
/// 576x1024x33 (3.57 s against 3.04 s) for four decodes instead of one. Spatial tiling is
/// worth engaging because it lifts a resolution ceiling, not because it is cheap.
public enum TiledDecode {

    // MARK: - Errors

    public enum Failure: Error, CustomStringConvertible, Equatable {
        case latentRank([Int])
        case notComplementary(axis: String, worst: Float)
        case tileLength(expected: Int, got: Int)
        case tileExtent(axis: String, expected: Int, got: Int)
        case policy(String)

        public var description: String {
            switch self {
            case let .latentRank(shape):
                return "a video latent must be [B, C, F, H, W], got \(shape)"
            case let .notComplementary(axis, worst):
                return "the \(axis) blend masks do not partition unity (worst sum "
                     + "\(worst)). The recorded boundary is the complementary path, "
                     + "which needs no weight denominator; a layout that needs one has "
                     + "not been measured."
            case let .tileLength(expected, got):
                return "a decoded tile is \(got) frames where the causal layout says "
                     + "\(expected); the decoder and the tiling disagree about the "
                     + "8(F-1)+1 lattice"
            case let .tileExtent(axis, expected, got):
                return "a decoded tile is \(got) wide on \(axis) where the layout says "
                     + "\(expected); the decoder and the tiling disagree about the "
                     + "spatial scale factor"
            case let .policy(why):
                return "illegal TileSizeConfig: \(why)"
            }
        }
    }

    // MARK: - The VAE grid

    /// `SpatioTemporalScaleFactors` — `(8, 32, 32)` for this checkpoint.
    public struct Scale: Equatable, Sendable {
        public var time: Int
        public var height: Int
        public var width: Int

        public init(time: Int, height: Int, width: Int) {
            self.time = time
            self.height = height
            self.width = width
        }

        public static let video = Scale(time: 8, height: 32, width: 32)
    }

    // MARK: - The policy

    /// `DimensionSizeConfig`: tile size and overlap **in pixel/frame units**, not latent.
    /// `tileSize == 0` means the axis is not tiled.
    public struct AxisPolicy: Equatable, Sendable {
        public var tileSize: Int
        public var overlap: Int

        public init(tileSize: Int, overlap: Int) {
            self.tileSize = tileSize
            self.overlap = overlap
        }

        public var isTiled: Bool { tileSize > 0 }
    }

    /// `TileSizeConfig`.
    public struct Policy: Equatable, Sendable {
        public var frames: AxisPolicy
        public var height: AxisPolicy
        public var width: AxisPolicy

        public init(frames: AxisPolicy, height: AxisPolicy, width: AxisPolicy) {
            self.frames = frames
            self.height = height
            self.width = width
        }

        /// `TileSizeConfig.default()` — 80/24 frames, 768/64 on both spatial axes.
        public static let reference = Policy(frames: AxisPolicy(tileSize: 80, overlap: 24),
                                             height: AxisPolicy(tileSize: 768, overlap: 64),
                                             width: AxisPolicy(tileSize: 768, overlap: 64))

        /// The conv VAE's production layout: the 768/64 long side and the 80/24 temporal
        /// chunk, with the short side derived from the long one.
        ///
        /// The short side is scaled in **latent** units — `round(size_lat · axis_lat /
        /// long_lat)` — and only then multiplied back up, because a pixel-space round
        /// plus a ceil-snap biases the short axis up by almost a whole latent. The round
        /// is to even, so `.toNearestOrEven` and not `.rounded()`.
        ///
        /// At 640x384 this gives width 768/64 and height 448/64, both untiled at this
        /// frame size.
        public static func conv(height: Int, width: Int,
                                scale: Scale = .video) -> Policy {
            let longSide = AxisPolicy(tileSize: 768, overlap: 64)
            let span = max(height, width)

            func axisSize(_ axisLength: Int, _ factor: Int) -> Int {
                let axisLatent = axisLength / factor
                let longLatent = span / factor
                let sizeLatent = longSide.tileSize / factor
                let overlapLatent = longSide.overlap / factor
                let lowerThreshold = max(2, overlapLatent + 1)
                let scaled = (Double(sizeLatent * axisLatent) / Double(longLatent))
                    .rounded(.toNearestOrEven)
                let tileLatent = max(lowerThreshold, Int(scaled))
                let minLegal = max(2 * factor, longSide.overlap + factor)
                return max(tileLatent * factor, minLegal)
            }

            return Policy(
                frames: AxisPolicy(tileSize: 80, overlap: 24),
                height: AxisPolicy(tileSize: axisSize(height, scale.height),
                                   overlap: longSide.overlap),
                width: AxisPolicy(tileSize: axisSize(width, scale.width),
                                  overlap: longSide.overlap))
        }
    }

    // MARK: - Splitting, in latent units

    /// A `DimensionInterval`: a half-open latent range plus its two ramp lengths.
    public struct Interval: Equatable, Sendable {
        public var start: Int
        public var end: Int
        public var leftRamp: Int
        public var rightRamp: Int
        public var length: Int { end - start }
    }

    /// Split an axis into overlapping tiles. The last tile may be short; there is no
    /// minimum-tile-size growth, because nothing on this path asks for one.
    static func splitBySize(length: Int, size: Int, overlap: Int) throws -> [Interval] {
        guard size > 0 else { throw Failure.policy("tile size must be > 0, got \(size)") }
        guard overlap >= 0, overlap < size else {
            throw Failure.policy("need 0 <= overlap < size, got \(overlap) and \(size)")
        }
        if length <= size {
            return [Interval(start: 0, end: length, leftRamp: 0, rightRamp: 0)]
        }
        let stride = size - overlap
        let amount = (length + size - 2 * overlap - 1) / stride
        var out = [Interval(start: 0, end: size, leftRamp: 0, rightRamp: overlap)]
        if amount > 2 {
            for i in 1 ..< (amount - 1) {
                out.append(Interval(start: i * stride, end: i * stride + size,
                                    leftRamp: overlap, rightRamp: overlap))
            }
        }
        out.append(Interval(start: (amount - 1) * stride, end: length,
                            leftRamp: overlap, rightRamp: 0))
        return out
    }

    /// The causal split: every tile after the first starts one latent frame earlier and
    /// ramps in one frame longer. The extra frame is the causal context the
    /// decoder needs, and it is why the second production tile is `[6, 13)` and not
    /// `[7, 13)`.
    static func splitTemporalCausal(length: Int, size: Int, overlap: Int) throws -> [Interval] {
        if length <= size {
            return [Interval(start: 0, end: length, leftRamp: 0, rightRamp: 0)]
        }
        var intervals = try splitBySize(length: length, size: size, overlap: overlap)
        guard intervals.count > 1 else { return intervals }
        for i in 1 ..< intervals.count {
            intervals[i].start -= 1
            intervals[i].leftRamp += 1
        }
        return intervals
    }

    // MARK: - Masks

    /// The 1-D trapezoidal blend mask, value for value.
    ///
    /// `leftStartsFromZero` is the temporal case: the fade-in is `i / rampLeft` and so
    /// *does* reach 0, because the tile before it holds a full-weight frame there. The
    /// spatial case is `(i + 1) / (rampLeft + 1)`, which never reaches 0. Both fade-outs
    /// are `(rampRight − j) / (rampRight + 1)`.
    ///
    /// These specific rationals matter: `24/25 + 1/25` is exactly `1.0` in fp32 (checked
    /// over every ramp position this policy produces), so the blend needs no denominator.
    ///
    /// Exactness is *sufficient* for skipping the denominator but it is not the test. The
    /// test is that each axis's masks sum to one within `1e-5`, and a decode that fails it
    /// has to divide by the accumulated weight instead. This implementation keeps the
    /// tighter behaviour — `plan` throws ``Failure/notComplementary(axis:worst:)`` at that
    /// same `1e-5` rather than falling back to a division no boundary here has measured —
    /// and the spatial ramps clear it by construction: on both axes the overlap pairs
    /// `(L−p)/(L+1)` with `(p+1)/(L+1)`, whose numerators sum to `L+1`.
    static func trapezoid(length: Int, rampLeft: Int, rampRight: Int,
                          leftStartsFromZero: Bool) -> [Float] {
        precondition(length > 0, "mask length must be positive")
        let l = max(0, min(rampLeft, length))
        let r = max(0, min(rampRight, length))
        var mask = [Float](repeating: 1, count: length)
        if l > 0 {
            for i in 0 ..< l {
                mask[i] *= leftStartsFromZero ? Float(i) / Float(l)
                                              : Float(i + 1) / Float(l + 1)
            }
        }
        if r > 0 {
            for j in 0 ..< r {
                mask[length - r + j] *= Float(r - j) / Float(r + 1)
            }
        }
        return mask.map { min(max($0, 0), 1) }
    }

    // MARK: - Layout

    /// One tile on one axis: where it is cut from the latent, where its decode lands in
    /// the output, and the 1-D blend mask over that output range.
    public struct Tile: Equatable, Sendable {
        public var latent: Range<Int>
        public var output: Range<Int>
        public var mask: [Float]
    }

    public struct AxisLayout: Equatable, Sendable {
        public var tiles: [Tile]
        public var outputLength: Int

        public var isTiled: Bool { tiles.count > 1 }

        /// The overlap length, in output units, between tile `i` and tile `i + 1`.
        public func overlap(after i: Int) -> Int {
            guard i + 1 < tiles.count else { return 0 }
            return tiles[i].output.upperBound - tiles[i + 1].output.lowerBound
        }

        /// The partition-of-unity check: these should sum to one at every position.
        public func maskSum() -> [Float] {
            var acc = [Float](repeating: 0, count: outputLength)
            for tile in tiles {
                for (i, m) in tile.mask.enumerated() { acc[tile.output.lowerBound + i] += m }
            }
            return acc
        }
    }

    public struct Layout: Equatable, Sendable {
        public var frames: AxisLayout
        public var height: AxisLayout
        public var width: AxisLayout
        public var scale: Scale

        public var outputShape: (frames: Int, height: Int, width: Int) {
            (frames.outputLength, height.outputLength, width.outputLength)
        }

        /// Total decoder invocations — the memory strategy, not a contract.
        public var tileCount: Int { frames.tiles.count * height.tiles.count * width.tiles.count }
    }

    /// The temporal mapping: latent `[b, e)` → output `[8b, 8(e−1)+1)`.
    ///
    /// The asymmetry between `start` and `stop` *is* the causality: the tile's first
    /// latent frame contributes one pixel frame and each subsequent one contributes
    /// eight. The left ramp maps the same way (`1 + (l−1)·8`), the right ramp does not
    /// (`r·8`), because only the left edge contains the tile's own first frame.
    static func mapTemporal(_ interval: Interval, scale: Int) -> Tile {
        let start = interval.start * scale
        let stop = 1 + (interval.end - 1) * scale
        let left = interval.leftRamp == 0 ? 0 : 1 + (interval.leftRamp - 1) * scale
        let right = interval.rightRamp * scale
        return Tile(latent: interval.start ..< interval.end,
                    output: start ..< stop,
                    mask: trapezoid(length: stop - start, rampLeft: left, rampRight: right,
                                    leftStartsFromZero: true))
    }

    /// The spatial mapping: a plain `× scale` on every quantity, no causal offset.
    static func mapSpatial(_ interval: Interval, scale: Int) -> Tile {
        let start = interval.start * scale
        let stop = interval.end * scale
        return Tile(latent: interval.start ..< interval.end,
                    output: start ..< stop,
                    mask: trapezoid(length: stop - start,
                                    rampLeft: interval.leftRamp * scale,
                                    rampRight: interval.rightRamp * scale,
                                    leftStartsFromZero: false))
    }

    /// Policy in pixel units → latent-grid splits → output tiles.
    ///
    /// Note that the pixel policy is divided by the VAE factor *before* splitting, so the
    /// 80/24 frame policy is a 10/3 latent split, and the lower threshold floors the tile
    /// at `overlap + 1` latent units.
    public static func plan(latentFrames: Int, latentHeight: Int, latentWidth: Int,
                            policy: Policy = .reference,
                            scale: Scale = .video) throws -> Layout {
        guard latentFrames > 0, latentHeight > 0, latentWidth > 0 else {
            throw Failure.policy("latent extents must be positive, got "
                + "\(latentFrames)x\(latentHeight)x\(latentWidth)")
        }

        func latentTile(_ axis: AxisPolicy, _ factor: Int, _ name: String) throws -> (Int, Int) {
            guard axis.tileSize >= 2 * factor else {
                throw Failure.policy("\(name).tileSize must be at least \(2 * factor), "
                    + "got \(axis.tileSize)")
            }
            guard axis.tileSize % factor == 0 else {
                throw Failure.policy("\(name).tileSize must be divisible by \(factor), "
                    + "got \(axis.tileSize)")
            }
            guard axis.overlap % factor == 0 else {
                throw Failure.policy("\(name).overlap must be divisible by \(factor), "
                    + "got \(axis.overlap)")
            }
            let size = axis.tileSize / factor
            let overlap = axis.overlap / factor
            return (max(max(2, overlap + 1), size), overlap)
        }

        func axisLayout(_ axis: AxisPolicy, latent: Int, factor: Int, name: String,
                        temporal: Bool) throws -> AxisLayout {
            let intervals: [Interval]
            if axis.isTiled {
                let (size, overlap) = try latentTile(axis, factor, name)
                intervals = temporal
                    ? try splitTemporalCausal(length: latent, size: size, overlap: overlap)
                    : try splitBySize(length: latent, size: size, overlap: overlap)
            } else {
                intervals = [Interval(start: 0, end: latent, leftRamp: 0, rightRamp: 0)]
            }
            let tiles = intervals.map { temporal ? mapTemporal($0, scale: factor)
                                                 : mapSpatial($0, scale: factor) }
            let outputLength = temporal ? 1 + (latent - 1) * factor : latent * factor
            let layout = AxisLayout(tiles: tiles, outputLength: outputLength)
            if let worst = layout.maskSum().first(where: { abs($0 - 1) > 1e-5 }) {
                throw Failure.notComplementary(axis: name, worst: worst)
            }
            return layout
        }

        return Layout(
            frames: try axisLayout(policy.frames, latent: latentFrames, factor: scale.time,
                                   name: "frames", temporal: true),
            height: try axisLayout(policy.height, latent: latentHeight, factor: scale.height,
                                   name: "height", temporal: false),
            width: try axisLayout(policy.width, latent: latentWidth, factor: scale.width,
                                  name: "width", temporal: false),
            scale: scale)
    }

    // MARK: - The decode

    /// Decode `[1, C, F, H, W]` of latent into `[1, 3, 8(F−1)+1, H·32, W·32]`, tiled.
    ///
    /// `decodeTile` is the whole decoder for one latent tile. It is a closure rather than
    /// a `VideoVAEDecoder` so that the composition — where every index and mask error
    /// lives — can be tested against a known ramp without loading 1.4 GB of weights.
    ///
    /// The returned tensor is the raw decoder range, **before** `rgb(_:)`: the caller
    /// applies the `(x + 1)/2` clamp once, on the assembled video.
    public static func decode(_ latent: MLXArray, layout: Layout,
                              decodeTile: (MLXArray) throws -> MLXArray) throws -> MLXArray {
        guard latent.ndim == 5 else { throw Failure.latentRank(latent.shape) }

        let tiles = layout.frames.tiles
        let spatiallyTiled = layout.height.tiles.count > 1 || layout.width.tiles.count > 1

        /// Decode one `(t, h, w)` sub-volume and check it against the layout on all three
        /// axes. Cast to bf16 on the way out: `decode(_:)` returns the checkpoint's own
        /// dtype and the accumulation buffer is the latent's.
        func decodeSub(_ t: Tile, _ h: Tile, _ w: Tile) throws -> MLXArray {
            let slice = latent[0..., 0..., t.latent.lowerBound ..< t.latent.upperBound,
                               h.latent.lowerBound ..< h.latent.upperBound,
                               w.latent.lowerBound ..< w.latent.upperBound]
            let decoded = try decodeTile(slice).asType(.bfloat16)
            guard decoded.shape[2] == t.output.count else {
                throw Failure.tileLength(expected: t.output.count, got: decoded.shape[2])
            }
            guard decoded.shape[3] == h.output.count else {
                throw Failure.tileExtent(axis: "height", expected: h.output.count,
                                         got: decoded.shape[3])
            }
            guard decoded.shape[4] == w.output.count else {
                throw Failure.tileExtent(axis: "width", expected: w.output.count,
                                         got: decoded.shape[4])
            }
            return decoded
        }

        /// One temporal group's contribution over the whole frame.
        ///
        /// **One spatial tile** — the overwhelmingly common case, every render at or below
        /// a 768 px long side — returns the fp32 masked product and rounds nothing: the
        /// mask is fp32 so the product promotes, and the first rounding happens where a
        /// piece is committed below. That is the expression the recorded boundary was
        /// measured through and it is untouched.
        ///
        /// **More than one** fills a group buffer allocated in the latent's dtype, into
        /// which each tile lands with `buffer[coords] += masked(decoded)`, so every spatial
        /// seam is `bf16( bf16(accumulated) + incoming_fp32 )`. The mask factors multiply
        /// in axis order — t, then h, then w — and the tiles are visited in `(h, w)` order
        /// so the four-tile corner rounds in one fixed sequence and not some other one.
        func slab(_ index: Int) throws -> MLXArray {
            let t = tiles[index]
            let tMask = MLXArray(t.mask).reshaped([1, 1, t.mask.count, 1, 1])

            if !spatiallyTiled {
                return try decodeSub(t, layout.height.tiles[0], layout.width.tiles[0]) * tMask
            }

            var buffer: MLXArray?
            for h in layout.height.tiles {
                let hMask = MLXArray(h.mask).reshaped([1, 1, 1, h.mask.count, 1])
                for w in layout.width.tiles {
                    let wMask = MLXArray(w.mask).reshaped([1, 1, 1, 1, w.mask.count])
                    let decoded = try decodeSub(t, h, w)
                    let contribution = decoded * tMask * hMask * wMask
                    if buffer == nil {
                        buffer = MLXArray.zeros(
                            [decoded.shape[0], decoded.shape[1], t.output.count,
                             layout.height.outputLength, layout.width.outputLength],
                            dtype: .bfloat16)
                    }
                    let rows = h.output.lowerBound ..< h.output.upperBound
                    let cols = w.output.lowerBound ..< w.output.upperBound
                    let accumulated = buffer![0..., 0..., 0..., rows, cols]
                    buffer![0..., 0..., 0..., rows, cols] =
                        (accumulated + contribution).asType(.bfloat16)
                    // Force the tile out of the graph. `eval` cannot change a value, and
                    // without it MLX would hold every sub-decode live until the whole
                    // frame was assembled — which is the memory spatial tiling exists to
                    // avoid spending.
                    buffer!.eval()
                }
            }
            return buffer!
        }

        func frames(_ x: MLXArray, _ range: Range<Int>) -> MLXArray {
            x[0..., 0..., range.lowerBound ..< range.upperBound, 0..., 0...]
        }

        // Only adjacent tiles ever overlap — the temporal stride is 56 output frames and
        // the seam is 25 — so the output is a strict alternation of solo runs and
        // two-tile seams, and never needs a third term.
        var pieces: [MLXArray] = []
        var carry: MLXArray?                      // the previous tile's bf16 seam tail

        for k in tiles.indices {
            let tile = tiles[k]
            let product = try slab(k)
            let headEnd = k > 0 ? layout.frames.overlap(after: k - 1) : 0
            let tailStart = k + 1 < tiles.count
                ? tiles[k + 1].output.lowerBound - tile.output.lowerBound
                : tile.output.count

            if let previous = carry, headEnd > 0 {
                // bf16(previous) + fp32(current): the measured seam. See the type note in
                // this file's header — a streamed decode rounds the second term as well,
                // and that would change the 13-frame answer.
                //
                // When the frame is spatially tiled the second term arrives already bf16,
                // because the spatial blend had to round it into the group buffer. That is
                // not a second policy: it is the same `previous[ovl] += buffer[ovl]`,
                // reached only on layouts that never had the fp32 boundary in the first
                // place. Every layout that did — one spatial tile — still gets fp32 here.
                pieces.append((previous + frames(product, 0 ..< headEnd)).asType(.bfloat16))
            }
            if tailStart > headEnd {
                pieces.append(frames(product, headEnd ..< tailStart).asType(.bfloat16))
            }
            carry = k + 1 < tiles.count
                ? frames(product, tailStart ..< tile.output.count).asType(.bfloat16)
                : nil
        }

        let assembled = pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 2)
        guard assembled.shape[2] == layout.frames.outputLength else {
            throw Failure.tileLength(expected: layout.frames.outputLength,
                                     got: assembled.shape[2])
        }
        return assembled
    }
}

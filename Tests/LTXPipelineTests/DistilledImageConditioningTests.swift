// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
import UniformTypeIdentifiers

import LTXFoundation
import LTXModules
@testable import LTXPipeline

/// `--two-stage` with `--image`: one still, two conditioning pairs, one per stage.
///
/// ## The contract, and what a wrong implementation would look like
///
/// Stage 1 conditions at half the output's pixel dimensions and stage 2 at the full ones.
/// **The image is preprocessed to each stage's pixel dimensions and encoded separately per
/// stage. It is not one latent that gets downscaled.**
///
/// The two wrong implementations that would still render:
///
/// * encode once at full size and pool or interpolate the *latent* down for stage 1. Right
///   shapes, wrong tensor — the VAE's 32x spatial compression is not an average, and the
///   half-resolution encode of an image is not the downsample of its full-resolution encode.
/// * condition stage 2 only. The anchor applies at stage 2 *as well as* stage 1, so a
///   stage-2-only version loses the draft's anchor and a stage-1-only version lets three
///   full-resolution refinement steps walk off it.
///
/// ## What this suite can hold without a checkpoint
///
/// Everything except the VAE forward. ``LTXPipeline/DistilledRenderer/conditioningPixels(at:spec:)``
/// is the split that makes that possible: it produces the two resized frames, which is where
/// the per-stage claim actually lives, and `RenderConditioning.build` then decides the rest
/// from shapes alone. The encode between them is the same `encodeImage` the single-stage i2v
/// path already uses and is not re-checked here.
@Suite("Distilled image conditioning")
struct DistilledImageConditioningTests {

    // MARK: - Fixtures

    static func scratch() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-distilled-cond-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// An opaque RGB png written from bytes, so nothing about the fixture is CoreGraphics'
    /// opinion. `value` is `(row, column, channel) -> 0...255`.
    static func writePNG(width: Int, height: Int, to url: URL,
                         _ value: (Int, Int, Int) -> UInt8) throws {
        var bytes = [UInt8]()
        bytes.reserveCapacity(width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                for c in 0 ..< 3 { bytes.append(value(y, x, c)) }
                bytes.append(255)
            }
        }
        let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)
        let image = try #require(CGImage(width: width, height: height, bitsPerComponent: 8,
                                         bitsPerPixel: 32, bytesPerRow: width * 4,
                                         space: space, bitmapInfo: info, provider: provider,
                                         decode: nil, shouldInterpolate: false,
                                         intent: .defaultIntent))
        let destination = try #require(CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [:] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination), "could not write \(url.path)")
    }

    /// The smallest legal two-stage geometry: 128x128 halves to 64x64, and both sit on the
    /// VAE's 32-pixel latent grid. 9 frames is one 8k+1 step.
    static func spec() -> DistilledRenderer.Spec {
        DistilledRenderer.Spec(
            prompt: "x",
            geometry: DiTForward.Geometry(frames: 9, height: 128, width: 128, frameRate: 24),
            seed: 0)
    }

    // MARK: - Two frames, two sizes, one file

    @Test("One source image resizes to stage 1's half size and stage 2's full size")
    func onePixelSourceGivesTwoStageSizes() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("anchor.png")
        // Deliberately neither square nor either target size: a resizer that happened to be
        // the identity, or that only handled the exact output shape, would pass on a
        // fixture already at 128x128.
        try Self.writePNG(width: 300, height: 200, to: url) { y, x, c in
            UInt8((y * 3 + x * 5 + c * 7) % 256)
        }

        let pixels = try DistilledRenderer.conditioningPixels(at: url, spec: Self.spec())

        // The whole per-stage claim, in two shapes: stage 1 is half of stage 2 in pixels,
        // which is what makes their latents half in each spatial axis.
        #expect(pixels.stage1.shape == [1, 64, 64, 3])
        #expect(pixels.stage2.shape == [1, 128, 128, 3])
        #expect(Self.spec().stage1Geometry.height == 64)
        #expect(Self.spec().stage1Geometry.width == 64)

        // Both are real content rather than a zero or constant fill — a resize that
        // collapsed would still have the right shape.
        for frame in [pixels.stage1, pixels.stage2] {
            let low = frame.min().item(Float.self), high = frame.max().item(Float.self)
            #expect(low >= 0 && high <= 1)
            #expect(high - low > 0.5)
        }
    }

    /// The stage-1 frame must come from the *source*, not from stage 2's frame.
    ///
    /// A composed resize (source → full size → half size) runs the bilinear filter twice and
    /// is a different image. The difference is not subtle once the filter is the
    /// **non-antialiased** one this path uses, and this drives it with a one-pixel-wide
    /// vertical line in a 256-wide source:
    ///
    /// * **direct, 256 → 64.** `scale = 4`, so the taps are `4d + 1.5`: `d = 31` samples
    ///   columns 125 and 126, `d = 32` samples 130 and 131. Column 128 is **never sampled**
    ///   and the line disappears completely — `max` is 0.
    /// * **composed, 256 → 128 → 64.** The first pass (`scale = 2`) puts half the line into
    ///   output column 63, and the second pass then splits that in half again — `max` is
    ///   0.25.
    ///
    /// So the two differ by the whole feature, and the numbers are hand-computable rather
    /// than "one is softer than the other". That a thin feature can vanish entirely is a
    /// property of bilinear sampling without an antialiasing prefilter, not a defect: the
    /// conditioning path resamples that way by specification.
    @Test("Stage 1 is resized from the source, not from stage 2's frame")
    func stage1IsNotAResizeOfStage2() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("line.png")
        try Self.writePNG(width: 256, height: 256, to: url) { _, x, _ in
            x == 128 ? 255 : 0
        }

        let pixels = try DistilledRenderer.conditioningPixels(at: url, spec: Self.spec())
        let composed = try MediaInput.resizeAndCenterCrop(pixels.stage2, height: 64, width: 64)

        #expect(pixels.stage2.max().item(Float.self) == 0.5)
        #expect(pixels.stage1.max().item(Float.self) == 0.0)
        #expect(composed.max().item(Float.self) == 0.25)
    }

    /// `resizeAndCenterCrop` is aspect-**fill** then crop, not fit-then-pad and not a
    /// stretch. A 2:1 source cropped to 1:1 must lose the sides and keep the full height.
    @Test("The resize fills the target and crops the overflow, rather than padding it")
    func resizeFillsAndCrops() throws {
        // 4 wide, 2 high, with a black border column at each end and white in the middle.
        // Filling a 2x2 target scales by max(2/2, 2/4) = 1.0, so new is 4x2 and the crop
        // takes columns 1 and 2 — the white ones. A fit-then-pad would keep the black.
        let source = MediaInputTests.clip(frames: 1, height: 2, width: 4) { _, _, x, _ in
            (x == 0 || x == 3) ? 0 : 1
        }
        let out = try MediaInput.resizeAndCenterCrop(source, height: 2, width: 2)
        #expect(out.shape == [1, 2, 2, 3])
        #expect(out.min().item(Float.self) == 1.0)
    }

    /// The two-tap map, computed by hand from `align_corners=False`.
    ///
    /// A 1-D ramp `[0, 1, 2, 3]` resized to 2 samples: `scale = 2`, so the sources are
    /// `2*(0.5) - 0.5 = 0.5` and `2*(1.5) - 0.5 = 2.5`, giving `0.5*(0+1) = 0.5` and
    /// `0.5*(2+3) = 2.5`. An `align_corners=True` resize would give `0` and `3`; an
    /// antialiased (area) one would give the same `0.5, 2.5` here, which is why the
    /// upscale below is the discriminating half — area upsampling repeats, bilinear ramps.
    @Test("Bilinear with align_corners=False, on a ramp worked out by hand")
    func bilinearMatchesTheHandComputedMap() throws {
        let ramp = MediaInputTests.clip(frames: 1, height: 1, width: 4) { _, _, x, _ in
            Float(x)
        }
        let down = MediaInput.bilinearAlongAxis(ramp, axis: 2, outSize: 2)
        #expect(down[0, 0, 0, 0].item(Float.self) == 0.5)
        #expect(down[0, 0, 1, 0].item(Float.self) == 2.5)

        // Upscale 2 -> 4 of [0, 1]: sources are 0.5*(d+0.5) - 0.5 = -0.25, 0.25, 0.75, 1.25,
        // clamped at 0 and pinned at index 1, giving 0, 0.25, 0.75, 1.
        let pair = MediaInputTests.clip(frames: 1, height: 1, width: 2) { _, _, x, _ in
            Float(x)
        }
        let up = MediaInput.bilinearAlongAxis(pair, axis: 2, outSize: 4)
        let got = (0 ..< 4).map { up[0, 0, $0, 0].item(Float.self) }
        #expect(got == [0.0, 0.25, 0.75, 1.0])
    }

    // MARK: - The pair each stage's geometry demands

    /// A VAE encode of one pixel frame at `H x W` is `[1, 128, 1, H/32, W/32]`, and
    /// `RenderConditioning.build` accepts exactly that and nothing else. So the two stages'
    /// latent shapes are decided here, without an encoder in the room — and swapping them is
    /// an error rather than a broadcast.
    @Test("Each stage's geometry demands its own latent shape, and refuses the other's")
    func eachStageDemandsItsOwnLatentShape() throws {
        let g1 = Self.spec().stage1Geometry, g2 = Self.spec().geometry
        let scale = g2.latent.spatialScale
        let channels = g2.latent.videoChannels

        func latent(_ g: DiTForward.Geometry) -> MLXArray {
            MLXArray.zeros([1, channels, 1, g.height / scale, g.width / scale],
                           dtype: .float32)
        }
        #expect(latent(g1).shape == [1, channels, 1, 2, 2])
        #expect(latent(g2).shape == [1, channels, 1, 4, 4])

        let ownG1 = try RenderConditioning.build(imageLatent: latent(g1), strength: 1.0,
                                                 geometry: g1)
        let ownG2 = try RenderConditioning.build(imageLatent: latent(g2), strength: 1.0,
                                                 geometry: g2)
        #expect(ownG1 != nil)
        #expect(ownG2 != nil)
        // The mistake this catches is the one a single shared pair would make.
        #expect(throws: RenderConditioning.Failure.self) {
            _ = try RenderConditioning.build(imageLatent: latent(g2), strength: 1.0,
                                             geometry: g1)
        }
        #expect(throws: RenderConditioning.Failure.self) {
            _ = try RenderConditioning.build(imageLatent: latent(g1), strength: 1.0,
                                             geometry: g2)
        }
    }

    /// Conditioning by latent index sets the denoise mask to `1 - strength` over one latent
    /// frame's tokens and touches nothing else.
    ///
    /// Checked at **both** stages because the token counts differ (4 against 16 here, 240
    /// against 960 at production) and a pair built for one stage would be silently wrong at
    /// the other.
    @Test("The mask is 1 - strength on exactly the first latent frame's tokens, per stage")
    func maskCoversOneLatentFrameAtEachStage() throws {
        let strength = 0.75
        let held = Float(1.0 - strength)

        for (stage, geometry) in [(1, Self.spec().stage1Geometry), (2, Self.spec().geometry)] {
            let scale = geometry.latent.spatialScale
            let image = MLXArray.zeros(
                [1, geometry.latent.videoChannels, 1,
                 geometry.height / scale, geometry.width / scale], dtype: .float32)
            let built = try #require(try RenderConditioning.build(
                imageLatent: image, strength: strength, geometry: geometry))

            let mask = built.denoiseMask.video
            let perFrame = geometry.tokensPerLatentFrame
            #expect(mask.shape == [1, geometry.videoTokens, 1], "stage \(stage)")
            #expect(perFrame == (geometry.height / scale) * (geometry.width / scale))

            let frozen = mask[0, 0 ..< perFrame, 0]
            let free = mask[0, perFrame ..< geometry.videoTokens, 0]
            #expect(frozen.min().item(Float.self) == held, "stage \(stage)")
            #expect(frozen.max().item(Float.self) == held, "stage \(stage)")
            #expect(free.min().item(Float.self) == 1.0, "stage \(stage)")
            // The boundary, stated as a count rather than as two ends: exactly one latent
            // frame's worth of tokens is held.
            let heldCount = MLX.sum((mask .< MLXArray(Float(1))).asType(.int32))
            #expect(heldCount.item(Int.self) == perFrame, "stage \(stage)")
            // fp32, because `postProcess` computes the blend in the mask's dtype.
            #expect(mask.dtype == .float32, "stage \(stage)")
            #expect(built.mode == .imageToVideo)
        }
    }

    /// Both stages carry the anchor: the same conditioning image is applied at each, so
    /// `--strength` is not a stage-1-only hint.
    @Test("The anchor is present at stage 2 as well as stage 1")
    func bothStagesCarryTheAnchor() throws {
        var pairs: [Int: RenderConditioning.Built] = [:]
        for (stage, geometry) in [(1, Self.spec().stage1Geometry), (2, Self.spec().geometry)] {
            let scale = geometry.latent.spatialScale
            pairs[stage] = try RenderConditioning.build(
                imageLatent: MLXArray.zeros(
                    [1, geometry.latent.videoChannels, 1,
                     geometry.height / scale, geometry.width / scale], dtype: .float32),
                strength: 1.0, geometry: geometry)
        }
        for stage in [1, 2] {
            let mask = try #require(pairs[stage]).denoiseMask.video
            #expect(mask.min().item(Float.self) == 0.0,
                    "stage \(stage) has no frozen token at all")
        }
        // And they are *different* pairs: the token counts differ, which is exactly what
        // `DistilledRenderer.checkConditioning` refuses to let be shared.
        #expect(try #require(pairs[1]).denoiseMask.video.dim(1)
                != (try #require(pairs[2]).denoiseMask.video.dim(1)))
    }

    /// The seam that refuses a shared pair, driven from the same two geometries.
    @Test("A stage-1 pair handed to stage 2 is refused at the seam")
    func mismatchedPairsAreRefused() throws {
        let g1 = Self.spec().stage1Geometry, g2 = Self.spec().geometry
        let scale = g1.latent.spatialScale
        let pair = try #require(try RenderConditioning.build(
            imageLatent: MLXArray.zeros([1, g1.latent.videoChannels, 1,
                                         g1.height / scale, g1.width / scale],
                                        dtype: .float32),
            strength: 1.0, geometry: g1))

        #expect(throws: Never.self) {
            try DistilledRenderer.checkConditioning(pair, geometry: g1, stage: 1)
        }
        #expect(throws: DistilledRenderer.Failure.self) {
            try DistilledRenderer.checkConditioning(pair, geometry: g2, stage: 2)
        }
    }
}

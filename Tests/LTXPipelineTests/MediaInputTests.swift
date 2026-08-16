// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MLX
import Testing
import UniformTypeIdentifiers

@testable import LTXPipeline

/// `MediaInput`, exercised against files this suite writes itself.
///
/// **No checkpoint and no sample media in the repo.** Every fixture is synthesised:
/// a PNG built from a byte array, a clip written by ``RenderOutput``, a WAV written by
/// ``RenderOutput/writeWAV(audio:to:sampleRate:)``. That is not only about speed — a reader
/// checked against a committed `.mp4` is checked against whatever that file happens to
/// contain, and the interesting properties here (the row stride, the channel order, the
/// alpha policy) are properties of *awkward* files, which have to be manufactured on
/// purpose.
///
/// The organising claim is that this is the inverse of ``RenderOutput``. Where a claim can
/// be made exact it is:
///
/// | check | strength |
/// |---|---|
/// | a lossless PNG round-trips to the exact `k / 255` | **exact** — hand-computed floats |
/// | alpha composites over the named background | **exact** at α = 0 and α = 1 |
/// | EXIF orientation lands each pixel where the tag says | **exact** — hand-computed positions |
/// | a WAV round trip through `writeWAV` | **exact** — fp32 in, fp32 out |
/// | an achromatic H.264 round trip | approximate, ±0.05; the tolerance is **measured and printed** |
/// | a colour H.264 round trip | approximate, ±0.08; measured, and a ratchet on the writer's colour tag |
/// | the row stride is not ignored | exact — the edge column is identical on every row |
/// | R is not B | approximate, ±0.2 on a 1.0 gap |
/// | each malformed input throws its own named case | exact |
///
/// ## Why the H.264 round trip is split in two
///
/// It was not split for tidiness. When this suite was first written ``RenderOutput`` wrote
/// files with no colour tags at all, so the encoder and the decoder each picked a YCbCr
/// matrix and picked differently — BT.709 in, BT.601 out, identified from a pure-red pixel
/// to three decimals. The colour round trip measured a mean of 6.5/255 against 0.46/255 for
/// the same clip made achromatic, and it was the 14× gap between those two numbers that
/// made the cause legible: a codec artefact would have moved both. ``RenderOutput`` now
/// tags Rec.709 and the colour mean is 0.74/255. The split stays because it is the only
/// place the two can be compared, and a writer that stopped tagging would still look
/// correct from its own side.
///
/// ## Three things here have no test, and this is the list
///
/// - ``MediaInput/Failure/frameSizeChanged(_:frame:from:to:)``. AVAssetWriter will not
///   produce a single track whose sample dimensions change mid-stream.
/// - ``MediaInput/Failure/noFrames(_:)``. See ``damagedContainerThrows()``: the file that
///   was supposed to trigger it cannot be opened at all, so the read fails a step earlier.
/// - The four mirrored EXIF orientations (2, 4, 5, 7). Two of the eight are checked against
///   a tagged file; the other six share the derivation but not the evidence, and 5 and 7 in
///   particular are the ones nobody can check by looking.
@Suite("Media input")
struct MediaInputTests {

    // MARK: - Fixtures

    static func scratch() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-media-input-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// `[F, H, W, 3]` from a closure over `(frame, row, column, channel)` — the same
    /// fixture shape ``RenderOutputTests`` uses, and for the same reason: an expected value
    /// worked out by hand is worth more than one produced by a second implementation.
    static func clip(frames: Int, height: Int, width: Int,
                     _ value: (Int, Int, Int, Int) -> Float) -> MLXArray {
        var flat = [Float]()
        flat.reserveCapacity(frames * height * width * 3)
        for f in 0 ..< frames {
            for y in 0 ..< height {
                for x in 0 ..< width {
                    for c in 0 ..< 3 { flat.append(value(f, y, x, c)) }
                }
            }
        }
        return MLXArray(flat, [frames, height, width, 3])
    }

    static func stereo(frames: Int, left: (Int) -> Float,
                       right: (Int) -> Float) -> MLXArray {
        var flat = [Float]()
        flat.reserveCapacity(frames * 2)
        for i in 0 ..< frames { flat.append(left(i)) }
        for i in 0 ..< frames { flat.append(right(i)) }
        return MLXArray(flat, [1, 2, frames])
    }

    /// A `CGImage` from straight-alpha RGBA bytes.
    ///
    /// Built through a `CGDataProvider` rather than by drawing, so the fixture's bytes are
    /// the fixture's bytes: drawing into a context to make the image under test would put
    /// the same CoreGraphics conversion on both sides of the comparison.
    static func makeImage(width: Int, height: Int, rgba: [UInt8]) throws -> CGImage {
        let data = Data(rgba)
        let provider = try #require(CGDataProvider(data: data as CFData))
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)
        return try #require(CGImage(width: width, height: height, bitsPerComponent: 8,
                                    bitsPerPixel: 32, bytesPerRow: width * 4, space: space,
                                    bitmapInfo: info, provider: provider, decode: nil,
                                    shouldInterpolate: false, intent: .defaultIntent))
    }

    static func write(_ image: CGImage, to url: URL, type: UTType,
                      properties: [CFString: Any] = [:]) throws {
        let destination = try #require(
            CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        #expect(CGImageDestinationFinalize(destination), "could not write \(url.path)")
    }

    /// Run `body`, require a `MediaInput.Failure`, and hand the case to `check`. `Failure`
    /// carries an `Error` in two cases, so it is not `Equatable` and matching is by
    /// pattern — the same shape ``RenderOutputTests/expectFailure(_:sourceLocation:_:matches:)``
    /// uses.
    static func expectFailure(_ description: Comment,
                              sourceLocation: SourceLocation = #_sourceLocation,
                              _ body: () async throws -> Void,
                              matches check: (MediaInput.Failure) -> Bool) async {
        do {
            try await body()
            Issue.record("expected a throw: \(description)", sourceLocation: sourceLocation)
        } catch let failure as MediaInput.Failure {
            #expect(check(failure), "wrong case: \(failure)", sourceLocation: sourceLocation)
        } catch {
            Issue.record("threw the wrong error type: \(error)",
                         sourceLocation: sourceLocation)
        }
    }

    /// The decoder's row stride for a file, read directly. Used to say out loud whether the
    /// stride test is exercising anything: if `bytesPerRow == width * 4` on this machine
    /// then a stride bug would be invisible and the assertion below is vacuous, and a test
    /// that cannot tell you that is worse than no test.
    static func decodedStride(_ url: URL) throws -> (width: Int, bytesPerRow: Int)? {
        let asset = AVURLAsset(url: url)
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: (Int, Int)?
        Task.detached {
            defer { semaphore.signal() }
            guard let track = try? await asset.loadTracks(withMediaType: .video).first,
                  let reader = try? AVAssetReader(asset: asset) else { return }
            let output = AVAssetReaderTrackOutput(
                track: track,
                outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ])
            reader.add(output)
            guard reader.startReading(), let sample = output.copyNextSampleBuffer(),
                  let image = CMSampleBufferGetImageBuffer(sample) else { return }
            result = (CVPixelBufferGetWidth(image), CVPixelBufferGetBytesPerRow(image))
            reader.cancelReading()
        }
        semaphore.wait()
        return result.map { (width: $0.0, bytesPerRow: $0.1) }
    }

    // MARK: - Images

    /// PNG is lossless, so the claim is identity: byte `k` in the file becomes exactly
    /// `k / 255` in the tensor.
    ///
    /// The fixture is asymmetric in every axis on purpose — a different value per row, per
    /// column and per channel. A vertical flip, a horizontal flip, a transpose and a
    /// channel rotation are each individually invisible against a fixture that is symmetric
    /// in the corresponding axis, and a bitmap context is a place where all four are one
    /// wrong constant away.
    @Test("a PNG round-trips to exactly k / 255, unflipped and unswapped")
    func readsAPNGExactly() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("known.png")

        let w = 5, h = 3
        func byte(_ x: Int, _ y: Int, _ c: Int) -> UInt8 {
            UInt8((17 + x * 31 + y * 71 + c * 13) % 256)
        }
        var rgba = [UInt8]()
        for y in 0 ..< h {
            for x in 0 ..< w {
                for c in 0 ..< 3 { rgba.append(byte(x, y, c)) }
                rgba.append(255)
            }
        }
        try Self.write(try Self.makeImage(width: w, height: h, rgba: rgba), to: url,
                       type: .png)

        let tensor = try MediaInput.image(at: url)
        #expect(tensor.shape == [1, h, w, 3], "shape \(tensor.shape)")
        let values = tensor.asArray(Float.self)
        #expect(values.allSatisfy { $0 >= 0 && $0 <= 1 }, "values escaped [0, 1]")

        for y in 0 ..< h {
            for x in 0 ..< w {
                for c in 0 ..< 3 {
                    let got = values[(y * w + x) * 3 + c]
                    let want = Float(byte(x, y, c)) / 255
                    #expect(abs(got - want) < 1e-6,
                            "pixel (\(x), \(y)) channel \(c) is \(got), expected \(want)")
                }
            }
        }
    }

    /// A greyscale source. The failure this is the control for is reading a `CGImage`'s
    /// bytes directly: a one-component-per-pixel image read as RGBA is not merely the wrong
    /// colour, it is the wrong *geometry*, and the redraw into a pinned RGBA context is what
    /// makes it a non-event.
    @Test("a greyscale PNG comes back as three equal channels")
    func readsGreyscale() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("grey.png")

        let w = 4, h = 2
        let space = try #require(CGColorSpace(name: CGColorSpace.genericGrayGamma2_2))
        let context = try #require(CGContext(data: nil, width: w, height: h,
                                             bitsPerComponent: 8, bytesPerRow: w,
                                             space: space,
                                             bitmapInfo: CGImageAlphaInfo.none.rawValue))
        // A different level per row, so a vertical flip would be visible here too.
        let data = try #require(context.data)
        let bytes = data.bindMemory(to: UInt8.self, capacity: w * h)
        for y in 0 ..< h {
            for x in 0 ..< w { bytes[y * w + x] = UInt8(40 + y * 100) }
        }
        try Self.write(try #require(context.makeImage()), to: url, type: .png)

        let tensor = try MediaInput.image(at: url)
        #expect(tensor.shape == [1, h, w, 3], "shape \(tensor.shape)")
        let values = tensor.asArray(Float.self)
        for i in stride(from: 0, to: values.count, by: 3) {
            #expect(abs(values[i] - values[i + 1]) < 1e-6)
            #expect(abs(values[i] - values[i + 2]) < 1e-6)
        }
        // Grey 40 and 140 through the gamma-2.2 grey space into sRGB are not 40/255 and
        // 140/255 — that is the colour conversion doing its job — but row 1 must still be
        // brighter than row 0, and neither row may be flat black or flat white.
        let row0 = values[0], row1 = values[w * 3]
        #expect(row1 > row0 + 0.2, "row 0 = \(row0), row 1 = \(row1)")
        #expect(row0 > 0.01 && row1 < 0.99, "row 0 = \(row0), row 1 = \(row1)")
    }

    @Test("a JPEG is read and is close to the values that went in")
    func readsAJPEG() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("known.jpg")

        // Flat blocks rather than a per-pixel pattern: JPEG is lossy and a high-frequency
        // fixture would be measuring the encoder's ringing rather than this reader.
        let w = 32, h = 16
        var rgba = [UInt8]()
        for y in 0 ..< h {
            for x in 0 ..< w {
                let left = x < w / 2
                rgba.append(left ? 200 : 20)
                rgba.append(y < h / 2 ? 200 : 20)
                rgba.append(60)
                rgba.append(255)
            }
        }
        try Self.write(try Self.makeImage(width: w, height: h, rgba: rgba), to: url,
                       type: .jpeg, properties: [kCGImageDestinationLossyCompressionQuality: 1.0])

        let tensor = try MediaInput.image(at: url)
        #expect(tensor.shape == [1, h, w, 3])
        let values = tensor.asArray(Float.self)
        func pixel(_ x: Int, _ y: Int) -> (Float, Float, Float) {
            let base = (y * w + x) * 3
            return (values[base], values[base + 1], values[base + 2])
        }
        // Sampled well inside each quadrant so the chroma subsample at the seams cannot
        // reach the sample points.
        let topLeft = pixel(w / 4, h / 4)
        let bottomRight = pixel(w * 3 / 4, h * 3 / 4)
        #expect(abs(topLeft.0 - 200.0 / 255) < 0.05, "top-left R \(topLeft.0)")
        #expect(abs(topLeft.1 - 200.0 / 255) < 0.05, "top-left G \(topLeft.1)")
        #expect(abs(bottomRight.0 - 20.0 / 255) < 0.05, "bottom-right R \(bottomRight.0)")
        #expect(abs(bottomRight.1 - 20.0 / 255) < 0.05, "bottom-right G \(bottomRight.1)")
    }

    // MARK: - Alpha

    @Test("an image with a transparent pixel is rejected by default")
    func alphaIsRejected() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("alpha.png")

        // Pixel 0 opaque green, pixel 1 fully transparent. α = 0 rather than α = 128 so the
        // composite assertions below are exact rather than premultiplication-rounded.
        let rgba: [UInt8] = [0, 255, 0, 255, 0, 0, 0, 0]
        try Self.write(try Self.makeImage(width: 2, height: 1, rgba: rgba), to: url,
                       type: .png)

        await Self.expectFailure("a transparent pixel must not be flattened silently") {
            _ = try MediaInput.image(at: url)
        } matches: {
            if case .imageHasAlpha(_, let minimum) = $0 { return minimum < 0.01 }
            return false
        }
    }

    @Test("an opaque image passes the alpha check even with an alpha channel present")
    func opaqueAlphaChannelIsFine() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("opaque.png")

        // The distinction the policy has to make: *having* an alpha channel is not the same
        // as having meaningful alpha, and rejecting every RGBA png would reject most of them.
        let rgba: [UInt8] = [10, 20, 30, 255, 40, 50, 60, 255]
        try Self.write(try Self.makeImage(width: 2, height: 1, rgba: rgba), to: url,
                       type: .png)

        let values = try MediaInput.image(at: url).asArray(Float.self)
        #expect(abs(values[0] - 10.0 / 255) < 1e-6, "\(values)")
        #expect(abs(values[5] - 60.0 / 255) < 1e-6, "\(values)")
    }

    @Test("compositing puts the named background where the transparency was")
    func alphaCompositesOverANamedBackground() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("alpha.png")
        let rgba: [UInt8] = [0, 255, 0, 255, 0, 0, 0, 0]
        try Self.write(try Self.makeImage(width: 2, height: 1, rgba: rgba), to: url,
                       type: .png)

        let overWhite = try MediaInput.image(at: url, alpha: .composite(over: .white))
            .asArray(Float.self)
        #expect(overWhite == [0, 1, 0, 1, 1, 1], "over white: \(overWhite)")

        let overBlack = try MediaInput.image(at: url, alpha: .composite(over: .black))
            .asArray(Float.self)
        #expect(overBlack == [0, 1, 0, 0, 0, 0], "over black: \(overBlack)")
    }

    // MARK: - Orientation

    /// EXIF 6 is "rotate 90° clockwise to display", so a wide file becomes a tall tensor and
    /// the stored top-left pixel lands at the displayed top-right.
    ///
    /// This is the assertion that the transform table is derived and not guessed. The
    /// failure it catches — a portrait photo arriving on its side — is one a human sees
    /// instantly in a render and no shape check ever sees at all.
    @Test("EXIF orientation 6 rotates the image and swaps the dimensions")
    func orientationSixIsApplied() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("rotated.tiff")

        // TIFF, because its orientation tag is native and lossless; a JPEG would put the
        // codec between the fixture and the assertion for no gain.
        let w = 4, h = 2
        func marker(_ x: Int, _ y: Int) -> UInt8 { UInt8(10 + x * 20 + y * 100) }
        var rgba = [UInt8]()
        for y in 0 ..< h {
            for x in 0 ..< w {
                let v = marker(x, y)
                rgba.append(contentsOf: [v, v, v, 255])
            }
        }
        try Self.write(try Self.makeImage(width: w, height: h, rgba: rgba), to: url,
                       type: .tiff, properties: [kCGImagePropertyOrientation: 6])

        // If ImageIO did not keep the tag there is nothing to test, and a green test that
        // silently checked the identity case would be a lie.
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let tag = properties?[kCGImagePropertyOrientation] as? UInt32
        try #require(tag == 6, "ImageIO dropped the orientation tag (got \(tag as Any))")

        let tensor = try MediaInput.image(at: url)
        #expect(tensor.shape == [1, w, h, 3], "shape \(tensor.shape); expected [1, 4, 2, 3]")
        let values = tensor.asArray(Float.self)

        // Stored (col, row) displays at (H_stored - 1 - row, col) under a 90° clockwise
        // rotation: the displayed width is `h` and the displayed height is `w`.
        for y in 0 ..< h {
            for x in 0 ..< w {
                let column = h - 1 - y, row = x
                let got = values[(row * h + column) * 3]
                let want = Float(marker(x, y)) / 255
                #expect(abs(got - want) < 1e-6,
                        "stored (\(x), \(y)) -> displayed (\(column), \(row)) is \(got), expected \(want)")
            }
        }
    }

    @Test("EXIF orientation 3 turns the image through 180 degrees")
    func orientationThreeIsApplied() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("upside-down.tiff")

        let w = 4, h = 2
        func marker(_ x: Int, _ y: Int) -> UInt8 { UInt8(10 + x * 20 + y * 100) }
        var rgba = [UInt8]()
        for y in 0 ..< h {
            for x in 0 ..< w {
                let v = marker(x, y)
                rgba.append(contentsOf: [v, v, v, 255])
            }
        }
        try Self.write(try Self.makeImage(width: w, height: h, rgba: rgba), to: url,
                       type: .tiff, properties: [kCGImagePropertyOrientation: 3])

        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        try #require((properties?[kCGImagePropertyOrientation] as? UInt32) == 3)

        let tensor = try MediaInput.image(at: url)
        #expect(tensor.shape == [1, h, w, 3], "shape \(tensor.shape)")
        let values = tensor.asArray(Float.self)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let column = w - 1 - x, row = h - 1 - y
                let got = values[(row * w + column) * 3]
                let want = Float(marker(x, y)) / 255
                #expect(abs(got - want) < 1e-6,
                        "stored (\(x), \(y)) -> displayed (\(column), \(row)) is \(got), expected \(want)")
            }
        }
    }

    // MARK: - The round trip against RenderOutput

    /// Absolute error between two tensors of the same shape, printed under `label`.
    ///
    /// Max, mean *and* the 99.9th percentile, because on a codec the first two disagree
    /// about what happened: the max is one pixel somewhere and the mean is dominated by the
    /// pixels where nothing went wrong. The percentile is what says whether the max is an
    /// outlier or the shape of the whole distribution — which is exactly the difference
    /// between "the encoder rounded a corner" and "this reader has a bug".
    ///
    /// Both round-trip tests report through here so their numbers are directly comparable;
    /// the whole point of having two is the gap between them.
    static func compare(_ want: MLXArray, _ got: MLXArray,
                        label: String) throws -> (max: Float, mean: Double) {
        let a = want.asArray(Float.self), b = got.asArray(Float.self)
        try #require(a.count == b.count)
        var errors = [Float]()
        errors.reserveCapacity(a.count)
        var total: Double = 0
        for i in 0 ..< a.count {
            let error = abs(a[i] - b[i])
            errors.append(error)
            total += Double(error)
        }
        errors.sort()
        let worst = errors.last ?? 0
        let mean = total / Double(a.count)
        let p999 = errors[min(errors.count - 1, Int(Double(errors.count) * 0.999))]
        print("[MediaInput] \(label) over \(a.count) values: "
            + "max |Δ| = \(worst) (\(worst * 255) / 255), "
            + "p99.9 = \(p999) (\(p999 * 255) / 255), "
            + "mean |Δ| = \(mean) (\(mean * 255) / 255)")
        return (worst, mean)
    }

    /// Write a clip with ``RenderOutput`` and read it back. The tolerance is **measured**.
    ///
    /// A guessed tolerance on a lossy codec is worthless in both directions — too tight and
    /// the suite is flaky, too loose and it asserts nothing. So the error is computed,
    /// printed on every run, and the bound is the measured value with headroom. The numbers
    /// this produced when it was written are in the comments below; if a future
    /// VideoToolbox moves them the printed line says by how much.
    ///
    /// **The fixture is achromatic (R = G = B) and that is the point of this test.** In
    /// 4:2:0 an achromatic frame has neutral chroma, so its round trip exercises the luma
    /// path, the geometry, the row stride and the range handling with the colour conversion
    /// taken out of the picture. Those are the things this reader is responsible for, and
    /// isolating them is what lets the bound here be tight rather than absorbing the very
    /// much larger colour error that ``colourSurvivesWithATaggedMatrix`` measures and
    /// explains. It is also low-frequency in space and time, so 4:2:0's spatial filter has
    /// nothing to blur.
    @Test("an achromatic clip written by RenderOutput reads back as the same pixels")
    func roundTripsAgainstRenderOutput() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("round-trip.mp4")

        let frames = 9, size = 64
        let source = Self.clip(frames: frames, height: size, width: size) { f, y, x, _ in
            let base = (Float(x) / Float(size - 1) + Float(y) / Float(size - 1)
                + Float(f) / Float(frames - 1)) / 3
            // Pulled off the endpoints: 0.0 and 1.0 are exactly where a limited-range
            // conversion clips, and this test is not about that — `RenderOutputTests`
            // owns the saturation claim.
            return 0.1 + 0.8 * base
        }
        try RenderOutput.writeVideo(rgb: source, to: url,
                                    spec: .init(fps: 24, averageBitRate: 20_000_000))

        let read = try await MediaInput.videoFrames(at: url)
        #expect(read.shape == [frames, size, size, 3], "shape \(read.shape)")

        let (worst, mean) = try Self.compare(source, read, label: "H.264 achromatic")
        // Measured on an M-series Mac, macOS 26 / Swift 6.3.3: max 0.0265 (6.8/255),
        // p99.9 0.0078 (2.0/255), mean 0.0018 (0.46/255). The percentile is the number to
        // read — the round trip is within a couple of counts almost everywhere and the max
        // is a handful of pixels where the encoder rounded a corner of the gradient.
        //
        // The bounds are ~2x the measurement, and the headroom costs nothing: the failures
        // this stands guard over — a vertical flip, a shear, a channel swap, an inverted
        // range — move a value by 0.3 to 1.0, not by a few counts.
        #expect(worst < 0.05, "max |Δ| = \(worst); the round trip is not the identity")
        #expect(mean < 0.006, "mean |Δ| = \(mean)")
    }

    /// The same round trip in colour. It is now nearly as good as the achromatic one, and
    /// the story of how it got there is why this test is kept separate.
    ///
    /// When this suite was first written, ``RenderOutput`` did not tag colour: the `.mp4`
    /// carried no `CVImageBufferYCbCrMatrix`, no primaries and no transfer function at all
    /// (checked directly with `CMFormatDescriptionGetExtensions` — every field nil, at
    /// 64×64 and at 320×320 alike). The encoder therefore chose a matrix by its default,
    /// the decoder chose one by its default, and they disagreed. The measurement was
    /// **max 0.135 (34.5/255), mean 0.0253 (6.5/255)**, against 0.0265 and 0.0018 for the
    /// same clip made achromatic — a 14× mean, which is not a codec artefact.
    ///
    /// The arithmetic identified the pair exactly. Pure red came back as
    /// `(0.9176, 0.0, 0.0078)`. Encoding `(1, 0, 0)` with BT.709 gives
    /// `Y = 0.2126, Cb = -0.1146, Cr = 0.5`; decoding that with BT.601 gives
    /// `R = Y + 1.402·Cr = 0.914`, `G = Y - 0.344·Cb - 0.714·Cr = -0.105 → 0`,
    /// `B = Y + 1.772·Cb = 0.010` — the measurement to three decimals. Not a channel swap,
    /// which would have put the 1.0 in B, and not a range error, which would have moved all
    /// three channels together.
    ///
    /// ``RenderOutput`` now sets `AVVideoColorPropertiesKey`, and the numbers below are
    /// **max 0.0417 (10.6/255), mean 0.0029 (0.74/255)**, with pure red returning exactly
    /// `(1.0, 0.0, 0.0)`. The achromatic numbers did not move at all, which is the
    /// corroboration: the tag is a chroma-path change and luma never depended on it.
    ///
    /// The test stays because the failure it caught is invisible from either side alone. A
    /// reader cannot recover a matrix a file does not state, so a writer that stopped
    /// tagging would look correct in `RenderOutputTests` and show up only here, as this
    /// number going back to 0.135.
    @Test("colour survives the round trip now that the writer states its matrix")
    func colourSurvivesWithATaggedMatrix() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("colour.mp4")

        let frames = 9, size = 64
        let source = Self.clip(frames: frames, height: size, width: size) { f, y, x, c in
            let base: Float
            switch c {
            case 0: base = Float(x) / Float(size - 1)
            case 1: base = Float(y) / Float(size - 1)
            default: base = Float(f) / Float(frames - 1)
            }
            return 0.1 + 0.8 * base
        }
        try RenderOutput.writeVideo(rgb: source, to: url,
                                    spec: .init(fps: 24, averageBitRate: 20_000_000))

        let read = try await MediaInput.videoFrames(at: url)
        #expect(read.shape == [frames, size, size, 3], "shape \(read.shape)")

        let (worst, mean) = try Self.compare(source, read, label: "H.264 colour")
        // Measured at 0.0417 max (10.6/255) and 0.0029 mean (0.74/255) with the writer
        // tagging Rec.709. The bounds are ~2x that and are a *ratchet*: an untagged writer
        // put these at 0.135 and 0.0253, which is 3x and 9x the bounds below, so a
        // regression in the tagging fails here loudly rather than degrading quietly.
        #expect(worst < 0.08, "max |Δ| = \(worst); at 0.135 the writer has stopped tagging")
        #expect(mean < 0.008, "mean |Δ| = \(mean); at 0.025 the writer has stopped tagging")
    }

    /// The control for the single most likely bug in the reader.
    ///
    /// 200 × 4 = 800 bytes per row, which is not a multiple of the 64-byte alignment
    /// VideoToolbox hands back, so the decoded buffer is padded and a reader that assumed
    /// `width * 4` would shear the frame by `(stride - 800) / 4` pixels per row. That
    /// produces an image which is the right size, the right length, the right channel order,
    /// and diagonally smeared — and every other assertion in this file passes on it.
    ///
    /// The claim is per-row: the vertical edge must land in the same column on all 120 rows.
    /// An absolute column check would also catch a whole-frame shift, but only the per-row
    /// invariance catches a shear, which is what a stride mistake actually is.
    @Test("the row stride is honoured at a width that is not a multiple of 16")
    func strideIsHonouredAtAnAwkwardWidth() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("stride.mp4")

        let width = 200, height = 120, edge = 100
        let source = Self.clip(frames: 3, height: height, width: width) { _, _, x, _ in
            x < edge ? 0.05 : 0.95
        }
        try RenderOutput.writeVideo(rgb: source, to: url,
                                    spec: .init(fps: 24, averageBitRate: 20_000_000))

        // Said out loud, because if the decoder does not pad on this machine the assertion
        // below is vacuous and the reader deserves to know that rather than a green tick.
        if let probe = try Self.decodedStride(url) {
            print("[MediaInput] decoded \(probe.width)-wide frame has bytesPerRow "
                + "\(probe.bytesPerRow); packed would be \(probe.width * 4)"
                + (probe.bytesPerRow > probe.width * 4
                    ? " — padded, so this test is exercising the stride"
                    : " — NOT padded, so this test is vacuous on this machine"))
        }

        let read = try await MediaInput.videoFrames(at: url, maxFrames: 1)
        #expect(read.shape == [1, height, width, 3], "shape \(read.shape)")
        let values = read.asArray(Float.self)

        var columns = [Int]()
        for y in 0 ..< height {
            var found = -1
            for x in 0 ..< width where values[(y * width + x) * 3] > 0.5 {
                found = x
                break
            }
            columns.append(found)
        }
        let unique = Set(columns)
        #expect(unique.count == 1,
                "the edge is at different columns on different rows, which is the shear ignoring bytesPerRow produces. Columns seen: \(unique.sorted())")
        // ±2 for the codec's own softening of a hard edge; the point of the test is the
        // line above, and this only rules out a uniform whole-frame shift.
        #expect(abs((columns.first ?? -1) - edge) <= 2,
                "the edge landed at column \(columns.first ?? -1), expected \(edge)")
    }

    /// R is not B. A BGRA/RGBA confusion in either direction passes every shape, size,
    /// stride and length check in this file, and shows up only as a render whose blues and
    /// reds are exchanged.
    ///
    /// The bounds are loose deliberately. Pure red currently comes back as exactly
    /// `(1.0, 0.0, 0.0)`, but before ``RenderOutput`` tagged its colour it came back as
    /// `(0.9176, 0, 0.0078)` — see ``colourSurvivesWithATaggedMatrix()`` — and this test
    /// should keep working either way. The gap it has to resolve is 1.0 against 0.0, so
    /// absorbing an 8% matrix deficit costs it nothing, and holding it tighter would be
    /// asserting the colour claim in the place that does not own it.
    @Test("a red clip reads back red and not blue")
    func channelOrderSurvivesTheRoundTrip() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("red.mp4")

        let size = 64
        let source = Self.clip(frames: 4, height: size, width: size) { _, _, _, c in
            c == 0 ? 1 : 0
        }
        try RenderOutput.writeVideo(rgb: source, to: url,
                                    spec: .init(fps: 24, averageBitRate: 20_000_000))

        let read = try await MediaInput.videoFrames(at: url, maxFrames: 1)
        let values = read.asArray(Float.self)
        let centre = ((size / 2) * size + size / 2) * 3
        let (r, g, b) = (values[centre], values[centre + 1], values[centre + 2])
        print("[MediaInput] pure red survived H.264 as (\(r), \(g), \(b))")
        #expect(r > 0.8, "R is \(r); a channel swap would put the 1.0 in B")
        #expect(g < 0.2, "G is \(g)")
        #expect(b < 0.2, "B is \(b); a channel swap would put it near 1.0")
    }

    @Test("maxFrames bounds the read and nil reads the whole clip")
    func maxFramesBoundsTheRead() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("nine.mp4")

        let source = Self.clip(frames: 9, height: 64, width: 64) { f, _, x, _ in
            0.1 + 0.05 * Float(f) + 0.4 * Float(x) / 63
        }
        try RenderOutput.writeVideo(rgb: source, to: url, spec: .init(fps: 24))

        #expect(try await MediaInput.videoFrames(at: url).shape == [9, 64, 64, 3])
        #expect(try await MediaInput.videoFrames(at: url, maxFrames: 3).shape == [3, 64, 64, 3])
        // Asking for more than there are is not an error — it is a bound, not a promise.
        #expect(try await MediaInput.videoFrames(at: url, maxFrames: 100).shape == [9, 64, 64, 3])

        // The bounded read must be the *first* frames and not an arbitrary three: a reader
        // that seeked, or that kept the tail of a ring buffer, would pass the shape check.
        let whole = try await MediaInput.videoFrames(at: url).asArray(Float.self)
        let head = try await MediaInput.videoFrames(at: url, maxFrames: 3).asArray(Float.self)
        for i in stride(from: 0, to: head.count, by: 997) {
            #expect(whole[i] == head[i], "frame \(i / (64 * 64 * 3)) differs at \(i)")
        }
    }

    // MARK: - Audio

    /// fp32 in, fp32 out, so the claim is identity and the tolerance is for the
    /// comparison's own arithmetic rather than for the format.
    @Test("a WAV written by RenderOutput reads back exactly")
    func audioRoundTripsAgainstWriteWAV() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.wav")

        let rate = 48_000, frames = 4800
        let step = 2 * Double.pi * 440 / Double(rate)
        // The right channel is a quarter-cycle behind the left, so the two are never equal
        // and a channel swap cannot hide.
        let source = Self.stereo(frames: frames,
                                 left: { 0.5 * Float(sin(step * Double($0))) },
                                 right: { 0.5 * Float(cos(step * Double($0))) })
        try RenderOutput.writeWAV(audio: source, to: url, sampleRate: rate)

        let (samples, readRate) = try await MediaInput.audio(at: url)
        #expect(readRate == rate, "rate \(readRate)")
        #expect(samples.shape == [1, 2, frames], "shape \(samples.shape)")

        let values = samples.asArray(Float.self)
        for i in [0, 1, 2, 37, 1000, frames - 1] {
            let wantLeft = Float(0.5 * sin(step * Double(i)))
            let wantRight = Float(0.5 * cos(step * Double(i)))
            #expect(values[i] == wantLeft, "L[\(i)] = \(values[i]), expected \(wantLeft)")
            #expect(values[frames + i] == wantRight,
                    "R[\(i)] = \(values[frames + i]), expected \(wantRight)")
        }
    }

    /// Channel-major and not interleaved. The layout `RenderOutput` refuses — `[1, samples, 2]`
    /// — is well-formed and plausible, so the reader has to be checked for producing the
    /// other one, not merely for producing three axes.
    @Test("the returned layout is [1, channels, samples] and the channels are not mixed")
    func audioLayoutIsChannelMajor() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("dc.wav")

        let frames = 480
        let source = Self.stereo(frames: frames, left: { _ in 0.5 }, right: { _ in -0.5 })
        try RenderOutput.writeWAV(audio: source, to: url, sampleRate: 48_000)

        let (samples, _) = try await MediaInput.audio(at: url)
        #expect(samples.shape == [1, 2, frames])
        let values = samples.asArray(Float.self)
        #expect(values[0 ..< frames].allSatisfy { $0 == 0.5 },
                "the first \(frames) floats are not a constant +0.5; an interleaved readback would alternate +0.5, -0.5")
        #expect(values[frames ..< 2 * frames].allSatisfy { $0 == -0.5 },
                "the second half is not a constant -0.5")

        // And the round trip closes: what came out goes back in and is accepted.
        let back = directory.appendingPathComponent("again.wav")
        try RenderOutput.writeWAV(audio: samples, to: back, sampleRate: 48_000)
        let (again, _) = try await MediaInput.audio(at: back)
        #expect(again.asArray(Float.self) == values, "the second round trip drifted")
    }

    @Test("resampling scales the length by the rate ratio")
    func resamplingScalesTheLength() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.wav")

        let rate = 48_000, frames = 4800
        let step = 2 * Double.pi * 440 / Double(rate)
        let source = Self.stereo(frames: frames,
                                 left: { 0.5 * Float(sin(step * Double($0))) },
                                 right: { 0.5 * Float(sin(step * Double($0))) })
        try RenderOutput.writeWAV(audio: source, to: url, sampleRate: rate)

        let halved = try await MediaInput.audio(at: url, sampleRate: 24_000, channels: 2)
        #expect(halved.shape.count == 3 && halved.shape[0] == 1 && halved.shape[1] == 2,
                "shape \(halved.shape)")
        let got = halved.shape[2]
        print("[MediaInput] 48000 -> 24000 turned \(frames) frames into \(got); "
            + "the exact ratio would be \(frames / 2)")
        // Loose on purpose: the converter's filter has a tail and the exact length is its
        // business. A caller that needs an exact count must trim, and the doc comment says so.
        #expect(abs(got - frames / 2) <= 64, "\(got) frames, expected about \(frames / 2)")

        // Length alone would pass on silence. 440 Hz at half the rate is still 440 Hz at
        // half the amplitude nowhere, so the peak has to survive.
        let peak = halved.asArray(Float.self).map { abs($0) }.max() ?? 0
        #expect(peak > 0.4 && peak < 0.6, "peak \(peak); expected the 0.5 tone to survive")

        // Up as well as down, because a converter wired backwards halves both.
        let doubled = try await MediaInput.audio(at: url, sampleRate: 96_000, channels: 2)
        print("[MediaInput] 48000 -> 96000 turned \(frames) frames into \(doubled.shape[2])")
        #expect(abs(doubled.shape[2] - frames * 2) <= 128, "shape \(doubled.shape)")
    }

    @Test("a stereo file can be asked for one channel and comes back mono")
    func downmixIsAvailableAndExplicit() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("dc.wav")

        let frames = 4800
        let source = Self.stereo(frames: frames, left: { _ in 0.5 }, right: { _ in 0.5 })
        try RenderOutput.writeWAV(audio: source, to: url, sampleRate: 48_000)

        let mono = try await MediaInput.audio(at: url, sampleRate: 48_000, channels: 1)
        #expect(mono.shape.count == 3 && mono.shape[0] == 1 && mono.shape[1] == 1,
                "shape \(mono.shape)")
        #expect(abs(mono.shape[2] - frames) <= 64, "shape \(mono.shape)")

        // Both channels are +0.5, so any downmix law — sum, average, -3 dB — leaves a
        // non-zero constant. The assertion is deliberately about the sign and the presence
        // of signal and not about the coefficient, which is Apple's to choose.
        let values = mono.asArray(Float.self)
        let middle = values[values.count / 2]
        #expect(middle > 0.2, "downmix produced \(middle); silence would be 0")
    }

    // MARK: - Negative controls

    @Test("a missing file throws .missingFile from every reader")
    func missingFileThrows() async {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-does-not-exist-" + UUID().uuidString + ".png")

        for (name, body) in [
            ("image", { _ = try MediaInput.image(at: url) }),
            ("videoFrames", { _ = try await MediaInput.videoFrames(at: url) }),
            ("audio", { _ = try await MediaInput.audio(at: url) }),
            ("audio(rate:)", { _ = try await MediaInput.audio(at: url, sampleRate: 48_000,
                                                         channels: 2) }),
        ] as [(String, () async throws -> Void)] {
            await Self.expectFailure("\(name) on a missing path", body) {
                if case .missingFile = $0 { return true }
                return false
            }
        }
    }

    @Test("a directory is refused rather than opened")
    func directoryThrows() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }

        await Self.expectFailure("a directory is not an image") {
            _ = try MediaInput.image(at: directory)
        } matches: {
            if case .notARegularFile(_, let kind) = $0 { return kind == "directory" }
            return false
        }
    }

    /// The extension is a claim, not evidence. ImageIO is asked and it says no.
    @Test("a text file named .png throws instead of decoding to something")
    func textFileNamedPNGThrows() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("liar.png")
        try Data("this is not a png, it is a sentence about one\n".utf8).write(to: url)

        await Self.expectFailure("the bytes are text") {
            _ = try MediaInput.image(at: url)
        } matches: {
            switch $0 {
            case .noDecodableImage, .unreadableImageContainer: return true
            default: return false
            }
        }
    }

    @Test("an audio-only file handed to the video reader throws .noVideoTrack")
    func audioOnlyFileHasNoVideoTrack() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.wav")

        let audio = Self.stereo(frames: 4800, left: { _ in 0.25 }, right: { _ in -0.25 })
        try RenderOutput.writeWAV(audio: audio, to: url, sampleRate: 48_000)

        await Self.expectFailure("a WAV has no video track") {
            _ = try await MediaInput.videoFrames(at: url)
        } matches: {
            if case .noVideoTrack = $0 { return true }
            return false
        }
    }

    /// The mistake this names: handing a silent render — which is what
    /// ``RenderOutput/writeVideo(rgb:to:spec:)`` produces, by design — to the audio reader.
    /// `AVAudioFile` reports it with the same opaque status it uses for corrupt bytes, so
    /// without the container re-ask this would come back as "unreadable".
    @Test("a silent mp4 handed to the audio reader throws .noAudioTrack")
    func silentVideoHasNoAudioTrack() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("silent.mp4")

        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, x, _ in
            0.1 + 0.5 * Float(x) / 63
        }
        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24))

        await Self.expectFailure("writeVideo is silent by design") {
            _ = try await MediaInput.audio(at: url)
        } matches: {
            if case .noAudioTrack = $0 { return true }
            return false
        }
    }

    /// An `.mp4` with a video track declared and no samples in it.
    ///
    /// This was written to exercise ``MediaInput/Failure/noFrames(_:)`` and it does not:
    /// AVAssetWriter finalises the file happily, and AVFoundation then refuses to open it
    /// at all (`-11829 "Cannot Open" … This media may be damaged`), so the read fails one
    /// step earlier at ``MediaInput/Failure/videoReaderSetup(_:detail:)``. The test is kept
    /// under an honest name, because a damaged container is a real thing to hand a reader
    /// and the assertion that matters is unchanged: it must throw, the error must name the
    /// path, and nothing must come back as an empty-but-plausible tensor.
    ///
    /// **`noFrames` therefore has no test.** A container that opens cleanly and yields zero
    /// samples is not something this suite can manufacture — `RenderOutput` refuses to
    /// write one by design, and AVAssetWriter's attempt is the damaged file above. The
    /// guard stays because the alternative to it is an empty tensor reaching the flicker
    /// detector, which measures inter-frame difference and would report a clean result over
    /// no frames.
    @Test("a damaged container throws and names the file rather than returning nothing")
    func damagedContainerThrows() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("empty.mp4")

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 64,
            AVVideoHeightKey: 64,
        ])
        input.expectsMediaDataInRealTime = false
        try #require(writer.canAdd(input))
        writer.add(input)
        try #require(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        input.markAsFinished()
        writer.endSession(atSourceTime: .zero)
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            writer.finishWriting {
                continuation.resume()
            }
        }
        try #require(writer.status == .completed,
                     "could not synthesise the fixture: \(String(describing: writer.error))")

        await Self.expectFailure("a track with no samples is not a clip") {
            _ = try await MediaInput.videoFrames(at: url)
        } matches: {
            switch $0 {
            case let .videoReaderSetup(named, _): return named == url
            case .noFrames, .noVideoTrack: return true
            default: return false
            }
        }
    }

    @Test("maxFrames of zero is refused rather than answered with an empty tensor")
    func zeroFrameLimitThrows() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, x, _ in
            0.1 + 0.5 * Float(x) / 63
        }
        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24))

        await Self.expectFailure("0 is not a bound, it is an empty request") {
            _ = try await MediaInput.videoFrames(at: url, maxFrames: 0)
        } matches: {
            if case .badFrameLimit(let got) = $0 { return got == 0 }
            return false
        }
    }

    @Test("a non-positive target rate or channel count is refused")
    func badConversionTargetsThrow() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.wav")
        let audio = Self.stereo(frames: 480, left: { _ in 0.1 }, right: { _ in -0.1 })
        try RenderOutput.writeWAV(audio: audio, to: url, sampleRate: 48_000)

        await Self.expectFailure("rate 0") {
            _ = try await MediaInput.audio(at: url, sampleRate: 0, channels: 2)
        } matches: {
            if case .badTargetSampleRate(let got) = $0 { return got == 0 }
            return false
        }
        await Self.expectFailure("channels 0") {
            _ = try await MediaInput.audio(at: url, sampleRate: 48_000, channels: 0)
        } matches: {
            if case .badTargetChannelCount(let got) = $0 { return got == 0 }
            return false
        }
    }

    @Test("a video file handed to the audio reader by mistake does not decode as audio")
    func textFileHandedToTheAudioReaderThrows() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("liar.wav")
        try Data("RIFF but not really\n".utf8).write(to: url)

        await Self.expectFailure("the bytes are not audio") {
            _ = try await MediaInput.audio(at: url)
        } matches: {
            switch $0 {
            case .unreadableAudio, .noAudioTrack: return true
            default: return false
            }
        }
    }
}

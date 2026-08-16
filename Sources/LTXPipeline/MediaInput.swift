// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MLX

/// The "and then you have a tensor" step, in the direction ``RenderOutput`` does not go:
/// a `.png`/`.jpg`/`.heic`, an `.mp4`, or a `.wav` on disk → an `MLXArray`.
///
/// Until this existed the port could only *produce* pixels. Every conditioning mode that
/// starts from something the user already has — image-to-video, video-to-video, and the
/// audio conditioning path — needs the opposite direction, and there was no code anywhere
/// in `Sources/` that could open an image or decode a video frame.
///
/// ## This is the inverse of `RenderOutput`, deliberately and exactly
///
/// The layouts are not chosen here; they are the ones the rest of the port already speaks:
///
/// | this | shape | mirrors |
/// |---|---|---|
/// | ``image(at:alpha:)`` | `[1, H, W, 3]`, `[0, 1]` | a one-frame ``VideoVAEDecoder/rgb(_:)`` stack |
/// | ``videoFrames(at:maxFrames:)`` | `[F, H, W, 3]`, `[0, 1]` | ``RenderOutput/writeVideo(rgb:to:spec:)``'s input |
/// | ``audio(at:)`` | `[1, C, samples]` | ``AudioVocoder/waveform(fromMel:)``'s output |
///
/// ``RenderOutput/writeWAV(audio:to:sampleRate:)`` then ``audio(at:)`` is the identity
/// *exactly*, checked at zero tolerance. `writeVideo` then ``videoFrames(at:maxFrames:)``
/// is the identity to within a handful of 8-bit counts — measured, printed on every test
/// run, and reported separately for luma and colour because the colour half depends on the
/// writer declaring its YCbCr matrix and the luma half does not. See
/// ``videoFrames(at:maxFrames:)`` for the numbers and for why that asymmetry is worth
/// keeping visible.
///
/// ## What the suite can and cannot check
///
/// Opening a JPEG has no recorded answer to be compared against. What is checkable is the
/// part that is mechanical — the geometry, the channel order, the row stride, the value
/// range, the alpha and colour policies, and the failure modes — and that is what the
/// suite checks. Whether a conditioning frame read through here *conditions* the way it
/// should is a question for a whole render, not for this file.
///
/// ## Three decisions that are policy rather than mechanism
///
/// **Colour is pinned to sRGB and the conversion is deliberate.** Images are drawn into a
/// bitmap context whose colour space is sRGB, so a Display P3 photo *is* converted on the
/// way in. That is a real change to the numbers and it is the right one: the alternative —
/// reading the `CGImage`'s bytes as they lie — hands the model whatever primaries the
/// camera happened to tag, silently, with no record of it. Untagged sources are treated as
/// sRGB, which is CoreGraphics' own assumption, so an untagged round trip through
/// ``RenderOutput`` (which does not tag colour either) is an identity.
///
/// **Alpha is never silently dropped.** ``AlphaPolicy/reject`` is the default: an image
/// with any non-opaque pixel throws. Compositing is available but must be asked for, and it
/// names the background it composites over, because "RGB from an RGBA png" is not one
/// answer — it is at least three, and picking one for the caller is how a logo arrives with
/// a black halo nobody can account for.
///
/// **Resampling audio is lossy and is a caller-visible choice.** ``audio(at:)`` returns the
/// file's own rate and channel count and does not touch either. The rate/channel conversion
/// lives in a *separate* overload, ``audio(at:sampleRate:channels:)``, so a caller who
/// needs 48 kHz stereo has to say so and knows a resample happened. A single entry point
/// that quietly resampled to whatever the model wanted would make a 44.1 kHz source and a
/// 48 kHz source indistinguishable downstream, which is exactly the distinction someone
/// debugging a pitch or length error needs.
public enum MediaInput {

    // MARK: - Failures

    /// Every case names the file. A media reader that throws "unsupported format" without a
    /// path is useless in a pipeline that opens a conditioning image, a conditioning clip
    /// and a reference audio file in the same call.
    public enum Failure: Error, CustomStringConvertible {
        case missingFile(URL)
        case notARegularFile(URL, kind: String)
        case unreadableImageContainer(URL)
        case noDecodableImage(URL, frames: Int)
        case degenerateImage(URL, width: Int, height: Int)
        case imageBitmapContext(URL, width: Int, height: Int)
        case unsupportedOrientation(URL, value: UInt32)
        case imageHasAlpha(URL, minimum: Float)
        case noVideoTrack(URL)
        case videoReaderSetup(URL, detail: String)
        case badFrameLimit(Int)
        case noFrames(URL)
        case frameNotAnImageBuffer(URL, frame: Int)
        case pixelBufferLock(URL, frame: Int, status: CVReturn)
        case pixelBufferBaseAddress(URL, frame: Int)
        case frameSizeChanged(URL, frame: Int, from: String, to: String)
        case videoReadFailed(URL, underlying: Error?)
        case noAudioTrack(URL)
        case unreadableAudio(URL, underlying: Error)
        case noSamples(URL)
        case audioBuffer(URL, frames: Int, format: String)
        case badTargetSampleRate(Int)
        case badTargetChannelCount(Int)
        case audioConverterSetup(URL, from: String, to: String)
        case audioConversionFailed(URL, underlying: Error?)
        case badResizeTarget(height: Int, width: Int)
        case resizeRank(got: [Int])

        public var description: String {
            switch self {
            case let .missingFile(url):
                return "there is no file at \(url.path)"
            case let .notARegularFile(url, kind):
                return "\(url.path) is a \(kind), not a regular file"
            case let .unreadableImageContainer(url):
                return "ImageIO would not open \(url.path) as an image container at all; "
                    + "the bytes are not a format this system decodes"
            case let .noDecodableImage(url, frames):
                return "\(url.path) is an image container reporting \(frames) image(s), but "
                    + "index 0 did not decode. A text file with a .png extension lands here"
            case let .degenerateImage(url, width, height):
                return "\(url.path) decoded to \(width)x\(height); both dimensions must be "
                    + "positive"
            case let .imageBitmapContext(url, width, height):
                return "could not create a \(width)x\(height) 8-bit sRGB RGBA bitmap "
                    + "context for \(url.path); the image is likely too large to fit in one "
                    + "allocation"
            case let .unsupportedOrientation(url, value):
                return "\(url.path) declares EXIF orientation \(value), which is outside "
                    + "the defined 1...8. Guessing would rotate the frame silently"
            case let .imageHasAlpha(url, minimum):
                return "\(url.path) has a meaningful alpha channel (minimum \(minimum)) and "
                    + "the policy is .reject. RGB from RGBA is not one answer — pass "
                    + ".composite(over:) and name the background, or flatten the file first"
            case let .noVideoTrack(url):
                return "\(url.path) has no video track; an audio-only container reaches "
                    + "here, and so does a still image handed to the video reader"
            case let .videoReaderSetup(url, detail):
                return "AVAssetReader setup failed for \(url.path): \(detail)"
            case let .badFrameLimit(limit):
                return "maxFrames \(limit) is not positive; pass nil to read the whole clip "
                    + "rather than 0, which would be a request for an empty tensor"
            case let .noFrames(url):
                return "\(url.path) has a video track that decoded 0 frames; refusing to "
                    + "return an empty tensor that a conditioning path would treat as a clip"
            case let .frameNotAnImageBuffer(url, frame):
                return "sample \(frame) of \(url.path) carries no CVImageBuffer"
            case let .pixelBufferLock(url, frame, status):
                return "CVPixelBufferLockBaseAddress failed with \(status) on frame "
                    + "\(frame) of \(url.path)"
            case let .pixelBufferBaseAddress(url, frame):
                return "frame \(frame) of \(url.path) has no base address; the decoder "
                    + "returned a planar buffer where 32BGRA was requested"
            case let .frameSizeChanged(url, frame, from, to):
                return "\(url.path) changes frame size at frame \(frame): \(from) -> \(to). "
                    + "A stacked tensor cannot hold that, and silently cropping or scaling "
                    + "to the first frame's size would hide a re-encode nobody asked for"
            case let .videoReadFailed(url, underlying):
                return "the reader stopped before the end of \(url.path)"
                    + (underlying.map { ": \($0)" } ?? " with no error set")
            case let .noAudioTrack(url):
                return "\(url.path) opens as a container but has no audio track; a silent "
                    + "render written by RenderOutput.writeVideo lands here"
            case let .unreadableAudio(url, underlying):
                return "AVAudioFile would not open \(url.path): \(underlying)"
            case let .noSamples(url):
                return "\(url.path) has an audio track of 0 frames; refusing to return "
                    + "silence that would look like a successful read"
            case let .audioBuffer(url, frames, format):
                return "AVAudioPCMBuffer would not allocate \(frames) frames of \(format) "
                    + "for \(url.path)"
            case let .badTargetSampleRate(rate):
                return "target sample rate \(rate) is not positive"
            case let .badTargetChannelCount(channels):
                return "target channel count \(channels) is not positive"
            case let .audioConverterSetup(url, from, to):
                return "AVAudioConverter refused \(from) -> \(to) for \(url.path); the "
                    + "channel counts are probably not a conversion it knows how to make"
            case let .audioConversionFailed(url, underlying):
                return "the resample of \(url.path) failed"
                    + (underlying.map { ": \($0)" } ?? " with no error set")
            case let .badResizeTarget(height, width):
                return "a resize target of \(width)x\(height) is not positive on both axes"
            case let .resizeRank(got):
                return "resizeAndCenterCrop wants [F, H, W, C] with a positive H and W, "
                    + "got \(got)"
            }
        }
    }

    // MARK: - Alpha

    /// A named background, because the whole point of the policy is that the caller says
    /// which one. Stored as the level each of R, G and B is composited against, so
    /// extending this to a grey or a colour is an added case and not a redesign.
    public enum Background: Sendable {
        case black
        case white

        var level: Float {
            switch self {
            case .black: return 0
            case .white: return 1
            }
        }
    }

    /// What to do with an image that has a non-opaque pixel in it.
    ///
    /// ``reject`` is the default and it is the loud one. The failure it prevents is not an
    /// error message anybody would see — it is a conditioning frame in which a transparent
    /// background became black (or white, or the premultiplied colour, depending on which
    /// of the three plausible readings the code happened to implement), conditioning the
    /// model on an edge that was never in the picture.
    public enum AlphaPolicy: Sendable {
        case reject
        case composite(over: Background)
    }

    // MARK: - Images

    /// A still image → `[1, H, W, 3]`, channels-last, RGB, in `[0, 1]`.
    ///
    /// PNG, JPEG and HEIC all arrive through ImageIO, so the set of formats this opens is
    /// whatever the OS decodes; nothing here is per-format.
    ///
    /// **The pixels are never read out of the `CGImage` directly.** They are re-drawn into
    /// a bitmap context this function fully specifies — 8 bits per component, sRGB,
    /// `premultipliedLast` with `byteOrder32Big`, i.e. R, G, B, A in memory order, and a
    /// row stride pinned to `width * 4` so the readback needs no stride arithmetic at all.
    /// That redraw is the entire reason this function is more than four lines. A `CGImage`
    /// off disk can be 16-bit, greyscale, CMYK, BGRA, premultiplied, big-endian, or tagged
    /// Display P3 or Adobe RGB, in any combination; `CGImageGetBytesPerRow` is padded as
    /// often as not; and every one of those differences produces an image that still looks
    /// like an image. Wrong colours, a blue-red swap and a diagonal shear are all silent
    /// bugs downstream of a model that has no idea what it was shown. Handing the
    /// conversion to CoreGraphics with the destination pinned is the only version of this
    /// that is right for a file nobody has inspected first.
    ///
    /// **Colour is converted, not preserved.** See the type comment: a P3 source becomes
    /// sRGB. That is a decision, not an accident.
    ///
    /// **EXIF orientation is applied.** A phone photo is stored landscape with a tag saying
    /// which way is up, and ignoring the tag is how a portrait conditioning image arrives
    /// on its side — a failure that is obvious to a human looking at the render and
    /// invisible to every shape check. The eight orientations are applied as a CTM on the
    /// destination context, so the returned `H` and `W` are the *displayed* dimensions and
    /// are swapped relative to the file for the four rotating cases.
    ///
    /// Multi-image containers (animated GIF, a HEIC burst) read index 0 and nothing else.
    /// Reading a GIF as a clip is ``videoFrames(at:maxFrames:)``'s job, not this one's.
    public static func image(at url: URL,
                             alpha: AlphaPolicy = .reject) throws -> MLXArray {
        try requireRegularFile(url)

        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw Failure.unreadableImageContainer(url)
        }
        let count = CGImageSourceGetCount(source)
        guard let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Failure.noDecodableImage(url, frames: count)
        }

        // The orientation lives in the *source properties* and not on the CGImage: the
        // decoded image is the stored pixels, unrotated, always.
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any]
        let raw = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        guard let orientation = Orientation(rawValue: raw) else {
            throw Failure.unsupportedOrientation(url, value: raw)
        }

        let stored = (width: decoded.width, height: decoded.height)
        guard stored.width > 0, stored.height > 0 else {
            throw Failure.degenerateImage(url, width: stored.width, height: stored.height)
        }
        let size = orientation.displayedSize(width: stored.width, height: stored.height)

        // `premultipliedLast | byteOrder32Big` is R, G, B, A in memory. Premultiplied
        // rather than straight because CoreGraphics does not offer an 8-bit straight-alpha
        // bitmap context at all, and — conveniently — premultiplied is the form the "over"
        // composite below actually wants: `src + bg * (1 - a)` is the Porter-Duff operator
        // written out, with no un-premultiply to undo first and no divide-by-zero at a
        // fully transparent pixel.
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: size.width, height: size.height,
                                      bitsPerComponent: 8, bytesPerRow: size.width * 4,
                                      space: space, bitmapInfo: info) else {
            throw Failure.imageBitmapContext(url, width: size.width, height: size.height)
        }
        // The draw below is 1:1 and the four rotating orientations are axis-aligned, so
        // there is nothing here that *needs* resampling — but the default quality would
        // still let CoreGraphics run a filter over a transformed draw, and a conditioning
        // frame that is imperceptibly softer than the file is a difference nobody would
        // ever attribute to the reader. `.none` makes the values the file's values.
        context.interpolationQuality = .none
        context.concatenate(orientation.transform(width: stored.width,
                                                  height: stored.height))
        context.draw(decoded, in: CGRect(x: 0, y: 0,
                                         width: stored.width, height: stored.height))

        guard let base = context.data else {
            throw Failure.imageBitmapContext(url, width: size.width, height: size.height)
        }
        let packed = [UInt8](UnsafeRawBufferPointer(start: base,
                                                    count: size.width * size.height * 4))

        // One crossing to MLX, already packed — the discipline `RenderOutput.bgraBytes`
        // sets out. Per-pixel indexing of an MLXArray at 4K would be eight million lazy
        // graph evaluations.
        let rgba = MLXArray(packed, [1, size.height, size.width, 4]).asType(.float32) / 255
        let rgb = rgba[0..., 0..., 0..., 0 ..< 3]
        let a = rgba[0..., 0..., 0..., 3 ..< 4]

        switch alpha {
        case .reject:
            let minimum = a.min().item(Float.self)
            guard minimum >= 1 else { throw Failure.imageHasAlpha(url, minimum: minimum) }
            return rgb
        case let .composite(background):
            return rgb + (MLXArray(Float(1)) - a) * background.level
        }
    }

    /// A still image → `[1, height, width, 3]`, resized the way a conditioning image is
    /// meant to be resized.
    ///
    /// **This exists because conditioning is per-stage, not per-render.** A two-stage
    /// render builds its stage-1 conditioning at `height // 2, width // 2` and its stage-2
    /// conditioning at the full `height, width`, each by resizing the source *pixels* and
    /// encoding them separately. It is **not** one latent that gets downscaled.
    /// ``image(at:alpha:)`` decodes 1:1 and cannot serve either stage on its own.
    ///
    /// See ``resizeAndCenterCrop(_:height:width:)`` for exactly which filter this is and
    /// what it approximates.
    ///
    /// **No CRF re-compression is applied, and that is a real divergence.** A conditioning
    /// still is meant to be re-encoded as a one-frame H.264 stream at crf 18 and decoded
    /// back before the resize, so the model sees the compression it was trained on. There
    /// is no CRF-addressable encoder on this platform — AVFoundation exposes bitrate and
    /// quality, not x264's rate factor — so the model gets an uncompressed still. That
    /// makes a conditioning frame here *cleaner*, never differently framed; it is recorded
    /// in the provenance sidecar rather than left to be rediscovered.
    public static func image(at url: URL, height: Int, width: Int,
                             alpha: AlphaPolicy = .reject) throws -> MLXArray {
        try resizeAndCenterCrop(try image(at: url, alpha: alpha),
                                height: height, width: width)
    }

    /// Resize and centre-crop, on a `[F, H, W, C]` channels-last tensor.
    ///
    /// ```
    /// scale        = max(height / srcH, width / srcW)
    /// newH, newW   = ceil(srcH * scale), ceil(srcW * scale)
    /// resized      = bilinear(tensor, (newH, newW), alignCorners: false)
    /// cropTop/Left = (newH - height) / 2, (newW - width) / 2
    /// ```
    ///
    /// Aspect-**fill** then centre-crop: the shorter axis is filled and the overflow is
    /// trimmed, so the framing is preserved and nothing is stretched. Getting this backwards
    /// (fit-then-pad, or a plain non-uniform resize) changes what the model is conditioned on
    /// without changing any shape.
    ///
    /// ## What is matched exactly, and what is approximated
    ///
    /// **Matched, read from the source rather than assumed:**
    ///
    /// * the filter is **bilinear with `align_corners=False`** and `antialias` left at its
    ///   default of `False` — so a downscale is a two-tap sample, *not* an area average.
    ///   This is the detail most likely to be got wrong by reaching for a system resampler:
    ///   `CGContext`'s `.high` interpolation is an antialiased Lanczos-family filter, and
    ///   `vImageScale` is a box/Lanczos hybrid. Neither is this operator, and both produce a
    ///   perfectly plausible image.
    /// * the source-index map that `align_corners=False` implies,
    ///   `src = (dst + 0.5) * (in / out) - 0.5`, clamped at 0, with the upper neighbour
    ///   pinned at `in - 1`;
    /// * the **order of association**: the two width taps combine first and the two height
    ///   taps second, so this resizes width fully and then height, rather than the other
    ///   way round;
    /// * `ceil` on the intermediate size and floor-division on the crop offsets.
    ///
    /// **Approximated, and stated so rather than glossed:**
    ///
    /// * the value range. The operator is defined over float32 pixels in `[0, 255]`,
    ///   normalised with `x / 127.5 - 1` afterwards; the readers here hand over `[0, 1]`
    ///   already, so the single division by 255 lands *before* the convex combination
    ///   instead of after. Bilinear interpolation is linear, so the two agree algebraically
    ///   and differ only in where one rounding happens.
    /// * the interpolation weights are computed in `Double` and rounded to `Float`, rather
    ///   than in `Float` throughout. Sub-ulp, and named here because "close enough" is how
    ///   a filter mismatch gets missed.
    /// * nothing records what a resizer "should" have produced, so the agreement above is
    ///   argued from the definition rather than measured.
    ///
    /// dtype is the input's — fp32 in, fp32 out.
    public static func resizeAndCenterCrop(_ frames: MLXArray, height: Int,
                                           width: Int) throws -> MLXArray {
        guard height > 0, width > 0 else {
            throw Failure.badResizeTarget(height: height, width: width)
        }
        guard frames.ndim == 4 else {
            throw Failure.resizeRank(got: frames.shape)
        }
        let srcH = frames.dim(1), srcW = frames.dim(2)
        guard srcH > 0, srcW > 0 else {
            throw Failure.resizeRank(got: frames.shape)
        }

        // `max`, not `min`: fill the target and crop the overflow.
        let scale = Swift.max(Double(height) / Double(srcH), Double(width) / Double(srcW))
        // `ceil`, not `round`: floating-point rounding can otherwise leave the intermediate
        // a pixel short of the target, which is a negative crop offset rather than a
        // visible error.
        let newH = Int(Foundation.ceil(Double(srcH) * scale))
        let newW = Int(Foundation.ceil(Double(srcW) * scale))

        var out = frames
        if newW != srcW { out = bilinearAlongAxis(out, axis: 2, outSize: newW) }
        if newH != srcH { out = bilinearAlongAxis(out, axis: 1, outSize: newH) }

        let top = (newH - height) / 2
        let left = (newW - width) / 2
        return out[0..., top ..< (top + height), left ..< (left + width), 0...]
    }

    /// One axis of a bilinear resample, `alignCorners: false`, no antialiasing.
    ///
    /// Bilinear is separable, so the 2-D kernel is this applied twice — but *only* if the
    /// two passes run in the kernel's own order, which is why the caller does width first.
    ///
    /// The index and weight tables are built in Swift rather than in MLX on purpose: they
    /// are `outSize` values of pure integer-and-double arithmetic that have to reproduce
    /// the source-index map exactly, and doing them in tensor ops would hide a clamp behind
    /// a broadcast.
    static func bilinearAlongAxis(_ x: MLXArray, axis: Int, outSize: Int) -> MLXArray {
        let inSize = x.dim(axis)
        // `align_corners=False` with no explicit scale: the scale is `in / out`, and the
        // half-pixel shift on both sides is what makes it "false" rather than "true".
        let scale = Double(inSize) / Double(outSize)

        var lower = [Int32](), upper = [Int32](), lambda = [Float]()
        lower.reserveCapacity(outSize)
        upper.reserveCapacity(outSize)
        lambda.reserveCapacity(outSize)
        for destination in 0 ..< outSize {
            var source = scale * (Double(destination) + 0.5) - 0.5
            // The clamp belongs to the kernel and is not a defensive addition: a
            // downscale's first output pixel maps below zero, and the definition pins it to
            // the first input pixel with weight 1 rather than extrapolating.
            if source < 0 { source = 0 }
            let floor = Foundation.floor(source)
            let index = Int(floor)
            lower.append(Int32(index))
            upper.append(Int32(index < inSize - 1 ? index + 1 : index))
            lambda.append(Float(source - floor))
        }

        var broadcast = [Int](repeating: 1, count: x.ndim)
        broadcast[axis] = outSize
        let weight = MLXArray(lambda, broadcast)
        let a = MLX.take(x, MLXArray(lower), axis: axis)
        let b = MLX.take(x, MLXArray(upper), axis: axis)
        // `(1 - lambda) * a + lambda * b`, spelled the way the kernel spells it. The
        // algebraically equal `a + lambda * (b - a)` is a different rounding.
        return (MLXArray(Float(1)) - weight) * a + weight * b
    }

    /// The eight EXIF orientations, as a destination transform.
    ///
    /// Derived rather than copied, and worth stating so the next reader can check it: in a
    /// `CGBitmapContext` an untransformed `draw(image, in: CGRect(0, 0, w, h))` already
    /// lands the image's top-left pixel at the buffer's first row, so image pixel
    /// `(col, row)` maps to user-space point `(col + 0.5, h - row - 0.5)`. Each case below
    /// is the affine map that takes that point to where the *displayed* pixel belongs, with
    /// the displayed position read straight off the EXIF definition ("row 0 = right,
    /// col 0 = top" for 6, and so on). `MediaInputTests` checks two of them against a
    /// tagged file rather than against this comment.
    enum Orientation: UInt32 {
        case up = 1
        case upMirrored = 2
        case down = 3
        case downMirrored = 4
        case leftMirrored = 5
        case right = 6
        case rightMirrored = 7
        case left = 8

        /// True for the four cases where the stored rows become displayed columns.
        var transposes: Bool {
            switch self {
            case .leftMirrored, .right, .rightMirrored, .left: return true
            default: return false
            }
        }

        func displayedSize(width w: Int, height h: Int) -> (width: Int, height: Int) {
            transposes ? (width: h, height: w) : (width: w, height: h)
        }

        func transform(width: Int, height: Int) -> CGAffineTransform {
            let w = CGFloat(width), h = CGFloat(height)
            switch self {
            case .up:
                return .identity
            case .upMirrored:                                   // (x, y) -> (w - x, y)
                return CGAffineTransform(a: -1, b: 0, c: 0, d: 1, tx: w, ty: 0)
            case .down:                                         // (x, y) -> (w - x, h - y)
                return CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: w, ty: h)
            case .downMirrored:                                 // (x, y) -> (x, h - y)
                return CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: h)
            case .leftMirrored:                                 // (x, y) -> (h - y, w - x)
                return CGAffineTransform(a: 0, b: -1, c: -1, d: 0, tx: h, ty: w)
            case .right:                                        // (x, y) -> (y, w - x)
                return CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: w)
            case .rightMirrored:                                // (x, y) -> (y, x)
                return CGAffineTransform(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0)
            case .left:                                         // (x, y) -> (h - y, x)
                return CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: h, ty: 0)
            }
        }
    }

    // MARK: - Video

    /// Video frames → `[F, H, W, 3]`, channels-last, RGB, in `[0, 1]`.
    ///
    /// The exact tensor ``RenderOutput/writeVideo(rgb:to:spec:)`` takes, and the exact
    /// tensor ``VideoVAEDecoder/rgb(_:)`` produces. `maxFrames` bounds the read — the
    /// reader is cancelled once it is reached, so a `maxFrames: 1` on a long clip does not
    /// decode the clip — and `nil` reads everything.
    ///
    /// **`kCVPixelFormatType_32BGRA` is requested explicitly**, so VideoToolbox performs
    /// the YCbCr→RGB conversion, including the limited↔full range expansion. Asking for the
    /// native 4:2:0 buffers instead would mean hand-rolling that conversion, with a
    /// matrix and a range that vary per file and are wrong by default.
    ///
    /// **The row stride is honoured and this is the most likely bug in this file.**
    /// `CVPixelBufferGetBytesPerRow` is routinely larger than `width * 4` — the decoder
    /// aligns rows for its own reasons, 64 bytes being typical — and a flat `memcpy` of
    /// `width * height * 4` from a padded buffer produces a progressively sheared image
    /// that is still recognisably an image, still the right size, still the right channel
    /// order, and completely wrong. `MediaInputTests.strideIsHonouredAtAnAwkwardWidth` is
    /// the control: it reads a 200-wide clip with a hard vertical edge and asserts the edge
    /// sits in the same column on every row.
    ///
    /// **A frame-size change mid-stream throws** rather than being cropped or scaled to the
    /// first frame. A stacked tensor cannot express it, and a reader that quietly resized
    /// would erase the only evidence that a file has been concatenated or re-encoded.
    /// This case is implemented and *not verified by a test* — synthesising a single track
    /// whose sample sizes change is not something AVAssetWriter will do.
    ///
    /// **No colour management is applied on this path, and the round trip depends on the
    /// writer having stated its matrix.** The decoded buffers are used as VideoToolbox
    /// produces them; the YCbCr→RGB conversion uses whatever matrix the file declares.
    /// That dependency was measured rather than assumed. Before ``RenderOutput`` set
    /// `AVVideoColorPropertiesKey`, its files carried no `CVImageBufferYCbCrMatrix`, no
    /// primaries and no transfer function, so the encoder chose a matrix by its default and
    /// the decoder chose one by its default and they disagreed — a pure red pixel came back
    /// as `(0.9176, 0, 0.0078)`, which is BT.709 in and BT.601 out to three decimals, and a
    /// colour gradient came back with a mean error of 6.5/255 against 0.46/255 for the same
    /// clip in luma only. With Rec.709 declared, pure red returns exactly `(1, 0, 0)` and
    /// the gradient's mean error is 0.74/255.
    ///
    /// A reader cannot recover a matrix a file never states, so that failure is not fixable
    /// here and this is the one property of the round trip that depends on the other side.
    /// `MediaInputTests` measures the luma and colour errors separately on every run and
    /// holds the colour one as a ratchet, so a writer that stopped tagging fails there.
    public static func videoFrames(at url: URL, maxFrames: Int? = nil) async throws -> MLXArray {
        if let maxFrames { guard maxFrames > 0 else { throw Failure.badFrameLimit(maxFrames) } }
        try requireRegularFile(url)

        let clip = try await decodeBGRA(url, maxFrames: maxFrames)
        guard clip.frames > 0 else { throw Failure.noFrames(url) }

        // One crossing, already packed, for the reason `RenderOutput.bgraBytes` gives in
        // the other direction. The channel reversal is the same `concatenated` trick it
        // uses, run backwards: B, G, R, A in memory → R, G, B, alpha discarded.
        let bgra = MLXArray(clip.packed, [clip.frames, clip.height, clip.width, 4])
        let r = bgra[0..., 0..., 0..., 2 ..< 3]
        let g = bgra[0..., 0..., 0..., 1 ..< 2]
        let b = bgra[0..., 0..., 0..., 0 ..< 1]
        return MLX.concatenated([r, g, b], axis: -1).asType(.float32) / 255
    }

    /// The video track's nominal frame rate.
    ///
    /// Needed by the training cache, and not derivable from the frames themselves. The rate
    /// is load-bearing rather than cosmetic: `DiTForward.Geometry.frameRate` drives the RoPE
    /// time axis, and the audio positions are in wall-clock **seconds**, so a clip cached at
    /// an assumed 24 fps when it is really 30 is positioned wrong along time — with no shape
    /// error anywhere to say so.
    ///
    /// `nominalFrameRate` is a `Float` and reports 23.976 as 23.976024. Callers that need the
    /// exact rational rate should read `minFrameDuration` instead; for the position grid this
    /// is well inside the resolution that matters.
    public static func frameRate(at url: URL) async throws -> Double {
        try requireRegularFile(url)
        let asset = AVURLAsset(url: url)
        let tracks: [AVAssetTrack]
        do {
            tracks = try await asset.loadTracks(withMediaType: .video)
        } catch {
            throw Failure.videoReaderSetup(url, detail: "loadTracks failed: \(error)")
        }
        guard let track = tracks.first else { throw Failure.noVideoTrack(url) }
        let rate = try await track.load(.nominalFrameRate)
        guard rate > 0, rate.isFinite else {
            throw Failure.videoReaderSetup(url, detail: "nominalFrameRate is \(rate)")
        }
        return Double(rate)
    }

    /// Packed BGRA for a whole clip, and the geometry it was packed at. `Sendable` because
    /// it is what crosses back out of the detached task in ``blocking(_:)`` — an
    /// `AVAssetTrack` is not `Sendable` and must not, so the entire decode happens inside.
    struct RawClip: Sendable {
        let frames: Int
        let width: Int
        let height: Int
        /// `frames * height * width * 4`, no padding: the source stride is removed here so
        /// nothing downstream has to know about it.
        let packed: [UInt8]
    }

    private static func decodeBGRA(_ url: URL, maxFrames: Int?) async throws -> RawClip {
        let asset = AVURLAsset(url: url)
        let tracks: [AVAssetTrack]
        do {
            tracks = try await asset.loadTracks(withMediaType: .video)
        } catch {
            throw Failure.videoReaderSetup(url, detail: "loadTracks failed: \(error)")
        }
        guard let track = tracks.first else { throw Failure.noVideoTrack(url) }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw Failure.videoReaderSetup(url, detail: "\(error)")
        }
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
        // The bytes are copied out row by row below and the buffer is released before the
        // next `copyNextSampleBuffer`, so the reader is free to recycle it.
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw Failure.videoReaderSetup(url, detail: "canAdd(32BGRA output) is false")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw Failure.videoReaderSetup(
                url, detail: "startReading returned false: "
                    + (reader.error.map { "\($0)" } ?? "no error set"))
        }

        var packed = [UInt8]()
        var frames = 0
        var width = 0, height = 0

        while let sample = output.copyNextSampleBuffer() {
            guard let image = CMSampleBufferGetImageBuffer(sample) else {
                reader.cancelReading()
                throw Failure.frameNotAnImageBuffer(url, frame: frames)
            }
            let w = CVPixelBufferGetWidth(image)
            let h = CVPixelBufferGetHeight(image)
            if frames == 0 {
                width = w
                height = h
                if let maxFrames {
                    packed.reserveCapacity(maxFrames * h * w * 4)
                }
            } else if w != width || h != height {
                reader.cancelReading()
                throw Failure.frameSizeChanged(url, frame: frames,
                                               from: "\(width)x\(height)", to: "\(w)x\(h)")
            }

            let lock = CVPixelBufferLockBaseAddress(image, .readOnly)
            guard lock == kCVReturnSuccess else {
                reader.cancelReading()
                throw Failure.pixelBufferLock(url, frame: frames, status: lock)
            }
            guard let base = CVPixelBufferGetBaseAddress(image) else {
                CVPixelBufferUnlockBaseAddress(image, .readOnly)
                reader.cancelReading()
                throw Failure.pixelBufferBaseAddress(url, frame: frames)
            }
            let sourceStride = CVPixelBufferGetBytesPerRow(image)
            let rowBytes = width * 4
            var frame = [UInt8](repeating: 0, count: rowBytes * height)
            frame.withUnsafeMutableBytes { destination in
                guard let start = destination.baseAddress else { return }
                for row in 0 ..< height {
                    // The whole reason this is a loop: `sourceStride` is the decoder's, not
                    // ours, and it is usually padded past `rowBytes`.
                    memcpy(start.advanced(by: row * rowBytes),
                           base.advanced(by: row * sourceStride), rowBytes)
                }
            }
            CVPixelBufferUnlockBaseAddress(image, .readOnly)

            packed.append(contentsOf: frame)
            frames += 1
            if let maxFrames, frames >= maxFrames { break }
        }

        // `.reading` means the loop left early on `maxFrames`, which is not a failure.
        // `.failed` is: a truncated file yields some frames and then stops, and returning
        // the prefix as though it were the clip is the quiet version of that bug.
        if reader.status == .failed {
            throw Failure.videoReadFailed(url, underlying: reader.error)
        }
        reader.cancelReading()

        return RawClip(frames: frames, width: width, height: height, packed: packed)
    }

    // MARK: - Audio

    /// Audio → `[1, channels, samples]` fp32, plus the file's own sample rate.
    ///
    /// Channel-major, which is ``AudioVocoder/waveform(fromMel:)``'s layout and therefore
    /// ``RenderOutput/write(rgb:audio:to:spec:audioSpec:)``'s input. `[1, samples, C]` is
    /// the transposed layout that is well-formed, plausible and wrong; `RenderOutput`'s own
    /// channel check exists because of it, and this reader emits the layout that check
    /// wants.
    ///
    /// **Nothing is resampled and nothing is downmixed.** The returned rate is the file's
    /// and the channel count is the file's — a mono 44.1 kHz file comes back as
    /// `[1, 1, samples]` at 44100, which ``RenderOutput`` will then refuse, correctly,
    /// because it is stereo-only. Silently making it fit is what
    /// ``audio(at:sampleRate:channels:)`` is for, and it is a separate call so that the
    /// conversion is in the caller's code where they can see it.
    ///
    /// The values are whatever is in the file, unclamped. `AVAudioFile`'s reading
    /// `processingFormat` is deinterleaved fp32, so a float WAV written by
    /// ``RenderOutput/writeWAV(audio:to:sampleRate:)`` round-trips bit-exactly; a 16-bit
    /// source is converted to float by AVFoundation and lands on the `k / 32768` lattice.
    public static func audio(at url: URL) async throws -> (samples: MLXArray, sampleRate: Int) {
        let (buffer, file) = try await openAudio(url)
        let channels = Int(buffer.format.channelCount)
        let frames = Int(buffer.frameLength)
        guard let data = buffer.floatChannelData else {
            throw Failure.audioBuffer(url, frames: frames, format: "\(buffer.format)")
        }

        // Deinterleaved on the way in, so each channel is already contiguous and the
        // channel-major tensor is a concatenation rather than a transpose. One crossing.
        var flat = [Float]()
        flat.reserveCapacity(channels * frames)
        for channel in 0 ..< channels {
            flat.append(contentsOf: UnsafeBufferPointer(start: data[channel], count: frames))
        }
        return (MLXArray(flat, [1, channels, frames]),
                Int(file.fileFormat.sampleRate.rounded()))
    }

    /// Audio → `[1, channels, samples]` fp32 at a **requested** rate and channel count.
    ///
    /// **This is lossy and it is the caller's choice, which is why it is a separate
    /// function.** A sample-rate conversion is an interpolation: it invents samples that
    /// were never recorded, it is not invertible, and running it twice is not the same as
    /// running it once. A channel-count change is worse than lossy in the downmix
    /// direction — stereo to mono discards the difference signal permanently. Neither is a
    /// fixup this reader is willing to apply behind a caller's back, so ``audio(at:)``
    /// never touches either and this overload makes both explicit at the call site.
    ///
    /// The conversion is `AVAudioConverter`'s, not a hand-rolled one. Its filter design and
    /// its downmix coefficients are Apple's and are the same ones the rest of the system
    /// uses; a resampler written here would be a second opinion that agrees with nothing.
    ///
    /// The output length is `round(inputFrames * targetRate / sourceRate)` give or take the
    /// converter's own edge handling — checked, loosely, in the suite. It is not exact and
    /// should not be relied on to be: a caller that needs an exact sample count must trim
    /// or pad and know that it did.
    public static func audio(at url: URL, sampleRate: Int, channels: Int) async throws -> MLXArray {
        guard sampleRate > 0 else { throw Failure.badTargetSampleRate(sampleRate) }
        guard channels > 0 else { throw Failure.badTargetChannelCount(channels) }

        let (input, _) = try await openAudio(url)
        guard let target = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate),
                                         channels: AVAudioChannelCount(channels)) else {
            throw Failure.audioConverterSetup(url, from: "\(input.format)",
                                              to: "\(sampleRate) Hz, \(channels) ch")
        }
        // A no-op conversion is still worth routing through the converter rather than
        // short-circuiting: the short circuit is a second code path that only runs when the
        // rates happen to match, i.e. the one path a test on a 48 kHz fixture would take
        // and the production path never would.
        guard let converter = AVAudioConverter(from: input.format, to: target) else {
            throw Failure.audioConverterSetup(url, from: "\(input.format)", to: "\(target)")
        }

        // Capacity with slack: the converter's anti-aliasing filter has a tail, so the
        // exact output length is its business and not something to predict tightly. Too
        // small a buffer silently truncates — `convert` fills what it can and returns.
        let ratio = Double(sampleRate) / input.format.sampleRate
        let capacity = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded()
            + Double(sampleRate) * 0.1 + 4096)
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw Failure.audioBuffer(url, frames: Int(capacity), format: "\(target)")
        }

        // `AVAudioConverterInputBlock` is typed `@Sendable`, but it is invoked
        // synchronously on this thread from inside `convert` — it is a pull callback, not a
        // queued one. The state it needs is therefore boxed rather than captured, which
        // satisfies the type without pretending the buffer crosses a thread.
        let source = Source(input)
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, outStatus in
            if source.exhausted {
                // `.endOfStream` and not `.noDataNow`: the latter makes `convert` return
                // `.inputRanDry` with the filter's tail still inside it, which is a clip
                // that is a few milliseconds short for no visible reason.
                outStatus.pointee = .endOfStream
                return nil
            }
            source.exhausted = true
            outStatus.pointee = .haveData
            return source.buffer
        }
        guard status == .endOfStream || status == .haveData else {
            throw Failure.audioConversionFailed(url, underlying: error)
        }

        let frames = Int(output.frameLength)
        guard frames > 0 else { throw Failure.noSamples(url) }
        guard let data = output.floatChannelData else {
            throw Failure.audioBuffer(url, frames: frames, format: "\(target)")
        }
        var flat = [Float]()
        flat.reserveCapacity(channels * frames)
        for channel in 0 ..< channels {
            flat.append(contentsOf: UnsafeBufferPointer(start: data[channel], count: frames))
        }
        return MLXArray(flat, [1, channels, frames])
    }

    /// Open a file and read the whole thing into one deinterleaved fp32 buffer.
    ///
    /// The `noAudioTrack` distinction is worth the extra work: `AVAudioFile` throws the
    /// same opaque `OSStatus` for "this mp4 has no audio track" as it does for "these bytes
    /// are not audio", and the first is a routine mistake — handing a silent render from
    /// ``RenderOutput/writeVideo(rgb:to:spec:)`` to the audio reader — while the second is
    /// a corrupt file. So a failed open is re-asked as a container question before the
    /// error is chosen.
    private static func openAudio(_ url: URL) async throws -> (AVAudioPCMBuffer, AVAudioFile) {
        try requireRegularFile(url)

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            let audioTracks = (try? await audioTrackCount(url)) ?? -1
            if audioTracks == 0 { throw Failure.noAudioTrack(url) }
            throw Failure.unreadableAudio(url, underlying: error)
        }

        let frames = file.length
        guard frames > 0 else { throw Failure.noSamples(url) }
        // `processingFormat` is deinterleaved fp32 whatever is on disk, which is what makes
        // the channel-major result below a concatenation of contiguous runs rather than a
        // stride walk over an interleaved block — the layout mistake `RenderOutput`'s
        // `plan(forAudio:spec:)` warns about, in reverse.
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(frames)) else {
            throw Failure.audioBuffer(url, frames: Int(frames), format: "\(format)")
        }
        do {
            try file.read(into: buffer)
        } catch {
            throw Failure.unreadableAudio(url, underlying: error)
        }
        guard buffer.frameLength > 0 else { throw Failure.noSamples(url) }
        return (buffer, file)
    }

    private static func audioTrackCount(_ url: URL) async throws -> Int {
        try await AVURLAsset(url: url).loadTracks(withMediaType: .audio).count
    }

    // MARK: - Plumbing

    /// The check is here rather than left to each framework because ImageIO, AVAssetReader
    /// and AVAudioFile each report a missing path as some flavour of unhelpful `OSStatus`,
    /// and a pipeline that opens a conditioning image, a conditioning clip and a reference
    /// waveform in one call needs to know *which* path was wrong.
    ///
    /// `attributesOfItem` is `lstat`, so a symlink reports as a symlink rather than as its
    /// target; symlinks are allowed through — pointing at media from elsewhere is what they
    /// are for — and a dangling one falls out of the frameworks below. What is refused is a
    /// directory, a socket or a device node, which are the arguments that come from a
    /// mistyped path rather than from a real file.
    private static func requireRegularFile(_ url: URL) throws {
        guard let attributes = try? FileManager.default
            .attributesOfItem(atPath: url.path) else {
            throw Failure.missingFile(url)
        }
        let type = (attributes[.type] as? FileAttributeType) ?? .typeUnknown
        guard type == .typeRegular || type == .typeSymbolicLink else {
            throw Failure.notARegularFile(url, kind: describe(type))
        }
    }

    private static func describe(_ type: FileAttributeType) -> String {
        switch type {
        case .typeDirectory: return "directory"
        case .typeSocket: return "socket"
        case .typeCharacterSpecial: return "character device"
        case .typeBlockSpecial: return "block device"
        default: return "non-regular file"
        }
    }

    /// The one-shot input to ``audio(at:sampleRate:channels:)``'s converter callback.
    /// `@unchecked Sendable` because `AVAudioPCMBuffer` is not `Sendable` and the callback
    /// is nominally one — see the call site for why nothing actually crosses a thread.
    private final class Source: @unchecked Sendable {
        let buffer: AVAudioPCMBuffer
        var exhausted = false
        init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
    }
}

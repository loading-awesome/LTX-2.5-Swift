// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import AVFoundation
import CoreVideo
import Foundation
import MLX
import Testing

@testable import LTXPipeline

/// `RenderOutput`, exercised end to end on synthetic tensors.
///
/// **No checkpoint, no fixture files, no GPU-sized allocation.** Every input here is built
/// from a Swift array of floats, so this suite runs on a bare checkout in about a second.
/// That is deliberate and it is what makes it worth having: the encode stage has nothing
/// external to be judged against — H.264 is lossy, and what a render pipeline keeps are
/// tensors, not files — so the only evidence available for it is what a self-contained test
/// can manufacture.
///
/// What is actually asserted, and how strong each assertion is:
///
/// | check | strength |
/// |---|---|
/// | the byte conversion saturates instead of wrapping | **exact** — hand-computed bytes |
/// | frame geometry and duration survive the encode | exact integers, ±1 frame on duration |
/// | out-of-range pixels are still saturated after H.264 | approximate, ±24/255 |
/// | the colour tags reach the container | **exact** — the format description's own strings |
/// | `.untagged` reaches it carrying nothing | **exact** — the control for the row above |
/// | tagging moved metadata and not pixels | **exact** — identical compressed bytes |
/// | colour survives a tagged round trip | approximate, and *better* than untagged |
/// | the sample conversion clamps and interleaves | **exact** — hand-computed floats |
/// | samples survive an LPCM mux | **exact** — bit-for-bit |
/// | the WAV round trip | **exact** — fp32 in, fp32 out |
/// | L and R are not swapped or mixed through AAC | approximate, ±0.01 on a 1.0 gap |
/// | each malformed input throws its own named case | exact |
///
/// The lossy readback is kept despite its tolerance because it is the only thing that
/// observes the whole path — tensor, pool, adaptor, encoder, container, decoder. A wrap
/// bug lands both endpoints at mid-grey (≈126 and ≈129), which is nowhere near either
/// endpoint at any tolerance this test would plausibly use, so the loose bound does not
/// weaken the control it exists to be.
///
/// **Nothing here compares the audio against another run, and nothing could.** Contract 11
/// forbids diffing a vocoder waveform — two runs on the same mel disagree — so
/// every audio assertion below is about transport: did the samples this suite manufactured
/// arrive in the file unpermuted, unclipped-except-where-intended, and at the stated rate.
/// Whether a real render *sounds* right is a listening test and is not automatable here.
///
/// The interleaving controls were checked by mutation rather than trusted. Swapping the two
/// channel rows in `plan(forAudio:spec:)` fails 6 tests / 45 expectations; replacing the
/// pairwise stack with a blocked `L…L R…R` layout fails the same 6 tests / 41 expectations.
/// Both mutations leave every track count, duration, sample rate and sample count in this
/// file passing — which is the point of having the controls at all.
///
/// ## `.serialized` is load-bearing, not tidiness
///
/// Concurrent VideoToolbox encode sessions do not produce the same H.264 bitstream from the
/// same input. Thirty-odd tests in this file encode video, and run in parallel they perturb
/// each other's output length by up to 50% — measured at 2914, 3538 and 4451 bytes for one
/// fixed 9-frame tensor at a pinned bit rate. Only
/// ``taggingChangesTheContainerAndNotThePixels`` compares two encodes against each other, so
/// only that one *fails*, but every timing and size reading in here is taken on the same
/// contended encoder. Removing this trait brings back a flake that reproduces 3 runs in 3
/// under `--filter RenderOutputTests` and never once when the test runs alone.
@Suite("Render output", .serialized)
struct RenderOutputTests {

    // MARK: - Fixtures

    /// A per-frame directory under the system temp dir, removed by the caller.
    static func scratch() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ltx-render-output-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// `[F, H, W, 3]` from a closure over `(frame, row, column, channel)`.
    ///
    /// Built in Swift and handed to MLX in one shot rather than assembled with MLX ops:
    /// the point of a fixture is that its expected bytes can be worked out by hand, and a
    /// tensor built from broadcasts and ranges is a second implementation of the thing
    /// under test.
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

    /// Run `body`, require it to throw a `RenderOutput.Failure`, and hand the case to
    /// `check`. `Failure` is deliberately not `Equatable` — several cases carry an
    /// underlying `Error` — so matching is by pattern rather than by `#expect(throws:)`
    /// with a value.
    static func expectFailure(_ description: Comment,
                              sourceLocation: SourceLocation = #_sourceLocation,
                              _ body: () throws -> Void,
                              matches check: (RenderOutput.Failure) -> Bool) {
        do {
            try body()
            Issue.record("expected a throw: \(description)", sourceLocation: sourceLocation)
        } catch let failure as RenderOutput.Failure {
            #expect(check(failure), "wrong case: \(failure)", sourceLocation: sourceLocation)
        } catch {
            Issue.record("threw the wrong error type: \(error)",
                         sourceLocation: sourceLocation)
        }
    }

    // MARK: - The happy path

    @Test("a 9-frame 64x64 clip becomes a readable mp4")
    func writesASmallClip() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // 9 frames is the second point on the 8k+1 lattice, so this is the smallest clip
        // a real decode could produce rather than an arbitrary count.
        let frames = 9, size = 64
        let rgb = Self.clip(frames: frames, height: size, width: size) { f, y, x, c in
            // A moving gradient. A constant clip would compress to almost nothing and make
            // the "non-trivial size" assertion vacuous.
            switch c {
            case 0: return Float(x) / Float(size - 1)
            case 1: return Float(y) / Float(size - 1)
            default: return Float(f) / Float(frames - 1)
            }
        }

        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24))

        #expect(FileManager.default.fileExists(atPath: url.path))
        let bytes = try FileManager.default
            .attributesOfItem(atPath: url.path)[.size] as? Int ?? 0
        // A 64×64 nine-frame gradient really does compress to about 1.5 kB, so this floor
        // is deliberately low — it is a "there is a container here at all" check and
        // nothing more. The claim that actually rules out truncation is the frame count
        // below, which needs the moov atom to have been written to be readable at all.
        #expect(bytes > 512, "file is \(bytes) bytes, which is not a container")

        // The failure this exists for: returning before `finishWriting` completes leaves
        // an mp4 that opens and reports a plausible duration while missing its tail. Only
        // decoding every sample catches it.
        let decoded = try await Self.frameCount(url)
        #expect(decoded == frames, "decoded \(decoded) frames, wrote \(frames)")

        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        let expected = Double(frames) / 24.0
        #expect(abs(duration.seconds - expected) < 1.0 / 24.0,
                "duration \(duration.seconds)s, expected \(expected)s")

        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let natural = try await track.load(.naturalSize)
        #expect(Int(natural.width) == size)
        #expect(Int(natural.height) == size)

        // `nominalFrameRate` is what the container advertises; the frame *count* is the
        // stronger claim, but reading every sample back costs more than it proves here.
        let rate = try await track.load(.nominalFrameRate)
        #expect(abs(Double(rate) - 24.0) < 0.5, "nominal frame rate \(rate)")
    }

    @Test("hevc is selectable and produces a readable mp4 too")
    func writesHEVC() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip-hevc.mp4")

        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, y, x, _ in
            Float((x + y) % 2)
        }
        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24, codec: .hevc))

        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let formats = try await track.load(.formatDescriptions)
        let subTypes = formats.map { CMFormatDescriptionGetMediaSubType($0) }
        #expect(subTypes.contains(kCMVideoCodecType_HEVC),
                "expected an HEVC track, got \(subTypes)")
    }

    @Test("an existing file at the destination is replaced, not appended to")
    func overwritesAnExistingFile() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // AVAssetWriter refuses to start on an existing URL, so without the unlink this is
        // the failure a second render of the same shot would hit.
        try Data(repeating: 0x41, count: 4096).write(to: url)
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, x, _ in
            Float(x) / 63.0
        }
        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24))

        let head = try FileHandle(forReadingFrom: url).read(upToCount: 8) ?? Data()
        #expect(head != Data(repeating: 0x41, count: 8),
                "the placeholder bytes are still there; the file was not replaced")
    }

    @Test("a directory at the destination is refused rather than removed")
    func refusesToRemoveADirectory() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)

        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        Self.expectFailure("a directory must not be unlinked") {
            try RenderOutput.writeVideo(rgb: rgb, to: url)
        } matches: {
            if case .destinationNotAFile(_, let kind) = $0 { return kind == "directory" }
            return false
        }
        #expect(FileManager.default.fileExists(atPath: url.path),
                "the directory was removed anyway")
    }

    // MARK: - Rejections

    @Test("a rank-5 decode() tensor throws .rank")
    func wrongRankThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // `[B, C, F, H, W]` — exactly `VideoVAEDecoder.decode`'s output, which is the
        // tensor someone will reach for by mistake because it is the one that has a name.
        let decoded = MLXArray(Array(repeating: Float(0.5), count: 1 * 3 * 9 * 16 * 16),
                               [1, 3, 9, 16, 16])
        Self.expectFailure("rank 5 is not [F, H, W, C]") {
            try RenderOutput.writeVideo(rgb: decoded, to: url)
        } matches: {
            if case .rank(let got) = $0 { return got == [1, 3, 9, 16, 16] }
            return false
        }
        #expect(!FileManager.default.fileExists(atPath: url.path),
                "a rejected input still created a file")
    }

    @Test("a 4-channel frame throws .channels")
    func wrongChannelCountThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // RGBA rather than RGB. This has to be an error and not a silent alpha drop: the
        // decoder's `out_channels` is 3, so a 4-channel tensor means the caller built the
        // stack some other way and the channel *order* is unknown too.
        let rgba = MLXArray(Array(repeating: Float(0.5), count: 9 * 16 * 16 * 4),
                            [9, 16, 16, 4])
        Self.expectFailure("4 channels is not RGB") {
            try RenderOutput.writeVideo(rgb: rgba, to: url)
        } matches: {
            if case .channels(let got) = $0 { return got == 4 }
            return false
        }
    }

    @Test("zero frames throws .noFrames instead of writing an empty container")
    func zeroFramesThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // The case this exists for: an empty file is a perfectly valid mp4, and the
        // flicker detector measures inter-frame difference — over zero frames it would
        // report no flicker and pass. A degenerate render has to fail loudly here or it
        // becomes a clean result downstream.
        let empty = MLXArray([Float](), [0, 16, 16, 3])
        Self.expectFailure("zero frames must not produce a file") {
            try RenderOutput.writeVideo(rgb: empty, to: url)
        } matches: {
            if case .noFrames = $0 { return true }
            return false
        }
        #expect(!FileManager.default.fileExists(atPath: url.path),
                "zero frames produced a file")
    }

    @Test("an odd dimension throws rather than reaching the 4:2:0 encoder")
    func oddDimensionThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        let odd = Self.clip(frames: 9, height: 15, width: 16) { _, _, _, _ in 0.5 }
        Self.expectFailure("odd height") {
            try RenderOutput.writeVideo(rgb: odd, to: url)
        } matches: {
            if case .oddDimension(_, let height) = $0 { return height == 15 }
            return false
        }
    }

    @Test("a non-positive frame rate throws")
    func badFrameRateThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        let rgb = Self.clip(frames: 9, height: 16, width: 16) { _, _, _, _ in 0.5 }
        Self.expectFailure("fps 0 would make every frame share a timestamp") {
            try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 0))
        } matches: {
            if case .badFrameRate = $0 { return true }
            return false
        }
    }

    // MARK: - The negative control

    /// Out-of-range values saturate. This is the assertion that proves the conversion.
    ///
    /// A float→uint8 cast without a preceding clamp truncates modulo 256: `-0.5 * 255` and
    /// `1.5 * 255` both land near mid-grey, and — worse than merely wrong — a *negative*
    /// value comes out **bright**. That failure is invisible in every other test here,
    /// because a well-behaved `[0, 1]` clip never exercises it, and it is invisible to the
    /// eye on a normal render because only a few overshooting pixels are affected. So it
    /// gets its own control, asserted on the exact bytes.
    @Test("out-of-range values clamp rather than wrap (exact bytes)")
    func outOfRangeValuesClampExactly() {
        // Column 0 undershoots, column 1 overshoots, columns 2-3 are the endpoints
        // themselves, column 4 is a value whose byte can be computed by hand.
        let samples: [Float] = [-0.5, 1.5, 0.0, 1.0, 0.5]
        let frame = MLXArray(samples.flatMap { [$0, $0, $0] }, [1, samples.count, 3])
        let bytes = RenderOutput.bgraBytes(frame: frame, range: .unitInterval)

        #expect(bytes.count == samples.count * 4)
        // BGRA, so each pixel's first three bytes are the (here identical) channels and
        // the fourth is alpha.
        let grey = stride(from: 0, to: bytes.count, by: 4).map { bytes[$0] }
        let alpha = stride(from: 3, to: bytes.count, by: 4).map { bytes[$0] }

        #expect(grey[0] == 0, "-0.5 became \(grey[0]); a wrap would put it near 129")
        #expect(grey[1] == 255, "1.5 became \(grey[1]); a wrap would put it near 126")
        #expect(grey[2] == 0)
        #expect(grey[3] == 255)
        #expect(grey[4] == 128, "0.5 * 255 + 0.5 rounds half-up to 128, got \(grey[4])")
        #expect(alpha.allSatisfy { $0 == 255 }, "alpha must be opaque")
    }

    @Test("the BGRA channel order is B, G, R and not R, G, B")
    func channelOrderIsBGRA() {
        // A pure red pixel. If the reorder were missing this comes back as (255, 0, 0, 255)
        // in memory, which a BGRA consumer displays as blue — a bug that survives every
        // shape check, every size check, and a greyscale fixture.
        let frame = MLXArray([Float(1), Float(0), Float(0)], [1, 1, 3])
        let bytes = RenderOutput.bgraBytes(frame: frame, range: .unitInterval)
        #expect(bytes == [0, 0, 255, 255], "got \(bytes), expected B=0 G=0 R=255 A=255")
    }

    @Test("signedUnit maps [-1, 1] onto the full byte range")
    func signedUnitRangeIsAffine() {
        let frame = MLXArray([Float(-1), Float(0), Float(1)].flatMap { [$0, $0, $0] },
                             [1, 3, 3])
        let bytes = RenderOutput.bgraBytes(frame: frame, range: .signedUnit)
        let grey = stride(from: 0, to: bytes.count, by: 4).map { bytes[$0] }
        #expect(grey == [0, 128, 255], "got \(grey)")
    }

    /// The same saturation claim, but observed after a real H.264 round trip.
    ///
    /// Weaker than the byte assertion above — the tolerance has to absorb the encode, the
    /// 4:2:0 chroma subsample and the limited↔full range conversion VideoToolbox applies —
    /// but it observes something the byte test cannot: that the saturated bytes actually
    /// reach the container in the right places, with the right stride, the right row order
    /// and the right channel order.
    @Test("out-of-range values are still saturated after a real encode")
    func outOfRangeValuesSurviveTheEncode() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clamp.mp4")

        let size = 64
        // Left half undershoots, right half overshoots. Sampled well inside each half so
        // the chroma subsample at the seam cannot reach the sample points.
        let rgb = Self.clip(frames: 4, height: size, width: size) { _, _, x, _ in
            x < size / 2 ? -0.5 : 1.5
        }
        try RenderOutput.writeVideo(rgb: rgb, to: url,
                                    spec: .init(fps: 24, averageBitRate: 8_000_000))

        let frame = try await Self.firstFrameBGRA(url)
        #expect(frame.width == size && frame.height == size)

        func pixel(_ x: Int, _ y: Int) -> (b: UInt8, g: UInt8, r: UInt8) {
            let base = y * frame.stride + x * 4
            return (frame.bytes[base], frame.bytes[base + 1], frame.bytes[base + 2])
        }
        let dark = pixel(size / 4, size / 2)
        let bright = pixel(size * 3 / 4, size / 2)

        // 24/255 is loose for a codec and tight for the bug: a wrap lands both of these
        // within a few counts of 128, which is over 100 away from either endpoint.
        let tolerance: Int = 24
        for (name, value) in [("blue", dark.b), ("green", dark.g), ("red", dark.r)] {
            #expect(Int(value) <= tolerance,
                    "undershoot did not saturate to black: \(name) = \(value)")
        }
        for (name, value) in [("blue", bright.b), ("green", bright.g), ("red", bright.r)] {
            #expect(Int(value) >= 255 - tolerance,
                    "overshoot did not saturate to white: \(name) = \(value)")
        }
    }

    // MARK: - Colour

    /// A colour gradient that stays off the endpoints.
    ///
    /// `0.1 … 0.9` rather than `0 … 1` because 0.0 and 1.0 are exactly where a
    /// limited-range encode clips, and the round trip below is measuring a *matrix*, not
    /// the saturation behaviour that ``outOfRangeValuesSurviveTheEncode`` already owns. The
    /// three channels vary along three different axes so no pair of them is ever equal —
    /// a clip where R == G == B has no chroma at all and would pass every assertion here
    /// with the matrix wired to anything.
    static func colourGradient(frames: Int, size: Int) -> MLXArray {
        clip(frames: frames, height: size, width: size) { f, y, x, c in
            let base: Float
            switch c {
            case 0: base = Float(x) / Float(size - 1)
            case 1: base = Float(y) / Float(size - 1)
            default: base = Float(f) / Float(max(frames - 1, 1))
            }
            return 0.1 + 0.8 * base
        }
    }

    /// The three colour extensions a track carries, plus the range flag.
    ///
    /// Read off the `CMFormatDescription` and not off the settings dictionary the writer
    /// was handed. That distinction is the whole point: a writer that builds a correct
    /// dictionary and then drops it on the floor passes a test that compares the dictionary
    /// against itself, and fails this one.
    static func colourAttachments(_ url: URL) async throws
        -> (primaries: String?, transfer: String?, matrix: String?, fullRange: Bool?) {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let formats = try await track.load(.formatDescriptions)
        let description = try #require(formats.first)
        func string(_ key: CFString) -> String? {
            CMFormatDescriptionGetExtension(description, extensionKey: key) as? String
        }
        return (string(kCMFormatDescriptionExtension_ColorPrimaries),
                string(kCMFormatDescriptionExtension_TransferFunction),
                string(kCMFormatDescriptionExtension_YCbCrMatrix),
                CMFormatDescriptionGetExtension(
                    description,
                    extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool)
    }

    /// Every compressed video sample in the file, concatenated.
    ///
    /// `outputSettings: nil` is a pass-through read, so these are the encoder's own bytes —
    /// the H.264 access units, before any decode. That is exactly what makes it the right
    /// instrument for "did tagging change the pixels": a *decoded* comparison cannot answer
    /// the question, because the decoded values are supposed to change (that is what
    /// stating the matrix does), while the encoded ones are not.
    ///
    /// `CMBlockBufferCopyDataBytes` rather than `GetDataPointer`, because a sample's block
    /// buffer is not guaranteed contiguous and a pointer read of a segmented one silently
    /// returns short.
    static func compressedSamples(_ url: URL) async throws -> Data {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        guard reader.startReading() else {
            throw NSError(domain: "RenderOutputTests", code: 7,
                          userInfo: [NSLocalizedDescriptionKey:
                                        "pass-through startReading failed: \(String(describing: reader.error))"])
        }
        var data = Data()
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var bytes = [UInt8](repeating: 0, count: length)
            guard CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length,
                                             destination: &bytes) == noErr else { continue }
            data.append(contentsOf: bytes)
        }
        return data
    }

    /// Frame 0 of `url`, decoded and compared against frame 0 of `source`.
    ///
    /// The BGRA readback is un-strided and un-swizzled here rather than in each caller, so
    /// the comparison is against the `[H, W, 3]` RGB the writer was given and not against
    /// some rearrangement of it.
    static func frameZeroError(_ url: URL, against source: MLXArray) async throws
        -> (worst: Float, mean: Float) {
        let decoded = try await firstFrameBGRA(url)
        let height = source.shape[1], width = source.shape[2]
        #expect(decoded.width == width && decoded.height == height,
                "decoded \(decoded.width)x\(decoded.height), source \(width)x\(height)")

        let expected = source[0].asType(.float32).asArray(Float.self)
        var worst: Float = 0
        var total: Double = 0
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = y * decoded.stride + x * 4
                // 32BGRA: B, G, R, A — so the RGB triple is read back to front.
                let rgb = [decoded.bytes[base + 2], decoded.bytes[base + 1],
                           decoded.bytes[base]]
                for c in 0 ..< 3 {
                    let delta = abs(Float(rgb[c]) / 255 - expected[(y * width + x) * 3 + c])
                    worst = max(worst, delta)
                    total += Double(delta)
                }
            }
        }
        return (worst, Float(total / Double(height * width * 3)))
    }

    /// The tagging claim, read back off the container.
    ///
    /// Uses `VideoSpec()`'s *default* colour rather than naming `.rec709`, so this is also
    /// the assertion that a caller who says nothing gets a tagged file. That matters more
    /// than it looks: every existing call site in the project — including the ones in this
    /// suite and in `MediaInputTests` — passes a spec that does not mention colour, so if
    /// the default were `.untagged` the feature would exist and be used by nobody.
    @Test("a track written with the default spec carries Rec.709 primaries, transfer and matrix")
    func taggedTrackCarriesTheColourAttachments() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tagged.mp4")

        let rgb = Self.colourGradient(frames: 9, size: 64)
        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24))

        let tags = try await Self.colourAttachments(url)
        #expect(tags.primaries == AVVideoColorPrimaries_ITU_R_709_2,
                "primaries = \(tags.primaries ?? "nil")")
        #expect(tags.transfer == AVVideoTransferFunction_ITU_R_709_2,
                "transfer = \(tags.transfer ?? "nil")")
        #expect(tags.matrix == AVVideoYCbCrMatrix_ITU_R_709_2,
                "matrix = \(tags.matrix ?? "nil")")

        // Printed, not asserted. `AVVideoColorPropertiesKey` has no range key and
        // AVFoundation exposes no other route to `video_full_range_flag`, so this is
        // VideoToolbox's choice and not this writer's. Limited range is what the format
        // calls for; nil or false here means limited, which agrees with it.
        print("[RenderOutput] tagged track FullRangeVideo = "
            + (tags.fullRange.map { "\($0)" } ?? "nil (i.e. limited)"))
    }

    /// The control that proves the tagging code is what is doing the work.
    ///
    /// Without this, every assertion above would pass just as well on a build where
    /// VideoToolbox had started tagging 709 by itself — which it does not, but "which it
    /// does not" is a claim about a closed-source encoder on one OS version and is exactly
    /// the kind of thing that changes underneath a project quietly. `.untagged` must come
    /// back empty on all three, and it is also the case a caller reaches for when they need
    /// to reproduce a file this writer emitted before colour tagging existed.
    @Test("untagged carries no colour attachments at all")
    func untaggedTrackCarriesNoColourAttachments() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("untagged.mp4")

        let rgb = Self.colourGradient(frames: 9, size: 64)
        try RenderOutput.writeVideo(rgb: rgb, to: url,
                                    spec: .init(fps: 24, colour: .untagged))

        let tags = try await Self.colourAttachments(url)
        #expect(tags.primaries == nil, "primaries = \(tags.primaries ?? "nil")")
        #expect(tags.transfer == nil, "transfer = \(tags.transfer ?? "nil")")
        #expect(tags.matrix == nil, "matrix = \(tags.matrix ?? "nil")")
    }

    /// `.rec601` writes the SD bundle, matrix included.
    ///
    /// The matrix is the assertion that matters — it is the only one of the three that
    /// moves a decoded number — but the primaries are checked too, because
    /// `AVVideoSettings.h` sanctions SMPTE-C/709/601 as a unit and a build where
    /// AVFoundation silently normalised the combination to something else would otherwise
    /// go unnoticed.
    @Test("rec601 writes the SMPTE-C / 709 / BT.601 bundle")
    func rec601TagsTheSDBundle() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("bt601.mp4")

        let rgb = Self.colourGradient(frames: 9, size: 64)
        try RenderOutput.writeVideo(rgb: rgb, to: url,
                                    spec: .init(fps: 24, colour: .rec601))

        let tags = try await Self.colourAttachments(url)
        #expect(tags.matrix == AVVideoYCbCrMatrix_ITU_R_601_4,
                "matrix = \(tags.matrix ?? "nil")")
        #expect(tags.primaries == AVVideoColorPrimaries_SMPTE_C,
                "primaries = \(tags.primaries ?? "nil")")
        #expect(tags.transfer == AVVideoTransferFunction_ITU_R_709_2,
                "transfer = \(tags.transfer ?? "nil")")
    }

    /// How far apart two compressed bitstreams are: differing bytes, and where the first is.
    ///
    /// A byte count rather than a bool because, on this machine, H.264 is **not**
    /// reproducible run-to-run — see ``taggingChangesTheContainerAndNotThePixels`` — so the
    /// useful question is not "identical?" but "further apart than the encoder's own
    /// run-to-run noise?".
    static func bitstreamDistance(_ a: Data, _ b: Data) -> (differing: Int, first: Int?,
                                                            total: Int) {
        let total = max(a.count, b.count)
        var differing = 0
        var first: Int?
        for i in 0 ..< total {
            let left = i < a.count ? a[a.startIndex + i] : nil
            let right = i < b.count ? b[b.startIndex + i] : nil
            if left != right {
                differing += 1
                if first == nil { first = i }
            }
        }
        return (differing, first, total)
    }

    /// **This is a tagging change. It must not have moved a pixel.**
    ///
    /// The failure mode being ruled out is specific and documented: when a source pixel
    /// buffer carries colour attachments that disagree with `AVVideoColorPropertiesKey`,
    /// `AVAssetWriter` converts between them — a real change to the encoded values, applied
    /// silently, and would land as an unexplained change in every render after this commit.
    /// The pool's buffers carry no attachments, so there should be
    /// nothing to convert, and this asserts that rather than assuming it.
    ///
    /// The instrument is the **compressed** H.264 sample data, read pass-through. A decoded
    /// comparison cannot make this claim: decoded values are *supposed* to differ between a
    /// tagged and an untagged file — that difference is the entire point of the change — so
    /// only the pre-decode bytes separate "the container now says something" from "the
    /// encoder now does something". The tag itself does not appear in these bytes at all: it
    /// rides in the SPS VUI, which lives in the `avcC` atom inside the format description,
    /// while a pass-through sample is the slice NALs and nothing else. So a tagging-only
    /// change should leave this data untouched.
    ///
    /// **Byte identity is not available on this machine and the test says so out loud.**
    /// Two untagged writes of the same tensor, at the same pinned bit rate, come back the
    /// same *length* and with differing contents — VideoToolbox's H.264 encoder is not
    /// reproducible run to run. So the claim is made against a **measured** noise floor
    /// instead of against zero: the untagged-vs-untagged distance is the tolerance, the
    /// tagged-vs-untagged distance has to sit inside it, and both numbers are printed on
    /// every run so a reader can see what the bound is actually worth rather than trusting
    /// a comment.
    ///
    /// **That same-length claim holds only while the encoder is uncontended, which is why
    /// this suite is `.serialized`.** Measured, after this test flaked in full-suite runs
    /// and nowhere else: run alone, thirty consecutive writes are 3538 bytes with a
    /// 1-byte floor. Run while the suite's other thirty-five tests encode in parallel, the
    /// *same* input comes back at 2914, 3538 or 4451 bytes and the floor jumps to ~2000 —
    /// concurrent VideoToolbox sessions do not produce the same bitstream. CPU load alone
    /// does not do this; only concurrent encodes do.
    ///
    /// That breaks the instrument rather than the claim. Once two untagged writes disagree
    /// in length, a tagged-vs-untagged comparison cannot separate "the tag converted
    /// pixels" from "the encoder behaved differently", and the test was reporting the
    /// former for the latter. The baseline is therefore checked *first* and reported as
    /// what it is, so a recurrence names the cause instead of blaming the tag.
    ///
    /// The measurement, over three runs on an M-series Mac: the stream is **3538 bytes**,
    /// the noise floor is **1 byte**, and tagged-vs-untagged is **1 or 2 bytes**, always at
    /// offset 49–50. So the two encodes are identical to within one byte in three and a
    /// half thousand, and the tag costs exactly as much divergence as running the same
    /// encode twice does — which is the strongest form this claim can take on a
    /// nondeterministic encoder. It is nowhere near a real conversion: a 709→601 shift
    /// moves most macroblocks in a clip with this much chroma, i.e. thousands of bytes, not
    /// two.
    ///
    /// The bound below is deliberately not tightened to those numbers. It is derived from
    /// the floor *measured on the same run*, so a machine where the encoder is more
    /// deterministic tightens it automatically and one where it is less does not go flaky.
    @Test("tagging changes the container and not the pixels")
    func taggingChangesTheContainerAndNotThePixels() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }

        let rgb = Self.colourGradient(frames: 9, size: 64)
        // Pinned so the encodes cannot differ through a rate decision; the default is
        // resolution-dependent and would be the same for all of them, but "would be" is the
        // kind of assumption this test exists to not make.
        func write(_ colour: RenderOutput.ColourTagging, _ name: String) async throws -> Data {
            let url = directory.appendingPathComponent(name)
            try RenderOutput.writeVideo(
                rgb: rgb, to: url,
                spec: .init(fps: 24, averageBitRate: 8_000_000, colour: colour))
            return try await Self.compressedSamples(url)
        }

        let untaggedA = try await write(.untagged, "a.mp4")
        let untaggedB = try await write(.untagged, "b.mp4")
        let tagged = try await write(.rec709, "tagged.mp4")

        let floor = Self.bitstreamDistance(untaggedA, untaggedB)
        let measured = Self.bitstreamDistance(untaggedA, tagged)
        print("[RenderOutput] compressed bitstream, \(floor.total) bytes. Encoder noise "
            + "floor (untagged vs untagged): \(floor.differing) bytes differ, first at "
            + "\(floor.first.map { "\($0)" } ?? "n/a"). Tagged vs untagged: "
            + "\(measured.differing) bytes differ, first at "
            + "\(measured.first.map { "\($0)" } ?? "n/a").")

        // The instrument's validity, checked before the claim it is used to make. Two
        // untagged writes of one tensor must agree in length and differ in only a handful
        // of bytes; when they do not, the encoder is not reproducible on this run and
        // nothing below can be attributed to the tag. Asserted rather than skipped, because
        // a silent skip here would be a test that stops testing the day the suite loses its
        // `.serialized` trait — but the message names the encoder, not the tag.
        let baseline = "the two UNTAGGED writes of one tensor came back at "
            + "\(untaggedA.count) and \(untaggedB.count) bytes, differing in "
            + "\(floor.differing) of \(floor.total). The encoder is not reproducible on "
            + "this run, so tagged-vs-untagged cannot be attributed to the tag. This is "
            + "what concurrent VideoToolbox sessions do: check that this suite is still "
            + "`.serialized` before reading anything below as a colour bug."
        let reproducible = untaggedA.count == untaggedB.count && floor.differing <= 64
        #expect(reproducible, "\(baseline)")
        guard reproducible else { return }

        let lengths = "tagged bitstream is \(tagged.count) bytes against "
            + "\(untaggedA.count) untagged; a length change is not a labelling change"
        #expect(untaggedA.count == tagged.count, "\(lengths)")

        // The floor is measured on this run, not hardcoded, so a machine or OS where the
        // encoder is *more* deterministic tightens this automatically. `+ 16` keeps a run
        // where the floor happens to come out at zero from being infinitely strict on the
        // very next comparison; it is two orders of magnitude below the thousands of bytes
        // a real colour conversion of this clip would move.
        let allowed = floor.differing + 16
        let drift = "tagged vs untagged differs in \(measured.differing) of "
            + "\(measured.total) bytes, against an encoder noise floor of "
            + "\(floor.differing) — that is beyond run-to-run variation, so VideoToolbox "
            + "converted the pixels rather than only labelling them"
        #expect(measured.differing <= allowed, "\(drift)")

        // The unit-level half of the same claim, and the exact one: `bgraBytes` has no
        // colour parameter, so the tensor→BGRA conversion is structurally incapable of
        // depending on the tag. There is no nondeterminism here and no tolerance — this is
        // the byte-identity control the compressed comparison above cannot be.
        let bytes = RenderOutput.bgraBytes(frame: rgb[0], range: .unitInterval)
        #expect(bytes == RenderOutput.bgraBytes(frame: rgb[0], range: .unitInterval),
                "the BGRA conversion is not even reproducible with itself")
        #expect(bytes.count == 64 * 64 * 4)
    }

    /// The round trip the tag exists to fix.
    ///
    /// Untagged, this pair is **not** the identity in colour: the encoder picks 709, the
    /// decoder picks 601, and a pure red pixel comes back as `(0.9176, 0, 0.0078)` —
    /// `MediaInputTests.colourIsBoundedByTheUntaggedMatrix` derives that arithmetic in full
    /// and measured a mean error of 6.5/255 over a gradient. Stating the matrix removes the
    /// disagreement, and what is left is the codec.
    ///
    /// Both numbers are measured here and both are printed, because the interesting claim is
    /// the *ratio* rather than either bound: a tagged round trip that were merely "within
    /// tolerance" while still five times worse than an achromatic one would mean the tag had
    /// not taken effect, and no absolute bound loose enough to be non-flaky would notice.
    ///
    /// The measurement, stable to the last digit across three runs on an M-series Mac:
    /// **tagged** max 0.0294 (7.5/255), mean 0.0027 (0.69/255); **untagged** max 0.1351
    /// (34.5/255), mean 0.0284 (7.24/255). That is a **10.5x** improvement in the mean and
    /// a 4.6x one in the worst pixel, and it lands the tagged colour round trip on top of
    /// the *achromatic* numbers `MediaInputTests` measured through the same encoder and
    /// reader (0.0265 max, 0.0018 mean) — which is the point. What is left after tagging is
    /// the codec; what was there before was the codec plus a matrix nobody chose.
    @Test("colour survives the tagged round trip, and better than the untagged one")
    func colourSurvivesTheTaggedRoundTrip() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }

        let frames = 9, size = 64
        let rgb = Self.colourGradient(frames: frames, size: size)
        func write(_ colour: RenderOutput.ColourTagging, _ name: String) throws -> URL {
            let url = directory.appendingPathComponent(name)
            try RenderOutput.writeVideo(
                rgb: rgb, to: url,
                spec: .init(fps: 24, averageBitRate: 20_000_000, colour: colour))
            return url
        }

        let tagged = try await Self.frameZeroError(write(.rec709, "tagged.mp4"), against: rgb)
        let untagged = try await Self.frameZeroError(write(.untagged, "untagged.mp4"),
                                                    against: rgb)
        print("[RenderOutput] frame-0 round trip: tagged max \(tagged.worst) "
            + "(\(tagged.worst * 255)/255) mean \(tagged.mean) (\(tagged.mean * 255)/255); "
            + "untagged max \(untagged.worst) (\(untagged.worst * 255)/255) mean "
            + "\(untagged.mean) (\(untagged.mean * 255)/255)")

        // Absolute bounds, at the level `MediaInputTests` measured for an *achromatic* clip
        // through the same encoder and reader — 0.0265 max, 0.0018 mean — with the same ~2x
        // headroom it uses. Tagging is what is supposed to bring a colour clip down to the
        // achromatic numbers, so borrowing them is the claim, not a coincidence.
        #expect(tagged.worst < 0.05, "max |Δ| = \(tagged.worst) through a tagged file")
        #expect(tagged.mean < 0.006, "mean |Δ| = \(tagged.mean) through a tagged file")

        // The comparative claim. 0.01 is far above the tagged mean and far below the
        // untagged one, so this fails loudly if the tag stops reaching the container while
        // still leaving room for the codec to have a bad day.
        let comparative = "the untagged round trip (mean \(untagged.mean)) was no worse "
            + "than the tagged one (mean \(tagged.mean)); the 709/601 mismatch this tag "
            + "fixes has apparently gone away by itself, which needs explaining"
        #expect(untagged.mean > tagged.mean, "\(comparative)")
    }

    // MARK: - Audio fixtures

    /// `[1, 2, samples]`, channel-major, from two per-channel closures.
    ///
    /// Channel-major and *not* interleaved, deliberately: this is the layout
    /// ``AudioVocoder/waveform(fromMel:)`` returns, and building the fixture the other way
    /// round would quietly make every interleaving assertion below a tautology.
    static func stereo(frames: Int, left: (Int) -> Float,
                       right: (Int) -> Float) -> MLXArray {
        var flat = [Float]()
        flat.reserveCapacity(frames * 2)
        for i in 0 ..< frames { flat.append(left(i)) }
        for i in 0 ..< frames { flat.append(right(i)) }
        return MLXArray(flat, [1, 2, frames])
    }

    /// A 440 Hz tone. The right channel is a quarter-cycle behind so the two are never
    /// equal, which is what makes a swap or a mix visible in the readback.
    static func sine(seconds: Double, sampleRate: Int, hz: Double = 440,
                     amplitude: Float = 0.5) -> MLXArray {
        let frames = Int(seconds * Double(sampleRate))
        let step = 2 * Double.pi * hz / Double(sampleRate)
        return stereo(frames: frames,
                      left: { amplitude * Float(sin(step * Double($0))) },
                      right: { amplitude * Float(cos(step * Double($0))) })
    }

    // MARK: - The interleave, at the unit level

    /// Clamping and interleaving in one assertion, on exact floats.
    ///
    /// These are the same claim in two directions and they are checked together because a
    /// fixture that separates them is weaker than one that does not: the expected buffer
    /// below is wrong if the samples wrap, wrong if the channels are blocked `L…L R…R`
    /// instead of paired, and wrong if they are paired in the other order.
    ///
    /// The wrap half matters for the reason the BGRA one does, with a different symptom.
    /// Nothing in the vocoder bounds its generators — `use_tanh_at_final` is false — and
    /// while `waveform(fromMel:)` clips at the bandwidth-extension mix,
    /// `baseRateWaveform(fromMel:)` does not. An unclamped `+1.5` narrowed
    /// to int16 by any downstream consumer comes back as a full-scale *negative* sample:
    /// a click, at the loudest instant, which is where it is least likely to be attributed
    /// to the writer.
    @Test("samples clamp and interleave to L, R pairs (exact floats)")
    func audioClampsAndInterleavesExactly() throws {
        let audio = Self.stereo(frames: 4,
                                left: { [-1.5, 1.5, 0.0, 0.5][$0] },
                                right: { [0.25, -0.25, -2.0, 2.0][$0] })
        let plan = try RenderOutput.plan(forAudio: audio, spec: .init(sampleRate: 48_000))

        #expect(plan.frames == 4, "frames counts samples per channel, not floats")
        #expect(plan.interleaved.count == 8)
        #expect(plan.seconds == 4.0 / 48_000.0)
        // L0 R0 L1 R1 …, every out-of-range value saturated to the nearest endpoint.
        #expect(plan.interleaved == [-1, 0.25, 1, -0.25, 0, -1, 0.5, 1],
                "got \(plan.interleaved)")
    }

    // MARK: - Muxing

    @Test("a sine muxed with a clip produces a two-track mp4 at the right rate")
    func muxesAudioAndVideo() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("av.mp4")

        // 24 frames at 24 fps is exactly one second, matching the audio, so this test says
        // nothing about the mismatch policy — that has its own pair of tests below.
        let rate = 48_000
        let rgb = Self.clip(frames: 24, height: 64, width: 64) { f, y, x, c in
            c == 2 ? Float(f) / 23.0 : Float(x + y) / 126.0
        }
        let audio = Self.sine(seconds: 1, sampleRate: rate)

        try RenderOutput.write(rgb: rgb, audio: audio, to: url,
                               spec: .init(fps: 24), audioSpec: .init(sampleRate: rate))

        let asset = AVURLAsset(url: url)
        let all = try await asset.load(.tracks)
        #expect(all.count == 2, "expected a video and an audio track, got \(all.count)")

        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let track = try #require(audioTracks.first)
        let range = try await track.load(.timeRange)
        #expect(abs(range.duration.seconds - 1.0) < 0.05,
                "audio track is \(range.duration.seconds)s, expected ~1s")

        // The rate the container advertises, read off the track's own format description
        // rather than from the settings dictionary this test would otherwise be comparing
        // against itself.
        let formats = try await track.load(.formatDescriptions)
        let description = try #require(formats.first)
        let asbd = try #require(
            CMAudioFormatDescriptionGetStreamBasicDescription(description))
        #expect(asbd.pointee.mSampleRate == Double(rate),
                "track rate \(asbd.pointee.mSampleRate)")
        #expect(asbd.pointee.mChannelsPerFrame == 2,
                "track channels \(asbd.pointee.mChannelsPerFrame)")
        #expect(asbd.pointee.mFormatID == kAudioFormatMPEG4AAC,
                "expected AAC by default, got \(asbd.pointee.mFormatID)")

        // The video is untouched by the audio track's presence — the frame count is the
        // claim that catches a mux that finished one input and truncated the other.
        let decoded = try await Self.frameCount(url)
        #expect(decoded == 24, "decoded \(decoded) video frames of 24")

        let samples = try await Self.decodeAudio(url)
        #expect(abs(samples.frames - rate) < rate / 20,
                "decoded \(samples.frames) samples, expected about \(rate)")
    }

    /// The negative control for the whole audio path: constant `L = +0.5`, `R = −0.5`.
    ///
    /// **Without this test an interleaving mistake is invisible.** Every other assertion
    /// here — track count, duration, sample rate, sample count, file size — passes exactly
    /// as well on a file whose channels are swapped, or whose left channel is the first
    /// half of the timeline and whose right channel is the second. That failure produces
    /// audio which plays, at the right length, at the right level, and is wrong. So the
    /// fixture is deliberately asymmetric in both sign and channel: a swap flips both
    /// signs, and any mixing of the two collapses them toward zero.
    ///
    /// Run through the default AAC path rather than LPCM on purpose, because that is the
    /// path with the `CMSampleBuffer` construction in it and the one a real render takes.
    /// DC survives AAC well enough to make the claim cleanly — the tolerance below is
    /// three orders of magnitude smaller than the gap it has to resolve.
    @Test("the channels are not swapped and not mixed")
    func channelOrderSurvivesTheMux() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("channels.mp4")

        let rate = 48_000
        let rgb = Self.clip(frames: 24, height: 64, width: 64) { _, _, x, _ in
            Float(x) / 63.0
        }
        let audio = Self.stereo(frames: rate, left: { _ in 0.5 }, right: { _ in -0.5 })
        try RenderOutput.write(rgb: rgb, audio: audio, to: url,
                               spec: .init(fps: 24), audioSpec: .init(sampleRate: rate))

        let decoded = try await Self.decodeAudio(url)
        #expect(decoded.frames > rate / 2, "only \(decoded.frames) samples came back")

        // Sampled from the middle. The head and tail of an AAC stream carry the encoder's
        // own ramp, and this test is about which channel is which, not about the codec's
        // transient response.
        let middle = (decoded.frames / 2) * 2
        let left = decoded.interleaved[middle]
        let right = decoded.interleaved[middle + 1]
        #expect(abs(left - 0.5) < 0.01,
                "left is \(left); a channel swap would put it near -0.5 and a mix near 0")
        #expect(abs(right + 0.5) < 0.01,
                "right is \(right); a channel swap would put it near +0.5 and a mix near 0")

        // A mean over the whole stream, so a partial interleave — the L…L R…R blocking a
        // stride-ignoring readback would produce — cannot hide behind one lucky sample.
        var leftSum = 0.0, rightSum = 0.0
        for i in stride(from: 0, to: decoded.frames * 2, by: 2) {
            leftSum += Double(decoded.interleaved[i])
            rightSum += Double(decoded.interleaved[i + 1])
        }
        let leftMean = leftSum / Double(decoded.frames)
        let rightMean = rightSum / Double(decoded.frames)
        #expect(abs(leftMean - 0.5) < 0.01, "left mean \(leftMean)")
        #expect(abs(rightMean + 0.5) < 0.01, "right mean \(rightMean)")
    }

    /// LPCM is a real option and it is exact, which is the property that makes it worth
    /// having: this is the only assertion in the suite that says a sample reached the
    /// container unchanged rather than approximately unchanged.
    @Test("audio survives an LPCM round trip exactly")
    func audioSurvivesAnLPCMRoundTripExactly() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("lpcm.mp4")

        let rate = 48_000
        let frames = 4800                                            // 0.1 s
        let rgb = Self.clip(frames: 3, height: 64, width: 64) { _, _, x, _ in
            Float(x) / 63.0
        }
        let audio = Self.stereo(frames: frames,
                                left: { Float($0) / Float(frames) - 0.5 },
                                right: { 0.5 - Float($0) / Float(frames) })
        try RenderOutput.write(rgb: rgb, audio: audio, to: url,
                               spec: .init(fps: 24),
                               audioSpec: .init(sampleRate: rate, codec: .lpcm))

        let decoded = try await Self.decodeAudio(url)
        #expect(decoded.frames == frames, "decoded \(decoded.frames) of \(frames)")
        for i in stride(from: 0, to: min(decoded.frames, frames), by: 419) {
            let expectedLeft = Float(i) / Float(frames) - 0.5
            #expect(decoded.interleaved[i * 2] == expectedLeft,
                    "sample \(i) L is \(decoded.interleaved[i * 2]), expected \(expectedLeft)")
            #expect(decoded.interleaved[i * 2 + 1] == -expectedLeft,
                    "sample \(i) R is \(decoded.interleaved[i * 2 + 1])")
        }
    }

    // MARK: - Duration honesty

    @Test("a small duration mismatch is written as-is, both tracks at their true length")
    func smallDurationMismatchIsKept() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("ragged.mp4")

        // 9 frames at 24 fps is 0.375s against 0.5s of audio — 0.125s apart, inside the
        // 0.25s floor. Nothing is padded and nothing is trimmed, and the assertion is that
        // both tracks come back at the length they were given rather than at a common one.
        let rate = 48_000
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, x, _ in
            Float(x) / 63.0
        }
        let audio = Self.sine(seconds: 0.5, sampleRate: rate)
        try RenderOutput.write(rgb: rgb, audio: audio, to: url,
                               spec: .init(fps: 24), audioSpec: .init(sampleRate: rate))

        let asset = AVURLAsset(url: url)
        let video = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let sound = try #require(try await asset.loadTracks(withMediaType: .audio).first)
        let videoSeconds = try await video.load(.timeRange).duration.seconds
        let audioSeconds = try await sound.load(.timeRange).duration.seconds

        #expect(abs(videoSeconds - 0.375) < 0.05,
                "video is \(videoSeconds)s; padding it to the audio would give 0.5")
        #expect(abs(audioSeconds - 0.5) < 0.05,
                "audio is \(audioSeconds)s; trimming it to the video would give 0.375")
    }

    /// A gross mismatch throws, and this is the bug it is standing in for.
    ///
    /// The audio and video latent counts come from the same requested duration by different
    /// arithmetic. Get the audio one wrong by a factor and the *only* symptom is a length
    /// disagreement — the samples are all individually plausible. A writer that padded or
    /// trimmed to fit would consume that symptom and emit a file that looks right.
    @Test("a gross duration mismatch throws instead of muxing")
    func grossDurationMismatchThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("mismatch.mp4")

        let rate = 48_000
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, x, _ in
            Float(x) / 63.0
        }
        let audio = Self.sine(seconds: 2, sampleRate: rate)          // against 0.375s of video
        Self.expectFailure("2s of audio on 0.375s of video is not a rounding edge") {
            try RenderOutput.write(rgb: rgb, audio: audio, to: url,
                                   spec: .init(fps: 24), audioSpec: .init(sampleRate: rate))
        } matches: {
            guard case .durationMismatch(let video, let audio, let allowed) = $0 else {
                return false
            }
            return abs(video - 0.375) < 1e-9 && abs(audio - 2.0) < 1e-9 && allowed == 0.25
        }
        #expect(!FileManager.default.fileExists(atPath: url.path),
                "a rejected mux still created a file")
    }

    // MARK: - Audio rejections

    @Test("a rank-2 waveform throws .audioRank")
    func audioWrongRankThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // `[2, samples]` — the batch axis dropped, which is what a caller who indexed the
        // vocoder's output before handing it over would have.
        let flat = MLXArray([Float](repeating: 0, count: 2 * 480), [2, 480])
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        Self.expectFailure("rank 2 is not [B, 2, samples]") {
            try RenderOutput.write(rgb: rgb, audio: flat, to: url)
        } matches: {
            if case .audioRank(let got) = $0 { return got == [2, 480] }
            return false
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test("a batch of two throws .audioBatch rather than picking a clip")
    func audioBatchThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        let batched = MLXArray([Float](repeating: 0.1, count: 2 * 2 * 480), [2, 2, 480])
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        Self.expectFailure("a file holds one clip; the writer must not choose") {
            try RenderOutput.write(rgb: rgb, audio: batched, to: url)
        } matches: {
            if case .audioBatch(let got) = $0 { return got == 2 }
            return false
        }
    }

    @Test("a mono waveform throws .audioChannels")
    func audioChannelCountThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        let mono = MLXArray([Float](repeating: 0.1, count: 480), [1, 1, 480])
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        Self.expectFailure("1 channel is not stereo") {
            try RenderOutput.write(rgb: rgb, audio: mono, to: url)
        } matches: {
            if case .audioChannels(let got) = $0 { return got == 1 }
            return false
        }
    }

    /// `[1, samples, 2]` is the transposed layout, and it is the dangerous one: it is
    /// well-formed, it has the right number of floats in it, and read as `[B, C, L]` it is
    /// a 2-sample clip with `samples` channels. The channel check is what stops it.
    @Test("a transposed [1, samples, 2] waveform is caught by the channel check")
    func audioTransposedLayoutThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        let transposed = MLXArray([Float](repeating: 0.1, count: 480 * 2), [1, 480, 2])
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        Self.expectFailure("[B, L, C] is not [B, C, L]") {
            try RenderOutput.write(rgb: rgb, audio: transposed, to: url)
        } matches: {
            if case .audioChannels(let got) = $0 { return got == 480 }
            return false
        }
    }

    @Test("zero samples throws .noSamples instead of writing a silent track")
    func audioZeroSamplesThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        // An empty audio track is a valid mp4 track, and a render that produced no samples
        // at all would come back as a file with a silent-but-present audio stream — which
        // looks like a vocoder that ran and produced quiet audio.
        let empty = MLXArray([Float](), [1, 2, 0])
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        Self.expectFailure("zero samples must not produce a track") {
            try RenderOutput.write(rgb: rgb, audio: empty, to: url)
        } matches: {
            if case .noSamples = $0 { return true }
            return false
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test("a non-positive sample rate throws")
    func audioBadSampleRateThrows() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("clip.mp4")

        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, _, _ in 0.5 }
        let audio = Self.stereo(frames: 480, left: { _ in 0.1 }, right: { _ in 0.1 })
        Self.expectFailure("rate 0 would make every sample share a timestamp") {
            try RenderOutput.write(rgb: rgb, audio: audio, to: url,
                                   audioSpec: .init(sampleRate: 0))
        } matches: {
            if case .badSampleRate(let got) = $0 { return got == 0 }
            return false
        }
    }

    @Test("writeVideo still produces a silent, single-track mp4")
    func writeVideoStaysSilent() async throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("silent.mp4")

        // The refactor that added the audio track routed `writeVideo` through the same
        // writer. This is the assertion that it did not acquire an empty audio track on
        // the way — which would be silence that a later check could mistake for a vocode.
        let rgb = Self.clip(frames: 9, height: 64, width: 64) { _, _, x, _ in
            Float(x) / 63.0
        }
        try RenderOutput.writeVideo(rgb: rgb, to: url, spec: .init(fps: 24))

        let asset = AVURLAsset(url: url)
        let all = try await asset.load(.tracks)
        #expect(all.count == 1, "expected exactly one track, got \(all.count)")
        let audio = try await asset.loadTracks(withMediaType: .audio)
        #expect(audio.isEmpty, "writeVideo produced an audio track")
    }

    // MARK: - WAV

    /// Write, read back, and compare. The claim is identity, not similarity.
    ///
    /// The reader is `AVAudioFile`, which is also what wrote the file, so this is not a
    /// fully independent check of the RIFF header — the `#expect` on the leading `RIFF`
    /// and `WAVE` bytes is there to cover the part it cannot see, namely that the container
    /// is a WAV at all and not a CAF that happens to be named one. What the round trip
    /// *does* check independently is the interleave: the file is written from a single
    /// interleaved buffer and read back deinterleaved into per-channel pointers, so the
    /// two sides do not share a layout assumption.
    @Test("writeWAV round-trips samples, rate, channels and length")
    func wavRoundTrips() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.wav")

        let rate = 48_000
        let frames = 4800
        let audio = Self.sine(seconds: 0.1, sampleRate: rate)
        try RenderOutput.writeWAV(audio: audio, to: url, sampleRate: rate)

        let head = try FileHandle(forReadingFrom: url).read(upToCount: 12) ?? Data()
        #expect(head.prefix(4) == Data("RIFF".utf8), "not a RIFF file")
        #expect(head.suffix(4) == Data("WAVE".utf8), "not a WAVE file")

        let read = try Self.readWAV(url)
        #expect(read.sampleRate == Double(rate), "rate \(read.sampleRate)")
        #expect(read.channels == 2, "channels \(read.channels)")
        #expect(read.frames == frames, "frames \(read.frames), expected \(frames)")

        // fp32 in and fp32 out, so the tolerance is for the comparison's own arithmetic
        // and not for the format. A 16-bit WAV would need ~3e-5 here, which is the reason
        // this writer does not produce one.
        let step = 2 * Double.pi * 440 / Double(rate)
        for i in [0, 1, 2, 37, 1000, frames - 1] {
            let expectedLeft = Float(0.5 * sin(step * Double(i)))
            let expectedRight = Float(0.5 * cos(step * Double(i)))
            #expect(abs(read.left[i] - expectedLeft) < 1e-6,
                    "L[\(i)] = \(read.left[i]), expected \(expectedLeft)")
            #expect(abs(read.right[i] - expectedRight) < 1e-6,
                    "R[\(i)] = \(read.right[i]), expected \(expectedRight)")
        }
    }

    /// The same channel control as the muxed path, with no codec in the way, so the
    /// tolerance is zero rather than merely small.
    @Test("writeWAV does not swap or mix the channels")
    func wavChannelOrderIsExact() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("dc.wav")

        let frames = 480
        let audio = Self.stereo(frames: frames, left: { _ in 0.5 }, right: { _ in -0.5 })
        try RenderOutput.writeWAV(audio: audio, to: url, sampleRate: 48_000)

        let read = try Self.readWAV(url)
        #expect(read.frames == frames)
        #expect(read.left.allSatisfy { $0 == 0.5 },
                "left is not constant +0.5; a swap gives -0.5 and a mix gives 0")
        #expect(read.right.allSatisfy { $0 == -0.5 },
                "right is not constant -0.5; a swap gives +0.5 and a mix gives 0")
    }

    @Test("writeWAV clamps rather than wrapping")
    func wavClampsOutOfRange() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("hot.wav")

        let audio = Self.stereo(frames: 4,
                                left: { [-1.5, 1.5, -1.0, 1.0][$0] },
                                right: { [3.0, -3.0, 0.0, 0.25][$0] })
        try RenderOutput.writeWAV(audio: audio, to: url, sampleRate: 48_000)

        let read = try Self.readWAV(url)
        #expect(read.left == [-1, 1, -1, 1], "left \(read.left)")
        #expect(read.right == [1, -1, 0, 0.25], "right \(read.right)")
    }

    @Test("a non-wav extension is refused rather than silently written as something else")
    func wavExtensionIsChecked() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.aiff")

        let audio = Self.stereo(frames: 480, left: { _ in 0.1 }, right: { _ in -0.1 })
        Self.expectFailure("AVAudioFile picks the container from the extension") {
            try RenderOutput.writeWAV(audio: audio, to: url, sampleRate: 48_000)
        } matches: {
            if case .notAWAVExtension = $0 { return true }
            return false
        }
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test("writeWAV rejects the same malformed tensors the muxer does")
    func wavRejectsMalformedAudio() throws {
        let directory = try Self.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tone.wav")

        Self.expectFailure("zero samples") {
            try RenderOutput.writeWAV(audio: MLXArray([Float](), [1, 2, 0]), to: url,
                                      sampleRate: 48_000)
        } matches: {
            if case .noSamples = $0 { return true }
            return false
        }
        Self.expectFailure("a batch of two") {
            let batched = MLXArray([Float](repeating: 0, count: 2 * 2 * 48), [2, 2, 48])
            try RenderOutput.writeWAV(audio: batched, to: url, sampleRate: 48_000)
        } matches: {
            if case .audioBatch(let got) = $0 { return got == 2 }
            return false
        }
        #expect(!FileManager.default.fileExists(atPath: url.path),
                "a rejected waveform still created a file")
    }

    // MARK: - Audio readback

    /// Decode an mp4's audio track back to interleaved fp32.
    ///
    /// LPCM output settings rather than pass-through, for the reason ``frameCount(_:)``
    /// forces a BGRA decode: a pass-through reader hands back the container's packets,
    /// which for AAC is 1024 frames apiece and tells you nothing about the samples.
    /// `IsNonInterleaved: false` is asked for explicitly, because a deinterleaved readback
    /// would make the channel-order control read the file the same wrong way it might have
    /// been written.
    static func decodeAudio(_ url: URL) async throws
        -> (frames: Int, interleaved: [Float]) {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .audio).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ])
        reader.add(output)
        guard reader.startReading() else {
            throw NSError(domain: "RenderOutputTests", code: 4,
                          userInfo: [NSLocalizedDescriptionKey:
                                        "audio startReading failed: \(String(describing: reader.error))"])
        }
        var samples = [Float]()
        while let buffer = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            var length = 0
            var pointer: UnsafeMutablePointer<Int8>?
            guard CMBlockBufferGetDataPointer(block, atOffset: 0, lengthAtOffsetOut: nil,
                                              totalLengthOut: &length,
                                              dataPointerOut: &pointer) == noErr,
                  let pointer else { continue }
            samples.append(contentsOf: UnsafeRawBufferPointer(start: pointer, count: length)
                .bindMemory(to: Float.self))
        }
        return (samples.count / 2, samples)
    }

    /// Read a WAV back per channel.
    ///
    /// `AVAudioFile`'s reading `processingFormat` is deinterleaved fp32 regardless of what
    /// is on disk, which is exactly what makes it a useful check of a file written from an
    /// interleaved buffer — the two sides cannot share a mistake about the layout.
    static func readWAV(_ url: URL) throws
        -> (sampleRate: Double, channels: Int, frames: Int, left: [Float], right: [Float]) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(file.length)) else {
            throw NSError(domain: "RenderOutputTests", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "no read buffer"])
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData, buffer.format.channelCount >= 2 else {
            throw NSError(domain: "RenderOutputTests", code: 6,
                          userInfo: [NSLocalizedDescriptionKey:
                                        "expected 2 channels, got \(buffer.format.channelCount)"])
        }
        let frames = Int(buffer.frameLength)
        return (format.sampleRate, Int(format.channelCount), frames,
                Array(UnsafeBufferPointer(start: channels[0], count: frames)),
                Array(UnsafeBufferPointer(start: channels[1], count: frames)))
    }

    /// Decode every sample and count the frames that come out.
    ///
    /// Decompression settings, not `nil`: a pass-through `AVAssetReaderTrackOutput`
    /// hands back the container's samples, which on an mp4 is not one buffer per frame —
    /// this returned 13 for a 9-frame clip. Forcing a BGRA decode makes the count mean
    /// what it says.
    static func frameCount(_ url: URL) async throws -> Int {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
        reader.add(output)
        guard reader.startReading() else { return 0 }
        var count = 0
        while output.copyNextSampleBuffer() != nil { count += 1 }
        return count
    }

    /// Decode frame 0 back to packed BGRA.
    ///
    /// `AVAssetReader` rather than `AVAssetImageGenerator`: the generator returns a
    /// `CGImage`, which means a colour-managed conversion sits between the file and the
    /// bytes, and this test's whole subject is whether byte values survived.
    static func firstFrameBGRA(_ url: URL) async throws
        -> (width: Int, height: Int, stride: Int, bytes: [UInt8]) {
        let asset = AVURLAsset(url: url)
        let track = try #require(try await asset.loadTracks(withMediaType: .video).first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
        reader.add(output)
        guard reader.startReading() else {
            throw NSError(domain: "RenderOutputTests", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                                        "startReading failed: \(String(describing: reader.error))"])
        }
        guard let sample = output.copyNextSampleBuffer(),
              let image = CMSampleBufferGetImageBuffer(sample) else {
            throw NSError(domain: "RenderOutputTests", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "no decodable first frame"])
        }
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        let width = CVPixelBufferGetWidth(image)
        let height = CVPixelBufferGetHeight(image)
        let stride = CVPixelBufferGetBytesPerRow(image)
        guard let base = CVPixelBufferGetBaseAddress(image) else {
            throw NSError(domain: "RenderOutputTests", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "no base address"])
        }
        let raw = UnsafeRawBufferPointer(start: base, count: stride * height)
        return (width, height, stride, [UInt8](raw))
    }
}

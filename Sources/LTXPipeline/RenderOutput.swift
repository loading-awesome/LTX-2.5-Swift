// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import AVFoundation
import CoreVideo
import Foundation
import MLX

/// The "and then you have a file" step: decoded RGB frames — and, now, vocoder samples —
/// → an H.264/HEVC `.mp4`.
///
/// `Pipeline.render`'s own doc note says decode is deliberately not folded into the
/// sampler and that there is therefore "no *and then you have a file* to reach anyway".
/// This is that file. Everything that judges a render by eye needs a real container on
/// disk — the temporal flicker detector reads a video, not a tensor — so until this
/// existed there were pixels nobody could be shown.
///
/// ## What this is not
///
/// **Nothing downstream of ``writeVideo(rgb:to:spec:)`` can be checked on value
/// agreement**, because H.264 is a lossy encode. What *is* checkable, and what
/// ``RenderOutputTests`` checks, is the part that is exact: the tensor→BGRA conversion,
/// the frame count, the geometry, and the failure modes. The encode itself is only
/// asserted to have happened and to be roughly the right shape when read back.
///
/// **The audio is in the same position, and less of it could be otherwise.** This file
/// used to say audio was not muxed in because there was no waveform to attach;
/// ``AudioVocoder`` is that waveform now, so ``write(rgb:audio:to:spec:audioSpec:)``
/// exists. What has not changed is contract 11: a bit-identical mel does not guarantee a
/// bit-identical waveform, so there is no waveform to be equal to and no value comparison
/// to run. The audio path is therefore **perceptual only** — it is judged by listening,
/// and the checks below are limited to the parts that are exact: the
/// tensor→interleaved-float conversion, the channel order, the sample count, the sample
/// rate, and the failure modes. A green `RenderOutputTests` says the samples reached the
/// container unmangled. It says nothing about whether they are the right samples.
///
/// ## The colour *is* stated, and that part is not a judgement call
///
/// This file used to write an untagged mp4 and say so as a known gap. It no longer does:
/// see ``ColourTagging`` for what is written and which third of the answer is a defensible
/// guess rather than a value read off a source. The short version is that
/// an untagged file is ambiguously coloured rather than neutrally coloured, and three
/// things here depended on the ambiguity — the flicker detector's ffmpeg decode, a
/// `MediaInput` write-then-read round trip, and a human looking at a render and having to
/// decide whether the colour is the model's fault or the container's.
///
/// ## Frame counts
///
/// The decoder emits frame counts on the `8k+1` lattice (`FrameLattice`): 9, 25, 97, and
/// so on, because the video VAE compresses time causally with one leading frame. This
/// writer does **not** enforce that — it writes whatever stack it is given, because a
/// caller trimming or slicing frames for a diagnostic clip is doing something legitimate
/// and an arithmetic rule from three stages up has no business failing it here. The
/// consequence to know is that duration is exactly `frames / fps`, so a 97-frame render
/// at 24 fps is 4.0416s and not a round number.
///
/// ## The two durations do not have to agree, and this writer will not make them
///
/// Video runs `frames / fps`; audio runs `samples / sampleRate`. Those are two independent
/// integer quotients and they are almost never equal — 97 frames at 24 fps is 4.041666s,
/// and the vocoder emits whatever `melFrames × 160 × 3` samples the audio latent count
/// gave it. **Neither track is padded or trimmed to match the other.** Both are written at
/// their true length and the container holds the ragged edge, which is a thing mp4 is
/// perfectly able to do.
///
/// This is a deliberate refusal, not an omission. A writer that quietly trimmed audio to
/// the video length would erase the visible symptom of a real bug: the audio and video
/// latent counts are derived from the same requested duration by different arithmetic, and
/// getting one of them wrong produces a length disagreement and nothing else. Hiding that
/// would leave the port silently generating the wrong amount of audio for every render.
/// A *gross* disagreement still throws — see ``durationAllowance(forVideo:)`` — because at
/// that point the mismatch is no longer a rounding edge and a muxed file would be actively
/// misleading.
public enum RenderOutput {

    // MARK: - Failures

    /// Every failure names the thing that was wrong, because the alternative — the one
    /// this enum exists to prevent — is an `.mp4` on disk that a later perceptual check
    /// happily consumes and reports a clean result for. A zero-frame render must be an
    /// error and not an empty-but-valid container.
    public enum Failure: Error, CustomStringConvertible {
        case rank(got: [Int])
        case channels(got: Int)
        case noFrames
        case degenerateFrame(width: Int, height: Int)
        case oddDimension(width: Int, height: Int)
        case badFrameRate(Double)
        case destinationNotAFile(URL, kind: String)
        case destinationUnremovable(URL, underlying: Error)
        case destinationDirectoryMissing(URL)
        case writerSetup(String)
        case noPixelBufferPool
        case pixelBufferAllocation(CVReturn)
        case pixelBufferLock(CVReturn)
        case pixelBufferBaseAddress
        case appendRefused(frame: Int, underlying: Error?)
        case inputStalled(frame: Int, seconds: Double)
        case finishFailed(underlying: Error?)
        case audioRank(got: [Int])
        case audioBatch(got: Int)
        case audioChannels(got: Int)
        case noSamples
        case badSampleRate(Int)
        case durationMismatch(video: Double, audio: Double, allowed: Double)
        case audioFormatDescription(OSStatus)
        case audioBlockBuffer(OSStatus)
        case audioSampleBuffer(OSStatus)
        case audioAppendRefused(sample: Int, underlying: Error?)
        case audioInputStalled(sample: Int, seconds: Double)
        case notAWAVExtension(URL)
        case wavSetup(String)

        public var description: String {
            switch self {
            case let .rank(got):
                return "rgb has shape \(got); expected rank 4 [F, H, W, C] as produced by "
                    + "VideoVAEDecoder.rgb. A rank-5 [B, C, F, H, W] is decode()'s own "
                    + "output — call rgb() on it first"
            case let .channels(got):
                return "rgb has \(got) channels; expected 3 (R, G, B). VideoVAEDecoder's "
                    + "out_channels is 3 and the layout is channels-last after rgb()"
            case .noFrames:
                return "rgb has 0 frames; refusing to write an empty container that a "
                    + "later perceptual check would treat as a valid clip"
            case let .degenerateFrame(width, height):
                return "frame is \(width)x\(height); both dimensions must be positive"
            case let .oddDimension(width, height):
                return "frame is \(width)x\(height); H.264/HEVC require even dimensions in "
                    + "4:2:0. A real render is a multiple of the VAE's spatial scale (32) "
                    + "and cannot hit this; a cropped or hand-built stack can"
            case let .badFrameRate(fps):
                return "fps \(fps) is not a positive finite number"
            case let .destinationNotAFile(url, kind):
                return "\(url.path) already exists and is a \(kind), not a regular file; "
                    + "refusing to remove it"
            case let .destinationUnremovable(url, underlying):
                return "could not remove the existing file at \(url.path): \(underlying)"
            case let .destinationDirectoryMissing(url):
                return "the parent directory of \(url.path) does not exist"
            case let .writerSetup(detail):
                return "AVAssetWriter setup failed: \(detail)"
            case .noPixelBufferPool:
                return "AVAssetWriterInputPixelBufferAdaptor has no pixel buffer pool "
                    + "after startWriting; the codec rejected the source pixel format"
            case let .pixelBufferAllocation(status):
                return "CVPixelBufferPoolCreatePixelBuffer failed with \(status)"
            case let .pixelBufferLock(status):
                return "CVPixelBufferLockBaseAddress failed with \(status)"
            case .pixelBufferBaseAddress:
                return "the locked pixel buffer has no base address; it is not a "
                    + "single-plane buffer"
            case let .appendRefused(frame, underlying):
                return "the writer refused frame \(frame)"
                    + (underlying.map { ": \($0)" } ?? " with no error set")
            case let .inputStalled(frame, seconds):
                return "the writer input was not ready for frame \(frame) after "
                    + "\(seconds)s; the encoder is wedged"
            case let .finishFailed(underlying):
                return "finishWriting ended in .failed"
                    + (underlying.map { ": \($0)" } ?? " with no error set")
            case let .audioRank(got):
                return "audio has shape \(got); expected rank 3 [B, 2, samples] as "
                    + "produced by AudioVocoder.waveform(fromMel:)"
            case let .audioBatch(got):
                return "audio has batch \(got); this writer takes one clip at a time, so "
                    + "slice the batch axis yourself rather than have the writer guess "
                    + "which clip belongs with these frames"
            case let .audioChannels(got):
                return "audio has \(got) channels; expected 2 (L, R). Every production "
                    + "vocoder checkpoint is stereo — AudioVocoder.stereoInputChannels is "
                    + "128, i.e. 2 x 64 mel bins — so a mono tensor is a slicing mistake"
            case .noSamples:
                return "audio has 0 samples; refusing to write a silent-but-present track "
                    + "that would look like a successful vocode"
            case let .badSampleRate(rate):
                return "sample rate \(rate) is not positive; the vocoder's "
                    + "config.outputSamplingRate is 48000"
            case let .durationMismatch(video, audio, allowed):
                return "video is \(video)s and audio is \(audio)s, which differ by "
                    + "\(abs(video - audio))s — more than the \(allowed)s this writer will "
                    + "mux without complaint. Neither track is padded or trimmed to fit, so "
                    + "a gap this large means the audio and video latent counts disagree "
                    + "upstream; fix that rather than the file"
            case let .audioFormatDescription(status):
                return "CMAudioFormatDescriptionCreate failed with \(status)"
            case let .audioBlockBuffer(status):
                return "CMBlockBuffer creation for the audio samples failed with \(status)"
            case let .audioSampleBuffer(status):
                return "CMSampleBufferCreate for the audio samples failed with \(status)"
            case let .audioAppendRefused(sample, underlying):
                return "the writer refused the audio chunk starting at sample \(sample)"
                    + (underlying.map { ": \($0)" } ?? " with no error set")
            case let .audioInputStalled(sample, seconds):
                return "the audio input was not ready for the chunk starting at sample "
                    + "\(sample) after \(seconds)s; the encoder is wedged"
            case let .notAWAVExtension(url):
                return "\(url.path) does not end in .wav; AVAudioFile picks the container "
                    + "from the extension, so writing here would silently produce some "
                    + "other format under a name that claims otherwise"
            case let .wavSetup(detail):
                return "AVAudioFile setup failed: \(detail)"
            }
        }
    }

    // MARK: - Spec

    public enum Codec: Sendable {
        case h264
        case hevc

        var avCodec: AVVideoCodecType {
            switch self {
            case .h264: return .h264
            case .hevc: return .hevc
            }
        }
    }

    /// Which range the incoming floats are in.
    ///
    /// ``VideoVAEDecoder/rgb(_:)`` is the `[0, 1]` tail — it applies
    /// `(x + 1) * 0.5` clamped to `[0, 1]` — so ``unitInterval`` is the default and
    /// the case a normal render takes. ``signedUnit`` exists because
    /// ``VideoVAEDecoder/decode(_:)`` on its own leaves the video in the decoder's raw
    /// range, roughly `[-1, 1]`, and a caller poking at an intermediate should be able to
    /// look at it without hand-rolling the affine and getting the direction wrong. It is
    /// an explicit field rather than a range sniff: a dark clip in `[-1, 1]` can easily
    /// have all its values inside `[0, 1]`, so inference would silently mis-expose it.
    public enum InputRange: Sendable {
        case unitInterval
        case signedUnit
    }

    /// What the file says its own colour means.
    ///
    /// **An untagged mp4 is not colourless, it is ambiguously coloured**, and that is a
    /// bug this writer used to have. `AVVideoColorPropertiesKey` was absent, so the file
    /// carried no `ColorPrimaries`, no `TransferFunction` and no `YCbCrMatrix` extension
    /// at all, and encoder and decoder each fell back to their own default. Those defaults
    /// do not agree: `MediaInputTests.colourIsBoundedByTheUntaggedMatrix` measured a pure
    /// red pixel coming back from an untagged round trip as `(0.9176, 0, 0.0078)`, which
    /// is BT.709 in and BT.601 out to three decimals — a mean error of 6.5/255 over a
    /// colour gradient against 0.46/255 for the same clip in luma only. **A reader cannot
    /// recover a matrix the file never states**, so the fix has to be here.
    ///
    /// Tagging closes it. `RenderOutputTests.colourSurvivesTheTaggedRoundTrip` measures the
    /// same gradient through both: mean 7.24/255 untagged against **0.69/255 tagged**, a
    /// 10.5x improvement that lands the colour round trip on top of the achromatic one. The
    /// residual is the codec. The part that was there before was a matrix nobody chose.
    ///
    /// ## What the decoder's values actually are
    ///
    /// ``VideoVAEDecoder/rgb(_:)`` is `(x + 1) / 2` clamped to `[0, 1]` and nothing else —
    /// there is no transfer function anywhere in the decode path. So the `[0, 1]` floats
    /// are **display-referred, transfer-encoded code values**: the same numbers an 8-bit
    /// `rgb24` frame divided by 255 gives, which is how the training tensors were built.
    /// Two details settle it rather than leaving it to omission:
    ///
    /// * an HDR decode is defined as "VAE decode `[F,H,W,C]` in `[0,1]` → scene-linear
    ///   HDR" and has to *apply* an inverse transfer to get there. The VAE's own output is
    ///   therefore not linear.
    /// * the source primaries for every non-ACES working space are Rec.709, and the HLG
    ///   sink treats the VAE signal as Rec.709 before converting to Rec.2020. The gamut is
    ///   709 — which is also sRGB's, since the two share primaries and a D65 white point,
    ///   so this half of the answer is the same whichever name you prefer.
    ///
    /// ## What the SDR format states, and what it leaves unstated
    ///
    /// **The matrix and the range, and deliberately nothing else**: BT.709 colorspace,
    /// limited range, RGB→YUV420p done with BT.709 luma weights (`0.2126, 0.7152, 0.0722`)
    /// and the limited-range `(219·E' + 16)` scaling. Primaries and transfer are never set
    /// on an SDR file; the only path that sets them is HLG, which tags BT.2020 primaries
    /// and ARIB STD-B67.
    ///
    /// So ``rec709`` **matches on the part that is stated** — the matrix — and deliberately
    /// adds the two parts that are not. The addition is not optional:
    /// `AVVideoColorPropertiesKey` is all-or-nothing, all three keys or none.
    ///
    /// ## Confidence, honestly split
    ///
    /// * **Matrix — high.** BT.709 is named in the format and tagged in the container. It
    ///   is a copied value, not a guess, and it is the *only* one of the three that changes
    ///   decoded RGB numbers rather than how a display interprets them.
    /// * **Primaries — high.** Rec.709, and identical to sRGB's regardless.
    /// * **Transfer — medium, and this is the soft one.** Nothing states a curve. The
    ///   training corpus is 8-bit video code values, whose curve is BT.709's
    ///   OETF or sRGB's depending on where each clip came from; the two differ by at most
    ///   a couple of counts and are conflated by most of the industry. If it is wrong,
    ///   what breaks is a colour-managed *display* — a slightly lifted or crushed shadow
    ///   on a wide-gamut screen — and not any number a diff would see. `IEC_sRGB` is not
    ///   offered as an alternative because `AVVideoTransferFunction_IEC_sRGB` is macOS 15
    ///   and this package targets macOS 14.
    ///
    /// ## Range is not settable here and is not tagged by this enum
    ///
    /// The format pins limited range. `AVVideoColorPropertiesKey` has no key for range —
    /// it rides in H.264's `video_full_range_flag` — and AVFoundation does
    /// not expose it in output settings, so this writer takes VideoToolbox's default.
    /// `RenderOutputTests.taggedTrackCarriesTheColourAttachments` prints what the format
    /// description reports so the assumption is at least observable; it is **not asserted**,
    /// because it is not a thing this file can control.
    public enum ColourTagging: Sendable {
        /// Rec.709 primaries, transfer and matrix. The default, and what any render meant
        /// to be compared with another should use.
        case rec709

        /// The SD bundle: SMPTE-C primaries, Rec.709 transfer, **BT.601 matrix**.
        ///
        /// Here because BT.601 is what a consumer applies to an untagged file when left to
        /// its own devices — VideoToolbox's decoder demonstrably does (that is the 6.5/255
        /// measurement above), and libswscale's `SWS_CS_DEFAULT` is ITU601, which is what
        /// any writer that tags nothing gets by omission on both the encode and the decode
        /// side. This case lets a caller comparing against such files **state** that
        /// assumption instead of inheriting it.
        ///
        /// Two things to know. It is **not** the same file as ``untagged``: tagging 601
        /// makes VideoToolbox *encode* with 601 too, where an untagged write encodes 709
        /// and only the reader disagrees. And the primaries come along for the ride —
        /// `AVVideoSettings.h` sanctions SMPTE-C/709/601 as a bundle and this does not
        /// deviate from it, on the grounds that an unsanctioned mixture is a combination
        /// AVFoundation is free to reject. Only the matrix moves any number.
        ///
        /// The libswscale half of that justification is read from its documented default
        /// and **was not measured here**; the VideoToolbox half was.
        case rec601

        /// Tag nothing — the behaviour this writer had before colour tagging existed.
        ///
        /// Kept reachable on purpose. Comparing renders made months apart means being able
        /// to reproduce an old file exactly, and "the writer changed under me" is precisely
        /// the confound that makes such a comparison worthless. It is the wrong choice for
        /// anything new: see the type comment for what it costs.
        case untagged

        /// The `AVVideoColorPropertiesKey` payload, or `nil` to leave the settings
        /// dictionary alone entirely. `nil` and an empty dictionary are not the same thing
        /// — an empty one is rejected, because AVFoundation requires all three keys.
        var properties: [String: String]? {
            switch self {
            case .untagged:
                return nil
            case .rec709:
                return [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
                ]
            case .rec601:
                return [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_SMPTE_C,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_601_4,
                ]
            }
        }
    }

    public struct VideoSpec: Sendable {
        public var fps: Double
        public var codec: Codec
        public var range: InputRange
        /// Average bit rate in bits per second. `nil` leaves VideoToolbox's own default,
        /// which is quality-reasonable but resolution-dependent; pin it when a test or a
        /// perceptual comparison needs two renders to be encoded alike.
        public var averageBitRate: Int?
        /// What the container says its colour means. Defaults to ``ColourTagging/rec709``;
        /// ``ColourTagging/untagged`` reproduces this writer's pre-tagging output.
        public var colour: ColourTagging

        public init(fps: Double = 24,
                    codec: Codec = .h264,
                    range: InputRange = .unitInterval,
                    averageBitRate: Int? = nil,
                    colour: ColourTagging = .rec709) {
            self.fps = fps
            self.codec = codec
            self.range = range
            self.averageBitRate = averageBitRate
            self.colour = colour
        }
    }

    public enum AudioCodec: Sendable {
        case aac
        /// Uncompressed 32-bit float, interleaved — the same thing ``writeWAV(audio:to:sampleRate:)``
        /// writes, but muxed alongside the video.
        ///
        /// mp4 does carry `lpcm`, and AVFoundation writes and reads it back bit-exactly
        /// here (checked, not assumed — `audioSurvivesAnLPCMRoundTripExactly`). The cost is
        /// size: 48 kHz stereo float is 384 kB/s, roughly 30× AAC, so this is for a clip
        /// something is going to *measure* rather than one anybody is going to ship.
        case lpcm

        var formatID: AudioFormatID {
            switch self {
            case .aac: return kAudioFormatMPEG4AAC
            case .lpcm: return kAudioFormatLinearPCM
            }
        }
    }

    public struct AudioSpec: Sendable {
        /// 48000 — ``AudioVocoder/Configuration/outputSamplingRate``. It is a parameter and
        /// not a constant because the vocoder also exposes its 16 kHz intermediate
        /// (`baseRateWaveform(fromMel:)`), and listening to that is a legitimate way to
        /// hear whether the bandwidth-extension stage is doing anything.
        public var sampleRate: Int
        public var codec: AudioCodec
        /// Bits per second for ``AudioCodec/aac``; ignored for ``AudioCodec/lpcm``, whose
        /// rate is fixed by the format. `nil` leaves the encoder's own default.
        public var averageBitRate: Int?

        public init(sampleRate: Int = 48_000,
                    codec: AudioCodec = .aac,
                    averageBitRate: Int? = nil) {
            self.sampleRate = sampleRate
            self.codec = codec
            self.averageBitRate = averageBitRate
        }
    }

    /// Stereo, everywhere. Named rather than spelled `2` so the places that depend on it —
    /// the byte stride, the channel layout tag, the validation — are greppable together.
    static let audioChannels = 2

    /// Interleaved 32-bit float: `L, R, L, R`. 8 bytes per frame.
    static let audioBytesPerFrame = audioChannels * MemoryLayout<Float>.size

    /// Per-channel frames per `CMSampleBuffer`. 4096 at 48 kHz is 85 ms, which is small
    /// enough that the video input never waits long behind a single audio append and large
    /// enough that a 5-second render is ~60 buffers rather than thousands.
    static let audioChunkFrames = 4096

    /// How far the two track durations may drift before ``Failure/durationMismatch(video:audio:allowed:)``.
    ///
    /// `max(0.25s, 10%)`, and the two terms cover different failures. The 10% term is what
    /// catches the interesting bug — an audio latent count that is a *factor* off (2×, 3×,
    /// the 160-vs-240 upsample confusion ``AudioVocoder`` warns about) scales with the clip
    /// and would slip under any fixed number on a long render. The 0.25s floor exists
    /// because 10% of a 9-frame test clip is 37 ms, which is tighter than the honest
    /// granularity of the thing being compared: audio arrives in whole mel frames worth
    /// 480 samples each, so a legitimate render can land a good fraction of a second either
    /// side of the video length with nothing wrong at all.
    static func durationAllowance(forVideo seconds: Double) -> Double {
        max(0.25, 0.10 * seconds)
    }

    /// How long to wait on a stalled encoder before declaring it wedged. Generous: a
    /// hardware encoder under contention can legitimately hold back for seconds, and a
    /// false positive here throws away a completed render.
    private static let readinessTimeout: TimeInterval = 60

    /// How long to sleep between readiness polls. `AVAssetWriterInput` offers
    /// `requestMediaDataWhenReady`, but that inverts control flow for what is a
    /// synchronous, already-materialised frame stack; a 2 ms sleep costs at most a
    /// handful of milliseconds across a 97-frame render and, unlike a bare `while
    /// !isReadyForMoreMediaData {}`, does not pin a core against the encoder that is
    /// trying to drain.
    private static let readinessPoll: TimeInterval = 0.002

    // MARK: - Writing

    /// Write decoded RGB frames to an `.mp4` at `url`.
    ///
    /// `rgb` is `[F, H, W, C]` channels-last with `C == 3`, which is exactly what
    /// ``VideoVAEDecoder/rgb(_:)`` returns:
    ///
    /// ```swift
    /// let video = decoded[0].transposed(1, 2, 3, 0)       // c f h w -> f h w c
    /// ```
    ///
    /// Note the `decoded[0]`: the batch axis is already gone by then, everything but clip 0
    /// having been dropped at the same point. A rank-5 tensor arriving here is `decode()`'s
    /// output that never went through `rgb()`, and that is what ``Failure/rank(got:)`` says
    /// rather than "bad shape".
    ///
    /// Any dtype is accepted; the decoder's tail stays in bf16 deliberately (its own doc
    /// explains why widening it would be *less* faithful), and this widens to fp32 for the
    /// byte conversion. That widening is lossless and happens after every value that
    /// matters has already been produced.
    ///
    /// The call blocks until the file is finalised on disk.
    ///
    /// The result is **silent**. This is still the right call for a video-only render and
    /// for every test that has no waveform to attach; reach for
    /// ``write(rgb:audio:to:spec:audioSpec:)`` when there is one.
    public static func writeVideo(rgb: MLXArray, to url: URL,
                                  spec: VideoSpec = VideoSpec()) throws {
        try writeTracks(rgb: rgb, audio: nil, to: url, spec: spec, audioSpec: AudioSpec())
    }

    /// Write decoded RGB frames and a vocoder waveform to one `.mp4` at `url`.
    ///
    /// `audio` is `[B, 2, samples]` — batch, stereo channel, sample — which is exactly what
    /// ``AudioVocoder/waveform(fromMel:)`` returns: it computes in `[B, L, C]` internally
    /// and transposes to `[B, C, L]` on the way out, so the middle axis is the channel and
    /// the last one is time. Getting that backwards is the reason ``Failure/audioChannels(got:)``
    /// exists and is checked before anything is written — a `[1, samples, 2]` tensor is
    /// well-formed, plausible, and would be read as a 2-sample stereo clip of `samples`
    /// channels.
    ///
    /// `B` must be 1. A batch axis survives this far because the vocoder is happy to run a
    /// batch, but a file holds one clip, and picking a clip for the caller is the kind of
    /// helpfulness that silently writes the wrong take.
    ///
    /// Any float dtype is accepted and widened to fp32 — though unlike the video tail this
    /// costs nothing in practice, because ``AudioVocoder`` runs the whole model in fp32
    /// already for reasons its own doc comment sets out.
    ///
    /// The two tracks are written at their true, generally unequal, lengths; see the type
    /// comment for why nothing is padded or trimmed to make them match.
    public static func write(rgb: MLXArray, audio: MLXArray, to url: URL,
                             spec: VideoSpec = VideoSpec(),
                             audioSpec: AudioSpec = AudioSpec()) throws {
        try writeTracks(rgb: rgb, audio: audio, to: url, spec: spec, audioSpec: audioSpec)
    }

    private static func writeTracks(rgb: MLXArray, audio: MLXArray?, to url: URL,
                                    spec: VideoSpec, audioSpec: AudioSpec) throws {
        guard rgb.ndim == 4 else { throw Failure.rank(got: rgb.shape) }
        let frames = rgb.shape[0], height = rgb.shape[1]
        let width = rgb.shape[2], channels = rgb.shape[3]
        guard channels == 3 else { throw Failure.channels(got: channels) }
        guard frames > 0 else { throw Failure.noFrames }
        guard width > 0, height > 0 else {
            throw Failure.degenerateFrame(width: width, height: height)
        }
        guard width % 2 == 0, height % 2 == 0 else {
            throw Failure.oddDimension(width: width, height: height)
        }
        guard spec.fps.isFinite, spec.fps > 0 else { throw Failure.badFrameRate(spec.fps) }

        // Both plans are built, and the durations compared, before the destination is
        // touched: a rejected render must not have unlinked the previous good one.
        let audio = try audio.map { try plan(forAudio: $0, spec: audioSpec) }
        let videoSeconds = Double(frames) / spec.fps
        if let audio {
            let allowed = durationAllowance(forVideo: videoSeconds)
            guard abs(videoSeconds - audio.seconds) <= allowed else {
                throw Failure.durationMismatch(video: videoSeconds, audio: audio.seconds,
                                               allowed: allowed)
            }
        }

        try clearDestination(url)

        // A timescale of `round(fps * 1000)` makes one frame exactly 1000 ticks, so the
        // presentation times are integers and the total duration is exactly `F / fps` —
        // including for the broadcast rates that are not integers at all (23.976 becomes
        // 23976/1000, not a float that drifts a frame over a long clip).
        let timescale = CMTimeScale((spec.fps * 1000).rounded())
        let frameTicks = CMTimeValue(1000)

        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            throw Failure.writerSetup("\(error)")
        }

        var settings: [String: Any] = [
            AVVideoCodecKey: spec.codec.avCodec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        if let bitRate = spec.averageBitRate {
            settings[AVVideoCompressionPropertiesKey] = [AVVideoAverageBitRateKey: bitRate]
        }
        // Colour goes on the *output settings* and not on the pixel buffers. Both routes
        // exist, and this one is the one that cannot silently resample: when a source
        // buffer carries colour attachments that disagree with `AVVideoColorPropertiesKey`,
        // the writer is documented to convert between them — i.e. to change pixel values.
        // The pool's buffers carry no attachments at all, so there is nothing to disagree
        // with and nothing to convert, and this is a pure metadata change.
        //
        // That is a claim rather than a hope, so it is checked:
        // `RenderOutputTests.taggingChangesTheContainerAndNotThePixels` writes the same
        // frames tagged and untagged and compares the compressed H.264 sample data. It is
        // not quite a byte-identity check, because VideoToolbox's encoder turns out not to
        // be reproducible run to run — measured at 1 byte in 3538 between two *identical*
        // untagged writes — so the test measures that noise floor on the same run and
        // requires the tagged-vs-untagged distance to sit inside it. It does: 1–2 bytes,
        // at the same offset. If VideoToolbox ever starts converting, that goes red rather
        // than the render quietly shifting.
        if let colour = spec.colour.properties {
            settings[AVVideoColorPropertiesKey] = colour
        }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        // False, because the frames already exist. With this true the writer will drop
        // data it cannot keep up with — correct for a camera, silently lossy for a render.
        input.expectsMediaDataInRealTime = false

        // `kCVPixelBufferIOSurfacePropertiesKey` is not decoration: without an
        // IOSurface-backed buffer VideoToolbox's hardware encoder will not take the frame
        // and the append fails with a format error that names nothing useful.
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: attributes)

        guard writer.canAdd(input) else {
            throw Failure.writerSetup("canAdd(input) is false for \(spec.codec) at "
                + "\(width)x\(height)")
        }
        writer.add(input)

        var audioInput: AVAssetWriterInput?
        var audioFormat: CMAudioFormatDescription?
        if audio != nil {
            let settings = audioSettings(audioSpec)
            let track = AVAssetWriterInput(mediaType: .audio, outputSettings: settings)
            track.expectsMediaDataInRealTime = false
            guard writer.canAdd(track) else {
                throw Failure.writerSetup("canAdd(audio) is false for \(audioSpec.codec) at "
                    + "\(audioSpec.sampleRate) Hz, \(audioChannels) channels")
            }
            writer.add(track)
            audioInput = track
            audioFormat = try audioSourceFormat(sampleRate: audioSpec.sampleRate)
        }

        guard writer.startWriting() else {
            throw Failure.writerSetup("startWriting returned false: "
                + (writer.error.map { "\($0)" } ?? "no error set"))
        }
        writer.startSession(atSourceTime: .zero)

        // The adaptor's pool only exists once the session has started, and using it rather
        // than a hand-rolled `CVPixelBufferPoolCreate` is what guarantees the buffers match
        // what the codec was configured for — including alignment and stride, which is why
        // `fill` below copies row by row instead of assuming `width * 4`.
        guard let pool = adaptor.pixelBufferPool else { throw Failure.noPixelBufferPool }

        // Both tracks are drained by AVAssetWriter's PULL model — `requestMediaDataWhenReady`
        // on a serial queue per input — and NOT by polling `isReadyForMoreMediaData` from
        // this thread. That is the difference between a file and a hang, and it was measured
        // rather than reasoned about.
        //
        // A standalone AVFoundation probe sharing no code with this port: with ONE input,
        // polling drains all 97 frames every time. Add a second input and the video track
        // stops being reported ready and never recovers — at frame 34, 37, 39, 40, 47, 85 or
        // 88 depending on chunk size and timing. It is a **race**, which is why this shipped
        // green for a while and then wedged a render that had already cost nine minutes of
        // sampling. The input COUNT is the whole variable: shape makes no difference (640x384
        // wedges exactly like 288x512), nor does codec (H.264, HEVC, AAC and LPCM all wedge),
        // nor IOSurface backing, nor chunk size — the last only moves where it dies. The same
        // probe in the pull model completes at both shapes.
        //
        // Interleaving by presentation time is now the writer's job rather than ours, and it
        // is better at it: each callback drains its own input for as long as that input stays
        // ready, so both tracks advance without this code arbitrating between them.
        let audioFrames = audio?.frames ?? 0
        let videoDrain = Drain()
        let audioDrain = Drain()
        let group = DispatchGroup()

        group.enter()
        input.requestMediaDataWhenReady(on: DispatchQueue(label: "ltx.render.video")) {
            while input.isReadyForMoreMediaData {
                let i = videoDrain.index
                guard i < frames else { videoDrain.finish(input, group); return }
                do {
                    var buffer: CVPixelBuffer?
                    let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
                    guard status == kCVReturnSuccess, let pixels = buffer else {
                        throw Failure.pixelBufferAllocation(status)
                    }

                    let bytes = bgraBytes(frame: rgb[i], range: spec.range)
                    try fill(pixels, with: bytes, width: width, height: height)

                    let time = CMTime(value: frameTicks * CMTimeValue(i), timescale: timescale)
                    guard adaptor.append(pixels, withPresentationTime: time) else {
                        throw Failure.appendRefused(frame: i, underlying: writer.error)
                    }
                    videoDrain.advance(1)
                } catch {
                    videoDrain.fail(error, input, group)
                    return
                }
            }
        }

        if let track = audioInput, let format = audioFormat, let audio {
            group.enter()
            track.requestMediaDataWhenReady(on: DispatchQueue(label: "ltx.render.audio")) {
                while track.isReadyForMoreMediaData {
                    let start = audioDrain.index
                    guard start < audioFrames else { audioDrain.finish(track, group); return }
                    do {
                        let count = min(audioChunkFrames, audioFrames - start)
                        let sample = try audioSampleBuffer(
                            audio.interleaved, frames: start ..< (start + count),
                            sampleRate: audioSpec.sampleRate, format: format)
                        guard track.append(sample) else {
                            throw Failure.audioAppendRefused(sample: start,
                                                             underlying: writer.error)
                        }
                        audioDrain.advance(count)
                    } catch {
                        audioDrain.fail(error, track, group)
                        return
                    }
                }
            }
        }

        // A STALL detector, not a total budget. A 20-second render at 1152x2048 legitimately
        // takes minutes to encode, so a fixed deadline would either fail honest work or wait
        // out a real wedge for just as long. This watches progress instead: as long as either
        // track is still advancing, it keeps waiting, and it gives up only when neither has
        // moved for `readinessTimeout`. The pull model is what AVAssetWriter documents, but
        // "documented" is not "cannot wedge", and the failure this replaced is exactly the
        // kind that a hopeful unbounded wait turns into an overnight hang.
        var lastProgress = (-1, -1)
        var idleSince = Date()
        while group.wait(timeout: .now() + 0.25) == .timedOut {
            let now = (videoDrain.index, audioDrain.index)
            if now != lastProgress {
                lastProgress = now
                idleSince = Date()
            } else if Date().timeIntervalSince(idleSince) > readinessTimeout {
                writer.cancelWriting()
                throw Failure.inputStalled(frame: videoDrain.index, seconds: readinessTimeout)
            }
        }

        // Leaving a half-written session open holds the encoder and the partial file. The
        // drain's error is the interesting one, so the cancellation itself is silent.
        if let error = videoDrain.failure ?? audioDrain.failure {
            writer.cancelWriting()
            throw error
        }
        // `finishWriting` is asynchronous, and returning before it completes is the classic
        // way to hand back a truncated `.mp4` that opens, reports a duration, and is missing
        // its last frames — or its moov atom entirely. Block on it.
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()

        guard writer.status == .completed else {
            throw Failure.finishFailed(underlying: writer.error)
        }
    }

    // MARK: - Destination

    /// Remove an existing file at `url`, but only if it really is a plain file.
    ///
    /// `AVAssetWriter` refuses to start when its output URL exists, so something has to
    /// unlink it. `attributesOfItem` uses `lstat`, so it reports the *symlink* rather than
    /// its target: a `--out` that happens to point at a symlink into a checkout, or at a
    /// directory, is a typo, and the damage from unlinking it silently is unbounded and
    /// unrecoverable. Only `.typeRegular` is removed; anything else is an error naming what
    /// it found.
    private static func clearDestination(_ url: URL) throws {
        let manager = FileManager.default
        let parent = url.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        guard manager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory),
              parentIsDirectory.boolValue else {
            throw Failure.destinationDirectoryMissing(url)
        }

        guard let attributes = try? manager.attributesOfItem(atPath: url.path) else {
            return                                          // nothing there; nothing to do
        }
        let type = (attributes[.type] as? FileAttributeType) ?? .typeUnknown
        guard type == .typeRegular else {
            throw Failure.destinationNotAFile(url, kind: describe(type))
        }
        do {
            try manager.removeItem(at: url)
        } catch {
            throw Failure.destinationUnremovable(url, underlying: error)
        }
    }

    private static func describe(_ type: FileAttributeType) -> String {
        switch type {
        case .typeDirectory: return "directory"
        case .typeSymbolicLink: return "symbolic link"
        case .typeSocket: return "socket"
        case .typeCharacterSpecial: return "character device"
        case .typeBlockSpecial: return "block device"
        default: return "non-regular file"
        }
    }

    // MARK: - Pixels

    /// One frame `[H, W, 3]` → a contiguous `H * W * 4` BGRA byte buffer.
    ///
    /// **The tensor crosses to the CPU exactly once, already as bytes.** Every step —
    /// affine, clamp, scale, round, channel reorder, alpha — happens inside MLX, and the
    /// single `asArray(UInt8.self)` pulls the finished buffer. The alternative that this
    /// avoids is indexing the MLXArray per pixel: a production frame is 640×384 and a
    /// production render is 97 of them, so per-pixel indexing is 71 million individual
    /// lazy-graph evaluations and turns a seconds-long encode into an overnight one.
    /// Measured at that shape — 97 frames of 640×384, `swift build` debug, M-series — the
    /// whole `writeVideo` call including the H.264 encode takes **0.27 s**. That number is
    /// from a throwaway probe rather than a committed test: a wall-clock assertion is a
    /// flaky test on a shared machine, and the shape of the bug it would guard against
    /// (orders of magnitude, not percent) is not one that needs a tight bound to notice.
    ///
    /// **Clamping, not rescaling.** The decoder's output can legitimately overshoot —
    /// `rgb()` clamps for that reason, the VAE having no `tanh_out` and nothing else
    /// bounding it. Rescaling by the observed extrema
    /// instead would make every pixel in the clip a function of its brightest one, so a
    /// single blown highlight would dim the entire render, and two clips of the same scene
    /// would be exposed differently — which is exactly the kind of global shift a
    /// perceptual or flicker check is supposed to be measuring, not receiving as an
    /// artefact of the writer. Clamping is local and idempotent: only the pixels that were
    /// already out of range move, and they move to the nearest representable value.
    ///
    /// The clamp must also precede the `uint8` cast rather than trailing it. MLX's
    /// float→uint8 conversion truncates the C way, so `-0.5 * 255` and `1.5 * 255` would
    /// wrap to mid-grey rather than saturating — a negative value would come out *bright*.
    /// `RenderOutputTests.outOfRangeValuesClampExactly` is the control for that.
    static func bgraBytes(frame: MLXArray, range: InputRange) -> [UInt8] {
        var unit = frame.asType(.float32)
        if case .signedUnit = range {
            unit = (unit + 1) * 0.5
        }
        // `+ 0.5` then truncate is a round-half-up; MLX's `round` is round-half-to-even,
        // which would map 0.5/255 and 1.5/255 to the same byte. Neither is wrong, but
        // half-up is what every other 8-bit image path does and what a hand-computed
        // expected value in a test will assume.
        let scaled = MLX.clip(unit, min: 0, max: 1) * 255 + 0.5
        let bytes = MLX.clip(scaled, min: 0, max: 255).asType(.uint8)

        let h = frame.shape[0], w = frame.shape[1]
        // 32BGRA is byte order B, G, R, A in memory, so the channel axis is reversed and
        // an opaque alpha appended. Doing it as one `concatenated` keeps this a single
        // device-side op and a single readback.
        let b = bytes[0..., 0..., 2 ..< 3]
        let g = bytes[0..., 0..., 1 ..< 2]
        let r = bytes[0..., 0..., 0 ..< 1]
        let a = MLXArray.full([h, w, 1], values: MLXArray(UInt8(255)), dtype: .uint8)
        return MLX.concatenated([b, g, r, a], axis: -1).asArray(UInt8.self)
    }

    /// Copy a packed BGRA buffer into a pool-allocated `CVPixelBuffer`.
    ///
    /// Row by row, because `CVPixelBufferGetBytesPerRow` is not `width * 4`: the pool pads
    /// rows to the encoder's alignment (64 bytes is typical), and a flat `memcpy` of
    /// `width * height * 4` into a padded buffer produces the diagonal-shear image that
    /// looks like a decoder bug and is not one.
    private static func fill(_ buffer: CVPixelBuffer, with bytes: [UInt8],
                             width: Int, height: Int) throws {
        let lock = CVPixelBufferLockBaseAddress(buffer, [])
        guard lock == kCVReturnSuccess else { throw Failure.pixelBufferLock(lock) }
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw Failure.pixelBufferBaseAddress
        }
        let destinationStride = CVPixelBufferGetBytesPerRow(buffer)
        let sourceStride = width * 4
        bytes.withUnsafeBufferPointer { source in
            guard let start = source.baseAddress else { return }
            for row in 0 ..< height {
                memcpy(base.advanced(by: row * destinationStride),
                       UnsafeRawPointer(start.advanced(by: row * sourceStride)),
                       sourceStride)
            }
        }
    }

    /// How far one writer input has got, and why it stopped.
    ///
    /// `requestMediaDataWhenReady` hands its block a serial queue and calls it there, so an
    /// input's own callback never races itself — but this type is also read from the thread
    /// waiting on the group, and capturing plain `var`s in an escaping `@Sendable` block is
    /// not expressible at all under the Swift 6 language mode this package builds in. Hence
    /// a box, one per input, with a lock rather than an `@unchecked` claim that the reads are
    /// somehow fine. A hundred-odd uncontended acquisitions per render is not a cost worth
    /// reasoning about; a torn read that made a stall detector fire early would be.
    private final class Drain: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        private var error: Error?
        private var done = false

        /// Frames (video) or per-channel samples (audio) appended so far.
        var index: Int { lock.withLock { count } }
        var failure: Error? { lock.withLock { error } }

        func advance(_ n: Int) { lock.withLock { count += n } }

        /// Idempotent, because it has to be: `requestMediaDataWhenReady`'s block can be
        /// re-entered after the input has been finished, and a second `group.leave()` traps
        /// rather than returning an error.
        func finish(_ input: AVAssetWriterInput, _ group: DispatchGroup) {
            let first: Bool = lock.withLock {
                if done { return false }
                done = true
                return true
            }
            guard first else { return }
            input.markAsFinished()
            group.leave()
        }

        /// Record the first failure and stop this input. The first is kept rather than the
        /// last because a wedged encoder produces a cascade, and the head of it is the one
        /// that says what actually went wrong.
        func fail(_ failure: Error, _ input: AVAssetWriterInput, _ group: DispatchGroup) {
            lock.withLock { if error == nil { error = failure } }
            finish(input, group)
        }
    }

    // MARK: - Samples

    /// A validated waveform, already interleaved and already on the CPU.
    struct AudioPlan {
        /// Per-channel sample count. **Not** the length of ``interleaved``, which is twice
        /// this; the two are easy to confuse and the confusion writes a file at half rate.
        let frames: Int
        /// `L, R, L, R, …`, `frames * 2` long.
        let interleaved: [Float]
        let seconds: Double
    }

    /// Validate `[1, 2, samples]` and flatten it to interleaved fp32 in one readback.
    ///
    /// **The tensor crosses to the CPU exactly once, already interleaved** — the same
    /// discipline as ``bgraBytes(frame:range:)`` and for the same reason: a 5-second stereo
    /// render is half a million samples, and pulling them out of MLX one index at a time is
    /// half a million lazy-graph evaluations.
    ///
    /// The interleave itself is a `stacked` of the two channel rows along a new trailing
    /// axis, giving `[samples, 2]` — pairs, in memory order, which *is* the interleaved
    /// layout. It is written that way rather than as `transposed(1, 0)` because `stacked`
    /// materialises a contiguous result, while a transpose is a stride view whose flat
    /// readback would be a question about MLX's internals rather than an obvious fact. The
    /// distinction matters more than it looks: a readback that ignored the strides would
    /// hand back `L L L … R R R`, which is not silence and not noise — it is the left
    /// channel played at double speed followed by the right, i.e. audio that plays.
    ///
    /// **Clamping, not scaling**, for the reason the BGRA path clamps. `waveform(fromMel:)`
    /// already applies `clip(-1, 1)` at the bandwidth-extension mix, so on the normal path
    /// this is a no-op — but `baseRateWaveform(fromMel:)` does not, and no
    /// generator in that model has a bounded output stage (`use_tanh_at_final` is false),
    /// so an overshoot is an expected value rather than an exceptional one. Left alone it
    /// does not merely sound loud: the float→int16 conversion inside any consumer that
    /// narrows the samples wraps, so a peak just over `+1.0` comes back as a full-scale
    /// *negative* sample — an audible click exactly at the loudest moment. Normalising by
    /// the observed peak instead would make the whole clip's level a function of its single
    /// worst sample, which is the same global-shift artefact the video path refuses.
    static func plan(forAudio audio: MLXArray, spec: AudioSpec) throws -> AudioPlan {
        guard audio.ndim == 3 else { throw Failure.audioRank(got: audio.shape) }
        guard audio.shape[0] == 1 else { throw Failure.audioBatch(got: audio.shape[0]) }
        guard audio.shape[1] == audioChannels else {
            throw Failure.audioChannels(got: audio.shape[1])
        }
        let frames = audio.shape[2]
        guard frames > 0 else { throw Failure.noSamples }
        guard spec.sampleRate > 0 else { throw Failure.badSampleRate(spec.sampleRate) }

        let clamped = MLX.clip(audio[0].asType(.float32), min: -1, max: 1)
        let pairs = MLX.stacked([clamped[0], clamped[1]], axis: -1)      // [frames, 2]
        return AudioPlan(frames: frames,
                         interleaved: pairs.asArray(Float.self),
                         seconds: Double(frames) / Double(spec.sampleRate))
    }

    private static func audioSettings(_ spec: AudioSpec) -> [String: Any] {
        switch spec.codec {
        case .aac:
            var settings: [String: Any] = [
                AVFormatIDKey: spec.codec.formatID,
                AVNumberOfChannelsKey: audioChannels,
                AVSampleRateKey: spec.sampleRate,
            ]
            if let bitRate = spec.averageBitRate {
                settings[AVEncoderBitRateKey] = bitRate
            }
            return settings
        case .lpcm:
            // Identical to the source format, so the writer copies rather than converts.
            // `IsNonInterleaved: false` is stated rather than left to the default for the
            // same reason the source ASBD omits `kAudioFormatFlagIsNonInterleaved`
            // explicitly: this is the one property of the whole audio path that fails
            // silently and audibly, so it is spelled out at every layer that can express it.
            return [
                AVFormatIDKey: spec.codec.formatID,
                AVNumberOfChannelsKey: audioChannels,
                AVSampleRateKey: spec.sampleRate,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ]
        }
    }

    /// The format the `CMSampleBuffer`s are *in* — always interleaved fp32 LPCM, whatever
    /// the output codec is. The AAC encoder sits behind the writer input and takes PCM.
    ///
    /// The channel layout is attached rather than inferred. A bare stereo ASBD does default
    /// to L-then-R everywhere in practice, but "in practice" is not a thing to rely on for
    /// the one property whose failure mode is a file that plays perfectly with the channels
    /// swapped, so `kAudioChannelLayoutTag_Stereo` says it out loud.
    private static func audioSourceFormat(sampleRate: Int) throws -> CMAudioFormatDescription {
        var asbd = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            // No `kAudioFormatFlagIsNonInterleaved`: its absence *is* the claim that the
            // frames are L, R, L, R.
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(audioBytesPerFrame),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(audioBytesPerFrame),
            mChannelsPerFrame: UInt32(audioChannels),
            mBitsPerChannel: 32,
            mReserved: 0)
        var layout = AudioChannelLayout()
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo

        var description: CMAudioFormatDescription?
        let status = CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault, asbd: &asbd,
            layoutSize: MemoryLayout<AudioChannelLayout>.size, layout: &layout,
            magicCookieSize: 0, magicCookie: nil, extensions: nil,
            formatDescriptionOut: &description)
        guard status == noErr, let description else {
            throw Failure.audioFormatDescription(status)
        }
        return description
    }

    /// One chunk of interleaved samples → a `CMSampleBuffer` the writer input will take.
    ///
    /// `frames` is a range of **per-channel** sample indices; it is doubled here to slice
    /// the interleaved array, which is the whole reason ``AudioPlan/frames`` is documented
    /// as not being the array's length.
    ///
    /// The block buffer is allocated by CoreMedia (`memoryBlock: nil`) and then filled,
    /// rather than wrapping the Swift array's storage. Wrapping would be one copy cheaper
    /// and would hand CoreMedia a pointer whose lifetime is an `Array`'s — the buffer
    /// outlives this call by design, sitting in the writer's queue, so that is a
    /// use-after-free that shows up as intermittent noise rather than a crash.
    private static func audioSampleBuffer(_ interleaved: [Float], frames: Range<Int>,
                                          sampleRate: Int,
                                          format: CMAudioFormatDescription) throws
        -> CMSampleBuffer {
        let byteCount = frames.count * audioBytesPerFrame

        var block: CMBlockBuffer?
        var status = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: byteCount,
            blockAllocator: kCFAllocatorDefault, customBlockSource: nil,
            offsetToData: 0, dataLength: byteCount, flags: 0, blockBufferOut: &block)
        guard status == noErr, let block else { throw Failure.audioBlockBuffer(status) }

        status = interleaved.withUnsafeBufferPointer { source -> OSStatus in
            guard let start = source.baseAddress else { return kCMBlockBufferBadPointerParameterErr }
            return CMBlockBufferReplaceDataBytes(
                with: start.advanced(by: frames.lowerBound * audioChannels),
                blockBuffer: block, offsetIntoDestination: 0, dataLength: byteCount)
        }
        guard status == noErr else { throw Failure.audioBlockBuffer(status) }

        // The timescale is the sample rate, so every presentation time is an exact integer
        // sample index. Using the video's timescale here would round each chunk's start to
        // the nearest tick and accumulate a drift against a track that has none.
        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
            presentationTimeStamp: CMTime(value: CMTimeValue(frames.lowerBound),
                                          timescale: CMTimeScale(sampleRate)),
            decodeTimeStamp: .invalid)
        // `sampleCount` is frames, and the single size entry is the size of *one* frame:
        // the duration above is likewise per-frame. Passing the chunk's totals instead
        // produces a buffer describing one enormous sample, which appends without
        // complaint and lands the timeline in the wrong place.
        var size = audioBytesPerFrame
        var sample: CMSampleBuffer?
        status = CMSampleBufferCreate(
            allocator: kCFAllocatorDefault, dataBuffer: block, dataReady: true,
            makeDataReadyCallback: nil, refcon: nil, formatDescription: format,
            sampleCount: CMItemCount(frames.count), sampleTimingEntryCount: 1,
            sampleTimingArray: &timing, sampleSizeEntryCount: 1, sampleSizeArray: &size,
            sampleBufferOut: &sample)
        guard status == noErr, let sample else { throw Failure.audioSampleBuffer(status) }
        return sample
    }

    // MARK: - WAV

    /// Write the waveform alone to an uncompressed `.wav` at `url`.
    ///
    /// This exists because the muxed track is the wrong thing to reason about. AAC is
    /// lossy, and a listener who hears something odd in the render has no way to tell a
    /// vocoder artefact from a codec one; anything that *measures* the audio — a spectrum,
    /// a level, a null test against a second run — must not be looking at an encoder's
    /// output either. So this path has no encoder in it at all.
    ///
    /// **32-bit float, not 16-bit.** The point of the file is exactness: at fp32 the
    /// samples on disk are the samples the vocoder produced, bit for bit, so a round trip
    /// through this function is an identity and any difference a tool reports is real.
    /// 16-bit would be more universally playable and would quantise every sample by up to
    /// 1/65536 — small, but not zero, and "small" is not a useful thing to be when the
    /// question is whether two waveforms are the same. Everything that plays or analyses
    /// WAV handles float32.
    ///
    /// The samples are clamped exactly as the muxed path clamps them; see
    /// ``plan(forAudio:spec:)``.
    ///
    /// As with the video writer, the call blocks until the file is complete on disk — but
    /// the mechanism is different and worth naming, because it is not obvious and it is not
    /// checkable by inspection: `AVAudioFile` has no `close()`. It finalises the RIFF header
    /// when it deallocates, so the write happens inside a nested function whose scope ends
    /// while the reference is still the only one. A version of this that kept the file alive
    /// until the end of an enclosing scope would return to a caller who could then read back
    /// zero frames from a file that exists and has a plausible size.
    public static func writeWAV(audio: MLXArray, to url: URL, sampleRate: Int) throws {
        let spec = AudioSpec(sampleRate: sampleRate, codec: .lpcm)
        let plan = try plan(forAudio: audio, spec: spec)

        let ext = url.pathExtension.lowercased()
        guard ext == "wav" || ext == "wave" else { throw Failure.notAWAVExtension(url) }
        try clearDestination(url)

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Double(sampleRate),
            AVNumberOfChannelsKey: audioChannels,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]

        func emit() throws {
            let file: AVAudioFile
            do {
                file = try AVAudioFile(forWriting: url, settings: settings,
                                       commonFormat: .pcmFormatFloat32, interleaved: true)
            } catch {
                throw Failure.wavSetup("\(error)")
            }
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                                frameCapacity: AVAudioFrameCount(plan.frames))
            else {
                throw Failure.wavSetup("AVAudioPCMBuffer would not allocate \(plan.frames) "
                    + "frames of \(file.processingFormat)")
            }
            buffer.frameLength = AVAudioFrameCount(plan.frames)
            // An *interleaved* format has one buffer in the list, not one per channel, so
            // `floatChannelData[0]` is the whole L,R,L,R block and `[1]` does not exist.
            // Filling it as though it were the left channel would write half the samples
            // and read past the end of the other half.
            guard let destination = buffer.floatChannelData?[0] else {
                throw Failure.wavSetup("the PCM buffer has no float channel data")
            }
            plan.interleaved.withUnsafeBytes { source in
                guard let start = source.baseAddress else { return }
                memcpy(destination, start, plan.frames * audioBytesPerFrame)
            }
            do {
                try file.write(from: buffer)
            } catch {
                throw Failure.wavSetup("write(from:) failed: \(error)")
            }
        }
        try emit()
    }
}

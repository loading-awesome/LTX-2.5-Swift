// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// Waveform → log-mel, the front end of the audio VAE **encoder**.
///
/// This is the stage ``AudioVAEEncoder`` deliberately stops short of, and the reason it
/// takes a tensor rather than a file path: the encoder consumes a stereo log-mel
/// `[B, 2, frames, 64]`, and the transform that produces one carries no weights and
/// belongs outside it. This type is that transform, minus a resampler — see
/// **Resampling** below, the one deliberate gap.
///
/// ```
/// [samples] or [channels, samples] @ 16 kHz
///   -> reflect pad n_fft/2 both sides      (center=True, pad_mode="reflect")
///   -> frame at hop 160, window 1024       (hann, periodic)
///   -> rfft, magnitude                     (power=1.0 — MAGNITUDE, not power)
///   -> slaney mel filterbank, slaney norm   [513 -> 64]
///   -> log(clamp(·, min: 1e-5))
///   -> [1, 2, frames, 64]                  (time before frequency)
/// ```
///
/// ## The four things this transform gets wrong quietly
///
/// Every one of these produces a correctly-shaped mel. None of them raises anything, and
/// the first place a wrong one shows up is as a value drift thirty layers into the DiT.
///
/// 1. **`power=1.0` is magnitude.** `2.0` is the more common default for a mel
///    spectrogram, and it is not what this is. Note carefully what a `power=2.0` mel is
///    *not*: it is **not** the square of this one, because the exponent is applied to the
///    linear spectrogram *before* the mel filterbank sums it, and
///    `Σ wᵢ|Sᵢ|²  ≠  (Σ wᵢ|Sᵢ|)²`. So the error is not a constant offset in log space
///    that a downstream scale could absorb — it runs between `+2.82` and `+11.51` nats
///    across a single frame of a 440 Hz sine
///    (`AudioMelProcessorTests.powerOneIsMagnitudeNotPower`). For a tone in
///    one filter of weight `w` the offset collapses to `−log w ≈ 2.75`; the spread comes
///    from spectral leakage. There is no single number to look for, which is exactly why
///    this is hard to spot by eye.
/// 2. **`mel_scale="slaney"` and `norm="slaney"` are two independent choices.** The
///    *scale* is the hz↔mel warping (linear below 1 kHz at `200/3` Hz per mel, log above,
///    versus HTK's single `2595 log₁₀(1 + f/700)`). The *norm* additionally divides each
///    filter by its own bandwidth, `2 / (f[i+2] − f[i])`. Getting either wrong moves
///    every filter's centre or every filter's height, and the resulting mel is still a
///    perfectly plausible-looking spectrogram. They are discriminated numerically in the
///    tests: the column sums of the slaney/slaney bank span `[0.06323, 0.06514]`, of the
///    slaney scale *without* the norm `[2.970, 23.368]`, and of HTK `[1.773, 20.770]`.
/// 3. **`center=True, pad_mode="reflect"`.** The signal is reflect-padded by `n_fft/2`
///    on both sides before framing, which fixes the frame count at `1 + samples / hop`
///    exactly (see ``frameCount(samples:)``). An off-by-one here changes the *audio
///    latent count*, which `FRAGILE_CONTRACTS.md` #3 records as a shape mismatch rather
///    than a value diff — cheap to find, but only if the shape is what you compare.
///    Reflect is not cosmetic: on a descending ramp it moves the first frame by
///    `7.375` nats against zero-padding, because zero-padding invents a step
///    discontinuity that broadens the whole spectrum.
/// 4. **Stereo.** `conv_in.conv.weight` is `[128, 2, 3, 3]`, so the encoder needs two
///    planes. **The transform itself does not replicate mono**: it is channel-agnostic
///    and simply maps `[B, C, T] → [B, C, T, mels]`, and normally it is the *upstream*
///    media loader that produces two channels. Replication is therefore this type's
///    choice, made here rather than left to the caller, and it is stated as such: a mono
///    source becomes two identical planes. Genuine stereo passes through with the
///    channels distinct — the planes never interact, so a two-channel run is identical
///    to two independent one-channel runs.
///
/// ## Resampling: the deliberate limit
///
/// The rate conversion this pipeline expects is a Kaiser-windowed sinc with
/// `lowpass_filter_width=6`, `rolloff=0.99`, default Kaiser `beta`, and a `gcd`-reduced
/// rational ratio. **This type does not implement it, and does not approximate it.**
/// `AVAudioConverter`'s resampler is a different filter with undocumented internals;
/// substituting it would put an unnameable difference in front of a stage that is
/// otherwise exact, and the difference would be largest exactly where audio conditioning
/// is most audible.
///
/// So ``logMel(waveform:sampleRate:)`` **refuses anything that is not the configured
/// rate** with ``Failure/unsupportedSampleRate(got:expected:)``. A caller who has 48 kHz
/// audio must resample deliberately, somewhere they own, and own the difference. That is
/// workable in a shipping app rather than a cop-out: a dedicated mono-16 kHz load path
/// plus a `precondition` that makes a 48 kHz clip fail loudly is the whole of what the
/// calling side needs.
///
/// **What lifting it would take**, concretely: writing out the sinc resample kernel here.
/// The kernel is closed-form, so it is portable; it is the *defaults* — filter width,
/// rolloff, Kaiser beta, how the ratio is reduced — and not the algorithm that are the
/// risk, and every one of them has to be pinned deliberately rather than inherited from
/// whatever library is nearest.
///
/// ## Precision, and contract 16
///
/// `FRAGILE_CONTRACTS.md` #16 is about reductions that narrow back to bf16 and the places
/// a port must narrow with them. **This transform has no such site.** The waveform is
/// fp32 on every path that reaches it and the mel carries the waveform's dtype, so the
/// whole transform runs in fp32 and nothing is narrowed mid-computation: the STFT and the
/// filterbank product are fp32, the output is fp32, and ``AudioVAEEncoder/encode(_:)``
/// does the bf16 cast at its own boundary as it already documents.
///
/// The **window and the filterbank are built in `Double` and cast to `Float`**, one step
/// wider than the fp32 they end up stored in. This is the same last-bit trade
/// ``AudioVocoder`` makes for its resampler filter. Building the filterbank in fp32
/// instead moves each tap by about `3.1e-09` and each column sum by `2.6e-08`, which is
/// below the precision it is stored at either way.
///
/// The window is where the extra width actually earns something: `0.5 − 0.5 cos(2πn/N)`
/// evaluated in fp32 cancels five significant digits near the edges, where `cos ≈ 1`. At
/// `N = 1024` the fp32 form gives `w[1] = 9.417533875e-06` against the exact
/// `sin²(π/1024) = 9.412358699e-06`, a relative error of `5.5e-04`; in `Double` the value
/// is exact. The absolute difference is `5.2e-09` on a window peaking at `1.0`, so it
/// does not propagate into the mel in any bin carrying energy — it is written down here
/// only because a window that disagrees with the obvious spelling is otherwise the shape
/// of a bug.
public struct AudioMelProcessor {

    // MARK: - Configuration

    /// The four numbers this transform takes, plus the two it derives.
    ///
    /// All four come **from the checkpoint**, never from a literal here. See
    /// ``AudioMelProcessor/configuration(from:)`` for the exact keys.
    ///
    /// `fMin` and `fMax` are *derived*, not configured: `f_min` is `0.0` and `f_max` is
    /// `sampleRate / 2`, and the checkpoint's own `preprocessing.mel.mel_fmin` /
    /// `mel_fmax` are never read. They are still fields here rather than constants, so
    /// that ``configuration(from:)`` can *check* those unused values agree and refuse a
    /// checkpoint that means something different by them — the same discipline
    /// ``AudioVocoder`` applies to `win_size`.
    public struct Configuration: Equatable, Sendable {
        /// `target_sample_rate`. Also the only rate ``logMel(waveform:sampleRate:)``
        /// accepts, because the resampler is not ported.
        public let sampleRate: Int
        /// `n_fft`, and `win_length` with it: `win_length` is always `n_fft`.
        public let nFFT: Int
        /// `mel_hop_length`. 160 at 16 kHz is 10 ms per frame.
        public let hopLength: Int
        /// `n_mels`.
        public let melBins: Int
        /// `f_min`. Unconditionally `0.0`.
        public let fMin: Double
        /// `f_max`. `sampleRate / 2`, derived rather than configured.
        public let fMax: Double

        public init(sampleRate: Int, nFFT: Int, hopLength: Int, melBins: Int,
                    fMin: Double, fMax: Double) {
            self.sampleRate = sampleRate
            self.nFFT = nFFT
            self.hopLength = hopLength
            self.melBins = melBins
            self.fMin = fMin
            self.fMax = fMax
        }

        /// The derivation, for a caller that has only the four transform parameters:
        /// `f_min = 0`, `f_max = sample_rate / 2`.
        public init(sampleRate: Int, nFFT: Int, hopLength: Int, melBins: Int) {
            self.init(sampleRate: sampleRate, nFFT: nFFT, hopLength: hopLength,
                      melBins: melBins, fMin: 0, fMax: Double(sampleRate) / 2)
        }

        /// `n_fft // 2 + 1`, the one-sided STFT bin count and the filterbank's row count.
        public var frequencyBins: Int { nFFT / 2 + 1 }
    }

    // MARK: - Failure

    /// Every way this can refuse, named. There is no fallback path anywhere in this type:
    /// a caller that hands it something it cannot faithfully transform gets an error, not
    /// a plausible tensor.
    public enum Failure: Error, CustomStringConvertible {
        /// The input is not at the configured rate, and this port has no resampler.
        case unsupportedSampleRate(got: Int, expected: Int)
        /// A checkpoint key is absent, malformed, or says something this port does not
        /// implement.
        case configuration(String)
        /// The waveform's rank, channel count or dtype is not one this accepts.
        case waveform(String)
        /// Fewer samples than reflect padding needs. `pad_mode="reflect"` cannot mirror
        /// more samples than the signal has.
        case tooShort(samples: Int, minimum: Int)

        public var description: String {
            switch self {
            case let .unsupportedSampleRate(got, expected):
                return "AudioMelProcessor: waveform is \(got) Hz, not \(expected) Hz. "
                    + "This type does not resample — the required filter is a "
                    + "Kaiser-windowed sinc that AVAudioConverter cannot match — so "
                    + "resampling must be done deliberately by the caller, who then "
                    + "owns the difference."
            case let .configuration(message):
                return "AudioMelProcessor configuration: \(message)"
            case let .waveform(message):
                return "AudioMelProcessor waveform: \(message)"
            case let .tooShort(samples, minimum):
                return "AudioMelProcessor: \(samples) samples is below the \(minimum) "
                    + "that reflect padding needs (n_fft / 2 + 1). Reflect padding "
                    + "cannot mirror more samples than the signal holds, so this throws "
                    + "rather than degrading to zero padding."
            }
        }
    }

    // MARK: - Stored state

    public let configuration: Configuration

    /// The analysis window, `[nFFT]` fp32: a periodic Hann window,
    /// `0.5 − 0.5 cos(2πn / N)` over `n ∈ [0, N)`.
    ///
    /// **Periodic, not symmetric**, and the difference is one sample: the symmetric form
    /// divides by `N − 1` and ends at `0` instead of `9.4175e-06`. It shows up as a
    /// window sum of `511.5` rather than exactly `512.0`, which is what the test asserts,
    /// because a half-sample-wide error in the window is invisible in the mel's shape and
    /// costs a fraction of a dB of leakage everywhere.
    public let window: MLXArray

    /// The mel filterbank, `[frequencyBins, melBins]` fp32 — slaney scale, slaney norm.
    ///
    /// Oriented so that `magnitude [.., frames, frequencyBins] @ filterbank` is the mel,
    /// which is one matmul and no rearranges.
    public let filterbank: MLXArray

    // MARK: - Init

    /// Build the window and the filterbank once. Both depend only on the configuration,
    /// and both are the expensive part; ``logMel(waveform:sampleRate:)`` is then a gather,
    /// an FFT and a matmul.
    public init(configuration: Configuration) throws {
        let c = configuration
        guard c.nFFT > 0, c.nFFT % 2 == 0 else {
            throw Failure.configuration("n_fft is \(c.nFFT); it must be positive and even "
                + "for the one-sided bin count n_fft / 2 + 1 to be exact")
        }
        guard c.hopLength > 0 else {
            throw Failure.configuration("hop_length is \(c.hopLength); it must be positive")
        }
        guard c.melBins > 0 else {
            throw Failure.configuration("n_mels is \(c.melBins); it must be positive")
        }
        guard c.sampleRate > 0 else {
            throw Failure.configuration("sample_rate is \(c.sampleRate); it must be positive")
        }
        guard c.fMax > c.fMin, c.fMin >= 0 else {
            throw Failure.configuration("f_min \(c.fMin) / f_max \(c.fMax) is not an "
                + "increasing non-negative pair")
        }
        self.configuration = c
        window = MLXArray(Self.hannWindow(length: c.nFFT), [c.nFFT])
        let bank = Self.melFilterbank(
            frequencyBins: c.frequencyBins, fMin: c.fMin, fMax: c.fMax,
            melBins: c.melBins, sampleRate: c.sampleRate,
            scale: .slaney, normalization: .slaney)
        // This refuses rather than warning and continuing. An all-zero filter means a mel
        // bin that is *always* `log(1e-5)` no matter the audio — a constant column in the
        // conditioning, which is a silent modelling error rather than a numeric one, and
        // there is no configuration in this repo where it is intended.
        for m in 0 ..< c.melBins {
            let column = bank[(m * c.frequencyBins) ..< ((m + 1) * c.frequencyBins)]
            if column.allSatisfy({ $0 == 0 }) {
                throw Failure.configuration("mel filter \(m) of \(c.melBins) is all zero: "
                    + "n_mels is too high or n_fft too low for \(c.sampleRate) Hz, and "
                    + "that bin would be a constant log(1e-5) column forever")
            }
        }
        // Built mel-major so the zero check above can slice contiguously; transposed once
        // here into the `[frequencyBins, melBins]` orientation the matmul wants.
        filterbank = MLXArray(bank, [c.melBins, c.frequencyBins]).transposed(1, 0)
    }

    /// Read the four constructor arguments **from the checkpoint**.
    ///
    /// * `sampleRate` from `audio_vae.model.params.sampling_rate` (default `16000`).
    /// * `nFFT` from `audio_vae.preprocessing.stft.filter_length` (default `1024`).
    /// * `hopLength` from `audio_vae.preprocessing.stft.hop_length` (default `160`).
    /// * `melBins` from `ddconfig.mel_bins`, else `preprocessing.mel.n_mel_channels`,
    ///   else `variables.mel_bins`.
    ///
    /// The defaults are defaults, not constants: a key that is *present* always wins, and
    /// `mel_bins` walks the same fallback ladder ``AudioVAEEncoder`` already walks.
    ///
    /// Three keys the transform never reads are nevertheless checked here, because a
    /// checkpoint that means something different by them would otherwise be silently
    /// mistransformed:
    ///
    /// * `preprocessing.stft.win_length` — `win_length` is always `n_fft`.
    /// * `preprocessing.mel.mel_fmin` / `mel_fmax` — always `0.0` and `sample_rate / 2.0`.
    ///   On this checkpoint they are `0` and `8000`, which agree.
    /// * `preprocessing.audio.sampling_rate` — a second copy of the rate, `16000` here.
    public static func configuration(from header: SafetensorsHeader) throws -> Configuration {
        guard let config = header.metadataJSON("config"),
              let audio = config["audio_vae"] as? [String: Any]
        else { throw Failure.configuration("__metadata__.config.audio_vae is absent") }

        let params = (audio["model"] as? [String: Any])?["params"] as? [String: Any] ?? [:]
        let dd = params["ddconfig"] as? [String: Any] ?? [:]
        let preprocessing = audio["preprocessing"] as? [String: Any] ?? [:]
        let stft = preprocessing["stft"] as? [String: Any] ?? [:]
        let mel = preprocessing["mel"] as? [String: Any] ?? [:]
        let variables = audio["variables"] as? [String: Any] ?? [:]

        // Present wins, absent falls back to the default.
        func number(_ dict: [String: Any], _ key: String) -> Double? {
            if let v = dict[key] as? Int { return Double(v) }
            if let v = dict[key] as? Double { return v }
            return nil
        }
        let sampleRate = Int(number(params, "sampling_rate") ?? 16000)
        let nFFT = Int(number(stft, "filter_length") ?? 1024)
        let hopLength = Int(number(stft, "hop_length") ?? 160)

        let melBins: Int
        if let m = number(dd, "mel_bins") ?? number(mel, "n_mel_channels")
            ?? number(variables, "mel_bins") {
            melBins = Int(m)
        } else {
            throw Failure.configuration("none of ddconfig.mel_bins, "
                + "preprocessing.mel.n_mel_channels or variables.mel_bins is present")
        }

        if let win = number(stft, "win_length"), Int(win) != nFFT {
            throw Failure.configuration("preprocessing.stft.win_length \(Int(win)) differs "
                + "from filter_length \(nFFT); AudioProcessor builds MelSpectrogram with "
                + "win_length = n_fft and ignores win_length entirely, so this checkpoint "
                + "means something this front end would not honour")
        }
        let derivedMax = Double(sampleRate) / 2
        if let low = number(mel, "mel_fmin"), low != 0 {
            throw Failure.configuration("preprocessing.mel.mel_fmin is \(low); "
                + "AudioProcessor passes f_min = 0.0 literally and never reads this key")
        }
        if let high = number(mel, "mel_fmax"), high != derivedMax {
            throw Failure.configuration("preprocessing.mel.mel_fmax is \(high) against a "
                + "derived sample_rate / 2 of \(derivedMax); AudioProcessor passes the "
                + "derived value and never reads this key")
        }
        if let rate = number(preprocessing["audio"] as? [String: Any] ?? [:], "sampling_rate"),
           Int(rate) != sampleRate {
            throw Failure.configuration("preprocessing.audio.sampling_rate \(Int(rate)) "
                + "differs from model.params.sampling_rate \(sampleRate); "
                + "AudioProcessor is built from the latter")
        }

        return Configuration(sampleRate: sampleRate, nFFT: nFFT, hopLength: hopLength,
                             melBins: melBins, fMin: 0, fMax: derivedMax)
    }

    // MARK: - Shape algebra

    /// Frames a signal of `samples` produces: **`1 + samples / hop`**, integer division,
    /// and nothing else.
    ///
    /// The derivation is worth writing down because the off-by-one is the expensive kind.
    /// `center=True` pads by `p = n_fft / 2` on each side, so the framed length is
    /// `samples + n_fft`. The STFT emits a frame at every offset `f·hop` for which
    /// `f·hop + n_fft ≤ samples + n_fft`, i.e. `f ≤ samples / hop`. The `n_fft` cancels
    /// exactly — the window length drops out of the frame count entirely, which is the
    /// part that surprises people porting from a `center=False` STFT where it does not.
    ///
    /// Downstream this is the *audio latent count*: ``AudioVAEEncoder`` downsamples time
    /// by 4, and `FRAGILE_CONTRACTS.md` #3 records a mismatch here as a shape error
    /// rather than a value diff. Checked at 513, 514, 640, 1600, 2400, 3200, 16000,
    /// 64000 and 64001 samples.
    public func frameCount(samples: Int) -> Int {
        1 + samples / configuration.hopLength
    }

    /// The shortest signal reflect padding can handle, `n_fft / 2 + 1`.
    ///
    /// Reflect padding mirrors *around* the edge sample without repeating it, so it needs
    /// `samples − 1 ≥ p`. At `n_fft = 1024`: 512 samples is refused, 513 gives 4 frames.
    public var minimumSamples: Int { configuration.nFFT / 2 + 1 }

    // MARK: - The transform

    /// `[samples]` or `[channels, samples]` fp32 at the configured rate → log-mel
    /// `[1, 2, frames, melBins]` fp32.
    ///
    /// The output is exactly what ``AudioVAEEncoder/encode(_:)`` accepts: rank 4, `NCHW`
    /// with **time on axis 2 and frequency on axis 3**, two channels, and fp32 — the
    /// encoder casts to its own bf16 weight dtype at its own boundary, as its docs say,
    /// and handing it a pre-narrowed mel would round twice.
    ///
    /// - Parameters:
    ///   - waveform: rank 1 (mono, replicated to two identical planes) or rank 2
    ///     `[channels, samples]` with 1 or 2 channels. Batch is always 1: the transform
    ///     batches naturally, but every caller feeds it one clip, and a batch axis here
    ///     would be an untested code path.
    ///   - sampleRate: the waveform's actual rate, **passed rather than assumed** so that
    ///     a mismatch is an error at this boundary instead of a silent pitch shift.
    public func logMel(waveform: MLXArray, sampleRate: Int) throws -> MLXArray {
        let c = configuration
        guard sampleRate == c.sampleRate else {
            throw Failure.unsupportedSampleRate(got: sampleRate, expected: c.sampleRate)
        }
        guard waveform.dtype == .float32 else {
            // Not a convenience cast. The mel carries the waveform's dtype, so a narrower
            // input would narrow the *mel* — contract 16's failure mode arriving through
            // the front door. A caller that widened a bf16 tensor should say so.
            throw Failure.waveform("dtype is \(waveform.dtype), not float32. The "
                + "reference computes this transform in fp32 and casts the mel back to "
                + "the waveform's dtype, so a narrower waveform would narrow the mel.")
        }

        let planes: MLXArray
        switch waveform.ndim {
        case 1:
            planes = MLX.stacked([waveform, waveform], axis: 0)
        case 2:
            switch waveform.shape[0] {
            case 1: planes = MLX.concatenated([waveform, waveform], axis: 0)
            case 2: planes = waveform
            default:
                throw Failure.waveform("\(waveform.shape[0]) channels; AudioEncoder's "
                    + "conv_in is [128, 2, 3, 3] and accepts exactly 2 planes. Downmix "
                    + "deliberately rather than have this pick two of them.")
            }
        default:
            throw Failure.waveform("rank \(waveform.ndim) \(waveform.shape); expected "
                + "[samples] or [channels, samples]")
        }

        let samples = planes.shape[1]
        guard samples >= minimumSamples else {
            throw Failure.tooShort(samples: samples, minimum: minimumSamples)
        }

        let frames = frameCount(samples: samples)
        let gather = MLXArray(Self.reflectFrameIndices(
            samples: samples, frames: frames, nFFT: c.nFFT, hop: c.hopLength),
            [frames, c.nFFT])

        // Reflect padding and framing in one gather. Materialising the padded signal
        // first would be the obvious spelling and would cost an extra copy of the whole
        // clip; more to the point, the reflect map is *the* thing to get right here, so
        // it is written once, in ``reflectFrameIndices(samples:frames:nFFT:hop:)``, where
        // it can be read.
        let framed = MLX.take(planes, gather, axis: -1)         // [C, frames, nFFT]
        let windowed = framed * window
        let spectrum = MLXFFT.rfft(windowed, axis: -1)          // [C, frames, freqBins]

        // `power=1.0` is magnitude. Spelled as sqrt(re² + im²) rather than a complex
        // `abs`, matching ``AudioVocoder``'s own STFT magnitude, so that the exponent is
        // visible in the source and cannot be changed by a library default.
        let real = spectrum.realPart()
        let imaginary = spectrum.imaginaryPart()
        let magnitude = MLX.sqrt(real * real + imaginary * imaginary)

        let mel = MLX.matmul(magnitude, filterbank)             // [C, frames, melBins]

        // `log(clamp(mel, min: 1e-5))`. The clamp is not defensive rounding —
        // it is the only thing standing between digital silence and `log(0) = -inf`
        // propagating into the encoder, and its value fixes the floor of the whole
        // representation at `log(1e-5) = -11.5129`.
        let floored = MLX.maximum(mel, MLXArray(Self.magnitudeFloor))
        return MLX.log(floored).expandedDimensions(axis: 0)     // [1, C, frames, melBins]
    }

    /// The clamp floor. Named because the value is load-bearing: it fixes the mel's floor
    /// at `log(1e-5) = -11.512925`, and every silent frame sits exactly on it.
    static let magnitudeFloor: Float = 1e-5

    // MARK: - Reflect padding

    /// The gather map: `indices[f, k]` is the sample that frame `f`, window tap `k`
    /// reads, with `mode="reflect"` padding already folded in.
    ///
    /// Frame `f` covers padded positions `[f·hop, f·hop + nFFT)`, and padded position `i`
    /// is signal position `j = i − p` where `p = nFFT / 2`. Reflect maps `j` back into
    /// `[0, samples)` by mirroring **around the edge samples without repeating them**:
    ///
    /// ```
    /// j < 0            ->  -j                 (x[-1] = x[1], not x[0])
    /// j >= samples     ->  2(samples-1) - j   (x[n] = x[n-2])
    /// ```
    ///
    /// which is why `samples ≥ p + 1` and not `samples ≥ p` — at `samples = p` the left
    /// pad would need `x[-p] = x[p]`, one past the end. The `edge` mode MLX *does* offer
    /// repeats the edge sample and is a different signal; there is no reflect in
    /// `MLX.PadMode`, which is the reason this is written out rather than delegated.
    static func reflectFrameIndices(samples: Int, frames: Int, nFFT: Int, hop: Int)
        -> [Int32] {
        let pad = nFFT / 2
        var indices = [Int32](repeating: 0, count: frames * nFFT)
        for f in 0 ..< frames {
            let base = f * hop - pad
            for k in 0 ..< nFFT {
                var j = base + k
                if j < 0 { j = -j } else if j >= samples { j = 2 * (samples - 1) - j }
                indices[f * nFFT + k] = Int32(j)
            }
        }
        return indices
    }

    // MARK: - Window

    /// A **periodic** Hann window: `0.5 − 0.5 cos(2πn / length)` over `n ∈ [0, length)`.
    ///
    /// Periodic, not symmetric. Divide by `length − 1` instead and the window sum drops
    /// from exactly `512` to `511.5`.
    static func hannWindow(length: Int) -> [Float] {
        (0 ..< length).map {
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(length)))
        }
    }

    // MARK: - Filterbank

    /// The hz↔mel warping.
    ///
    /// Only ``slaney`` is used by this port; ``htk`` exists so the tests can build the
    /// wrong one and show it is materially different, which is the only way to demonstrate
    /// that "a filterbank exists" is not the claim being made.
    enum MelScale { case slaney, htk }

    /// The filterbank normalisation. **Independent of the scale**, which is the whole
    /// point of it being a separate type: `norm="slaney"` divides filter `i` by its bandwidth
    /// `(f[i+2] − f[i]) / 2`, and can be applied to an HTK-warped bank or omitted from a
    /// slaney-warped one.
    enum Normalization { case slaney, none }

    /// Hz → mel.
    ///
    /// The slaney branch is piecewise: **linear** at `200/3` Hz per mel below 1 kHz, and
    /// **logarithmic** above it with `logstep = ln(6.4) / 27`. HTK is a single
    /// `2595 log₁₀(1 + f/700)` everywhere. They agree nowhere except `f = 0`, so choosing
    /// the wrong one moves every filter's centre frequency — a mel that looks entirely
    /// normal and whose energy is in the wrong bins.
    static func hzToMel(_ hz: Double, scale: MelScale) -> Double {
        switch scale {
        case .htk:
            return 2595.0 * log10(1.0 + hz / 700.0)
        case .slaney:
            let fSp = 200.0 / 3.0
            let minLogHz = 1000.0
            let minLogMel = minLogHz / fSp
            let logstep = log(6.4) / 27.0
            return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logstep : hz / fSp
        }
    }

    /// Mel → hz, the inverse of ``hzToMel(_:scale:)``.
    ///
    /// The branch condition is on the **mel** value against `minLogMel = 1000 / (200/3)`,
    /// not on the resulting hz. Testing the hz instead is self-referential and silently
    /// shifts the single filter that straddles the knee.
    static func melToHz(_ mel: Double, scale: MelScale) -> Double {
        switch scale {
        case .htk:
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        case .slaney:
            let fSp = 200.0 / 3.0
            let minLogHz = 1000.0
            let minLogMel = minLogHz / fSp
            let logstep = log(6.4) / 27.0
            return mel >= minLogMel ? minLogHz * exp(logstep * (mel - minLogMel)) : fSp * mel
        }
    }

    /// The mel filterbank, returned **mel-major** as a flat `[melBins × frequencyBins]`
    /// array; ``init(configuration:)`` transposes it once.
    ///
    /// Three details are spelled out rather than left to a from-memory formula, because
    /// each is a place such a formula goes wrong:
    ///
    /// 1. **The STFT bin grid is `linspace(0, sampleRate / 2, frequencyBins)` with
    ///    *integer* division.** It is the Nyquist rate, *not* `fMax`. They coincide here
    ///    because `fMax` is derived as `sampleRate / 2` anyway, but a checkpoint with a
    ///    band-limited `fMax` would place every bin wrongly if `fMax` were used.
    /// 2. **The triangles are the slope form**, `max(0, min(down, up))` with
    ///    `down = −(f_pts[m] − freq) / (f_pts[m+1] − f_pts[m])` and
    ///    `up = (f_pts[m+2] − freq) / (f_pts[m+2] − f_pts[m+1])`. Peaks are *not*
    ///    normalised to 1, so a bank built by "unit triangles, then normalise" is a
    ///    different bank.
    /// 3. **The slaney norm is `2 / (f_pts[m+2] − f_pts[m])`**, applied *after* the
    ///    triangles, using the outer edges and not the half-widths.
    ///
    /// Evaluated in `Double` and cast to `Float`; see the type's precision note.
    static func melFilterbank(frequencyBins: Int, fMin: Double, fMax: Double, melBins: Int,
                              sampleRate: Int, scale: MelScale,
                              normalization: Normalization) -> [Float] {
        // `linspace(0, sample_rate / 2, n_freqs)` — integer division on the rate.
        let nyquist = Double(sampleRate / 2)
        let freqStep = nyquist / Double(frequencyBins - 1)
        let allFreqs = (0 ..< frequencyBins).map { Double($0) * freqStep }

        let melMin = hzToMel(fMin, scale: scale)
        let melMax = hzToMel(fMax, scale: scale)
        let melStep = (melMax - melMin) / Double(melBins + 1)
        let edges = (0 ..< melBins + 2).map {
            melToHz(melMin + Double($0) * melStep, scale: scale)
        }

        var bank = [Float](repeating: 0, count: melBins * frequencyBins)
        for m in 0 ..< melBins {
            let lower = edges[m], centre = edges[m + 1], upper = edges[m + 2]
            let enorm: Double = normalization == .slaney ? 2.0 / (upper - lower) : 1.0
            for (b, freq) in allFreqs.enumerated() {
                let down = (freq - lower) / (centre - lower)
                let up = (upper - freq) / (upper - centre)
                let value = max(0.0, min(down, up))
                bank[m * frequencyBins + b] = Float(value * enorm)
            }
        }
        return bank
    }
}

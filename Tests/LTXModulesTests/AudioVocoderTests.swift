// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The vocoder: mel → 48 kHz stereo waveform.
///
/// ## What this suite can and cannot assert
///
/// `docs/FRAGILE_CONTRACTS.md` #11 rules out the obvious test: a vocoder of this shape is
/// not reproducible sample for sample, and two runs over a bit-identical mel can differ at
/// `max_abs` 1.9e-03 while every other stage is bit-exact. So nothing below asserts a
/// waveform value.
///
/// What is left is worth more than it sounds:
///
/// * **Key accounting.** Every one of the checkpoint's 1227 `vocoder.*` tensors is
///   consumed at the shape the metadata's config predicts, and nothing the module tree
///   wants is absent. This is the single highest-value test here — it is what catches a
///   module tree that is structurally wrong but still runs, which no check on the waveform
///   itself could.
/// * **Arithmetic pinning.** The four primitives whose conventions are guessable and
///   whose errors are silent — the two convolution weight permutations, the depthwise
///   filter, and SnakeBeta's log-scale — are checked against values computed by hand.
/// * **Rate algebra.** `T` mel frames must give exactly `160T` samples at 16 kHz and
///   `480T` at 48 kHz, stereo. An off-by-one anywhere in the padding shows up here.
///
/// ## Why most of it runs without the checkpoint
///
/// ``AudioVocoder/init(weights:configuration:)`` takes a weight dictionary, so the module
/// tree can be built at 1/500th scale from ``syntheticWeights(for:)`` and exercised
/// end-to-end in milliseconds. The tests that genuinely need the real 365 MB file — key
/// accounting, config parsing, and one real synthesis — guard on its presence and print a
/// `SKIP` line rather than failing on a machine that does not have it.
@Suite("Audio vocoder")
struct AudioVocoderTests {

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root!
        + "/vae/ltx-2.5-audio-vae-bf16.safetensors"

    static var checkpointURL: URL? {
        FileManager.default.fileExists(atPath: checkpointPath)
            ? URL(fileURLWithPath: checkpointPath) : nil
    }

    // MARK: - Primitive pinning
    //
    // These four are properties of MLX or of a formula, not of the checkpoint, and every
    // one of them has a wrong version that produces a same-shaped tensor. Contract 11
    // means no downstream diff will ever notice, so they are pinned against arithmetic
    // done on paper.

    /// Transposed-convolution weights are stored `[in, out, k]`; MLX wants `[out, k, in]`.
    ///
    /// Both the permutation and MLX's `padding` semantics are asserted here, because
    /// confusing `(1, 2, 0)` with `conv1d`'s `(0, 2, 1)` is a no-op on any square stage —
    /// and `ups.0` is `[1536, 768, 11]`, so five of the vocoder's six stages would catch
    /// it on shape and the sixth would not.
    ///
    /// `x = [1, 2, 3]`, kernel `[1, 10, 100]`, stride 2. Transposed convolution scatters
    /// each input across the kernel at `stride` spacing and sums the overlaps:
    /// `[1, 10, 100+2, 20, 200+3, 30, 300]`.
    @Test("transposed convolution matches PyTorch's layout and padding")
    func transposedConvolutionLayout() {
        let x = MLXArray([Float(1), 2, 3], [1, 3, 1])
        let w = MLXArray([Float(1), 10, 100], [1, 1, 3])        // stored [in, out, k]

        let full = AudioVocoder.convTransposed1d(x, weight: w, bias: nil, stride: 2,
                                                 padding: 0)
        #expect(full.shape == [1, 7, 1])
        #expect(full.asArray(Float.self) == [1, 10, 102, 20, 203, 30, 300])

        // `padding` crops the output symmetrically; it does not pad the input.
        // Length is (L-1)*stride - 2*padding + k = 5.
        let cropped = AudioVocoder.convTransposed1d(x, weight: w, bias: nil, stride: 2,
                                                    padding: 1)
        #expect(cropped.shape == [1, 5, 1])
        #expect(cropped.asArray(Float.self) == [10, 102, 20, 203, 30])
    }

    /// Convolution here is cross-correlation with `[out, in, k]` weights, and it is
    /// *dilated* in `convs1` — the dilation and its padding have to move together or the
    /// block stops preserving length.
    @Test("convolution matches PyTorch's layout, padding and dilation")
    func convolutionLayout() {
        let x = MLXArray([Float(1), 2, 3, 4], [1, 4, 1])
        let w = MLXArray([Float(1), 10, 100], [1, 1, 3])        // stored [out, in, k]

        let plain = AudioVocoder.conv1d(x, weight: w, bias: nil, padding: 1, dilation: 1)
        #expect(plain.asArray(Float.self) == [210, 321, 432, 43])

        // The padding for kernel 3 at dilation 2 is 2, which is what keeps the length at 4.
        let dilated = AudioVocoder.conv1d(x, weight: w, bias: nil,
                                          padding: AudioVocoder.padding(kernel: 3,
                                                                        dilation: 2),
                                          dilation: 2)
        #expect(dilated.asArray(Float.self) == [310, 420, 31, 42])

        let biased = AudioVocoder.conv1d(x, weight: w, bias: MLXArray([Float(5)]),
                                         padding: 1, dilation: 1)
        #expect(biased.asArray(Float.self) == [215, 326, 437, 48])
    }

    /// The anti-aliasing filters are applied per channel with **one shared kernel**: a
    /// depthwise convolution over an expanded buffer, not a mixing one.
    ///
    /// A mono collapse here — summing the channels instead of filtering each — would
    /// still produce a stereo-shaped output downstream, because `conv_post` maps to 2
    /// channels regardless. So the two channels are given different values and required
    /// to stay different.
    @Test("shared-filter resampling is depthwise, not mixing")
    func sharedFilterIsDepthwise() {
        // Two channels: [1,2,3,4] and [10,20,30,40]. Filter [1, 0.5].
        let x = MLXArray([Float(1), 10, 2, 20, 3, 30, 4, 40], [1, 4, 2])
        let filter = MLXArray([Float(1), 0.5], [1, 1, 2])

        let down = AudioVocoder.sharedFilterConv(x, filter: filter, stride: 1)
        #expect(down.shape == [1, 3, 2])
        #expect(down.asArray(Float.self) == [2, 20, 3.5, 35, 5, 50])

        // Transposed: each sample scatters to `stride`-spaced positions, per channel.
        let y = MLXArray([Float(1), 10, 2, 20], [1, 2, 2])
        let up = AudioVocoder.sharedFilterConvTransposed(y, filter: filter, stride: 2)
        #expect(up.shape == [1, 4, 2])
        #expect(up.asArray(Float.self) == [1, 10, 0.5, 5, 2, 20, 1, 10])
    }

    /// `SnakeBeta`: `x + (1 / (exp(beta) + 1e-9)) · sin²(exp(alpha) · x)`.
    ///
    /// **Both parameters are log-scale**, and the checkpoint says so on its own:
    /// `vocoder.act_post.act.alpha` runs from −1.289 to −0.127, entirely negative, which a
    /// linear-scale alpha initialised to ones could not be. Reading them as linear runs,
    /// keeps every shape, and turns the `1/beta` gain negative wherever beta is — 44% of
    /// the first resblock's channels.
    ///
    /// The three cases are computed by hand in `Double`:
    ///
    /// * `alpha = beta = 0` → both exponentiate to 1, so `f(0.5) = 0.5 + sin²(0.5)`.
    /// * `alpha = ln 2, beta = ln 0.5` → `f(-1) = -1 + 2·sin²(-2)`, which exercises the
    ///   two parameters *differently*; a port that used alpha for both would give
    ///   `-1 + 0.5·sin²(-2) = -0.587`, not `+0.654`.
    /// * `x = 0` → exactly 0 whatever the parameters, since `sin²(0)` is 0.
    @Test("SnakeBeta matches hand-computed values in fp32")
    func snakeBetaScalars() {
        func evaluate(_ x: Float, alpha: Float, beta: Float) -> Float {
            AudioVocoder.snakeBeta(MLXArray([x], [1, 1, 1]),
                                   alpha: MLXArray([alpha]),
                                   beta: MLXArray([beta])).asArray(Float.self)[0]
        }

        // 0.5 + sin(0.5)^2 = 0.5 + 0.4794255386^2
        #expect(abs(evaluate(0.5, alpha: 0, beta: 0) - 0.729_848_847) < 2e-6)
        // -1 + (1/0.5) * sin(-2)^2 = -1 + 2 * 0.9092974268^2
        #expect(abs(evaluate(-1, alpha: Float(log(2.0)), beta: Float(log(0.5)))
                    - 0.653_643_621) < 2e-6)
        #expect(evaluate(0, alpha: 1.7, beta: -0.3) == 0)

        // Per-channel, and the two parameters must not be interchangeable.
        let x = MLXArray([Float(0.5), 0.5], [1, 1, 2])
        let mixed = AudioVocoder.snakeBeta(x, alpha: MLXArray([Float(0), 0]),
                                           beta: MLXArray([Float(0), Float(log(2.0))]))
        let v = mixed.asArray(Float.self)
        #expect(abs(v[0] - 0.729_848_847) < 2e-6)
        #expect(abs(v[1] - (0.5 + 0.229_848_847 / 2)) < 2e-6)
    }

    /// The 3× skip resampler is the one filter that is **not** in the checkpoint: there is
    /// no `resampler.*` among the 1227 keys. Every other filter (`*.upsample.filter`,
    /// `*.downsample.lowpass.filter`) is stored and is loaded rather than recomputed,
    /// because a Kaiser window needs a Bessel `i0` and reproducing it would be free
    /// divergence. This one is safe to rebuild for a specific reason: the
    /// `window_type: "hann"` branch is closed form, `cos²` times `sinc`, with no special
    /// function in it.
    ///
    /// Pinned on the three landmarks that would move if any constant were wrong: the
    /// length `2·⌈6/0.99⌉·3 + 1 = 43`, the exact centre value `0.99/3` (where `sinc` and
    /// the window are both 1), and the vanishing endpoints (where the clamp puts the
    /// window argument at exactly `π/2`).
    @Test("the unstored Hann resampler filter rebuilds to its reference form")
    func hannResamplerFilter() {
        let (filter, pad, padLeft, padRight) = AudioVocoder.hannResamplerFilter(ratio: 3)
        #expect(filter.shape == [1, 1, 43])
        #expect(pad == 7)
        #expect(padLeft == 42)
        #expect(padRight == 40)

        let taps = filter.asArray(Float.self)
        #expect(abs(taps[21] - Float(0.99 / 3.0)) < 1e-7)
        // The endpoints vanish rather than being exactly zero: the clamp puts the window
        // argument at exactly π/2, and `cos(π/2)` is 6.1e-17 in double, not 0. Asserting
        // equality with zero would be asserting something about libm, not about the
        // filter — the tap is 1.2e-35, thirty-five orders below the centre.
        #expect(abs(taps[0]) < 1e-30)
        #expect(abs(taps[42]) < 1e-30)
        for i in 0 ..< 21 {
            #expect(abs(taps[i] - taps[42 - i]) < 1e-9)      // even in time
        }
        // The three pads must cancel exactly. Replicate-padding by `pad` on each side and
        // then transposing at stride 3 gives `(L + 2·pad − 1)·3 + 43` samples; trimming
        // `padLeft + padRight` has to leave `3L` and nothing else, or the skip path and
        // the BWE residual come out different lengths and the add throws.
        #expect(padLeft + padRight == (2 * pad - 1) * 3 + 43)
    }

    // MARK: - Structure against the real checkpoint

    /// **Every `vocoder.*` tensor is consumed, and nothing wanted is missing.**
    ///
    /// The highest-value assertion in the file. A module tree can be wrong in ways that
    /// leave the forward pass running and the shapes plausible — a stage counted wrong, a
    /// resblock index laid out kernel-major instead of stage-major, a `conv_post.bias`
    /// invented — and every one of those shows up here as a set difference and nowhere
    /// else, because contract 11 rules out the waveform check that would otherwise catch
    /// it.
    ///
    /// Reads the header only, so it costs no memory: the 365 MB payload is never mapped.
    @Test("every vocoder tensor in the checkpoint is consumed, at the right shape")
    func checkpointKeyAccounting() throws {
        guard let url = Self.checkpointURL else {
            print("SKIP checkpointKeyAccounting: no \(Self.checkpointPath)")
            return
        }
        let header = try SafetensorsHeader.read(from: url)
        let config = try AudioVocoder.configuration(from: header)
        let expected = AudioVocoder.parameterShapes(for: config)

        var actual: [String: [Int]] = [:]
        for name in header.names(withPrefix: AudioVocoder.checkpointPrefix) {
            let stripped = String(name.dropFirst(AudioVocoder.checkpointPrefix.count))
            actual[stripped] = header.tensors[name]!.shape
        }

        let unconsumed = Set(actual.keys).subtracting(expected.keys).sorted()
        let absent = Set(expected.keys).subtracting(actual.keys).sorted()
        #expect(unconsumed.isEmpty, Comment(rawValue:
            "\(unconsumed.count) checkpoint tensors no module reads: "
            + "\(unconsumed.prefix(8).joined(separator: ", "))"))
        #expect(absent.isEmpty, Comment(rawValue:
            "\(absent.count) tensors the module tree wants are absent: "
            + "\(absent.prefix(8).joined(separator: ", "))"))

        for (name, shape) in expected where actual[name] != nil {
            #expect(actual[name]! == shape,
                    Comment(rawValue: "\(name) is \(actual[name]!), predicted \(shape)"))
        }
        print("vocoder key accounting: \(actual.count) tensors, "
            + "\(expected.count) predicted, \(unconsumed.count) unconsumed, "
            + "\(absent.count) absent")
        // The count is a fact about this checkpoint, recorded so a future file that is
        // quietly a different size does not slip past a set comparison that happens to
        // balance.
        #expect(actual.count == 1227)
    }

    /// The config is *read*, never assumed — and this is what "read" has to mean.
    ///
    /// Every number here has a plausible wrong value that loads. The one that matters most
    /// is `upsample_rates`: a 2.3 checkpoint's `[6, 5, 2, 2, 2]` is 240×, not 160×, so a
    /// hardcoded 2.5 rate list would emit a waveform 1.5× too long at the wrong pitch,
    /// from a load that raised nothing at all.
    @Test("configuration comes from the checkpoint metadata")
    func configurationFromMetadata() throws {
        guard let url = Self.checkpointURL else {
            print("SKIP configurationFromMetadata: no \(Self.checkpointPath)")
            return
        }
        let config = try AudioVocoder.configuration(from: SafetensorsHeader.read(from: url))

        #expect(config.vocoder.upsampleInitialChannel == 1536)
        #expect(config.vocoder.upsampleRates == [5, 2, 2, 2, 2, 2])
        #expect(config.vocoder.upsampleKernelSizes == [11, 4, 4, 4, 4, 4])
        #expect(config.vocoder.totalUpsampleFactor == 160)      // == the mel hop length
        #expect(config.vocoder.finalChannels == 24)
        #expect(config.vocoder.resblockKernelSizes == [3, 7, 11])
        #expect(config.vocoder.resblockDilationSizes == [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        #expect(config.vocoder.useTanhAtFinal == false)
        #expect(config.vocoder.useBiasAtFinal == false)
        // Not a config key: it is fixed by construction, true for the vocoder and false
        // for the BWE, and any JSON key of the same name is ignored.
        #expect(config.vocoder.applyFinalActivation == true)

        #expect(config.bwe.upsampleInitialChannel == 512)
        #expect(config.bwe.upsampleRates == [6, 5, 2, 2, 2])
        #expect(config.bwe.upsampleKernelSizes == [12, 11, 4, 4, 4])
        #expect(config.bwe.totalUpsampleFactor == 240)          // 3x rate change x hop 80
        #expect(config.bwe.finalChannels == 16)
        #expect(config.bwe.applyFinalActivation == false)

        #expect(config.inputSamplingRate == 16000)
        #expect(config.outputSamplingRate == 48000)
        #expect(config.hopLength == 80)
        #expect(config.nFFT == 512)
        #expect(config.numMels == 64)
        // The BWE's own analysis rate must close the loop: 240 upsamples of a mel taken
        // every 80 samples is exactly the 3x rate change. If these disagreed the residual
        // and the skip would be different lengths and `waveform(fromMel:)` would throw.
        #expect(config.bwe.totalUpsampleFactor
                == config.hopLength * config.resampleRatio)
    }

    /// One real synthesis, checked for the properties that are available.
    ///
    /// Not a value test — see the suite comment. What it does establish is that the real
    /// 1227-tensor tree runs to completion, produces the exact sample count the rate
    /// algebra predicts, contains no NaN, and does not sit pinned against the output
    /// clamp. That last one is the only *amplitude* check with any teeth: because
    /// `apply_final_activation` is true and `use_tanh_at_final` is false, the output is
    /// clamped to `[-1, 1]` no matter how wrong it is, so "bounded" proves nothing and
    /// "mostly not saturated" proves a little.
    @Test("the real checkpoint synthesises a well-formed waveform")
    func checkpointSynthesis() throws {
        guard let url = Self.checkpointURL else {
            print("SKIP checkpointSynthesis: no \(Self.checkpointPath)")
            return
        }
        let vocoder = try AudioVocoder(checkpoint: url)
        let frames = 16
        let mel = Self.syntheticMel(frames: frames, bins: vocoder.config.numMels,
                                    seed: 0x5EED)

        let low = try vocoder.baseRateWaveform(fromMel: mel)
        #expect(low.shape == [1, 2, frames * 160])

        let waveform = try vocoder.waveform(fromMel: mel)
        #expect(waveform.shape == [1, 2, frames * 480])
        #expect(waveform.shape[2] == vocoder.sampleCount(melFrames: frames))

        let samples = waveform.asArray(Float.self)
        #expect(samples.allSatisfy { $0.isFinite })
        let saturated = samples.filter { abs($0) >= 0.999_999 }.count
        let peak = samples.map { abs($0) }.max() ?? 0
        let rms = (samples.reduce(0.0) { $0 + Double($1 * $1) }
                   / Double(samples.count)).squareRoot()
        print("checkpoint synthesis: \(samples.count) samples, peak \(peak), "
            + "rms \(rms), saturated \(saturated)")
        #expect(Double(saturated) / Double(samples.count) < 0.05,
                Comment(rawValue: "\(saturated) of \(samples.count) samples are pinned at "
                    + "the clamp; the model is not producing a signal, it is producing a "
                    + "square wave"))

        // Determinism on the real weights too — the synthetic run below cannot see a
        // nondeterminism that only the 768-channel reductions would expose.
        let again = try vocoder.waveform(fromMel: mel).asArray(Float.self)
        #expect(again == samples)
    }

    // MARK: - Behaviour, on a synthetic tree

    /// `T` mel frames → exactly `160T` samples at 16 kHz and `480T` at 48 kHz, stereo.
    ///
    /// Every stage is an integer rate change, so this is exact rather than approximate,
    /// and a single mis-derived padding anywhere — the transposed convolutions, the
    /// anti-aliased activation's asymmetric trims, the causal STFT's left-only pad, the
    /// skip resampler's `[42:-40]` — changes it. That is why this runs on the synthetic
    /// tree, where it can be cheap enough to run on several frame counts.
    @Test("mel frames map to exact sample counts at both rates", arguments: [4, 8, 13])
    func sampleCountAlgebra(frames: Int) throws {
        let vocoder = try Self.syntheticVocoder()
        let mel = Self.syntheticMel(frames: frames, bins: vocoder.config.numMels, seed: 11)

        let low = try vocoder.baseRateWaveform(fromMel: mel)
        #expect(low.shape == [1, 2, frames * 160])
        #expect(low.shape[2] == vocoder.baseRateSampleCount(melFrames: frames))

        let high = try vocoder.waveform(fromMel: mel)
        #expect(high.shape == [1, 2, frames * 480])
        #expect(high.shape[2] == vocoder.sampleCount(melFrames: frames))
        #expect(high.shape[2] == low.shape[2] * 3)
        // Stereo, and not by accident: `conv_post` is the only thing that fixes it at 2.
        #expect(high.shape[1] == 2)
    }

    /// The same mel twice gives a bit-identical waveform.
    ///
    /// Bit-reproducibility is not a property a vocoder of this shape has in general —
    /// contract 11 — so a passing result here says only that nothing in this
    /// implementation is order-dependent or uninitialised. It is deliberately not
    /// evidence of correctness, and would pass just as well on a wrong network.
    @Test("this implementation is deterministic (which the reference is not)")
    func determinism() throws {
        let vocoder = try Self.syntheticVocoder()
        let mel = Self.syntheticMel(frames: 6, bins: vocoder.config.numMels, seed: 7)
        let first = try vocoder.waveform(fromMel: mel).asArray(Float.self)
        let second = try vocoder.waveform(fromMel: mel).asArray(Float.self)
        #expect(first == second)
    }

    /// No NaN, no Inf, and — separately — the intermediate 16 kHz signal is bounded too.
    ///
    /// The 48 kHz output is clamped by construction, so its bound is uninformative. The
    /// 16 kHz signal is clamped as well (`apply_final_activation` true), which leaves
    /// finiteness as the only real content: a NaN survives `clamp`, so it reaches the
    /// output rather than being hidden by the bound.
    @Test("output is finite at both rates")
    func finiteness() throws {
        let vocoder = try Self.syntheticVocoder()
        let mel = Self.syntheticMel(frames: 6, bins: vocoder.config.numMels, seed: 3)

        for stage in ["16 kHz", "48 kHz"] {
            let y = stage == "16 kHz"
                ? try vocoder.baseRateWaveform(fromMel: mel)
                : try vocoder.waveform(fromMel: mel)
            let samples = y.asArray(Float.self)
            #expect(samples.allSatisfy { $0.isFinite }, Comment(rawValue: "\(stage) NaN/Inf"))
            #expect(samples.allSatisfy { abs($0) <= 1.0 }, Comment(rawValue: "\(stage) range"))
        }
    }

    /// A degenerate all-zero mel.
    ///
    /// It does **not** produce digital silence, and asserting that it did would be wrong:
    /// `conv_pre` has a bias, so a zero mel becomes a nonzero constant and every later
    /// stage has a bias of its own. What it must produce is a finite, non-saturating,
    /// time-constant-ish signal rather than a NaN or a full-scale square wave — which is
    /// the difference between "degenerate input handled" and "arithmetic fell over".
    @Test("a degenerate zero mel produces a finite, unsaturated signal, not garbage")
    func degenerateMel() throws {
        let vocoder = try Self.syntheticVocoder()
        let zeros = MLXArray.zeros([1, 2, 6, vocoder.config.numMels], dtype: .float32)
        let samples = try vocoder.waveform(fromMel: zeros).asArray(Float.self)

        #expect(samples.count == 6 * 480 * 2)
        #expect(samples.allSatisfy { $0.isFinite })
        let saturated = samples.filter { abs($0) >= 0.999_999 }.count
        #expect(Double(saturated) / Double(samples.count) < 0.05,
                Comment(rawValue: "\(saturated) of \(samples.count) samples pinned at the "
                    + "clamp on a zero mel"))
    }

    /// Malformed input throws rather than being coerced into something plausible.
    ///
    /// The mono case is the one that matters. `conv_pre` is 128 channels wide and a mono
    /// mel is 64, so a port that flattened whatever it was given would either fail on a
    /// matmul deep inside or — worse, if it "helpfully" duplicated — emit stereo audio
    /// from a mono analysis and sound almost right. Contract 9b records that the mel's
    /// stereo-ness does not follow from anything upstream, so it is checked at the door.
    @Test("malformed mel input throws")
    func malformedInput() throws {
        let vocoder = try Self.syntheticVocoder()
        let bins = vocoder.config.numMels

        #expect(throws: AudioVocoder.Failure.self) {           // 3D, no channel axis
            _ = try vocoder.waveform(fromMel: MLXArray.zeros([1, 6, bins], dtype: .float32))
        }
        #expect(throws: AudioVocoder.Failure.self) {           // mono
            _ = try vocoder.waveform(
                fromMel: MLXArray.zeros([1, 1, 6, bins], dtype: .float32))
        }
        #expect(throws: AudioVocoder.Failure.self) {           // wrong bin count
            _ = try vocoder.waveform(
                fromMel: MLXArray.zeros([1, 2, 6, bins / 2], dtype: .float32))
        }
        #expect(throws: AudioVocoder.Failure.self) {           // no frames
            _ = try vocoder.waveform(
                fromMel: MLXArray.zeros([1, 2, 0, bins], dtype: .float32))
        }
    }

    /// Loading is fail-closed in all three directions.
    ///
    /// This is the same accounting ``checkpointKeyAccounting()`` does against the real
    /// file, asserted as a *property of the initialiser* so it holds on machines without
    /// the checkpoint. A loader that returned its input unchanged when a weight was
    /// missing would convert a structural mistake into slightly wrong audio; this one
    /// refuses instead.
    @Test("loading rejects missing, extra and mis-shaped tensors")
    func loadingIsFailClosed() throws {
        let config = Self.syntheticConfiguration()
        let good = Self.syntheticWeights(for: config)

        var missing = good
        missing.removeValue(forKey: "vocoder.resblocks.3.convs2.1.bias")
        #expect(throws: AudioVocoder.Failure.self) {
            _ = try AudioVocoder(weights: missing, configuration: config)
        }

        var extra = good
        extra["vocoder.resblocks.99.convs1.0.weight"] = MLXArray.zeros([4, 4, 3])
        #expect(throws: AudioVocoder.Failure.self) {
            _ = try AudioVocoder(weights: extra, configuration: config)
        }

        // Transposing a ConvTranspose1d weight the way a Conv1d weight would be is the
        // realistic version of this: on a stage whose channel counts differ it is a shape
        // error, and this is where it must surface.
        var misshaped = good
        let w = good["vocoder.ups.0.weight"]!
        misshaped["vocoder.ups.0.weight"] = w.transposed(1, 0, 2)
        #expect(throws: AudioVocoder.Failure.self) {
            _ = try AudioVocoder(weights: misshaped, configuration: config)
        }

        // And the good set loads, so the three rejections are not vacuous.
        _ = try AudioVocoder(weights: good, configuration: config)
    }

    /// A config whose two generators disagree about the rate change is rejected at load,
    /// not at the `residual + skip` add several seconds later.
    @Test("a config that cannot close the rate loop is rejected")
    func inconsistentRatesRejected() throws {
        let base = Self.syntheticConfiguration()
        let broken = AudioVocoder.Configuration(
            vocoder: base.vocoder, bwe: base.bwe,
            inputSamplingRate: 16000, outputSamplingRate: 44100,   // not an integer ratio
            hopLength: base.hopLength, nFFT: base.nFFT, numMels: base.numMels)
        #expect(throws: AudioVocoder.Failure.self) {
            _ = try AudioVocoder(weights: Self.syntheticWeights(for: broken),
                                 configuration: broken)
        }
    }

    // MARK: - Synthetic fixtures

    /// The real rate structure at 1/500th the width.
    ///
    /// The rates, kernels, hop, FFT size and mel count are the checkpoint's, because those
    /// are what the algebra tests are about. Only the channel counts shrink, and only down
    /// to the smallest values that still halve the right number of times: 128 → 2 over six
    /// stages for the vocoder, 64 → 2 over five for the BWE. `resblock_kernel_sizes` is
    /// cut to one entry, which makes the per-stage mean a mean of one — deliberately, so
    /// that the tests that need speed are not paying for three.
    static func syntheticConfiguration() -> AudioVocoder.Configuration {
        AudioVocoder.Configuration(
            vocoder: AudioVocoder.GeneratorConfiguration(
                upsampleInitialChannel: 128,
                upsampleRates: [5, 2, 2, 2, 2, 2],
                upsampleKernelSizes: [11, 4, 4, 4, 4, 4],
                resblockKernelSizes: [3],
                resblockDilationSizes: [[1, 3, 5]],
                useTanhAtFinal: false, useBiasAtFinal: false, applyFinalActivation: true),
            bwe: AudioVocoder.GeneratorConfiguration(
                upsampleInitialChannel: 64,
                upsampleRates: [6, 5, 2, 2, 2],
                upsampleKernelSizes: [12, 11, 4, 4, 4],
                resblockKernelSizes: [3],
                resblockDilationSizes: [[1, 3, 5]],
                useTanhAtFinal: false, useBiasAtFinal: false, applyFinalActivation: false),
            inputSamplingRate: 16000, outputSamplingRate: 48000,
            hopLength: 80, nFFT: 512, numMels: 64)
    }

    static func syntheticVocoder() throws -> AudioVocoder {
        let config = syntheticConfiguration()
        return try AudioVocoder(weights: syntheticWeights(for: config),
                                configuration: config)
    }

    /// Deterministic pseudo-random weights at exactly the shapes the module tree predicts.
    ///
    /// Built *from* ``AudioVocoder/parameterShapes(for:)`` rather than by hand, which is
    /// what makes the synthetic tests structural rather than circular: they exercise
    /// whatever tree the config describes, and it is
    /// ``AudioVocoderTests/checkpointKeyAccounting()`` — comparing that same prediction
    /// against the real file — that establishes the tree is 2.5's.
    ///
    /// The magnitudes are small (0.02) so six stages of convolution neither vanish nor
    /// saturate the output clamp; the resampling filters are made non-negative and
    /// normalised so they behave like lowpasses rather than amplifying; and `mel_basis` is
    /// non-negative so the log is taken of something a real filterbank could produce.
    /// None of that is claimed to resemble the trained values.
    static func syntheticWeights(for config: AudioVocoder.Configuration)
        -> [String: MLXArray] {
        var noise = Noise(state: 0xA17D_1E55)
        var weights: [String: MLXArray] = [:]
        for (name, shape) in AudioVocoder.parameterShapes(for: config).sorted(by: {
            $0.key < $1.key
        }) {
            let count = shape.reduce(1, *)
            var values = [Float](repeating: 0, count: count)
            if name.hasSuffix(".filter") {
                // A normalised non-negative lowpass, so the anti-aliased activation's
                // round trip neither rings nor gains.
                var total: Float = 0
                for i in 0 ..< count {
                    values[i] = abs(noise.next()) + 0.1
                    total += values[i]
                }
                for i in 0 ..< count { values[i] /= total }
            } else if name.hasSuffix("mel_basis") {
                for i in 0 ..< count { values[i] = abs(noise.next()) * 0.05 }
            } else if name.contains("act.") {
                // alpha/beta are log-scale, so they belong near zero.
                for i in 0 ..< count { values[i] = noise.next() * 0.3 }
            } else {
                for i in 0 ..< count { values[i] = noise.next() * 0.02 }
            }
            weights[name] = MLXArray(values, shape)
        }
        return weights
    }

    /// A mel in the range a real log-mel occupies — `AudioVAEDecoder`'s own comment puts
    /// it at roughly `[-10.9, -2.9]`. Feeding zeros instead would exercise the biases only.
    static func syntheticMel(frames: Int, bins: Int, seed: UInt64) -> MLXArray {
        var noise = Noise(state: seed)
        let count = 2 * frames * bins
        var values = [Float](repeating: 0, count: count)
        for i in 0 ..< count { values[i] = -6.9 + noise.next() * 4.0 }
        return MLXArray(values, [1, 2, frames, bins])
    }

    /// A 64-bit LCG. Deterministic across platforms and runs, which is the only property
    /// wanted here — `MLXRandom` would tie the fixtures to a generator whose stream is not
    /// this test's to pin.
    struct Noise {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let bits = UInt32(truncatingIfNeeded: state >> 32)
            return Float(bits) / Float(UInt32.max) * 2 - 1
        }
    }
}

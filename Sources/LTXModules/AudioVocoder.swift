// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The audio **vocoder**: log-mel spectrogram → 48 kHz stereo waveform.
///
/// This is the stage ``AudioVAEDecoder`` deliberately stops short of. Together they are
/// the whole audio branch: the VAE decoder turns a latent into a stereo log-mel, and this
/// turns that mel into samples a file can hold.
///
/// ## Three models, not one
///
/// This is a BigVGAN v2 generator followed by a *bandwidth extension* stage, not a plain
/// BigVGAN. The full path is three models:
///
/// ```
/// mel [B, 2, T, 64]  --vocoder-->        16 kHz stereo [B, 2, 160T]
///                    --mel_stft-->       mel [B, 2, 2T, 64]   (n_fft 512, hop 80)
///                    --bwe_generator-->  48 kHz residual [B, 2, 480T]
///                    + sinc-resampled skip, clamped                 -> [B, 2, 480T]
/// ```
///
/// The two generators are the same architecture at different rates, and the BWE one
/// predicts a *residual* that is added to a 3× resampled copy of the 16 kHz signal.
/// Stopping after the first generator produces audio that plays, is the right length at
/// 16 kHz, and is missing the entire top two octaves.
///
/// ## Contract 11: the waveform is not a thing to compare numerically
///
/// `FRAGILE_CONTRACTS.md` #11. Identical mel inputs can produce different waveforms
/// wherever the transposed convolutions are free to pick their own accumulation order, so
/// the mel is the boundary to compare at and the waveform is judged by ear. Nothing here
/// is a source of that spread — there is no noise injection, no phase initialisation, no
/// dropout, and this forward pass is deterministic — but determinism is not correctness.
/// What this file does establish is structural: every one of the checkpoint's 1227
/// `vocoder.*` tensors is consumed at the shape the config predicts, and the sample-rate
/// algebra is exact.
///
/// ## Precision: this whole model runs in fp32, unlike its neighbour
///
/// ``AudioVAEDecoder`` casts activations to bf16 at every boundary because that decoder
/// runs in bf16. This one is the opposite: bf16 accumulation errors compound through 108
/// sequential convolutions and degrade the spectral metrics by 40–90%, so the *entire*
/// pass runs in fp32, `exp(alpha)` inside SnakeBeta included.
///
/// So the fp32 is not per-op here: it is the whole model, and the weights are widened once
/// at load. Getting this backwards — storing bf16 activations the way the VAE decoder
/// does — is the failure this paragraph exists to prevent, and it produces audio that is
/// merely somewhat worse rather than obviously broken.
///
/// ## Layout
///
/// MLX convolves `NLC` with weights `[out, kernel, in]`; the checkpoint is PyTorch `NCL`
/// with `[out, in, kernel]` (and `[in, out, kernel]` for the transposed convolutions —
/// a different permutation, which is the one that silently produces a same-shaped wrong
/// answer when the two are confused). **Everything inside this type is `[B, L, C]`**, and
/// the only `NCL` tensors are the public return values.
public struct AudioVocoder {

    // MARK: - Configuration

    /// One generator's config. Both generators are the same architecture.
    public struct GeneratorConfiguration: Equatable, Sendable {
        public let upsampleInitialChannel: Int
        /// Product is the total time upsampling: 160 for the vocoder (matching the mel
        /// hop length), 240 for the BWE (3× rate change times the 80-sample BWE hop).
        public let upsampleRates: [Int]
        public let upsampleKernelSizes: [Int]
        public let resblockKernelSizes: [Int]
        /// One dilation triple per resblock kernel size. An AMP block has exactly three
        /// conv pairs and reads `dilation[0..2]` positionally, so three is the only
        /// length that builds.
        public let resblockDilationSizes: [[Int]]
        public let useTanhAtFinal: Bool
        public let useBiasAtFinal: Bool
        /// **Not read from the config dict.** It is a property of the generator's role:
        /// `true` for the vocoder, `false` for the BWE generator, whose output is a
        /// residual and must not be clamped before the skip is added. The checkpoint's
        /// `bwe.apply_final_activation: false` agrees, but nothing reads it.
        public let applyFinalActivation: Bool

        public var totalUpsampleFactor: Int { upsampleRates.reduce(1, *) }
        public var finalChannels: Int {
            upsampleInitialChannel / (1 << upsampleRates.count)
        }
        /// Channels leaving upsample stage `index`.
        public func channels(afterStage index: Int) -> Int {
            upsampleInitialChannel / (1 << (index + 1))
        }
        /// Resblocks per stage = one per entry in `resblockKernelSizes`, all evaluated on
        /// the same input and **averaged**, not chained.
        public var resblocksPerStage: Int { resblockKernelSizes.count }

        public init(upsampleInitialChannel: Int, upsampleRates: [Int],
                    upsampleKernelSizes: [Int], resblockKernelSizes: [Int],
                    resblockDilationSizes: [[Int]], useTanhAtFinal: Bool,
                    useBiasAtFinal: Bool, applyFinalActivation: Bool) {
            self.upsampleInitialChannel = upsampleInitialChannel
            self.upsampleRates = upsampleRates
            self.upsampleKernelSizes = upsampleKernelSizes
            self.resblockKernelSizes = resblockKernelSizes
            self.resblockDilationSizes = resblockDilationSizes
            self.useTanhAtFinal = useTanhAtFinal
            self.useBiasAtFinal = useBiasAtFinal
            self.applyFinalActivation = applyFinalActivation
        }
    }

    /// `config.vocoder`, read from the checkpoint's `__metadata__` rather than declared.
    ///
    /// The repo's standing rule, and it bites hardest here: a 2.3 checkpoint carries a
    /// *flat* vocoder config with `resblock: "1"` and rates `[6,5,2,2,2]`. Hardcoding
    /// 2.5's `[5,2,2,2,2,2]` would decode a 2.3 file at 240× instead of 160× — a
    /// waveform 1.5× too long at the wrong pitch, from a load that raised nothing.
    public struct Configuration: Equatable, Sendable {
        public let vocoder: GeneratorConfiguration
        public let bwe: GeneratorConfiguration
        /// 16000. The rate the first generator emits.
        public let inputSamplingRate: Int
        /// 48000. The rate the whole thing emits.
        public let outputSamplingRate: Int
        /// 80. The BWE mel's hop, **not** the audio VAE's 160.
        public let hopLength: Int
        public let nFFT: Int
        public let numMels: Int

        public var resampleRatio: Int { outputSamplingRate / inputSamplingRate }

        public init(vocoder: GeneratorConfiguration, bwe: GeneratorConfiguration,
                    inputSamplingRate: Int, outputSamplingRate: Int, hopLength: Int,
                    nFFT: Int, numMels: Int) {
            self.vocoder = vocoder
            self.bwe = bwe
            self.inputSamplingRate = inputSamplingRate
            self.outputSamplingRate = outputSamplingRate
            self.hopLength = hopLength
            self.nFFT = nFFT
            self.numMels = numMels
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case missing(String)
        case unconsumed([String])
        case shape(String, got: [Int], expected: String)
        case unsupported(String)
        case configuration(String)

        public var description: String {
            switch self {
            case let .missing(name): return "weight \(name) is absent"
            case let .unconsumed(names):
                return "\(names.count) checkpoint tensors are not consumed by the module "
                    + "tree, first few: \(names.prefix(5).joined(separator: ", "))"
            case let .shape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case let .unsupported(why): return why
            case let .configuration(why): return "checkpoint config: \(why)"
            }
        }
    }

    // MARK: - Constants the architecture fixes

    /// `conv_pre`'s input width, fixed at 128: every production checkpoint is stereo, and
    /// that is 2 stereo channels x 64 mel bins each.
    ///
    /// Both generators use it, so the BWE's `num_mels` must be 64 for the model to build
    /// at all; ``configuration(from:)`` checks that rather than leaving it to a shape
    /// error 200 tensors later.
    public static let stereoInputChannels = 128

    /// `conv_pre` and `conv_post` are `kernel_size=7, padding=3`. Not configurable, so
    /// not in `Configuration`.
    static let stemKernelSize = 7

    /// An AMP block has exactly three `convs1`/`convs2`/`acts1`/`acts2`, one per entry of
    /// its dilation triple.
    static let convsPerBlock = 3

    /// The SnakeBeta epsilon. Guards `1/(beta + eps)`; beta is `exp(·)` and therefore
    /// already strictly positive, so this only ever matters for a beta that underflowed.
    static let snakeEpsilon: Float = 1e-9

    /// The anti-aliased activation's resampling ratio, up and down alike. 2 is the only
    /// value in this checkpoint — confirmed by every stored filter being 12 taps, which
    /// is the shared default kernel size at ratio 2.
    static let activationResampleRatio = 2

    /// The mel floor: the analysis takes `log(max(mel, 1e-5))`.
    static let melFloor: Float = 1e-5

    // MARK: - State

    /// Every `vocoder.*` tensor, prefix stripped once, widened to fp32.
    ///
    /// `inverse_basis` is the one checkpoint tensor deliberately absent: it is the iSTFT
    /// basis, this direction never runs an inverse STFT, and holding 263k unused floats
    /// resident for symmetry is not a virtue. Its *presence and shape* are still checked
    /// at load, so it stays inside the accounting rather than becoming a key nobody looks
    /// at.
    public let weights: [String: MLXArray]
    public let config: Configuration

    /// The 3× sinc resampler's filter — **computed, not loaded.**
    ///
    /// This is the one buffer in the whole model that is not in the checkpoint — it is
    /// non-persistent, and the enumeration confirms it: there is no `resampler.filter`
    /// among the 1227 keys. Everything else (`*.upsample.filter`,
    /// `*.downsample.lowpass.filter`) *is* stored and is loaded verbatim, because
    /// recomputing a Kaiser window needs a Bessel `i0` and would be a pure source of
    /// divergence.
    ///
    /// Recomputing *this* one is safe in a way recomputing those would not be: the
    /// Hann-window form is closed-form — `cos²` window times `sinc`, no Bessel function,
    /// no iteration — so the only difference from a stored copy would be last-bit rounding
    /// from evaluating it in `Double`. See ``hannResamplerFilter``.
    let resamplerFilter: MLXArray
    let resamplerPad: Int
    let resamplerPadLeft: Int
    let resamplerPadRight: Int

    public static let checkpointPrefix = "vocoder."

    // MARK: - Loading

    /// Load the vocoder half of the audio VAE checkpoint.
    ///
    /// The file is 1329 tensors; 1227 of them are this model and the other 102 are
    /// ``AudioVAEDecoder``'s. The split is by prefix, and it strips `vocoder.` **once**,
    /// so `vocoder.vocoder.conv_pre` becomes `vocoder.conv_pre` and not `conv_pre`. Stripping
    /// it twice would merge the two generators' key spaces and load the BWE's 512-channel
    /// stem into the vocoder's 1536-channel one.
    public init(checkpoint url: URL) throws {
        let header = try SafetensorsHeader.read(from: url)
        let config = try Self.configuration(from: header)

        let all = try MLX.loadArrays(url: url)
        var vocoder: [String: MLXArray] = [:]
        for (name, value) in all where name.hasPrefix(Self.checkpointPrefix) {
            vocoder[String(name.dropFirst(Self.checkpointPrefix.count))] = value
        }
        guard !vocoder.isEmpty else { throw Failure.missing(Self.checkpointPrefix + "*") }
        try self.init(weights: vocoder, configuration: config)
    }

    /// Build from an already-filtered weight dictionary.
    ///
    /// Exists so the module tree can be exercised without the 365 MB checkpoint, and so
    /// the *whole* consumed/missing accounting happens here rather than only in a test:
    /// the initialiser is where a structural mistake should surface, because a wrong tree
    /// that merely leaves tensors unused would otherwise run to completion and emit
    /// plausible audio.
    public init(weights: [String: MLXArray], configuration: Configuration) throws {
        self.config = configuration

        let expected = Self.parameterShapes(for: configuration)
        var kept: [String: MLXArray] = [:]
        kept.reserveCapacity(expected.count)
        for (name, shape) in expected {
            guard let value = weights[name] else { throw Failure.missing(name) }
            guard value.shape == shape else {
                throw Failure.shape(name, got: value.shape, expected: "\(shape)")
            }
            // fp32 for the whole model — see the type comment. Done once, here, so no
            // forward path has to remember to widen anything.
            if name != Self.unusedInverseBasisKey {
                kept[name] = value.asType(.float32)
            }
        }
        let extra = Set(weights.keys).subtracting(expected.keys).sorted()
        guard extra.isEmpty else { throw Failure.unconsumed(extra) }
        self.weights = kept

        let ratio = configuration.resampleRatio
        guard ratio >= 1, ratio * configuration.inputSamplingRate
                == configuration.outputSamplingRate
        else {
            throw Failure.unsupported(
                "output rate \(configuration.outputSamplingRate) is not an integer multiple "
                + "of input rate \(configuration.inputSamplingRate); UpSample1d only "
                + "resamples by an integer ratio")
        }
        let (filter, pad, padLeft, padRight) = Self.hannResamplerFilter(ratio: ratio)
        self.resamplerFilter = filter
        self.resamplerPad = pad
        self.resamplerPadLeft = padLeft
        self.resamplerPadRight = padRight
    }

    static let unusedInverseBasisKey = "mel_stft.stft_fn.inverse_basis"

    /// Parse `config.vocoder` and reject every shape this type does not implement, at
    /// load time, by name.
    ///
    /// Each field is *required*. Defaulting a missing key to its 2.2-era value —
    /// `upsample_initial_channel: 1024`, rates `[6,5,2,2,2]`, kernels `[16,15,8,4,4]` —
    /// would not fail, it would build a different network. Here it throws.
    public static func configuration(from header: SafetensorsHeader) throws -> Configuration {
        guard let root = header.metadataJSON("config"),
              let outer = root["vocoder"] as? [String: Any]
        else { throw Failure.configuration("config.vocoder is absent") }

        guard let vocoderCfg = outer["vocoder"] as? [String: Any],
              let bweCfg = outer["bwe"] as? [String: Any]
        else {
            // A config without "bwe" is a pre-2.3 flat checkpoint: a bare vocoder with
            // `resblock: "1"` — ResBlock1, LeakyReLU, no anti-aliased activation. That is
            // a different architecture, not a special case of this one.
            throw Failure.unsupported(
                "config.vocoder has no \"bwe\" section, which marks a pre-2.3 flat "
                + "checkpoint using ResBlock1/LeakyReLU; this port implements the 2.3+ "
                + "AMP1 + bandwidth-extension model only")
        }

        func generator(_ cfg: [String: Any], _ label: String,
                       applyFinalActivation: Bool) throws -> GeneratorConfiguration {
            func require<T>(_ key: String, _ type: T.Type) throws -> T {
                guard let v = cfg[key] as? T else {
                    throw Failure.configuration(
                        "vocoder.\(label).\(key) is absent or not a \(T.self); the "
                        + "reference would silently substitute a 2.2-era default")
                }
                return v
            }
            func check(_ key: String, _ expected: String) throws {
                let actual = cfg[key] as? String
                guard actual == expected else {
                    throw Failure.unsupported(
                        "vocoder.\(label).\(key) is \(actual.map { "\"\($0)\"" } ?? "absent"), "
                        + "expected \"\(expected)\"")
                }
            }
            // Three values that must be what they are; anything else is another model.
            try check("resblock", "AMP1")
            try check("activation", "snakebeta")
            guard (cfg["stereo"] as? Bool) == true else {
                throw Failure.unsupported(
                    "vocoder.\(label).stereo is not true; a mono checkpoint's conv_pre "
                    + "would be 64 channels wide, not the hardcoded 128")
            }

            let rates: [Int] = try require("upsample_rates", [Int].self)
            let kernels: [Int] = try require("upsample_kernel_sizes", [Int].self)
            let dilations: [[Int]] = try require("resblock_dilation_sizes", [[Int]].self)
            let rbKernels: [Int] = try require("resblock_kernel_sizes", [Int].self)
            guard rates.count == kernels.count else {
                throw Failure.configuration("vocoder.\(label): \(rates.count) upsample "
                    + "rates against \(kernels.count) kernel sizes")
            }
            guard rbKernels.count == dilations.count else {
                throw Failure.configuration("vocoder.\(label): \(rbKernels.count) resblock "
                    + "kernel sizes against \(dilations.count) dilation lists")
            }
            guard dilations.allSatisfy({ $0.count == convsPerBlock }) else {
                throw Failure.unsupported("vocoder.\(label).resblock_dilation_sizes has a "
                    + "list that is not \(convsPerBlock) long; AMPBlock1 indexes "
                    + "dilation[0..2] literally and cannot build any other width")
            }
            for (i, k) in kernels.enumerated() where (k - rates[i]) % 2 != 0 {
                // The padding is `(kernel_size - stride) / 2`, floored. An odd difference
                // means the transposed convolution's output is one sample short of
                // `rate × L`, and the shortfall accumulates across stages.
                throw Failure.unsupported("vocoder.\(label): upsample \(i) has kernel \(k) "
                    + "and rate \(rates[i]); (kernel - rate) must be even for the "
                    + "transposed convolution to land on an exact multiple")
            }
            let initial: Int = try require("upsample_initial_channel", Int.self)
            guard initial % (1 << rates.count) == 0 else {
                throw Failure.configuration("vocoder.\(label): \(initial) initial channels "
                    + "does not halve \(rates.count) times")
            }
            return GeneratorConfiguration(
                upsampleInitialChannel: initial,
                upsampleRates: rates,
                upsampleKernelSizes: kernels,
                resblockKernelSizes: rbKernels,
                resblockDilationSizes: dilations,
                useTanhAtFinal: try require("use_tanh_at_final", Bool.self),
                useBiasAtFinal: try require("use_bias_at_final", Bool.self),
                applyFinalActivation: applyFinalActivation)
        }

        func requireInt(_ key: String) throws -> Int {
            guard let v = bweCfg[key] as? Int else {
                throw Failure.configuration("vocoder.bwe.\(key) is absent or not an integer")
            }
            return v
        }
        let numMels = try requireInt("num_mels")
        let nFFT = try requireInt("n_fft")
        // The mel analysis is built with `win_length = n_fft`, so `win_size` is read by
        // nobody. Checking it agrees keeps a future checkpoint that means something by it
        // from being silently ignored.
        if let winSize = bweCfg["win_size"] as? Int, winSize != nFFT {
            throw Failure.unsupported("vocoder.bwe.win_size \(winSize) differs from n_fft "
                + "\(nFFT); the analysis mel uses win_length = n_fft and ignores "
                + "win_size entirely")
        }
        guard 2 * numMels == stereoInputChannels else {
            throw Failure.unsupported("vocoder.bwe.num_mels is \(numMels); conv_pre's "
                + "input width is fixed at \(stereoInputChannels), which only works "
                + "for 2 x 64")
        }

        return Configuration(
            // `apply_final_activation` is a property of the role, not a config key: true
            // for the vocoder (clamped to [-1,1] because `use_tanh_at_final` is false),
            // false for the BWE generator, whose output is a residual and must not be
            // clamped before the skip is added.
            vocoder: try generator(vocoderCfg, "vocoder", applyFinalActivation: true),
            bwe: try generator(bweCfg, "bwe", applyFinalActivation: false),
            inputSamplingRate: try requireInt("input_sampling_rate"),
            outputSamplingRate: try requireInt("output_sampling_rate"),
            hopLength: try requireInt("hop_length"),
            nFFT: nFFT,
            numMels: numMels)
    }

    // MARK: - The module tree, derived from the config

    /// Every tensor this type reads, and the PyTorch shape it must have.
    ///
    /// The structural claim, in one place and in a form a test can compare against the
    /// checkpoint's own key list. The count it predicts for 2.5 is 1227,
    /// which is the count the file has; that agreement is not decorative, because a
    /// module tree can be wrong in ways that only show up as *unused* tensors, and unused
    /// tensors do not raise anything.
    public static func parameterShapes(for config: Configuration) -> [String: [Int]] {
        var shapes: [String: [Int]] = [:]
        let freqBins = config.nFFT / 2 + 1
        shapes["mel_stft.mel_basis"] = [config.numMels, freqBins]
        // The DFT bases are stored as a conv1d kernel: real rows then imaginary rows,
        // already multiplied by the causal Hann window. They are stored rather than
        // recomputed so that the mel values fed to the BWE generator come from the exact
        // bf16 bases it was trained on.
        shapes["mel_stft.stft_fn.forward_basis"] = [2 * freqBins, 1, config.nFFT]
        shapes[unusedInverseBasisKey] = [2 * freqBins, 1, config.nFFT]
        generatorShapes("vocoder.", config.vocoder, into: &shapes)
        generatorShapes("bwe_generator.", config.bwe, into: &shapes)
        return shapes
    }

    static func generatorShapes(_ p: String, _ g: GeneratorConfiguration,
                                into shapes: inout [String: [Int]]) {
        let initial = g.upsampleInitialChannel
        shapes[p + "conv_pre.weight"] = [initial, stereoInputChannels, stemKernelSize]
        shapes[p + "conv_pre.bias"] = [initial]

        for (i, kernel) in g.upsampleKernelSizes.enumerated() {
            // `nn.ConvTranspose1d(in, out, k)` stores `[in, out, k]` — in-major, the
            // opposite of `nn.Conv1d`. This is the single most dangerous transpose in the
            // file: for the vocoder's stages 1..5 the two are 768 vs 384 and would fail
            // loudly, but any square stage would load and convolve nonsense.
            shapes[p + "ups.\(i).weight"] =
                [initial / (1 << i), g.channels(afterStage: i), kernel]
            shapes[p + "ups.\(i).bias"] = [g.channels(afterStage: i)]
        }

        // Resblocks are numbered **stage-major**: `stage * len(kernels) + j`, so the
        // vocoder's 18 run 768, 768, 768, 384, ... and not 768, 384, 192, .... Laying them
        // out kernel-major loads every tensor and mismatches every channel count from
        // index 1 onward — which is a shape error at load here, and would have been
        // silently averaged nonsense had the channel counts happened to line up.
        var index = 0
        for stage in 0 ..< g.upsampleRates.count {
            let channels = g.channels(afterStage: stage)
            for kernel in g.resblockKernelSizes {
                // The dilations change the receptive field, not the shapes, so they play
                // no part here; `ampBlock` is where they are read.
                let block = p + "resblocks.\(index)."
                for m in 0 ..< convsPerBlock {
                    shapes[block + "convs1.\(m).weight"] = [channels, channels, kernel]
                    shapes[block + "convs1.\(m).bias"] = [channels]
                    shapes[block + "convs2.\(m).weight"] = [channels, channels, kernel]
                    shapes[block + "convs2.\(m).bias"] = [channels]
                    for acts in ["acts1", "acts2"] {
                        activationShapes(block + "\(acts).\(m).", channels: channels,
                                         into: &shapes)
                    }
                }
                index += 1
            }
        }

        activationShapes(p + "act_post.", channels: g.finalChannels, into: &shapes)
        // Stereo out, and `bias=use_bias_at_final` — false on 2.5, so `conv_post.bias`
        // is genuinely absent rather than missing. Predicting it unconditionally would
        // fail the load on a correct checkpoint.
        shapes[p + "conv_post.weight"] = [2, g.finalChannels, stemKernelSize]
        if g.useBiasAtFinal { shapes[p + "conv_post.bias"] = [2] }
    }

    /// One `Activation1d`: the SnakeBeta parameters plus the two **stored** anti-aliasing
    /// filters.
    ///
    /// The filters are persistent buffers, so they live in the checkpoint and are loaded
    /// rather than recomputed — see ``resamplerFilter`` for the single exception and why
    /// it is one.
    static func activationShapes(_ p: String, channels: Int,
                                 into shapes: inout [String: [Int]]) {
        shapes[p + "act.alpha"] = [channels]
        shapes[p + "act.beta"] = [channels]
        // `6 * ratio / 2 * 2` — the up/downsamplers' shared default kernel size, 12 at
        // ratio 2. Written out rather than hardcoded because it is the only thing tying
        // the filter length to the resampling ratio, and the checkpoint's `[1, 1, 12]`
        // buffers are what confirm the ratio is 2 in the first place.
        let taps = 6 * activationResampleRatio / 2 * 2
        shapes[p + "upsample.filter"] = [1, 1, taps]
        shapes[p + "downsample.lowpass.filter"] = [1, 1, taps]
    }

    // MARK: - Public inference

    /// Mel frames → 48 kHz samples per channel. Exact, not approximate: every stage is an
    /// integer rate change with no boundary trimming.
    public func sampleCount(melFrames: Int) -> Int {
        baseRateSampleCount(melFrames: melFrames) * config.resampleRatio
    }

    /// Mel frames → 16 kHz samples per channel.
    public func baseRateSampleCount(melFrames: Int) -> Int {
        melFrames * config.vocoder.totalUpsampleFactor
    }

    /// Log-mel `[B, 2, T, 64]` (the shape ``AudioVAEDecoder/decode(_:)`` returns) → 48 kHz
    /// stereo waveform `[B, 2, 480T]`, clamped to `[-1, 1]`.
    public func waveform(fromMel mel: MLXArray) throws -> MLXArray {
        try bandwidthExtended(baseRate(fromMel: mel)).transposed(0, 2, 1)
    }

    /// Log-mel → the intermediate **16 kHz** stereo waveform `[B, 2, 160T]`.
    ///
    /// Exposed because it is a real boundary — it is what the bandwidth-extension stage
    /// re-analyses into its own mel — and because it is the only place the rate algebra
    /// can be observed without paying for the BWE generator.
    public func baseRateWaveform(fromMel mel: MLXArray) throws -> MLXArray {
        try baseRate(fromMel: mel).transposed(0, 2, 1)
    }

    /// `[B, L, 2]`, channels-last, 16 kHz.
    func baseRate(fromMel mel: MLXArray) throws -> MLXArray {
        guard mel.ndim == 4, mel.shape[1] == 2, mel.shape[3] == config.numMels,
              2 * mel.shape[3] == Self.stereoInputChannels
        else {
            throw Failure.shape("mel", got: mel.shape,
                                expected: "[B, 2, frames, \(config.numMels)]")
        }
        guard mel.shape[2] > 0 else {
            throw Failure.shape("mel", got: mel.shape, expected: "at least one frame")
        }
        return try generator(mel.asType(.float32), "vocoder.", config.vocoder)
    }

}

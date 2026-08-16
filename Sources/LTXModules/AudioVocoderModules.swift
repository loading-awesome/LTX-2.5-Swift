// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX

/// ``AudioVocoder``'s forward arithmetic, split out of the type's own file purely for
/// length — the loading, configuration and module-tree derivation live next door in
/// `AudioVocoder.swift`.
///
/// The division is by responsibility rather than by convenience: nothing here reads the
/// checkpoint's metadata or decides what the tree *is*. Everything here takes the tree as
/// given and evaluates it, so a structural mistake belongs on the other side of this line
/// and an arithmetic one belongs on this side.
extension AudioVocoder {
    // MARK: - Bandwidth extension

    /// 16 kHz `[B, L, 2]` → 48 kHz `[B, 3L, 2]`.
    ///
    /// The BWE is not a second vocoder run: it re-analyses the 16 kHz signal into its own
    /// mel (a different one — 512-point, hop 80, 64 bins, against the audio VAE's
    /// 1024-point hop 160), predicts a **residual** from it, and adds that to a
    /// sinc-resampled copy of the input. Treating the generator's output as the final
    /// signal, rather than as a correction to the skip, discards the entire low band.
    func bandwidthExtended(_ x: MLXArray) throws -> MLXArray {
        let lowRateLength = x.shape[1]
        let outputLength = lowRateLength * config.resampleRatio

        // Pad to a whole number of BWE hops so the mel frame count is exact. On this
        // checkpoint the vocoder's 160× and the BWE's 80-sample hop make the remainder
        // always zero; the branch exists because a future rate pair could make it fire.
        var padded = x
        let remainder = lowRateLength % config.hopLength
        if remainder != 0 {
            padded = MLX.padded(padded, widths: [
                IntOrPair(0), IntOrPair((0, config.hopLength - remainder)), IntOrPair(0),
            ])
        }

        let mel = melSpectrogram(of: padded)
        let residual = try generator(mel, "bwe_generator.", config.bwe)
        let skip = resample(padded)
        guard residual.shape == skip.shape else {
            throw Failure.shape("bwe residual", got: residual.shape,
                                expected: "\(skip.shape) (the resampled skip)")
        }
        // The clamp makes the output trivially bounded, which is why a "waveform is
        // bounded" assertion proves less than it looks like it does.
        let mixed = MLX.clip(residual + skip, min: -1, max: 1)
        return mixed[0..., 0 ..< outputLength, 0...]
    }

    /// Causal log-mel of a `[B, L, C]` waveform → `[B, C, frames, numMels]`.
    ///
    /// Shaped for ``generator(_:_:_:)`` directly. The conventional layout for a mel
    /// analysis is `(B, C, n_mels, T)`, which the generator would only transpose back;
    /// producing its layout here is the same arithmetic without the round trip.
    ///
    /// A full mel analysis also yields phase and energy. Nobody on this path reads either,
    /// so they are not computed.
    func melSpectrogram(of x: MLXArray) -> MLXArray {
        let batch = x.shape[0], channels = x.shape[2]
        var flat = Self.channelsToBatch(x)                  // [B*C, L, 1]

        // Causal: the whole `win_length - hop_length` overlap goes on the **left**, with
        // zeros. Splitting it symmetrically (which is what every non-causal STFT does)
        // gives the identical frame count and lets every frame see the future.
        //
        // `nFFT` stands in for `win_length`: the analysis is built with
        // `win_length = n_fft` and never reads the config's own `win_size`.
        // ``AudioVocoder/configuration(from:)`` refuses a checkpoint where the two
        // disagree rather than quietly ignoring one of them.
        let leftPad = max(0, config.nFFT - config.hopLength)
        flat = MLX.padded(flat, widths: [
            IntOrPair(0), IntOrPair((leftPad, 0)), IntOrPair(0),
        ])

        // Force-unwrapped deliberately: ``AudioVocoder/init(weights:configuration:)``
        // checks every key in ``AudioVocoder/parameterShapes(for:)`` before this type
        // exists, so an absent basis here would mean the initialiser's accounting is
        // broken — which should crash, not degrade into a silent fallback.
        let basis = weights["mel_stft.stft_fn.forward_basis"]!
        let spec = MLX.conv1d(flat, basis.transposed(0, 2, 1), stride: config.hopLength)
        let freqBins = spec.shape[2] / 2
        let real = spec[0..., 0..., 0 ..< freqBins]
        let imaginary = spec[0..., 0..., freqBins...]
        let magnitude = MLX.sqrt(real * real + imaginary * imaginary)

        let melBasis = weights["mel_stft.mel_basis"]!
        let mel = MLX.matmul(magnitude, melBasis.transposed(1, 0))
        let logMel = MLX.log(MLX.maximum(mel, Self.melFloor))
        return logMel.reshaped([batch, channels, logMel.shape[1], logMel.shape[2]])
    }

    /// The skip path: integer-ratio sinc upsampling of `[B, L, C]` to `[B, ratio·L, C]`.
    func resample(_ x: MLXArray) -> MLXArray {
        Self.upsample1d(x, filter: resamplerFilter, ratio: config.resampleRatio,
                        pad: resamplerPad, padLeft: resamplerPadLeft,
                        padRight: resamplerPadRight)
    }

    /// The Hann-windowed sinc filter the skip-path upsampler uses, plus its three padding
    /// constants.
    ///
    /// Computed rather than loaded because it is a non-persistent buffer — see
    /// ``resamplerFilter``. Evaluated in `Double` and narrowed to fp32 at the end, which
    /// is a last-bit difference on a filter whose output is a *skip* path.
    static func hannResamplerFilter(ratio: Int)
        -> (filter: MLXArray, pad: Int, padLeft: Int, padRight: Int) {
        let rolloff = 0.99
        let lowpassWidth = 6.0
        let width = Int((lowpassWidth / rolloff).rounded(.up))          // 7 for rolloff .99
        let kernelSize = 2 * width * ratio + 1
        var taps = [Float](repeating: 0, count: kernelSize)
        for i in 0 ..< kernelSize {
            let time = (Double(i) / Double(ratio) - Double(width)) * rolloff
            let clamped = min(max(time, -lowpassWidth), lowpassWidth)
            let window = pow(cos(clamped * Double.pi / lowpassWidth / 2), 2)
            // The *normalised* sinc, sin(pi x)/(pi x), which is exactly 1 at x = 0 — and
            // the midpoint tap hits x = 0 exactly, so the guard is load bearing rather
            // than defensive.
            let sinc = time == 0 ? 1.0 : sin(Double.pi * time) / (Double.pi * time)
            taps[i] = Float(sinc * window * rolloff / Double(ratio))
        }
        return (MLXArray(taps, [1, 1, kernelSize]),
                pad: width,
                padLeft: 2 * width * ratio,
                padRight: kernelSize - ratio)
    }

    // MARK: - One generator

    /// One generator: mel `[B, C, T, M]` → waveform `[B, L, 2]`, channels-last.
    ///
    /// The mel enters as `(batch, stereo, time, bins)` and flattens to
    /// `(batch, stereo·bins, time)` — **stereo-major**, `s * bins + c`. The other
    /// interleaving is the same 128 channels in a different order and would be silently
    /// accepted by `conv_pre`.
    func generator(_ mel: MLXArray, _ p: String,
                   _ g: GeneratorConfiguration) throws -> MLXArray {
        let batch = mel.shape[0], frames = mel.shape[2]
        // [B, C, T, M] -> [B, T, C, M] -> [B, T, C*M]: channel-last with the stereo index
        // outermost, so the channel axis is `(s c)`.
        var x = mel.transposed(0, 2, 1, 3)
            .reshaped([batch, frames, mel.shape[1] * mel.shape[3]])
        x = try conv(x, p + "conv_pre", padding: (Self.stemKernelSize - 1) / 2)

        for stage in 0 ..< g.upsampleRates.count {
            // This is an AMP model, so there is **no** LeakyReLU before the upsample. The
            // non-AMP variant has one; running it here would insert an activation the
            // trained model has never seen.
            x = try convTransposed(
                x, p + "ups.\(stage)", stride: g.upsampleRates[stage],
                padding: (g.upsampleKernelSizes[stage] - g.upsampleRates[stage]) / 2)

            // All resblocks see the *same* input and their outputs are averaged. This is
            // not a residual chain: chaining them would give the same shape, roughly the
            // same magnitude, and a different network.
            let start = stage * g.resblocksPerStage
            var outputs: [MLXArray] = []
            outputs.reserveCapacity(g.resblocksPerStage)
            for j in 0 ..< g.resblocksPerStage {
                outputs.append(try ampBlock(
                    x, p + "resblocks.\(start + j).",
                    kernel: g.resblockKernelSizes[j],
                    dilations: g.resblockDilationSizes[j]))
            }
            x = MLX.stacked(outputs, axis: 0).mean(axis: 0)
        }

        x = try activation1d(x, p + "act_post.")
        x = try conv(x, p + "conv_post", padding: (Self.stemKernelSize - 1) / 2)
        if g.applyFinalActivation {
            x = g.useTanhAtFinal ? MLX.tanh(x) : MLX.clip(x, min: -1, max: 1)
        }
        return x
    }

    /// One AMP block: three (activation, dilated conv, activation, conv) residual passes,
    /// accumulating into `x`.
    ///
    /// Note which convolution is dilated: `convs1[m]` carries `dilation[m]`, `convs2[m]`
    /// is always dilation 1. Dilating both would keep every shape (the padding is derived
    /// from the dilation) and change the receptive field of the whole model.
    func ampBlock(_ input: MLXArray, _ p: String, kernel: Int,
                  dilations: [Int]) throws -> MLXArray {
        var x = input
        for m in 0 ..< Self.convsPerBlock {
            var xt = try activation1d(x, p + "acts1.\(m).")
            xt = try conv(xt, p + "convs1.\(m)",
                          padding: Self.padding(kernel: kernel, dilation: dilations[m]),
                          dilation: dilations[m])
            xt = try activation1d(xt, p + "acts2.\(m).")
            xt = try conv(xt, p + "convs2.\(m)",
                          padding: Self.padding(kernel: kernel, dilation: 1))
            x = x + xt
        }
        return x
    }

    /// `(k*d - d) / 2`. Length-preserving for odd `k`, which every resblock kernel here
    /// is.
    static func padding(kernel: Int, dilation: Int) -> Int {
        (kernel * dilation - dilation) / 2
    }

    // MARK: - Anti-aliased activation

    /// `Activation1d`: **upsample → activation → downsample**, in that order.
    ///
    /// This is the whole point of BigVGAN v2. The Snake activation is strongly
    /// harmonic-generating, so applying it at the signal's own rate folds everything
    /// above Nyquist back down as aliasing. Running it at 2× with band-limited resampling
    /// on both sides pushes that content where the lowpass can remove it.
    ///
    /// Reordering the three — activation first, or dropping the downsample and keeping
    /// the doubled length — produces a tensor of the right shape in two of the three
    /// cases and audible aliasing in all of them. No shape check sees it.
    func activation1d(_ x: MLXArray, _ p: String) throws -> MLXArray {
        let ratio = Self.activationResampleRatio
        let up = try weight(p + "upsample.filter")
        let down = try weight(p + "downsample.lowpass.filter")
        let alpha = try weight(p + "act.alpha")
        let beta = try weight(p + "act.beta")

        let taps = up.shape[2]
        let pad = taps / ratio - 1
        var y = Self.upsample1d(
            x, filter: up, ratio: ratio, pad: pad,
            padLeft: pad * ratio + (taps - ratio) / 2,
            padRight: pad * ratio + (taps - ratio + 1) / 2)
        y = Self.snakeBeta(y, alpha: alpha, beta: beta)
        return Self.downsample1d(y, filter: down, ratio: ratio)
    }

    /// `SnakeBeta`: `x + (1 / (beta + eps)) · sin²(alpha · x)`.
    ///
    /// **The stored parameters are log-scale**, so both are exponentiated first. The
    /// checkpoint says so on its own: `act_post.act.alpha` is entirely negative (min
    /// −1.289, max −0.127) and 60% of the resblock alphas are, which is impossible for a
    /// parameter that is meant to be a frequency and initialised to ones.
    ///
    /// Reading them as linear is the failure mode this comment exists for: it runs, the
    /// shapes are identical, and every activation in the model becomes a different
    /// function — with a *negative* `1/beta` gain wherever beta is negative, which is
    /// 44% of them.
    ///
    /// `alpha` and `beta` are per-channel `[C]` and broadcast against the last axis here,
    /// which is the `[1, C, 1]` broadcast of an NCL layout. Same product.
    static func snakeBeta(_ x: MLXArray, alpha: MLXArray, beta: MLXArray) -> MLXArray {
        let a = MLX.exp(alpha)
        let b = MLX.exp(beta)
        let s = MLX.sin(x * a)
        return x + (1.0 / (b + Self.snakeEpsilon)) * (s * s)
    }

    /// The anti-aliasing upsampler: replicate-pad, transposed convolution by the shared
    /// filter, scale by the ratio, then trim asymmetrically.
    ///
    /// The two trims are **not** equal when `kernel - ratio` is odd — `padRight` takes the
    /// extra sample — and the padding is `replicate`, not zeros. A zero pad here rings at
    /// both ends of every activation in the model, 108 convolutions deep.
    static func upsample1d(_ x: MLXArray, filter: MLXArray, ratio: Int,
                           pad: Int, padLeft: Int, padRight: Int) -> MLXArray {
        var y = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((pad, pad)), IntOrPair(0)],
                           mode: .edge)
        y = sharedFilterConvTransposed(y, filter: filter, stride: ratio)
        y = y * MLXArray(Float(ratio))
        return y[0..., padLeft ..< (y.shape[1] - padRight), 0...]
    }

    /// The matching downsampler: replicate-pad `(k/2 − 1, k/2)` for an even kernel, then a
    /// strided depthwise lowpass convolution.
    ///
    /// The asymmetry is deliberate: it is what makes the round trip land back on exactly
    /// `L` samples rather than `L + 1`.
    static func downsample1d(_ x: MLXArray, filter: MLXArray, ratio: Int) -> MLXArray {
        let taps = filter.shape[2]
        let even = taps % 2 == 0
        let padLeft = taps / 2 - (even ? 1 : 0)
        let padRight = taps / 2
        let y = MLX.padded(x, widths: [
            IntOrPair(0), IntOrPair((padLeft, padRight)), IntOrPair(0),
        ], mode: .edge)
        return sharedFilterConv(y, filter: filter, stride: ratio)
    }

    // MARK: - Primitives

    private func weight(_ name: String) throws -> MLXArray {
        guard let w = weights[name] else { throw Failure.missing(name) }
        return w
    }

    /// A PyTorch `nn.Conv1d`: weight `[out, in, k]`, optional bias, symmetric padding.
    ///
    /// MLX wants `[out, k, in]` over `NLC`, hence `(0, 2, 1)`. Everything is already fp32
    /// (see the type comment), so unlike ``AudioVAEDecoder/conv(_:_:)`` there is no
    /// accumulate-then-round dance to get right — the accumulator and the storage are the
    /// same type.
    func conv(_ x: MLXArray, _ stem: String, padding: Int, dilation: Int = 1) throws
        -> MLXArray {
        Self.conv1d(x, weight: try weight(stem + ".weight"), bias: weights[stem + ".bias"],
                    padding: padding, dilation: dilation)
    }

    /// The weight-dictionary-free half of ``conv(_:_:padding:dilation:)``, so a test can
    /// pin the permutation against a hand-evaluated example without a model.
    static func conv1d(_ x: MLXArray, weight: MLXArray, bias: MLXArray?, padding: Int,
                       dilation: Int) -> MLXArray {
        var y = MLX.conv1d(x, weight.transposed(0, 2, 1), stride: 1, padding: padding,
                           dilation: dilation)
        if let bias { y = y + bias }
        return y
    }

    /// A PyTorch `nn.ConvTranspose1d`: weight `[in, out, k]` — **in-major**, the opposite
    /// of `nn.Conv1d` — so the permutation to MLX's `[out, k, in]` is `(1, 2, 0)` and not
    /// `(0, 2, 1)`.
    ///
    /// MLX's `convTransposed1d` already matches PyTorch's convention on both the kernel
    /// orientation (it flips internally) and the meaning of `padding` (a crop of the
    /// output, giving `(L−1)·stride − 2·padding + k`). Both claims are asserted
    /// numerically in `AudioVocoderTests` against a hand-evaluated example, because they
    /// are properties of a dependency rather than of this code.
    func convTransposed(_ x: MLXArray, _ stem: String, stride: Int, padding: Int) throws
        -> MLXArray {
        Self.convTransposed1d(x, weight: try weight(stem + ".weight"),
                              bias: weights[stem + ".bias"], stride: stride,
                              padding: padding)
    }

    /// The weight-dictionary-free half of ``convTransposed(_:_:stride:padding:)``.
    static func convTransposed1d(_ x: MLXArray, weight: MLXArray, bias: MLXArray?,
                                 stride: Int, padding: Int) -> MLXArray {
        var y = MLX.convTransposed1d(x, weight.transposed(1, 2, 0), stride: stride,
                                     padding: padding)
        if let bias { y = y + bias }
        return y
    }

    /// Depthwise convolution where **every channel shares one filter**.
    ///
    /// A grouped convolution with one group per channel, each group a `1 → 1` kernel, so
    /// the `[1, 1, k]` buffer broadcasts to MLX's `[C_out, k, C_in/groups]` = `[C, k, 1]`
    /// unchanged. The alternative — folding the channels into the batch — is arithmetically
    /// identical and needs no grouped-convolution support, but costs four full transposes
    /// per activation across roughly 450 activation sites. The grouped form is used
    /// instead, and its arithmetic is pinned by a hand-computed test.
    static func sharedFilterConv(_ x: MLXArray, filter: MLXArray, stride: Int) -> MLXArray {
        let channels = x.shape[2]
        let kernel = MLX.broadcast(filter.transposed(0, 2, 1),
                                   to: [channels, filter.shape[2], 1])
        return MLX.conv1d(x, kernel, stride: stride, groups: channels)
    }

    /// The transposed twin of ``sharedFilterConv(_:filter:stride:)``.
    ///
    /// A `1 → 1` per-group kernel is its own transpose layout — PyTorch's `[C_in,
    /// C_out/groups, k]` and MLX's `[C_out, k, C_in/groups]` are both `[C, k, 1]` here —
    /// so the same broadcast serves both directions. That coincidence is why the
    /// dangerous `ConvTranspose1d` permutation in ``convTransposed(_:_:stride:padding:)``
    /// does not recur in this one.
    static func sharedFilterConvTransposed(_ x: MLXArray, filter: MLXArray, stride: Int)
        -> MLXArray {
        let channels = x.shape[2]
        let kernel = MLX.broadcast(filter.transposed(0, 2, 1),
                                   to: [channels, filter.shape[2], 1])
        return MLX.convTransposed1d(x, kernel, stride: stride, groups: channels)
    }

    /// `[B, L, C]` → `[B·C, L, 1]`, channel-major so the inverse is a plain reshape.
    ///
    /// Only the mel analysis needs this: the STFT runs one shared 514-row basis over each
    /// channel independently, which is a batch dimension and not a group.
    static func channelsToBatch(_ x: MLXArray) -> MLXArray {
        x.transposed(0, 2, 1).reshaped([x.shape[0] * x.shape[2], x.shape[1], 1])
    }
}

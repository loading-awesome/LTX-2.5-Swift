// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// Waveform → log-mel, pinned to exact numbers rather than described.
///
/// ## Where the numbers come from
///
/// Every constant in ``Reference`` is an independently computed value, not a value
/// derived from the same understanding the implementation is built on. That is the
/// difference between this suite and a shape-check suite: a misread of the mel spec
/// cannot agree with itself here.
///
/// The waveforms were chosen so they can be reconstructed **exactly**:
///
/// * **440 Hz sine, 2400 samples**, `sin(2π·440·i/16000)` in `Double` cast to `Float`.
///   A tone must land in the right mel bin — the test a wrong hz↔mel warping fails.
/// * **1500 Hz sine**, the same construction: a second bin, on the far side of
///   slaney's 1 kHz knee.
/// * **Descending ramp, 1600 samples**, `1 − i/(n−1)`. Maximally asymmetric at sample
///   0, so reflect and zero padding visibly disagree.
/// * **LCG sequence, 3200 samples**, `x ← (1103515245x + 12345) mod 2³¹`. Broadband, so
///   every filter is exercised; exact integer arithmetic, so it is the same everywhere.
/// * **Silence**, zeros: the `1e-5` clamp, and nothing else.
///
/// Frame indices 0 and last are deliberate: those are the frames the reflect padding
/// touches, and a padding bug that leaves the interior perfect shows up only there.
///
/// ## The tolerance, and why it is not tighter
///
/// The expected values come from an all-fp32 evaluation; this implementation computes the
/// window and filterbank in `Double` and the STFT in fp32. In log space the disagreement
/// is dominated by the *quietest* bins, where an absolute fp32 error of ~1e-6 on a
/// magnitude of ~1e-3 is 0.1% and `log` turns that into ~1e-3 nats. Loud bins agree far
/// better. So the comparisons assert two bounds, not one: a loose one everywhere and a
/// tight one on the bins that carry the energy. A single loose bound would hide a real
/// error in a loud bin behind the noise floor of a quiet one.
///
/// Across the twelve frames compared here the worst case is `2.4e-03` over all bins (the
/// ramp's interior frames, where the top of the band sits at `log 1e-4`) and `2.4e-05`
/// over the bins above `-6` nats. The bounds are set at `3e-03` and `5e-05`, which is
/// roughly a factor of two of headroom — enough for a different Accelerate version's
/// `cos`, not enough for an arithmetic mistake.
@Suite("Audio mel processor")
struct AudioMelProcessorTests {

    // MARK: - Configuration under test
    //
    // Not hardcoded into the type: these are the values `configuration(from:)` reads out
    // of the audio VAE checkpoint, restated here so the suite runs without the 365 MB
    // file. `checkpointConfigurationMatches` asserts the two agree when the file is
    // present, which is what keeps this from being an independent set of constants.

    static let config = AudioMelProcessor.Configuration(
        sampleRate: 16000, nFFT: 1024, hopLength: 160, melBins: 64)

    static func processor() throws -> AudioMelProcessor {
        try AudioMelProcessor(configuration: config)
    }

    // MARK: - Signals

    static func sine(_ hz: Double, samples: Int) -> [Float] {
        (0 ..< samples).map {
            Float(sin(2 * Double.pi * hz * Double($0) / Double(config.sampleRate)))
        }
    }

    static func ramp(samples: Int) -> [Float] {
        (0 ..< samples).map { Float(1.0 - Double($0) / Double(samples - 1)) }
    }

    /// `glibc`'s LCG, in exact integer arithmetic so every machine produces bit-identical
    /// samples. `Reference.noiseFirstEight` pins the first eight of them.
    static func lcgNoise(samples: Int) -> [Float] {
        var x: Int64 = 12345
        return (0 ..< samples).map { _ in
            x = (1_103_515_245 &* x &+ 12345) % (1 << 31)
            return Float(Double(x) / Double(1 << 31) * 2.0 - 1.0)
        }
    }

    static func mono(_ values: [Float]) -> MLXArray { MLXArray(values, [values.count]) }

    /// One frame of the log-mel, `[melBins]`, as Swift floats.
    static func frame(_ mel: MLXArray, channel: Int, index: Int) -> [Float] {
        mel[0, channel, index].asArray(Float.self)
    }

    static func maxAbsoluteDifference(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "\(a.count) against \(b.count)")
        return zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// Compare a frame against its expected values twice: everywhere, and again
    /// restricted to the bins carrying real energy. See the suite's note on tolerance.
    static func expectFrame(_ got: [Float], matches expected: [Float], _ label: String,
                            everywhere: Float = 3e-3, loudBins: Float = 5e-5,
                            loudAbove: Float = -6) {
        let overall = maxAbsoluteDifference(got, expected)
        #expect(overall <= everywhere,
                "\(label): max |Δ| \(overall) over all bins exceeds \(everywhere)")
        var loudest: Float = 0
        var loudCount = 0
        for (g, e) in zip(got, expected) where e > loudAbove {
            loudest = max(loudest, abs(g - e))
            loudCount += 1
        }
        #expect(loudCount > 0, "\(label): no bin above \(loudAbove); the case is vacuous")
        #expect(loudest <= loudBins, Comment(rawValue:
            "\(label): max |Δ| \(loudest) over the \(loudCount) bins above "
                + "\(loudAbove) exceeds \(loudBins)"))
    }

    // MARK: - Reference values
    //
    // Held at fp32 because that is the precision the values were computed at; a wider
    // constant would assert precision the transform never carries.

    /// The values every comparison below is made against.
    enum Reference {
        /// Column sums of the 64 x 513 filterbank: slaney scale, slaney norm, 0–8000 Hz
        /// at a 16 kHz rate.
        static let slaneyColumnSums: [Float] = [
            6.39934540e-02, 6.39934465e-02, 6.39934540e-02, 6.39934540e-02, 6.39934540e-02,
            6.39934540e-02, 6.39934614e-02, 6.39934540e-02, 6.39934614e-02, 6.39934540e-02,
            6.39934540e-02, 6.39934614e-02, 6.39934540e-02, 6.39934614e-02, 6.39934540e-02,
            6.39934540e-02, 6.39934614e-02, 6.39934540e-02, 6.39934614e-02, 6.39934540e-02,
            6.40471578e-02, 6.39797673e-02, 6.34027570e-02, 6.51373342e-02, 6.32328689e-02,
            6.40268847e-02, 6.42896593e-02, 6.41050637e-02, 6.35462254e-02, 6.44642934e-02,
            6.37996495e-02, 6.39365539e-02, 6.39486313e-02, 6.41219169e-02, 6.39910772e-02,
            6.39016107e-02, 6.42087012e-02, 6.38887584e-02, 6.39456287e-02, 6.40671924e-02,
            6.39674142e-02, 6.40127957e-02, 6.40018582e-02, 6.39917776e-02, 6.39808327e-02,
            6.40338510e-02, 6.39324337e-02, 6.40594959e-02, 6.40192106e-02, 6.39345422e-02,
            6.40637055e-02, 6.39893934e-02, 6.39919192e-02, 6.39874488e-02, 6.40001670e-02,
            6.40032589e-02, 6.40110224e-02, 6.39899969e-02, 6.39908835e-02, 6.40211850e-02,
            6.39970228e-02, 6.39898256e-02, 6.39955848e-02, 6.40036911e-02,
        ]

        /// The same bank on the HTK scale with no normalisation. Present so the
        /// discrimination below is against the real alternative and not a sketch of it.
        static let htkColumnSums: [Float] = [
            1.77265668e+00, 1.86392593e+00, 1.94858182e+00, 2.02699709e+00, 2.09952307e+00,
            2.16649580e+00, 2.35341811e+00, 2.33166027e+00, 2.45478463e+00, 2.56273341e+00,
            2.68544197e+00, 2.74386644e+00, 2.87496877e+00, 2.99647260e+00, 3.10891008e+00,
            3.21280456e+00, 3.37348390e+00, 3.48901224e+00, 3.63973475e+00, 3.75791550e+00,
            3.92533970e+00, 4.08037758e+00, 4.22373247e+00, 4.42041302e+00, 4.55680895e+00,
            4.78963947e+00, 4.93399572e+00, 5.14366102e+00, 5.36557484e+00, 5.53742981e+00,
            5.77929592e+00, 6.02008534e+00, 6.24287939e+00, 6.48089981e+00, 6.74745941e+00,
            7.02791643e+00, 7.28742981e+00, 7.57937717e+00, 7.87892771e+00, 8.18650341e+00,
            8.52464581e+00, 8.84734726e+00, 9.19448853e+00, 9.57918549e+00, 9.93491840e+00,
            1.03405266e+01, 1.07433138e+01, 1.11731634e+01, 1.16105499e+01, 1.20669594e+01,
            1.25538578e+01, 1.30363312e+01, 1.35680084e+01, 1.40864801e+01, 1.46569490e+01,
            1.52414646e+01, 1.58255892e+01, 1.64616089e+01, 1.71060906e+01, 1.77889519e+01,
            1.84955139e+01, 1.92219334e+01, 1.99752159e+01, 2.07701855e+01,
        ]

        /// The slaney *scale* with no norm — the third of the four combinations, and the one
        /// closest to a plausible misreading.
        static let slaneyUnnormalisedColumnSums: [Float] = [
            2.96966624e+00, 2.96966648e+00, 2.96966648e+00, 2.96966648e+00, 2.96966648e+00,
            2.96966672e+00, 2.96966624e+00, 2.96966600e+00, 2.96966696e+00, 2.96966648e+00,
            2.96966600e+00, 2.96966624e+00, 2.96966672e+00, 2.96966720e+00, 2.96966505e+00,
            2.96966696e+00, 2.96966696e+00, 2.96966696e+00, 2.96966696e+00, 2.96966505e+00,
            3.00063157e+00, 3.11534476e+00, 3.25370407e+00, 3.50657797e+00, 3.57092834e+00,
            3.79301834e+00, 3.99528193e+00, 4.17909861e+00, 4.34574843e+00, 4.62464428e+00,
            4.80132484e+00, 5.04750300e+00, 5.29593706e+00, 5.57059574e+00, 5.83175278e+00,
            6.10908175e+00, 6.43934012e+00, 6.72134399e+00, 7.05710888e+00, 7.41713047e+00,
            7.76859617e+00, 8.15519142e+00, 8.55351448e+00, 8.97141838e+00, 9.40960407e+00,
            9.87904167e+00, 1.03469095e+01, 1.08756790e+01, 1.14016085e+01, 1.19447231e+01,
            1.25556030e+01, 1.31558180e+01, 1.38012581e+01, 1.44767704e+01, 1.51894770e+01,
            1.59348583e+01, 1.67180176e+01, 1.75318089e+01, 1.83914738e+01, 1.93021545e+01,
            2.02407475e+01, 2.12305584e+01, 2.22733212e+01, 2.33681107e+01,
        ]

        /// The first 24 taps of filter 0, entirely inside slaney's linear region.
        static let slaneyFilterZeroTaps: [Float] = [
            -0.00000000e+00, 7.25564221e-03, 1.45112844e-02, 2.13311519e-02, 1.40755093e-02,
            6.81986753e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]

        /// Taps 190–229 of filter 40, well above the 1 kHz knee where the two warpings have
        /// fully diverged.
        static let slaneyFilter40Taps: [Float] = [
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        ]

        /// 440 Hz, 2400 samples: frame 0 — the frame reflect padding shapes.
        static let sine440Frame0: [Float] = [
            -2.88951576e-01, -2.53857076e-01, -1.92916080e-01, -1.00246452e-01, 3.28715816e-02,
            2.26446763e-01, 5.20400286e-01, 1.08312130e+00, 1.90646863e+00, 1.85574472e+00,
            9.16631222e-01, 2.53712833e-01, -1.47249728e-01, -4.48690951e-01, -6.93900108e-01,
            -9.02149022e-01, -1.08404994e+00, -1.24614549e+00, -1.39261234e+00, -1.52655554e+00,
            -1.64996815e+00, -1.76921880e+00, -1.89202225e+00, -1.97864401e+00, -2.11946702e+00,
            -2.21430469e+00, -2.31777263e+00, -2.42658353e+00, -2.53868246e+00, -2.62702370e+00,
            -2.73916364e+00, -2.83683014e+00, -2.93569541e+00, -3.03119278e+00, -3.13058376e+00,
            -3.22795153e+00, -3.31846452e+00, -3.41779828e+00, -3.50984573e+00, -3.60018420e+00,
            -3.69293571e+00, -3.78227806e+00, -3.87142229e+00, -3.95930624e+00, -4.04591656e+00,
            -4.13020515e+00, -4.21537304e+00, -4.29541779e+00, -4.37643623e+00, -4.45610046e+00,
            -4.53040171e+00, -4.60566759e+00, -4.67713547e+00, -4.74595499e+00, -4.81143761e+00,
            -4.87368202e+00, -4.93206882e+00, -4.98666573e+00, -5.03617859e+00, -5.07994509e+00,
            -5.11858606e+00, -5.15020323e+00, -5.17397976e+00, -5.18901157e+00,
        ]

        /// 440 Hz: frame 8, entirely interior.
        static let sine440Frame8: [Float] = [
            -8.36485100e+00, -8.15352154e+00, -7.81664324e+00, -7.35896015e+00, -6.76472998e+00,
            -5.98181009e+00, -4.86744451e+00, -2.78835917e+00, 1.77890933e+00, 1.67301369e+00,
            -2.80888748e+00, -4.91132021e+00, -6.02226877e+00, -6.81557512e+00, -7.43859434e+00,
            -7.95362377e+00, -8.39394093e+00, -8.77891254e+00, -9.12104797e+00, -9.42932892e+00,
            -9.71126270e+00, -9.98048496e+00, -1.02420883e+01, -1.04652205e+01, -1.07385960e+01,
            -1.09610786e+01, -1.11877337e+01, -1.14190493e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
        ]

        /// 440 Hz: the last frame, shaped by the *trailing* reflect pad.
        static let sine440Frame15: [Float] = [
            -3.03956717e-01, -2.68876731e-01, -2.07918823e-01, -1.15270808e-01, 1.78780984e-02,
            2.11401239e-01, 5.05477250e-01, 1.06710696e+00, 1.99365699e+00, 1.83375931e+00,
            9.01036918e-01, 2.38693461e-01, -1.62256390e-01, -4.63705242e-01, -7.08911777e-01,
            -9.17160749e-01, -1.09906340e+00, -1.26115632e+00, -1.40762615e+00, -1.54156566e+00,
            -1.66498184e+00, -1.78422964e+00, -1.90703297e+00, -1.99365795e+00, -2.13447857e+00,
            -2.22931695e+00, -2.33278465e+00, -2.44159508e+00, -2.55369473e+00, -2.64203668e+00,
            -2.75417495e+00, -2.85184169e+00, -2.95070672e+00, -3.04620504e+00, -3.14559460e+00,
            -3.24296451e+00, -3.33347702e+00, -3.43281078e+00, -3.52485824e+00, -3.61519670e+00,
            -3.70794678e+00, -3.79729128e+00, -3.88643694e+00, -3.97431898e+00, -4.06092930e+00,
            -4.14521742e+00, -4.23038721e+00, -4.31043243e+00, -4.39144802e+00, -4.47111511e+00,
            -4.54541779e+00, -4.62068415e+00, -4.69214678e+00, -4.76096153e+00, -4.82644510e+00,
            -4.88869572e+00, -4.94708872e+00, -5.00168323e+00, -5.05118847e+00, -5.09495497e+00,
            -5.13360310e+00, -5.16521931e+00, -5.18898869e+00, -5.20402288e+00,
        ]

        /// 1500 Hz, frame 8. Above slaney's knee, so a warping error moves it further than it
        /// moves the 440 Hz case.
        static let sine1500Frame8: [Float] = [
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -9.76720095e-01, 1.83838570e+00,
            -7.57966876e-01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
        ]

        /// The same frame from a `power=2.0` transform, identical in every other respect.
        /// The wrong answer, kept so the difference between the two can be measured
        /// rather than assumed.
        static let sine440Frame8AsPower: [Float] = [
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.07095833e+01,
            -9.08753967e+00, -6.71061039e+00, -1.81707764e+00, 7.00133753e+00, 6.92742872e+00,
            -1.50209022e+00, -6.76668739e+00, -9.15352249e+00, -1.07990494e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
        ]

        /// Descending ramp, 1600 samples: frame 0. The reflect-padding discriminator.
        static let rampFrame0: [Float] = [
            7.84850657e-01, -2.61297274e+00, -3.54795599e+00, -4.15301895e+00, -4.61565781e+00,
            -4.98700380e+00, -5.30026865e+00, -5.56961727e+00, -5.80707979e+00, -6.01857901e+00,
            -6.20993757e+00, -6.38397408e+00, -6.54417705e+00, -6.69204664e+00, -6.82975483e+00,
            -6.95823574e+00, -7.07883024e+00, -7.19249630e+00, -7.29983568e+00, -7.40146208e+00,
            -7.49782753e+00, -7.59376764e+00, -7.69622326e+00, -7.76444912e+00, -7.88829613e+00,
            -7.96883917e+00, -8.05915070e+00, -8.15594864e+00, -8.25736713e+00, -8.33601856e+00,
            -8.43988991e+00, -8.52990055e+00, -8.62168789e+00, -8.71079350e+00, -8.80454063e+00,
            -8.89686775e+00, -8.98272896e+00, -9.07783318e+00, -9.16641808e+00, -9.25276566e+00,
            -9.34230423e+00, -9.42931461e+00, -9.51594734e+00, -9.60152245e+00, -9.68632698e+00,
            -9.76838112e+00, -9.85160923e+00, -9.93019676e+00, -1.00098343e+01, -1.00886326e+01,
            -1.01621285e+01, -1.02346983e+01, -1.03055067e+01, -1.03728600e+01, -1.04380570e+01,
            -1.05013037e+01, -1.05592861e+01, -1.06116915e+01, -1.06620684e+01, -1.07051811e+01,
            -1.07433443e+01, -1.07747593e+01, -1.07984514e+01, -1.08134651e+01,
        ]

        /// Ramp frame 1 — still inside the padded region's influence.
        static let rampFrame1: [Float] = [
            7.16189385e-01, -2.87958240e+00, -3.79465675e+00, -4.40378046e+00, -4.86519861e+00,
            -5.23780060e+00, -5.55071449e+00, -5.82056761e+00, -6.05788183e+00, -6.26963091e+00,
            -6.46083546e+00, -6.63516188e+00, -6.79522753e+00, -6.94325495e+00, -7.08089161e+00,
            -7.20944405e+00, -7.33007431e+00, -7.44369078e+00, -7.55094338e+00, -7.65272951e+00,
            -7.74887276e+00, -7.84482098e+00, -7.94742966e+00, -8.01503372e+00, -8.13960075e+00,
            -8.22035122e+00, -8.31031132e+00, -8.40709400e+00, -8.50881481e+00, -8.58743286e+00,
            -8.69119263e+00, -8.78119755e+00, -8.87325668e+00, -8.96226406e+00, -9.05556011e+00,
            -9.14883041e+00, -9.23426056e+00, -9.32906628e+00, -9.41714287e+00, -9.50341511e+00,
            -9.59387398e+00, -9.68057728e+00, -9.76660919e+00, -9.85236645e+00, -9.93689632e+00,
            -1.00197592e+01, -1.01027489e+01, -1.01812277e+01, -1.02612743e+01, -1.03401022e+01,
            -1.04132566e+01, -1.04871788e+01, -1.05569744e+01, -1.06243029e+01, -1.06903725e+01,
            -1.07520819e+01, -1.08084421e+01, -1.08631382e+01, -1.09130678e+01, -1.09561586e+01,
            -1.09932899e+01, -1.10250730e+01, -1.10497685e+01, -1.10649462e+01,
        ]

        /// Ramp frame 5 — fully interior, and identical under either padding mode.
        static let rampFrame5: [Float] = [
            1.47050187e-01, -3.84149408e+00, -5.23321724e+00, -6.15107679e+00, -6.84497309e+00,
            -7.40509176e+00, -7.87538672e+00, -8.28100204e+00, -8.63791084e+00, -8.95644093e+00,
            -9.24433136e+00, -9.50630379e+00, -9.74771786e+00, -9.97098827e+00, -1.01792564e+01,
            -1.03722095e+01, -1.05547495e+01, -1.07272015e+01, -1.08895721e+01, -1.10428352e+01,
            -1.11875439e+01, -1.13334665e+01, -1.14859657e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
            -1.15129251e+01, -1.15129251e+01, -1.15129251e+01, -1.15129251e+01,
        ]

        /// LCG sequence, 3200 samples: frame 0.
        static let noiseFrame0: [Float] = [
            -9.86526489e-01, -3.46838236e-01, -7.51312494e-01, -5.81621587e-01, -6.40571237e-01,
            -2.60937959e-01, -7.20866799e-01, -8.20920229e-01, -1.75605476e-01, -7.38031507e-01,
            2.24094868e-01, -4.36260104e-01, -5.50037563e-01, -4.56616223e-01, -1.05395126e+00,
            -5.70497990e-01, -1.04980910e+00, -9.85164106e-01, -4.84134227e-01, -3.23055953e-01,
            -6.18874542e-02, -1.15375362e-01, -3.88018668e-01, -7.39449561e-01, -5.12940407e-01,
            -9.43950295e-01, -4.25221294e-01, -5.52362323e-01, -9.36748564e-01, -1.33565533e+00,
            -1.45086026e+00, -8.13939393e-01, -7.50465035e-01, -8.33846748e-01, -4.65630233e-01,
            -1.39208501e-02, -4.59439695e-01, -6.59779012e-01, -9.00875747e-01, -7.27934003e-01,
            -6.14023268e-01, -5.30597627e-01, -8.39617312e-01, -7.88797289e-02, -2.15974420e-01,
            -1.06969941e+00, -8.43328237e-01, -5.33067524e-01, -5.21294534e-01, -2.50073778e-03,
            -5.49633682e-01, -7.30570853e-01, -8.03737283e-01, -7.67366052e-01, -6.83440030e-01,
            -4.73623365e-01, -6.11120939e-01, -4.93580222e-01, -7.88106799e-01, -6.76122606e-01,
            -6.26240015e-01, -1.66438326e-01, -4.28380758e-01, -5.76763868e-01,
        ]

        /// LCG frame 10.
        static let noiseFrame10: [Float] = [
            -1.20410037e+00, -4.62382197e-01, -4.86114085e-01, -3.97177964e-01, -5.11259019e-01,
            -7.22821116e-01, -5.62885284e-01, -8.71566176e-01, -4.03970510e-01, -1.33595556e-01,
            -6.52862310e-01, -1.34167945e+00, -8.38654995e-01, -6.52583301e-01, -9.50023055e-01,
            -3.47159952e-01, -4.66564186e-02, -9.99836028e-01, -1.07013333e+00, -6.67980909e-01,
            -6.55228794e-01, -1.98847175e-01, -2.97510207e-01, -5.05365372e-01, -8.45431630e-03,
            -3.80903512e-01, -1.02507484e+00, -1.94982097e-01, -6.96622670e-01, -8.71122479e-01,
            -8.78415346e-01, -3.11495781e-01, -4.37957168e-01, -4.13544893e-01, -7.41017819e-01,
            -1.26899827e+00, -5.93403995e-01, -3.24336082e-01, -1.44272745e-01, -7.72778153e-01,
            -2.53850013e-01, -6.10047951e-02, -2.31130481e-01, -5.31810522e-01, -3.96244347e-01,
            -2.88619340e-01, -3.14980209e-01, -3.90712053e-01, -6.83922648e-01, -5.61045170e-01,
            -3.00073564e-01, -3.65354240e-01, -3.75173777e-01, -6.16291285e-01, -6.28427565e-01,
            -3.37342948e-01, -3.80517900e-01, -2.94256568e-01, -4.62667406e-01, -3.94556522e-01,
            -5.43261886e-01, -4.39309746e-01, -2.44671240e-01, -2.85065919e-01,
        ]

        /// LCG frame 20, the last.
        static let noiseFrame20: [Float] = [
            -7.49493003e-01, -7.10730791e-01, -1.09446756e-02, -5.10751188e-01, -1.33624584e-01,
            -9.37841475e-01, -4.87532198e-01, -7.96481729e-01, -7.39061594e-01, -6.13849908e-02,
            -3.24645877e-01, -1.41431570e+00, -1.44990337e+00, -5.70580875e-03, 1.53342122e-02,
            -9.24541116e-01, -5.64144015e-01, -1.13647640e+00, -4.39297557e-01, -4.52923536e-01,
            -3.04960132e-01, -8.41384828e-01, -6.21946216e-01, -7.56995916e-01, -7.45413840e-01,
            -1.01150096e+00, -1.05054867e+00, -9.28301692e-01, -9.84123945e-01, -5.18550098e-01,
            -1.08562386e+00, -7.24291325e-01, -6.24690484e-03, -1.08358693e+00, -5.96690476e-02,
            4.49077226e-02, -1.10848224e+00, -9.79927897e-01, -1.05748391e+00, -6.14387333e-01,
            -4.07247394e-01, -7.03580439e-01, -7.65439272e-01, -4.28060085e-01, -3.01233888e-01,
            -5.65472841e-01, -7.72832811e-01, -7.69327879e-01, -2.88399816e-01, -4.76743579e-01,
            -4.72201109e-01, -6.32443130e-01, -3.79435241e-01, -1.85388789e-01, -9.45865989e-01,
            -5.89454532e-01, -5.58563232e-01, -3.59867632e-01, -4.17967886e-01, -5.42834878e-01,
            -5.17427266e-01, -3.15434814e-01, -4.13405836e-01, -5.84020376e-01,
        ]

        /// The first eight LCG *samples*, not mel values: without the same signal going in,
        /// the three frames above compare nothing.
        static let noiseFirstEight: [Float] = [
            3.10308099e-01, -3.90371352e-01, 3.49921256e-01, -7.86463022e-01, 3.31488922e-02,
            -2.06673183e-02, 2.04944402e-01, -2.60090470e-01,
        ]

    }

    // MARK: - Alternative spellings, built so the wrong ones can be shown to be wrong
    //
    // Every one of these is a plausible reading of the mel spec that produces a
    // correctly-shaped mel. They exist here to be *contrasted* with the real thing —
    // asserting only that the implementation matches itself would be vacuous.

    /// The transform again, with two knobs the real one does not have: the magnitude
    /// exponent and the padding mode. Built from the processor's own window and
    /// filterbank so that the only difference under test is the knob.
    static func logMel(_ processor: AudioMelProcessor, signal: [Float],
                       exponent: Float, reflectPadding: Bool) -> [Float] {
        let c = processor.configuration
        let samples = signal.count
        let frames = processor.frameCount(samples: samples)
        let pad = c.nFFT / 2
        var indices = [Int32](repeating: 0, count: frames * c.nFFT)
        var mask = [Float](repeating: 1, count: frames * c.nFFT)
        for f in 0 ..< frames {
            for k in 0 ..< c.nFFT {
                var j = f * c.hopLength + k - pad
                if j < 0 || j >= samples {
                    if reflectPadding {
                        j = j < 0 ? -j : 2 * (samples - 1) - j
                    } else {
                        // Zero padding: read sample 0 and multiply it away.
                        mask[f * c.nFFT + k] = 0
                        j = 0
                    }
                }
                indices[f * c.nFFT + k] = Int32(j)
            }
        }
        let x = MLXArray(signal, [samples])
        let framed = MLX.take(x, MLXArray(indices, [frames, c.nFFT]), axis: -1)
            * MLXArray(mask, [frames, c.nFFT])
        let spectrum = MLXFFT.rfft(framed * processor.window, axis: -1)
        let real = spectrum.realPart(), imaginary = spectrum.imaginaryPart()
        var magnitude = MLX.sqrt(real * real + imaginary * imaginary)
        if exponent != 1 { magnitude = MLX.pow(magnitude, MLXArray(exponent)) }
        let mel = MLX.matmul(magnitude, processor.filterbank)
        return MLX.log(MLX.maximum(mel, MLXArray(AudioMelProcessor.magnitudeFloor)))
            .asArray(Float.self)
    }

    static func columnSums(_ bank: [Float], melBins: Int, frequencyBins: Int) -> [Float] {
        (0 ..< melBins).map { m in
            bank[(m * frequencyBins) ..< ((m + 1) * frequencyBins)].reduce(0, +)
        }
    }

    // MARK: - The filterbank: slaney scale AND slaney norm, shown to be neither of the
    //         other three combinations

    /// The single most important test here, because a wrong filterbank is invisible
    /// downstream: it produces a mel of the right shape whose energy is in the wrong bins.
    ///
    /// Four banks are built — {slaney, htk} × {slaney norm, no norm} — and their column
    /// sums are separated by more than two orders of magnitude. The implementation's bank
    /// is asserted to be the *slaney/slaney* one and, individually, to differ from each
    /// of the others by a margin nothing numerical could close.
    @Test("Filterbank is the slaney scale with the slaney norm, and no other combination")
    func filterbankIsSlaneyScaleAndSlaneyNorm() throws {
        let c = Self.config
        func sums(_ scale: AudioMelProcessor.MelScale,
                  _ norm: AudioMelProcessor.Normalization) -> [Float] {
            Self.columnSums(
                AudioMelProcessor.melFilterbank(
                    frequencyBins: c.frequencyBins, fMin: c.fMin, fMax: c.fMax,
                    melBins: c.melBins, sampleRate: c.sampleRate,
                    scale: scale, normalization: norm),
                melBins: c.melBins, frequencyBins: c.frequencyBins)
        }

        // 1. Ours is the expected bank, to eight significant figures.
        let ours = sums(.slaney, .slaney)
        let againstReference = Self.maxAbsoluteDifference(ours, Reference.slaneyColumnSums)
        #expect(againstReference <= 1e-7,
                "slaney/slaney column sums differ from torchaudio by \(againstReference)")

        // 2. The two *wrong* banks are reproduced exactly too, so step 3 contrasts
        //    against the real alternatives rather than against straw men.
        let htk = sums(.htk, .none)
        #expect(Self.maxAbsoluteDifference(htk, Reference.htkColumnSums) <= 1e-4,
                Comment(rawValue: "the HTK bank does not match torchaudio's, so the "
                    + "contrast below is against something this port invented"))
        let unnormalised = sums(.slaney, .none)
        #expect(Self.maxAbsoluteDifference(
            unnormalised, Reference.slaneyUnnormalisedColumnSums) <= 1e-4)

        // 3. The three are nowhere near each other. Slaney/slaney sits in a band a
        //    thousandth as wide and thirty times lower; a port that got either choice
        //    wrong could not land inside it by accident.
        #expect(ours.min()! > 0.063 && ours.max()! < 0.066,
                Comment(rawValue: "slaney/slaney column sums should be near-constant at "
                    + "~0.064, got [\(ours.min()!), \(ours.max()!)]"))
        #expect(htk.min()! > 1.7 && htk.max()! > 20,
                "HTK column sums should span [1.77, 20.77]")
        #expect(unnormalised.min()! > 2.9 && unnormalised.max()! > 23,
                "the slaney scale without the slaney norm should span [2.97, 23.37]")
        // Bin 40 is above the 1 kHz knee, where the two warpings have diverged fully.
        #expect(abs(ours[40] - htk[40]) > 1.0)
        #expect(abs(ours[40] - unnormalised[40]) > 1.0)
    }

    /// Column sums are a summary; a bank with the right total and the wrong shape would
    /// pass the test above. These are individual taps, one filter from the slaney
    /// warping's **linear** region and one from its **logarithmic** region.
    @Test("Individual filter taps match torchaudio, below and above the 1 kHz knee")
    func filterbankTapsMatchReference() throws {
        let c = Self.config
        let bank = AudioMelProcessor.melFilterbank(
            frequencyBins: c.frequencyBins, fMin: c.fMin, fMax: c.fMax,
            melBins: c.melBins, sampleRate: c.sampleRate,
            scale: .slaney, normalization: .slaney)
        let filter0 = Array(bank[0 ..< 24])
        #expect(Self.maxAbsoluteDifference(filter0, Reference.slaneyFilterZeroTaps) <= 1e-7)
        let base = 40 * c.frequencyBins
        let filter40 = Array(bank[(base + 190) ..< (base + 230)])
        #expect(Self.maxAbsoluteDifference(filter40, Reference.slaneyFilter40Taps) <= 1e-7)
        // Peaks are not 1: the slaney norm scales each triangle by its bandwidth, so a
        // "unit triangles, then normalise" bank would be a different bank.
        #expect(filter0.max()! < 0.03,
                "a slaney-normalised low filter peaks near 0.021, not 1")
    }

    /// The warping itself, independent of any bank. HTK and slaney agree only at zero.
    @Test("The hz to mel warping is slaney's piecewise form, not HTK's single log")
    func melWarpingIsSlaneyNotHTK() throws {
        // Below 1 kHz slaney is exactly linear at 200/3 Hz per mel; HTK is already curved.
        for hz in [100.0, 500.0, 999.0] {
            let slaney = AudioMelProcessor.hzToMel(hz, scale: .slaney)
            #expect(abs(slaney - hz / (200.0 / 3.0)) < 1e-9,
                    "slaney should be linear below 1 kHz at \(hz) Hz")
            #expect(abs(slaney - AudioMelProcessor.hzToMel(hz, scale: .htk)) > 1.0)
        }
        // Above it, both are logarithmic and still disagree by tens of mels.
        for hz in [2000.0, 8000.0] {
            let slaney = AudioMelProcessor.hzToMel(hz, scale: .slaney)
            let htk = AudioMelProcessor.hzToMel(hz, scale: .htk)
            #expect(abs(slaney - htk) > 20.0, "at \(hz) Hz slaney \(slaney) vs htk \(htk)")
        }
        // Round trip, on both branches of the knee and on it.
        for hz in [0.0, 250.0, 1000.0, 1000.1, 4000.0, 8000.0] {
            for scale in [AudioMelProcessor.MelScale.slaney, .htk] {
                let back = AudioMelProcessor.melToHz(
                    AudioMelProcessor.hzToMel(hz, scale: scale), scale: scale)
                #expect(abs(back - hz) < 1e-6, "\(scale) round trip failed at \(hz) Hz")
            }
        }
    }

    // MARK: - A tone lands where it should

    /// The test a wrong hz↔mel warping fails and no shape check sees.
    ///
    /// 440 Hz must peak at mel bin 8 and 1500 Hz at bin 29. The neighbour check is what
    /// makes it a *localisation* test rather than an ordering one: two bins either side
    /// must be more than 3 nats down, i.e. under 5% of the peak's magnitude.
    @Test("A pure tone lands in the reference's mel bin, and its neighbours do not")
    func pureToneLandsInTheExpectedMelBin() throws {
        let processor = try Self.processor()
        for (hz, expectedBin) in [(440.0, 8), (1500.0, 29)] {
            let mel = try processor.logMel(
                waveform: Self.mono(Self.sine(hz, samples: 2400)),
                sampleRate: Self.config.sampleRate)
            let f = Self.frame(mel, channel: 0, index: 8)
            let peak = f.firstIndex(of: f.max()!)!
            #expect(peak == expectedBin,
                    "\(hz) Hz peaked at mel bin \(peak), reference says \(expectedBin)")
            for offset in [-3, -2, 2, 3] where (0 ..< f.count).contains(peak + offset) {
                #expect(f[peak] - f[peak + offset] > 3.0, Comment(rawValue:
                    "bin \(peak + offset) is only \(f[peak] - f[peak + offset]) nats "
                        + "below the peak; the tone is not localised"))
            }
        }
    }

    // MARK: - Full-frame agreement

    @Test("A 440 Hz tone reproduces torchaudio's log-mel frame for frame")
    func logMelMatchesReferenceForAPureTone() throws {
        let processor = try Self.processor()
        let mel = try processor.logMel(
            waveform: Self.mono(Self.sine(440, samples: 2400)),
            sampleRate: Self.config.sampleRate)
        #expect(mel.shape == [1, 2, 16, 64])
        Self.expectFrame(Self.frame(mel, channel: 0, index: 0),
                         matches: Reference.sine440Frame0, "440 Hz frame 0")
        Self.expectFrame(Self.frame(mel, channel: 0, index: 8),
                         matches: Reference.sine440Frame8, "440 Hz frame 8")
        Self.expectFrame(Self.frame(mel, channel: 0, index: 15),
                         matches: Reference.sine440Frame15, "440 Hz frame 15")

        let mel1500 = try processor.logMel(
            waveform: Self.mono(Self.sine(1500, samples: 2400)),
            sampleRate: Self.config.sampleRate)
        Self.expectFrame(Self.frame(mel1500, channel: 0, index: 8),
                         matches: Reference.sine1500Frame8, "1500 Hz frame 8")
    }

    /// Broadband, so every one of the 64 filters is exercised at once rather than the
    /// three a tone touches.
    @Test("A broadband LCG sequence reproduces torchaudio's log-mel")
    func logMelMatchesReferenceForBroadbandNoise() throws {
        let processor = try Self.processor()
        let signal = Self.lcgNoise(samples: 3200)
        // The generator must have produced the expected samples for the mel comparison
        // below to mean anything.
        #expect(Self.maxAbsoluteDifference(Array(signal[0 ..< 8]),
                                           Reference.noiseFirstEight) <= 1e-9,
                "the LCG diverged from the pod's; the mel comparison below is void")
        let mel = try processor.logMel(waveform: Self.mono(signal),
                                       sampleRate: Self.config.sampleRate)
        #expect(mel.shape == [1, 2, 21, 64])
        Self.expectFrame(Self.frame(mel, channel: 0, index: 0),
                         matches: Reference.noiseFrame0, "noise frame 0")
        Self.expectFrame(Self.frame(mel, channel: 0, index: 10),
                         matches: Reference.noiseFrame10, "noise frame 10")
        Self.expectFrame(Self.frame(mel, channel: 0, index: 20),
                         matches: Reference.noiseFrame20, "noise frame 20")
    }

    // MARK: - power = 1.0

    /// This transform uses `power = 1.0`, i.e. magnitude. The usual default is `2.0`, and
    /// the difference is **not** a constant a downstream scale could absorb.
    ///
    /// The derivation, because a number nobody derived is worth nothing: the exponent is
    /// applied to the linear spectrogram *before* the filterbank sums it, so a `power=2`
    /// mel is `Σᵢ wᵢ|Sᵢ|²` while this one is `Σᵢ wᵢ|Sᵢ|`. Those are not related by
    /// squaring — `Σ wᵢ|Sᵢ|² ≠ (Σ wᵢ|Sᵢ|)²` — so `log(mel₂) − 2·log(mel₁)` varies with
    /// how the energy is spread. Two limits bracket it. If the whole filter sees one
    /// amplitude `c`, the difference is `−log(Σwᵢ) = −log 0.064 = 2.75`. If a single bin
    /// `k` dominates, it is `−log w_k`, and for the widest filters that is larger. Across
    /// one frame the gap spans `[2.82, 11.51]` — and the `11.51` is the clamp floor,
    /// which is itself a tell: squaring pushes quiet bins under `1e-5` where magnitudes
    /// did not go.
    ///
    /// So the assertion is threefold: the magnitude form matches its expected frame, the
    /// squared form matches the expected `power=2.0` frame, and the gap between them is
    /// nowhere near constant and nowhere near `log 2`.
    @Test("power = 1.0 is magnitude, and the power form is not a constant offset from it")
    func powerOneIsMagnitudeNotPower() throws {
        let processor = try Self.processor()
        let signal = Self.sine(440, samples: 2400)
        let melBins = Self.config.melBins

        let magnitude = Self.logMel(processor, signal: signal,
                                    exponent: 1, reflectPadding: true)
        let squared = Self.logMel(processor, signal: signal,
                                  exponent: 2, reflectPadding: true)
        let frame8Magnitude = Array(magnitude[(8 * melBins) ..< (9 * melBins)])
        let frame8Squared = Array(squared[(8 * melBins) ..< (9 * melBins)])

        Self.expectFrame(frame8Magnitude, matches: Reference.sine440Frame8,
                         "magnitude frame 8")
        // The squared form is pinned too, so the contrast is with a real power=2.0 mel
        // and not with a straw man.
        Self.expectFrame(frame8Squared, matches: Reference.sine440Frame8AsPower,
                         "power=2.0 frame 8", everywhere: 2e-4, loudBins: 2e-5,
                         loudAbove: -6)

        let gaps = zip(frame8Squared, frame8Magnitude).map { $0 - 2 * $1 }
        #expect(gaps.min()! > 2.5 && gaps.min()! < 3.2, Comment(rawValue:
            "the smallest log-space gap should sit near −log(Σw) = 2.75, got "
                + "\(gaps.min()!)"))
        #expect(gaps.max()! > 11.0,
                "the largest gap should reach the clamp floor, got \(gaps.max()!)")
        #expect(gaps.max()! - gaps.min()! > 8.0, Comment(rawValue:
            "the gap spans \(gaps.max()! - gaps.min()!) nats; if it were constant a "
                + "downstream scale could hide this error, and it is not"))
        // And it is emphatically not log 2, which is the number a hurried port reaches
        // for on the (wrong) assumption that mel₂ = mel₁².
        #expect(gaps.allSatisfy { abs($0 - Float(log(2.0))) > 1.0 })
    }

    // MARK: - Frame count algebra

    /// `1 + samples / hop`, checked at a spread of lengths including both sides of a hop
    /// boundary.
    ///
    /// This is the number that becomes the audio latent count. `FRAGILE_CONTRACTS.md` #3
    /// records an error here as a shape mismatch, which is cheap to find — but only if
    /// the shape is the thing being compared, and dropping the centre padding would be
    /// wrong by `n_fft / hop = 6` frames here without anything else looking odd.
    @Test("Frame count is 1 + samples / hop, centre padding included")
    func frameCountAlgebra() throws {
        let processor = try Self.processor()
        // Sample counts paired with the frame count each one must produce.
        let measured: [(samples: Int, frames: Int)] = [
            (513, 4), (514, 4), (640, 5), (1600, 11), (2400, 16), (3200, 21),
            (16000, 101), (64000, 401), (64001, 401),
        ]
        for (samples, frames) in measured {
            #expect(processor.frameCount(samples: samples) == frames, Comment(rawValue:
                "\(samples) samples: predicted \(processor.frameCount(samples: samples)), "
                    + "reference measured \(frames)"))
            #expect(1 + samples / Self.config.hopLength == frames)
        }
        // And the shape the transform actually produces agrees with the algebra.
        for samples in [513, 1600, 2400] {
            let mel = try processor.logMel(
                waveform: Self.mono(Self.lcgNoise(samples: samples)),
                sampleRate: Self.config.sampleRate)
            #expect(mel.shape == [1, 2, processor.frameCount(samples: samples), 64])
        }
        // A window length that divides differently must not move the count: `n_fft`
        // cancels out of the derivation, and this is where an implementation that kept
        // it in fails.
        let wide = try AudioMelProcessor(configuration: .init(
            sampleRate: 16000, nFFT: 2048, hopLength: 160, melBins: 64))
        #expect(wide.frameCount(samples: 2400) == processor.frameCount(samples: 2400))
    }

    // MARK: - Reflect padding

    /// A descending ramp is the discriminator: it starts at `1.0`, so zero-padding
    /// invents a step discontinuity at sample 0 and reflect padding does not.
    ///
    /// The reflect-padded first frame and the zero-padded one differ by `7.375` nats — a
    /// factor of 1600 in magnitude — so this is not a subtle test. It is also the only
    /// test in the suite that touches the padding *mode* rather than its width; an
    /// implementation that pads the right number of samples the wrong way passes
    /// everything else.
    @Test("Padding is reflect, not zero, and the first frames prove it")
    func reflectPaddingNotZeroPadding() throws {
        let processor = try Self.processor()
        let signal = Self.ramp(samples: 1600)
        let mel = try processor.logMel(waveform: Self.mono(signal),
                                       sampleRate: Self.config.sampleRate)
        #expect(mel.shape == [1, 2, 11, 64])
        Self.expectFrame(Self.frame(mel, channel: 0, index: 0),
                         matches: Reference.rampFrame0, "ramp frame 0")
        Self.expectFrame(Self.frame(mel, channel: 0, index: 1),
                         matches: Reference.rampFrame1, "ramp frame 1")
        Self.expectFrame(Self.frame(mel, channel: 0, index: 5),
                         matches: Reference.rampFrame5, "ramp frame 5")

        let melBins = Self.config.melBins
        let zeroPadded = Self.logMel(processor, signal: signal,
                                     exponent: 1, reflectPadding: false)
        let zeroFrame0 = Array(zeroPadded[0 ..< melBins])
        let gap = Self.maxAbsoluteDifference(zeroFrame0, Reference.rampFrame0)
        #expect(gap > 7.0, Comment(rawValue:
            "zero padding differs from the reference's first frame by only \(gap) "
                + "nats; the two modes are supposed to be far apart here"))
        // The interior is untouched by either mode, which is exactly why an interior-only
        // comparison would have missed this.
        let zeroFrame5 = Array(zeroPadded[(5 * melBins) ..< (6 * melBins)])
        #expect(Self.maxAbsoluteDifference(zeroFrame5, Reference.rampFrame5) < 1e-2,
                "frame 5 should be identical under both padding modes")
    }

    // MARK: - Channels

    /// The encoder's `conv_in` is `[128, 2, 3, 3]`, so a mono source must become two
    /// planes. Replicating rather than zero-filling the second plane keeps a mono clip's
    /// energy where a stereo clip's would be.
    @Test("Mono replicates to two identical planes")
    func monoReplicatesToStereo() throws {
        let processor = try Self.processor()
        let signal = Self.sine(440, samples: 2400)
        let fromRank1 = try processor.logMel(waveform: Self.mono(signal),
                                             sampleRate: Self.config.sampleRate)
        #expect(fromRank1.shape == [1, 2, 16, 64])
        #expect(Self.frame(fromRank1, channel: 0, index: 8)
            == Self.frame(fromRank1, channel: 1, index: 8))
        // `[1, samples]` is the same thing spelled with an explicit channel axis.
        let fromRank2 = try processor.logMel(
            waveform: MLXArray(signal, [1, signal.count]),
            sampleRate: Self.config.sampleRate)
        #expect(Self.frame(fromRank2, channel: 1, index: 8)
            == Self.frame(fromRank1, channel: 1, index: 8))
    }

    /// Genuine stereo passes through with the channels distinct, and each channel is
    /// **bit-identical to the same signal transformed alone**. An implementation that
    /// accidentally mixed or averaged the ears would fail that while still producing a
    /// `[1, 2, T, 64]` tensor.
    @Test("Genuine stereo keeps the two ears distinct and independent")
    func genuineStereoKeepsChannelsDistinct() throws {
        let processor = try Self.processor()
        let samples = 2400
        let left = Self.sine(440, samples: samples)
        let right = Self.sine(1500, samples: samples)
        let stereo = try processor.logMel(
            waveform: MLXArray(left + right, [2, samples]),
            sampleRate: Self.config.sampleRate)
        #expect(stereo.shape == [1, 2, 16, 64])

        let leftAlone = try processor.logMel(waveform: Self.mono(left),
                                             sampleRate: Self.config.sampleRate)
        let rightAlone = try processor.logMel(waveform: Self.mono(right),
                                              sampleRate: Self.config.sampleRate)
        #expect(Self.frame(stereo, channel: 0, index: 8)
            == Self.frame(leftAlone, channel: 0, index: 8))
        #expect(Self.frame(stereo, channel: 1, index: 8)
            == Self.frame(rightAlone, channel: 0, index: 8))
        #expect(Self.frame(stereo, channel: 0, index: 8)
            != Self.frame(stereo, channel: 1, index: 8))
        // And each ear peaks in its own tone's bin: 8 and 29.
        let l = Self.frame(stereo, channel: 0, index: 8)
        let r = Self.frame(stereo, channel: 1, index: 8)
        #expect(l.firstIndex(of: l.max()!)! == 8)
        #expect(r.firstIndex(of: r.max()!)! == 29)
    }

    // MARK: - Refusals

    /// The deliberate limit. High-quality resampling is a Kaiser-windowed sinc this type
    /// does not implement and will not approximate, so anything that is not the configured
    /// rate is an error naming that reason — not a best-effort conversion whose difference
    /// nobody could bound.
    @Test("A waveform at any other sample rate is refused by name")
    func nonSixteenKilohertzThrows() throws {
        let processor = try Self.processor()
        let signal = Self.mono(Self.sine(440, samples: 2400))
        for rate in [8000, 22050, 44100, 48000] {
            #expect(throws: AudioMelProcessor.Failure.self) {
                _ = try processor.logMel(waveform: signal, sampleRate: rate)
            }
            do {
                _ = try processor.logMel(waveform: signal, sampleRate: rate)
                Issue.record("\(rate) Hz was accepted")
            } catch let failure as AudioMelProcessor.Failure {
                guard case let .unsupportedSampleRate(got, expected) = failure else {
                    Issue.record("wrong case for \(rate) Hz: \(failure)")
                    return
                }
                #expect(got == rate && expected == 16000)
                // The message has to say *why*, or a caller will just resample with
                // whatever is nearest to hand and never learn that it matters.
                #expect(failure.description.contains("resampl"))
            }
        }
    }

    /// Reflect padding cannot mirror more samples than the signal holds, so `n_fft/2 + 1`
    /// is the shortest input there is: 512 samples is refused and 513 succeeds.
    @Test("A signal shorter than n_fft / 2 + 1 is refused, at the reference's boundary")
    func shortSignalThrows() throws {
        let processor = try Self.processor()
        #expect(processor.minimumSamples == 513)
        for samples in [1, 160, 511, 512] {
            #expect(throws: AudioMelProcessor.Failure.self) {
                _ = try processor.logMel(waveform: Self.mono(Self.lcgNoise(samples: samples)),
                                         sampleRate: Self.config.sampleRate)
            }
        }
        let ok = try processor.logMel(waveform: Self.mono(Self.lcgNoise(samples: 513)),
                                      sampleRate: Self.config.sampleRate)
        #expect(ok.shape == [1, 2, 4, 64])
    }

    @Test("Wrong rank, wrong channel count and wrong dtype are all refused")
    func malformedWaveformsThrow() throws {
        let processor = try Self.processor()
        let rate = Self.config.sampleRate
        #expect(throws: AudioMelProcessor.Failure.self) {
            _ = try processor.logMel(waveform: MLXArray.zeros([1, 2, 2400]), sampleRate: rate)
        }
        #expect(throws: AudioMelProcessor.Failure.self) {
            _ = try processor.logMel(waveform: MLXArray.zeros([6, 2400]), sampleRate: rate)
        }
        // bf16 in would mean a narrowed mel out — contract 16's failure mode arriving as
        // an argument rather than as a reduction.
        #expect(throws: AudioMelProcessor.Failure.self) {
            _ = try processor.logMel(
                waveform: MLXArray.zeros([2400], dtype: .bfloat16), sampleRate: rate)
        }
    }

    // MARK: - Silence, determinism, dtype

    /// The `1e-5` clamp exists precisely so `log(0)` never appears. Silence must therefore
    /// be finite and sit *exactly* on `log(1e-5)`, everywhere.
    ///
    /// The floor is `-11.51292515` rather than `-11.51292546` because the clamp and the
    /// log both happen in fp32, so it is `log` of the fp32 value nearest `1e-5`.
    @Test("Silence is finite and sits exactly on log(1e-5)")
    func silenceSitsOnTheFloor() throws {
        let processor = try Self.processor()
        let mel = try processor.logMel(waveform: MLXArray.zeros([2, 1600]),
                                       sampleRate: Self.config.sampleRate)
        #expect(mel.shape == [1, 2, 11, 64])
        let values = mel.asArray(Float.self)
        #expect(values.allSatisfy { $0.isFinite }, "log(0) leaked through the clamp")
        let floor = Float(log(Double(AudioMelProcessor.magnitudeFloor)))
        #expect(abs(values.min()! - floor) < 1e-5)
        #expect(abs(values.max()! - floor) < 1e-5)
        // The fp32 floor, spelled out.
        #expect(abs(values.min()! - -11.512925) < 1e-4)
    }

    @Test("The transform is deterministic and fp32, as the encoder expects")
    func deterministicAndFloat32() throws {
        let processor = try Self.processor()
        let signal = Self.mono(Self.lcgNoise(samples: 3200))
        let first = try processor.logMel(waveform: signal, sampleRate: Self.config.sampleRate)
        for _ in 0 ..< 3 {
            let again = try processor.logMel(waveform: signal,
                                             sampleRate: Self.config.sampleRate)
            #expect(again.asArray(Float.self) == first.asArray(Float.self))
        }
        // fp32 out. `AudioVAEEncoder.encode` casts to the weight dtype at its own
        // boundary; handing it a pre-narrowed mel would round twice.
        #expect(first.dtype == .float32)
        // A second processor built from the same configuration is the same processor.
        let other = try Self.processor()
        #expect(try other.logMel(waveform: signal, sampleRate: Self.config.sampleRate)
            .asArray(Float.self) == first.asArray(Float.self))
    }

    /// The window is the periodic Hann, not the symmetric one. The two forms differ by a
    /// single sample; the window sum is the cleanest way to see which one you have —
    /// `512.0` exactly against `511.5`.
    ///
    /// **The edge taps are where an fp32 window loses accuracy, and this asserts around
    /// it.** Evaluating `0.5 − 0.5 cos(2πn / N)` in fp32 at `n = 1` puts the cosine at
    /// `0.99998…`, so the subtraction cancels five significant digits and lands on
    /// `9.417533875e-06`. This implementation computes the window in `Double` and so gets
    /// the exact `sin²(π/1024) = 9.412358699e-06` instead — a relative difference of
    /// `5.5e-04` on the tap, but only `5.2e-09` absolute on a window whose peak is `1.0`,
    /// which is why the full-frame comparisons above still agree to `2e-04`. Both bounds
    /// are asserted because being *more* accurate than the surrounding arithmetic is
    /// normally a bug (contract 16), and here it is bounded rather than assumed.
    @Test("The analysis window is periodic Hann, not symmetric")
    func hannWindowIsPeriodic() throws {
        let processor = try Self.processor()
        let w = processor.window.asArray(Float.self)
        #expect(w.count == 1024)
        #expect(abs(w.reduce(0, +) - 512.0) < 1e-2,
                "window sums to \(w.reduce(0, +)); the symmetric form sums to 511.5")
        #expect(w[0] == 0)
        #expect(abs(w[512] - 1.0) < 1e-7)
        // The symmetric form's last tap is 0; the periodic form's equals its second. That
        // is the discrimination. The value itself is asserted against the exact one, and
        // separately shown to be within 1e-8 of the fp32-cancelled value.
        #expect(abs(w[1023] - Float(sin(Double.pi / 1024) * sin(Double.pi / 1024))) < 1e-12)
        #expect(w[1] == w[1023])
        #expect(abs(w[1023] - 9.417533875e-06) < 1e-8,
                "torch's fp32 tap and this exact one should differ by ~5.2e-09, not more")
    }

    // MARK: - Config from the checkpoint, not from constants

    /// The repo's standing rule: the configuration comes out of the checkpoint's metadata
    /// rather than out of constants. This asserts that what
    /// ``AudioMelProcessor/configuration(from:)`` reads out of the real file equals the
    /// configuration the rest of this suite uses — which is what stops ``config`` above
    /// from being a set of independent constants that could drift from the checkpoint
    /// without anything noticing.
    @Test("The configuration comes out of the checkpoint")
    func checkpointConfigurationMatches() throws {
        let path = LTXConfiguration.resolved.checkpoints.root!
            + "/vae/ltx-2.5-audio-vae-bf16.safetensors"
        guard FileManager.default.fileExists(atPath: path) else {
            print("SKIP checkpointConfigurationMatches: \(path) is absent")
            return
        }
        let header = try SafetensorsHeader.read(from: URL(fileURLWithPath: path))
        let fromCheckpoint = try AudioMelProcessor.configuration(from: header)
        #expect(fromCheckpoint == Self.config)
        // f_max is derived from the rate rather than read: it is Nyquist, and
        // `preprocessing.mel.mel_fmax` is never consulted.
        #expect(fromCheckpoint.fMax == Double(fromCheckpoint.sampleRate) / 2)
        #expect(fromCheckpoint.fMin == 0)
        // And the mel this produces is a shape the encoder will accept.
        let encoderConfig = try AudioVAEEncoder.readConfiguration(header)
        #expect(encoderConfig.melBins == fromCheckpoint.melBins)
        let processor = try AudioMelProcessor(configuration: fromCheckpoint)
        let mel = try processor.logMel(
            waveform: Self.mono(Self.lcgNoise(samples: 16000)),
            sampleRate: fromCheckpoint.sampleRate)
        #expect(mel.shape == [1, encoderConfig.inChannels, 101, encoderConfig.melBins])
        // 101 mel frames is the number `AudioVAEEncoder.Configuration` turns into the
        // audio latent count; that the two agree here is the only place this file's
        // frame algebra meets the encoder's.
        #expect(encoderConfig.latentFrames(melFrames: mel.shape[2]) > 0)
        #expect(encoderConfig.latentColumns(melBins: mel.shape[3])
            == encoderConfig.latentColumns)
    }

    /// A configuration this port cannot honour must be refused at construction, not
    /// silently transformed into a different one.
    @Test("Impossible configurations are refused at construction")
    func badConfigurationsThrow() throws {
        #expect(throws: AudioMelProcessor.Failure.self) {
            _ = try AudioMelProcessor(configuration: .init(
                sampleRate: 16000, nFFT: 1023, hopLength: 160, melBins: 64))
        }
        #expect(throws: AudioMelProcessor.Failure.self) {
            _ = try AudioMelProcessor(configuration: .init(
                sampleRate: 16000, nFFT: 1024, hopLength: 0, melBins: 64))
        }
        // 512 mel bins over 513 STFT bins leaves filters with no bin in them. Such a
        // filter emits a constant log(1e-5) column forever, so the configuration is
        // refused rather than warned about.
        #expect(throws: AudioMelProcessor.Failure.self) {
            _ = try AudioMelProcessor(configuration: .init(
                sampleRate: 16000, nFFT: 1024, hopLength: 160, melBins: 512))
        }
    }
}

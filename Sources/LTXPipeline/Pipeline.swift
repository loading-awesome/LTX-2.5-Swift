// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX

/// Conditioning, latent layout, sampler, guidance composition, decode, mux.
///
/// The guidance stack is additive and each layer costs a forward pass:
/// base CFG (or dual CFG over the video/audio split), then modality guidance,
/// then STG. A step can be four passes. An observer that records tensors here must be
/// branch-aware or it will attribute one pass's tensor to another.
///
/// `Sampler` owns the arithmetic. This file owns the wiring: what a guidance pass *means*
/// in terms of `DiTForward` + `DiTOutputHead`, where the initial latent comes from, and
/// how the trajectory's token-space output becomes something a VAE will accept.
///
/// ## The four passes
///
/// | pass | what changes | how |
/// |---|---|---|
/// | `.conditional` | nothing | the positive context |
/// | `.unconditional` | the text context | the negative context |
/// | `.modality` | both cross-modal attentions severed | `crossPerturbationMask = 0` |
/// | `.perturbed` | selected self-attentions return their value projection | STG blocks |
///
/// The perturbed path passes `GuidanceParams.stgBlocks` separately for video and audio to
/// `DiTForward`.  The attention implementation retains `to_v`, per-head gating and
/// `to_out`, while bypassing q/k projection, q/k norm, RoPE and SDPA in selected blocks.
/// It never substitutes the conditional pass: that would silently zero the STG term.
public enum Pipeline {

    public enum Failure: Error, CustomStringConvertible {
        case missingTap(String)
        case latentShape(String, got: [Int], expected: String)
        /// Residual cache skips blocks 1–47, so the tap and AdaLN observers would record
        /// a hole where those blocks should be.
        case cacheWithReplayTaps

        public var description: String {
            switch self {
            case let .missingTap(name):
                return "tap \(name) is absent"
            case let .latentShape(name, got, expected):
                return "\(name) is \(got), expected \(expected)"
            case .cacheWithReplayTaps:
                return "residual cache and replay tap observers cannot run together: "
                    + "skipped blocks 1–47 omit taps"
            }
        }
    }

    /// Skipped blocks 1–47 omit every tap those observers would have recorded, leaving a
    /// hole in the record that reads as a missing block rather than a skipped one.
    public static func refuseCacheWithReplayTaps(hasReplayObserver: Bool,
                                                 cacheCount: Int) throws {
        if hasReplayObserver && cacheCount > 0 {
            throw Failure.cacheWithReplayTaps
        }
    }

    // MARK: - Text conditioning

    /// One prompt polarity's text features, per stream: `enc.features.video` and
    /// `enc.features.audio`.
    ///
    /// The connector output, not the Gemma hidden states and not the tokenizer's mask —
    /// see `DiTForward`'s note on why the encoder's attention mask is the wrong tensor to
    /// reach for. Taken as an argument because the encoder is a separate stage; the
    /// sampler's only interest in text is which of two tensors a pass uses.
    public struct Conditioning {
        public var video: MLXArray
        public var audio: MLXArray
        public init(video: MLXArray, audio: MLXArray) {
            self.video = video
            self.audio = audio
        }
    }

    // MARK: - The initial latent

    /// **Contract 1.** The initial latent is a *recorded* tensor. There is no seed here,
    /// and there is deliberately no overload that takes one.
    ///
    /// A CUDA RNG stream is not portable and bf16 rounding of one is not either, so
    /// redrawing from `seed: 0` lands on a different starting point — and since the
    /// trajectory saturates within two steps, a different starting point is a different
    /// render, not a slightly different one. `docs/FRAGILE_CONTRACTS.md` #1 says the seed
    /// is not a contract; this function is where that is enforced by there being no
    /// alternative.
    ///
    /// `noise.video` / `noise.audio` are the draws themselves. The first patchify input is
    /// the fallback when they are absent: on a text-only run it holds the same value,
    /// because the noised state starts from a zero clean latent with an all-ones denoise
    /// mask. Prefer the independent draw when it exists.
    public static func recordedNoise(_ taps: [String: MLXArray]) throws -> Sampler.Streams {
        func read(noise: String, legacy: String) throws -> MLXArray {
            guard let t = taps[noise] ?? taps[legacy] else { throw Failure.missingTap(noise) }
            // The tap is stored fp32 and holds exactly bf16 values; the latent itself is
            // bf16 and every rounding downstream depends on it being so. Casting here is
            // lossless and casting nowhere is not.
            return t.asType(.bfloat16)
        }
        return Sampler.Streams(video: try read(noise: "noise.video",
                                               legacy: "patchify_proj.in.call000"),
                               audio: try read(noise: "noise.audio",
                                               legacy: "audio_patchify_proj.in.call000"))
    }

    /// Exact equality for the recorded draw seam. `-0.0` and `+0.0` deliberately fold:
    /// their bit patterns differ but neither a noiser nor an algebraic lerp has a
    /// semantically meaningful preference between them, and numeric metrics cannot observe
    /// the distinction. Every non-zero bf16 value must have identical bits.
    public static func noiseExactlyMatchesInitialLatent(_ noise: MLXArray,
                                                         _ initialLatent: MLXArray) -> Bool {
        guard noise.shape == initialLatent.shape else { return false }
        // `asType(.uint16)` numerically converts (and would collapse both 1.0 and
        // 1.0078125 to 1). `view` is the MLX bit reinterpretation needed for the
        // actual bfloat16 encoding.
        let lhs = noise.asType(.bfloat16).view(dtype: .uint16).flattened()
            .asArray(UInt16.self)
        let rhs = initialLatent.asType(.bfloat16).view(dtype: .uint16).flattened()
            .asArray(UInt16.self)
        guard lhs.count == rhs.count else { return false }
        for (a, b) in zip(lhs, rhs) {
            if a == b { continue }
            // bfloat16 ±0 differ only by sign bit. Folding only the two zero bit patterns
            // does not accidentally accept subnormals, infinities, or NaN payloads.
            if (a == 0x0000 || a == 0x8000) && (b == 0x0000 || b == 0x8000) { continue }
            return false
        }
        return true
    }

    // MARK: - The denoiser

    /// `DiTForward` + `DiTOutputHead` behind `Sampler.Denoiser`.
    ///
    /// One instance serves every pass of every step; the only per-pass state is which
    /// context goes in and whether the cross-modal attentions are severed.
    public struct DiTDenoiser: Sampler.Denoiser {

        /// Observes the exact tensor handed to an AdaLN module for a live sampler pass.
        /// The optional hook is intentionally at the DiT boundary rather than rebuilding
        /// a timestep from `sigma` in a caller: the latter would exercise a copy of the
        /// convention while this sees the value the module consumes.
        public typealias AdaLNInputObserver = (Sampler.Pass, Int, String, MLXArray) throws -> Void
        /// Pass-aware tap surface.  `tap` deliberately carries no call counter: the
        /// sampler is the sole owner of that counter, so the observer names each tensor
        /// from the actual pass and step rather than keeping a second ordering here.
        public typealias L1TapObserver = (Sampler.Pass, Int, String, MLXArray) throws -> Void

        public let forward: DiTForward
        public let head: DiTOutputHead
        public let geometry: DiTForward.Geometry
        public let positive: Conditioning
        public let negative: Conditioning
        /// Video and audio self-attention are perturbed independently, so the two block
        /// lists are separate. They are required here rather than defaulting to `[]`: a
        /// perturbed sampler pass with an inferred empty selection is a conditional pass
        /// in disguise.
        public let videoSTGBlocks: [Int]
        public let audioSTGBlocks: [Int]
        public let adaLNInputObserver: AdaLNInputObserver?
        public let l1TapObserver: L1TapObserver?
        /// One cache per guidance pass. Empty means the dense loop. Sharing a
        /// single cache across passes would compare a conditional residual
        /// against an unconditional one and reuse across the gap.
        public let stepCaches: [Sampler.Pass: StepCache]
        public let cacheStepCount: Int

        public init(forward: DiTForward, head: DiTOutputHead,
                    geometry: DiTForward.Geometry,
                    positive: Conditioning, negative: Conditioning,
                    videoSTGBlocks: [Int], audioSTGBlocks: [Int],
                    adaLNInputObserver: AdaLNInputObserver? = nil,
                    l1TapObserver: L1TapObserver? = nil,
                    stepCaches: [Sampler.Pass: StepCache] = [:],
                    cacheStepCount: Int = 0) {
            self.forward = forward
            self.head = head
            self.geometry = geometry
            self.positive = positive
            self.negative = negative
            self.videoSTGBlocks = videoSTGBlocks
            self.audioSTGBlocks = audioSTGBlocks
            self.adaLNInputObserver = adaLNInputObserver
            self.l1TapObserver = l1TapObserver
            self.stepCaches = stepCaches
            self.cacheStepCount = cacheStepCount
        }

        public init(weights: [String: MLXArray], topology: TransformerTopology,
                    geometry: DiTForward.Geometry,
                    positive: Conditioning, negative: Conditioning,
                    videoSTGBlocks: [Int], audioSTGBlocks: [Int],
                    adaLNInputObserver: AdaLNInputObserver? = nil,
                    l1TapObserver: L1TapObserver? = nil,
                    stepCaches: [Sampler.Pass: StepCache] = [:],
                    cacheStepCount: Int = 0) {
            self.init(forward: DiTForward(weights: weights, topology: topology),
                      head: DiTOutputHead(weights: weights, topology: topology),
                      geometry: geometry, positive: positive, negative: negative,
                      videoSTGBlocks: videoSTGBlocks, audioSTGBlocks: audioSTGBlocks,
                      adaLNInputObserver: adaLNInputObserver, l1TapObserver: l1TapObserver,
                      stepCaches: stepCaches, cacheStepCount: cacheStepCount)
        }

        /// One pass: patchify, 48 blocks, both output heads, back to latent channels.
        ///
        /// The sigma reaches the transformer as `sigma * timestep_scale_multiplier`.
        /// Scaling it here rather than inside the transformer is a seam choice — it is
        /// the same single fp32 multiply either way, and `adaln_single.in` lands on
        /// 999.999755859375 from either side. What would *not* be the same is folding it
        /// into the x0 conversion, which reads the unscaled `Modality.timesteps` and is
        /// 4.29e+02 away if given the scaled one.
        public func velocity(pass: Sampler.Pass, step: Int, sigma: Float,
                             latents: Sampler.Streams,
                             denoiseMask: Sampler.Streams?) throws -> Sampler.Streams {
            try Pipeline.refuseCacheWithReplayTaps(
                hasReplayObserver: l1TapObserver != nil || adaLNInputObserver != nil,
                cacheCount: stepCaches.count)
            let context = pass == .unconditional ? negative : positive
            // `.masked` and not `.uniform`. With a nil mask the two are identical bit for
            // bit, so an unconditioned render is unaffected; a uniform timestep on a
            // *conditioned* render is what made i2v hold its first frame and generate the
            // rest as if nothing had been conditioned.
            let sampling = DiTForward.Sampling.masked(
                scaledTimestep: FlowSchedule.scaledTimestep(sigma),
                videoMask: denoiseMask?.video, audioMask: denoiseMask?.audio,
                // WithReference, because the per-token timestep has to cover the tokens
                // that actually exist. An IC-LoRA reference is appended to the video
                // stream, and it is precisely those extra tokens that need timestep 0 —
                // sizing this from the generated count alone is what makes them invisible.
                // With no reference the two are equal and this is unchanged.
                videoTokens: geometry.videoTokensWithReference,
                audioTokens: geometry.audioTokens)

            // `DiTForward.Output`'s memberwise initialiser is internal, so the severed
            // path cannot return one. A tuple keeps both branches in the same shape
            // without reaching into another target's access control.
            let body: (video: MLXArray, audio: MLXArray,
                       videoEmbeddedTimestep: MLXArray, audioEmbeddedTimestep: MLXArray)
            switch pass {
            case .conditional, .unconditional, .perturbed, .modality:
                // These are the inputs actually consumed by the patchifiers, recorded at
                // the DiT seam so an observer sees every live guidance call rather than a
                // reconstruction of one.
                try l1TapObserver?(pass, step, "patchify_proj.in", latents.video)
                try l1TapObserver?(pass, step, "audio_patchify_proj.in", latents.audio)
                let mask = pass == .modality ? MLXArray([Float(0)], [1, 1, 1]) : nil
                let out = try forward(
                    videoLatent: latents.video, audioLatent: latents.audio,
                    videoContext: context.video, audioContext: context.audio,
                    sampling: sampling, geometry: geometry,
                    videoSelfAttentionPassthroughBlocks:
                        pass == .perturbed ? Set(videoSTGBlocks) : [],
                    audioSelfAttentionPassthroughBlocks:
                        pass == .perturbed ? Set(audioSTGBlocks) : [],
                    crossPerturbationMask: mask,
                    adaLNInputObserver: { module, input in
                        try adaLNInputObserver?(pass, step, module, input)
                        try l1TapObserver?(pass, step, module + ".in", input)
                    },
                    tapObserver: { tap, tensor in
                        try l1TapObserver?(pass, step, tap, tensor)
                    },
                    blockTapObserver: { index, suffix, tensor in
                        let block = String(format: "block_%02d", index)
                        try l1TapObserver?(pass, step, block + "." + suffix, tensor)
                    },
                    blockOutputObserver: { index, suffix, tensor in
                        let block = String(format: "block_%02d", index)
                        try l1TapObserver?(pass, step, block + "." + suffix, tensor)
                    },
                    observer: { index, video, audio in
                        let block = String(format: "block_%02d", index)
                        try l1TapObserver?(pass, step, block + ".out", video)
                        try l1TapObserver?(pass, step, block + ".out.el1", audio)
                    },
                    stepCache: stepCaches[pass],
                    stepIndex: step,
                    stepCount: cacheStepCount)
                body = (out.video, out.audio,
                        out.videoEmbeddedTimestep, out.audioEmbeddedTimestep)
            }

            let heads = try head(
                video: (x: body.video,
                        embeddedTimestep: body.videoEmbeddedTimestep),
                audio: (x: body.audio,
                        embeddedTimestep: body.audioEmbeddedTimestep))
            try l1TapObserver?(pass, step, "norm_out.in", body.video)
            try l1TapObserver?(pass, step, "norm_out.out", heads.video.normed)
            try l1TapObserver?(pass, step, "proj_out.in", heads.video.modulated)
            try l1TapObserver?(pass, step, "proj_out.out", heads.video.latent)
            try l1TapObserver?(pass, step, "audio_norm_out.in", body.audio)
            try l1TapObserver?(pass, step, "audio_norm_out.out", heads.audio.normed)
            try l1TapObserver?(pass, step, "audio_proj_out.in", heads.audio.modulated)
            try l1TapObserver?(pass, step, "audio_proj_out.out", heads.audio.latent)
            return Sampler.Streams(video: heads.video.latent, audio: heads.audio.latent)
        }
    }

    // MARK: - The render

    /// A finished trajectory, in both layouts.
    public struct Rendered {
        /// Token space, `[B, T, C]` — what the sampler works in and what
        /// `patchify_proj.in` records.
        public var patchified: Sampler.Streams
        /// `[1, C, F, H, W]` and `[1, C, T, F]` — what the VAEs accept.
        public var videoLatent: MLXArray
        public var audioLatent: MLXArray
        public var trajectory: Sampler.Trajectory
    }

    /// The sampling loop, from a recorded initial latent to VAE-shaped latents.
    ///
    /// Decode is deliberately not folded in. `VideoVAEDecoder.decode` then `.rgb`, and
    /// `AudioVAEDecoder.decode`, are separate stages with their own memory profile; a
    /// `render` that also decoded would make the sampler untestable without a second
    /// 1.5 GB checkpoint. The vocoder lives on the live `Renderer` path; contract 11
    /// forbids diffing its waveform, not running it.
    ///
    /// `denoiseMask` and `cleanLatent` are forwarded rather than defaulted away. They are
    /// the identity on a text-to-audio-video render, but `Sampler.postProcess` is the seam
    /// image conditioning enters through, and a `render` that quietly dropped them would
    /// make the sampler's only conditioning hook unreachable from the only entry point
    /// anything calls.
    public static func render(sampler: Sampler, steps: Int, initial: Sampler.Streams,
                              denoiser: Sampler.Denoiser,
                              geometry: DiTForward.Geometry,
                              denoiseMask: Sampler.Streams? = nil,
                              cleanLatent: Sampler.Streams? = nil,
                              observer: ((Int, Sampler.Streams) throws -> Void)? = nil)
        throws -> Rendered {
        let trajectory = try sampler.run(steps: steps, initial: initial,
                                         denoiser: denoiser, denoiseMask: denoiseMask,
                                         cleanLatent: cleanLatent,
                                         retainHistory: false, observer: observer)
        let latents = trajectory.latents
        return Rendered(
            patchified: latents,
            videoLatent: try Sampler.unpatchifyVideo(
                latents.video, latentFrames: geometry.latentFrames,
                height: geometry.height / geometry.latent.spatialScale,
                width: geometry.width / geometry.latent.spatialScale),
            audioLatent: Sampler.unpatchifyAudio(
                latents.audio, channels: geometry.latent.audioChannels),
            trajectory: trajectory)
    }
}

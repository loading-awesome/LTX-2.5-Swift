// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXRecipes
import LTXPipeline
import LTXModules
import MLX

/// One active render, running off the engine actor.
public final class RenderJob: @unchecked Sendable {

    public enum Event: Sendable {
        case started
        case textEncodingStarted
        case textEncodingCompleted(validTokens: [Int])
        case samplingProgress(step: Int, totalSteps: Int)
        case completed(LTX25.RenderResult)
        case failed(Error)
        case cancelled
    }

    private let request: LTX25.Request
    /// The recipe, already reconciled with the request by `RenderEngine` on admission.
    ///
    /// Held rather than re-derived: admission planned memory against *this* shape and step
    /// count, and a job that resolved its own would be free to disagree with the plan that
    /// let it start.
    private let resolved: ResolvedRecipe
    private let checkpoints: Renderer.Checkpoints
    /// Present only on a two-stage render; `RenderEngine` refuses the combination that
    /// would leave this nil while `request.twoStage` is set.
    private let upsampler: URL?
    private let engine: RenderEngine

    private let state = JobState()
    private var task: Task<Void, Never>?

    /// What the waveform went through on its way to a mel, for the sidecar.
    ///
    /// Written by `encodeConditioning` and read by `write`, both on this job's own task, so
    /// the plain `var` is not a race. It is carried rather than recomputed because the facts
    /// worth recording — that a resample happened, and that the clip was cut to fit — are
    /// only knowable at the point the decision was made.
    private var audioFrontEnd: String?

    /// How the two-stage route built its image conditioning, for the sidecar. `nil` on the
    /// single-stage route and on an unconditioned two-stage render.
    ///
    /// The sidecar otherwise records `RenderConditioning.Built.provenance`, which describes
    /// **one** pair — and on this route there are two, at different resolutions, from one
    /// file. A record that named only stage 2's would be true and would hide the whole
    /// per-stage mechanism.
    private var twoStageConditioning: String?

    /// Wall seconds between completed sampler steps, for the unconditional bench sidecar.
    /// Written on this job's own task, same as `audioFrontEnd`.
    private var stepSeconds: [Double] = []
    private var stepMark = Date()
    private var samplingStarted = Date()
    /// Sum of `stepSeconds` — DiT load sits in the first interval. Not decode, not mux.
    private var samplingSeconds = 0.0
    /// `Renderer.render` wall clock minus `samplingSeconds`: VAE + vocoder.
    private var decodeSeconds = 0.0

    private let continuation: AsyncStream<Event>.Continuation
    public let events: AsyncStream<Event>

    init(request: LTX25.Request, resolved: ResolvedRecipe,
         checkpoints: Renderer.Checkpoints,
         upsampler: URL? = nil, engine: RenderEngine) {
        self.request = request
        self.resolved = resolved
        self.checkpoints = checkpoints
        self.upsampler = upsampler
        self.engine = engine

        var cont: AsyncStream<Event>.Continuation!
        self.events = AsyncStream { c in
            cont = c
        }
        self.continuation = cont
    }

    public var isFinished: Bool { state.isFinished }

    public func cancel() {
        state.cancel()
        task?.cancel()
    }

    private func emit(_ event: Event) {
        continuation.yield(event)
    }

    /// That the conditioning files are actually there, before a byte of checkpoint opens.
    ///
    /// Deliberately ONLY this. Every other cheap contradiction — image against video, audio
    /// against a mel, strength out of range, a negative prompt with no guidance to bite on —
    /// belongs to ``LTX25/Request/validate()``, which `RenderEngine` already runs before the
    /// job starts. Restating any of it here would mean two checks with two wordings for one
    /// rule, and the copy that drifts is the one nobody is looking at.
    ///
    /// Existence is the exception because it needs the filesystem rather than the request,
    /// and without it a mistyped path is discovered a minute into a 26 GB Gemma load instead
    /// of in the four `stat`s below.
    private func validateConditioningFiles() throws {
        let imageFiles = request.images.map { ("--image/--keyframe", Optional($0.url)) }
        for (flag, url) in imageFiles + [("--video", request.video),
                            ("--audio-mel", request.audioMel), ("--audio", request.audio)] {
            guard let url else { continue }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else {
                throw LTX25.LTXError.invalidRequest("\(flag) is not a file: \(url.path)")
            }
        }
    }

    /// The output geometry, shared by both routes. Extracted so the single-stage and
    /// two-stage specs cannot resolve a shape differently — the two-stage one *derives* its
    /// stage-1 shape from this, so a divergence here would put the two stages at unrelated
    /// resolutions.
    private func resolveGeometry() throws -> DiTForward.Geometry {
        do {
            let geometry = DiTForward.Geometry(
                frames: resolved.frames,
                height: resolved.output.height,
                width: resolved.output.width,
                frameRate: resolved.frameRate)
            try geometry.validate()
            return geometry
        } catch let error as LTX25.LTXError {
            throw error
        } catch {
            throw LTX25.LTXError.invalidRequest("\(error)")
        }
    }

    /// Every adapter this render applies, resolved to files and checked to be adapters.
    ///
    /// Two sources, stacked in that order: the recipe's own stage adapters, then anything
    /// the request added. Overlays compose by summation and each delta is computed from the
    /// base input, so order does not change the result — it changes only what the provenance
    /// reads like, and recipe-then-request is the order a reader expects.
    ///
    /// Hints resolve against the model tree the DiT sits in, so a recipe stays a description
    /// of a pipeline rather than a set of paths. An absolute hint is verified where it
    /// points instead.
    private func resolveAdapters() throws -> [LoRAOverlay.Request] {
        let requested = (resolved.stages.flatMap { $0.stage.adapters }) + request.adapters
        guard !requested.isEmpty else { return [] }

        let root = AdapterCatalog.modelRoot(containing: checkpoints.dit)
        return try requested.map { adapter in
            let file: CheckpointIdentity.File
            do {
                file = adapter.filenameHint.hasPrefix("/")
                    ? try AdapterCatalog.verify(at: URL(fileURLWithPath: adapter.filenameHint))
                    : try AdapterCatalog.resolve(hint: adapter.filenameHint, in: root)
            } catch {
                throw LTX25.LTXError.invalidRequest("\(error)")
            }
            return LoRAOverlay.Request(
                url: file.url,
                strength: adapter.strength,
                filter: adapter.include.isEmpty ? .all
                                                : LoRAOverlay.ModuleFilter(include: adapter.include))
        }
    }

    /// `DistilledRenderer.Spec` — no steps and no guidance, because the distilled pipeline
    /// has neither. ``LTXRecipes/RecipeRequest/validate()`` has already refused a request
    /// that set any of them.
    private func resolveDistilledSpec() throws -> DistilledRenderer.Spec {
        var spec = DistilledRenderer.Spec(prompt: request.prompt,
                                          geometry: try resolveGeometry(),
                                          seed: request.seed)
        // Was silently dropped: `resolveAdapters` is reachable only from `resolveSpec`, so
        // `--lora` on this route resolved an adapter, validated nothing, and rendered
        // without it. A knob that does nothing is the failure this file's neighbours are
        // written to prevent.
        spec.adapters = try resolveAdapters()
        do {
            try spec.validate()
        } catch {
            throw LTX25.LTXError.invalidRequest("\(error)")
        }
        return spec
    }

    private func distilledCheckpoints() throws -> DistilledRenderer.Checkpoints {
        guard let upsampler else {
            throw LTX25.LTXError.invalidRequest(
                "a two-stage distilled render needs the x2 latent spatial upsampler "
                    + "checkpoint; pass --upsampler")
        }
        return DistilledRenderer.Checkpoints(base: checkpoints, upsampler: upsampler)
    }

    private func resolveSpec() throws -> Renderer.Spec {
        do {
            let geometry = try resolveGeometry()

            // The recipe's own guidance, not a hardcoded production pair. A request field
            // still overrides per knob, which is what keeps `--video-cfg` a sweep tool.
            guard let stage = resolved.stages.first?.stage else {
                throw LTX25.LTXError.invalidRequest(
                    "recipe '\(resolved.recipeID)' declares no stages to sample")
            }
            let videoGuide = stage.videoGuidance
            let audioGuide = stage.audioGuidance
            let videoGuidance = GuidanceParams(
                cfgScale: request.videoCFG ?? videoGuide.cfgScale,
                stgScale: request.stgScale ?? videoGuide.stgScale,
                stgBlocks: request.stgBlocks ?? videoGuide.stgBlocks,
                rescaleScale: request.rescaleScale ?? videoGuide.rescaleScale,
                modalityScale: request.modalityScale ?? videoGuide.modalityScale,
                skipStep: 0,
                stgStartPercent: request.stgStartPercent,
                stgEndPercent: request.stgEndPercent,
                modalityStartPercent: request.modalityStartPercent,
                modalityEndPercent: request.modalityEndPercent)
            let audioGuidance = GuidanceParams(
                cfgScale: request.audioCFG ?? audioGuide.cfgScale,
                stgScale: request.stgScale ?? audioGuide.stgScale,
                stgBlocks: request.stgBlocks ?? audioGuide.stgBlocks,
                rescaleScale: request.rescaleScale ?? audioGuide.rescaleScale,
                modalityScale: request.modalityScale ?? audioGuide.modalityScale,
                skipStep: 0,
                stgStartPercent: request.stgStartPercent,
                stgEndPercent: request.stgEndPercent,
                modalityStartPercent: request.modalityStartPercent,
                modalityEndPercent: request.modalityEndPercent)

            return Renderer.Spec(
                prompt: request.prompt,
                negativePrompt: request.negativePrompt ?? "",
                geometry: geometry,
                steps: request.steps ?? stage.sigmas.steps,
                seed: request.seed,
                videoGuidance: videoGuidance,
                audioGuidance: audioGuidance,
                cacheThreshold: request.cacheThreshold,
                cacheMaxSkips: request.cacheMaxSkips,
                adapters: try resolveAdapters())
        } catch let error as LTX25.LTXError {
            throw error
        } catch {
            throw LTX25.LTXError.invalidRequest("\(error)")
        }
    }

    /// Spawned from the actor; the work itself must not inherit that isolation.
    func start() {
        let engine = self.engine
        task = Task.detached { [self] in
            defer {
                state.finish()
                Task { await engine.jobDidFinish(self) }
            }

            do {
                emit(.started)
                try validateConditioningFiles()
                try checkCancellation()

                let written = request.twoStage
                    ? try runTwoStage()
                    : try await runSingleStage()
                emit(.completed(written))
                continuation.finish()
            } catch is CancellationError {
                emit(.cancelled)
                continuation.finish()
            } catch let error as LTX25.LTXError {
                if case .cancelled = error {
                    emit(.cancelled)
                } else {
                    emit(.failed(error))
                }
                continuation.finish()
            } catch {
                if state.checkCancelled() {
                    emit(.cancelled)
                } else {
                    emit(.failed(LTX25.LTXError.executionFailed(error)))
                }
                continuation.finish()
            }
        }
    }

    // MARK: - The two routes

    /// `Renderer.render` — one sampler run at the output resolution, with guidance.
    private func runSingleStage() async throws -> LTX25.RenderResult {
        let spec = try resolveSpec()
        try checkCancellation()

        emit(.textEncodingStarted)
        let text = try Renderer.conditioning(spec: spec, checkpoints: checkpoints)
        emit(.textEncodingCompleted(validTokens: text.validTokenCounts))
        try checkCancellation()
        // Encoder is out of scope inside `conditioning`; hand its buffers back so the
        // VAE encode and the DiT load do not sit on top of Gemma's residue.
        MLX.Memory.clearCache()

        let conditioning = try await encodeConditioning(spec: spec)
        try checkCancellation()
        MLX.Memory.clearCache()

        stepSeconds = []
        samplingStarted = Date()
        stepMark = samplingStarted
        let rendered = try Renderer.render(
            spec: spec,
            checkpoints: checkpoints,
            text: text,
            conditioning: conditioning,
            observer: { [self] step, total in
                try self.checkCancellation()
                let now = Date()
                self.stepSeconds.append(now.timeIntervalSince(self.stepMark))
                self.stepMark = now
                self.emit(.samplingProgress(step: step, totalSteps: total))
            })
        markRenderFinished()
        return try write(rendered, spec: spec, conditioning: conditioning)
    }

    /// `DistilledRenderer.render` — draft, upsample, refine.
    ///
    /// **`--image` is honoured here and nothing else is.** The distilled route takes still
    /// images as its only conditioning: no video, no audio, and no audio conditioning in
    /// either stage. ``LTXRecipes/RecipeRequest/validate()`` refuses `--video`, `--audio`
    /// and `--audio-mel` by name for that reason rather than accepting and dropping them.
    ///
    /// The image conditioning is built **per stage** — see
    /// ``LTXPipeline/DistilledRenderer/imageConditioning(at:strength:spec:encoder:)``.
    ///
    /// Progress is reported as one 11-step run rather than 8 + 3. The stage boundary is
    /// still visible where it matters — the sidecar names both stages and their sigmas —
    /// and a progress counter that restarted at 1 partway through reads as a crash-and-retry.
    private func runTwoStage() throws -> LTX25.RenderResult {
        let spec = try resolveDistilledSpec()
        let distilled = try distilledCheckpoints()
        try checkCancellation()

        emit(.textEncodingStarted)
        let text = try DistilledRenderer.conditioning(spec: spec, checkpoints: distilled)
        emit(.textEncodingCompleted(validTokens: text.validTokenCounts))
        try checkCancellation()
        MLX.Memory.clearCache()

        // One encoder load for both stages, and both pairs built before the 42 GB DiT is
        // opened: a mistyped strength or an unreadable still should cost a VAE load, not a
        // transformer load.
        var stage1Conditioning: RenderConditioning.Built?
        var stage2Conditioning: RenderConditioning.Built?
        let images = try request.resolvedImages(frames: spec.geometry.frames)
        if !images.isEmpty {
            let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
            let pairs = try DistilledRenderer.imageConditioning(
                images, defaultStrength: request.strength, spec: spec, encoder: encoder)
            stage1Conditioning = pairs?.stage1
            stage2Conditioning = pairs?.stage2
        }
        try checkCancellation()
        MLX.Memory.clearCache()

        let total = spec.stage1Steps + spec.stage2Steps
        stepSeconds = []
        samplingStarted = Date()
        stepMark = samplingStarted
        let rendered = try DistilledRenderer.render(
            spec: spec,
            checkpoints: distilled,
            text: text,
            stage1Conditioning: stage1Conditioning,
            stage2Conditioning: stage2Conditioning,
            observer: { [self] stage, step, _ in
                try self.checkCancellation()
                let now = Date()
                self.stepSeconds.append(now.timeIntervalSince(self.stepMark))
                self.stepMark = now
                let global = stage == 1 ? step : spec.stage1Steps + step
                self.emit(.samplingProgress(step: global, totalSteps: total))
            })
        markRenderFinished()

        // `write` reads only the frame rate off this, for the mux. The guidance fields it
        // carries are the single-stage defaults and are meaningless here, which is why the
        // provenance the sidecar actually records is `DistilledRenderer`'s own — assembled
        // in `Renderer.Result.provenance` before this point, and never rebuilt from a spec.
        // The adapters ARE carried, unlike the guidance fields: `write` records them, and a
        // sidecar reporting "none (base weights)" for an adapted render describes a
        // different model than the one that ran.
        let shim = Renderer.Spec(prompt: spec.prompt, geometry: spec.geometry, seed: spec.seed,
                                 adapters: spec.adapters)
        if stage2Conditioning != nil {
            let g1 = spec.stage1Geometry, g2 = spec.geometry
            twoStageConditioning =
                "built TWICE from one source image, once per stage: "
                + "stage 1 at \(g1.width)x\(g1.height), stage 2 at \(g2.width)x\(g2.height). "
                + "Resize is bilinear align_corners=False, NOT antialiased, aspect-fill then "
                + "centre-crop (resize_and_center_crop). The reference's H.264 crf "
                + "re-compression of the still (ImageConditioner.resolve_crf, 18 from LTX "
                + "2.4) is NOT applied — there is no crf-addressable encoder here, so the "
                + "model is conditioned on an uncompressed frame."
        }
        return try write(rendered, spec: shim, conditioning: stage2Conditioning)
    }

    private func encodeConditioning(spec: Renderer.Spec) async throws -> RenderConditioning.Built? {
        var keyframes: [RenderConditioning.Keyframe] = []
        var videoLatent: MLXArray?
        var audioLatent: MLXArray?

        let images = try request.resolvedImages(frames: spec.geometry.frames)
        if !images.isEmpty {
            let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
            for image in images {
                var rgb = try MediaInput.image(at: image.url)
                // Resize only when it is actually needed. This path used to demand a still
                // already at the render's exact size and threw a shape error otherwise;
                // resizing unconditionally would instead run a bilinear filter over stills
                // that are already correct, quietly changing renders that work today.
                if rgb.dim(1) != spec.geometry.height || rgb.dim(2) != spec.geometry.width {
                    rgb = try MediaInput.resizeAndCenterCrop(rgb,
                                                             height: spec.geometry.height,
                                                             width: spec.geometry.width)
                }
                keyframes.append(RenderConditioning.Keyframe(
                    latent: try RenderConditioning.encodeImage(rgb, encoder: encoder),
                    pixelFrame: image.pixelFrame,
                    strength: image.strength ?? request.strength))
            }
        }
        if let videoURL = request.video {
            let rgb = try await MediaInput.videoFrames(at: videoURL, maxFrames: spec.geometry.frames)
            let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
            videoLatent = try RenderConditioning.encodeVideo(rgb, encoder: encoder)
        }
        if let audioMelURL = request.audioMel {
            let arrays = try MLX.loadArrays(url: audioMelURL)
            guard let mel = arrays["mel"] ?? arrays.values.first else {
                throw LTX25.LTXError.invalidRequest("--audio-mel contains no tensors")
            }
            let encoder = try AudioVAEEncoder(checkpoint: checkpoints.audioVAE)
            audioLatent = try RenderConditioning.encodeAudioMel(mel, encoder: encoder)
        }
        if let audioURL = request.audio {
            let encoder = try AudioVAEEncoder(checkpoint: checkpoints.audioVAE)
            let melConfig = try AudioMelProcessor.configuration(
                from: SafetensorsHeader.read(from: checkpoints.audioVAE))
            let processor = try AudioMelProcessor(configuration: melConfig)
            let melFrames = encoder.config.melFrames(latentFrames: spec.geometry.audioTokens)
            let needed = (melFrames - 1) * melConfig.hopLength

            let (native, nativeRate) = try await MediaInput.audio(at: audioURL)
            var planes = native[0]
            var notes: [String] = []
            if nativeRate != melConfig.sampleRate {
                // `AudioMelProcessor` refuses an off-rate waveform rather than approximate
                // a Kaiser-windowed sinc resampler, and that refusal is correct for a
                // measurement. A render is not a measurement, so the conversion happens
                // here — but it is NAMED here too, because the mel it produces is not the
                // mel an exact resampler would give for the same file, and a silent
                // resample would be indistinguishable from an exact one afterwards.
                planes = try await MediaInput.audio(at: audioURL,
                                                    sampleRate: melConfig.sampleRate,
                                                    channels: planes.dim(0))[0]
                notes.append("resampled \(nativeRate) -> \(melConfig.sampleRate) Hz by "
                    + "AVAudioConverter, not a Kaiser-windowed sinc")
            } else {
                notes.append("already at \(melConfig.sampleRate) Hz, unresampled")
            }
            // The GEOMETRY decides the length, never the file — so a clip is cut or extended
            // to fit, and the sidecar says which. A five-second take silently becoming four
            // is the kind of thing that is obvious while you are doing it and invisible a
            // week later.
            let have = planes.dim(1)
            if have > needed {
                planes = planes[0..., 0 ..< needed]
                notes.append("trimmed \(have) -> \(needed) samples")
            } else if have < needed {
                planes = MLX.concatenated(
                    [planes, MLXArray.zeros([planes.dim(0), needed - have], dtype: .float32)],
                    axis: 1)
                notes.append("zero-padded \(have) -> \(needed) samples")
            } else {
                notes.append("exactly \(needed) samples, unfitted")
            }
            let mel = try processor.logMel(waveform: planes, sampleRate: melConfig.sampleRate)
            audioLatent = try RenderConditioning.encodeAudioMel(mel, encoder: encoder)
            audioFrontEnd = notes.joined(separator: "; ")
                + "; mel \(mel.shape) via AudioMelProcessor"
        }

        return try RenderConditioning.build(
            keyframes: keyframes, videoLatent: videoLatent,
            audioLatent: audioLatent, strength: request.strength, geometry: spec.geometry)
    }

    private func write(_ result: Renderer.Result, spec: Renderer.Spec,
                       conditioning: RenderConditioning.Built?) throws -> LTX25.RenderResult {
        let writeStarted = Date()
        let outURL = request.videoOutput
        let wavURL = request.audioOutput
            ?? outURL.deletingPathExtension().appendingPathExtension("wav")
        let sidecarURL = outURL.deletingPathExtension().appendingPathExtension("provenance.json")
        let waveformURL = outURL.deletingPathExtension()
            .appendingPathExtension("waveform.safetensors")

        try RenderOutput.writeWAV(audio: result.audioWaveform, to: wavURL,
                                  sampleRate: result.audioSampleRate)
        try TensorFile.save(arrays: ["waveform": result.audioWaveform.asType(.float32),
                                     "mel": result.audioMel.asType(.float32)],
                            to: waveformURL)

        var muxFailed: String?
        do {
            try RenderOutput.write(
                rgb: result.videoRGB, audio: result.audioWaveform,
                to: outURL,
                spec: RenderOutput.VideoSpec(fps: spec.geometry.frameRate),
                audioSpec: RenderOutput.AudioSpec(sampleRate: result.audioSampleRate))
        } catch {
            muxFailed = "\(error)"
            let framesURL = outURL.deletingPathExtension()
                .appendingPathExtension("videoRGB.safetensors")
            try TensorFile.save(arrays: ["videoRGB": result.videoRGB.asType(.float32)],
                                to: framesURL)
        }
        let writeSeconds = Date().timeIntervalSince(writeStarted)

        var manifest = result.provenance
        manifest["mp4"] = outURL.lastPathComponent
        manifest["audio_mel_shape"] = "\(result.audioMel.shape)"
        manifest["audio_waveform_shape"] = "\(result.audioWaveform.shape)"
        manifest["audio_sample_rate"] = String(result.audioSampleRate)
        manifest["audio_waveform_file"] = waveformURL.lastPathComponent
        manifest["wav"] = wavURL.lastPathComponent
        // The conditioner's OWN account of what it did, not a label derived from which flag
        // was set. Those are not the same thing and the difference is not cosmetic: a
        // derived label reports the first match, so an image-plus-audio render recorded
        // "image" and dropped the audio from the record entirely. `Built.provenance` also
        // carries the strength, the mask polarity, and the fact that audio is frozen
        // outright and ignores strength.
        manifest["conditioning"] = conditioning?.provenance ?? "none (text to audio-video)"
        manifest["conditioning_mode"] = conditioning?.mode.rawValue ?? "textToAV"
        // Every still and its position. A two-keyframe render recorded as one path would be
        // unreproducible in exactly the way the sidecar exists to prevent.
        if !request.images.isEmpty {
            // Resolved positions, not the request's. `--keyframe end.png@last` stores the
            // placeholder -1, and a sidecar recording "@-1" describes a render nobody can
            // reproduce without also knowing the duration and the rule that turns one into
            // the other. Positions are already validated by admission, so this cannot throw;
            // the fallback exists so a record is written even if that ever stops being true.
            let images = (try? request.resolvedImages(frames: resolved.frames)) ?? request.images
            manifest["conditioning_images"] = images
                .map { "\($0.url.path)@\($0.pixelFrame)"
                    + ($0.strength.map { s in ":\(s)" } ?? "") }
                .joined(separator: ", ")
        }
        // An adapter changes the weights the render ran against, so a record that omits it
        // describes a different model. "none (base weights)" is written explicitly rather
        // than left as an absent key, so an unadapted render is distinguishable from an
        // incomplete record.
        if spec.adapters.isEmpty {
            manifest["adapters"] = "none (base weights)"
        } else {
            manifest["adapters"] = spec.adapters
                .map { "\($0.url.lastPathComponent)@\($0.strength) [\($0.filter)]" }
                .joined(separator: ", ")
            manifest["adapters_note"] = "no full render with an adapter applied has been "
                + "checked end to end"
        }
        if let video = request.video { manifest["conditioning_video"] = video.path }
        if let audioMel = request.audioMel { manifest["conditioning_audio_mel"] = audioMel.path }
        if let audio = request.audio { manifest["conditioning_audio"] = audio.path }
        if let audioFrontEnd { manifest["conditioning_audio_front_end"] = audioFrontEnd }
        if let twoStageConditioning {
            manifest["conditioning_per_stage"] = twoStageConditioning
        }
        if let muxFailed { manifest["mux_failed"] = muxFailed }
        let benchURL = outURL.deletingPathExtension().appendingPathExtension("bench.json")
        try writeBench(to: benchURL, spec: spec, provenance: manifest,
                       writeSeconds: writeSeconds)
        manifest["bench"] = benchURL.lastPathComponent
        let json = try JSONSerialization.data(withJSONObject: manifest,
                                              options: [.prettyPrinted, .sortedKeys])
        try json.write(to: sidecarURL)

        return LTX25.RenderResult(
            video: outURL, audioWAV: wavURL, waveform: waveformURL, sidecar: sidecarURL,
            muxFailed: muxFailed, provenance: manifest)
    }

    /// Unconditional. Timings you have to remember to switch on get recorded for the runs
    /// somebody expected to care about, which is never the run that turns out to matter.
    private func writeBench(to url: URL, spec: Renderer.Spec,
                            provenance: [String: String], writeSeconds: Double) throws {
        let mean = stepSeconds.isEmpty
            ? 0.0
            : stepSeconds.reduce(0, +) / Double(stepSeconds.count)
        let body: [String: Any] = [
            "schema": "ltx-bench/1",
            "seed": NSNumber(value: spec.seed),
            "steps": request.twoStage ? stepSeconds.count : spec.steps,
            "geometry": provenance["geometry"] ?? "",
            "cache": provenance["cache"] ?? "off",
            "cache_threshold": spec.cacheThreshold,
            "cache_max_skips": spec.cacheMaxSkips,
            "two_stage": request.twoStage,
            "step_seconds": stepSeconds,
            "mean_full_step_seconds": mean,
            "sampling_seconds": samplingSeconds,
            "decode_seconds": decodeSeconds,
            "write_seconds": writeSeconds,
            "mlx_peak_bytes": MLX.Memory.peakMemory,
            "mlx_active_bytes": MLX.Memory.activeMemory,
        ]
        let data = try JSONSerialization.data(withJSONObject: body,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    /// Split sampling (sum of per-step observer intervals) from VAE + vocoder.
    /// The first step interval includes the DiT load.
    private func markRenderFinished() {
        let renderWall = Date().timeIntervalSince(samplingStarted)
        samplingSeconds = stepSeconds.reduce(0, +)
        decodeSeconds = max(0, renderWall - samplingSeconds)
    }

    private func checkCancellation() throws {
        if Task.isCancelled || state.checkCancelled() {
            throw LTX25.LTXError.cancelled
        }
    }
}

private final class JobState: @unchecked Sendable {
    private let lock = NSLock()
    private var _isFinished = false
    private var _isCancelled = false

    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isFinished
    }

    func finish() {
        lock.lock()
        defer { lock.unlock() }
        _isFinished = true
    }

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        _isCancelled = true
    }

    func checkCancelled() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isCancelled
    }
}

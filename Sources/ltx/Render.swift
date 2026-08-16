// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTX25
import LTXCatalog

struct Render: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "render",
        abstract: "Render a clip from a prompt to an mp4.")

    @Option(help: "The prompt.")
    var prompt: String

    @Option(help: ArgumentHelp("The negative prompt. Empty means the pipeline's default "
                + "negative, which is 223 valid tokens and not an empty sequence (contract 9)."))
    var negativePrompt: String = ""

    @Option(help: "Where to write the mp4. A .provenance.json is written beside it.")
    var out: String

    @Option(help: ArgumentHelp("Video CFG scale. This is what makes --negative-prompt "
                + "bite on the picture. 1.0 is neutral and disables it. Defaults to 3.0; "
                + "refused with --two-stage, which builds no guider."))
    var videoCfg: Double?

    @Option(help: ArgumentHelp("Audio CFG scale, independent of the video one "
                + "(contract 7). 7.0 is a much higher setting that works well. "
                + "Refused with --two-stage."))
    var audioCfg: Double?

    @Option(help: ArgumentHelp("Spatio-temporal guidance scale. Neutral at 0.0, NOT 1.0 "
                + "— and note an empty --stg-blocks does not disable it on this "
                + "reference: it runs a full-price pass that perturbs nothing. Defaults "
                + "to 1.0; refused with --two-stage."))
    var stgScale: Double?

    @Option(help: ArgumentHelp("Comma-separated blocks STG perturbs. Defaults to [28]."))
    var stgBlocks: String?

    @Option(help: ArgumentHelp("Modality guidance. Neutral at 1.0; anything else costs "
                + "one extra forward pass per step, shared across both directions. "
                + "Defaults to 3.0; refused with --two-stage."))
    var modalityScale: Double?

    @Option(help: ArgumentHelp("Guidance rescale. Neutral at 0.0; applied to the "
                + "combination, costs no pass. Defaults to 0.7; refused with --two-stage."))
    var rescaleScale: Double?

    @Option(help: ArgumentHelp("STG window start, 0…1. p=0 is the first sampled sigma "
                + "(~1), p=1 is the end (sigma 0). Default 0. A window excluding every "
                + "sampled sigma costs the STG pass nothing and is bit-identical to "
                + "--stg-scale 0. Refused with --two-stage."))
    var stgStart: Double = 0

    @Option(help: ArgumentHelp("STG window end, 0…1. Default 1 (the whole schedule)."))
    var stgEnd: Double = 1

    @Option(help: ArgumentHelp("Modality-guidance window start, 0…1. Same mapping as "
                + "--stg-start. Default 0. Refused with --two-stage."))
    var modalityStart: Double = 0

    @Option(help: ArgumentHelp("Modality-guidance window end, 0…1. Default 1."))
    var modalityEnd: Double = 1

    @Option(help: ArgumentHelp("Seed for the initial latent. Drives MLX's own RNG, so a "
                + "render is reproducible on this machine."))
    var seed: UInt64 = 0

    @Option(help: ArgumentHelp("Pixel frames. Must be on the 8k+1 lattice; the decode "
                + "policy this port has measured covers 97."))
    var frames: Int = 97

    @Option(help: "Width in pixels. Overrides megapixels. Must be a multiple of 64.")
    var width: Int?

    @Option(help: "Height in pixels. Overrides megapixels. Must be a multiple of 64.")
    var height: Int?

    @Option(help: "Megapixels (e.g. 0.2, 0.7, 1.0, 2.0).")
    var megapixels: Double?

    @Option(help: ArgumentHelp("Aspect ratio (16:9, 9:16, 1:1, 21:9, 4:3, 3:4)."))
    var aspectRatio: String = "16:9"

    @Option(help: ArgumentHelp("Frame rate. 24 is the default — at 25 the audio stream "
                + "silently gets 97 latents instead of 101."))
    var fps: Double = 24

    @Option(help: ArgumentHelp("Still image to condition on (image-to-video). Encoded to "
                + "latent frame 0 and re-imposed every step, so a high --strength holds "
                + "the shot there rather than merely starting from it. Works with "
                + "--two-stage: the image is resized and encoded ONCE PER STAGE, at half "
                + "size for the draft and full size for the refine."))
    var image: String?

    @Option(help: ArgumentHelp("Still to condition the LAST frame on. Shorthand for "
                + "--keyframe PATH@last. Combine with --image for a start-and-end render."))
    var lastFrame: String?

    @Option(help: ArgumentHelp("A still pinned to a frame, repeatable: PATH@FRAME or "
                + "PATH@FRAME:STRENGTH. FRAME accepts an integer, 'first' or 'last', and "
                + "must land on a latent boundary — the VAE is causal, so frame 0 encodes "
                + "alone and every 8 frames after it become one latent. A frame inside a "
                + "chunk is refused rather than rounded, because rounding would freeze all "
                + "8 of that chunk's frames to the one still. The optional strength "
                + "overrides --strength for that keyframe alone, so an anchored opening and "
                + "a loose closing frame are one command. e.g. --keyframe end.png@last:0.6"))
    var keyframe: [String] = []

    @Option(help: ArgumentHelp("Video to condition on (video-to-video). Must match the "
                + "render's frame count and dimensions. Refused with --two-stage: the "
                + "distilled pipeline takes only images."))
    var video: String?

    @Option(help: ArgumentHelp("Mel spectrogram (.safetensors) to condition on "
                + "(audio-to-video); the audio stream is frozen entirely and video is "
                + "denoised against it. Use --audio to supply a waveform instead."))
    var audioMel: String?

    @Option(help: ArgumentHelp("Audio file (.wav, .mp4, anything AVAudioFile opens) to "
                + "condition on (audio-to-video). Goes through the measured mel front "
                + "end. Trimmed or zero-padded to the exact length this geometry needs, "
                + "and resampled if it is not already at the audio VAE's rate — the "
                + "resampler is AVAudioConverter, and that it ran at all is recorded "
                + "in the provenance sidecar."))
    var audio: String?

    @Option(help: ArgumentHelp("Conditioning strength, 0…1. 1.0 pins the conditioned "
                + "tokens exactly; 0.0 ignores them. Does not apply to audio, which is "
                + "frozen outright."))
    var strength: Double = 1.0

    @Option(help: ArgumentHelp("Cross-step residual cache threshold. 0 (default) is "
                + "dense — every block of every pass. >0 reuses the previous step's "
                + "stack residual when block 0 has barely moved. Swept: 0.01 skips "
                + "nothing and proves the probe is free, 0.05 costs detail for 1.10x, "
                + "and 0.10 is 2.78x and fails on viewing. Judge any setting above 0 "
                + "by eye. One cache per guidance pass; a shared cache would silently "
                + "mix CFG/STG/modality residuals. Refused with --two-stage."))
    var cacheThreshold: Double = 0

    @Option(help: ArgumentHelp("Consecutive-reuse ceiling for the residual cache. "
                + "5 is the cap that survived viewing. Ignored when --cache-threshold is 0."))
    var cacheMaxSkips: Int = 5

    @Flag(help: ArgumentHelp("Run the two-stage distilled pipeline: "
                + "a draft at half width and "
                + "height over 8 ancestral steps, an x2 spatial upsample of the VIDEO "
                + "latent alone, then 3 deterministic Euler steps at full size. No "
                + "guidance and no negative prompt — 11 transformer forwards for the whole "
                + "render against the single-stage path's 120. Accepts --image (built per "
                + "stage) and --strength; refuses --steps, the guidance knobs, --video and "
                + "the audio conditioning flags by name. Defaults --checkpoint to the "
                + "distilled transformer and requires --upsampler. "
                + "EQUIVALENT TO --recipe distilled, which is the spelling to prefer."))
    var twoStage: Bool = false

    @Option(help: ArgumentHelp("A LoRA adapter, repeatable: NAME or NAME@STRENGTH "
                + "(default 1.0). NAME is either a path or a fragment of a filename, "
                + "resolved against the model tree the --checkpoint sits in — a fragment "
                + "matching nothing, or more than one adapter, is refused by name. Only "
                + "files whose own keys carry LoRA tensors are candidates, so pointing this "
                + "at a latent upscaler is caught before the 42 GB transformer is read. "
                + "Applied as a residual overlay, never baked, so stacking two is well "
                + "defined and the resident cost is the adapter file. STACKS ON TOP of "
                + "whatever the recipe already asks for."))
    var lora: [String] = []

    @Option(help: ArgumentHelp("The pipeline to run, by name. `ltx recipes` lists them with "
                + "the shape each resolves to and the evidence behind it. Defaults to "
                + "'prod'. The recipe decides the stages, the schedule, the sampler, the "
                + "guidance and the transformer — so --steps and the guidance flags are "
                + "overrides on top of it, and are refused outright by a recipe whose "
                + "schedule is a literal table."))
    var recipe: String?

    // MARK: Paths — configured, not typed
    //
    // These were five `String` options with hardcoded defaults. They are optional overrides
    // now: the config file supplies them, and naming one here prints an OVERRIDE line to
    // stderr. See `Paths` for why an unannounced override is the dangerous kind.

    @Option(help: ArgumentHelp("Config file. Defaults to ~/.config/ltx/config.json, and to "
                + "built-in paths when that does not exist. `ltx doctor` writes one."))
    var config: String?

    @Option(help: ArgumentHelp("OVERRIDE the DiT checkpoint. Defaults to whichever "
                + "transformer the recipe's first stage names — dev for 'prod', distilled "
                + "for 'distilled'. The two files are byte-identical in structure, so "
                + "naming the wrong one produces a render rather than an error."))
    var checkpoint: String?

    @Option(help: "OVERRIDE the x2 latent spatial upsampler (multi-stage recipes only).")
    var upsampler: String?

    @Option(help: "OVERRIDE the text encoder, which also carries the tokenizer.")
    var textEncoder: String?

    @Option(help: "OVERRIDE the video VAE.")
    var videoVae: String?

    @Option(help: "OVERRIDE the audio VAE.")
    var audioVae: String?

    mutating func run() async throws {
        var blocks: [Int]?
        if let stgBlocks {
            let parts = stgBlocks.split(separator: ",")
            let parsed = parts.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard parsed.count == parts.count else {
                throw ValidationError("--stg-blocks must be comma-separated integers, got '\(stgBlocks)'")
            }
            blocks = parsed
        }
        guard let reqAspect = LTX25.Aspect(rawValue: aspectRatio) else {
            throw ValidationError(
                "--aspect-ratio must be one of "
                    + LTX25.Aspect.allCases.map(\.rawValue).joined(separator: ", "))
        }

        // One selector, resolved here. `--two-stage` predates the registry and is kept as a
        // spelling of `--recipe distilled`; the two disagreeing is a typo worth reporting
        // rather than a precedence rule worth inventing.
        if twoStage, let recipe, recipe != LTX25.Recipes.distilled.id {
            throw ValidationError(
                "--two-stage is --recipe \(LTX25.Recipes.distilled.id); it cannot be "
                    + "combined with --recipe \(recipe)")
        }
        // The config's `default_recipe` is the fallback, not the registry's — otherwise a
        // caller who set it would watch `render` ignore it while `doctor` reported it.
        let configuredDefault = (try? Paths(configPath: config))?
            .config.policy.defaultRecipe
        let recipeID = recipe ?? (twoStage ? LTX25.Recipes.distilled.id
                                           : (configuredDefault ?? LTX25.Recipes.defaultID))
        let chosen: LTX25.Pipeline
        do {
            chosen = try LTX25.Recipes.recipe(recipeID)
        } catch {
            throw ValidationError("\(error). Known recipes: "
                + LTX25.Recipes.ids.joined(separator: ", "))
        }
        // Guarded on the command rather than on the kind. `ingredients` and `msr` are
        // generative — prompt, seed, a size the caller chooses — and still not routes this
        // command can run: their reference tokens make the transformer's sequence longer
        // than the one `render` sizes its noise, masks and unpatchify from. Refusing them by
        // kind would have let them through and rendered without the reference at all, which
        // is the failure that completes successfully.
        guard chosen.command == "render" else {
            let why = chosen.kind == LTX25.Pipeline.Kind.transform
                ? "transforms a clip that already exists rather than rendering from a prompt"
                : "conditions on reference images that ride in the video sequence, which "
                    + "this command cannot size"
            throw ValidationError(
                "recipe '\(recipeID)' \(why); run it through `ltx \(chosen.command)`")
        }

        let outURL = URL(fileURLWithPath: out)
        let request = LTX25.Request(
            prompt: prompt,
            videoOutput: outURL,
            audioOutput: outURL.deletingPathExtension().appendingPathExtension("wav"),
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seconds: Double(frames) / fps,
            frames: frames,
            frameRate: fps,
            // The recipe owns the schedule. `prod` is 30 steps, `distilled` is a
            // pair of fixed tables; neither is a number a caller supplies.
            steps: nil,
            seed: seed,
            megapixels: megapixels,
            aspectRatio: reqAspect,
            width: width,
            height: height,
            recipeID: recipeID,
            image: image.map { URL(fileURLWithPath: $0) },
            images: try keyframe.map { try LTX25.Keyframe.parse($0) }
                + (lastFrame.map {
                    [LTX25.Keyframe(url: URL(fileURLWithPath: $0),
                                       pixelFrame: LTX25.Request.lastFramePlaceholder)]
                } ?? []),
            video: video.map { URL(fileURLWithPath: $0) },
            audio: audio.map { URL(fileURLWithPath: $0) },
            audioMel: audioMel.map { URL(fileURLWithPath: $0) },
            strength: strength,
            adapters: try lora.map { try LTX25.Adapter.parse($0) },
            videoCFG: videoCfg,
            audioCFG: audioCfg,
            stgScale: stgScale,
            stgBlocks: blocks,
            modalityScale: modalityScale,
            rescaleScale: rescaleScale,
            stgStartPercent: stgStart,
            stgEndPercent: stgEnd,
            modalityStartPercent: modalityStart,
            modalityEndPercent: modalityEnd,
            cacheThreshold: cacheThreshold,
            cacheMaxSkips: cacheMaxSkips
        )

        // The transformer follows the recipe's own role, not a flag. The dev and distilled
        // files are byte-for-byte the same size with identical embedded configs, so running
        // the wrong one produces a render rather than an error — which is exactly why the
        // choice should follow the named pipeline.
        var paths = try Paths(configPath: config)
        let usesDistilled = chosen.stages.first?.transformer == .distilled
        let checkpoints = LTX25.Checkpoints(
            textEncoder: try paths.url(.textEncoder, override: textEncoder),
            dit: try paths.url(usesDistilled ? .ditDistilled : .ditDev, override: checkpoint),
            videoVAE: try paths.url(.videoVAE, override: videoVae),
            audioVAE: try paths.url(.audioVAE, override: audioVae),
            upsampler: chosen.stages.count > 1
                ? try paths.url(.upsampler, override: upsampler) : nil)
        paths.announce()

        // Before the engine, not inside it: a missing checkpoint found here costs a header
        // read, and found later costs the 26 GB text-encoder pass that preceded it.
        var slots = [
            CheckpointInventory.Slot(role: "dit", url: checkpoints.dit,
                                     expected: .transformer,
                                     nameMustContain: usesDistilled ? "distilled" : "dev"),
            CheckpointInventory.Slot(role: "text encoder", url: checkpoints.textEncoder,
                                     expected: .textEncoder),
            CheckpointInventory.Slot(role: "video vae", url: checkpoints.videoVAE,
                                     expected: .videoVAECausal),
            CheckpointInventory.Slot(role: "audio vae", url: checkpoints.audioVAE,
                                     expected: .audioVAE),
        ]
        if let upsamplerURL = checkpoints.upsampler {
            slots.append(CheckpointInventory.Slot(role: "upsampler", url: upsamplerURL,
                                                 expected: .latentUpscaler))
        }
        try Preflight.check(paths: paths, slots: slots)

        let started = Date()
        var result: LTX25.RenderResult?

        let job = try await RenderEngine.shared.startJob(
            request: request, checkpoints: checkpoints,
            memoryMarginFraction: paths.config.policy.memoryMarginFraction)
        for await event in job.events {
            switch event {
            case .started:
                FileHandle.standardError.write(Data("starting render job...\n".utf8))
            case .textEncodingStarted:
                FileHandle.standardError.write(Data(
                    "encoding text (loading the 26 GB encoder; it is released before the DiT)\n"
                        .utf8))
            case let .textEncodingCompleted(validTokens):
                FileHandle.standardError.write(Data(
                    "text encoded: valid tokens per branch \(validTokens)\n".utf8))
            case let .samplingProgress(step, total):
                let elapsed = Int(Date().timeIntervalSince(started))
                FileHandle.standardError.write(Data(
                    "step \(step + 1)/\(total)  \(elapsed)s elapsed\n".utf8))
            case let .completed(res):
                result = res
            case let .failed(error):
                throw error
            case .cancelled:
                FileHandle.standardError.write(Data("job cancelled\n".utf8))
                throw ExitCode(1)
            }
        }

        guard let result else {
            throw ValidationError("job completed without a result")
        }
        if let muxFailed = result.muxFailed {
            FileHandle.standardError.write(Data(
                ("wrote \(result.sidecar.path)\nNO mp4: \(muxFailed)\n").utf8))
            throw ExitCode(1)
        }
        FileHandle.standardError.write(Data(
            ("wrote \(result.video.path)\nwrote \(result.sidecar.path)\n")
                .utf8))
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXCatalog
import LTXFoundation
import LTXPipeline
import LTXRecipes

/// `ltx ingredients` — render from a prompt plus a reference sheet.
///
/// A focused command rather than a recipe. In-context reference conditioning is a different
/// shape of job from the generative ladder: the sequence the transformer sees is longer than
/// the one that gets decoded, and the recipe machinery sizes noise, masks and the unpatchify
/// from a single token count. Threading that through is a larger change than this adapter
/// needs in order to be *tried*, and trying it is the point — so this reaches the
/// reference-guided sampler directly and says so.
struct Ingredients: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "ingredients",
        abstract: "Render a clip from a prompt and a reference sheet, through an IC-LoRA.",
        discussion: """
            The Ingredients IC-LoRA conditions on a sheet holding the characters, objects and \
            locations a shot should contain. The sheet is encoded at each STAGE's resolution \
            — the adapter's reference_downscale_factor is 1 — and its latent tokens ride \
            alongside the generated ones, held clean, then dropped before the decode.

            THERE IS NO --width OR --height. Every shape belongs to the recipe: the output \
            comes from --megapixels and --aspect-ratio through the shape ladder, or from the \
            recipe's own measured shape when you name neither, and what each STAGE samples at \
            comes from that stage's scale. This is not tidiness. Draft resolution is the knob \
            that decides whether rigid structure survives — a subject on a 10x6 latent grid \
            has nowhere to put mechanical detail, and a refine that re-noises to 0.909 then \
            invents it independently per frame, which is what melting is. It was briefly a \
            flag, and a release where --width meant the output rather than the sampled size \
            halved everyone's draft without changing their command line.

            Three recipes run here; `ltx recipes` prints them with what each is measured at. \
            'ingredients' is the adapter's own arrangement — the distilled transformer, its \
            8-step draft table, no guidance, one forward per step — in two stages, drafting at \
            half and refining after an x2 upsample. 'ingredients-prod' is the dev transformer \
            on a continuous 30-step schedule with production guidance, ONE stage at full \
            resolution: four forwards per step. 'ingredients-prod-cfg' is that with CFG alone, \
            which is two forwards per step and half the wall clock for the same resolution.

            Two stages is not a speed setting. It makes a LARGER output cheaper than rendering \
            it natively — measured at 1.95x — and it does not make a given output faster, \
            because the draft it saves on is the draft the picture is built from. The prod \
            recipes buy their time back by dropping guidance passes instead of pixels.

            The shipped adapter is a 2.3 checkpoint. All 480 of its modules resolve against \
            the 2.5 transformer with matching shapes, which is a statement about key sets and \
            not about whether its features transfer.
            """)

    static let hint = "ic-lora-ingredients"

    @Option(name: .shortAndLong, help: "The reference sheet image.")
    var reference: String

    @Option(name: .shortAndLong, help: "The prompt.")
    var prompt: String

    @Option(name: .shortAndLong, help: "Output .mp4.")
    var out: String = "ingredients.mp4"

    @Option(help: ArgumentHelp("Megapixel target, resolved against --aspect-ratio through "
                + "the shape ladder. Omitted takes the recipe's own measured shape."))
    var megapixels: Double?

    @Option(help: ArgumentHelp("Aspect ratio: "
                + AspectRatio.allCases.map(\.rawValue).joined(separator: ", ") + "."))
    var aspectRatio: String = "16:9"

    @Option(help: "Frame count; must satisfy frames % 8 == 1.")
    var frames: Int = 49

    @Option(help: "Frame rate.")
    var fps: Double = 24

    @Option(help: ArgumentHelp("Which registered recipe to run. Every shape, schedule and "
                + "guidance setting comes from it — see `ltx recipes` for the menu and what "
                + "each has been measured at."))
    var recipe: String = RecipeRegistry.ingredients.id


    @Option(help: ArgumentHelp("Force the latent every n-th step rather than every step. "
                + "Each force is a barrier where the GPU drains; this port measures ~0.4 s "
                + "of per-forward cost that does not scale with tokens, and a cadence "
                + "amortises it. It also keeps n steps of graph live instead of one, so the "
                + "working set grows with it. UNMEASURED for quality-neutrality at >1 — it "
                + "should be exactly neutral, since nothing about the arithmetic changes."))
    var evalCadence: Int = 1

    @Option(help: ArgumentHelp("Reuse the previous step's residual when block 0's moved less "
                + "than this. 0 (default) is the dense loop and is what every measurement "
                + "here was taken with. 0.10 is the measured knee, ON A DIFFERENT MODEL — "
                + "the mechanism transfers, the number has not been checked against LTX."))
    var cacheThreshold: Double = 0

    @Option(help: "Seed.")
    var seed: UInt64 = 42

    @Option(help: "IC-LoRA strength.")
    var strength: Float = 1.0

    @Option(help: ArgumentHelp("OVERRIDE the IC-LoRA. Defaults to resolving the hint "
                + "'\(Ingredients.hint)' against the adapter roots."))
    var icLora: String?

    @Option(help: ArgumentHelp("OVERRIDE the DiT. Defaults to the transformer the recipe's "
                + "stages declare — dev or distilled."))
    var checkpoint: String?

    @Option(help: "OVERRIDE the text encoder.")
    var textEncoder: String?

    @Option(help: "OVERRIDE the video VAE.")
    var videoVae: String?

    @Option(help: "OVERRIDE the x2 latent spatial upsampler, which stage 2 runs.")
    var upsampler: String?

    @Option(help: "OVERRIDE the audio VAE. Every render carries sound, so this is always read.")
    var audioVae: String?

    @Option(help: "Path to the config file.")
    var config: String?

    mutating func run() async throws {
        var paths = try Paths(configPath: config)
        guard let reqAspect = AspectRatio(rawValue: aspectRatio) else {
            throw ValidationError("--aspect-ratio must be one of "
                + AspectRatio.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let chosen: Recipe
        do {
            chosen = try RecipeRegistry.recipe(recipe)
        } catch {
            throw ValidationError("\(error). Recipes for this command: "
                + RecipeRegistry.all.filter { $0.command == "ingredients" }
                    .map(\.id).joined(separator: ", "))
        }
        // Guarded on the command the recipe declares, not on its kind. `msr` is generative
        // and reference-conditioned too, and running it here would attach the wrong adapter
        // and render something plausible.
        guard chosen.command == "ingredients" else {
            throw ValidationError("recipe '\(recipe)' runs through `ltx \(chosen.command)`")
        }
        // Every shape decided here, before a checkpoint is opened. The stage geometry comes
        // from each stage's `scale` against this output — which is why no flag on this
        // command can move what stage 1 samples at.
        let plan = try chosen.resolve(RecipeRequest(
            prompt: prompt, videoOutput: URL(fileURLWithPath: out),
            seconds: Double(frames) / fps, frames: frames, frameRate: fps,
            seed: seed, megapixels: megapixels, aspectRatio: reqAspect))

        // The transformer follows the recipe's declared role. A 30-step continuous schedule
        // on weights distilled to need eight renders, and is wrong.
        let isDev = plan.stages.contains { $0.stage.transformer == .dev }
        let ditURL = try paths.url(isDev ? .ditDev : .ditDistilled, override: checkpoint)

        let adapterURL: URL
        if let icLora {
            adapterURL = try AdapterCatalog.verify(at: URL(fileURLWithPath: icLora)).url
            FileHandle.standardError.write(Data("OVERRIDE --ic-lora \(icLora)\n".utf8))
        } else {
            // Searched across every configured adapter root, not just the one beside the
            // transformer: the 2.3 IC-LoRAs live in their own tree.
            adapterURL = try paths.resolveAdapter(hint: Self.hint)
        }

        let checkpoints = VideoUpscaler.Checkpoints(
            videoVAE: try paths.url(.videoVAE, override: videoVae),
            upsampler: try paths.url(.upsampler, override: upsampler),
            dit: ditURL,
            textEncoder: try paths.url(.textEncoder, override: textEncoder),
            audioVAE: try paths.url(.audioVAE, override: audioVae),
            icLoRA: adapterURL)
        paths.announce()

        try Preflight.check(paths: paths, slots: [
            CheckpointInventory.Slot(role: "dit", url: ditURL, expected: .transformer,
                                     nameMustContain: isDev ? "dev" : "distilled"),
            CheckpointInventory.Slot(role: "text encoder", url: checkpoints.textEncoder!,
                                     expected: .textEncoder),
            CheckpointInventory.Slot(role: "video vae", url: checkpoints.videoVAE,
                                     expected: .videoVAECausal),
            CheckpointInventory.Slot(role: "upsampler", url: checkpoints.upsampler,
                                     expected: .latentUpscaler),
        ])

        // Identified, not just opened: the version it declares is the thing worth saying
        // out loud here, because a 2.3 adapter on a 2.5 base is the open question.
        let identity = try AdapterCatalog.verify(at: adapterURL)
        print("adapter  \(identity.name)")
        print("         v\(identity.modelVersion ?? "?")  \(identity.tensorCount) tensors")

        print("recipe   \(plan.recipeID)  \(plan.output.description), "
              + "\(plan.stages.count) stage(s), \(plan.forwardCount()) forwards")
        print("         \(plan.evidence.label)")

        var options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: self.reference),
            prompt: prompt,
            output: URL(fileURLWithPath: out),
            plan: plan, seed: seed, strength: strength)
        // Machine tuning, set after construction rather than threaded through the recipe
        // initialiser: neither changes what is rendered, only what rendering it costs.
        options.evalCadence = evalCadence
        options.cacheThreshold = cacheThreshold

        let report = try IngredientsRenderer.render(
            options,
            checkpoints: checkpoints,
            note: { FileHandle.standardError.write(Data("  \($0)\n".utf8)) })

        print(String(format: """
            reference  %@ -> %d tokens
            generated  %d tokens
            output     %dx%d, %d frames
            encode     %.1f s
            stage 1    %.1f s
            stage 2    %.1f s
            decode     %.1f s
            peak       %.1f GB
            wrote      %@
            """,
            "\(report.referenceLatentShape)", report.referenceTokens,
            report.generatedTokens, report.outputWidth, report.outputHeight,
            report.framesWritten, report.encodeSeconds, report.sampleSeconds,
            report.refineSeconds, report.decodeSeconds, report.peakGB, out))
    }
}

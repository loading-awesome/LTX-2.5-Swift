// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXCatalog
import LTXFoundation
import LTXPipeline
import LTXRecipes

/// `ltx msr` — render a shot with a cast of references, each held apart.
///
/// A focused command rather than a recipe, for the reason ``Ingredients`` gives and one
/// more. The recipe machinery sizes noise, masks and the unpatchify from a *single* token
/// count, and in-context conditioning makes the sequence the transformer sees longer than
/// the one that gets decoded — with five references it can be longer than the target itself.
/// `RecipeStage` has no field for a reference at all, so an MSR "recipe" would be a stage
/// whose declared shape omits most of what it runs.
struct MSR: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "msr",
        abstract: "Render from a prompt and up to five reference stills, each in its own slot.",
        discussion: """
            Multiple-Subject-Reference conditions on several stills at once and keeps them \
            distinguishable, which single-reference IC-LoRAs cannot do. Two mechanisms carry \
            that: a learned slot vector added to each reference's latent channels, and a \
            distinct NEGATIVE time offset per slot — -(n - i) pixel frames, so pic1 sits \
            furthest back and no reference shares coordinates with another or with the \
            target's own frame 0.

            References are given in slot order. Pass --background for a location or plate: \
            it takes the last slot and is centre-cropped to the frame, where a subject is \
            padded rather than cropped when its aspect ratio disagrees with the output's.

            Each still is repeated to --reference-frames frames and encoded at the STAGE's \
            resolution, so the cast costs real sequence length: five references at 33 frames \
            occupy five times what a 33-frame clip would.

            THERE IS NO --width OR --height. The output comes from --megapixels and \
            --aspect-ratio through the shape ladder, or from the recipe's own measured shape \
            when you name neither, and what each STAGE samples at comes from that stage's \
            scale. Draft resolution decides whether rigid structure survives, so it is a \
            recipe's decision rather than a flag's.

            Three recipes run here; `ltx recipes` prints them. 'msr' is the adapter's own \
            arrangement — the distilled transformer on its 8-step table, no guidance, one \
            forward per step. 'msr-prod' is the dev transformer on a continuous 30-step \
            schedule with production guidance, four forwards per step. 'msr-prod-cfg' is that \
            with CFG alone: two forwards per step, half the wall clock, same resolution.

            All three are single stage. A refine would re-prepare, re-encode and re-tag every \
            slot at the output resolution — reference_downscale_factor is 1, so a reference \
            must sit on its own stage's grid, and a five-slot cast would pay five encodes \
            twice for a draft that had less structure in it to begin with.
            """)

    static let hint = "licon-msr"

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: ArgumentHelp("Reference stills, in slot order. 1-5, or 1-4 with "
                + "--background.", valueName: "image"))
    var reference: [String]

    @Option(help: ArgumentHelp("A location or plate. Takes the last slot and is "
                + "centre-cropped rather than padded.", valueName: "image"))
    var background: String?

    @Option(name: .shortAndLong, help: "The prompt.")
    var prompt: String

    @Option(name: .shortAndLong, help: "Output .mp4.")
    var out: String = "msr.mp4"

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
                + "guidance setting comes from it — see `ltx recipes` for the menu."))
    var recipe: String = RecipeRegistry.msr.id



    @Option(help: "Seed.")
    var seed: UInt64 = 42

    @Option(help: "Conditioning strength; 1.0 holds the references perfectly clean.")
    var strength: Float = 1.0

    @Option(help: ArgumentHelp("Frames each still is repeated to before encoding: 25 or 33. "
                + "33 is the plugin's default and gives 5 latent frames per slot.",
                valueName: "n"))
    var referenceFrames: Int = 33

    @Option(help: ArgumentHelp("OVERRIDE the MSR adapter. Defaults to resolving the hint "
                + "'\(MSR.hint)' against the adapter roots."))
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
        guard let referenceFrames = MSRRenderer.ReferenceFrames(rawValue: self.referenceFrames)
        else {
            throw ValidationError("--reference-frames must be 25 or 33, not "
                + "\(self.referenceFrames). Both sit on the VAE's 8k+1 lattice; other counts "
                + "do not, and the plugin offers exactly these two.")
        }

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
                + RecipeRegistry.all.filter { $0.command == "msr" }
                    .map(\.id).joined(separator: ", "))
        }
        guard chosen.command == "msr" else {
            throw ValidationError("recipe '\(recipe)' runs through `ltx \(chosen.command)`")
        }
        // Every shape decided before a checkpoint is opened, from the stages' own scales.
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

        let identity = try AdapterCatalog.verify(at: adapterURL)
        print("adapter  \(identity.name)")
        print("         v\(identity.modelVersion ?? "?")  \(identity.tensorCount) tensors")

        print("recipe   \(plan.recipeID)  \(plan.output.description), "
              + "\(plan.stages.count) stage(s), \(plan.forwardCount()) forwards")
        print("         \(plan.evidence.label)")

        let report = try MSRRenderer.render(
            MSRRenderer.Options(
                references: reference.map { URL(fileURLWithPath: $0) },
                background: background.map { URL(fileURLWithPath: $0) },
                prompt: prompt,
                output: URL(fileURLWithPath: out),
                plan: plan, seed: seed, strength: strength,
                referenceFrames: referenceFrames),
            checkpoints: checkpoints,
            note: { FileHandle.standardError.write(Data("  \($0)\n".utf8)) })

        print(String(format: """
            slots      %d, offsets %@ frames
            reference  %@ each -> %d tokens total
            generated  %d tokens
            output     %dx%d, %d frames
            encode     %.1f s
            stage 1    %.1f s
            stage 2    %.1f s
            decode     %.1f s
            peak       %.1f GB
            wrote      %@
            """,
            report.slots, "\(report.frameOffsets)",
            "\(report.referenceLatentShape)", report.referenceTokens,
            report.generatedTokens, report.outputWidth, report.outputHeight,
            report.framesWritten, report.encodeSeconds, report.sampleSeconds,
            report.refineSeconds, report.decodeSeconds, report.peakGB, out))
    }
}

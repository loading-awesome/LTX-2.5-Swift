// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXCatalog
import LTXFoundation
import LTXPipeline

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
            locations a shot should contain. The sheet is encoded at the OUTPUT's resolution \
            — the adapter's reference_downscale_factor is 1 — and its latent tokens ride \
            alongside the generated ones, held clean, then dropped before the decode.

            Runs on the DISTILLED transformer at the distilled 3-step schedule, which is what \
            the reference's ICLoraPipeline does.

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

    @Option(help: "Width in pixels; a multiple of 64.")
    var width: Int = 640

    @Option(help: "Height in pixels; a multiple of 64.")
    var height: Int = 384

    @Option(help: "Frame count; must satisfy frames % 8 == 1.")
    var frames: Int = 49

    @Option(help: "Frame rate.")
    var fps: Double = 24

    @Option(help: "Sampler steps. 3 is the distilled schedule's own count.")
    var steps: Int = 3

    @Option(help: "Seed.")
    var seed: UInt64 = 42

    @Option(help: "IC-LoRA strength.")
    var strength: Float = 1.0

    @Option(help: ArgumentHelp("OVERRIDE the IC-LoRA. Defaults to resolving the hint "
                + "'\(Ingredients.hint)' against the adapter roots."))
    var icLora: String?

    @Option(help: "OVERRIDE the DiT. Defaults to the distilled transformer.")
    var checkpoint: String?

    @Option(help: "OVERRIDE the text encoder.")
    var textEncoder: String?

    @Option(help: "OVERRIDE the video VAE.")
    var videoVae: String?

    @Flag(name: .customLong("upsample"),
          help: ArgumentHelp("Run the pipeline's second stage: x2 latent upsample then "
              + "refine, so the file written is twice --width x --height."))
    var upsampleStage2 = false

    @Option(help: ArgumentHelp("Stage-2 refine steps.", valueName: "n"))
    var refineSteps = 3

    @Option(help: ArgumentHelp("OVERRIDE the latent spatial upsampler. Unused by this "
                + "command; the checkpoint bundle requires it."))
    var upsampler: String?

    @Option(help: "OVERRIDE the audio VAE. Every render carries sound, so this is always read.")
    var audioVae: String?

    @Option(help: "Path to the config file.")
    var config: String?

    mutating func run() async throws {
        var paths = try Paths(configPath: config)
        let ditURL = try paths.url(.ditDistilled, override: checkpoint)

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
                                     nameMustContain: "distilled"),
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

        let report = try IngredientsRenderer.render(
            IngredientsRenderer.Options(
                reference: URL(fileURLWithPath: self.reference),
                prompt: prompt,
                output: URL(fileURLWithPath: out),
                width: width, height: height, frames: frames, frameRate: fps,
                steps: steps, seed: seed, strength: strength,
            upsample: upsampleStage2, refineSteps: refineSteps),
            checkpoints: checkpoints,
            note: { FileHandle.standardError.write(Data("  \($0)\n".utf8)) })

        print(String(format: """
            reference  %@ -> %d tokens
            generated  %d tokens
            encode     %.1f s
            sample     %.1f s
            decode     %.1f s
            peak       %.1f GB
            wrote      %@
            """,
            "\(report.referenceLatentShape)", report.referenceTokens,
            report.generatedTokens, report.encodeSeconds, report.sampleSeconds,
            report.decodeSeconds, report.peakGB, out))
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXCatalog
import LTXFoundation
import LTXPipeline

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

            Each still is repeated to --reference-frames frames and encoded at the OUTPUT's \
            resolution, so the cast costs real sequence length: five references at 33 frames \
            occupy five times what a 33-frame clip would. Runs on the DISTILLED transformer \
            at the distilled schedule.
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

    @Option(help: "Conditioning strength; 1.0 holds the references perfectly clean.")
    var strength: Float = 1.0

    @Option(help: ArgumentHelp("Frames each still is repeated to before encoding: 25 or 33. "
                + "33 is the plugin's default and gives 5 latent frames per slot.",
                valueName: "n"))
    var referenceFrames: Int = 33

    @Option(help: ArgumentHelp("OVERRIDE the MSR adapter. Defaults to resolving the hint "
                + "'\(MSR.hint)' against the adapter roots."))
    var icLora: String?

    @Option(help: "OVERRIDE the DiT. Defaults to the distilled transformer.")
    var checkpoint: String?

    @Option(help: "OVERRIDE the text encoder.")
    var textEncoder: String?

    @Option(help: "OVERRIDE the video VAE.")
    var videoVae: String?

    @Option(help: ArgumentHelp("OVERRIDE the latent spatial upsampler. Unused by this "
                + "command; the checkpoint bundle requires it."))
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
        let ditURL = try paths.url(.ditDistilled, override: checkpoint)

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
                                     nameMustContain: "distilled"),
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

        let report = try MSRRenderer.render(
            MSRRenderer.Options(
                references: reference.map { URL(fileURLWithPath: $0) },
                background: background.map { URL(fileURLWithPath: $0) },
                prompt: prompt,
                output: URL(fileURLWithPath: out),
                width: width, height: height, frames: frames, frameRate: fps,
                steps: steps, seed: seed, strength: strength,
                referenceFrames: referenceFrames),
            checkpoints: checkpoints,
            note: { FileHandle.standardError.write(Data("  \($0)\n".utf8)) })

        print(String(format: """
            slots      %d, offsets %@ frames
            reference  %@ each -> %d tokens total
            generated  %d tokens
            encode     %.1f s
            sample     %.1f s
            decode     %.1f s
            peak       %.1f GB
            wrote      %@ (%d frames)
            """,
            report.slots, "\(report.frameOffsets)",
            "\(report.referenceLatentShape)", report.referenceTokens,
            report.generatedTokens, report.encodeSeconds, report.sampleSeconds,
            report.decodeSeconds, report.peakGB, out, report.framesWritten))
    }
}

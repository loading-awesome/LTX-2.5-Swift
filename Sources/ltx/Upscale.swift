// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import LTXCatalog
import Foundation
import LTXPipeline

/// `ltx upscale` — x2 an existing clip through the latent spatial upsampler.
///
/// A sibling of `render` rather than a mode of it, for the reason `VideoUpscaler` gives:
/// there is no prompt, no seed, no schedule and no output size to choose here. Every
/// guidance flag `render` carries is already refused under `--two-stage`, so routing a
/// video through them would be a wider seam for less.
struct Upscale: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "upscale",
        abstract: "Upscale an existing video x2 through the latent spatial upsampler.")

    @Option(help: "The video to upscale. Both axes must be a multiple of 32.")
    var input: String

    @Option(help: ArgumentHelp("Where to write the mp4. The source's audio is carried over "
                + "unchanged by default and trimmed to the new picture duration; pass "
                + "--silent to drop it."))
    var out: String

    @Flag(help: ArgumentHelp("Write picture only. By default the input's audio track is "
                + "copied into the output at its own rate — carried, never regenerated, "
                + "because every mode here changes the picture and none of them has "
                + "anything to say about the sound."))
    var silent: Bool = false

    @Option(help: ArgumentHelp("plain: encode, x2 upsample, decode — no transformer and no "
                + "text encoder, 3.4 GB of checkpoints. refined: adds the recipe's stage-2, "
                + "3 deterministic Euler steps on sigmas [0.909375, 0.725, 0.421875, 0.0] "
                + "on the distilled checkpoint, which costs the 42 GB DiT and a 26 GB "
                + "text-encoder pass. Note sigma[0]=0.909375 — the refine re-noises to 91% "
                + "and re-denoises, so it is licensed to move away from the source."))
    var mode: String = "plain"

    @Option(help: ArgumentHelp("How far --mode refined re-noises before re-denoising, as a "
                + "starting sigma. The recipe's own value is 0.909375 and that is the "
                + "default — but it is chosen for cleaning up an 8-step DRAFT, and on a "
                + "finished clip it re-renders rather than upscales. Lower it until the "
                + "picture stops changing identity. Any value below 0.909375 rescales the "
                + "curve and is a generalisation of the recipe."))
    var denoise: Double = 0.909375

    @Option(help: ArgumentHelp("Prompt for --mode refined. NOT decoration: whatever the "
                + "refine re-derives, it re-derives toward this, and an empty prompt points "
                + "at the base model's own prior, which for this checkpoint is photographic. "
                + "Pass the film's style string."))
    var prompt: String = ""

    @Option(help: ArgumentHelp("Read the prompt from a file instead. Takes precedence over "
                + "--prompt; convenient for a style string that must stay byte-identical."))
    var promptFile: String?

    @Option(help: ArgumentHelp("Frame rate written into the output container. Does not "
                + "change the picture. Read it off the source."))
    var fps: Double = 24

    @Option(help: ArgumentHelp("Cap the frames read, before the 8k+1 lattice snap. For a "
                + "cheap probe of a long clip."))
    var maxFrames: Int?

    @Option(help: "Config file. Defaults to ~/.config/ltx/config.json.")
    var config: String?

    @Option(help: ArgumentHelp("OVERRIDE the video VAE. Supplies both the codec and the "
                + "upsampler's statistics."))
    var videoVae: String?

    @Option(help: "OVERRIDE the x2 latent spatial upsampler.")
    var upsampler: String?

    @Option(help: ArgumentHelp("OVERRIDE the DiT. Read only by --mode refined and "
                + "--mode ic-lora; defaults to the distilled transformer."))
    var checkpoint: String?

    @Option(help: "OVERRIDE the text encoder. Read only by --mode refined and --mode ic-lora.")
    var textEncoder: String?

    @Option(help: ArgumentHelp("The IC-LoRA adapter, read only by --mode ic-lora. NOT the "
                + "file in --upsampler: that is the latent spatial upscaler that sits "
                + "between the two stages of a distilled render. This one is a rank-32 "
                + "adapter on the transformer and is the checkpoint actually built to "
                + "upscale a clip. Both are about 327 MB and both say 'spatial upscaler'. "
                + "Defaults to resolving the hint '\(Upscale.icLoRAHint)' against the model "
                + "tree, which only considers files whose own keys carry LoRA tensors."))
    var icLora: String?

    @Option(help: ArgumentHelp("Sampler steps for --mode ic-lora. 3 is the recommended "
                + "setting for upscaling and the default; the full recorded schedule is 8. Fewer steps keep the output CLOSER to the "
                + "reference (the model card's own fidelity dial, alongside guidance, which "
                + "this pipeline pins at neutral). Thinning comes out of the schedule's "
                + "near-stationary head, so the working sigmas 0.909/0.725/0.422/0 survive "
                + "down to 3 steps, where the schedule is [1.0, 0.909, 0.422, 0.0]."))
    var steps: Int = 3

    @Option(help: ArgumentHelp("Seed for --mode ic-lora, which renders from noise rather "
                + "than resampling. Change it for a different take of the same upscale."))
    var seed: UInt64 = 0

    /// The filename fragment that identifies the x2 pixel spatial upscaler adapter.
    ///
    /// A hint rather than a path, so this command and `RecipeRegistry.upscaleICLoRA` name
    /// the same adapter the same way and neither carries a machine's directory layout.
    static let icLoRAHint = "ic-lora-pixel-spatial-upscaler"

    mutating func run() async throws {
        // Case- and dash-insensitive: the raw values are Swift-cased (`refineOnly`,
        // `icLoRA`) and nobody types a CLI flag that way.
        func normalise(_ s: String) -> String {
            s.lowercased().replacingOccurrences(of: "-", with: "")
                .replacingOccurrences(of: "_", with: "")
        }
        let wanted = normalise(self.mode)
        guard let mode = VideoUpscaler.Mode.allCases
            .first(where: { normalise($0.rawValue) == wanted }) else {
            throw ValidationError("--mode must be one of "
                + VideoUpscaler.Mode.allCases.map(\.rawValue).joined(separator: ", ")
                + " (dashes and case are ignored)")
        }

        let needsTransformer = mode == .refined || mode == .refineOnly || mode == .icLoRA
        var paths = try Paths(configPath: config)
        let ditURL = try paths.url(.ditDistilled, override: checkpoint)

        // The adapter is resolved by hint against the model tree rather than carried as a
        // sixth configured path: `AdapterCatalog` only considers files whose own keys hold
        // LoRA tensors, so it cannot land on the latent upscaler sitting beside it under a
        // near-identical name — which is the mistake this flag's help text is about.
        var icLoRAURL: URL?
        if mode == .icLoRA {
            if let icLora {
                icLoRAURL = try AdapterCatalog.verify(
                    at: URL(fileURLWithPath: icLora)).url
                FileHandle.standardError.write(
                    Data("OVERRIDE --ic-lora \(icLora)\n".utf8))
            } else {
                let root = AdapterCatalog.modelRoot(containing: ditURL)
                icLoRAURL = try AdapterCatalog.resolve(hint: Self.icLoRAHint, in: root).url
            }
        }

        let checkpoints = VideoUpscaler.Checkpoints(
            videoVAE: try paths.url(.videoVAE, override: videoVae),
            upsampler: try paths.url(.upsampler, override: upsampler),
            dit: needsTransformer ? ditURL : nil,
            textEncoder: needsTransformer
                ? try paths.url(.textEncoder, override: textEncoder) : nil,
            icLoRA: icLoRAURL)
        paths.announce()

        // Only what this mode opens. `plain` runs no transformer and no text encoder, and
        // telling someone their transformer is missing when the command was never going to
        // read it is worse than saying nothing.
        var slots = [
            CheckpointInventory.Slot(role: "video vae", url: checkpoints.videoVAE,
                                     expected: .videoVAECausal),
            CheckpointInventory.Slot(role: "upsampler", url: checkpoints.upsampler,
                                     expected: .latentUpscaler),
        ]
        if needsTransformer {
            slots.append(CheckpointInventory.Slot(role: "dit", url: ditURL,
                                                  expected: .transformer,
                                                  nameMustContain: "distilled"))
            if let encoder = checkpoints.textEncoder {
                slots.append(CheckpointInventory.Slot(role: "text encoder", url: encoder,
                                                      expected: .textEncoder))
            }
        }
        try Preflight.check(paths: paths, slots: slots)

        var text = prompt
        if let promptFile {
            text = try String(contentsOf: URL(fileURLWithPath: promptFile), encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard denoise > 0, denoise <= 1 else {
            throw ValidationError("--denoise must be in (0, 1]; got \(denoise)")
        }

        let report = try await VideoUpscaler.upscale(
            input: URL(fileURLWithPath: input),
            output: URL(fileURLWithPath: out),
            checkpoints: checkpoints,
            mode: mode,
            prompt: text,
            denoise: Float(denoise),
            seed: seed,
            steps: steps,
            keepAudio: !silent,
            fps: fps,
            maxFrames: maxFrames,
            note: { line in FileHandle.standardError.write(Data((line + "\n").utf8)) })

        FileHandle.standardError.write(Data((report.summary + "\n"
            + "wrote \(out)\n").utf8))
    }
}

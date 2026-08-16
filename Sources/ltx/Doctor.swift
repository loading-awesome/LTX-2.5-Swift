// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTX25
import LTXCatalog
import LTXFoundation
import LTXHardware
import LTXRecipes

/// Everything the port decided before it ran anything, and why.
///
/// This exists because the alternative is a support conversation. "It picked the wrong
/// checkpoint", "it says it will not fit", "which transformer am I actually running",
/// "why is that recipe refused" are questions with one answer each, and printing all of
/// them at once costs milliseconds: every checkpoint is identified from its header, which
/// is a few hundred kilobytes regardless of a 39 GiB body.
///
/// **It reports rather than throws.** A doctor that stops at the first missing file makes
/// you run it four times to find four problems.
///
/// No MLX. Every target it imports is MLX-free by construction (`Package.swift`), so this
/// runs on a bare checkout with no Metal, no GPU and no 39 GiB load — which is exactly the
/// property that makes it worth running first.
struct Doctor: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Show the machine, the resolved checkpoints, and what can actually run.",
        discussion: """
            Reads headers only. Nothing here loads a checkpoint, touches the GPU or
            renders anything, so it is safe and fast to run at any time.

              ltx doctor
              ltx doctor --tokens 7488        plan memory for a larger shape
              ltx doctor --checkpoint <path>  check a transformer other than the default
            """
    )

    @Option(help: ArgumentHelp("Video tokens to plan memory for. Defaults to 3120, the "
                + "production shape (640x384x97) every measured figure here is quoted at."))
    var tokens: Int = MemoryPlan.productionVideoTokens

    @Option(help: ArgumentHelp("Check the checkpoints against this recipe's expectations. "
                + "Defaults to 'prod'."))
    var recipe: String = RecipeRegistry.defaultID

    @Option(help: ArgumentHelp("The DiT checkpoint to inspect. Defaults to the transformer "
                + "the chosen recipe's first stage names."))
    var checkpoint: String?

    @Option(help: "OVERRIDE the text encoder.")
    var textEncoder: String?

    @Option(help: "OVERRIDE the video VAE.")
    var videoVae: String?

    @Option(help: "OVERRIDE the audio VAE.")
    var audioVae: String?

    @Option(help: "OVERRIDE the x2 latent spatial upsampler.")
    var upsampler: String?

    @Option(help: ArgumentHelp("Config file to read and, if absent, write. Defaults to "
                + "~/.config/ltx/config.json."))
    var config: String?

    @Flag(help: ArgumentHelp("Rewrite the config from the built-in defaults plus any "
                + "overrides given here, discarding what is there. Without this, an "
                + "existing config is read and left alone."))
    var writeConfig: Bool = false

    /// The recipe being checked against, or `nil` when the name is not registered.
    private var chosen: Recipe? { try? RecipeRegistry.recipe(recipe) }

    /// True when the chosen recipe's first stage runs the distilled transformer.
    private var wantsDistilled: Bool {
        chosen?.stages.first?.transformer == .distilled
    }

    private var ditRole: LTXConfiguration.Role { wantsDistilled ? .ditDistilled : .ditDev }

    func run() throws {
        var problems = 0
        func problem() { problems += 1 }

        // Held locally, not as a stored property: ArgumentParser synthesises `Decodable`
        // over every stored property, and `Paths` is not one of the argument types.
        var paths = printConfiguration(problem: problem)
        let ditURL = try paths.url(ditRole, override: checkpoint)

        let machine = Machine.detect()
        let available = Machine.availableBytes()

        // Reported rather than swallowed. An unrecognised name would otherwise leave
        // `chosen` nil and quietly disable the transformer cross-check, so the run would
        // look healthier for having been misconfigured.
        if chosen == nil {
            problem()
            print("configuration")
            print("  PROBLEM: '\(recipe)' is not a registered recipe, so the checkpoints")
            print("           below are not being checked against any pipeline's")
            print("           expectations. Known: \(RecipeRegistry.ids.joined(separator: ", "))")
            print("")
        }

        // ---- machine
        print("machine")
        print("  \(machine.summary)")
        print(String(format: "  %.0f GB available right now (free + inactive + speculative)",
                     Double(available) / 1e9))

        // ---- environment
        //
        // First, because it is the first thing that fails and the least self-explanatory
        // when it does: without the kernels every checkpoint below resolves perfectly and
        // the render dies on its opening GPU op with an untyped C++ error carrying no path.
        print("\nMetal kernels")
        if let lib = MetalLibrary.locate() {
            print("  \(lib.path)")
            if MetalLibrary.locatedOnlyViaWorkingDirectory() {
                problem()
                print("  WARNING: found only relative to the current directory, so this binary")
                print("           works from here and nowhere else. Run")
                print("           tools/build_mlx_metallib.sh to place mlx.metallib beside the")
                print("           executable instead.")
            }
        } else {
            problem()
            print("  MISSING — the first GPU operation will fail with an untyped C++ error.")
            print("  looked in:")
            for path in MetalLibrary.searchPaths { print("    \(path.path)") }
            print("  remedy: tools/build_mlx_metallib.sh")
        }

        printGEMMPatch(machine: machine, problem: problem)

        // ---- checkpoints
        let slots = [
            CheckpointInventory.Slot(role: "dit", url: ditURL, expected: .transformer,
                                     nameMustContain: expectedTransformerFragment()),
            CheckpointInventory.Slot(role: "text encoder",
                                     url: try paths.url(.textEncoder, override: textEncoder),
                                     expected: .textEncoder),
            CheckpointInventory.Slot(role: "video vae",
                                     url: try paths.url(.videoVAE, override: videoVae),
                                     expected: .videoVAECausal),
            CheckpointInventory.Slot(role: "audio vae",
                                     url: try paths.url(.audioVAE, override: audioVae),
                                     expected: .audioVAE),
            CheckpointInventory.Slot(role: "upsampler",
                                     url: try paths.url(.upsampler, override: upsampler),
                                     expected: .latentUpscaler),
        ]
        let rows = CheckpointInventory.rows(slots)
        print("\ncheckpoints")
        for row in rows {
            switch row.result {
            case let .success(file):
                let size = fileSizeGB(file.url)
                print("  \(pad(row.role, 15))\(pad(file.name, 54))"
                    + String(format: "%6.1f GB  %d tensors", size, file.tensorCount))
            case let .failure(mismatch):
                problem()
                print("  \(pad(row.role, 15))PROBLEM")
                print("      \(mismatch)")
            }
        }
        if let disagreement = CheckpointInventory.versionDisagreement(rows) {
            problem()
            print("  PROBLEM: files declare different model versions — "
                + disagreement.sorted { $0.key < $1.key }
                    .map { "\($0.key)=\($0.value)" }.joined(separator: ", "))
        }
        // Said on every run, not only when something is wrong: it is a limit of the data,
        // and a limit nobody is told about is one they will assume away.
        print("  NOTE: dev and distilled transformers are byte-identical in structure —")
        print("        same header length, 4349 tensors, same metadata including")
        print("        model_version. Which one is loaded is known from the FILENAME only.")

        // ---- adapters
        //
        // Every configured root, not just the transformer's own directory. Two model
        // generations coexist here: the 2.3 IC-LoRA set is a sibling of the 2.5 tree, and a
        // scan rooted at the DiT cannot see it — which is how an adapter sitting on disk
        // reads as absent.
        var roots = [AdapterCatalog.modelRoot(containing: ditURL)]
        for extra in paths.config.checkpoints.adapterRoots {
            if let url = paths.config.resolve(extra) { roots.append(url) }
        }
        var adapters: [CheckpointIdentity.File] = []
        print("\nadapters")
        for root in roots {
            guard FileManager.default.fileExists(atPath: root.path) else {
                print("  \(root.path) — not present, skipped")
                continue
            }
            let found = (try? AdapterCatalog.discover(in: root)) ?? []
            adapters.append(contentsOf: found)
            print("  \(root.path)")
            if found.isEmpty { print("    none found") }
            for adapter in found {
                // Version and reference factor, because both decide whether an adapter can
                // be used and neither is in the filename. A 2.3 adapter on a 2.5 base is a
                // vendor claim this repo has not measured; a reference factor is what places
                // the conditioning clip in the target's frame.
                let reference = try? ICLoRAReference.read(contentsOf: adapter.url)
                let version = adapter.modelVersion ?? "?"
                // Standard LoRA or IC-LoRA, said outright: they are applied the same way but
                // they are not the same thing, and only the second needs a reference clip.
                let kind: String
                if let reference, reference.isDeclared {
                    kind = "IC-LoRA ref x\(reference.downscaleFactor)"
                } else if reference != nil {
                    kind = "standard LoRA"
                } else {
                    kind = "UNREADABLE reference metadata"
                }
                print("    \(pad(adapter.name, 62))"
                    + String(format: "%6.2f GB  ", fileSizeGB(adapter.url))
                    + pad("v\(version)", 8) + pad(kind, 22)
                    + (version.hasPrefix("2.5") ? "" : "[not 2.5 — unmeasured here]"))
            }
        }

        // ---- recipes
        problems += printRecipes(rows: rows, adapters: adapters, roots: roots,
                                 available: available, machine: machine)

        // ---- memory
        printMemory(available: available,
                    margin: paths.config.policy.memoryMarginFraction)

        print("")
        print(problems == 0 ? "no problems found" : "\(problems) problem(s) above")
        if problems > 0 { throw ExitCode(1) }
    }

    // MARK: - Sections

    /// Read the config, writing one first when there is none.
    ///
    /// **Writing is the point.** Before this existed the six paths were `static let`s on a
    /// CLI struct: compiled in, invisible and uneditable. A first run
    /// on a new machine now leaves a file on disk that says exactly what the port expects to
    /// find and where, which turns "it cannot find my checkpoints" from a code question into
    /// a one-line edit.
    ///
    /// An existing config is **never** silently rewritten: it may carry edits this command
    /// knows nothing about, and a diagnostic that destroys the configuration it was run to
    /// diagnose is worse than no diagnostic. `--write-config` is the explicit opt-in.
    private func printConfiguration(problem: () -> Void) -> Paths {
        let target = config.map(URL.init(fileURLWithPath:)) ?? LTXConfiguration.defaultURL
        let existed = FileManager.default.fileExists(atPath: target.path)

        print("configuration")
        if !existed || writeConfig {
            // Seed from the built-ins, then fold in whatever this invocation named, so
            // `ltx doctor --text-encoder /elsewhere/enc.safetensors` writes a config that
            // already reflects the machine it was run on.
            var seed = LTXConfiguration.builtIn
            if let checkpoint {
                seed.checkpoints.dit[wantsDistilled ? "distilled" : "dev"] = checkpoint
            }
            if let textEncoder { seed.checkpoints.textEncoder = textEncoder }
            if let videoVae { seed.checkpoints.videoVAE = videoVae }
            if let audioVae { seed.checkpoints.audioVAE = audioVae }
            if let upsampler { seed.checkpoints.upsampler = upsampler }
            seed.policy.defaultRecipe = recipe

            do {
                let wrote = try seed.write(to: target, force: writeConfig)
                print("  \(wrote ? (existed ? "REWROTE" : "wrote") : "kept") \(target.path)")
                if wrote, !existed {
                    print("  Paths came from the built-in defaults. Edit `checkpoints.root`")
                    print("  in this file if your checkpoints are somewhere else.")
                }
            } catch {
                problem()
                print("  PROBLEM: could not write \(target.path): \(error)")
            }
        }

        var paths: Paths
        do {
            paths = try Paths(configPath: target.path)
            print("  \(paths.configURL?.path ?? "built-in defaults (no file)")")
            let policy = paths.config.policy
            print("  default recipe        \(policy.defaultRecipe ?? RecipeRegistry.defaultID)")
            print(String(format: "  memory margin         %.0f%%",
                         policy.memoryMarginFraction * 100))
            if let root = paths.config.checkpoints.root {
                print("  checkpoint root       \(root)")
            }
        } catch {
            problem()
            print("  PROBLEM: \(error)")
            // A parse failure must not silently fall back to defaults — that is how a
            // render ends up using a checkpoint nobody chose — but the remaining sections
            // are still worth printing, so carry on against the built-ins and say so.
            paths = (try? Paths(configPath: nil)) ?? Paths.builtIn
            print("  continuing against built-in defaults for the sections below")
        }
        print("")
        return paths
    }

    private func printGEMMPatch(machine: Machine, problem: () -> Void) {
        let patch = MetalLibrary.gemmPatch(packageRoot: packageRoot())
        print("\nMLX large-M GEMM patch")
        switch patch {
        case .applied:
            print("  applied")
        case let .missing(target):
            // Only a problem on the hardware it tunes for. Reporting it on a machine that
            // will never take the branch would be noise dressed as a finding.
            if MetalLibrary.isUltraClass(machine) {
                problem()
                print("  MISSING on an Ultra/Max-class chip, where it is worth 8.7-14.6% on")
                print("  the feed-forward GEMMs above M=8192. This failure has NO other")
                print("  symptom: the build succeeds, the tests pass, the numbers stay")
                print("  bit-identical, and the binary is permanently slower on long clips.")
                print("  remedy: ./tools/check-mlx-patch.sh --apply")
                print("  target: \(target)")
            } else {
                print("  not applied; not needed on this chip (Ultra/Max tuning)")
            }
        case let .notCheckedOut(looked):
            print("  unknown — no mlx-swift checkout to inspect")
            print("  looked: \(looked)")
        }
    }

    /// One row per recipe: whether it can run **here**, and what stands behind the result.
    ///
    /// This is the part that answers "will this work" rather than "is this file present".
    /// It resolves each recipe exactly as `render` does, checks the components it needs
    /// against what was actually found, resolves its adapter hints against the tree on this
    /// machine, and plans memory at the shape the recipe itself lands on.
    /// Resolve an adapter hint the way the render commands do: across **every** configured
    /// root, first match wins.
    ///
    /// Rooting this at the transformer's own directory is what made `ingredients` report
    /// BLOCKED on a machine where it runs — the 2.3 IC-LoRA set is a sibling of the 2.5
    /// tree, exactly as the adapters section above says. A doctor that invents a problem is
    /// worse than one that misses it: this is the command people run to decide whether
    /// something is worth debugging.
    private func resolveAdapter(hint: String, roots: [URL]) throws -> CheckpointIdentity.File {
        var last: Error?
        for root in roots {
            do { return try AdapterCatalog.resolve(hint: hint, in: root) }
            catch { last = error }
        }
        throw last ?? AdapterCatalog.Failure.noMatch(hint: hint, root: "(no roots)",
                                                    available: [])
    }

    private func printRecipes(rows: [CheckpointInventory.Row],
                              adapters: [CheckpointIdentity.File], roots: [URL],
                              available: UInt64, machine: Machine) -> Int {
        var problems = 0
        // Adapters are components too. Building the capability set from the five named
        // slots alone reported "needs the lora checkpoint, which was not found" on the very
        // line above one that had just resolved the adapter successfully.
        var present = CheckpointInventory.components(rows)
        for adapter in adapters { present.formUnion(adapter.components) }
        let capability = Capability(unifiedMemoryBytes: available, components: present)

        print("\nrecipes")
        print("  \(pad("id", 18))\(pad("run with", 16))\(pad("shape", 12))status")
        for recipe in RecipeRegistry.all {
            var notes: [String] = []
            var blocked = false

            // Resolve at the recipe's OWN measured configuration, not at a default
            // request. A bare request defaults to 4 seconds, which snaps to 89 frames and
            // misses the 97-frame match — so every row read `sibling` and the menu
            // understated what this port has evidence for.
            let gate = recipe.gates.first
            let request = LTX25.Request(
                prompt: "", videoOutput: URL(fileURLWithPath: "/dev/null"),
                seconds: gate.map { Double($0.frames) / $0.frameRate } ?? 97.0 / 24.0,
                frames: gate?.frames,
                frameRate: gate?.frameRate ?? 24)
            let resolved = try? recipe.resolve(request)
            let evidence = resolved?.evidence
                ?? recipe.evidence(output: recipe.gates.first?.shape
                                       ?? Shape(width: 640, height: 384),
                                   frames: 97, frameRate: 24, steps: recipe.declaredSteps)
            for reason in capability.reasonsCannotRun(recipe.id, evidence: evidence,
                                                      requires: recipe.requires) {
                notes.append("\(reason)")
                blocked = true
            }

            // Adapter hints, resolved against this machine's tree. Deduplicated: a
            // two-stage recipe names the same adapter twice and would otherwise print the
            // same line — or the same failure — once per stage.
            var seenHints = Set<String>()
            for adapter in recipe.stages.flatMap({ $0.adapters })
            where seenHints.insert(adapter.filenameHint).inserted {
                do {
                    let file = try resolveAdapter(hint: adapter.filenameHint, roots: roots)
                    notes.append("adapter '\(adapter.filenameHint)' -> \(file.name) "
                        + String(format: "(%.2f GB resident, strength %g)",
                                 fileSizeGB(file.url), adapter.strength))
                } catch {
                    notes.append("adapter: \(error)")
                    blocked = true
                }
            }

            // Memory, at the shape this recipe actually resolves to — and at the length
            // the transformer actually attends over.
            //
            // `videoTokens()` is the *generated* count, which is what sizes the noise draw
            // and the decode. A reference-conditioned recipe carries more than that: its
            // reference tokens ride in the same sequence, so at five MSR slots the real
            // peak is 3120 + 6000 rather than 3120. Planning on the generated count alone
            // would clear a shape that then runs out of memory, which is the one answer
            // this table must never give.
            var fits = true
            if let resolved, var peakTokens = resolved.videoTokens().max() {
                if let references = recipe.contextReferences,
                   let firstStage = try? recipe.stages.first?.shape(forOutput: resolved.output)
                       ?? resolved.output {
                    let latent = LatentGeometry()
                    peakTokens += references.tokens(
                        latentHeight: firstStage.height / latent.spatialScale,
                        latentWidth: firstStage.width / latent.spatialScale).upperBound
                }
                let adapterBytes = Set(recipe.stages.flatMap { $0.adapters }
                    .map(\.filenameHint))
                    .compactMap { hint -> UInt64? in
                        guard let file = try? resolveAdapter(hint: hint, roots: roots)
                        else { return nil }
                        return fileSizeBytes(file.url)
                    }
                    .reduce(0, +)
                let plan = MemoryPlan.plan(videoTokens: peakTokens,
                                           availableBytes: available &- min(available, adapterBytes))
                fits = plan.fits
                if !fits {
                    notes.append(String(format: "does not fit: peak %.1f GB, short by %.1f GB",
                                        plan.peakGB, -Double(plan.headroomBytes) / 1e9))
                    blocked = true
                }
            }

            let shape = resolved?.output.description
                ?? (recipe.kind == .transform ? "from input" : "—")
            let status = blocked ? "BLOCKED" : (recipe.kind == .transform
                                                ? "ok" : evidence.label)
            print("  \(pad(recipe.id, 18))\(pad("ltx " + recipe.command, 16))"
                + "\(pad(shape, 12))\(status)")
            for note in notes { print("      \(note)") }
            if blocked { problems += 1 }
        }
        return problems
    }

    private func printMemory(available: UInt64, margin: Double) {
        print("\nmemory plan at \(tokens) video tokens")
        let plan = MemoryPlan.plan(videoTokens: tokens, availableBytes: available,
                                   marginFraction: margin)
        print(plan.explanation)
        // The planner is fitted to the single-stage fused path. Measured peaks on the
        // two-stage route came in below it, so it errs toward refusing renders that would
        // have fit rather than admitting ones that would not — worth saying, because a
        // conservative planner reads like a broken one when the render then succeeds.
        print("  NOTE: coefficients are fitted to the SINGLE-STAGE fused path. A measured")
        print("        two-stage render at 3120 tokens peaked at 41.0 GB against this")
        print("        plan's estimate, so the plan is conservative there, not wrong.")
        print("  NOTE: adapter weights are NOT in the phase figures. The rank-450 distilled")
        print("        LoRA measured +6.9 GB resident; the recipes section above subtracts")
        print("        each recipe's adapters from available before planning.")
    }

    // MARK: - Helpers

    /// Which transformer the chosen recipe expects, as a filename fragment.
    ///
    /// Applied **always**, including to an explicitly named `--checkpoint` — that is the
    /// case worth catching, because pointing `--checkpoint` at the distilled file while
    /// running a recipe built for dev is precisely the mistake that produces a render rather
    /// than an error. `--recipe` is how a caller says which pairing they meant.
    ///
    /// The two filenames discriminate cleanly: neither contains the other's fragment. A
    /// renamed file defeats it, which is a limitation of the only signal that exists —
    /// see the note printed with the checkpoint table.
    private func expectedTransformerFragment() -> String? {
        guard chosen != nil else { return nil }
        return wantsDistilled ? "distilled" : "dev"
    }

    private func packageRoot() -> URL {
        // The binary sits in .build/<config>/ or .build/<triple>/<config>/. Walk up until a
        // Package.swift appears, so this works from either and from an installed copy
        // (where it simply finds nothing and reports the patch as unknown).
        var dir = MetalLibrary.binaryDirectory
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0 ..< 6 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("Package.swift").path) { return dir }
            dir = dir.deletingLastPathComponent()
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private func fileSizeBytes(_ url: URL) -> UInt64 {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64) ?? 0
    }

    private func fileSizeGB(_ url: URL) -> Double { Double(fileSizeBytes(url)) / 1e9 }

    /// Columns padded in Swift: Darwin's Foundation ignores width flags for `%@`, so a
    /// `%-15@` does not pad and the table collapses into a ragged list.
    private func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
}

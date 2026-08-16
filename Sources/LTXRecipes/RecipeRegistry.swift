// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation

/// A named pipeline, and the evidence behind it.
///
/// ## Why the registry exists
///
/// Until now the pipeline was chosen by `RecipeRequest.twoStage`, a `Bool`. That boolean is
/// a pipeline selector wearing a disguise, and it is why
/// ``RecipeRequest/validate()`` needs forty-five lines of "these knobs are silently ignored
/// under the other pipeline": one request type was describing two pipelines, so every field
/// that belonged to one of them had to be refused by hand under the other. A recipe owns
/// the pipeline — stages, schedules, sampler, guidance, transformer role, adapters — and the
/// request keeps only intent: prompt, duration, seed, conditioning, output. Most of that
/// refusal then becomes structural rather than hand-written, because a recipe that
/// constructs no guider has no guidance fields to ignore.
///
/// ## What is registered here
///
/// **Only routes that run today.** Every entry below was traced end to end through this
/// repository before it was written down, because a menu that lists a pipeline the port
/// cannot execute is worse than no menu: it converts a five-second refusal into a discovery
/// made twenty minutes into a render.
///
/// That rule cuts both ways, and it did. Ingredients was listed here as deliberately absent
/// on the grounds that the 2.3 adapters "have no 2.5 release and no route here" — true of
/// the release, no longer true of the route. `ltx ingredients` runs it end to end, so the
/// absence had outlived its reason and the entry is now present with its 2.3 provenance in
/// its own summary, where a caller reads it. Union Control, Motion Track and In-Outpainting
/// remain absent because they still have no route, which is the actual criterion.
///
/// **A recipe is not the same thing as `ltx render --recipe`.** The registry is the menu of
/// *how a pipeline is conditioned and run*, and three of its entries are executed by other
/// commands — see ``Recipe/command``. Keeping a route out of the registry because `render`
/// cannot run it would hide the conditioning choice in the one place a caller goes to
/// compare conditioning choices.
///
/// ## Evidence is derived, not declared
///
/// A recipe does not carry an ``Evidence`` value. It carries its measured ``Gate``s, and
/// ``evidence(output:frames:frameRate:steps:)`` decides what a *particular* resolution is
/// entitled to claim. A measurement at one shape is not a measurement at another, and the
/// menu has to say so rather than letting a resolution inherit a status from the pipeline
/// it shares. Render the measured pipeline at 1024x576
/// and the answer is `sibling`, naming the shape it is a sibling of and what differs.
public struct Recipe: Sendable {

    /// Whether a recipe generates a clip or transforms one that already exists.
    ///
    /// The split is not cosmetic: the two share no request. A transform has no prompt to
    /// resolve a schedule against, no seed in the generative sense, and its output geometry
    /// is fixed by its input rather than chosen — which is exactly why `VideoUpscaler` is a
    /// separate command and not a flag on `render`. They live in one registry anyway,
    /// because a menu split across two commands is a menu somebody reads half of.
    public enum Kind: String, Sendable, CaseIterable {
        case generative
        case transform
    }

    /// One exact configuration this recipe has been measured at.
    ///
    /// Every field participates in the match. A measurement at 97 frames says nothing about
    /// 121, and one at 30 steps says nothing about 8 — so a `Gate` that matched on shape
    /// alone would launder an unmeasured render into a measured one, which is the single
    /// failure this type exists to prevent.
    public struct Gate: Sendable, Equatable {
        public let shape: Shape
        public let frames: Int
        public let frameRate: Double
        public let steps: Int
        /// The file the measurement is read from, so a claim can be checked rather than
        /// trusted.
        public let source: String

        public init(shape: Shape, frames: Int, frameRate: Double, steps: Int, source: String) {
            self.shape = shape
            self.frames = frames
            self.frameRate = frameRate
            self.steps = steps
            self.source = source
        }
    }

    public let id: String
    /// One line for the menu. Says what the recipe *is*, not what it is good for.
    public let summary: String
    public let kind: Kind
    /// The subcommand that executes this recipe.
    ///
    /// Separate from ``Kind`` because the two answer different questions and used to be
    /// conflated. `render` guarded on `kind == .generative` and told everything else to run
    /// through `ltx upscale`, which was true while the only non-render recipes were
    /// transforms. It stops being true for a reference-conditioned generative route:
    /// `ingredients` and `msr` generate from a prompt at a chosen size — every part of the
    /// generative definition — and are still not things `ltx render` can run, because the
    /// sequence the transformer sees is longer than the one that gets decoded and `render`
    /// sizes its noise, masks and unpatchify from a single token count.
    ///
    /// Left as data rather than derived, so registering a route that `render` cannot execute
    /// is a field the author fills in rather than a guard somebody remembers to extend.
    public let command: String
    public let stages: [RecipeStage]
    public let grid: Grid
    /// Checkpoints this recipe opens. Fed to ``Capability/reasonsCannotRun(_:evidence:requires:)``.
    public let requires: Set<LTXComponent>
    public let gates: [Gate]
    /// `nil` means **no measured ceiling**, not "unlimited".
    ///
    /// The `8...1001` range this port inherited came from a 2.3-era coherence
    /// investigation, and no frame-count ceiling has been established for 2.5 here.
    /// Carrying that number forward would dress a guess as a limit. The real ceiling on
    /// this machine is the memory planner, which measures.
    public let frameLimits: ClosedRange<Int>?
    /// Non-nil registers the recipe as unrunnable and carries the reason into every refusal.
    public let unroutable: String?

    /// Reference clips this recipe carries **in context**, beside the generated tokens.
    ///
    /// Declared because the menu's token column is otherwise a lie for these routes. An
    /// in-context reference makes the sequence the transformer attends over longer than the
    /// one that gets decoded — at five MSR slots on a short target, several times longer —
    /// and a caller reading "384 tokens" to budget memory would be off by a factor of three.
    /// `RecipeCatalogue` prints the reference cost beside the generated count.
    public struct ContextReferences: Sendable, Equatable {
        public let minimum: Int
        public let maximum: Int
        /// Latent frames each reference occupies. 1 for a single still; 5 for a still
        /// repeated onto 33 pixel frames through the causal VAE's `8k + 1`.
        public let latentFramesEach: Int

        public init(minimum: Int, maximum: Int, latentFramesEach: Int) {
            self.minimum = minimum
            self.maximum = maximum
            self.latentFramesEach = latentFramesEach
        }

        /// Tokens added to the video stream at a given latent grid, per the maximum slots.
        public func tokens(latentHeight: Int, latentWidth: Int) -> ClosedRange<Int> {
            let each = latentFramesEach * latentHeight * latentWidth
            return (each * minimum) ... (each * maximum)
        }
    }

    public let contextReferences: ContextReferences?

    public init(id: String, summary: String, kind: Kind = .generative,
                command: String = "render",
                stages: [RecipeStage], grid: Grid = .single,
                requires: Set<LTXComponent>, gates: [Gate] = [],
                frameLimits: ClosedRange<Int>? = nil, unroutable: String? = nil,
                contextReferences: ContextReferences? = nil) {
        self.id = id
        self.summary = summary
        self.kind = kind
        self.command = command
        self.stages = stages
        self.grid = grid
        self.requires = requires
        self.gates = gates
        self.frameLimits = frameLimits
        self.unroutable = unroutable
        self.contextReferences = contextReferences
    }

    // MARK: - Schedule

    /// True when every stage's schedule is a literal table, so the step count is not a knob.
    ///
    /// Both distilled stages are literal sigma tables. Accepting a `--steps` against them
    /// and dropping it is the mistake ``RecipeRequest/validate()`` already refuses by name
    /// for the two-stage case; derived here, it covers every fixed-schedule recipe without
    /// anyone remembering to add one.
    public var scheduleIsFixed: Bool {
        stages.allSatisfy { if case .fixed = $0.sigmas { return true } else { return false } }
    }

    /// The recipe's own step count, summed across stages.
    public var declaredSteps: Int { stages.reduce(0) { $0 + $1.sigmas.steps } }

    // MARK: - Evidence

    /// What *this* resolution is entitled to claim.
    ///
    /// Ordering matters. `noRoute` outranks everything, because
    /// reporting an unroutable recipe as "unmeasured" is true and misleading — it implies a
    /// measurement would fix it, when nothing about the pipeline is reachable at any shape.
    public func evidence(output: Shape, frames: Int, frameRate: Double, steps: Int) -> Evidence {
        if let unroutable { return .noRoute(unroutable) }

        if let hit = gates.first(where: {
            $0.shape == output && $0.frames == frames
                && $0.frameRate == frameRate && $0.steps == steps
        }) {
            return .gated(shape: hit.shape, frames: hit.frames, steps: hit.steps,
                          source: hit.source)
        }

        guard let nearest = nearestGate(to: output) else {
            return .unmeasured("'\(id)' has not been measured at any shape")
        }

        // Name every axis that differs. "sibling of 640x384" alone invites the reading that
        // only the shape moved, which is wrong the moment a duration or a step count did.
        var differs: [String] = []
        if nearest.shape != output {
            differs.append("shape \(output.description) vs \(nearest.shape.description)")
        }
        if nearest.frames != frames { differs.append("\(frames) frames vs \(nearest.frames)") }
        if nearest.frameRate != frameRate {
            differs.append("\(frameRate) fps vs \(nearest.frameRate)")
        }
        if nearest.steps != steps { differs.append("\(steps) steps vs \(nearest.steps)") }

        return .sibling(of: nearest.shape,
                        note: "same pipeline, measured at \(nearest.source); differs by "
                            + differs.joined(separator: ", "))
    }

    /// The gate closest in pixel count. Only used to name what a sibling is a sibling *of*.
    private func nearestGate(to output: Shape) -> Gate? {
        gates.min { abs($0.shape.pixels - output.pixels) < abs($1.shape.pixels - output.pixels) }
    }

    // MARK: - Resolution

    /// A request and this recipe, reconciled. Every shape decided, nothing loaded.
    ///
    /// The order is deliberate: the refusals that need no arithmetic come first, so a
    /// caller who named an unroutable recipe is not first told about their frame count.
    public func resolve(_ request: RecipeRequest,
                        lattice: FrameLattice = FrameLattice()) throws -> ResolvedRecipe {
        if let unroutable {
            throw RecipeFailure.noRoute(recipe: id, reason: unroutable)
        }
        guard kind == .generative else {
            throw RecipeFailure.notGenerative(
                recipe: id,
                detail: "it transforms a clip that already exists, so it has no prompt, no "
                    + "seed and no output geometry to choose — the input supplies all three. "
                    + "Run it through `ltx upscale`")
        }
        try request.validate()

        if scheduleIsFixed, request.steps != nil {
            throw RecipeFailure.scheduleIsFixed(
                recipe: id,
                detail: "--steps: every stage's schedule is a fixed table, "
                    + "so the count is \(stages.map { String($0.sigmas.steps) }.joined(separator: " + "))")
        }

        let frames = try request.resolvedFrames(lattice: lattice)
        if let frameLimits, !frameLimits.contains(frames) {
            throw RecipeFailure.framesOutOfRange(frames, allowed: frameLimits)
        }

        let output = try resolvedOutput(request)
        let staged = try stages.map { (stage: $0, shape: try $0.shape(forOutput: output)) }
        let steps = request.steps ?? declaredSteps

        return ResolvedRecipe(
            recipeID: id, output: output, stages: staged,
            frames: frames, frameRate: request.frameRate,
            evidence: evidence(output: output, frames: frames,
                               frameRate: request.frameRate, steps: steps))
    }

    /// The output shape, with the recipe's own measured shape as the default.
    ///
    /// A request that names no size takes this recipe's **measured** shape rather than a rung
    /// the ladder derived, because a derived rung near 0.25 MP has nothing behind it while
    /// the measured shape does. An explicit size or a megapixel target overrides it and is
    /// resolved by ``RecipeRequest/resolvedShape(grid:)`` — which validates rather than
    /// snaps, so the caller is told when their number is off the grid instead of being moved
    /// onto it.
    ///
    /// **The measured shape only wins when its orientation matches the request's.** Every
    /// measurement here is landscape, and taking `gates.first` unconditionally handed a
    /// caller who asked for 9:16 a 640x384 landscape clip — the request silently ignored, the
    /// output plausible. Falling through to ``RecipeRequest/resolvedShape(grid:)`` picks up
    /// ``Ladder/pinned``, which carries that shape's transpose for exactly this case.
    private func resolvedOutput(_ request: RecipeRequest) throws -> Shape {
        if request.width == nil, request.height == nil, request.megapixels == nil,
           let gate = gates.first(where: {
               Self.orientation(of: $0.shape.aspect) == Self.orientation(of: request.aspectRatio.ratio)
           }) {
            return gate.shape
        }
        return try request.resolvedShape(grid: grid)
    }

    /// Landscape, portrait or square. Compared rather than the ratio itself because the
    /// pinned shape is 1.667 and 16:9 is 1.778 — they are the same *request*, and an equality
    /// test on the ratio would never match.
    private static func orientation(of aspect: Double) -> Int {
        if aspect > 1.0 { return 1 }
        if aspect < 1.0 { return -1 }
        return 0
    }
}

// MARK: - The registry

/// Every pipeline this port can actually run, by name.
///
/// Five entries. Two generate, three transform, and each one was traced through to the
/// executing code before being registered — see the per-entry notes for which type does the
/// work. Nothing aspirational is listed.
public enum RecipeRegistry {

    /// The default when no recipe is named. Preserves what `ltx render` does today.
    public static let defaultID = "prod"

    public static var all: [Recipe] {
        [production, devLoRA, distilled, ingredients, msr,
         upscalePlain, upscaleRefined, upscaleICLoRA]
    }

    /// Recipes `ltx render` can execute. The rest name their own command.
    public static func renderable() -> [Recipe] { all.filter { $0.command == "render" } }

    public static var ids: [String] { all.map(\.id) }

    public static func recipe(_ id: String) throws -> Recipe {
        guard let hit = all.first(where: { $0.id == id }) else {
            throw RecipeFailure.unknownRecipe(id)
        }
        return hit
    }

    public static func generative() -> [Recipe] { all.filter { $0.kind == .generative } }

    // MARK: Generative

    /// The ladder surface: one stage on the dev transformer with production guidance.
    ///
    /// This is the raw-render route. Shape comes from a megapixel target and an aspect
    /// ratio through ``Ladder``, or from the measured shape when the caller names neither.
    /// Runs through `LTXPipeline.Renderer`.
    ///
    /// **Two measurements, one pipeline.** The same arithmetic was measured at two shapes,
    /// so they are two ``Gate``s on one recipe rather than two recipes — which is what lets a
    /// render at 320x192x25 report `gated` instead of inheriting the production shape's
    /// status.
    ///
    /// Four passes per step at these settings (cfg 3.0, STG 1.0, modality 3.0), measured
    /// at 4 x 7.29 s at 3120 tokens.
    public static let production = Recipe(
        id: "prod",
        summary: "single-stage on the dev transformer, production guidance — the raw-render ladder",
        stages: [
            RecipeStage(name: "sample",
                        sigmas: .continuous(steps: 30),
                        sampler: .euler,
                        videoGuidance: .productionVideo,
                        audioGuidance: .productionAudio,
                        transformer: .dev)
        ],
        requires: [.transformer, .textEncoder, .videoVAECausal, .audioVAE],
        gates: [
            Recipe.Gate(shape: Shape(width: 640, height: 384), frames: 97, frameRate: 24, steps: 30,
                 source: "Tests/Fixtures/reference/l05_drift_ltx25_prod.json"),
            Recipe.Gate(shape: Shape(width: 320, height: 192), frames: 25, frameRate: 24, steps: 30,
                 source: "Tests/Fixtures/reference/l05_drift_ltx25_tiny.json"),
        ])

    /// The dev transformer with the distilled LoRA applied as an overlay.
    ///
    /// The route the 2.3 lineage took before 2.5 shipped a distilled checkpoint: base dev
    /// weights plus `ltx-2.5-22b-distilled-lora-450-bf16`, rank 450. Registered because the
    /// adapter is on disk and the overlay is wired.
    ///
    /// **Not measured, and the gap is specific.** What has been checked is that the adapter
    /// is applied to the right tensors at the first step — not the trajectory that consumes
    /// it, and not the picture. Nothing here says this route is better or worse than `prod`.
    ///
    /// **It costs 8.29 GB of resident memory** that `MemoryPlan` does not model: the overlay
    /// deliberately never bakes, so the rank-450 file stays alongside the DiT's 42.06 GB for
    /// the whole render.
    ///
    /// Strength 1.0 and the full step schedule are this port's choices, not a recorded
    /// recipe. A 2.3-era strength of 0.6 exists for the equivalent adapter, but it was
    /// fitted against different weights and so does not transfer — see
    /// `docs/FRAGILE_CONTRACTS.md` contract 14 — which is why it is not used here.
    public static let devLoRA = Recipe(
        id: "dev-lora",
        summary: "dev transformer + the rank-450 distilled LoRA, overlaid",
        stages: [
            RecipeStage(name: "sample",
                        sigmas: .continuous(steps: 30),
                        sampler: .euler,
                        videoGuidance: .productionVideo,
                        audioGuidance: .productionAudio,
                        transformer: .dev,
                        adapters: [
                            AdapterRequest(filenameHint: "distilled-lora-450", strength: 1.0)
                        ]),
        ],
        requires: [.transformer, .textEncoder, .videoVAECausal, .audioVAE, .lora])

    /// The two-stage distilled route: draft at half size, latent x2 upsample, refine.
    ///
    /// Runs through `LTXPipeline.DistilledRenderer`. **16-17x faster than single-stage**,
    /// measured — the largest single lever there is, and the reason this is a first-class
    /// entry rather than a variant.
    ///
    /// No guidance anywhere: this route constructs no guider at all, so both stages are one
    /// forward per step — 8 + 3 = 11 in total.
    ///
    /// Stage 1 is ancestral (`eta = 1, s_noise = 1`) and stage 2 is deterministic Euler.
    /// `Sampler.ancestralStep` carries the warning that matters here: the more widely known
    /// ancestral-step formula disagrees with this one everywhere except `eta = 0`, which is
    /// precisely not the eta this pipeline uses.
    public static let distilled = Recipe(
        id: "distilled",
        summary: "two-stage distilled, 8 + 3 steps, no guidance — 16-17x faster than prod",
        stages: [
            RecipeStage(name: "draft", scale: 0.5,
                        sigmas: .distilledStage1,
                        sampler: .eulerAncestral,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .distilled,
                        decodes: false),
            RecipeStage(name: "refine", scale: 1.0,
                        sigmas: .distilledStage2,
                        sampler: .euler,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .distilled),
        ],
        grid: .twoStage,
        requires: [.transformer, .textEncoder, .videoVAECausal, .audioVAE, .latentUpscaler],
        gates: [
            Recipe.Gate(shape: Shape(width: 640, height: 384), frames: 97, frameRate: 24, steps: 11,
                 source: "640x384x97 @ 24 fps, 8 + 3 steps"),
        ])

    // MARK: Generative, reference-conditioned

    /// A prompt plus a **reference sheet** — one composite image of the cast, props and
    /// location — through the Ingredients IC-LoRA. Runs through `IngredientsRenderer`.
    ///
    /// Generative by every part of the definition: a prompt, a seed, an output size the
    /// caller chooses. It is not a `render` route because the reference's tokens ride in the
    /// video sequence beside the generated ones, held clean at timestep 0 and dropped before
    /// the decode — a longer sequence than the one `render` sizes its noise and masks from.
    ///
    /// **A 2.3 adapter on a 2.5 base.** All 480 of its modules resolve with matching shapes,
    /// which is a statement about key sets and not about whether its features transfer.
    /// `reference_downscale_factor` is 1, so the sheet is encoded at the target's own
    /// resolution rather than a fraction of it.
    ///
    /// Two stages: stage 1 at half the output, then a x2 latent upsample and a refine that
    /// carries **the same conditioning** — not `upscale-refined`, which attaches no adapter,
    /// appends no reference, and melts terrain. `--upsample` selects it; without the flag
    /// only stage 1 runs, at the requested size.
    public static let ingredients = Recipe(
        id: "ingredients",
        summary: "prompt + a reference sheet through the Ingredients IC-LoRA — a 2.3 adapter "
            + "on a 2.5 base, an experiment rather than a supported path",
        command: "ingredients",
        stages: [
            RecipeStage(name: "reference-guided", scale: 0.5,
                        sigmas: .distilledStage1,
                        sampler: .eulerAncestral,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .distilled,
                        adapters: [
                            AdapterRequest(filenameHint: "ic-lora-ingredients", strength: 1.0)
                        ],
                        decodes: false),
            RecipeStage(name: "refine", scale: 1.0,
                        sigmas: .distilledStage2,
                        sampler: .eulerAncestral,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .distilled,
                        adapters: [
                            AdapterRequest(filenameHint: "ic-lora-ingredients", strength: 1.0)
                        ]),
        ],
        grid: .twoStage,
        requires: [.transformer, .textEncoder, .videoVAECausal, .audioVAE, .lora,
                   .latentUpscaler],
        // One sheet, one latent frame: a still encodes to a single latent frame through the
        // causal VAE, and `IngredientsRenderer` refuses a reference that does not.
        contextReferences: Recipe.ContextReferences(minimum: 1, maximum: 1, latentFramesEach: 1))

    /// A prompt plus **a cast** — up to five reference stills, each in its own slot, through
    /// the Multiple-Subject-Reference IC-LoRA. Runs through `MSRRenderer`.
    ///
    /// The route that answers "several subjects, each staying itself", which no
    /// single-reference adapter can do. What separates the slots is not the prompt: a
    /// learned slot vector is added to each reference's latent channels, and each slot takes
    /// a distinct **negative** time offset, `-(n - i)` pixel frames, so no two references
    /// share coordinates with each other or with the target's own frame 0. Contracts 18 and
    /// 19 in `docs/FRAGILE_CONTRACTS.md` carry both, and the reason neither has a symptom of
    /// its own when broken.
    ///
    /// **A genuine 2.5 adapter**, unlike Ingredients — 960 LoRA tensors across all 48 blocks
    /// plus the 5 slot-embedding tensors, video stream only.
    ///
    /// Single stage. This route has no second stage and no upsample; adding one would be an
    /// invention, and `upscale-ic-lora` is the registered route for enlarging what it
    /// produces.
    public static let msr = Recipe(
        id: "msr",
        summary: "prompt + up to five reference stills, each held apart by a learned slot "
            + "vector and its own negative time offset",
        command: "msr",
        stages: [
            RecipeStage(name: "reference-guided",
                        sigmas: .distilledStage1,
                        sampler: .eulerAncestral,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .distilled,
                        adapters: [
                            AdapterRequest(filenameHint: "licon-msr", strength: 1.0)
                        ]),
        ],
        requires: [.transformer, .textEncoder, .videoVAECausal, .audioVAE, .lora],
        // Each still is repeated onto 33 pixel frames and encodes to 5 latent frames. At
        // five slots that is 25 latent frames of reference against a target that may have
        // fewer — which is exactly why the menu prints it.
        contextReferences: Recipe.ContextReferences(minimum: 1, maximum: 5, latentFramesEach: 5))

    // MARK: Transform

    /// Encode, x2 latent upsample, decode. No transformer and no text encoder.
    ///
    /// The honest floor: whatever the learned x2 operator does on its own, at 3.4 GB of
    /// checkpoint and no sampling at all. Registered because it is the baseline the other
    /// two upscale routes have to beat, and a menu that hides the baseline is a menu that
    /// cannot be used to choose.
    public static let upscalePlain = Recipe(
        id: "upscale-plain",
        summary: "encode, x2 latent upsample, decode — no transformer, no sampling",
        kind: .transform,
        command: "upscale",
        stages: [],
        grid: .twoStage,
        requires: [.videoVAECausal, .latentUpscaler])

    /// Plain upscale plus the distilled recipe's documented second stage.
    ///
    /// Three deterministic Euler steps on `[0.909375, 0.725, 0.421875, 0.0]`, no guider, on
    /// the distilled checkpoint — the distilled recipe's own second-stage schedule, not an
    /// invented one. Costs the 42 GB DiT and a text-encoder pass for three forwards.
    ///
    /// Read `sigmas[0] = 0.909375` before using it: stage 2 does not merely clean the
    /// upsample, it **re-noises to 91% and re-denoises**. On a generated draft that is how
    /// the refine continues the draft; on a finished clip it is also licence to move away
    /// from it.
    public static let upscaleRefined = Recipe(
        id: "upscale-refined",
        summary: "x2 upsample then the distilled stage-2 schedule, 3 steps — re-noises to 91%",
        kind: .transform,
        command: "upscale",
        stages: [
            RecipeStage(name: "refine",
                        sigmas: .distilledStage2,
                        sampler: .euler,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .distilled),
        ],
        grid: .twoStage,
        requires: [.transformer, .textEncoder, .videoVAECausal, .latentUpscaler])

    /// The x2 pixel spatial upscaler IC-LoRA — **the only IC-LoRA released for 2.5**.
    ///
    /// A reference-guided route, not a resample: the low-resolution clip is carried
    /// in-context as *clean* tokens beside the generated ones, and the picture is generated
    /// as though unconditioned. `DiTForward.Geometry.Reference` models it and
    /// `VideoUpscaler.referenceGuided` runs it, with the adapter reaching 960 tensors across
    /// all 48 blocks, video stream only.
    ///
    /// Default 3 steps rather than the recorded 8: fewer steps keep the output closer to the
    /// reference, which is the model card's own fidelity dial. The thinning comes out of the
    /// schedule's near-stationary head, so the working sigmas survive down to 3.
    ///
    /// Every other IC-LoRA in the LTX catalogue — Union Control, Motion Track, Ingredients,
    /// In-Outpainting, Clean Plate, Colorization, Deblur, Decompression — is released
    /// against **LTX-2.3-22b** and has no 2.5 build. They are absent rather than registered.
    public static let upscaleICLoRA = Recipe(
        id: "upscale-ic-lora",
        summary: "x2 pixel spatial upscaler IC-LoRA — the only IC-LoRA released for 2.5",
        kind: .transform,
        command: "upscale",
        stages: [
            RecipeStage(name: "reference-guided",
                        sigmas: .distilledStage1,
                        sampler: .eulerAncestral,
                        videoGuidance: .distilled,
                        audioGuidance: .distilled,
                        transformer: .dev,
                        adapters: [
                            AdapterRequest(
                                filenameHint: "ic-lora-pixel-spatial-upscaler",
                                strength: 1.0)
                        ]),
        ],
        grid: .twoStage,
        requires: [.transformer, .textEncoder, .videoVAECausal, .lora])
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXRecipes

/// The registry's contract, and above all the evidence rule.
///
/// The tests that matter here are the ones that stop a render from claiming a status it has
/// not earned. Everything else in this file is arithmetic; `gatedOnlyOnAnExactMatch` and its
/// neighbours are the ones that would let a silent mistake through if they were deleted.
@Suite("Recipe registry")
struct RecipeRegistryTests {

    private func request(width: Int? = nil, height: Int? = nil,
                         megapixels: Double? = nil, frames: Int? = 97,
                         steps: Int? = nil, fps: Double = 24,
                         aspect: AspectRatio = .r16x9) -> RecipeRequest {
        RecipeRequest(
            prompt: "x",
            videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"),
            frames: frames,
            frameRate: fps,
            steps: steps,
            megapixels: megapixels,
            aspectRatio: aspect,
            width: width,
            height: height)
    }

    // MARK: - Registration

    @Test("every registered id is unique and resolvable by name")
    func idsAreUniqueAndResolvable() throws {
        let ids = RecipeRegistry.ids
        #expect(Set(ids).count == ids.count)
        for id in ids {
            #expect(try RecipeRegistry.recipe(id).id == id)
        }
    }

    @Test("an unknown id is refused rather than defaulted")
    func unknownRecipeThrows() {
        #expect(throws: RecipeFailure.self) { try RecipeRegistry.recipe("nope") }
    }

    @Test("the default id is registered")
    func defaultIsRegistered() throws {
        #expect(try RecipeRegistry.recipe(RecipeRegistry.defaultID).id == "prod")
    }

    // MARK: - Evidence

    @Test("the production gate is claimed only at its exact configuration")
    func gatedAtProductionShape() throws {
        let resolved = try RecipeRegistry.production.resolve(request())
        #expect(resolved.output == Shape(width: 640, height: 384))
        guard case let .gated(shape, frames, steps, source) = resolved.evidence else {
            Issue.record("expected gated, got \(resolved.evidence)")
            return
        }
        #expect(shape == Shape(width: 640, height: 384))
        #expect(frames == 97)
        #expect(steps == 30)
        #expect(source.contains("l05_drift_ltx25_prod"))
    }

    @Test("the tiny gate is a second gate on the same recipe, not a second recipe")
    func gatedAtTinyShape() throws {
        let resolved = try RecipeRegistry.production.resolve(
            request(width: 320, height: 192, frames: 25))
        guard case let .gated(_, _, _, source) = resolved.evidence else {
            Issue.record("expected gated, got \(resolved.evidence)")
            return
        }
        #expect(source.contains("l05_drift_ltx25_tiny"))
    }

    /// The rule the whole type exists for: a gate at one shape is not a gate at another.
    @Test("a different shape on the gated pipeline is a sibling, naming what differs")
    func siblingNamesTheDifference() throws {
        let resolved = try RecipeRegistry.production.resolve(request(width: 1024, height: 576))
        guard case let .sibling(of, note) = resolved.evidence else {
            Issue.record("expected sibling, got \(resolved.evidence)")
            return
        }
        #expect(of == Shape(width: 640, height: 384))
        #expect(note.contains("1024x576"))
        #expect(note.contains("640x384"))
    }

    /// A step count alone must break the gate. If it did not, a 4-step render at the
    /// production shape would report itself as carrying the 30-step evidence.
    @Test("a matching shape at a different step count is NOT gated")
    func gatedOnlyOnAnExactMatch() throws {
        let resolved = try RecipeRegistry.production.resolve(request(steps: 4))
        #expect(resolved.output == Shape(width: 640, height: 384))
        if case .gated = resolved.evidence {
            Issue.record("30-step gate claimed at 4 steps")
        }
        guard case let .sibling(_, note) = resolved.evidence else {
            Issue.record("expected sibling, got \(resolved.evidence)")
            return
        }
        #expect(note.contains("4 steps vs 30"))
    }

    /// Same shape, same steps, different duration. The evidence behind the production shape
    /// was taken at 97 frames and says nothing about 121.
    @Test("a matching shape at a different frame count is NOT gated")
    func frameCountBreaksTheGate() throws {
        let resolved = try RecipeRegistry.production.resolve(request(frames: 121))
        if case .gated = resolved.evidence {
            Issue.record("97-frame gate claimed at 121 frames")
        }
    }

    @Test("a megapixel target resolves through the ladder and is a sibling")
    func megapixelTargetIsASibling() throws {
        let resolved = try RecipeRegistry.production.resolve(request(megapixels: 0.59))
        #expect(resolved.output == Shape(width: 1024, height: 576))
        guard case .sibling = resolved.evidence else {
            Issue.record("expected sibling, got \(resolved.evidence)")
            return
        }
    }

    // MARK: - Defaults

    /// A request naming no size takes the recipe's *gated* shape, not a derived rung. The
    /// ladder's own fallback is 0.25 MP, which has nothing behind it.
    @Test("no size named takes the recipe's gated shape")
    func defaultShapeIsTheGate() throws {
        #expect(try RecipeRegistry.production.resolve(request()).output
            == Shape(width: 640, height: 384))
        #expect(try RecipeRegistry.distilled.resolve(request()).output
            == Shape(width: 640, height: 384))
    }

    /// Regression. Every gate in this repository is landscape, so defaulting to the first
    /// one handed a caller who asked for 9:16 a 640x384 landscape clip — request ignored,
    /// output plausible, nothing in the result saying so.
    @Test("a portrait request does not silently take the landscape gate")
    func portraitDoesNotTakeALandscapeGate() throws {
        for recipe in RecipeRegistry.generative() {
            let resolved = try recipe.resolve(request(aspect: .r9x16))
            #expect(resolved.output == Shape(width: 384, height: 640),
                    "\(recipe.id) resolved portrait to \(resolved.output.description)")
            #expect(resolved.output.aspect < 1.0)
        }
    }

    /// The transpose carries the same token count, and orientation measures as free at
    /// equal tokens (0.38% at 3120). It is still only a sibling.
    @Test("the portrait transpose is a sibling at equal token count")
    func portraitIsASiblingAtEqualTokens() throws {
        let landscape = try RecipeRegistry.production.resolve(request())
        let portrait = try RecipeRegistry.production.resolve(request(aspect: .r9x16))
        #expect(landscape.videoTokens() == portrait.videoTokens())
        guard case .sibling = portrait.evidence else {
            Issue.record("expected sibling, got \(portrait.evidence)")
            return
        }
    }

    // MARK: - Schedules

    @Test("the distilled schedule is fixed at 8 + 3 and prod's is not")
    func fixedScheduleDetection() {
        #expect(RecipeRegistry.distilled.scheduleIsFixed)
        #expect(!RecipeRegistry.production.scheduleIsFixed)
        // 11 forwards is the recorded observed_transformer_forwards_total: 8 + 3.
        #expect(RecipeRegistry.distilled.declaredSteps == 11)
        #expect(RecipeRegistry.production.declaredSteps == 30)
    }

    /// Accepting `--steps` here and dropping it is the failure mode that survives a whole
    /// session: the render completes, looks plausible, and was not what was asked for.
    @Test("--steps against a fixed schedule is refused, not ignored")
    func stepsAgainstFixedScheduleRefused() {
        #expect(throws: RecipeFailure.self) {
            try RecipeRegistry.distilled.resolve(request(steps: 20))
        }
    }

    @Test("a transform recipe cannot be resolved from a generative request")
    func transformRefusesGenerativeRequest() {
        for recipe in RecipeRegistry.all where recipe.kind == .transform {
            #expect(throws: RecipeFailure.self) { try recipe.resolve(request()) }
        }
    }

    // MARK: - Which command runs it

    /// Every registered recipe must name a command that exists.
    ///
    /// The registry is the menu of *how a pipeline is conditioned and run*, so half its
    /// entries are executed by something other than `render`. A recipe naming a command the
    /// CLI does not have sends the reader to a subcommand that is not there — worse than an
    /// unlisted route, because it reads as a typo on the user's side.
    @Test("every recipe names a real subcommand")
    func commandsExist() {
        let commands: Set<String> = ["render", "upscale", "ingredients", "msr"]
        for recipe in RecipeRegistry.all {
            #expect(commands.contains(recipe.command),
                    "\(recipe.id) names '\(recipe.command)'" as Comment)
        }
    }

    /// A transform is never run by `render`, and the reference-conditioned routes are not
    /// either — for a different reason, which is the whole point of the field.
    ///
    /// This is the invariant that broke when `render` guarded on `kind == .generative`:
    /// `ingredients` and `msr` are generative and would have passed that guard, then
    /// rendered with no reference at all. A failure that completes successfully.
    @Test("no transform is renderable, and the reference routes are generative but are not")
    func renderableIsNotKind() {
        for recipe in RecipeRegistry.all where recipe.kind == .transform {
            #expect(recipe.command != "render", "\(recipe.id)" as Comment)
        }
        for id in ["ingredients", "msr"] {
            let recipe = try! RecipeRegistry.recipe(id)
            #expect(recipe.kind == .generative)
            #expect(recipe.command != "render")
            #expect(!RecipeRegistry.renderable().contains { $0.id == id })
        }
        #expect(RecipeRegistry.renderable().map(\.id) == ["prod", "dev-lora", "distilled"])
    }

    /// A recipe carrying in-context references must declare them.
    ///
    /// The menu's token column sizes a caller's memory budget, and an undeclared reference
    /// block makes it wrong by more than a rounding — at five MSR slots the references
    /// outweigh the picture. Tied to the adapter list so a future reference-conditioned
    /// recipe cannot be added without either declaring its references or deciding not to.
    @Test("the reference-conditioned recipes declare what they carry in context")
    func contextReferencesDeclared() throws {
        let ingredients = try RecipeRegistry.recipe("ingredients")
        #expect(ingredients.contextReferences == Recipe.ContextReferences(
            minimum: 1, maximum: 1, latentFramesEach: 1))
        let msr = try RecipeRegistry.recipe("msr")
        #expect(msr.contextReferences == Recipe.ContextReferences(
            minimum: 1, maximum: 5, latentFramesEach: 5))

        // At 640x384 the MSR references are 20x12 latent, 5 frames each: 1200 per slot.
        let tokens = msr.contextReferences!.tokens(latentHeight: 12, latentWidth: 20)
        #expect(tokens == 1200 ... 6000)
        // Which is not a rounding against the 3120 the target generates.
        #expect(tokens.upperBound > 3120)
    }

    /// The menu has to say how to run each row.
    @Test("the catalogue prints the command and the reference cost")
    func catalogueReportsCommandAndReferences() {
        let report = RecipeCatalogue().report
        #expect(report.contains("ltx msr"))
        #expect(report.contains("ltx ingredients"))
        #expect(report.contains("ltx upscale"))
        #expect(report.contains("in-context reference tokens"))
        // The upscale rows used to claim `render` because the field defaulted.
        #expect(!report.contains("upscale-plain     ltx render"))
    }

    // MARK: - Stage geometry

    @Test("the distilled draft stage is half the output and lands on the grid")
    func distilledStageShapes() throws {
        let resolved = try RecipeRegistry.distilled.resolve(request())
        #expect(resolved.stages.count == 2)
        #expect(resolved.stages[0].shape == Shape(width: 320, height: 192))
        #expect(resolved.stages[1].shape == Shape(width: 640, height: 384))
        #expect(resolved.stages[0].stage.decodes == false)
        #expect(resolved.stages[1].stage.decodes == true)
    }

    /// Documents a real constraint rather than asserting it is the right one.
    ///
    /// `RecipeStage.shape(forOutput:)` re-checks the scaled stage against `Grid.single`,
    /// whose value is 64 — so a half-resolution stage forces the *output* to be a multiple
    /// of 128, not 64. 1024x576 is a legal output on the 64 grid and its draft, 512x288, is
    /// not. A looser stage-1 rule of ÷32 would admit this shape; the stricter one is what
    /// ships.
    @Test("two-stage effectively requires a multiple of 128 on both axes")
    func twoStageGridIsStricterThanSingleStage() throws {
        // Legal single-stage, refused two-stage.
        _ = try RecipeRegistry.production.resolve(request(width: 1024, height: 576))
        #expect(throws: RecipeFailure.self) {
            try RecipeRegistry.distilled.resolve(request(width: 1024, height: 576))
        }
    }

    // MARK: - Token counts

    @Test("production token counts match the benchmarked shape")
    func productionTokens() throws {
        let resolved = try RecipeRegistry.production.resolve(request())
        // 13 latent frames x 12 x 20 = 3120, the count the cost law is priced against.
        #expect(resolved.videoTokens() == [3120])
        // round(97 / 24 * 25) = 101, contract 3.
        #expect(resolved.audioTokens() == 101)
    }

    @Test("the distilled draft carries a quarter of the refine's tokens")
    func distilledTokens() throws {
        let tokens = try RecipeRegistry.distilled.resolve(request()).videoTokens()
        #expect(tokens == [780, 3120])
    }
}

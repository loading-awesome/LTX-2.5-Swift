// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

import LTXFoundation
import LTXRecipes
@testable import LTXPipeline

/// Stage geometry and guidance belong to the recipe, not to the caller.
///
/// This suite exists because of a specific regression. `--width`/`--height` were introduced
/// as the *output* size on the reference-conditioned commands, with stage 1 sampling at half
/// them. The flags a caller already had then meant something new: `--width 640` went from
/// "sample at 640" to "sample at 320 and upscale". Nobody's command line changed and every
/// rigid subject started melting, because a draft is not a smaller picture — it is a picture
/// with less structure in it, and a refine that re-noises to 0.909 invents what is missing
/// independently per frame.
///
/// The fix is structural rather than a better default: the shapes come from
/// ``RecipeStage/scale`` against a resolved output, so moving what a stage samples at means
/// editing a named recipe. These tests hold that boundary shut.
@Suite("Recipes own stage geometry")
struct HalfResolutionStageTests {

    // MARK: - Geometry comes from the recipe

    @Test("Options take every shape from the resolved plan, and none from the caller")
    func shapesComeFromThePlan() throws {
        let plan = try RecipePlan.ingredients()
        let options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: "/dev/null"), prompt: "",
            output: URL(fileURLWithPath: "/dev/null"), plan: plan)

        #expect(options.width == plan.output.width)
        #expect(options.height == plan.output.height)
        #expect(options.sampledWidth == plan.stages[0].shape.width)
        #expect(options.sampledHeight == plan.stages[0].shape.height)
        #expect(options.frames == plan.frames)
    }

    /// The two-stage recipe still drafts at half — that arrangement was never the bug.
    @Test("A two-stage recipe samples stage 1 at half its output")
    func twoStageDraftsAtHalf() throws {
        let plan = try RecipePlan.ingredients()
        let options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: "/dev/null"), prompt: "",
            output: URL(fileURLWithPath: "/dev/null"), plan: plan)

        #expect(options.twoStage)
        #expect(options.sampledWidth * 2 == options.width)
        #expect(options.sampledHeight * 2 == options.height)
    }

    /// The bug, stated as an invariant: a single-stage recipe must sample the full output.
    /// If a `scale` below 1.0 ever appears on a first-and-only stage, the draft has moved
    /// without a refine to carry it back up, which is the melting case with no upside.
    @Test("Every single-stage recipe samples exactly what it writes")
    func singleStageSamplesItsOutput() throws {
        for recipe in RecipeRegistry.all where recipe.kind == .generative {
            let plan = try RecipePlan.resolve(recipe)
            guard plan.stages.count == 1 else { continue }
            #expect(plan.stages[0].shape == plan.output,
                    "\(recipe.id) samples \(plan.stages[0].shape) and writes \(plan.output)")
        }
    }

    /// The prod routes are single stage **on purpose**. A refine cannot restore structure a
    /// draft never had, so the expensive recipes keep their resolution and buy time back on
    /// passes instead.
    @Test("The prod reference recipes are single stage at full resolution")
    func prodRoutesAreSingleStage() throws {
        for recipe in [RecipeRegistry.ingredientsProd, RecipeRegistry.ingredientsProdCFG,
                       RecipeRegistry.msrProd, RecipeRegistry.msrProdCFG] {
            let plan = try RecipePlan.resolve(recipe)
            #expect(plan.stages.count == 1, "\(recipe.id) should not draft")
            #expect(plan.sampledShape == plan.output)
        }
    }

    // MARK: - Guidance is named by the recipe

    /// The pass count is the cost, and it follows from the declared scales alone.
    @Test("CFG-only halves the forwards against full production guidance")
    func cfgOnlyHalvesTheForwards() throws {
        let full = try RecipePlan.ingredientsProd()
        let cfg = try RecipePlan.ingredientsProdCFG()

        #expect(full.forwardCount() == 120, "30 steps x 4 passes")
        #expect(cfg.forwardCount() == 60, "30 steps x 2 passes")
        #expect(cfg.forwardCount() * 2 == full.forwardCount())
        // Same schedule and same resolution — only the guidance differs, which is the whole
        // claim being made about this recipe.
        #expect(cfg.stages[0].stage.sigmas.steps == full.stages[0].stage.sigmas.steps)
        #expect(cfg.output == full.output)
    }

    @Test("A distilled recipe builds no guider at all")
    func distilledRunsOneForwardPerStep() throws {
        let plan = try RecipePlan.ingredients()
        let options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: "/dev/null"), prompt: "",
            output: URL(fileURLWithPath: "/dev/null"), plan: plan)
        #expect(options.guided == nil, "distilled guidance means no guider is built")
        #expect(plan.forwardCount() == plan.stages.reduce(0) { $0 + $1.stage.sigmas.steps },
                "one forward per step across every stage")
    }

    @Test("A guided recipe carries the recipe's own scales, not a hardcoded production pair")
    func guidedCarriesTheRecipesScales() throws {
        let options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: "/dev/null"), prompt: "",
            output: URL(fileURLWithPath: "/dev/null"), plan: try RecipePlan.ingredientsProdCFG())
        let guided = try #require(options.guided)

        #expect(guided.video.cfgScale == RecipeGuidance.cfgOnlyVideo.cfgScale)
        #expect(guided.video.stgScale == 0, "STG off is what makes this two passes")
        #expect(guided.video.modalityScale == 1, "modality neutral at 1, not 0")
        #expect(guided.audio.cfgScale == RecipeGuidance.cfgOnlyAudio.cfgScale)
    }

    // MARK: - Schedules come from the stage's own plan

    @Test("A fixed table is passed through verbatim rather than rebuilt from a step count")
    func fixedTablesArePassedThrough() throws {
        let options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: "/dev/null"), prompt: "",
            output: URL(fileURLWithPath: "/dev/null"), plan: try RecipePlan.ingredients())
        #expect(options.stage1Sigmas == FlowSchedule.distilledStage1)
    }

    /// A second stage does not start at pure noise: it re-noises the upsampled draft and
    /// descends. A curve starting at 1.0 would lerp the draft *entirely* to noise, turning
    /// the refine into a fresh full-resolution render — which is a bug that completes.
    @Test("A refine stage starts at the re-noise level, not at 1.0")
    func refineStartsPartWayDown() throws {
        let options = try IngredientsRenderer.Options(
            reference: URL(fileURLWithPath: "/dev/null"), prompt: "",
            output: URL(fileURLWithPath: "/dev/null"), plan: try RecipePlan.ingredients())
        let stage2 = try #require(options.stage2Sigmas)

        #expect(stage2[0] == FlowSchedule.distilledStage2[0])
        #expect(stage2[0] == 0.909375)
        #expect(stage2.last == 0.0, "the curve must land on zero or the last step is partial")
        #expect(options.refineSteps == stage2.count - 1)
    }

    /// Scaling rather than truncating is what makes a refine *the same curve entered part
    /// way* instead of a different curve that happens to start at the right number.
    @Test("Scaling a continuous plan keeps its shape")
    func scalingKeepsTheShape() throws {
        let plan = SigmaPlan.continuous(steps: 8)
        let base = try plan.sigmas()
        let scaled = try plan.sigmas(denoise: 0.909375)

        #expect(scaled.count == base.count)
        for i in 0 ..< scaled.count where base[i] != 0 {
            #expect(abs(scaled[i] / base[i] - 0.909375) < 1e-6,
                    "sigma \(i) scaled by a different factor than the re-noise level")
        }
    }

    // MARK: - The grid

    /// `LTXRecipes.Grid` is 64 because a halved stage must still land on the VAE's 32.
    /// Pinned so a stricter rule cannot silently refuse rungs that render correctly today.
    @Test("Every multiple of 64 halves onto the VAE's 32 grid")
    func theGridRuleIsSufficient() {
        for size in stride(from: 64, through: 2048, by: 64) {
            #expect((size / 2) % 32 == 0,
                    "\(size) halves to \(size / 2), which is off the VAE grid")
        }
    }

    /// Resolution is where an off-grid shape is refused — before a checkpoint is opened,
    /// rather than by a second rule inside each renderer.
    @Test("An off-grid request is refused during resolution")
    func offGridIsRefusedEarly() {
        #expect(throws: (any Error).self) {
            _ = try RecipeRegistry.ingredients.resolve(RecipeRequest(
                prompt: "", videoOutput: URL(fileURLWithPath: "/dev/null"),
                seconds: 2, frameRate: 24, aspectRatio: .r16x9,
                width: 641, height: 384))
        }
    }
}

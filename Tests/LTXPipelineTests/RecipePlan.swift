// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

import LTXRecipes

/// Resolved recipes for tests, through the registry's own path rather than a hand-built
/// `ResolvedRecipe`.
///
/// Constructing one directly would let a test assert a stage shape that
/// ``RecipeStage/shape(forOutput:)`` would have refused — which is the class of thing these
/// suites exist to catch, so the resolution is real and only the request is a fixture.
enum RecipePlan {

    static func resolve(_ recipe: Recipe, frames: Int = 49, frameRate: Double = 24,
                        megapixels: Double? = nil,
                        aspect: AspectRatio = .r16x9) throws -> ResolvedRecipe {
        try recipe.resolve(RecipeRequest(
            prompt: "", videoOutput: URL(fileURLWithPath: "/dev/null"),
            seconds: Double(frames) / frameRate, frames: frames, frameRate: frameRate,
            megapixels: megapixels, aspectRatio: aspect))
    }

    static func ingredients(frames: Int = 49) throws -> ResolvedRecipe {
        try resolve(RecipeRegistry.ingredients, frames: frames)
    }

    static func ingredientsProd(frames: Int = 49) throws -> ResolvedRecipe {
        try resolve(RecipeRegistry.ingredientsProd, frames: frames)
    }

    static func ingredientsProdCFG(frames: Int = 49) throws -> ResolvedRecipe {
        try resolve(RecipeRegistry.ingredientsProdCFG, frames: frames)
    }

    static func msr(frames: Int = 49) throws -> ResolvedRecipe {
        try resolve(RecipeRegistry.msr, frames: frames)
    }
}

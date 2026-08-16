// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXRecipes

/// Prints the recipe menu, resolved.
///
/// Argument plumbing only. The shapes, the token counts and the evidence rules live in
/// `RecipeCatalogue` and `Recipe`, where they are tested without spawning a process and
/// where they are the same code `render` resolves against — a listing that resolved shapes
/// independently would eventually advertise one size and render another.
struct Recipes: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "recipes",
        abstract: "List the pipelines this port can run, with shape and evidence.",
        discussion: """
            Every registered pipeline, resolved at one duration and size target.

              ltx recipes
              ltx recipes --megapixels 0.6
              ltx recipes --aspect-ratio 9:16 --seconds 8

            With no --megapixels, each recipe reports its own gated shape — the shape
            `ltx render` would pick if you named no size either.

            Only routes that run today are listed. The IC-LoRA adapters LTX ships for
            2.3 — Union Control, Motion Track, Ingredients, In-Outpainting — have no 2.5
            release, so they are absent rather than listed as unavailable.
            """
    )

    @Option(help: "Megapixel target. Omit to see each recipe's own gated shape.")
    var megapixels: Double?

    @Option(help: "Aspect ratio (16:9, 9:16, 1:1, 21:9, 4:3, 3:4).")
    var aspectRatio: String = "16:9"

    @Option(help: "Duration to resolve the menu at. Snaps down onto the 8k+1 lattice.")
    var seconds: Double = 97.0 / 24.0

    @Option(help: "Frame rate to resolve the menu at.")
    var fps: Double = 24

    func run() throws {
        guard let aspect = AspectRatio(rawValue: aspectRatio) else {
            throw ValidationError(
                "--aspect-ratio must be one of "
                    + AspectRatio.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let catalogue = RecipeCatalogue(megapixels: megapixels, aspect: aspect,
                                        seconds: seconds, frameRate: fps)
        print("")
        print(catalogue.report)
    }
}

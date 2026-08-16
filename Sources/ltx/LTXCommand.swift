// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import LTX25

/// A thin CLI over the public API. Every subcommand is argument plumbing over
/// a call into `LTX25`; logic that needs a test belongs in a library target.
///
/// `AsyncParsableCommand` is not a preference. `render` drives an actor, so it
/// is async, and ArgumentParser only awaits an async subcommand when the root
/// is async too. Declared synchronous it warns and never calls `run()`.
@main
struct LTXCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ltx",
        abstract: "Generate video and audio together with LTX 2.5, on Apple silicon.",
        version: LTX25.version,
        // `bench` produces neither a clip nor a quality result, only timings, which is
        // exactly why it is not a mode of `render`.
        // `upscale` is a third kind: it takes a clip that already exists rather than a
        // prompt, so it shares no request with `render` — no prompt, no seed, no schedule,
        // and an output size fixed at 2x the input by construction.
        // `recipes` is a fourth kind again: it produces no artifact at all, only a reading
        // of what the port can run and what stands behind each route. It resolves through
        // the same `Recipe.resolve` the render path uses, so the menu cannot advertise a
        // shape the render then disagrees with.
        // `doctor` is a fifth kind: it produces no artifact and makes no claim about a
        // render, only a reading of this machine, these files and what the port can do with
        // them. It links no MLX and reads no payload, so it stays runnable when everything
        // else is broken — which is the state it exists for.
        subcommands: [Render.self, Bench.self, Upscale.self, Recipes.self,
                      Ingredients.self, MSR.self, Doctor.self]
    )
}

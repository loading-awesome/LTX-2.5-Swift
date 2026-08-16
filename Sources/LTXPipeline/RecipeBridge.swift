// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

import LTXFoundation
import LTXRecipes

/// The two conversions that turn a declared recipe stage into something the sampler runs.
///
/// Both existed already, inline and private, inside `LTX25.RenderJob` — which is why the
/// reference-conditioned commands could not use them and grew their own geometry and
/// guidance instead. That divergence is what let `--width` mean the sampled size on one
/// route and the output size on another. One conversion, in one place, so a recipe means
/// the same thing whichever command runs it.
extension GuidanceParams {

    /// A recipe's declared guidance, as the sampler's parameters.
    ///
    /// A straight field copy, and worth having as a named conversion anyway: the pass count
    /// — the thing that decides what a render costs — is a function of these numbers and
    /// nothing else. ``Sampler/plan(step:sigma:)`` adds `unconditional` when `cfgScale != 1`,
    /// `perturbed` when `stgScale != 0`, and `modality` when `modalityScale != 1`. A recipe
    /// that wants two forwards per step rather than four says so here and nowhere else.
    public init(_ guidance: RecipeGuidance) {
        self.init(cfgScale: guidance.cfgScale,
                  stgScale: guidance.stgScale,
                  stgBlocks: guidance.stgBlocks,
                  rescaleScale: guidance.rescaleScale,
                  modalityScale: guidance.modalityScale)
    }
}

extension SigmaPlan {

    /// The stage's sigmas, whether it declared a table or a step count.
    ///
    /// `.fixed` is returned verbatim — a recorded table is the schedule, not a sample of
    /// one, so nothing here interpolates or thins it. `.continuous` is built from
    /// ``FlowSchedule``, which is what the dev checkpoint samples on.
    ///
    /// `denoise` scales the whole curve so it starts there instead of at its own head. It
    /// exists for a second stage, which does not begin at pure noise: it re-noises an
    /// upsampled draft to a partial level and descends from that. Scaling rather than
    /// truncating keeps the schedule's shape — the ratios between live sigmas — which is
    /// what separates "the same curve, entered partway" from "a different curve that
    /// happens to start at the right number".
    public func sigmas(denoise: Float? = nil) throws -> [Float] {
        let curve: [Float]
        switch self {
        case let .fixed(values):
            curve = values
        case let .continuous(steps):
            curve = try FlowSchedule().sigmas(steps: steps)
        }
        guard let denoise, let head = curve.first, head > 0 else { return curve }
        let k = denoise / head
        return curve.map { $0 * k }
    }
}

extension ResolvedRecipe {

    /// Whether this resolution runs a second stage at all.
    public var isTwoStage: Bool { stages.count > 1 }

    /// The shape stage 1 samples at — **not** the output, when a refine follows.
    ///
    /// The distinction is the whole reason this is read from the recipe rather than typed.
    /// Draft resolution decides whether rigid structure survives: a subject occupying a
    /// 10x6 latent grid has nowhere to put mechanical detail, and a refine that re-noises
    /// to 0.909 then invents it independently per frame, which reads as melting. Tying it
    /// to `RecipeStage.scale` means changing it is a recipe edit with a name on it.
    public var sampledShape: Shape { stages.first?.shape ?? output }

    /// Forwards this resolution will make, across every stage. Exact, and made without
    /// making any of them — the pass count follows from the declared guidance.
    public func forwardCount() -> Int {
        stages.reduce(0) { total, entry in
            let sampler = Sampler(video: GuidanceParams(entry.stage.videoGuidance),
                                  audio: GuidanceParams(entry.stage.audioGuidance))
            return total + sampler.passCount(steps: entry.stage.sigmas.steps)
        }
    }
}

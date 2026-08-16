// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// The recipe menu, resolved at one duration and one size target.
///
/// **A list of names is not a menu.** Three of the things a caller most needs to know about
/// a recipe — the shape it will actually render, whether it can run at all at that shape,
/// and what evidence stands behind the result — are nowhere in its name. Discovering any of
/// them by starting a render is the failure this type exists to prevent.
///
/// The rows are produced by ``Recipe/resolve(_:lattice:)``, the same call `render` makes. A
/// listing that resolved shapes independently would eventually advertise one size here and
/// render another there.
///
/// No time estimate. The cost law is known — 7.29 s per pass at 3120 tokens, linear in
/// steps, `tokens^0.75` in shape — but the pass count comes from
/// guidance settings whose authority is `Sampler.plan`, in an MLX-bound target this one
/// cannot import. Deriving a second pass-count rule here is how a menu starts quoting a
/// number the sampler disagrees with, so the column is absent until the two can share one
/// implementation.
public struct RecipeCatalogue: Sendable {

    public struct Row: Sendable {
        public let id: String
        public let summary: String
        public let kind: Recipe.Kind
        /// The subcommand that runs it. Printed because half the registry is not `render`,
        /// and a menu that names a route without naming how to invoke it sends the reader
        /// to `--help` to find out.
        public let command: String
        /// Nil for a transform recipe, whose geometry comes from its input file.
        public let shape: Shape?
        public let stageCount: Int
        public let videoTokens: [Int]
        /// Tokens the in-context references add to the *first* stage's video stream, at the
        /// resolved shape. Nil when the recipe carries none.
        ///
        /// Reported separately rather than folded into ``videoTokens``, which is the count
        /// that sizes the noise draw and the unpatchify. The two genuinely differ, and it is
        /// the sum that has to fit in memory.
        public let referenceTokens: ClosedRange<Int>?
        public let audioTokens: Int?
        public let evidence: Evidence?
        /// Why this recipe could not be resolved at these parameters.
        ///
        /// Recorded rather than dropped. A recipe that vanishes from the menu at one size
        /// reads as "not available", when the truth is usually "not available *here*" and
        /// the reason is actionable.
        public let failure: String?
    }

    public let rows: [Row]
    public let frameRate: Double
    public let seconds: Double

    /// Resolve every registered recipe against one set of parameters.
    ///
    /// `megapixels` nil means "each recipe's own measured shape", which is the default
    /// `render` itself takes and therefore the most useful thing to show first.
    public init(megapixels: Double? = nil,
                aspect: AspectRatio = .r16x9,
                seconds: Double = 97.0 / 24.0,
                frameRate: Double = 24,
                recipes: [Recipe] = RecipeRegistry.all) {
        self.frameRate = frameRate
        self.seconds = seconds

        self.rows = recipes.map { recipe in
            guard recipe.kind == .generative else {
                return Row(id: recipe.id, summary: recipe.summary, kind: recipe.kind,
                           command: recipe.command,
                           shape: nil, stageCount: recipe.stages.count,
                           videoTokens: [], referenceTokens: nil, audioTokens: nil,
                           evidence: recipe.unroutable.map { Evidence.noRoute($0) },
                           failure: nil)
            }
            let request = RecipeRequest(
                prompt: "", videoOutput: URL(fileURLWithPath: "/dev/null"),
                seconds: seconds, frameRate: frameRate,
                megapixels: megapixels, aspectRatio: aspect)
            do {
                let resolved = try recipe.resolve(request)
                // Against the first stage's grid, which is where the references are appended
                // — for a two-stage recipe that is the half-size draft, not the output.
                let first = try recipe.stages.first?.shape(forOutput: resolved.output)
                    ?? resolved.output
                let latent = LatentGeometry()
                let references = recipe.contextReferences?.tokens(
                    latentHeight: first.height / latent.spatialScale,
                    latentWidth: first.width / latent.spatialScale)
                return Row(id: recipe.id, summary: recipe.summary, kind: recipe.kind,
                           command: recipe.command,
                           shape: resolved.output, stageCount: resolved.stages.count,
                           videoTokens: resolved.videoTokens(),
                           referenceTokens: references,
                           audioTokens: resolved.audioTokens(),
                           evidence: resolved.evidence, failure: nil)
            } catch {
                return Row(id: recipe.id, summary: recipe.summary, kind: recipe.kind,
                           command: recipe.command,
                           shape: nil, stageCount: recipe.stages.count,
                           videoTokens: [], referenceTokens: nil, audioTokens: nil,
                           evidence: nil, failure: "\(error)")
            }
        }
    }

    /// Columns padded in Swift, not by `String(format:)`.
    ///
    /// Darwin's Foundation ignores width flags for `%@`, so `%-18@` does not pad and the
    /// table silently collapses into a ragged list.
    private func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }

    public var report: String {
        var out: [String] = []
        out.append(pad("id", 18) + pad("run with", 16) + pad("shape", 12) + pad("stages", 8)
                   + pad("tokens", 16) + "status")

        for row in rows {
            // A generative recipe with no shape was refused, not fed from a file. Printing
            // "from input" for it would misattribute the reason.
            let shape = row.shape?.description
                ?? (row.kind == .transform ? "from input" : "—")
            let stages = row.kind == .transform && row.stageCount == 0
                ? "-" : String(row.stageCount)
            let tokens = row.videoTokens.isEmpty
                ? "-"
                : row.videoTokens.map(String.init).joined(separator: " + ")
            let status = row.failure == nil ? (row.evidence?.label ?? "ok") : "REFUSED"
            out.append(pad(row.id, 18) + pad("ltx " + row.command, 16) + pad(shape, 12)
                       + pad(stages, 8) + pad(tokens, 16) + status)
            out.append("    " + row.summary)
            if let references = row.referenceTokens {
                let span = references.lowerBound == references.upperBound
                    ? String(references.lowerBound)
                    : "\(references.lowerBound)-\(references.upperBound)"
                out.append("    + \(span) in-context reference tokens on stage 1, which the "
                           + "transformer attends over and the decode never sees")
            }
            if let failure = row.failure { out.append("    refused: \(failure)") }
            if case let .sibling(_, note) = row.evidence { out.append("    \(note)") }
            if case let .gated(_, _, _, source) = row.evidence {
                out.append("    gate: \(source)")
            }
            if case let .unmeasured(note) = row.evidence { out.append("    \(note)") }
            if case let .noRoute(note) = row.evidence { out.append("    no route: \(note)") }
        }

        out.append("")
        out.append("Audio tokens are round(frames / fps * 25); video tokens are exact, not "
                   + "estimates.")
        out.append("`gated` means this pipeline has been measured at this exact shape, "
                   + "duration and step count.")
        out.append("`sibling` means the same pipeline measured at a different shape. It is "
                   + "not a claim about this shape.")
        return out.joined(separator: "\n")
    }
}

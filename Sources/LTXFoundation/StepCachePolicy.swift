// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Whether to reuse the previous step's residual, decided from two scalars.
///
/// **Deliberately free of MLX.** Measuring how much block 0's residual moved
/// needs tensors; deciding what to do about it does not. Every rule that can
/// fail silently — warmup, cooldown, consecutive cap, which stream gets to
/// veto — lives here so it tests in microseconds with no GPU and no checkpoint.
///
/// Each rule below was found the hard way:
///
/// **Never reuse on the first step.** There is no history to compare against.
///
/// **Never reuse on the last step.** Its residual lands directly in the decoded
/// pixels, with no later step to correct it.
///
/// **Bound consecutive reuse because the cached residual ages.** Block 0 runs
/// on every step, skipped or not, and the probe is refreshed unconditionally.
/// What goes stale is the **total stack residual**, which only a full step
/// refreshes. The cap bounds residual *age*.
///
/// **Gate on the worse stream, not the average.** LTX's video tokens dominate
/// the packed count, so a probe averaged over everything is a video-only
/// decision. LTX also runs up to four guidance passes per step; this policy
/// does not know about that. The caller holds **one policy instance's worth
/// of state per pass**, because a conditional residual is not comparable to
/// an unconditional one.
public struct StepCachePolicy: Sendable, Equatable {

    public let threshold: Double
    public let maxConsecutiveSkips: Int
    public let warmupSteps: Int
    public let cooldownSteps: Int
    /// When false, the whole-sequence change alone decides. Kept so the two
    /// probes can be A/B'd rather than argued about.
    public let perStream: Bool

    /// The measured knee in block 0's residual change: below it the total stack
    /// residual has barely moved. **Not a quality claim for LTX** — the mechanism
    /// is the same, the schedule and the model the number came off are not.
    /// Default in the recipe is 0 (off) until a sweep fills the quality column.
    public static let measuredThreshold: Double = 0.10
    /// The consecutive-reuse cap that survived a viewing on 2026-08-06. Same caveat.
    public static let measuredCap: Int = 5

    public init(threshold: Double, maxConsecutiveSkips: Int = 5,
                warmupSteps: Int = 1, cooldownSteps: Int = 1,
                perStream: Bool = true) {
        self.threshold = threshold
        self.maxConsecutiveSkips = maxConsecutiveSkips
        self.warmupSteps = warmupSteps
        self.cooldownSteps = cooldownSteps
        self.perStream = perStream
    }

    public enum Decision: Sendable, Equatable {
        case runFull
        case reuse
    }

    public enum Reason: String, Sendable, Equatable, CaseIterable {
        case noHistory
        case warmup
        case cooldown
        case consecutiveCap
        case nonFinite
        case wholeSequenceAboveThreshold
        case audioAboveThreshold
        /// Advisory only — video does not vote. Recorded so the case for
        /// giving it one can be made from data.
        case videoAboveThreshold = "videoRowsAboveThreshold"
        case belowThreshold
    }

    public struct Verdict: Sendable, Equatable {
        public let decision: Decision
        public let reason: Reason
        /// Every constraint that independently forced a full step, not just
        /// the headline. Under a tight cap the audio veto is invisible in
        /// `reason` and load-bearing the moment the cap is relaxed.
        public let constraints: [Reason]
        public let advisory: [Reason]
        public let effectiveChange: Double
    }

    public func explain(wholeSequenceChange: Double, audioChange: Double?,
                        videoChange: Double? = nil,
                        step: Int, totalSteps: Int,
                        consecutiveSkips: Int, haveCachedResidual: Bool) -> Verdict {
        var constraints: [Reason] = []
        if !haveCachedResidual { constraints.append(.noHistory) }
        if step < warmupSteps { constraints.append(.warmup) }
        if step >= totalSteps - cooldownSteps { constraints.append(.cooldown) }
        if consecutiveSkips >= maxConsecutiveSkips { constraints.append(.consecutiveCap) }

        let effective = (perStream && audioChange != nil)
            ? Swift.max(wholeSequenceChange, audioChange!)
            : wholeSequenceChange
        if !effective.isFinite {
            constraints.append(.nonFinite)
        } else {
            if wholeSequenceChange >= threshold {
                constraints.append(.wholeSequenceAboveThreshold)
            }
            if perStream, let a = audioChange, a.isFinite, a >= threshold {
                constraints.append(.audioAboveThreshold)
            }
        }

        var advisory: [Reason] = []
        if let v = videoChange, v.isFinite, v >= threshold {
            advisory.append(.videoAboveThreshold)
        }

        guard constraints.isEmpty else {
            let structural = constraints.first {
                $0 == .noHistory || $0 == .warmup || $0 == .cooldown
                    || $0 == .consecutiveCap || $0 == .nonFinite
            }
            let headline: Reason
            if let structural {
                headline = structural
            } else if constraints.contains(.audioAboveThreshold),
                      (audioChange ?? -.infinity) >= wholeSequenceChange {
                headline = .audioAboveThreshold
            } else {
                headline = .wholeSequenceAboveThreshold
            }
            return Verdict(decision: .runFull, reason: headline, constraints: constraints,
                           advisory: advisory, effectiveChange: effective)
        }
        return Verdict(decision: .reuse, reason: .belowThreshold, constraints: [],
                       advisory: advisory, effectiveChange: effective)
    }

    public func decide(wholeSequenceChange: Double, audioChange: Double?,
                       step: Int, totalSteps: Int,
                       consecutiveSkips: Int, haveCachedResidual: Bool) -> Decision {
        explain(wholeSequenceChange: wholeSequenceChange, audioChange: audioChange,
                step: step, totalSteps: totalSteps, consecutiveSkips: consecutiveSkips,
                haveCachedResidual: haveCachedResidual).decision
    }
}

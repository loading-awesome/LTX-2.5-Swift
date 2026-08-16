// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
@testable import LTXFoundation

/// The cache's decision rules, each of which fails silently if broken.
///
/// No MLX, no GPU, no checkpoint — scalar arithmetic, which is the point of
/// separating the policy from the measurement.
@Suite("step cache policy")
struct StepCachePolicyTests {

    static let p = StepCachePolicy(threshold: 0.5, maxConsecutiveSkips: 3)
    static let quiet = 0.01

    @Test("nothing is reused before a full step has been recorded")
    func needsHistory() {
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: false) == .runFull)
    }

    @Test("the first step always runs in full")
    func warmup() {
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 0, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
    }

    @Test("the last step always runs in full")
    func cooldown() {
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 19, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 18, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .reuse)
    }

    @Test("consecutive reuse is capped")
    func consecutiveCap() {
        for n in 0 ..< 3 {
            #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                                  step: 5, totalSteps: 20, consecutiveSkips: n,
                                  haveCachedResidual: true) == .reuse)
        }
        #expect(Self.p.decide(wholeSequenceChange: Self.quiet, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 3,
                              haveCachedResidual: true) == .runFull)
    }

    @Test("the per-stream probe lets the audio veto a skip")
    func audioVetoes() {
        let perStream = StepCachePolicy(threshold: 0.5, perStream: true)
        #expect(perStream.decide(wholeSequenceChange: 0.01, audioChange: 0.9,
                                 step: 5, totalSteps: 20, consecutiveSkips: 0,
                                 haveCachedResidual: true) == .runFull)
        #expect(perStream.decide(wholeSequenceChange: 0.01, audioChange: 0.02,
                                 step: 5, totalSteps: 20, consecutiveSkips: 0,
                                 haveCachedResidual: true) == .reuse)
    }

    @Test("the whole-sequence probe misses exactly what the per-stream probe catches")
    func wholeSequenceMisses() {
        let whole = StepCachePolicy(threshold: 0.5, perStream: false)
        #expect(whole.decide(wholeSequenceChange: 0.01, audioChange: 0.9,
                             step: 5, totalSteps: 20, consecutiveSkips: 0,
                             haveCachedResidual: true) == .reuse)
    }

    @Test("a non-finite change never reuses")
    func infiniteChangeRuns() {
        #expect(Self.p.decide(wholeSequenceChange: .infinity, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
        #expect(Self.p.decide(wholeSequenceChange: .nan, audioChange: nil,
                              step: 5, totalSteps: 20, consecutiveSkips: 0,
                              haveCachedResidual: true) == .runFull)
    }

    @Test("a threshold of zero disables reuse entirely")
    func zeroThresholdNeverReuses() {
        let off = StepCachePolicy(threshold: 0)
        #expect(off.decide(wholeSequenceChange: 0, audioChange: nil,
                           step: 5, totalSteps: 20, consecutiveSkips: 0,
                           haveCachedResidual: true) == .runFull)
    }

    @Test("explain reports every independent constraint, not just the headline")
    func constraintsAreComplete() {
        // Cap AND audio both object. Headline is the cap (structural first);
        // audio must still appear so a cap sweep cannot conclude the veto is idle.
        let v = Self.p.explain(wholeSequenceChange: 0.01, audioChange: 0.9,
                               step: 5, totalSteps: 20, consecutiveSkips: 3,
                               haveCachedResidual: true)
        #expect(v.decision == .runFull)
        #expect(v.reason == .consecutiveCap)
        #expect(v.constraints.contains(.consecutiveCap))
        #expect(v.constraints.contains(.audioAboveThreshold))
    }
}

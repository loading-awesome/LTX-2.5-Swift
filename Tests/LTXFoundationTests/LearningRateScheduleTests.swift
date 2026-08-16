// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXFoundation

/// The standard schedules, each against the formula that defines it.
///
/// A wrong schedule does not fail. It trains, and produces an adapter that is undertrained or
/// overcooked, and the only symptom is a result that could have been the data's fault.
@Suite("Learning rate schedule")
struct LearningRateScheduleTests {

    private let base = 1e-4

    @Test("constant is constant, including past the budget")
    func constantHolds() {
        let schedule = LearningRateSchedule.constant
        for step in [0, 1, 500, 1000, 5000] {
            #expect(schedule.rate(base: base, step: step, steps: 1000) == base)
        }
    }

    /// Linear decay interpolates the *factor*, and it ends at `end_factor = 0.1`. Decaying
    /// to a tenth rather than to zero is the easy thing to get wrong here, so it is the
    /// thing asserted.
    @Test("linear decays to a tenth, not to zero")
    func linearEndsAtATenth() {
        let schedule = LearningRateSchedule.linear()
        #expect(schedule.rate(base: base, step: 0, steps: 1000) == base)
        #expect(abs(schedule.rate(base: base, step: 500, steps: 1000) - base * 0.55)
            <= 1e-18)
        let end = schedule.rate(base: base, step: 1000, steps: 1000)
        #expect(abs(end - base * 0.1) <= 1e-18, "\(end)" as Comment)
        #expect(end > 0, "linear must not reach zero")
    }

    @Test("linear holds its end factor past the budget")
    func linearClamps() {
        let schedule = LearningRateSchedule.linear()
        let atEnd = schedule.rate(base: base, step: 1000, steps: 1000)
        #expect(schedule.rate(base: base, step: 4000, steps: 1000) == atEnd)
    }

    /// `eta_min + (base - eta_min) * (1 + cos(pi t / T)) / 2`.
    @Test("cosine matches its formula at the points that pin it")
    func cosineMatches() {
        let schedule = LearningRateSchedule.cosine()
        #expect(abs(schedule.rate(base: base, step: 0, steps: 1000) - base) <= 1e-18)
        // Halfway is exactly half, which is the property that separates a cosine from a
        // linear decay with the same endpoints.
        #expect(abs(schedule.rate(base: base, step: 500, steps: 1000) - base / 2) <= 1e-18)
        #expect(schedule.rate(base: base, step: 1000, steps: 1000) <= 1e-20)

        let floored = LearningRateSchedule.cosine(minimum: 1e-5)
        #expect(abs(floored.rate(base: base, step: 1000, steps: 1000) - 1e-5) <= 1e-18)
    }

    @Test("polynomial with power 1 is a straight line to zero")
    func polynomialIsLinearAtPowerOne() {
        let schedule = LearningRateSchedule.polynomial()
        #expect(abs(schedule.rate(base: base, step: 250, steps: 1000) - base * 0.75) <= 1e-18)
        #expect(schedule.rate(base: base, step: 1000, steps: 1000) <= 1e-20)
    }

    /// An odd power past the budget would go negative without the clamp, and a negative
    /// learning rate ascends the loss.
    @Test("polynomial cannot go negative past the budget")
    func polynomialClamps() {
        for power in [1.0, 3.0, 0.5] {
            let rate = LearningRateSchedule.polynomial(power: power)
                .rate(base: base, step: 2000, steps: 1000)
            #expect(rate >= 0, "power \(power) gave \(rate)" as Comment)
        }
    }

    @Test("the rate depends only on the step, so a resumed run continues the schedule")
    func rateIsAFunctionOfStep() {
        // The property that matters on resume: asking for step 400 twice, or after a
        // restart, gives the same answer. A stepped scheduler object restarts at its base
        // rate unless its counter is restored, and that is silent.
        for schedule in [LearningRateSchedule.constant, .linear(), .cosine(), .polynomial()] {
            let first = schedule.rate(base: base, step: 400, steps: 1000)
            let again = schedule.rate(base: base, step: 400, steps: 1000)
            #expect(first == again)
            #expect(first != schedule.rate(base: base, step: 0, steps: 1000)
                || schedule == .constant)
        }
    }

    @Test("a zero-step budget does not divide by zero")
    func zeroStepsIsSafe() {
        #expect(LearningRateSchedule.cosine().rate(base: base, step: 0, steps: 0) == base)
    }

    // MARK: - Names

    @Test("the documented spellings parse")
    func namesParse() throws {
        #expect(try LearningRateSchedule.named("constant") == .constant)
        #expect(try LearningRateSchedule.named("linear") == .linear())
        #expect(try LearningRateSchedule.named("cosine") == .cosine())
        #expect(try LearningRateSchedule.named("polynomial") == .polynomial())
    }

    /// The two schedulers this does not implement are refused **by name**, so a config
    /// naming one says what is missing instead of silently getting a constant rate.
    @Test("unported schedulers are refused by name, not silently substituted")
    func unportedAreNamed() {
        for name in ["cosine_with_restarts", "step"] {
            #expect(throws: LearningRateSchedule.Failure.notPorted(name)) {
                try LearningRateSchedule.named(name)
            }
        }
        #expect(throws: LearningRateSchedule.Failure.unknown("cosin")) {
            try LearningRateSchedule.named("cosin")
        }
    }
}

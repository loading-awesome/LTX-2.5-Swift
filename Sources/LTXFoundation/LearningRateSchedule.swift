// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The learning-rate schedules this trainer offers, as closed forms.
///
/// These are conventionally built as mutable scheduler objects, stepped once per optimiser
/// step. The closed form is equivalent and better suited to a run
/// that resumes: `rate(at: 4000)` after a restart is the rate step 4000 would have had,
/// where a stepped object has to be replayed or its internal counter restored, and a schedule
/// that silently restarts at its initial rate on resume is a real and quiet way to ruin a
/// long run.
///
/// `cosine_with_restarts` and `step` are **not** ported. Their defaults are derived from the
/// step budget (`T_0 = steps // 4`, `step_size = steps // 2`) and their behaviour under
/// resume needs care that a schedule nothing here uses does not justify. They are refused by
/// name rather than silently mapped onto something else.
public enum LearningRateSchedule: Equatable, Sendable {

    /// The base rate, unchanged, forever.
    case constant

    /// Interpolates the *factor* from `start` to `end` over `steps`, then holds `end`.
    ///
    /// The conventional linear schedule, with its usual defaults: start 1.0, end **0.1**.
    /// The end factor being 0.1 rather than 0 is easy to miss and worth stating —
    /// this decays to a tenth of the base rate, not to nothing.
    case linear(start: Double = 1.0, end: Double = 0.1)

    /// Cosine annealing: `minimum + (base - minimum) * (1 + cos(pi * t / T)) / 2`.
    case cosine(minimum: Double = 0)

    /// Polynomial decay: `base * (1 - t / total)^power`, reaching zero at `total`.
    case polynomial(power: Double = 1.0)

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case unknown(String)
        case notPorted(String)

        public var description: String {
            switch self {
            case let .unknown(name):
                return "unknown scheduler \(name.debugDescription). Known: "
                    + "constant, linear, cosine, polynomial"
            case let .notPorted(name):
                return "scheduler \(name.debugDescription) is not supported "
                    + "— its defaults are derived from the step budget and "
                    + "its resume behaviour needs care. Use constant, linear, cosine or "
                    + "polynomial."
            }
        }
    }

    /// Parsed from a config's `scheduler_type` spelling, so a config can be copied across.
    public static func named(_ name: String) throws -> LearningRateSchedule {
        switch name {
        case "constant": return .constant
        case "linear": return .linear()
        case "cosine": return .cosine()
        case "polynomial": return .polynomial()
        case "cosine_with_restarts", "step": throw Failure.notPorted(name)
        default: throw Failure.unknown(name)
        }
    }

    /// The rate at `step`, counting from 0.
    ///
    /// Clamped at both ends. Past `steps` every schedule holds its final value rather than
    /// continuing to decay — a run extended beyond its planned budget should flatten out, not
    /// go negative, and `polynomial` with an odd power would.
    public func rate(base: Double, step: Int, steps: Int) -> Double {
        guard steps > 0 else { return base }
        let t = Double(min(max(step, 0), steps)) / Double(steps)

        switch self {
        case .constant:
            return base
        case let .linear(start, end):
            return base * (start + (end - start) * t)
        case let .cosine(minimum):
            return minimum + (base - minimum) * (1 + cos(.pi * t)) / 2
        case let .polynomial(power):
            return base * pow(1 - t, power)
        }
    }
}

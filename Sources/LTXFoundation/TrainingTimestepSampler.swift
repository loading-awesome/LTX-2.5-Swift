// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Which sigma a training example is noised to.
///
/// ## Why this one is worth the care
///
/// Every other part of a training loop fails loudly. This one does not: get the
/// distribution wrong and the loop still runs, the loss still falls, and the adapter is
/// trained against a schedule the model was never trained on. Nothing downstream notices,
/// because there is nothing downstream to notice it — the loss is computed against the
/// sampler's own choice, so a wrong choice is self-consistent.
///
/// ## The transform is a pure function; the draw is not
///
/// Three random draws are consumed per example, in this order:
///
/// ```
/// randn(batch)   the normal sample, before the shift
/// rand(batch)    the uniform fallback
/// rand(batch)    the mixing probability
/// ```
///
/// ``timestep(normal:uniform:prob:sequenceLength:)`` takes those three as arguments rather
/// than drawing them, so the transform can be checked exactly — same draws in, same sigma
/// out — while where the draws come from stays this port's own business (contract 1).
///
/// ## Arithmetic in `Float`, deliberately
///
/// The distribution is defined over float32 values with the shift folded in at double
/// precision, so every operation below is `Float` and the shift alone is computed in
/// `Double` before it narrows. Computing the whole thing in `Double` would be *more
/// precise*, and would not be this distribution.
public enum TrainingTimestepSampler {

    /// The uniform sampler. One line, but it is a selectable option in the trainer's
    /// config, so a port that skipped it would be silently wrong for anyone who chose it.
    public struct Uniform: Sendable, Equatable {
        public let minValue: Float
        public let maxValue: Float

        public init(minValue: Float = 0.0, maxValue: Float = 1.0) {
            self.minValue = minValue
            self.maxValue = maxValue
        }

        public func timestep(uniform: Float) -> Float {
            uniform * (maxValue - minValue) + minValue
        }
    }

    /// The shifted logit-normal sampler.
    ///
    /// A logit-normal whose mean is shifted by sequence length, stretched between two
    /// percentiles so it covers `[0, 1]` evenly, reflected near zero for numerical
    /// stability, and mixed with a uniform draw a tenth of the time so the distribution
    /// does not collapse at high token counts.
    public struct ShiftedLogitNormal: Sendable, Equatable {
        public let std: Float
        public let eps: Float
        public let uniformProb: Float

        /// 99.9th and 0.5th percentiles of the standard normal, scaled by `std`, carried
        /// at the precision they are written at.
        public var normal999Percentile: Double { 3.0902 * Double(std) }
        public var normal005Percentile: Double { -2.5758 * Double(std) }

        public init(std: Float = 1.0, eps: Float = 1e-3, uniformProb: Float = 0.1) {
            self.std = std
            self.eps = eps
            self.uniformProb = uniformProb
        }

        /// The sequence-length shift — a linear interpolation with **no clamp**.
        ///
        /// The absence of a clamp is the part worth stating: at 8192 tokens this returns
        /// 3.52, well past `maxShift`, and it extrapolates rather than saturating. Clamping
        /// it "for safety" would train long sequences at a different noise distribution,
        /// which is precisely the silent divergence this file exists to prevent.
        public static func shift(forSequenceLength sequenceLength: Int,
                                 minTokens: Int = 1024, maxTokens: Int = 4096,
                                 minShift: Double = 0.95, maxShift: Double = 2.05) -> Double {
            let m = (maxShift - minShift) / Double(maxTokens - minTokens)
            let b = minShift - m * Double(minTokens)
            return m * Double(sequenceLength) + b
        }

        /// One timestep from one set of draws. Pure.
        public func timestep(normal: Float, uniform: Float, prob: Float,
                             sequenceLength: Int) -> Float {
            let mu = Self.shift(forSequenceLength: sequenceLength)

            // The sum is formed at double precision and only the result narrows to float32,
            // so it is built in `Double` here too.
            let p999 = Self.sigmoid(Float(mu + normal999Percentile))
            let p005 = Self.sigmoid(Float(mu + normal005Percentile))

            let shifted = normal * std + Float(mu)
            let logitNormal = Self.sigmoid(shifted)
            let raw = (logitNormal - p005) / (p999 - p005)

            // Reflection, not a clamp: values below eps are mirrored to `2·eps - raw`, which
            // keeps them off zero without piling them onto it.
            let reflected = raw >= eps ? raw : 2 * eps - raw
            let stretched = min(max(reflected, 0), 1)

            // The uniform fallback is offset by eps for the same reason.
            let uniformValue = (1 - eps) * uniform + eps
            return prob > uniformProb ? stretched : uniformValue
        }

        /// A batch, consuming one draw of each kind per example.
        public func timesteps(normal: [Float], uniform: [Float], prob: [Float],
                              sequenceLength: Int) -> [Float] {
            precondition(normal.count == uniform.count && uniform.count == prob.count,
                         "one draw of each kind per example")
            return (0 ..< normal.count).map {
                timestep(normal: normal[$0], uniform: uniform[$0], prob: prob[$0],
                         sequenceLength: sequenceLength)
            }
        }

        /// Sigmoid in float32.
        static func sigmoid(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }
    }
}

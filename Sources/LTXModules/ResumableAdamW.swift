// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import MLXNN
import MLXOptimizers

/// AdamW that owns its moment estimates, so a run can be stopped and resumed exactly.
///
/// ## Why this exists rather than `MLXOptimizers.AdamW`
///
/// Not because the library's arithmetic is wrong — this reproduces it deliberately and a
/// test pins the two together step for step. Because the library's state cannot be read back
/// **in**. `OptimizerBase.stateStorage` is internal to `MLXOptimizers`, `AdamState`'s
/// initialisers are internal, and the only public surface is `innerState() -> [MLXArray]`,
/// which is an unlabelled read. There is no supported way to construct an optimiser holding
/// a previous run's moments, so an optimiser that survives a process boundary has to own
/// them.
///
/// What that cost bought: a resumed run continues the trajectory instead of restarting the
/// moment estimates. The difference is not cosmetic. Adam's first steps after a reset are
/// large — `m` and `v` are zero, bias correction divides by `1 - beta^1`, and the effective
/// step size is at its maximum — so resuming a converged run kicks it, and the loss curve
/// shows a step change at exactly the resume point that reads like a bad checkpoint.
///
/// ## The arithmetic is the library's, transcribed
///
/// From `Adam.applySingle` and `AdamW.applySingle`, in their order, which matters:
///
/// ```
/// parameter *= (1 - learningRate * weightDecay)     // decoupled, before the moments
/// step      += 1
/// m          = b1 * m + (1 - b1) * gradient
/// v          = b2 * v + (1 - b2) * gradient^2
/// c1         = learningRate / (1 - b1^step)
/// c2         = rsqrt(1 - b2^step)
/// parameter -= (c1 * m) / (sqrt(v) * c2 + eps)
/// ```
///
/// Two details are transcribed rather than rederived, because both are silent when wrong:
///
/// * **The weight decay multiplies the parameter, not the gradient.** That is what
///   "decoupled" means, and an L2 form — adding `weightDecay * parameter` to the gradient —
///   trains, converges, and follows a different path. `TrainingBackwardGateTests` needs
///   three steps to separate them, because at step 1 they nearly agree.
/// * **`eps` is added after `sqrt(v)` is scaled by `c2`**, not to `v` and not before. The
///   library writes `sqrt(v) * c2 + eps`; `sqrt(v * c2 * c2 + eps)` is a different function
///   of the same numbers.
///
/// ``biasCorrection`` defaults to **true** here, the opposite of the library's default and
/// the same value the trainer was already passing explicitly. The default is inverted on
/// purpose: mlx-swift follows the original Adam paper and omits the correction, while the
/// AdamW that training recipes are written against applies it. Uncorrected, the first
/// handful of updates are several times larger than the configured learning rate implies —
/// `m` and `v` are still biased towards their zero initialisation — and the run leaves the
/// trajectory it was configured for with nothing else looking wrong.
public final class ResumableAdamW: Evaluatable, Updatable {

    /// Per-parameter first and second moment estimates.
    public struct Moments {
        public var m: MLXArray
        public var v: MLXArray

        public init(m: MLXArray, v: MLXArray) {
            self.m = m
            self.v = v
        }
    }

    public var learningRate: Float
    public let betas: (Float, Float)
    public let eps: Float
    public let weightDecay: Float
    public let biasCorrection: Bool

    /// Updates applied so far. Drives bias correction, and is the reason a resumed run needs
    /// more than the moments: restoring `m` and `v` with `step` back at 0 divides them by
    /// `1 - beta^1` and produces the same kick a full reset would.
    public private(set) var step: Int

    /// Keyed by the module's flattened parameter path — `a.0`, `b.7`.
    private var moments: [String: Moments]

    public enum Failure: Error, CustomStringConvertible {
        case unknownParameter(String, known: [String])
        case missingTensor(String)
        case shapeMismatch(String, state: [Int], parameter: [Int])
        case partialGradients(missing: [String])

        public var description: String {
            switch self {
            case let .unknownParameter(key, known):
                return "the optimiser state names \(key), which this adapter does not have. "
                    + "It was saved for a different target set. Known: \(known.prefix(3))…"
            case let .missingTensor(name):
                return "the optimiser state is missing \(name)"
            case let .shapeMismatch(key, state, parameter):
                return "the optimiser state for \(key) is \(state), the parameter is "
                    + "\(parameter). Resuming would apply another run's moments."
            case let .partialGradients(missing):
                return "no gradient for \(missing.prefix(3))… — this optimiser keeps one "
                    + "step counter for the whole adapter, which is only equivalent to a "
                    + "per-parameter counter while every parameter is updated every step"
            }
        }
    }

    public init(learningRate: Float,
                betas: (Float, Float) = (0.9, 0.999),
                eps: Float = 1e-8,
                weightDecay: Float = 0.01,
                biasCorrection: Bool = true,
                step: Int = 0,
                moments: [String: Moments] = [:]) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
        self.biasCorrection = biasCorrection
        self.step = step
        self.moments = moments
    }

    // MARK: - The step

    /// Apply `gradients` to `module`'s parameters, advancing the moments.
    ///
    /// Mirrors `Optimizer.update(model:gradients:)`: the new parameters are written back
    /// through `Module.update(parameters:)` rather than assigned, so the module's own
    /// bookkeeping stays consistent.
    public func update(module: LoRAAdapterModule, gradients: ModuleParameters) throws {
        let parameters = module.parameters().flattened()
        let gradientsByKey = Dictionary(uniqueKeysWithValues: gradients.flattened())

        // One counter for the whole adapter is only the same thing as the library's
        // per-parameter counter while every parameter is stepped every time. That holds for
        // this trainer — the gradient tree comes from differentiating a closure over the
        // whole module — and is checked rather than assumed, because a partial gradient set
        // would leave some parameters bias-corrected against the wrong step number.
        let missing = parameters.map(\.0).filter { gradientsByKey[$0] == nil }
        guard missing.isEmpty else { throw Failure.partialGradients(missing: missing) }

        step += 1
        let (b1, b2) = betas
        // Computed once per step rather than per parameter: they depend only on the counter.
        let c1 = learningRate / (1 - pow(b1, Float(step)))
        let c2 = 1 / (1 - pow(b2, Float(step))).squareRoot()

        var updated: [(String, MLXArray)] = []
        updated.reserveCapacity(parameters.count)

        for (key, parameter) in parameters {
            let gradient = gradientsByKey[key]!
            var state = moments[key] ?? Moments(m: MLXArray.zeros(like: parameter),
                                                v: MLXArray.zeros(like: parameter))

            // Decoupled weight decay: on the parameter, before the moments are touched.
            let decayed = weightDecay == 0
                ? parameter
                : parameter * (1 - learningRate * weightDecay)

            state.m = b1 * state.m + (1 - b1) * gradient
            state.v = b2 * state.v + (1 - b2) * MLX.square(gradient)

            let delta: MLXArray
            if biasCorrection {
                delta = (c1 * state.m) / (MLX.sqrt(state.v) * c2 + eps)
            } else {
                delta = learningRate * state.m / (MLX.sqrt(state.v) + eps)
            }

            moments[key] = state
            updated.append((key, decayed - delta))
        }

        module.update(parameters: ModuleParameters.unflattened(updated))
    }

    /// The arrays this optimiser owns, for ``MLX/eval(_:)``.
    ///
    /// Without this the moments stay lazy and the graph grows across steps — the same reason
    /// the trainer evaluates the module and the optimiser together after every update.
    public func innerState() -> [MLXArray] {
        moments.keys.sorted().flatMap { [moments[$0]!.m, moments[$0]!.v] }
    }

    // MARK: - Persistence

    /// Suffixes, in torch's own naming, so a reader who has seen a `.pt` optimiser file
    /// recognises what these are.
    public static let firstMomentSuffix = ".exp_avg"
    public static let secondMomentSuffix = ".exp_avg_sq"

    /// The state, keyed by the **adapter's** own weight names rather than by parameter path.
    ///
    /// `a.0` is a position in a module; `…attn1.to_q.lora_A.exp_avg` is an identity. Saving
    /// positions would silently pair one run's moments with another run's targets the moment
    /// the target set or block range changed — the shapes would often still match, and Adam
    /// would carry the wrong history for every module in the adapter. Naming them means
    /// ``load(_:for:)`` can refuse.
    public func stateDict(for module: LoRAAdapterModule) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, state) in moments {
            let name = try Self.tensorStem(parameterPath: key, module: module)
            out[name + Self.firstMomentSuffix] = state.m
            out[name + Self.secondMomentSuffix] = state.v
        }
        return out
    }

    /// `a.3` -> the adapter's third key with the `lora_A` factor, in ``TrainableLoRA``'s
    /// on-disk spelling.
    static func tensorStem(parameterPath: String, module: LoRAAdapterModule) throws -> String {
        let parts = parameterPath.split(separator: ".")
        guard parts.count == 2, let index = Int(parts[1]), index < module.keys.count,
              parts[0] == "a" || parts[0] == "b" else {
            throw Failure.unknownParameter(parameterPath, known: module.keys)
        }
        let factor = parts[0] == "a" ? "lora_A" : "lora_B"
        return module.keys[index] + "." + factor
    }

    /// Rebuild the moments for `module` from a saved state dict.
    ///
    /// Every parameter must be present and shaped correctly. A partial restore is refused
    /// rather than filled with zeros: a mix of restored and fresh moments is the one outcome
    /// that looks like it worked and trains along a path neither run would have taken.
    public static func load(_ tensors: [String: MLXArray],
                            for module: LoRAAdapterModule) throws -> [String: Moments] {
        var out: [String: Moments] = [:]
        for (key, parameter) in module.parameters().flattened() {
            let stem = try tensorStem(parameterPath: key, module: module)
            guard let m = tensors[stem + firstMomentSuffix] else {
                throw Failure.missingTensor(stem + firstMomentSuffix)
            }
            guard let v = tensors[stem + secondMomentSuffix] else {
                throw Failure.missingTensor(stem + secondMomentSuffix)
            }
            guard m.shape == parameter.shape else {
                throw Failure.shapeMismatch(stem, state: m.shape, parameter: parameter.shape)
            }
            guard v.shape == parameter.shape else {
                throw Failure.shapeMismatch(stem, state: v.shape, parameter: parameter.shape)
            }
            out[key] = Moments(m: m, v: v)
        }
        return out
    }
}

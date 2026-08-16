// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import MLXNN
import MLXOptimizers

/// The LoRA parameters as an `MLXNN.Module`, so the library's optimisers can drive them.
///
/// ``TrainableLoRA`` owns the *convention* — the initialisation, the `alpha/rank` scaling,
/// the on-disk key form. This owns the *plumbing*: MLX's optimisers take
/// `(model: Module, gradients: ModuleParameters)`, and `MLXNN.valueAndGrad`
/// returns gradients already shaped as that tree. Going through `Module` therefore replaces
/// the flat-array path and its `argumentNumbers` hazard with something the library keeps
/// consistent on its own.
///
/// Only the factors are parameters. The base transformer is **not** a submodule and is never
/// registered, which is what keeps the optimiser's state proportional to the adapter rather
/// than to 22 B frozen weights.
public final class LoRAAdapterModule: Module {

    /// `[rank, in]` per target, in ``keys`` order.
    public var a: [MLXArray]
    /// `[out, rank]` per target, in ``keys`` order.
    public var b: [MLXArray]

    public let keys: [String]
    public let scaling: Float
    public let rank: Int
    public let alpha: Int

    public init(_ lora: TrainableLoRA) {
        self.keys = lora.keys
        self.scaling = lora.scaling
        self.rank = lora.rank
        self.alpha = lora.alpha
        self.a = lora.factors.map(\.a)
        self.b = lora.factors.map(\.b)
        super.init()
    }

    /// The delta for one target's linear output.
    public func delta(input: MLXArray, at index: Int) -> MLXArray {
        TrainableLoRA.delta(input: input, a: a[index], b: b[index], scaling: scaling)
    }

    /// Back to the value type, for saving through ``TensorFile``.
    public func snapshot() -> TrainableLoRA {
        TrainableLoRA(keys: keys,
                      factors: zip(a, b).map { TrainableLoRA.Factors(a: $0, b: $1) },
                      rank: rank, alpha: alpha)
    }

    /// Groups these factors by transformer block, for ``DiTForward/checkpointing``.
    ///
    /// Call this on the model ``MLXNN/valueAndGrad(model:)`` hands to the traced closure, not
    /// on the instance outside it. The arrays must be the traced ones: the whole point of the
    /// two closures is that the factors cross the checkpoint boundary as inputs, and factors
    /// captured from a stale instance get no gradient and report no error.
    ///
    /// Targets whose key carries no block index are skipped. On the released-adapter target
    /// set there are none — all 480 sit in blocks — but the default suffix list also matches
    /// connectors outside them, and those cannot be checkpointed per block.
    public func blockScopedCheckpointing() -> DiTForward.Checkpointing {
        var byBlock: [Int: [Int]] = [:]
        for (index, key) in keys.enumerated() {
            guard let block = DiTLoRATargets.blockIndex(in: key) else { continue }
            byBlock[block, default: []].append(index)
        }
        // Both closures walk `byBlock[block]` in the same order, which is what keeps them in
        // agreement. A mismatch would pair a key with another module's factors — wrong, and
        // not obviously so, since the shapes still line up within a block.
        return DiTForward.Checkpointing(
            parameters: { block in
                (byBlock[block] ?? []).flatMap { [self.a[$0], self.b[$0]] }
            },
            overlay: { block, arrays in
                let indices = byBlock[block] ?? []
                guard !indices.isEmpty else { return nil }
                return LoRAOverlay.trainable(
                    indices.enumerated().map { offset, index in
                        (baseKey: self.keys[index],
                         a: arrays[2 * offset], b: arrays[2 * offset + 1])
                    },
                    scaling: self.scaling)
            })
    }
}

/// One optimiser step of LoRA training, composed from the types that own the rules.
///
/// ## What this type owns
///
/// The **sigma** comes from ``LTXFoundation/TrainingTimestepSampler`` and the **noising,
/// target and loss** from ``LTXFoundation/TrainingObjective``. What this type adds is
/// composition and the optimiser call — so a mistake here is a mistake in *wiring*, which a
/// loss that fails to fall will show, rather than a mistake in the objective, which it
/// would not.
///
/// The forward is supplied by the caller rather than reached for. That is not indirection
/// for its own sake: it lets the whole step be exercised against a small stand-in, which is
/// the only way to test this without a 42 GB transformer and the activation memory to
/// differentiate through it.
public enum LoRATrainingStep {

    /// `(module, noisyLatents, perTokenTimesteps) -> prediction`.
    ///
    /// The module is passed in rather than captured so `MLXNN.valueAndGrad` can substitute
    /// the parameters it is differentiating; capturing the outer instance would silently
    /// differentiate a stale copy and yield zero gradients.
    public typealias Forward = (LoRAAdapterModule, MLXArray, MLXArray) -> MLXArray

    /// One example's worth of pre-encoded, pre-patchified latents.
    public struct Batch {
        /// `[batch, sequence, channels]`, clean, straight from the cache.
        public let latents: MLXArray
        /// True where a token is conditioning — kept clean and excluded from the loss.
        public let conditioningMask: [Bool]
        public let shape: TrainingObjective.Shape

        public init(latents: MLXArray, conditioningMask: [Bool],
                    shape: TrainingObjective.Shape) {
            self.latents = latents
            self.conditioningMask = conditioningMask
            self.shape = shape
        }

        public static func unconditioned(latents: MLXArray,
                                         shape: TrainingObjective.Shape) -> Batch {
            Batch(latents: latents,
                  conditioningMask: [Bool](repeating: false, count: shape.tokens),
                  shape: shape)
        }
    }

    public struct Result {
        /// Per-example loss, unreduced. The trainer reduces; keeping it unreduced here is
        /// what lets loss be tracked per sigma bucket.
        public let perExampleLoss: [Float]
        public let mean: Float
        /// The sigma each example was noised to, for that bucketing.
        public let sigmas: [Float]
    }

    /// Draw sigmas for a batch from this repo's own randomness.
    ///
    /// Only the transform is fixed; the stream itself is ours, and contract 1 is why it has
    /// to be. A seeded, reproducible draw is therefore the most that can be asked of this —
    /// and it is asked, because a training run that cannot be repeated cannot be debugged.
    ///
    /// - Parameter key: an explicit PRNG key. See ``gradients(module:batch:sampler:sigmas:key:forward:)``
    ///   for why one is worth passing; `nil` draws from the global stream, unchanged.
    public static func drawSigmas(sampler: TrainingTimestepSampler.ShiftedLogitNormal,
                                  batch: Int, sequenceLength: Int,
                                  key: MLXArray? = nil) -> [Float] {
        // Three draws, three **sub**keys. Handing one key to three calls is not "seeded",
        // it is the same stream three times — `normal(key: k)` and `uniform(key: k)` are
        // deterministic functions of `k`, so the uniform and the probability draw would be
        // rigidly coupled to each other for every batch this ever trains on. The split is
        // the whole difference between a key and a seed.
        let keys = key.map { MLX.split(key: $0, into: 3) }
        let normal = MLX.normal([batch], key: keys?[0]).asArray(Float.self)
        let uniform = MLX.uniform(low: 0, high: 1, [batch], key: keys?[1]).asArray(Float.self)
        let prob = MLX.uniform(low: 0, high: 1, [batch], key: keys?[2]).asArray(Float.self)
        return sampler.timesteps(normal: normal, uniform: uniform, prob: prob,
                                 sequenceLength: sequenceLength)
    }

    /// The loss and gradients for one batch, **without** touching the parameters.
    ///
    /// Split out from ``step(module:optimizer:batch:sampler:sigmas:forward:)`` so a caller can
    /// accumulate over several batches before updating. Gradient accumulation is not optional
    /// here: ``DiTTrainingForward`` runs one example per step because `DiTForward.Sampling`
    /// has no batch axis, so accumulation is the only route to an effective batch above 1.
    public struct Gradients {
        public let value: ModuleParameters
        public let perExampleLoss: [Float]
        public let mean: Float
        public let sigmas: [Float]
    }

    /// Take one step: noise, predict, score, differentiate, update.
    ///
    /// Returns the loss **before** the update, which is the value that corresponds to the
    /// gradients just applied. Reporting the post-update loss would understate it by exactly
    /// one step and make a learning-rate sweep read better than it is.
    @discardableResult
    public static func step(module: LoRAAdapterModule,
                            optimizer: MLXOptimizers.Optimizer,
                            batch: Batch,
                            sampler: TrainingTimestepSampler.ShiftedLogitNormal
                                = .init(),
                            sigmas: [Float]? = nil,
                            key: MLXArray? = nil,
                            forward: @escaping Forward) -> Result {
        let computed = gradients(module: module, batch: batch, sampler: sampler,
                                 sigmas: sigmas, key: key, forward: forward)
        optimizer.update(model: module, gradients: computed.value)
        MLX.eval(module, optimizer)
        return Result(perExampleLoss: computed.perExampleLoss,
                      mean: computed.mean,
                      sigmas: computed.sigmas)
    }

    /// ``step(module:optimizer:batch:sampler:sigmas:key:forward:)`` without the update.
    ///
    /// - Parameter key: an explicit PRNG key for this step's sigma and noise draws.
    ///
    ///   **`nil` keeps the global stream, and that is the historical behaviour rather than
    ///   the good one.** `MLX.normal(shape)` with no key advances a process-wide generator,
    ///   so this step's answer depends on everything else that has drawn from it since the
    ///   last `MLX.seed`. In a training loop on one thread that is invisible and harmless.
    ///   Two places it is neither:
    ///
    ///   * **Resume.** ``LTXPipeline/LoRATrainer`` seeds once and lets the loop advance the
    ///     stream, so step 5 of a fresh run and step 5 of a run resumed at 5 see *different*
    ///     noise — the schedule resumes exactly and the noise does not. A key derived from
    ///     the step index makes both reproduce the same draw, which is what the trainer's
    ///     own "a run that cannot be repeated cannot be debugged" asks for.
    ///   * **Parallel tests.** Two identically-seeded calls agree only if nothing else draws
    ///     between them. `DiTTrainingForwardTests.checkpointingMatchesThroughTheStep`
    ///     measured that as an "exactness" failure varying between 0.011 and 3.48 across
    ///     runs while being bit-exact alone — a difference in the *inputs*, reported as a
    ///     difference in checkpointing.
    ///
    ///   Defaulted to `nil` rather than made mandatory because every existing caller and
    ///   test in this repository draws through the global path, and a key would change the
    ///   draws. Passing one is the improvement; the default is the compatibility.
    public static func gradients(module: LoRAAdapterModule,
                                 batch: Batch,
                                 sampler: TrainingTimestepSampler.ShiftedLogitNormal
                                     = .init(),
                                 sigmas: [Float]? = nil,
                                 key: MLXArray? = nil,
                                 forward: @escaping Forward) -> Gradients {
        let shape = batch.shape
        // Two independent subkeys: the sigmas and the noise must not be functions of each
        // other. Split before either is used, so that supplying `sigmas` explicitly does
        // not change which key the noise draw gets — otherwise a run with fixed sigmas and
        // a run without would disagree on the noise for the same key.
        let (sigmaKey, noiseKey) = key.map { k -> (MLXArray?, MLXArray?) in
            let parts = MLX.split(key: k, into: 2)
            return (parts[0], parts[1])
        } ?? (nil, nil)

        let drawn = sigmas ?? drawSigmas(sampler: sampler, batch: shape.batch,
                                         sequenceLength: shape.sequence, key: sigmaKey)

        // Noise and target, both from `TrainingObjective`'s rules. Computed once, outside
        // the differentiated closure: they are constants of this step, and putting them
        // inside would make the graph re-derive them on every backward for no reason.
        let clean = batch.latents.asArray(Float.self)
        let noiseArray = MLX.normal(batch.latents.shape, key: noiseKey).asType(.float32)
        let noise = noiseArray.asArray(Float.self)

        let noisyFlat = TrainingObjective.noised(clean: clean, noise: noise, sigmas: drawn,
                                                 conditioningMask: batch.conditioningMask,
                                                 shape: shape)
        let targetFlat = TrainingObjective.velocityTarget(clean: clean, noise: noise)
        let timestepsFlat = TrainingObjective.perTokenTimesteps(
            sigmas: drawn, conditioningMask: batch.conditioningMask,
            batch: shape.batch, sequence: shape.sequence)

        let noisy = MLXArray(noisyFlat, [shape.batch, shape.sequence, shape.channels])
        let target = MLXArray(targetFlat, [shape.batch, shape.sequence, shape.channels])
        let timesteps = MLXArray(timestepsFlat, [shape.batch, shape.sequence])
        let lossMask = batch.conditioningMask.map { !$0 }

        // The loss must be a single MLXArray for the gradient, so the per-example reduction
        // is done in tensor space here and re-derived on the host below for reporting. Both
        // follow `TrainingObjective.videoLoss`, and `LoRATrainingStepTests` checks the two
        // against each other.
        let maskArray = MLXArray(lossMask.map { $0 ? Float(1) : Float(0) },
                                 [shape.batch, shape.sequence, 1])
        let elements = Float(shape.sequence * shape.channels)

        // The per-example vector is captured out of the closure rather than recomputed:
        // the gradient needs a scalar, but the loss is per example and the trainer buckets
        // by sigma with it. Recomputing would mean a second forward.
        var perExample: MLXArray?
        let valueAndGrad = MLXNN.valueAndGrad(model: module) {
            (model: LoRAAdapterModule, arrays: [MLXArray]) -> [MLXArray] in
            let prediction = forward(model, arrays[0], arrays[1])
            let error = prediction - target
            let masked = (error * error) * maskArray
            let numerator = masked.sum(axes: [1, 2]) / elements
            let denominator = MLX.maximum(maskArray.sum(axes: [1, 2]) / Float(shape.sequence),
                                          MLXArray(Float(1e-8)))
            let vector = numerator / denominator
            perExample = vector
            return [vector.sum()]
        }

        // Differentiated on a deep stack. `mlx::core::vjp` recurses over the graph, and under
        // gradient checkpointing each checkpointed segment's VJP is another `vjp` call — 48
        // blocks nest 48 traversals, which overflows the 512 KB stack of a `Task` or a
        // `swift-testing` body and lands as SIGBUS with an unreadable backtrace. See
        // ``DeepStack``. It costs one thread hop per step against a step measured in seconds,
        // and it means callers cannot get this wrong by scheduling training on the wrong
        // executor.
        let (losses, gradients) = DeepStack.run {
            let output = valueAndGrad(module, [noisy, timesteps])
            // Evaluated BEFORE the update, so the reported loss is the one these gradients
            // were computed from. Reporting after would understate it by exactly one step and
            // make any learning-rate sweep read better than it is. Evaluated *here*, on the
            // deep stack, because eval is what forces the backward to actually run.
            MLX.eval(output.0, output.1)
            return output
        }
        let vector = perExample.map { array -> [Float] in
            MLX.eval(array)
            return array.asArray(Float.self)
        } ?? []
        let total = losses[0].item(Float.self)

        return Gradients(value: gradients,
                         perExampleLoss: vector,
                         mean: total / Float(shape.batch),
                         sigmas: drawn)
    }
}

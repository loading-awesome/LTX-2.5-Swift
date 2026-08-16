// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX
import MLXOptimizers
import MLXNN

/// The outer loop: sample, accumulate, clip, step, schedule, snapshot.
///
/// Everything below it is already measured — the objective, the gradient against the real
/// transformer, the memory against a sweep. What this adds is the part that runs for hours
/// unattended, so its failures are the ones that waste a day rather than a second: a
/// schedule that restarts on resume, a snapshot that cannot be loaded back, an order that
/// is not reproducible from its seed.
public struct LoRATrainer {

    public struct Configuration: Sendable {
        /// Optimiser steps, not samples seen. With accumulation, one step consumes
        /// ``gradientAccumulationSteps`` samples.
        public var steps: Int
        public var learningRate: Double
        public var schedule: LearningRateSchedule
        /// Samples per optimiser step. ``DiTTrainingForward`` runs one example at a time —
        /// `DiTForward.Sampling` has no batch axis — so this is the only way to a larger
        /// effective batch.
        public var gradientAccumulationSteps: Int
        /// The gradient-norm clip; 0 or less disables clipping.
        public var maxGradientNorm: Float
        /// AdamW's decoupled weight decay, at the usual default.
        public var weightDecay: Float
        public var rank: Int
        public var alpha: Int
        /// Transformer blocks to adapt. Nil is all 48.
        public var blocks: Int?
        public var gradientCheckpointing: Bool
        /// Drives the sample order and the noise. A run that cannot be repeated cannot be
        /// debugged.
        public var seed: UInt64
        /// Write an adapter every N steps. Nil disables intermediate snapshots.
        public var snapshotInterval: Int?
        public var outputDirectory: URL

        public init(steps: Int,
                    learningRate: Double = 1e-4,
                    schedule: LearningRateSchedule = .constant,
                    gradientAccumulationSteps: Int = 1,
                    maxGradientNorm: Float = 1.0,
                    weightDecay: Float = 0.01,
                    rank: Int = 64,
                    alpha: Int = 64,
                    blocks: Int? = nil,
                    gradientCheckpointing: Bool = true,
                    seed: UInt64 = 42,
                    snapshotInterval: Int? = 250,
                    outputDirectory: URL) {
            self.steps = steps
            self.learningRate = learningRate
            self.schedule = schedule
            self.gradientAccumulationSteps = gradientAccumulationSteps
            self.maxGradientNorm = maxGradientNorm
            self.weightDecay = weightDecay
            self.rank = rank
            self.alpha = alpha
            self.blocks = blocks
            self.gradientCheckpointing = gradientCheckpointing
            self.seed = seed
            self.snapshotInterval = snapshotInterval
            self.outputDirectory = outputDirectory
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case badStepCount(Int)
        case badAccumulation(Int)
        case resumeBeyondBudget(resumed: Int, steps: Int)

        public var description: String {
            switch self {
            case let .badStepCount(steps):
                return "steps must be positive, got \(steps)"
            case let .badAccumulation(count):
                return "gradientAccumulationSteps must be positive, got \(count)"
            case let .resumeBeyondBudget(resumed, steps):
                return "resuming at step \(resumed) of a \(steps)-step run: nothing to do. "
                    + "Raise `steps` to continue training this adapter."
            }
        }
    }

    /// What one optimiser step did. Handed to ``Progress`` as it happens rather than
    /// accumulated, so a long run reports as it goes.
    public struct StepReport: Sendable {
        public let step: Int
        public let loss: Float
        public let learningRate: Double
        /// Pre-clip gradient norm. Clipping and moving on would hide it; reporting the norm
        /// is what makes "the clip is doing everything" visible rather than inferred from a
        /// loss that will not move.
        public let gradientNorm: Float
        public let seconds: Double
        public let samples: [String]
    }

    public typealias Progress = (StepReport) -> Void

    public var configuration: Configuration
    public var dataset: TrainingDataset

    public init(configuration: Configuration, dataset: TrainingDataset) {
        self.configuration = configuration
        self.dataset = dataset
    }

    /// The order samples are visited in, for a given epoch.
    ///
    /// A seeded shuffle rather than the manifest's order: repeatedly walking a sorted list
    /// pairs the same clips with the same schedule position on every epoch. Derived from the
    /// seed **and the epoch**, so the order differs between epochs and is still reproducible.
    ///
    /// Deliberately not `MLXRandom`: this must not consume the same stream the noise does, or
    /// changing the dataset size would change every noise draw in the run.
    public static func order(entries: [String], seed: UInt64, epoch: Int) -> [String] {
        var state = seed &* 0x9E37_79B9_7F4A_7C15 &+ UInt64(bitPattern: Int64(epoch)) &+ 1
        func next() -> UInt64 {
            // SplitMix64. Small, exactly specified, and identical on every platform — which
            // is the whole requirement for a shuffle that has to be reproducible.
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        var out = entries
        guard out.count > 1 else { return out }
        for i in stride(from: out.count - 1, to: 0, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            out.swapAt(i, j)
        }
        return out
    }

    /// The PRNG key a given step draws its sigmas and noise from.
    ///
    /// A pure function of `(seed, step)`, which is the whole point: it makes step N's noise
    /// addressable rather than positional. `MLX.key` is a counter-based generator, so
    /// distinct step indices give independent streams without any of them having to be
    /// *reached* by advancing through the ones before.
    ///
    /// `&+` rather than `+` so a seed near `UInt64.max` wraps instead of trapping — a seed is
    /// an opaque label and there is no reason for one value of it to crash a run.
    ///
    /// Deliberately not the sample order's generator: ``order(entries:seed:epoch:)`` says why
    /// it is SplitMix64 and not `MLXRandom`, and the same argument applies in reverse here.
    /// Two independent streams from one seed, neither able to shift the other.
    public static func stepKey(seed: UInt64, step: Int) -> MLXArray {
        MLX.key(seed &+ UInt64(bitPattern: Int64(step)))
    }

    /// Where a resumed run picks up: the adapter, the step, and the optimiser's moments.
    ///
    /// The moments are the part that is easy to leave out and expensive to leave out.
    /// Without them AdamW restarts from `m = v = 0` with its bias-correction counter at
    /// zero, which is not a neutral state — it is the state with the *largest* effective
    /// step size, so resuming a converged run kicks it. The loss shows a step change at
    /// exactly the resume point, which reads like a damaged checkpoint.
    ///
    /// `nil` is still accepted, because a snapshot written before the optimiser state
    /// existed has no moments to offer and refusing it would strand those adapters. What it
    /// must not do is pretend: a resume without moments restarts the optimiser's step
    /// counter at 0 rather than at ``step``, so the run is honestly a fresh optimiser on a
    /// trained adapter rather than a continuation wearing a continuation's step numbers.
    public struct Resume {
        public var adapter: TrainableLoRA
        public var step: Int
        /// As written by ``snapshot(_:optimizer:step:)``, keyed by the adapter's own weight
        /// names. Loaded through ``LTXModules/ResumableAdamW/load(_:for:)``, which refuses a
        /// state whose keys or shapes disagree with this adapter.
        public var optimizerState: [String: MLXArray]?

        public init(adapter: TrainableLoRA, step: Int,
                    optimizerState: [String: MLXArray]? = nil) {
            self.adapter = adapter
            self.step = step
            self.optimizerState = optimizerState
        }
    }

    /// Train, returning the adapter.
    ///
    /// - Parameters:
    ///   - base: the loaded transformer, shared across every step. Loading it per step would
    ///     dominate everything else.
    ///   - resumeFrom: an adapter to continue, the step it stopped at, and the optimiser
    ///     moments to continue with.
    public func run(base: DiTForward,
                    head: DiTOutputHead,
                    header: SafetensorsHeader,
                    resumeFrom: Resume? = nil,
                    progress: Progress? = nil) throws -> TrainableLoRA {
        guard configuration.steps > 0 else {
            throw Failure.badStepCount(configuration.steps)
        }
        guard configuration.gradientAccumulationSteps > 0 else {
            throw Failure.badAccumulation(configuration.gradientAccumulationSteps)
        }
        let startStep = resumeFrom?.step ?? 0
        guard startStep < configuration.steps else {
            throw Failure.resumeBeyondBudget(resumed: startStep, steps: configuration.steps)
        }

        // Seeds the **initialisation** below, and nothing in the loop. Every draw the steps
        // make now comes from an explicit key derived from the step index — see
        // `stepKey(_:)` — so this line stops being load-bearing for the trajectory and stays
        // only because `makeLoRA` initialises `A` from the global stream.
        MLXRandom.seed(configuration.seed &+ UInt64(startStep))

        let lora = resumeFrom?.adapter ?? DiTLoRATargets.makeLoRA(
            header: header, rank: configuration.rank, alpha: configuration.alpha,
            suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly,
            blocks: configuration.blocks.map { 0 ..< $0 })
        let module = LoRAAdapterModule(lora)

        // `ResumableAdamW`, not `MLXOptimizers.AdamW`, and the reason is resume rather than
        // arithmetic — the two compute the same step. The library's moments cannot be
        // written back: `OptimizerBase.stateStorage` and `AdamState`'s initialisers are
        // internal to `MLXOptimizers`, so its state is readable and not restorable.
        //
        // `biasCorrection: true` is still stated rather than defaulted, because it is the
        // value that matters and it is the opposite of the library's default: mlx-swift
        // follows the original Adam paper, while the AdamW this objective was tuned against
        // applies the correction. At the default the parameters are 0.108 out on the first
        // step while the loss falls the whole way and nothing looks wrong.
        //
        // The step counter resumes only when the moments do. Restoring `m` and `v` while
        // leaving the counter at 0 divides them by `1 - beta^1` and produces the same kick a
        // full reset would; starting the counter at `startStep` with zero moments produces
        // the opposite error, a first step far smaller than either run would take.
        var moments: [String: ResumableAdamW.Moments] = [:]
        var optimizerStep = 0
        if let state = resumeFrom?.optimizerState {
            moments = try ResumableAdamW.load(state, for: module)
            optimizerStep = startStep
        }
        let optimizer = ResumableAdamW(learningRate: Float(configuration.learningRate),
                                       weightDecay: configuration.weightDecay,
                                       biasCorrection: true,
                                       step: optimizerStep,
                                       moments: moments)

        try FileManager.default.createDirectory(at: configuration.outputDirectory,
                                                withIntermediateDirectories: true)

        var consumed = startStep * configuration.gradientAccumulationSteps
        for step in startStep ..< configuration.steps {
            let started = Date()

            // The rate is recomputed from the step index, never advanced. That is what makes
            // a resumed run pick the schedule up where it left off instead of restarting it.
            let rate = configuration.schedule.rate(base: configuration.learningRate,
                                                   step: step, steps: configuration.steps)
            optimizer.learningRate = Float(rate)

            var accumulated: ModuleParameters?
            var lossSum: Float = 0
            var names: [String] = []

            // One key per micro-batch, derived from the **step index** rather than taken
            // from a stream the loop advances. This is the noise's version of what the
            // learning rate already does two lines up: recomputed from `step`, never
            // advanced, so a resumed run sees what a fresh run saw at the same step.
            //
            // Before this, resume silently disagreed. `MLXRandom.seed(seed + startStep)` and
            // then N steps of stream advance is not the same stream position as `seed(seed)`
            // and then `startStep + N` steps of it, so a run resumed at step 500 trained on
            // different noise than an uninterrupted run did at step 500 — with the schedule,
            // the sample order and the optimiser state all resuming exactly, which is what
            // makes the disagreement hard to see.
            let microKeys = MLX.split(key: Self.stepKey(seed: configuration.seed, step: step),
                                      into: configuration.gradientAccumulationSteps)

            for micro in 0 ..< configuration.gradientAccumulationSteps {
                let epoch = consumed / max(dataset.entries.count, 1)
                let position = consumed % max(dataset.entries.count, 1)
                let name = Self.order(entries: dataset.entries,
                                      seed: configuration.seed, epoch: epoch)[position]
                consumed += 1
                names.append(name)

                let sample = try dataset.sample(name)
                let forward = try dataset.forward(
                    for: sample, base: base, head: head,
                    gradientCheckpointing: configuration.gradientCheckpointing,
                    blocks: configuration.blocks)

                let computed = LoRATrainingStep.gradients(
                    module: module, batch: sample.batch, key: microKeys[micro],
                    forward: forward())
                lossSum += computed.mean
                accumulated = accumulated.map { sum in
                    // Summed, then divided once below. Averaging incrementally would weight
                    // the earliest sample most.
                    sum.mapValues(computed.value) { $0 + ($1 ?? MLXArray(0)) }
                } ?? computed.value
            }

            var gradients = accumulated!
            if configuration.gradientAccumulationSteps > 1 {
                let scale = Float(1) / Float(configuration.gradientAccumulationSteps)
                gradients = gradients.mapValues { $0 * scale }
            }

            let norm: Float
            if configuration.maxGradientNorm > 0 {
                let (clipped, total) = clipGradNorm(gradients: gradients,
                                                    maxNorm: configuration.maxGradientNorm)
                gradients = clipped
                norm = total.item(Float.self)
            } else {
                norm = Float.nan
            }

            try optimizer.update(module: module, gradients: gradients)
            MLX.eval(module, optimizer)

            progress?(StepReport(
                step: step,
                loss: lossSum / Float(configuration.gradientAccumulationSteps),
                learningRate: rate, gradientNorm: norm,
                seconds: -started.timeIntervalSinceNow, samples: names))

            if let interval = configuration.snapshotInterval,
               interval > 0, (step + 1) % interval == 0, step + 1 < configuration.steps {
                try snapshot(module.snapshot(), optimizer: (optimizer, module),
                             step: step + 1)
            }
            MLX.Memory.clearCache()
        }

        let final = module.snapshot()
        try snapshot(final, optimizer: (optimizer, module), step: configuration.steps)
        return final
    }

    /// The file name an optimiser snapshot takes for a step.
    public static func optimizerFilename(step: Int) -> String {
        String(format: "optimizer-%06d.safetensors", step)
    }

    /// Adapter file for a step, in the released adapters' key convention — and, beside it,
    /// the optimiser's moments.
    ///
    /// Written through ``TensorFile`` so it is byte-stable, and named by step so a run leaves
    /// a trail rather than one file whose provenance is a modification time.
    ///
    /// **Two files, not one.** The adapter is the deliverable: it is what a render loads,
    /// what gets published, and it must stay loadable by everything that reads an adapter.
    /// The moments are twice its size, are useless to anything but this trainer, and would
    /// have to be stripped before shipping. Folding them into the same file would make every
    /// consumer's key set depend on whether the adapter came out of a resumable run.
    public func snapshot(_ adapter: TrainableLoRA,
                         optimizer: (ResumableAdamW, LoRAAdapterModule)? = nil,
                         step: Int) throws {
        if let (state, module) = optimizer {
            let url = configuration.outputDirectory
                .appendingPathComponent(Self.optimizerFilename(step: step))
            try TensorFile.save(arrays: try state.stateDict(for: module),
                                metadata: [
                                    "step": String(state.step),
                                    "learning_rate": String(configuration.learningRate),
                                    "weight_decay": String(configuration.weightDecay),
                                    "beta1": String(state.betas.0),
                                    "beta2": String(state.betas.1),
                                    "eps": String(state.eps),
                                    "bias_correction": String(state.biasCorrection),
                                ],
                                to: url)
        }
        let url = configuration.outputDirectory
            .appendingPathComponent(String(format: "adapter-%06d.safetensors", step))
        try TensorFile.save(arrays: adapter.stateDict(),
                            metadata: [
                                "rank": String(adapter.rank),
                                "alpha": String(adapter.alpha),
                                "step": String(step),
                            ],
                            to: url)
    }
}

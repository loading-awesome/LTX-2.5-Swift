// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX
import Testing
@testable import LTXPipeline

/// The outer loop, end to end on the real transformer.
///
/// Small — 4 blocks, rank 8, the minimum geometry, a handful of steps — because what is under
/// test is the loop, not the model. The loop's failures are the expensive ones: they surface
/// after hours, and they all look like "the training didn't work".
@Suite("LoRA trainer", .serialized)
struct LoRATrainerTests {

    static let devTransformer =
        LTXConfiguration.resolved.checkpoints.root! + "/diffusion_models/"
        + "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    private func skipIfAbsent() -> Bool {
        if FileManager.default.fileExists(atPath: Self.devTransformer) { return false }
        print("SKIP LoRATrainer: transformer absent at \(Self.devTransformer)")
        return true
    }

    // MARK: - Order, which needs no checkpoint

    /// Reproducible from the seed, and different between epochs. Both halves matter: without
    /// the first a run cannot be debugged, and without the second every epoch pairs the same
    /// clip with the same schedule position.
    @Test("sample order is seeded, stable, and varies by epoch")
    func orderIsSeededAndVaries() {
        let entries = (0 ..< 50).map { "clip\($0)" }

        let a = LoRATrainer.order(entries: entries, seed: 42, epoch: 0)
        #expect(a == LoRATrainer.order(entries: entries, seed: 42, epoch: 0))
        #expect(a != LoRATrainer.order(entries: entries, seed: 43, epoch: 0),
                "a different seed must give a different order")
        #expect(a != LoRATrainer.order(entries: entries, seed: 42, epoch: 1),
                "a different epoch must give a different order")
        // A shuffle that drops or duplicates entries would train on a subset and never say so.
        #expect(Set(a) == Set(entries))
        #expect(a.count == entries.count)
        #expect(a != entries, "the order should not be the manifest's")
    }

    @Test("a one-entry dataset shuffles to itself rather than trapping")
    func orderHandlesDegenerateSizes() {
        #expect(LoRATrainer.order(entries: ["only"], seed: 1, epoch: 0) == ["only"])
        #expect(LoRATrainer.order(entries: [], seed: 1, epoch: 0).isEmpty)
    }

    // MARK: - The loop

    private func withTemporaryRoot(_ body: (URL) throws -> Void) rethrows {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltx-train-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    /// A cache of `count` clips sharing one caption, shapes taken from the checkpoint.
    private func writeCache(root: URL, count: Int, topology: TransformerTopology) throws
        -> TrainingCacheLayout.Fingerprint {
        let fingerprint = TrainingCacheLayout.Fingerprint(
            textEncoder: "gemma:1:1", videoVAE: "vae:1:1", audioVAE: nil,
            transformer: "dit:1:1", bucket: "128x128x9", trigger: nil)
        let geometry = DiTForward.Geometry(frames: 9, height: 128, width: 128, frameRate: 24)
        let latentHeight = geometry.height / geometry.latent.spatialScale
        let latentWidth = geometry.width / geometry.latent.spatialScale
        let latentFrames = geometry.videoTokens / (latentHeight * latentWidth)
        let sequence = GemmaTokenizer.conditioningLength

        MLXRandom.seed(9)
        var names: [String] = []
        for index in 0 ..< count {
            let name = "clip\(index)"
            names.append(name)
            let latents = MLXRandom.normal(
                [topology.latentChannels, latentFrames, latentHeight, latentWidth])
                .asType(.bfloat16)
            MLX.eval(latents)
            try TrainingCache.write(
                TrainingCache.VideoEntry(latents: latents, numFrames: geometry.frames,
                                         height: geometry.height, width: geometry.width,
                                         fps: 24),
                root: root, name: name)
            try TrainingCache.write(
                TrainingCache.ConditionEntry(
                    videoPromptEmbeds:
                        (MLXRandom.normal([sequence, topology.videoWidth]) * 0.05)
                        .asType(.float32),
                    audioPromptEmbeds:
                        (MLXRandom.normal([sequence, topology.audioWidth]) * 0.05)
                        .asType(.float32),
                    promptAttentionMask: MLXArray(
                        (0 ..< sequence).map { Int32($0 < 11 ? 1 : 0) }, [sequence])),
                root: root, name: name)
        }
        try TrainingCache.writeManifest(
            TrainingCacheLayout.Manifest(fingerprint: fingerprint, entries: names,
                                         entriesWithAudio: []),
            root: root)
        return fingerprint
    }

    private struct Loaded {
        let base: DiTForward
        let head: DiTOutputHead
        let header: SafetensorsHeader
        let topology: TransformerTopology
    }

    private func loadTransformer() throws -> Loaded {
        let url = URL(fileURLWithPath: Self.devTransformer)
        let header = try SafetensorsHeader.read(from: url)
        let topology = try TransformerTopology.read(header)
        let weights = try MLX.loadArrays(url: url)
        return Loaded(base: DiTForward(weights: weights, topology: topology,
                                       attentionPath: .fused),
                      head: DiTOutputHead(weights: weights, topology: topology),
                      header: header, topology: topology)
    }

    /// The property the whole loop exists for. Everything else here checks one piece; this
    /// is the only test that says the assembled thing learns.
    @Test("a short run's loss falls and leaves a loadable adapter")
    func shortRunLearns() throws {
        if skipIfAbsent() { return }
        let loaded = try loadTransformer()

        try withTemporaryRoot { root in
            let cache = root.appendingPathComponent("cache")
            let output = root.appendingPathComponent("out")
            let fingerprint = try writeCache(root: cache, count: 3, topology: loaded.topology)

            let dataset = try TrainingDataset(
                root: cache, dit: URL(fileURLWithPath: Self.devTransformer),
                requiring: fingerprint)
            let trainer = LoRATrainer(
                configuration: .init(steps: 12, learningRate: 2e-3, schedule: .cosine(),
                                     rank: 8, alpha: 8, blocks: 4, seed: 7,
                                     snapshotInterval: 6, outputDirectory: output),
                dataset: dataset)

            var reports: [LoRATrainer.StepReport] = []
            let adapter = try trainer.run(base: loaded.base, head: loaded.head,
                                          header: loaded.header) { reports.append($0) }

            #expect(reports.count == 12)
            #expect(reports.allSatisfy { $0.loss.isFinite })

            let first = reports.prefix(4).map(\.loss).reduce(0, +) / 4
            let last = reports.suffix(4).map(\.loss).reduce(0, +) / 4
            print(String(format: "TRAIN first4 %.4f last4 %.4f  lr %.2e -> %.2e",
                         first, last, reports.first!.learningRate,
                         reports.last!.learningRate))
            // Windowed, because each step draws its own noise and a single-sample comparison
            // measures the draw rather than the learning — the mistake
            // `LoRATrainingStepTests` was making until it was run deterministically.
            #expect(last < first, "loss did not fall: \(first) -> \(last)" as Comment)

            // The schedule must have moved. A cosine that reported its base rate every step
            // would mean the rate is never reaching the optimiser.
            #expect(reports.first!.learningRate > reports.last!.learningRate)

            // The gradient norm is reported so a run that is entirely clip-limited is
            // visible; here it should be a real number, not NaN or zero.
            #expect(reports.allSatisfy { $0.gradientNorm.isFinite && $0.gradientNorm > 0 })

            // Snapshots: one at step 6, and the final one at 12 — each with the optimiser
            // moments beside it. Paired deliberately: an adapter whose moments are missing
            // can only be resumed as a fresh optimiser, and the pairing is what makes that
            // a visible absence rather than a silent one.
            let files = try FileManager.default
                .contentsOfDirectory(atPath: output.path).sorted()
            #expect(files == ["adapter-000006.safetensors", "adapter-000012.safetensors",
                              "optimizer-000006.safetensors", "optimizer-000012.safetensors"],
                    "\(files)" as Comment)

            // The moments are their own file rather than extra keys in the adapter. That
            // matters to every consumer: an adapter is loaded by the render path, published
            // and shipped, and a key set that changed depending on whether the run was
            // resumable would have to be stripped somewhere.
            let optimizerURL = output.appendingPathComponent("optimizer-000012.safetensors")
            let moments = try MLX.loadArrays(url: optimizerURL)
            #expect(moments.count == adapter.keys.count * 4,
                    "\(moments.count) moment tensors for \(adapter.keys.count) targets"
                        as Comment)
            #expect(moments.keys.allSatisfy {
                $0.hasSuffix(ResumableAdamW.firstMomentSuffix)
                    || $0.hasSuffix(ResumableAdamW.secondMomentSuffix)
            })

            // A snapshot that cannot be read is worth nothing after a six-hour run, and
            // "the file parses" is too weak a claim. The contract `stateDict` documents is
            // that the key form is the one `LoRAOverlay.load` resolves — so the check is
            // that the saved adapter loads back through the *inference* path and that every
            // key it names is a weight the base checkpoint actually has. A key form that
            // was self-consistent but wrong would satisfy a string comparison and fail here.
            let snapshotURL = output.appendingPathComponent("adapter-000012.safetensors")
            let reloaded = try MLX.loadArrays(url: snapshotURL)
            #expect(reloaded.count == adapter.keys.count * 2)

            let overlay = try LoRAOverlay.load(url: snapshotURL)
            try overlay.validate(againstBaseWeights: loaded.base.weights)
        }
    }

    /// A step's noise must be addressable by step index, not reached by advancing a stream.
    ///
    /// The cheap half of the resume claim, and the one that holds without a transformer:
    /// `stepKey` is a pure function of `(seed, step)`, so whatever the process has drawn in
    /// the meantime cannot move it. Before the key existed the trainer seeded once and let
    /// the loop advance the global stream, which made step 5's noise a function of how many
    /// steps had run *in this process* — the thing that made resume disagree.
    @Test("a step's key is a function of the step index, not of what ran before it")
    func stepKeyIsAddressable() {
        let first = LoRATrainer.stepKey(seed: 7, step: 5)

        // Everything a different run might have done in between.
        MLXRandom.seed(999)
        MLX.eval(MLXRandom.normal([256]))
        let again = LoRATrainer.stepKey(seed: 7, step: 5)
        #expect(first.asArray(UInt32.self) == again.asArray(UInt32.self),
                "the same (seed, step) gave two different keys")

        // And it has to actually vary, or every step trains on one draw.
        let otherStep = LoRATrainer.stepKey(seed: 7, step: 6)
        let otherSeed = LoRATrainer.stepKey(seed: 8, step: 5)
        #expect(first.asArray(UInt32.self) != otherStep.asArray(UInt32.self))
        #expect(first.asArray(UInt32.self) != otherSeed.asArray(UInt32.self))
    }

    /// The end-to-end version: the **first** resumed step trains on the noise the
    /// uninterrupted run trained on, so its loss reproduces exactly.
    ///
    /// This is what the key was added for and what the schedule test above cannot see. The
    /// schedule resumed correctly all along — it is recomputed from the step index — while
    /// the noise did not, and a run whose rate, sample order and parameters all resume
    /// exactly is precisely the run where a different noise draw is hardest to notice.
    ///
    /// Measured with the key removed and everything else identical, this run's step 4 came
    /// back at `42.630455` against the uninterrupted `42.918125`. With the key it is equal
    /// to the bit. That difference is the whole fix.
    ///
    /// ## Why only the first step, and what the rest measures
    ///
    /// **`AdamW`'s moments are not in the snapshot.** ``LoRATrainer/run(base:head:header:resumeFrom:progress:)``
    /// constructs a fresh optimiser on every call, so a resumed run restarts `m`, `v` and
    /// the bias-correction counter while the uninterrupted run carries five steps of them.
    /// The loss is reported *before* the update, so step 4 is unaffected and every later
    /// step inherits a different update:
    ///
    /// ```
    /// step 4   35.44 vs 36.10 -> equal once keyed
    /// step 5   35.44 vs 36.10
    /// step 6   45.26 vs 46.87
    /// step 7   20.75 vs 22.46
    /// ```
    ///
    /// So the tail is asserted as a *known* divergence rather than left silent or asserted
    /// as equality that cannot hold. It is a real gap — a resumed long run is not the run it
    /// would have been — and when the optimiser state is added to the snapshot this test
    /// should tighten to full equality across all four steps, which is the signal that the
    /// fix landed rather than a note in a file somewhere.
    ///
    /// `.constant` rather than `.linear()` on purpose: with a decaying schedule the 4-step
    /// prefix run would train at different rates than the 8-step run's first four steps, so
    /// the two would diverge before the resume even happened and the test would be measuring
    /// the schedule instead of the noise.
    @Test("a resumed run reproduces the uninterrupted run's first step exactly")
    func resumeReproducesTheLosses() throws {
        if skipIfAbsent() { return }
        let loaded = try loadTransformer()

        try withTemporaryRoot { root in
            let cache = root.appendingPathComponent("cache")
            let fingerprint = try writeCache(root: cache, count: 2, topology: loaded.topology)
            let dataset = try TrainingDataset(
                root: cache, dit: URL(fileURLWithPath: Self.devTransformer),
                requiring: fingerprint)

            func configuration(steps: Int, _ output: URL) -> LoRATrainer.Configuration {
                .init(steps: steps, learningRate: 1e-3, schedule: .constant,
                      rank: 8, alpha: 8, blocks: 4, seed: 7,
                      snapshotInterval: nil, outputDirectory: output)
            }

            func straightRun(_ name: String) throws -> [LoRATrainer.StepReport] {
                var out: [LoRATrainer.StepReport] = []
                _ = try LoRATrainer(configuration: configuration(
                    steps: 8, root.appendingPathComponent(name)), dataset: dataset)
                    .run(base: loaded.base, head: loaded.head,
                         header: loaded.header) { out.append($0) }
                return out
            }
            // Twice, to measure this machine's own run-to-run floor on the steps the resume
            // will be compared over. The comparison below is against *that*, not against
            // zero — see `runsAreDeterministic` for why zero is not available.
            let straight = try straightRun("whole")
            let again = try straightRun("whole-again")
            var floor: Float = 0
            for i in 4 ..< 8 {
                floor = max(floor, abs(straight[i].loss - again[i].loss)
                    / max(abs(straight[i].loss), 1))
            }

            // The same first four steps, standing in for "the run so far".
            let prefixDirectory = root.appendingPathComponent("prefix")
            var prefix: [LoRATrainer.StepReport] = []
            let half = try LoRATrainer(configuration: configuration(
                steps: 4, prefixDirectory), dataset: dataset)
                .run(base: loaded.base, head: loaded.head,
                     header: loaded.header) { prefix.append($0) }
            for i in 0 ..< 4 {
                #expect(prefix[i].loss == straight[i].loss,
                        "step \(i): prefix \(prefix[i].loss) vs straight \(straight[i].loss)"
                            as Comment)
            }

            // Read back off disk rather than carried in memory. That is the path a real
            // resume takes — a new process, a directory of snapshots — so a state that
            // serialises lossily, or under names `load` cannot resolve, fails here rather
            // than the first time someone actually restarts a run.
            let optimizerURL = prefixDirectory.appendingPathComponent(
                LoRATrainer.optimizerFilename(step: 4))
            #expect(FileManager.default.fileExists(atPath: optimizerURL.path),
                    "no optimiser snapshot at \(optimizerURL.path)" as Comment)
            let optimizerState = try MLX.loadArrays(url: optimizerURL)
            // Two tensors per factor, two factors per target, 4 blocks x 10 modules.
            #expect(optimizerState.count == 2 * 2 * 4 * 10,
                    "\(optimizerState.count) optimiser tensors" as Comment)

            var resumed: [LoRATrainer.StepReport] = []
            _ = try LoRATrainer(configuration: configuration(
                steps: 8, root.appendingPathComponent("resumed")), dataset: dataset)
                .run(base: loaded.base, head: loaded.head, header: loaded.header,
                     resumeFrom: .init(adapter: half, step: 4,
                                       optimizerState: optimizerState)) { resumed.append($0) }

            #expect(resumed.count == 4)

            // **The two assertions that carry the claim, both bit-exact.**
            //
            // Step 4's loss is computed before any update, from the restored parameters
            // alone: it tests the adapter handover and nothing else.
            //
            // Step 5's loss is computed after one update *made with the restored moments*,
            // and is therefore the moments' own assertion. It is the step that moved by
            // 1.8e-2 relative when the optimiser restarted — 35.44 against 36.10 — so an
            // equality here is the whole fix, stated at the one place a single step of
            // accumulated kernel noise cannot yet reach.
            for offset in 0 ..< 2 {
                let want = straight[4 + offset]
                let restored = "step \(resumed[offset].step): resumed "
                    + "\(resumed[offset].loss) vs uninterrupted \(want.loss). Step 4 is the "
                    + "parameter handover and step 5 is the restored moments; both are one "
                    + "step from identical state and must be exact."
                #expect(resumed[offset].step == want.step)
                #expect(resumed[offset].loss == want.loss, "\(restored)" as Comment)
            }

            // Beyond that, kernel nondeterminism compounds. A resumed run starts from
            // arrays read off disk rather than from a chain of operations, and MLX can pick
            // a different reduction split for the same arithmetic — so two runs that agree
            // exactly for two steps can part in the last digits by the fourth. Measured
            // worst case across invocations: 7.7e-4 relative at step 7, against a
            // no-optimiser-state failure of 1.8e-2 at step 5. The bound sits between them.
            let bound = max(floor, 3e-3)
            print(String(format: "RESUME floor %.3e, bound %.3e", floor, bound))
            for (offset, report) in resumed.enumerated() {
                let want = straight[4 + offset]
                let gap = abs(report.loss - want.loss) / max(abs(want.loss), 1)
                print(String(format: "RESUME step %d loss %.7f vs %.7f (%.3e)",
                             report.step, report.loss, want.loss, gap))
                #expect(report.step == want.step)
                let drift = "step \(report.step): resumed \(report.loss) vs uninterrupted "
                    + "\(want.loss) — \(gap) relative against a floor of \(floor)"
                #expect(gap <= bound, "\(drift)" as Comment)
            }
        }
    }

    /// How reproducible is the loop, actually? Every resume claim is bounded by this.
    ///
    /// **Not bit-exact, and the number matters.** Two runs with the same seed, cache and
    /// configuration agree exactly for the first several steps and then part company in the
    /// last few digits: measured on the gradient norm at step 5 as `25.6527824` in one
    /// invocation and `25.6545677` in another, which reaches ~5e-5 relative on the loss by
    /// step 7. MLX's GPU kernels split reductions by occupancy, so the order of a float32
    /// sum is not fixed run to run, and the difference compounds through the optimiser.
    ///
    /// So this measures rather than asserts zero, and every resume comparison is made
    /// against the floor it measures instead of against bit equality. The bound still has
    /// teeth: resuming *without* the optimiser moments moved the loss by 1.8e-2 relative,
    /// which is three orders of magnitude above this.
    @Test("the loop is reproducible to within a small, measured floor")
    func runsAreDeterministic() throws {
        if skipIfAbsent() { return }
        let loaded = try loadTransformer()

        try withTemporaryRoot { root in
            let cache = root.appendingPathComponent("cache")
            let fingerprint = try writeCache(root: cache, count: 2, topology: loaded.topology)
            let dataset = try TrainingDataset(
                root: cache, dit: URL(fileURLWithPath: Self.devTransformer),
                requiring: fingerprint)

            func losses(_ name: String) throws -> [Float] {
                var out: [Float] = []
                _ = try LoRATrainer(
                    configuration: .init(steps: 8, learningRate: 1e-3, schedule: .constant,
                                         rank: 8, alpha: 8, blocks: 4, seed: 7,
                                         snapshotInterval: nil,
                                         outputDirectory: root.appendingPathComponent(name)),
                    dataset: dataset)
                    .run(base: loaded.base, head: loaded.head,
                         header: loaded.header) { out.append($0.loss) }
                return out
            }
            let a = try losses("a"), b = try losses("b")
            var worst: Float = 0
            for i in 0 ..< min(a.count, b.count) {
                worst = max(worst, abs(a[i] - b[i]) / max(abs(a[i]), 1))
            }
            print(String(format: "TRAIN reproducibility: worst relative gap %.3e over %d steps",
                         worst, min(a.count, b.count)))
            let unstable = "two identical runs differ by \(worst) relative — beyond kernel "
                + "nondeterminism, so something in the loop is genuinely unstable"
            #expect(worst <= 1e-3, "\(unstable)" as Comment)
        }
    }

    /// Resume must continue the schedule, not restart it. A restarted schedule is silent: the
    /// run trains, at the wrong rate, and only a rate log would show it.
    @Test("resuming continues the schedule rather than restarting it")
    func resumeContinuesTheSchedule() throws {
        if skipIfAbsent() { return }
        let loaded = try loadTransformer()

        try withTemporaryRoot { root in
            let cache = root.appendingPathComponent("cache")
            let fingerprint = try writeCache(root: cache, count: 2, topology: loaded.topology)
            let dataset = try TrainingDataset(
                root: cache, dit: URL(fileURLWithPath: Self.devTransformer),
                requiring: fingerprint)

            func configuration(_ output: URL) -> LoRATrainer.Configuration {
                .init(steps: 8, learningRate: 1e-3, schedule: .linear(),
                      rank: 8, alpha: 8, blocks: 4, seed: 7,
                      snapshotInterval: nil, outputDirectory: output)
            }

            // A straight 8-step run, for its rate at step 4.
            var straight: [LoRATrainer.StepReport] = []
            let whole = LoRATrainer(configuration: configuration(
                root.appendingPathComponent("whole")), dataset: dataset)
            let adapter = try whole.run(base: loaded.base, head: loaded.head,
                                        header: loaded.header) { straight.append($0) }

            // Resume the same budget at step 4 and compare the rate it picks up at.
            var resumed: [LoRATrainer.StepReport] = []
            let second = LoRATrainer(configuration: configuration(
                root.appendingPathComponent("resumed")), dataset: dataset)
            _ = try second.run(base: loaded.base, head: loaded.head, header: loaded.header,
                               resumeFrom: .init(adapter: adapter, step: 4)) { resumed.append($0) }

            #expect(resumed.count == 4, "a resumed run must do the remaining steps only")
            #expect(resumed.first!.step == 4)
            #expect(resumed.first!.learningRate == straight[4].learningRate,
                    "\(resumed.first!.learningRate) vs \(straight[4].learningRate)" as Comment)
            #expect(resumed.last!.learningRate == straight.last!.learningRate)
        }
    }

    /// Accumulation must average, not sum: summing scales the effective learning rate by the
    /// accumulation count, which looks like a tuning problem rather than a bug.
    @Test("gradient accumulation consumes several samples for one step")
    func accumulationConsumesSeveralSamples() throws {
        if skipIfAbsent() { return }
        let loaded = try loadTransformer()

        try withTemporaryRoot { root in
            let cache = root.appendingPathComponent("cache")
            let fingerprint = try writeCache(root: cache, count: 4, topology: loaded.topology)
            let dataset = try TrainingDataset(
                root: cache, dit: URL(fileURLWithPath: Self.devTransformer),
                requiring: fingerprint)

            var reports: [LoRATrainer.StepReport] = []
            let trainer = LoRATrainer(
                configuration: .init(steps: 2, learningRate: 1e-3,
                                     gradientAccumulationSteps: 3,
                                     rank: 8, alpha: 8, blocks: 4, seed: 7,
                                     snapshotInterval: nil,
                                     outputDirectory: root.appendingPathComponent("out")),
                dataset: dataset)
            _ = try trainer.run(base: loaded.base, head: loaded.head,
                                header: loaded.header) { reports.append($0) }

            #expect(reports.count == 2)
            #expect(reports.allSatisfy { $0.samples.count == 3 },
                    "each step must consume gradientAccumulationSteps samples")
            #expect(reports.allSatisfy { $0.loss.isFinite })
        }
    }

    @Test("a resume past the budget is refused instead of doing nothing")
    func resumePastBudgetIsRefused() throws {
        if skipIfAbsent() { return }
        let loaded = try loadTransformer()

        try withTemporaryRoot { root in
            let cache = root.appendingPathComponent("cache")
            let fingerprint = try writeCache(root: cache, count: 1, topology: loaded.topology)
            let dataset = try TrainingDataset(
                root: cache, dit: URL(fileURLWithPath: Self.devTransformer),
                requiring: fingerprint)
            let trainer = LoRATrainer(
                configuration: .init(steps: 4, rank: 8, alpha: 8, blocks: 4,
                                     snapshotInterval: nil,
                                     outputDirectory: root.appendingPathComponent("out")),
                dataset: dataset)
            let lora = DiTLoRATargets.makeLoRA(
                header: loaded.header, rank: 8, alpha: 8,
                suffixes: TrainableLoRA.releasedAdapterTargetModules,
                scope: .videoStreamOnly, blocks: 0 ..< 4)

            #expect(throws: LoRATrainer.Failure.self) {
                _ = try trainer.run(base: loaded.base, head: loaded.head,
                                    header: loaded.header,
                                    resumeFrom: .init(adapter: lora, step: 4))
            }
        }
    }
}

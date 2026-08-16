// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX
import MLXNN
import MLXOptimizers
import Testing
@testable import LTXModules

/// A real gradient through the real transformer.
///
/// Everything else in the training suite runs against a stand-in, which proves the wiring
/// and says nothing about whether the thing is *possible*. This one loads the 39 GiB
/// checkpoint, attaches trainable factors, and differentiates through `DiTForward` — the
/// question being activation memory.
///
/// `LTX_TRAIN_CHECKPOINT=1` switches from ``LoRAOverlay/trainable(_:scaling:)`` on the whole
/// forward to ``DiTForward/checkpointing`` per block. Running the sweep both ways is what
/// produced the two series in `docs/TRAINING.md`: without it activations cost 101.9 MB per
/// video token, with it 19.5 MB, which is the difference between the production shape needing
/// ~350 GB and needing 100.8 GB.
///
/// Skips without the checkpoint, in the suite's usual style. Deliberately **not** part of a
/// routine run: it costs a 42 GB load and tens of GB of transient allocation.
///
/// Text context is synthesised rather than encoded. The gradient does not care what the
/// prompt says, and encoding one would add a 26 GB Gemma load to a measurement that is about
/// the transformer.
@Suite("DiT LoRA backward", .serialized)
struct DiTLoRABackwardTests {

    static let devTransformer =
        LTXConfiguration.resolved.checkpoints.root! + "/diffusion_models/"
        + "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    /// Overridable so a sweep needs no recompile: the whole point of this probe is the
    /// scaling, and re-linking between points would make the sweep the slow part.
    ///
    ///     LTX_TRAIN_BLOCKS=48 LTX_TRAIN_W=640 LTX_TRAIN_H=384 LTX_TRAIN_F=97 swift test ...
    static func env(_ name: String, _ fallback: Int) -> Int {
        ProcessInfo.processInfo.environment[name].flatMap(Int.init) ?? fallback
    }
    static var width: Int { env("LTX_TRAIN_W", 128) }
    static var height: Int { env("LTX_TRAIN_H", 128) }
    static var frames: Int { env("LTX_TRAIN_F", 9) }
    static var blocks: Int { env("LTX_TRAIN_BLOCKS", 4) }
    static var rank: Int { env("LTX_TRAIN_RANK", 16) }
    static var checkpointed: Bool { env("LTX_TRAIN_CHECKPOINT", 0) != 0 }

    private func skipIfAbsent() -> Bool {
        if FileManager.default.fileExists(atPath: Self.devTransformer) { return false }
        print("SKIP DiTLoRABackward: transformer absent at \(Self.devTransformer)")
        return true
    }

    @Test("a gradient reaches the LoRA factors through the real transformer")
    func realBackward() throws {
        if skipIfAbsent() { return }

        let url = URL(fileURLWithPath: Self.devTransformer)
        let header = try SafetensorsHeader.read(from: url)
        let topology = try TransformerTopology.read(header)

        // A few blocks, not all 48. The point of the first run is to establish that the
        // gradient flows and to measure what one block costs, which is what makes the
        // 48-block figure predictable rather than a surprise.
        let blockCount = Self.blocks
        let lora = DiTLoRATargets.makeLoRA(
            header: header, rank: Self.rank, alpha: Self.rank,
            suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly, blocks: 0 ..< blockCount)
        #expect(!lora.keys.isEmpty, "no targets discovered")
        print("TARGETS \(lora.keys.count) over \(blockCount) blocks, rank \(lora.rank)")

        let module = LoRAAdapterModule(lora)

        let loadStart = Date()
        let weights = try MLX.loadArrays(url: url)
        MLX.eval(Array(weights.values.prefix(1)))
        print(String(format: "LOAD %.1f s", -loadStart.timeIntervalSinceNow))

        let geometry = DiTForward.Geometry(frames: Self.frames, height: Self.height,
                                           width: Self.width, frameRate: 24)
        let videoTokens = geometry.videoTokens
        let audioTokens = geometry.audioTokens
        print("GEOMETRY \(Self.width)x\(Self.height)x\(Self.frames) -> "
            + "\(videoTokens) video tokens, \(audioTokens) audio tokens")

        let videoLatent = MLXRandom.normal([1, videoTokens, topology.latentChannels])
            .asType(.bfloat16)
        let audioLatent = MLXRandom.normal([1, audioTokens, topology.latentChannels])
            .asType(.bfloat16)
        // Context width is the stream width; 32 tokens of prompt is representative and the
        // gradient is indifferent to the content.
        let videoContext = MLXRandom.normal([1, 32, topology.videoWidth]).asType(.bfloat16)
        let audioContext = MLXRandom.normal([1, 32, topology.audioWidth]).asType(.bfloat16)

        func constant(_ value: Float, _ count: Int) -> MLXArray {
            MLXArray([Float](repeating: value, count: count), [count]).asType(.bfloat16)
        }
        let sampling = DiTForward.Sampling(
            videoTimestep: constant(0.5, videoTokens),
            audioTimestep: constant(0.5, audioTokens),
            videoPromptTimestep: constant(0.5, 1),
            audioPromptTimestep: constant(0.5, 1),
            a2vGateTimestep: constant(0.5, 1),
            v2aGateTimestep: constant(0.5, 1))

        let base = DiTForward(weights: weights, topology: topology, attentionPath: .fused)

        MLX.GPU.resetPeakMemory()
        let beforeForward = MLX.GPU.activeMemory
        let stepStart = Date()

        let useCheckpointing = Self.checkpointed
        print("CHECKPOINTING \(useCheckpointing ? "on" : "off")")

        let valueAndGrad = MLXNN.valueAndGrad(model: module) {
            (model: LoRAAdapterModule, _: [MLXArray]) -> [MLXArray] in
            var forward = base
            if useCheckpointing {
                // Per-block. Note there is no `attach` here: the checkpointed path builds a
                // block-scoped overlay from factors that crossed the boundary. Attaching
                // globally as well would capture the traced arrays, and captured arrays get
                // no gradient — silently.
                forward.checkpointing = model.blockScopedCheckpointing()
            } else {
                // The overlay is rebuilt from the *traced* parameters on every call. Building
                // it outside would capture arrays the transform has already replaced, and the
                // gradient would be structurally zero without anything failing.
                forward.attach(lora: LoRAOverlay.trainable(
                    zip(model.keys, zip(model.a, model.b)).map {
                        (baseKey: $0.0, a: $0.1.0, b: $0.1.1)
                    },
                    scaling: model.scaling))
            }
            let out = try! forward(videoLatent: videoLatent, audioLatent: audioLatent,
                                   videoContext: videoContext, audioContext: audioContext,
                                   sampling: sampling, geometry: geometry,
                                   blocks: blockCount)
            // Any scalar the whole stream feeds is enough to exercise the backward.
            return [(out.video.asType(.float32) * out.video.asType(.float32)).mean()]
        }

        // On the cooperative pool this overflows past ~16 checkpointed blocks — see
        // ``DeepStack``. Uncheckpointed it would run anywhere; routing both paths through the
        // same thread keeps the comparison honest.
        let (losses, gradients) = DeepStack.run {
            let out = valueAndGrad(module, [])
            MLX.eval(out.0, out.1)
            return out
        }
        let stepSeconds = -stepStart.timeIntervalSinceNow

        let peak = MLX.GPU.peakMemory
        print(String(format: "FORWARD+BACKWARD %.1f s", stepSeconds))
        print(String(format: "MEMORY active before %.2f GB, peak %.2f GB",
                     Double(beforeForward) / 1e9, Double(peak) / 1e9))

        // The gradient must exist and be non-trivial for every factor. A structurally
        // disconnected overlay produces zeros here while everything above still runs.
        var counted = 0, nonZero = 0
        for (_, value) in gradients.flattened() {
            counted += 1
            if MLX.abs(value).max().item(Float.self) > 0 { nonZero += 1 }
        }
        print("GRADIENTS \(nonZero)/\(counted) non-zero")
        #expect(counted > 0, "no gradients came back")
        // `B` starts at zero, so dL/dA is zero at step 0 by construction — only the B half
        // can be non-zero on the very first step. That is the correct behaviour, not a bug,
        // and asserting "all non-zero" here would be asserting the wrong thing.
        #expect(nonZero >= counted / 2,
                "\(nonZero)/\(counted) — at least the B half must carry gradient")
        #expect(losses[0].item(Float.self).isFinite)
    }

    /// Checkpointing is a memory transform, not a numerical one: it must return the same loss
    /// and the same gradients. Four blocks at the smallest geometry, so both paths fit in one
    /// test.
    ///
    /// `B` is perturbed off zero first. Left at its zero initialisation every `dL/dA` is
    /// exactly 0 on both paths, and half the comparison would be `0 == 0` — agreement that
    /// would survive the checkpointed path being wired to nothing at all.
    @Test("checkpointing changes neither the loss nor the gradients")
    func checkpointingIsExact() throws {
        if skipIfAbsent() { return }

        let url = URL(fileURLWithPath: Self.devTransformer)
        let header = try SafetensorsHeader.read(from: url)
        let topology = try TransformerTopology.read(header)
        let blockCount = 4, rank = 8

        let lora = DiTLoRATargets.makeLoRA(
            header: header, rank: rank, alpha: rank,
            suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly, blocks: 0 ..< blockCount)
        let module = LoRAAdapterModule(lora)
        MLXRandom.seed(4242)
        module.b = module.b.map { MLXRandom.normal($0.shape) * 0.02 }
        MLX.eval(module.b)

        let weights = try MLX.loadArrays(url: url)
        let geometry = DiTForward.Geometry(frames: 9, height: 128, width: 128, frameRate: 24)
        let videoLatent = MLXRandom.normal([1, geometry.videoTokens, topology.latentChannels])
            .asType(.bfloat16)
        let audioLatent = MLXRandom.normal([1, geometry.audioTokens, topology.latentChannels])
            .asType(.bfloat16)
        let videoContext = MLXRandom.normal([1, 32, topology.videoWidth]).asType(.bfloat16)
        let audioContext = MLXRandom.normal([1, 32, topology.audioWidth]).asType(.bfloat16)
        MLX.eval(videoLatent, audioLatent, videoContext, audioContext)

        func constant(_ value: Float, _ count: Int) -> MLXArray {
            MLXArray([Float](repeating: value, count: count), [count]).asType(.bfloat16)
        }
        let sampling = DiTForward.Sampling(
            videoTimestep: constant(0.5, geometry.videoTokens),
            audioTimestep: constant(0.5, geometry.audioTokens),
            videoPromptTimestep: constant(0.5, 1),
            audioPromptTimestep: constant(0.5, 1),
            a2vGateTimestep: constant(0.5, 1),
            v2aGateTimestep: constant(0.5, 1))

        let base = DiTForward(weights: weights, topology: topology, attentionPath: .fused)

        func run(checkpointed: Bool) -> (Float, [MLXArray]) {
            let valueAndGrad = MLXNN.valueAndGrad(model: module) {
                (model: LoRAAdapterModule, _: [MLXArray]) -> [MLXArray] in
                var forward = base
                if checkpointed {
                    forward.checkpointing = model.blockScopedCheckpointing()
                } else {
                    forward.attach(lora: LoRAOverlay.trainable(
                        zip(model.keys, zip(model.a, model.b)).map {
                            (baseKey: $0.0, a: $0.1.0, b: $0.1.1)
                        },
                        scaling: model.scaling))
                }
                let out = try! forward(videoLatent: videoLatent, audioLatent: audioLatent,
                                       videoContext: videoContext, audioContext: audioContext,
                                       sampling: sampling, geometry: geometry,
                                       blocks: blockCount)
                return [(out.video.asType(.float32) * out.video.asType(.float32)).mean()]
            }
            let (losses, gradients) = valueAndGrad(module, [])
            MLX.eval(losses, gradients)
            return (losses[0].item(Float.self), gradients.flattened().map(\.1))
        }

        let (plainLoss, plainGradients) = run(checkpointed: false)
        let (checkpointLoss, checkpointGradients) = run(checkpointed: true)

        print(String(format: "LOSS plain %.6f, checkpointed %.6f", plainLoss, checkpointLoss))
        #expect(plainLoss.isFinite)
        #expect(abs(plainLoss - checkpointLoss) <= 1e-6 * max(abs(plainLoss), 1),
                "\(plainLoss) vs \(checkpointLoss)" as Comment)

        #expect(plainGradients.count == checkpointGradients.count)
        #expect(plainGradients.count == 2 * lora.keys.count)

        var compared = 0, live = 0
        for (plain, checkpointed) in zip(plainGradients, checkpointGradients) {
            let magnitude = MLX.abs(plain).max().item(Float.self)
            compared += 1
            if magnitude > 0 { live += 1 }
            let error = MLX.abs(plain - checkpointed).max().item(Float.self)
            #expect(error <= 2e-5 * max(magnitude, 1e-3),
                    "max |Δ| \(error) against magnitude \(magnitude)" as Comment)
        }
        print("GRADIENTS compared \(compared), non-zero \(live)")
        // With B perturbed off zero, *both* halves must carry signal — otherwise the
        // comparison above is only testing the half that does.
        #expect(live == compared, "\(live)/\(compared) carried gradient" as Comment)
    }
}

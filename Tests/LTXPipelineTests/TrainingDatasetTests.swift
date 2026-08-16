// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX
import MLXOptimizers
import Testing
@testable import LTXPipeline

/// A training step fed entirely from a cache.
///
/// The cache's parts are all tested on their own — layout in `LTXFoundationTests`, the tensor
/// round trip in `LTXModulesTests`, the step in `DiTTrainingForwardTests`. What none of them
/// shows is that the pieces *compose*: that a `[C, F, H, W]` latent off disk patchifies to the
/// token count the geometry implies, that cached pre-connector embeds run through the
/// connector to the width the transformer wants, and that the result is something
/// ``LoRATrainingStep`` can take a step on.
///
/// Every one of those is a shape agreement between two components derived independently, which
/// is the kind of thing that is either exactly right or fails loudly — so it is worth one test
/// that actually runs it.
///
/// The cache is synthesised rather than encoded. Running the VAE and Gemma here would add a
/// 26 GB load and minutes per run to prove something ``TrainingCacheBuilder`` is responsible
/// for; what is under test is the *reading* path.
@Suite("Training dataset", .serialized)
struct TrainingDatasetTests {

    static let devTransformer =
        LTXConfiguration.resolved.checkpoints.root! + "/diffusion_models/"
        + "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    private func skipIfAbsent() -> Bool {
        if FileManager.default.fileExists(atPath: Self.devTransformer) { return false }
        print("SKIP TrainingDataset: transformer absent at \(Self.devTransformer)")
        return true
    }

    private func withTemporaryRoot(_ body: (URL) throws -> Void) rethrows {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltx-ds-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    private static let fingerprint = TrainingCacheLayout.Fingerprint(
        textEncoder: "gemma:1:1", videoVAE: "vae:1:1", audioVAE: nil,
        transformer: "dit:1:1", bucket: "128x128x9", trigger: nil)

    /// A cache whose shapes are derived from the checkpoint's own topology and geometry,
    /// not hardcoded — hardcoding them would make this test agree with itself rather than
    /// with the model.
    @discardableResult
    private func writeSyntheticEntry(root: URL, name: String,
                                     topology: TransformerTopology) throws
        -> DiTForward.Geometry {
        let geometry = DiTForward.Geometry(frames: 9, height: 128, width: 128, frameRate: 24)
        let latentHeight = geometry.height / geometry.latent.spatialScale
        let latentWidth = geometry.width / geometry.latent.spatialScale
        let latentFrames = geometry.videoTokens / (latentHeight * latentWidth)

        MLXRandom.seed(5)
        let latents = MLXRandom.normal(
            [topology.latentChannels, latentFrames, latentHeight, latentWidth])
            .asType(.bfloat16)
        MLX.eval(latents)
        try TrainingCache.write(
            TrainingCache.VideoEntry(latents: latents, numFrames: geometry.frames,
                                     height: geometry.height, width: geometry.width,
                                     fps: 24),
            root: root, name: name)

        // Pre-connector, at the stream widths the connector consumes.
        let sequence = GemmaTokenizer.conditioningLength
        let mask = MLXArray((0 ..< sequence).map { Int32($0 < 11 ? 1 : 0) }, [sequence])
        try TrainingCache.write(
            TrainingCache.ConditionEntry(
                videoPromptEmbeds: (MLXRandom.normal([sequence, topology.videoWidth]) * 0.05)
                    .asType(.float32),
                audioPromptEmbeds: (MLXRandom.normal([sequence, topology.audioWidth]) * 0.05)
                    .asType(.float32),
                promptAttentionMask: mask),
            root: root, name: name)
        return geometry
    }

    @Test("a cached clip becomes a sample the transformer's shapes accept")
    func sampleShapesAgree() throws {
        if skipIfAbsent() { return }
        let url = URL(fileURLWithPath: Self.devTransformer)
        let topology = try TransformerTopology.read(try SafetensorsHeader.read(from: url))

        try withTemporaryRoot { root in
            let geometry = try writeSyntheticEntry(root: root, name: "a/clip",
                                                   topology: topology)
            try TrainingCache.writeManifest(
                TrainingCacheLayout.Manifest(fingerprint: Self.fingerprint,
                                             entries: ["a/clip"], entriesWithAudio: []),
                root: root)

            let dataset = try TrainingDataset(root: root, dit: url,
                                              requiring: Self.fingerprint)
            #expect(dataset.entries == ["a/clip"])

            let sample = try dataset.sample("a/clip")
            // The token count the patchify produced must be the one the geometry implies.
            // These are derived independently — one from the latent's shape, one from the
            // frame lattice — so agreement is the thing worth checking.
            #expect(sample.batch.shape.sequence == geometry.videoTokens)
            #expect(sample.batch.shape.channels == topology.latentChannels)
            #expect(sample.batch.latents.shape
                == [1, geometry.videoTokens, topology.latentChannels])
            #expect(sample.geometry.frames == 9)
            #expect(sample.geometry.frameRate == 24)

            // Post-connector, at the widths `DiTForward` cross-attends to.
            #expect(sample.videoContext.dim(2) == topology.videoWidth)
            #expect(sample.audioContext.dim(2) == topology.audioWidth)
            // No audio in the manifest, so the stream is silence at the right length.
            #expect(sample.audioLatent.shape
                == [1, geometry.audioTokens, topology.latentChannels])
        }
    }

    /// The whole point, end to end: a step whose every input came off disk.
    @Test("a training step runs on a sample read from the cache")
    func stepFromCache() throws {
        if skipIfAbsent() { return }
        let url = URL(fileURLWithPath: Self.devTransformer)
        let header = try SafetensorsHeader.read(from: url)
        let topology = try TransformerTopology.read(header)

        try withTemporaryRoot { root in
            try writeSyntheticEntry(root: root, name: "a/clip", topology: topology)
            try TrainingCache.writeManifest(
                TrainingCacheLayout.Manifest(fingerprint: Self.fingerprint,
                                             entries: ["a/clip"], entriesWithAudio: []),
                root: root)

            let dataset = try TrainingDataset(root: root, dit: url,
                                              requiring: Self.fingerprint)
            let sample = try dataset.sample("a/clip")

            let weights = try MLX.loadArrays(url: url)
            let base = DiTForward(weights: weights, topology: topology,
                                  attentionPath: .fused)
            let forward = try dataset.forward(
                for: sample, base: base,
                head: DiTOutputHead(weights: weights, topology: topology),
                blocks: 4)

            let lora = DiTLoRATargets.makeLoRA(
                header: header, rank: 8, alpha: 8,
                suffixes: TrainableLoRA.releasedAdapterTargetModules,
                scope: .videoStreamOnly, blocks: 0 ..< 4)
            let module = LoRAAdapterModule(lora)

            let before = module.a[0].asArray(Float.self)
            let result = LoRATrainingStep.step(
                module: module, optimizer: AdamW(learningRate: 1e-3),
                batch: sample.batch, sigmas: [0.5], forward: forward())
            let after = module.a[0].asArray(Float.self)

            print(String(format: "CACHED STEP loss %.6f", result.mean))
            #expect(result.mean.isFinite)
            #expect(result.mean > 0)
            #expect(zip(before, after).contains { $0 != $1 }, "the factors did not move")
        }
    }

    /// The failure the fingerprint exists to prevent. A cache from a different text encoder
    /// trains without complaint on features from the wrong model.
    @Test("a stale cache is refused at construction, before any step runs")
    func staleCacheIsRefused() throws {
        if skipIfAbsent() { return }
        let url = URL(fileURLWithPath: Self.devTransformer)

        try withTemporaryRoot { root in
            try TrainingCache.writeManifest(
                TrainingCacheLayout.Manifest(fingerprint: Self.fingerprint,
                                             entries: ["a"], entriesWithAudio: []),
                root: root)
            var wanted = Self.fingerprint
            wanted.textEncoder = "a-different-gemma"
            #expect(throws: TrainingCacheLayout.Failure.self) {
                _ = try TrainingDataset(root: root, dit: url, requiring: wanted)
            }
        }
    }

    @Test("an empty cache is refused rather than yielding zero steps")
    func emptyCacheIsRefused() throws {
        if skipIfAbsent() { return }
        let url = URL(fileURLWithPath: Self.devTransformer)
        try withTemporaryRoot { root in
            try TrainingCache.writeManifest(
                TrainingCacheLayout.Manifest(fingerprint: Self.fingerprint,
                                             entries: [], entriesWithAudio: []),
                root: root)
            #expect(throws: TrainingDataset.Failure.self) {
                _ = try TrainingDataset(root: root, dit: url, requiring: Self.fingerprint)
            }
        }
    }
}

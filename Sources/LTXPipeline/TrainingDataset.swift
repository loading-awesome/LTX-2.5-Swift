// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX

/// Feeds ``LTXModules/LoRATrainingStep`` from a preprocessed cache.
///
/// This is the half of the caching work that makes training possible: the VAE and Gemma ran
/// once, in ``TrainingCacheBuilder``, and every step from here on reads tensors off disk.
/// What is left per step is the connector — a few small matmuls whose weights live in the DiT
/// rather than the text encoder, and which therefore cannot be baked into the cache without
/// tying every cached caption to one transformer.
///
/// The fingerprint is checked at construction, not at first use. A cache built with a
/// different Gemma trains perfectly happily on the wrong features, so the only useful place
/// to catch it is before the run starts.
public struct TrainingDataset {

    public let root: URL
    public let manifest: TrainingCacheLayout.Manifest
    private let processor: EmbeddingsProcessor

    /// Entry names, sorted. Sorted so a run is reproducible from a seed: shuffling a set
    /// whose order comes from the file system reproduces nothing.
    public var entries: [String] { manifest.entries }

    public enum Failure: Error, CustomStringConvertible {
        case empty(URL)
        case audioAbsent(entry: String)
        case latentRank(entry: String, got: [Int])

        public var description: String {
            switch self {
            case let .empty(root):
                return "the cache at \(root.path) has no entries"
            case let .audioAbsent(entry):
                return "\(entry): the manifest lists audio for this entry but the file is gone"
            case let .latentRank(entry, got):
                return "\(entry): video latent is \(got), expected [C, F, H, W]"
            }
        }
    }

    /// - Parameter fingerprint: what this run needs. Non-nil means the cache is refused if it
    ///   does not match — pass nil only for inspection.
    public init(root: URL, dit ditURL: URL,
                requiring fingerprint: TrainingCacheLayout.Fingerprint?) throws {
        self.root = root
        self.manifest = try TrainingCache.readManifest(root: root)
        if let fingerprint {
            let differences = manifest.fingerprint.differences(from: fingerprint)
            guard differences.isEmpty else {
                throw TrainingCacheLayout.Failure.fingerprintMismatch(differences)
            }
        }
        guard !manifest.entries.isEmpty else { throw Failure.empty(root) }
        // The connector alone, not the whole text pipeline. See
        // ``TextConditioningPipeline/connect(projected:attentionMask:processor:...)``.
        self.processor = try EmbeddingsProcessor(checkpoint: ditURL)
    }

    /// Everything one step needs for one clip.
    public struct Sample {
        public let name: String
        /// Patchified, `[1, videoTokens, channels]`, float32 — the objective's space.
        public let batch: LoRATrainingStep.Batch
        public let geometry: DiTForward.Geometry
        /// `enc.features.video`, after the connector.
        public let videoContext: MLXArray
        public let audioContext: MLXArray
        /// Patchified audio, `[1, audioTokens, channels]`, in the transformer's dtype.
        public let audioLatent: MLXArray
    }

    /// Load one clip and shape it for a training step.
    ///
    /// Nothing here runs a VAE or Gemma. The cost is a safetensors read, a patchify, and the
    /// connector.
    public func sample(_ name: String, dtype: DType = .bfloat16) throws -> Sample {
        let video = try TrainingCache.readVideo(root: root, entry: name)
        guard video.latents.ndim == 4 else {
            throw Failure.latentRank(entry: name, got: video.latents.shape)
        }

        // The cache stores `[C, F, H, W]` — no batch axis on write, and `patchifyVideo`
        // wants it back.
        let batched = video.latents.expandedDimensions(axis: 0)
        let tokens = try Sampler.patchifyVideo(batched).asType(.float32)

        let geometry = DiTForward.Geometry(frames: video.numFrames,
                                           height: video.height,
                                           width: video.width,
                                           frameRate: video.fps)

        let condition = try TrainingCache.readCondition(root: root, entry: name)
        // `[S, width] -> [1, S, width]`, and the mask `[S] -> [1, S]`: the connector works in
        // batched shapes, and the cache stores one clip.
        let mask = condition.promptAttentionMask.ndim == 1
            ? condition.promptAttentionMask.expandedDimensions(axis: 0)
            : condition.promptAttentionMask
        var projected: [TextConditioningLayout.Stream: MLXArray] = [
            .video: condition.videoPromptEmbeds.ndim == 2
                ? condition.videoPromptEmbeds.expandedDimensions(axis: 0)
                : condition.videoPromptEmbeds,
        ]
        if let audio = condition.audioPromptEmbeds {
            projected[.audio] = audio.ndim == 2 ? audio.expandedDimensions(axis: 0) : audio
        }
        let encoding = try TextConditioningPipeline.connect(
            projected: projected, attentionMask: mask, processor: processor, branch: 0)

        let audioLatent: MLXArray
        if manifest.entriesWithAudio.contains(name) {
            guard TrainingCache.exists(root: root, kind: .audioLatents, entry: name) else {
                throw Failure.audioAbsent(entry: name)
            }
            let audio = try TrainingCache.readAudio(root: root, entry: name)
            audioLatent = try Sampler.patchifyAudio(
                audio.latents.expandedDimensions(axis: 0)).asType(dtype)
        } else {
            // Silence. The cross-modal bridges read the audio stream on every block whether
            // or not the clip had a track, so a clip without audio still has to supply one —
            // and zeros are what the encoder produces for silence anyway. It stays clean and
            // is excluded from the loss, exactly as a conditioning stream is.
            audioLatent = MLXArray.zeros(
                [1, geometry.audioTokens, tokens.dim(2)]).asType(dtype)
        }

        let shape = TrainingObjective.Shape(batch: 1, sequence: tokens.dim(1),
                                            channels: tokens.dim(2))
        return Sample(name: name,
                      batch: .unconditioned(latents: tokens, shape: shape),
                      geometry: geometry,
                      videoContext: encoding.features[.video]!,
                      audioContext: encoding.features[.audio]!,
                      audioLatent: audioLatent)
    }

    /// A ``DiTTrainingForward`` for one sample, sharing an already-loaded transformer.
    public func forward(for sample: Sample, base: DiTForward, head: DiTOutputHead,
                        gradientCheckpointing: Bool = true,
                        blocks: Int? = nil) throws -> DiTTrainingForward {
        let audioTimestep = MLXArray(
            [Float](repeating: 0, count: sample.geometry.audioTokens),
            [sample.geometry.audioTokens]).asType(sample.audioLatent.dtype)
        let forward = DiTTrainingForward(
            base: base, head: head, geometry: sample.geometry,
            videoContext: sample.videoContext, audioContext: sample.audioContext,
            audioLatent: sample.audioLatent, audioTimestep: audioTimestep,
            gradientCheckpointing: gradientCheckpointing, blocks: blocks)
        try forward.validate(shape: sample.batch.shape)
        return forward
    }
}

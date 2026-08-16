// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXModules
import MLX

/// Encodes a dataset into the cache ``TrainingDataset`` reads.
///
/// The whole point is that this runs **once**. A 26 GB Gemma load and a VAE pass per clip is
/// unremarkable as preprocessing and ruinous as a per-step cost: at 20 s of transformer work
/// per step, re-encoding would more than double it, for a result identical every time.
///
/// ## Order of work
///
/// Captions first, then video. Both stages load a large model and neither needs the other's,
/// so they are done in separate passes and the encoder is dropped in between — the same
/// phase-scoped dropping the render pipeline uses to keep peak memory near the largest single
/// component rather than their sum.
public struct TrainingCacheBuilder {

    /// One row of the dataset.
    public struct Item: Sendable, Equatable {
        /// Path to the clip, absolute or relative to ``Options/mediaRoot``.
        public var video: String
        public var caption: String

        public init(video: String, caption: String) {
            self.video = video
            self.caption = caption
        }
    }

    public struct Options: Sendable {
        public var root: URL
        public var mediaRoot: URL
        /// Every clip is resized and centre-cropped to this. One bucket, not a list: mixing
        /// shapes needs batch-1 training and per-bucket sampling, which is not written.
        public var width: Int
        public var height: Int
        public var frames: Int
        /// Prepended to every caption, for a trigger-word LoRA.
        public var trigger: String?
        /// Re-encode entries that already have files. Off by default so an interrupted run
        /// resumes; the fingerprint is what catches a *stale* cache, not this.
        public var overwrite: Bool

        public init(root: URL, mediaRoot: URL, width: Int, height: Int, frames: Int,
                    trigger: String? = nil, overwrite: Bool = false) {
            self.root = root
            self.mediaRoot = mediaRoot
            self.width = width
            self.height = height
            self.frames = frames
            self.trigger = trigger
            self.overwrite = overwrite
        }

        public var bucket: String { "\(width)x\(height)x\(frames)" }
    }

    public struct Checkpoints: Sendable {
        public var textEncoder: URL
        public var dit: URL
        public var videoVAE: URL

        public init(textEncoder: URL, dit: URL, videoVAE: URL) {
            self.textEncoder = textEncoder
            self.dit = dit
            self.videoVAE = videoVAE
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case noItems
        case frameShortfall(item: String, got: Int, wanted: Int)

        public var description: String {
            switch self {
            case .noItems:
                return "the dataset is empty"
            case let .frameShortfall(item, got, wanted):
                return "\(item): decoded \(got) frames, the bucket needs \(wanted). Use a "
                    + "shorter bucket or drop the clip — padding it would train on "
                    + "frames that are not in the source."
            }
        }
    }

    public var options: Options
    public var checkpoints: Checkpoints

    public init(options: Options, checkpoints: Checkpoints) {
        self.options = options
        self.checkpoints = checkpoints
    }

    /// Identity of this configuration, for the staleness check.
    ///
    /// Checkpoints are identified by content — ``CheckpointIdentity`` reads the safetensors
    /// header — rather than by path. A path says nothing about whether the file behind it
    /// changed, and swapping a checkpoint in place while keeping its name is exactly the case
    /// the fingerprint exists to catch.
    public func fingerprint() throws -> TrainingCacheLayout.Fingerprint {
        func identity(_ url: URL) throws -> String {
            let header = try SafetensorsHeader.read(from: url)
            return "\(url.lastPathComponent):\(header.tensors.count):\(header.declaredDataSize)"
        }
        return TrainingCacheLayout.Fingerprint(
            textEncoder: try identity(checkpoints.textEncoder),
            videoVAE: try identity(checkpoints.videoVAE),
            audioVAE: nil,
            transformer: try identity(checkpoints.dit),
            bucket: options.bucket,
            trigger: options.trigger)
    }

    /// Encode every item, writing the manifest last.
    ///
    /// The manifest is written last on purpose. It is what ``TrainingDataset`` reads to decide
    /// the cache is usable, so writing it up front would make a crashed run look like a
    /// complete one.
    public func build(_ items: [Item],
                      note: ((String) -> Void)? = nil) async throws -> TrainingCacheLayout.Manifest {
        guard !items.isEmpty else { throw Failure.noItems }
        let fingerprint = try fingerprint()

        try encodeCaptions(items, note: note)
        MLX.Memory.clearCache()
        try await encodeVideo(items, note: note)
        MLX.Memory.clearCache()

        let manifest = TrainingCacheLayout.Manifest(
            fingerprint: fingerprint,
            entries: items.map { TrainingCacheLayout.entryName(forMediaPath: $0.video) },
            // Audio is not encoded yet — see `docs/TRAINING.md`. Recording it as absent is
            // what makes `TrainingDataset` substitute silence rather than fail.
            entriesWithAudio: [])
        try TrainingCache.writeManifest(manifest, root: options.root)
        note?("manifest: \(manifest.entries.count) entries")
        return manifest
    }

    // MARK: - Captions

    private func encodeCaptions(_ items: [Item], note: ((String) -> Void)?) throws {
        let pending = items.filter {
            options.overwrite || !TrainingCache.exists(
                root: options.root, kind: .conditions,
                entry: TrainingCacheLayout.entryName(forMediaPath: $0.video))
        }
        guard !pending.isEmpty else {
            note?("captions: all \(items.count) cached")
            return
        }

        note?("captions: loading the text encoder")
        let pipeline = try TextConditioningPipeline(textEncoder: checkpoints.textEncoder,
                                                    dit: checkpoints.dit)
        for (index, item) in pending.enumerated() {
            let name = TrainingCacheLayout.entryName(forMediaPath: item.video)
            let caption = options.trigger.map { "\($0) \(item.caption)" } ?? item.caption
            // Branch 0 is the positive prompt. Training has no negative branch — guidance is
            // an inference-time construction — so the default-negative substitution that
            // `branchPrompts` performs must not be reached here.
            let encoding = try pipeline.encode(prompt: caption, branch: 0)

            // Stored pre-connector and with the batch axis dropped, which is the layout the
            // cache format defines. `projected`, not `features`.
            try TrainingCache.write(
                TrainingCache.ConditionEntry(
                    videoPromptEmbeds: encoding.projected[.video]!.squeezed(axis: 0),
                    audioPromptEmbeds: encoding.projected[.audio]?.squeezed(axis: 0),
                    promptAttentionMask: encoding.attentionMask.squeezed(axis: 0)),
                root: options.root, name: name)
            note?("captions: \(index + 1)/\(pending.count) \(name)")
        }
    }

    // MARK: - Video

    private func encodeVideo(_ items: [Item], note: ((String) -> Void)?) async throws {
        let pending = items.filter {
            options.overwrite || !TrainingCache.exists(
                root: options.root, kind: .videoLatents,
                entry: TrainingCacheLayout.entryName(forMediaPath: $0.video))
        }
        guard !pending.isEmpty else {
            note?("video: all \(items.count) cached")
            return
        }

        note?("video: loading the VAE")
        let encoder = try VideoVAEEncoder(checkpoint: checkpoints.videoVAE)
        for (index, item) in pending.enumerated() {
            let name = TrainingCacheLayout.entryName(forMediaPath: item.video)
            let url = URL(fileURLWithPath: item.video, relativeTo: options.mediaRoot)

            var rgb = try await MediaInput.videoFrames(at: url, maxFrames: options.frames)
            guard rgb.dim(0) >= options.frames else {
                throw Failure.frameShortfall(item: name, got: rgb.dim(0),
                                             wanted: options.frames)
            }
            rgb = rgb[0 ..< options.frames]
            rgb = try MediaInput.resizeAndCenterCrop(rgb, height: options.height,
                                                     width: options.width)
            // Read from the file, never assumed. See ``MediaInput/frameRate(at:)``.
            let fps = try await MediaInput.frameRate(at: url)

            // bf16 deliberately: an fp32 clip promotes all 42 convolutions and computes more
            // precisely than the encoder is meant to, which is the expensive way to be
            // wrong. Same choice `VideoUpscaler` makes on the same encoder.
            let latent = try encoder.encode(encoder.pixels(fromRGB: rgb).asType(.bfloat16))
            MLX.eval(latent)

            // `[1, C, F, H, W] -> [C, F, H, W]`: the cache stores no batch axis.
            try TrainingCache.write(
                TrainingCache.VideoEntry(latents: latent.squeezed(axis: 0),
                                         numFrames: options.frames,
                                         height: options.height,
                                         width: options.width,
                                         fps: fps),
                root: options.root, name: name)
            note?("video: \(index + 1)/\(pending.count) \(name) \(latent.shape)")
            MLX.Memory.clearCache()
        }
    }
}

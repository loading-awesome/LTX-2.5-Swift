// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// Reads and writes the preprocessed training cache described by ``TrainingCacheLayout``.
///
/// Tensors go through ``TensorFile``, so every file is byte-stable for identical content: a
/// cache you cannot checksum is a cache you cannot tell apart from a stale one.
public enum TrainingCache {

    // MARK: - Payloads

    /// One clip's encoded video.
    public struct VideoEntry {
        /// `[C, F', H', W']`.
        public var latents: MLXArray
        /// Source frame count, before the VAE's temporal downsample.
        public var numFrames: Int
        public var height: Int
        public var width: Int
        /// Effective fps — source fps over the temporal subsample factor.
        public var fps: Double

        public init(latents: MLXArray, numFrames: Int, height: Int, width: Int, fps: Double) {
            self.latents = latents
            self.numFrames = numFrames
            self.height = height
            self.width = width
            self.fps = fps
        }
    }

    public struct AudioEntry {
        public var latents: MLXArray
        public var numTimeSteps: Int
        public var frequencyBins: Int
        public var duration: Double

        public init(latents: MLXArray, numTimeSteps: Int, frequencyBins: Int,
                    duration: Double) {
            self.latents = latents
            self.numTimeSteps = numTimeSteps
            self.frequencyBins = frequencyBins
            self.duration = duration
        }
    }

    /// One clip's encoded caption, **pre-connector** — see
    /// ``TrainingCacheLayout/ConditionField/videoPromptEmbeds``.
    public struct ConditionEntry {
        /// `[S, width]`. The batch axis is dropped on write.
        public var videoPromptEmbeds: MLXArray
        public var audioPromptEmbeds: MLXArray?
        /// `[S]`.
        public var promptAttentionMask: MLXArray

        public init(videoPromptEmbeds: MLXArray, audioPromptEmbeds: MLXArray?,
                    promptAttentionMask: MLXArray) {
            self.videoPromptEmbeds = videoPromptEmbeds
            self.audioPromptEmbeds = audioPromptEmbeds
            self.promptAttentionMask = promptAttentionMask
        }
    }

    public enum Failure: Error, CustomStringConvertible {
        case absent(kind: String, entry: String, path: String)
        case missingTensor(String, entry: String)

        public var description: String {
            switch self {
            case let .absent(kind, entry, path):
                return "\(entry): no \(kind) in the cache at \(path)"
            case let .missingTensor(name, entry):
                return "\(entry): cache file has no tensor named \(name)"
            }
        }
    }

    // MARK: - Writing

    public static func write(_ entry: VideoEntry, root: URL, name: String) throws {
        try write(
            arrays: [TrainingCacheLayout.VideoField.latents: entry.latents],
            metadata: [
                TrainingCacheLayout.VideoField.numFrames:
                    TrainingCacheLayout.format(entry.numFrames),
                TrainingCacheLayout.VideoField.height:
                    TrainingCacheLayout.format(entry.height),
                TrainingCacheLayout.VideoField.width:
                    TrainingCacheLayout.format(entry.width),
                TrainingCacheLayout.VideoField.fps:
                    TrainingCacheLayout.format(entry.fps),
            ],
            to: TrainingCacheLayout.url(root: root, kind: .videoLatents, entry: name))
    }

    public static func write(_ entry: AudioEntry, root: URL, name: String) throws {
        try write(
            arrays: [TrainingCacheLayout.AudioField.latents: entry.latents],
            metadata: [
                TrainingCacheLayout.AudioField.numTimeSteps:
                    TrainingCacheLayout.format(entry.numTimeSteps),
                TrainingCacheLayout.AudioField.frequencyBins:
                    TrainingCacheLayout.format(entry.frequencyBins),
                TrainingCacheLayout.AudioField.duration:
                    TrainingCacheLayout.format(entry.duration),
            ],
            to: TrainingCacheLayout.url(root: root, kind: .audioLatents, entry: name))
    }

    public static func write(_ entry: ConditionEntry, root: URL, name: String) throws {
        var arrays = [
            TrainingCacheLayout.ConditionField.videoPromptEmbeds: entry.videoPromptEmbeds,
            TrainingCacheLayout.ConditionField.promptAttentionMask: entry.promptAttentionMask,
        ]
        if let audio = entry.audioPromptEmbeds {
            arrays[TrainingCacheLayout.ConditionField.audioPromptEmbeds] = audio
        }
        try write(arrays: arrays, metadata: [:],
                  to: TrainingCacheLayout.url(root: root, kind: .conditions, entry: name))
    }

    /// Written to a temporary sibling and moved into place.
    ///
    /// Preprocessing a dataset is long and gets interrupted. A half-written file that *looks*
    /// complete is the worst outcome — it would be skipped as done on the next run and then
    /// train on truncated data.
    private static func write(arrays: [String: MLXArray], metadata: [String: String],
                              to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        let temporary = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(ProcessInfo.processInfo.processIdentifier).tmp")
        try TensorFile.save(arrays: arrays, metadata: metadata, to: temporary)
        // `replaceItemAt` is atomic and clobbers an existing file; `moveItem` refuses one.
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: url)
        }
    }

    // MARK: - Reading

    public static func exists(root: URL, kind: TrainingCacheLayout.Kind,
                              entry: String) -> Bool {
        FileManager.default.fileExists(
            atPath: TrainingCacheLayout.url(root: root, kind: kind, entry: entry).path)
    }

    public static func readVideo(root: URL, entry: String) throws -> VideoEntry {
        let url = TrainingCacheLayout.url(root: root, kind: .videoLatents, entry: entry)
        let (arrays, metadata) = try load(url, kind: "video latent", entry: entry)
        return VideoEntry(
            latents: try tensor(TrainingCacheLayout.VideoField.latents, arrays, entry),
            numFrames: try TrainingCacheLayout.integer(
                TrainingCacheLayout.VideoField.numFrames, in: metadata, entry: entry),
            height: try TrainingCacheLayout.integer(
                TrainingCacheLayout.VideoField.height, in: metadata, entry: entry),
            width: try TrainingCacheLayout.integer(
                TrainingCacheLayout.VideoField.width, in: metadata, entry: entry),
            fps: try TrainingCacheLayout.double(
                TrainingCacheLayout.VideoField.fps, in: metadata, entry: entry))
    }

    public static func readAudio(root: URL, entry: String) throws -> AudioEntry {
        let url = TrainingCacheLayout.url(root: root, kind: .audioLatents, entry: entry)
        let (arrays, metadata) = try load(url, kind: "audio latent", entry: entry)
        return AudioEntry(
            latents: try tensor(TrainingCacheLayout.AudioField.latents, arrays, entry),
            numTimeSteps: try TrainingCacheLayout.integer(
                TrainingCacheLayout.AudioField.numTimeSteps, in: metadata, entry: entry),
            frequencyBins: try TrainingCacheLayout.integer(
                TrainingCacheLayout.AudioField.frequencyBins, in: metadata, entry: entry),
            duration: try TrainingCacheLayout.double(
                TrainingCacheLayout.AudioField.duration, in: metadata, entry: entry))
    }

    public static func readCondition(root: URL, entry: String) throws -> ConditionEntry {
        let url = TrainingCacheLayout.url(root: root, kind: .conditions, entry: entry)
        let (arrays, _) = try load(url, kind: "condition", entry: entry)
        return ConditionEntry(
            videoPromptEmbeds: try tensor(
                TrainingCacheLayout.ConditionField.videoPromptEmbeds, arrays, entry),
            audioPromptEmbeds: arrays[TrainingCacheLayout.ConditionField.audioPromptEmbeds],
            promptAttentionMask: try tensor(
                TrainingCacheLayout.ConditionField.promptAttentionMask, arrays, entry))
    }

    private static func load(_ url: URL, kind: String, entry: String) throws
        -> ([String: MLXArray], [String: String]) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Failure.absent(kind: kind, entry: entry, path: url.path)
        }
        let header = try SafetensorsHeader.read(from: url)
        return (try MLX.loadArrays(url: url), header.metadata)
    }

    private static func tensor(_ name: String, _ arrays: [String: MLXArray],
                               _ entry: String) throws -> MLXArray {
        guard let array = arrays[name] else {
            throw Failure.missingTensor(name, entry: entry)
        }
        return array
    }

    // MARK: - Manifest

    public static func writeManifest(_ manifest: TrainingCacheLayout.Manifest,
                                     root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try manifest.encoded().write(to: TrainingCacheLayout.manifestURL(root: root),
                                     options: .atomic)
    }

    public static func readManifest(root: URL) throws -> TrainingCacheLayout.Manifest {
        try TrainingCacheLayout.Manifest.decode(
            Data(contentsOf: TrainingCacheLayout.manifestURL(root: root)))
    }

    /// Refuses a cache built with different settings.
    ///
    /// Call this before a run, not after. A mismatched cache trains perfectly well on the
    /// wrong features — 2.5's Gemma 4 embeddings are not 2.3's Gemma 3, and nothing else in
    /// the pipeline notices — so the only useful place to catch it is before the first step.
    public static func validate(root: URL,
                                against wanted: TrainingCacheLayout.Fingerprint) throws {
        let manifest = try readManifest(root: root)
        let differences = manifest.fingerprint.differences(from: wanted)
        guard differences.isEmpty else {
            throw TrainingCacheLayout.Failure.fingerprintMismatch(differences)
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import Testing
@testable import LTXModules

/// Round-tripping the cache. Synthetic tensors only — no checkpoints, no GPU work worth
/// mentioning, so this stays in the routine suite where a format regression should surface.
@Suite("Training cache", .serialized)
struct TrainingCacheTests {

    private func withTemporaryRoot(_ body: (URL) throws -> Void) throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltx-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    private var fingerprint: TrainingCacheLayout.Fingerprint {
        TrainingCacheLayout.Fingerprint(
            textEncoder: "gemma:1:2", videoVAE: "vae:3:4", audioVAE: nil,
            transformer: "dit:5:6", bucket: "320x192x97", trigger: nil)
    }

    @Test("a video entry survives the round trip, scalars included")
    func videoRoundTrips() throws {
        try withTemporaryRoot { root in
            let latents = MLXRandom.normal([128, 13, 6, 10]).asType(.bfloat16)
            MLX.eval(latents)
            // 23.976 rather than 24: a rate that a sloppy formatter would round, and the
            // position grid is in wall-clock seconds.
            let entry = TrainingCache.VideoEntry(latents: latents, numFrames: 97,
                                                 height: 192, width: 320, fps: 23.976023976)
            try TrainingCache.write(entry, root: root, name: "scene1/clip")

            let back = try TrainingCache.readVideo(root: root, entry: "scene1/clip")
            #expect(back.latents.shape == [128, 13, 6, 10])
            #expect(back.numFrames == 97)
            #expect(back.height == 192)
            #expect(back.width == 320)
            #expect(back.fps == 23.976023976)
            #expect(MLX.abs(back.latents.asType(.float32)
                - latents.asType(.float32)).max().item(Float.self) == 0)
        }
    }

    @Test("a condition entry round trips, with and without audio")
    func conditionRoundTrips() throws {
        try withTemporaryRoot { root in
            let video = MLXRandom.normal([1024, 64]).asType(.float32)
            let audio = MLXRandom.normal([1024, 32]).asType(.float32)
            let mask = MLXArray((0 ..< 1024).map { Int32($0 < 11 ? 1 : 0) }, [1024])
            MLX.eval(video, audio, mask)

            try TrainingCache.write(
                TrainingCache.ConditionEntry(videoPromptEmbeds: video,
                                            audioPromptEmbeds: audio,
                                            promptAttentionMask: mask),
                root: root, name: "withAudio")
            let both = try TrainingCache.readCondition(root: root, entry: "withAudio")
            #expect(both.audioPromptEmbeds != nil)
            #expect(both.videoPromptEmbeds.shape == [1024, 64])
            #expect(both.promptAttentionMask.sum().item(Int32.self) == 11)

            // Absent audio must read back as nil, not as zeros. A zero-filled audio stream
            // would be a valid-looking prompt that says nothing.
            try TrainingCache.write(
                TrainingCache.ConditionEntry(videoPromptEmbeds: video,
                                            audioPromptEmbeds: nil,
                                            promptAttentionMask: mask),
                root: root, name: "videoOnly")
            #expect(try TrainingCache.readCondition(root: root,
                                                    entry: "videoOnly").audioPromptEmbeds == nil)
        }
    }

    @Test("an audio entry round trips")
    func audioRoundTrips() throws {
        try withTemporaryRoot { root in
            let latents = MLXRandom.normal([8, 26, 16]).asType(.bfloat16)
            MLX.eval(latents)
            try TrainingCache.write(
                TrainingCache.AudioEntry(latents: latents, numTimeSteps: 26,
                                         frequencyBins: 16, duration: 4.041666),
                root: root, name: "a")
            let back = try TrainingCache.readAudio(root: root, entry: "a")
            #expect(back.numTimeSteps == 26)
            #expect(back.frequencyBins == 16)
            #expect(back.duration == 4.041666)
        }
    }

    /// Nested names must produce nested files, not a collision. The layout test proves the
    /// naming; this proves the writer actually creates the directories.
    @Test("two clips with the same basename do not overwrite each other")
    func nestedEntriesCoexist() throws {
        try withTemporaryRoot { root in
            for (index, name) in ["scene1/clip", "scene2/clip"].enumerated() {
                let latents = MLXArray([Float(index)], [1, 1, 1, 1]).asType(.bfloat16)
                try TrainingCache.write(
                    TrainingCache.VideoEntry(latents: latents, numFrames: 9, height: 128,
                                             width: 128, fps: 24),
                    root: root, name: name)
            }
            let a = try TrainingCache.readVideo(root: root, entry: "scene1/clip")
            let b = try TrainingCache.readVideo(root: root, entry: "scene2/clip")
            #expect(a.latents.item(Float.self) == 0)
            #expect(b.latents.item(Float.self) == 1)
        }
    }

    /// Byte-stability is what makes a cache checksummable, and the sidecar-ordering bug this
    /// repository already fixed once is exactly what it guards against.
    @Test("writing the same entry twice produces identical bytes")
    func writesAreDeterministic() throws {
        try withTemporaryRoot { root in
            let latents = MLXRandom.normal([4, 3, 2, 2]).asType(.bfloat16)
            let mask = MLXArray([Int32](repeating: 1, count: 8), [8])
            MLX.eval(latents, mask)
            let entry = TrainingCache.ConditionEntry(
                videoPromptEmbeds: latents.reshaped([4, 12]).asType(.float32),
                audioPromptEmbeds: latents.reshaped([4, 12]).asType(.float32),
                promptAttentionMask: mask)

            try TrainingCache.write(entry, root: root, name: "x")
            let first = try Data(contentsOf: TrainingCacheLayout.url(
                root: root, kind: .conditions, entry: "x"))
            // Overwrite in place: this also exercises the atomic replace path.
            try TrainingCache.write(entry, root: root, name: "x")
            let second = try Data(contentsOf: TrainingCacheLayout.url(
                root: root, kind: .conditions, entry: "x"))
            #expect(first == second, "the cache is not byte-stable")
        }
    }

    @Test("a missing entry is a named error, not a crash")
    func missingEntryIsNamed() throws {
        try withTemporaryRoot { root in
            #expect(throws: TrainingCache.Failure.self) {
                try TrainingCache.readVideo(root: root, entry: "nope")
            }
            #expect(!TrainingCache.exists(root: root, kind: .videoLatents, entry: "nope"))
        }
    }

    // MARK: - Staleness

    @Test("a cache built with a different text encoder is refused before the run")
    func staleCacheIsRefused() throws {
        try withTemporaryRoot { root in
            try TrainingCache.writeManifest(
                TrainingCacheLayout.Manifest(fingerprint: fingerprint, entries: ["a"],
                                             entriesWithAudio: []),
                root: root)

            // Same settings: usable.
            try TrainingCache.validate(root: root, against: fingerprint)

            var wanted = fingerprint
            wanted.textEncoder = "gemma3:9:9"
            #expect(throws: TrainingCacheLayout.Failure.self) {
                try TrainingCache.validate(root: root, against: wanted)
            }
        }
    }

    @Test("the manifest round trips through disk")
    func manifestRoundTrips() throws {
        try withTemporaryRoot { root in
            let manifest = TrainingCacheLayout.Manifest(
                fingerprint: fingerprint, entries: ["b", "a"], entriesWithAudio: ["a"])
            try TrainingCache.writeManifest(manifest, root: root)
            #expect(try TrainingCache.readManifest(root: root) == manifest)
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXFoundation

/// Layout, naming, scalars and staleness — the parts of a cache that fail without saying so.
///
/// A cache with a mis-encoded frame count or a colliding entry name still trains. It converges,
/// it produces an adapter, and the adapter is wrong. None of that needs a GPU to prevent, so
/// none of it is tested with one.
@Suite("Training cache layout")
struct TrainingCacheLayoutTests {

    // MARK: - Naming

    @Test("directory structure is preserved, so same-named clips do not collide")
    func nestedNamesDoNotCollide() {
        let a = TrainingCacheLayout.entryName(forMediaPath: "scene1/clip.mp4")
        let b = TrainingCacheLayout.entryName(forMediaPath: "scene2/clip.mp4")
        #expect(a == "scene1/clip")
        #expect(b == "scene2/clip")
        #expect(a != b, "flattening to a basename would silently overwrite one with the other")
    }

    @Test("the extension is replaced, not appended")
    func extensionIsReplaced() {
        #expect(TrainingCacheLayout.entryName(forMediaPath: "a/b.mp4") == "a/b")
        #expect(TrainingCacheLayout.entryName(forMediaPath: "a/b.MOV") == "a/b")
        // A dot in a directory name is not an extension.
        #expect(TrainingCacheLayout.entryName(forMediaPath: "v1.2/clip.mp4") == "v1.2/clip")
    }

    /// An absolute path must not escape the cache root. `appendingPathComponent` on a string
    /// that starts with "/" is fine, but the resulting tree would mirror the whole filesystem
    /// under the cache; stripping the root keeps it relative.
    @Test("an absolute media path stays inside the cache root")
    func absolutePathsStayInside() {
        let name = TrainingCacheLayout.entryName(forMediaPath: "/data/clips/x.mp4")
        #expect(!name.hasPrefix("/"))
        let url = TrainingCacheLayout.url(root: URL(fileURLWithPath: "/cache"),
                                          kind: .videoLatents, entry: name)
        #expect(url.path.hasPrefix("/cache/latents/"), "\(url.path)")
    }

    @Test("each kind lands in its own directory")
    func directoriesMatchTheReference() {
        let root = URL(fileURLWithPath: "/cache")
        #expect(TrainingCacheLayout.url(root: root, kind: .videoLatents, entry: "a").path
            == "/cache/latents/a.safetensors")
        #expect(TrainingCacheLayout.url(root: root, kind: .audioLatents, entry: "a").path
            == "/cache/audio_latents/a.safetensors")
        #expect(TrainingCacheLayout.url(root: root, kind: .conditions, entry: "a").path
            == "/cache/conditions/a.safetensors")
    }

    /// The on-disk field names, pinned so a tidy-up rename shows up as a visible diff rather
    /// than as a cache that only this build can read.
    @Test("field names match the on-disk layout")
    func fieldNamesMatchTheReference() {
        #expect(TrainingCacheLayout.VideoField.latents == "latents")
        #expect(TrainingCacheLayout.VideoField.numFrames == "num_frames")
        #expect(TrainingCacheLayout.VideoField.fps == "fps")
        #expect(TrainingCacheLayout.AudioField.numTimeSteps == "num_time_steps")
        #expect(TrainingCacheLayout.AudioField.frequencyBins == "frequency_bins")
        #expect(TrainingCacheLayout.ConditionField.videoPromptEmbeds == "video_prompt_embeds")
        #expect(TrainingCacheLayout.ConditionField.audioPromptEmbeds == "audio_prompt_embeds")
        #expect(TrainingCacheLayout.ConditionField.promptAttentionMask
            == "prompt_attention_mask")
    }

    // MARK: - Scalars

    @Test("a frame rate survives the round trip")
    func frameRateRoundTrips() throws {
        for rate in [24.0, 30.0, 23.976023976, 59.94, 25.0] {
            let text = TrainingCacheLayout.format(rate)
            let back = try TrainingCacheLayout.double("fps", in: ["fps": text], entry: "e")
            // Exactly, not nearly: `%.17g` round-trips every double. At 10 digits
            // 23.976023976 came back 4e-09 away — harmless downstream, but it made this
            // property a tolerance rather than a fact.
            #expect(back == rate, "\(rate) -> \(text) -> \(back)" as Comment)
        }
    }

    /// The important half. A missing or corrupt scalar must refuse, not default — a default
    /// frame count is a plausible number attached to the wrong clip.
    @Test("a missing or unparseable scalar is refused, never defaulted")
    func badScalarsAreRefused() {
        #expect(throws: TrainingCacheLayout.Failure.missingField("num_frames", entry: "e")) {
            try TrainingCacheLayout.integer("num_frames", in: [:], entry: "e")
        }
        #expect(throws: TrainingCacheLayout.Failure.self) {
            try TrainingCacheLayout.integer("num_frames", in: ["num_frames": "twenty"],
                                            entry: "e")
        }
        #expect(throws: TrainingCacheLayout.Failure.self) {
            try TrainingCacheLayout.double("fps", in: ["fps": "nan"], entry: "e")
        }
        #expect(throws: TrainingCacheLayout.Failure.self) {
            try TrainingCacheLayout.double("fps", in: ["fps": "inf"], entry: "e")
        }
    }

    // MARK: - Staleness

    private func fingerprint(textEncoder: String = "gemma:1:2",
                             videoVAE: String = "vae:3:4",
                             transformer: String = "dit:5:6",
                             bucket: String = "320x192x97",
                             trigger: String? = nil) -> TrainingCacheLayout.Fingerprint {
        TrainingCacheLayout.Fingerprint(
            textEncoder: textEncoder, videoVAE: videoVAE, audioVAE: nil,
            transformer: transformer, bucket: bucket, trigger: trigger)
    }

    @Test("an identical configuration reports no differences")
    func identicalFingerprintsAgree() {
        #expect(fingerprint().differences(from: fingerprint()).isEmpty)
    }

    /// The hazard this exists for: a different text encoder produces different embeddings,
    /// and a stale `conditions/` trains happily on the wrong ones.
    @Test("a changed text encoder is caught and named")
    func changedTextEncoderIsCaught() {
        let differences = fingerprint()
            .differences(from: fingerprint(textEncoder: "gemma3:1:2"))
        #expect(differences.count == 1)
        #expect(differences[0].contains("text encoder"), "\(differences[0])")
    }

    @Test("every field is compared, not just the obvious ones")
    func everyFieldIsCompared() {
        #expect(fingerprint().differences(from: fingerprint(videoVAE: "other")).count == 1)
        #expect(fingerprint().differences(from: fingerprint(transformer: "other")).count == 1)
        #expect(fingerprint().differences(from: fingerprint(bucket: "640x384x97")).count == 1)
        // A trigger word appearing or disappearing changes every cached caption.
        #expect(fingerprint().differences(from: fingerprint(trigger: "ohwx")).count == 1)
        #expect(fingerprint(trigger: "ohwx").differences(from: fingerprint()).count == 1)
    }

    @Test("several changes are all reported, not just the first")
    func differencesAccumulate() {
        let differences = fingerprint()
            .differences(from: fingerprint(textEncoder: "x", bucket: "y"))
        #expect(differences.count == 2)
    }

    // MARK: - Manifest

    @Test("the manifest round trips and is byte-stable")
    func manifestIsStable() throws {
        // Deliberately unsorted: a manifest whose order came from the file system would make
        // a seeded run unreproducible and the file's checksum meaningless.
        let manifest = TrainingCacheLayout.Manifest(
            fingerprint: fingerprint(),
            entries: ["b", "a", "c"],
            entriesWithAudio: ["c", "a"])
        #expect(manifest.entries == ["a", "b", "c"])
        #expect(manifest.entriesWithAudio == ["a", "c"])

        let once = try manifest.encoded()
        let twice = try TrainingCacheLayout.Manifest.decode(once).encoded()
        #expect(once == twice, "re-encoding changed the bytes")
        #expect(try TrainingCacheLayout.Manifest.decode(once) == manifest)
    }
}

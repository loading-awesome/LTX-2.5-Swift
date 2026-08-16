// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Where a preprocessed training cache lives on disk, and what is in it.
///
/// The VAE and Gemma must run **once per clip, not once per step**. Encoding a 97-frame clip
/// costs seconds and a 26 GB Gemma load; a training run does tens of thousands of steps over
/// the same clips. So the cache is not an optimisation, it is the only arrangement in which
/// training is possible at all.
///
/// ## The schema is fixed, not ours to tidy
///
/// Directory names and field names — `latents/`, `audio_latents/`, `conditions/`, and the
/// fields inside each — follow the established layout for an LTX training cache, so a cache
/// built here holds the same things, under the same names, as one built by any other tool.
///
/// **The container is different.** The usual container is a Python pickle (`.pt`); this
/// writes safetensors. Reading a pickle from Swift means implementing a pickle VM, which is
/// a large and fragile surface for no gain, and this repository already has a deterministic
/// safetensors writer. The consequence is honest and worth stating: a cache written here is
/// not directly consumable by a `.pt`-based trainer, nor one of those by this port, without
/// a converter. What *is* preserved is the schema, so a converter is a field-for-field copy
/// rather than a reinterpretation.
///
/// This type is MLX-free on purpose. Layout, naming, scalar encoding and staleness are
/// exactly the parts where a mistake is silent — a cache keyed by the wrong name, or reused
/// after the checkpoint changed — and they are all decidable without a GPU or a 39 GB file.
public enum TrainingCacheLayout {

    // MARK: - Directories

    /// Video latents, one file per clip.
    public static let videoLatentsDirectory = "latents"
    /// Audio latents, one file per clip that has an audio track.
    public static let audioLatentsDirectory = "audio_latents"
    /// Encoded captions, one file per clip.
    public static let conditionsDirectory = "conditions"

    public static let manifestName = "ltx-cache.json"

    /// The extension this port writes. The usual one is `.pt`; see the type's discussion.
    public static let fileExtension = "safetensors"

    // MARK: - Field names
    //
    // Fixed by the schema, not chosen here. Renaming any of them to something tidier would
    // make the cache readable only by this port, and would make a field-for-field diff
    // against another tool's cache impossible.

    public enum VideoField {
        /// `[C, F', H', W']` — the encoded clip.
        public static let latents = "latents"
        /// Source frame count, **before** the VAE's temporal downsample.
        public static let numFrames = "num_frames"
        public static let height = "height"
        public static let width = "width"
        /// Effective fps: source fps divided by the temporal subsample factor. Downstream
        /// position maths expects the rate the *saved latents* have, not the source's.
        public static let fps = "fps"
    }

    public enum AudioField {
        public static let latents = "latents"
        public static let numTimeSteps = "num_time_steps"
        public static let frequencyBins = "frequency_bins"
        public static let duration = "duration"
    }

    public enum ConditionField {
        /// `[S, width]` — the projected video-stream prompt features, **pre-connector**.
        ///
        /// Pre-connector matters. The connector's weights live in the transformer, not the
        /// text encoder, so caching after it would tie every cached caption to a particular
        /// DiT checkpoint and rule out ever adapting the connector itself. It is also cheap
        /// to run per step, and caching before it means caption encoding needs nothing but
        /// the text encoder loaded.
        public static let videoPromptEmbeds = "video_prompt_embeds"
        public static let audioPromptEmbeds = "audio_prompt_embeds"
        public static let promptAttentionMask = "prompt_attention_mask"
    }

    // MARK: - Naming

    /// The cache-relative name for a media path, with the extension replaced.
    ///
    /// Directory structure is preserved: two clips called `clip.mp4` in different folders
    /// must not collide, and flattening to a basename is exactly how they would.
    public static func entryName(forMediaPath path: String) -> String {
        let url = URL(fileURLWithPath: path)
        let stem = url.deletingPathExtension().relativePath
        // A leading "/" would make `appendingPathComponent` below discard the root.
        return stem.hasPrefix("/") ? String(stem.dropFirst()) : stem
    }

    public enum Kind: String, Sendable, CaseIterable {
        case videoLatents, audioLatents, conditions

        public var directory: String {
            switch self {
            case .videoLatents: return videoLatentsDirectory
            case .audioLatents: return audioLatentsDirectory
            case .conditions: return conditionsDirectory
            }
        }
    }

    public static func url(root: URL, kind: Kind, entry: String) -> URL {
        root.appendingPathComponent(kind.directory)
            .appendingPathComponent(entry)
            .appendingPathExtension(fileExtension)
    }

    // MARK: - Staleness

    /// What a cache was built from.
    ///
    /// Preprocessing tools generally leave this hazard to the operator: rerun with an
    /// overwrite flag when the parameters change, or keep the stale outputs from the previous
    /// run. `conditions/` files in particular are not interchangeable between model versions,
    /// because 2.5's Gemma 4 embeddings differ from 2.3's Gemma 3.
    ///
    /// A stale cache does not fail. It trains, and it converges, on features from the wrong
    /// encoder — so this is recorded and compared instead of trusted.
    public struct Fingerprint: Codable, Equatable, Sendable {
        /// Identity of the text encoder that produced `conditions/`.
        public var textEncoder: String
        /// Identity of the video VAE that produced `latents/`.
        public var videoVAE: String
        /// Identity of the audio VAE that produced `audio_latents/`, if audio was encoded.
        public var audioVAE: String?
        /// Identity of the transformer whose projection produced the prompt embeds.
        public var transformer: String
        /// `WxHxF`, the bucket every entry was resized and cropped to.
        public var bucket: String
        /// Prepended to every caption before encoding, or nil.
        public var trigger: String?

        public init(textEncoder: String, videoVAE: String, audioVAE: String?,
                    transformer: String, bucket: String, trigger: String?) {
            self.textEncoder = textEncoder
            self.videoVAE = videoVAE
            self.audioVAE = audioVAE
            self.transformer = transformer
            self.bucket = bucket
            self.trigger = trigger
        }

        /// Every field that differs, named. Empty means the cache is usable.
        ///
        /// Returns the differences rather than a Bool because "your cache is stale" is not
        /// an actionable message and "the text encoder differs" is.
        public func differences(from other: Fingerprint) -> [String] {
            var out: [String] = []
            func compare(_ label: String, _ a: String?, _ b: String?) {
                guard a != b else { return }
                out.append("\(label): cache has \(a ?? "none"), run wants \(b ?? "none")")
            }
            compare("text encoder", textEncoder, other.textEncoder)
            compare("video VAE", videoVAE, other.videoVAE)
            compare("audio VAE", audioVAE, other.audioVAE)
            compare("transformer", transformer, other.transformer)
            compare("bucket", bucket, other.bucket)
            compare("trigger", trigger, other.trigger)
            return out
        }
    }

    /// The index written beside the cache.
    public struct Manifest: Codable, Equatable, Sendable {
        public var fingerprint: Fingerprint
        /// Entry names, sorted. Sorted so the file is byte-stable across runs — the same
        /// reason ``SafetensorsWriter`` sorts, and the same class of bug it was written for.
        public var entries: [String]
        /// Entries that have an `audio_latents/` file. A subset of ``entries``.
        public var entriesWithAudio: [String]

        public init(fingerprint: Fingerprint, entries: [String], entriesWithAudio: [String]) {
            self.fingerprint = fingerprint
            self.entries = entries.sorted()
            self.entriesWithAudio = entriesWithAudio.sorted()
        }

        public func encoded() throws -> Data {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            return try encoder.encode(self)
        }

        public static func decode(_ data: Data) throws -> Manifest {
            try JSONDecoder().decode(Manifest.self, from: data)
        }
    }

    public static func manifestURL(root: URL) -> URL {
        root.appendingPathComponent(manifestName)
    }

    // MARK: - Scalars
    //
    // safetensors metadata is a string-to-string map, so the scalars stored alongside each
    // tensor are formatted here. Both directions are strict: a cache with an
    // unparseable frame count is refused, never defaulted. A default here would be a
    // plausible number attached to the wrong clip.

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case missingField(String, entry: String)
        case unparseableField(String, value: String, entry: String)
        case fingerprintMismatch([String])

        public var description: String {
            switch self {
            case let .missingField(name, entry):
                return "\(entry): cache entry has no \(name)"
            case let .unparseableField(name, value, entry):
                return "\(entry): \(name) is \(value.debugDescription), which is not a number"
            case let .fingerprintMismatch(differences):
                return """
                    the cache was built with different settings and would train on the wrong \
                    features — \(differences.joined(separator: "; ")). Rebuild it, or point \
                    at a different cache root.
                    """
            }
        }
    }

    public static func format(_ value: Int) -> String { String(value) }

    /// `%.17g` — the shortest precision that round-trips **every** IEEE-754 double exactly.
    ///
    /// 10 significant digits look like plenty for a frame rate and are not: 23.976023976 comes
    /// back 4e-09 away, which is a value near the one that was cached rather than the one that
    /// was cached. Nothing downstream would notice — the resulting position error is ~1e-09
    /// seconds against a 20-second grid — but "survives the round trip" is worth being true
    /// rather than nearly true, and the extra seven characters are written once per clip.
    public static func format(_ value: Double) -> String { String(format: "%.17g", value) }

    public static func integer(_ name: String, in metadata: [String: String],
                               entry: String) throws -> Int {
        guard let raw = metadata[name] else {
            throw Failure.missingField(name, entry: entry)
        }
        guard let value = Int(raw) else {
            throw Failure.unparseableField(name, value: raw, entry: entry)
        }
        return value
    }

    public static func double(_ name: String, in metadata: [String: String],
                              entry: String) throws -> Double {
        guard let raw = metadata[name] else {
            throw Failure.missingField(name, entry: entry)
        }
        guard let value = Double(raw), value.isFinite else {
            throw Failure.unparseableField(name, value: raw, entry: entry)
        }
        return value
    }
}

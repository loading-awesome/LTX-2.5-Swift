// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Where the checkpoints are, so they do not have to travel on the command line.
///
/// Three sources, each overriding the one before: **built-in defaults, then the config
/// file, then a CLI flag.** The last of those is announced when it happens — a path that
/// silently differs from the configured one is how a session ends up benchmarking a file
/// nobody chose. A timing number only means something if you know which file produced it.
///
/// JSON rather than TOML for one reason: `JSONDecoder` is in the standard library and a
/// TOML parser is a dependency. The schema is small enough that the difference is a few
/// quotation marks, and a configuration format is a poor place to spend a supply-chain risk.
///
/// ## Every key is explicit, and that is not style
///
/// `.convertFromSnakeCase` rewrites keys *before* `CodingKeys` are consulted, so the two
/// mechanisms fight and the strategy wins silently. In the port this design is taken from,
/// `video_vae` became `videoVae`, never matched `videoVAE`, and both VAE paths decoded as
/// nil — silently, because every field here is optional. Its doctor then listed eight
/// checkpoints, neither VAE, and looked entirely healthy doing it. A unit test caught it;
/// the integration output did not.
public struct LTXConfiguration: Codable, Sendable, Equatable {

    public struct Checkpoints: Codable, Sendable, Equatable {
        /// Everything else is resolved relative to this unless it starts with `/`.
        public var root: String?
        /// Keyed by transformer role — `dev` and `distilled`. A dictionary rather than two
        /// fields because the recipe names the role, and a recipe added later that wants a
        /// third build should not need a schema change.
        ///
        /// The two files are byte-identical in structure, so nothing downstream can tell
        /// them apart. This table is the only place the association is recorded.
        public var dit: [String: String]
        public var textEncoder: String?
        public var videoVAE: String?
        public var audioVAE: String?
        /// The x2 latent **spatial** upsampler. Not the temporal one that ships beside it.
        public var upsampler: String?
        /// Extra directories to scan for LoRA adapters, beyond the model root.
        ///
        /// Exists because two model generations now coexist on one disk: the 2.3 IC-LoRA
        /// set sits in a sibling tree of the 2.5 checkpoints, and a search rooted at the
        /// transformer's own directory cannot see it. Relative entries hang off `root`.
        public var adapterRoots: [String]

        enum CodingKeys: String, CodingKey {
            case root, dit
            case adapterRoots = "adapter_roots"
            case textEncoder = "text_encoder"
            case videoVAE = "video_vae"
            case audioVAE = "audio_vae"
            case upsampler
        }

        public init(root: String? = nil, dit: [String: String] = [:],
                    textEncoder: String? = nil, videoVAE: String? = nil,
                    audioVAE: String? = nil, upsampler: String? = nil,
                    adapterRoots: [String] = []) {
            self.root = root
            self.dit = dit
            self.adapterRoots = adapterRoots
            self.textEncoder = textEncoder
            self.videoVAE = videoVAE
            self.audioVAE = audioVAE
            self.upsampler = upsampler
        }

        /// Every key optional, defaulting to empty.
        ///
        /// Swift's synthesised `Codable` ignores property defaults for absent keys, so
        /// without this a config naming only the paths its author cares about — a perfectly
        /// reasonable thing to write — fails to decode with `keyNotFound`. A partial config
        /// is the normal case, not an error.
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            root = try c.decodeIfPresent(String.self, forKey: .root)
            dit = try c.decodeIfPresent([String: String].self, forKey: .dit) ?? [:]
            adapterRoots = try c.decodeIfPresent([String].self, forKey: .adapterRoots) ?? []
            textEncoder = try c.decodeIfPresent(String.self, forKey: .textEncoder)
            videoVAE = try c.decodeIfPresent(String.self, forKey: .videoVAE)
            audioVAE = try c.decodeIfPresent(String.self, forKey: .audioVAE)
            upsampler = try c.decodeIfPresent(String.self, forKey: .upsampler)
        }
    }

    public struct Policy: Codable, Sendable, Equatable {
        /// `nil` takes the registry's own default rather than restating it here, so the two
        /// cannot drift apart.
        public var defaultRecipe: String?
        /// Fraction of available memory held back when planning. Memory pressure is not a
        /// cliff the planner can see coming.
        public var memoryMarginFraction: Double

        enum CodingKeys: String, CodingKey {
            case defaultRecipe = "default_recipe"
            case memoryMarginFraction = "memory_margin_fraction"
        }

        public init(defaultRecipe: String? = nil, memoryMarginFraction: Double = 0.15) {
            self.defaultRecipe = defaultRecipe
            self.memoryMarginFraction = memoryMarginFraction
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            defaultRecipe = try c.decodeIfPresent(String.self, forKey: .defaultRecipe)
            memoryMarginFraction =
                try c.decodeIfPresent(Double.self, forKey: .memoryMarginFraction) ?? 0.15
        }
    }

    public var checkpoints: Checkpoints
    public var policy: Policy

    public init(checkpoints: Checkpoints = .init(), policy: Policy = .init()) {
        self.checkpoints = checkpoints
        self.policy = policy
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        checkpoints = try c.decodeIfPresent(Checkpoints.self, forKey: .checkpoints) ?? .init()
        policy = try c.decodeIfPresent(Policy.self, forKey: .policy) ?? .init()
    }

    // MARK: - Roles

    public enum Role: String, Sendable, CaseIterable {
        case ditDev, ditDistilled, textEncoder, videoVAE, audioVAE, upsampler

        /// The flag a caller would use to override this path, for the announcement.
        public var flag: String {
            switch self {
            case .ditDev, .ditDistilled: "--checkpoint"
            case .textEncoder: "--text-encoder"
            case .videoVAE: "--video-vae"
            case .audioVAE: "--audio-vae"
            case .upsampler: "--upsampler"
            }
        }

        public var label: String {
            switch self {
            case .ditDev: "dit (dev)"
            case .ditDistilled: "dit (distilled)"
            case .textEncoder: "text encoder"
            case .videoVAE: "video vae"
            case .audioVAE: "audio vae"
            case .upsampler: "upsampler"
            }
        }
    }

    /// The configured path for a role, before the `root` prefix is applied.
    public func rawPath(_ role: Role) -> String? {
        switch role {
        case .ditDev: checkpoints.dit["dev"]
        case .ditDistilled: checkpoints.dit["distilled"]
        case .textEncoder: checkpoints.textEncoder
        case .videoVAE: checkpoints.videoVAE
        case .audioVAE: checkpoints.audioVAE
        case .upsampler: checkpoints.upsampler
        }
    }

    /// The absolute URL for a role, falling back to the built-in default.
    public func url(_ role: Role) -> URL? {
        resolve(rawPath(role)) ?? LTXConfiguration.builtIn.resolve(
            LTXConfiguration.builtIn.rawPath(role))
    }

    /// Absolute URL for a path that may be relative to `checkpoints.root`.
    public func resolve(_ path: String?) -> URL? {
        guard let path, !path.isEmpty else { return nil }
        if path.hasPrefix("/") { return URL(fileURLWithPath: path) }
        guard let root = checkpoints.root, !root.isEmpty else {
            return URL(fileURLWithPath: path)
        }
        return URL(fileURLWithPath: root).appendingPathComponent(path)
    }

    // MARK: - Loading

    public static var defaultURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/ltx/config.json")
    }

    public enum Failure: Error, CustomStringConvertible {
        case unreadable(path: String, reason: String)

        public var description: String {
            switch self {
            case let .unreadable(path, reason): "cannot read \(path): \(reason)"
            }
        }
    }

    /// Load from `url`, or return the built-in defaults when no file exists.
    ///
    /// A missing config is not an error — passing every path on the command line is a
    /// legitimate way to use this. A config that exists and **cannot be parsed is** an
    /// error, because silently falling back to defaults after somebody edited a file is how
    /// a render ends up using a checkpoint nobody chose.
    public static func load(from url: URL? = nil) throws -> (LTXConfiguration, URL?) {
        let target = url ?? defaultURL
        guard FileManager.default.fileExists(atPath: target.path) else {
            if url != nil {
                throw Failure.unreadable(path: target.path, reason: "no such config file")
            }
            return (builtIn, nil)
        }
        let data: Data
        do { data = try Data(contentsOf: target) } catch {
            throw Failure.unreadable(path: target.path, reason: error.localizedDescription)
        }
        do {
            // No `.convertFromSnakeCase`. See the note on this type.
            return (try JSONDecoder().decode(LTXConfiguration.self, from: data), target)
        } catch {
            throw Failure.unreadable(path: target.path,
                                     reason: "not valid configuration JSON: \(error)")
        }
    }

    /// The configuration in effect: the file `ltx doctor` wrote, or the built-in
    /// defaults when there is none.
    ///
    /// This is the accessor for callers that only want paths and treat a checkpoint that
    /// is not there as a reason to skip — tests, mostly. It **swallows an unreadable
    /// config**, which ``load(from:)`` deliberately does not: a command that is about to
    /// render has to hear that somebody's edit did not parse, while a suite that would
    /// skip anyway gains nothing from the distinction. Use ``load(from:)`` anywhere the
    /// answer matters.
    public static var resolved: LTXConfiguration {
        ((try? load()) ?? (builtIn, nil)).0
    }

    /// The parent of the configured 2.5 tree, where sibling model trees live.
    ///
    /// The 2.3 IC-LoRA set sits beside the 2.5 tree rather than inside it, so a caller
    /// that wants it needs the level above ``Checkpoints/root``. Derived rather than
    /// configured separately, because two independently-set roots drift.
    public static var resolvedModelsRoot: String {
        guard let root = resolved.checkpoints.root else { return defaultModelsRoot }
        return URL(fileURLWithPath: root).deletingLastPathComponent().path
    }

    /// Where the built-in defaults look, before `ltx doctor` writes a config.
    ///
    /// A neutral guess, and it is meant to be replaced rather than matched: the whole
    /// point of the config file is that a machine says where its tree is once, in a file
    /// it can edit, instead of every caller carrying a path.
    public static let defaultModelsRoot = FileManager.default
        .homeDirectoryForCurrentUser.appendingPathComponent("models/ltx").path

    /// The fallback when no config file exists, and the seed `ltx doctor` writes out as
    /// the starting file to edit.
    public static var builtIn: LTXConfiguration { LTXConfiguration(
        checkpoints: .init(
            root: defaultModelsRoot + "/2.5",
            dit: ["dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
                  "distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"],
            textEncoder: "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            videoVAE: "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
            audioVAE: "vae/ltx-2.5-audio-vae-bf16.safetensors",
            upsampler: "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            // The 2.3 IC-LoRA set, which lives beside the 2.5 tree rather than inside it.
            // A directory that does not exist is skipped rather than reported, so listing
            // it costs nothing on a machine that has no 2.3 adapters.
            adapterRoots: [defaultModelsRoot + "/ic-lora-2.3"]),
        policy: .init())
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    /// Write to `url`, creating the directory. Refuses to clobber unless `force`.
    @discardableResult
    public func write(to url: URL, force: Bool = false) throws -> Bool {
        if FileManager.default.fileExists(atPath: url.path), !force { return false }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try encoded().write(to: url)
        return true
    }
}

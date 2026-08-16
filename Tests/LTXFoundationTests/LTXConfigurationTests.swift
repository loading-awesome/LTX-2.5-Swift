// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXFoundation

/// The config file's contract.
///
/// `snakeCaseKeysRoundTrip` is the one that would have caught the bug this schema is
/// designed against: `.convertFromSnakeCase` silently turns `video_vae` into `videoVae`,
/// which never matches `videoVAE`, so both VAE paths decode as nil and the doctor lists
/// eight checkpoints and neither VAE while looking healthy.
@Suite("Configuration")
struct LTXConfigurationTests {

    private func withFile(_ body: (URL) throws -> Void) throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltxcfg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try body(dir.appendingPathComponent("config.json"))
    }

    // MARK: - Encoding

    /// Every acronym key must survive a write/read cycle.
    @Test("snake_case keys round-trip, acronyms included")
    func snakeCaseKeysRoundTrip() throws {
        let data = try LTXConfiguration.builtIn.encoded()
        let text = String(decoding: data, as: UTF8.self)
        for key in ["video_vae", "audio_vae", "text_encoder", "upsampler", "dit", "root"] {
            #expect(text.contains("\"\(key)\""), "missing key \(key) in the encoded form")
        }
        let decoded = try JSONDecoder().decode(LTXConfiguration.self, from: data)
        #expect(decoded == LTXConfiguration.builtIn)
        #expect(decoded.checkpoints.videoVAE != nil, "video_vae must not decode as nil")
        #expect(decoded.checkpoints.audioVAE != nil, "audio_vae must not decode as nil")
    }

    /// A config naming only what its author cares about is the normal case, not an error.
    @Test("a partial config decodes, with the rest falling back")
    func partialConfigDecodes() throws {
        let json = #"{"checkpoints":{"root":"/models","video_vae":"vae/v.safetensors"}}"#
        let config = try JSONDecoder().decode(LTXConfiguration.self,
                                              from: Data(json.utf8))
        #expect(config.checkpoints.root == "/models")
        #expect(config.checkpoints.videoVAE == "vae/v.safetensors")
        #expect(config.checkpoints.textEncoder == nil)
        #expect(config.checkpoints.dit.isEmpty)
        #expect(config.policy.memoryMarginFraction == 0.15, "absent policy takes its default")
    }

    @Test("an empty object decodes to the defaults")
    func emptyObjectDecodes() throws {
        let config = try JSONDecoder().decode(LTXConfiguration.self, from: Data("{}".utf8))
        #expect(config == LTXConfiguration())
    }

    // MARK: - Resolution

    @Test("relative paths hang off root; absolute paths ignore it")
    func rootResolution() {
        let config = LTXConfiguration(checkpoints: .init(root: "/models/ltx"))
        #expect(config.resolve("vae/v.safetensors")?.path == "/models/ltx/vae/v.safetensors")
        #expect(config.resolve("/elsewhere/v.safetensors")?.path == "/elsewhere/v.safetensors")
        #expect(config.resolve(nil) == nil)
        #expect(config.resolve("") == nil)
    }

    @Test("with no root, a relative path stays relative")
    func noRoot() {
        let config = LTXConfiguration(checkpoints: .init(root: nil, videoVAE: "v.safetensors"))
        #expect(config.resolve("v.safetensors")?.path.hasSuffix("v.safetensors") == true)
    }

    @Test("every role resolves from the built-ins")
    func everyRoleResolves() {
        for role in LTXConfiguration.Role.allCases {
            #expect(LTXConfiguration.builtIn.url(role) != nil, "\(role) has no built-in path")
        }
    }

    /// A config that names some paths must not lose the rest.
    @Test("an unset role falls back to the built-in path")
    func unsetRoleFallsBack() {
        let config = LTXConfiguration(
            checkpoints: .init(root: "/models", videoVAE: "mine.safetensors"))
        #expect(config.url(.videoVAE)?.path == "/models/mine.safetensors")
        #expect(config.url(.audioVAE) == LTXConfiguration.builtIn.url(.audioVAE))
    }

    @Test("the two transformer roles read different keys")
    func ditRoles() {
        let config = LTXConfiguration(
            checkpoints: .init(root: "/m", dit: ["dev": "d.safetensors",
                                                 "distilled": "x.safetensors"]))
        #expect(config.url(.ditDev)?.path == "/m/d.safetensors")
        #expect(config.url(.ditDistilled)?.path == "/m/x.safetensors")
    }

    // MARK: - Loading and writing

    @Test("a missing default config is not an error; a missing named one is")
    func missingConfig() throws {
        try withFile { url in
            let (config, source) = try LTXConfiguration.load(from: nil as URL?)
            #expect(source == nil || FileManager.default.fileExists(atPath: source!.path))
            #expect(config.checkpoints.root != nil)

            #expect(throws: LTXConfiguration.Failure.self) {
                _ = try LTXConfiguration.load(from: url)
            }
        }
    }

    /// Falling back to defaults after somebody edited a file is how a render ends up using
    /// a checkpoint nobody chose.
    @Test("a malformed config is an error, never a silent fallback")
    func malformedConfigThrows() throws {
        try withFile { url in
            try Data(#"{"checkpoints": {"#.utf8).write(to: url)
            #expect(throws: LTXConfiguration.Failure.self) {
                _ = try LTXConfiguration.load(from: url)
            }
        }
    }

    @Test("write creates the file, and refuses to clobber without force")
    func writeAndClobber() throws {
        try withFile { url in
            let created = try LTXConfiguration.builtIn.write(to: url)
            #expect(created)
            #expect(FileManager.default.fileExists(atPath: url.path))

            var edited = LTXConfiguration.builtIn
            edited.checkpoints.root = "/edited"
            let clobbered = try edited.write(to: url)
            #expect(!clobbered, "must not overwrite by default")
            let (kept, _) = try LTXConfiguration.load(from: url)
            #expect(kept.checkpoints.root == LTXConfiguration.builtIn.checkpoints.root)

            let forced = try edited.write(to: url, force: true)
            #expect(forced)
            let (rewritten, _) = try LTXConfiguration.load(from: url)
            #expect(rewritten.checkpoints.root == "/edited")
        }
    }

    @Test("write creates missing intermediate directories")
    func writeCreatesDirectories() throws {
        try withFile { url in
            let nested = url.deletingLastPathComponent()
                .appendingPathComponent("a/b/config.json")
            let created = try LTXConfiguration.builtIn.write(to: nested)
            #expect(created)
            #expect(FileManager.default.fileExists(atPath: nested.path))
        }
    }

    @Test("a written config reloads to exactly what was written")
    func roundTripThroughDisk() throws {
        try withFile { url in
            var config = LTXConfiguration.builtIn
            config.checkpoints.root = "/somewhere/else"
            config.policy.defaultRecipe = "distilled"
            config.policy.memoryMarginFraction = 0.25
            _ = try config.write(to: url, force: true)
            let (loaded, source) = try LTXConfiguration.load(from: url)
            #expect(loaded == config)
            #expect(source == url)
        }
    }
}

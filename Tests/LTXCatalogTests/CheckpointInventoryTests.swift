// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXCatalog

/// The inventory's contract: report everything, never throw, and tell the truth about
/// which build of the transformer is loaded.
@Suite("Checkpoint inventory")
struct CheckpointInventoryTests {

    /// Minimal safetensors: 8-byte little-endian header length, JSON header, zero payload.
    /// The payload is required — `SafetensorsHeader` checks `data_offsets` against the size
    /// of the data section, so a header-only file reads as truncated.
    private func write(_ name: String, keys: [String], version: String? = nil,
                       in directory: URL) throws -> URL {
        var entries: [String] = []
        var offset = 0
        for key in keys {
            entries.append("\"\(key)\":{\"dtype\":\"F32\",\"shape\":[2,2],"
                + "\"data_offsets\":[\(offset),\(offset + 16)]}")
            offset += 16
        }
        if let version {
            entries.append("\"__metadata__\":{\"model_version\":\"\(version)\"}")
        }
        let json = "{" + entries.joined(separator: ",") + "}"
        var data = Data()
        withUnsafeBytes(of: UInt64(json.utf8.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: Array(json.utf8))
        data.append(Data(count: offset))
        let url = directory.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    private let ditKeys = ["model.diffusion_model.blocks.0.attn1.to_q.weight"]
    private let audioKeys = ["audio_vae.encoder.conv_in.weight"]

    private func withTree(_ body: (URL) throws -> Void) throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("inventory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    /// The reason this type exists rather than reusing `bind`.
    @Test("every problem is reported, not just the first")
    func reportsAllProblems() throws {
        try withTree { root in
            let audio = try write("audio.safetensors", keys: audioKeys, in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: root.appendingPathComponent("absent.safetensors"),
                      expected: .transformer),
                .init(role: "audio vae", url: audio, expected: .audioVAE),
                .init(role: "video vae", url: root.appendingPathComponent("gone.safetensors"),
                      expected: .videoVAECausal),
            ])
            let failures = rows.filter { if case .failure = $0.result { return true }; return false }
            #expect(failures.count == 2, "both missing files should be reported in one pass")
            #expect(rows[1].result.isSuccess)
        }
    }

    @Test("a file in the wrong slot is a reported mismatch, not a pass")
    func wrongComponent() throws {
        try withTree { root in
            let audio = try write("audio.safetensors", keys: audioKeys, in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: audio, expected: .transformer),
            ])
            guard case let .failure(mismatch) = rows[0].result else {
                Issue.record("expected a mismatch"); return
            }
            #expect(rows[0].isWrongComponent)
            #expect("\(mismatch)".contains("audioVAE"))
        }
    }

    /// The dev/distilled discriminator. Byte-identical structure means the filename is the
    /// only signal, so this check is the only thing standing between a caller and a render
    /// against the transformer they did not choose.
    @Test("the wrong transformer build is caught by name, and says why")
    func wrongVariant() throws {
        try withTree { root in
            let distilled = try write("ltx-2.5-22b-distilled-transformer-bf16.safetensors",
                                      keys: ditKeys, in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: distilled, expected: .transformer,
                      nameMustContain: "dev"),
            ])
            guard case let .failure(mismatch) = rows[0].result else {
                Issue.record("expected a mismatch"); return
            }
            let text = "\(mismatch)"
            #expect(text.contains("'dev'"))
            #expect(text.contains("byte-identical"),
                    "the message must explain that the filename is the only signal")
        }
    }

    @Test("the right build passes the name check")
    func rightVariantPasses() throws {
        try withTree { root in
            let dev = try write("ltx-2.5-22b-dev-transformer-bf16.safetensors",
                                keys: ditKeys, in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: dev, expected: .transformer, nameMustContain: "dev"),
            ])
            #expect(rows[0].result.isSuccess)
        }
    }

    @Test("components are unioned across the rows that resolved")
    func componentsUnion() throws {
        try withTree { root in
            let dit = try write("dit.safetensors", keys: ditKeys, in: root)
            let audio = try write("audio.safetensors", keys: audioKeys, in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: dit, expected: .transformer),
                .init(role: "audio vae", url: audio, expected: .audioVAE),
                .init(role: "video vae", url: root.appendingPathComponent("nope.safetensors"),
                      expected: .videoVAECausal),
            ])
            let found = CheckpointInventory.components(rows)
            #expect(found.contains(.transformer))
            #expect(found.contains(.audioVAE))
            #expect(!found.contains(.videoVAECausal), "a missing file contributes nothing")
        }
    }

    /// Mixing releases is a shape error three stages later, not a load failure.
    @Test("disagreeing model versions are detected")
    func versionDisagreement() throws {
        try withTree { root in
            let dit = try write("dit.safetensors", keys: ditKeys, version: "2.5.0", in: root)
            let audio = try write("audio.safetensors", keys: audioKeys, version: "2.3.0",
                                  in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: dit, expected: .transformer),
                .init(role: "audio vae", url: audio, expected: .audioVAE),
            ])
            let disagreement = CheckpointInventory.versionDisagreement(rows)
            #expect(disagreement?.count == 2)
        }
    }

    @Test("matching model versions are not reported")
    func versionsAgree() throws {
        try withTree { root in
            let dit = try write("dit.safetensors", keys: ditKeys, version: "2.5.0", in: root)
            let audio = try write("audio.safetensors", keys: audioKeys, version: "2.5.0",
                                  in: root)
            let rows = CheckpointInventory.rows([
                .init(role: "dit", url: dit, expected: .transformer),
                .init(role: "audio vae", url: audio, expected: .audioVAE),
            ])
            #expect(CheckpointInventory.versionDisagreement(rows) == nil)
        }
    }
}

private extension Result {
    var isSuccess: Bool { if case .success = self { return true }; return false }
}

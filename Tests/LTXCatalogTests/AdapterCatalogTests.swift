// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXCatalog

/// Adapter discovery and hint resolution, against synthesised headers.
///
/// Header-only files, so these run in milliseconds and need no checkpoint. The behaviour
/// worth protecting is that candidacy is decided by the **key set** and not the filename:
/// the 2.5 tree ships a spatial and a temporal latent upscaler whose names differ by one
/// word and which are not adapters at all.
@Suite("Adapter catalogue")
struct AdapterCatalogTests {

    /// A minimal valid safetensors file: 8-byte little-endian header length, the JSON
    /// header, then a zero payload.
    ///
    /// The payload is not optional even though nothing here reads it: `SafetensorsHeader`
    /// checks every entry's `data_offsets` against the size of the data section, so a
    /// header-only file is rejected as truncated rather than parsed.
    private func write(_ name: String, keys: [String], in directory: URL) throws {
        var entries: [String] = []
        var offset = 0
        for key in keys {
            entries.append("\"\(key)\":{\"dtype\":\"F32\",\"shape\":[2,2],"
                + "\"data_offsets\":[\(offset),\(offset + 16)]}")
            offset += 16
        }
        let json = "{" + entries.joined(separator: ",") + "}"
        var data = Data()
        withUnsafeBytes(of: UInt64(json.utf8.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: Array(json.utf8))
        data.append(Data(count: offset))
        try data.write(to: directory.appendingPathComponent(name))
    }

    /// Keys that make a file read as a LoRA to `CheckpointIdentity`.
    private let loraKeys = ["diffusion_model.blocks.0.attn1.to_q.lora_A.weight",
                            "diffusion_model.blocks.0.attn1.to_q.lora_B.weight"]

    private func withTree(_ body: (URL) throws -> Void) throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("adapter-catalogue-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    @Test("only files whose keys carry LoRA tensors are discovered")
    func discoversByKeySet() throws {
        try withTree { root in
            try write("an-adapter.safetensors", keys: loraKeys, in: root)
            // Named like an adapter, keyed like an upscaler. This is the file the catalogue
            // exists to refuse.
            try write("ltx-2.5-latent-temporal-upscaler-x2.safetensors",
                      keys: ["initial_conv.weight", "post_upsample_res_blocks.0.weight"],
                      in: root)

            let found = try AdapterCatalog.discover(in: root)
            #expect(found.count == 1)
            #expect(found.first?.name == "an-adapter.safetensors")
        }
    }

    @Test("a hint resolves to the one adapter that matches")
    func resolvesHint() throws {
        try withTree { root in
            try write("ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
                      keys: loraKeys, in: root)
            try write("ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2.safetensors",
                      keys: loraKeys, in: root)

            #expect(try AdapterCatalog.resolve(hint: "distilled-lora-450", in: root).name
                == "ltx-2.5-22b-distilled-lora-450-bf16.safetensors")
            #expect(try AdapterCatalog.resolve(hint: "pixel-spatial", in: root).name
                == "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2.safetensors")
        }
    }

    @Test("hint matching ignores case")
    func caseInsensitive() throws {
        try withTree { root in
            try write("Distilled-LoRA-450.safetensors", keys: loraKeys, in: root)
            #expect(try AdapterCatalog.resolve(hint: "distilled-lora-450", in: root).name
                == "Distilled-LoRA-450.safetensors")
        }
    }

    @Test("a hint matching two adapters is refused and names both")
    func ambiguousRefused() throws {
        try withTree { root in
            try write("lora-alpha.safetensors", keys: loraKeys, in: root)
            try write("lora-beta.safetensors", keys: loraKeys, in: root)
            #expect(throws: AdapterCatalog.Failure.self) {
                try AdapterCatalog.resolve(hint: "lora", in: root)
            }
        }
    }

    /// An exact basename beats a substring, so a hint that is also a prefix of a longer
    /// sibling still resolves instead of reporting an ambiguity the caller cannot act on.
    @Test("an exact basename wins over a longer sibling")
    func exactBeatsSubstring() throws {
        try withTree { root in
            try write("detail.safetensors", keys: loraKeys, in: root)
            try write("detail-v2.safetensors", keys: loraKeys, in: root)
            #expect(try AdapterCatalog.resolve(hint: "detail", in: root).name
                == "detail.safetensors")
        }
    }

    @Test("a hint matching nothing says where it looked and what was there")
    func noMatchRefused() throws {
        try withTree { root in
            try write("lora-alpha.safetensors", keys: loraKeys, in: root)
            #expect(throws: AdapterCatalog.Failure.self) {
                try AdapterCatalog.resolve(hint: "inpainting", in: root)
            }
        }
    }

    @Test("verify refuses a named path that holds no LoRA tensors")
    func verifyRefusesNonAdapter() throws {
        try withTree { root in
            try write("upscaler.safetensors",
                      keys: ["initial_conv.weight", "post_upsample_res_blocks.0.weight"],
                      in: root)
            #expect(throws: AdapterCatalog.Failure.self) {
                try AdapterCatalog.verify(at: root.appendingPathComponent("upscaler.safetensors"))
            }
            try write("real.safetensors", keys: loraKeys, in: root)
            #expect(try AdapterCatalog.verify(
                at: root.appendingPathComponent("real.safetensors")).name == "real.safetensors")
        }
    }

    @Test("adapters are found in subdirectories, as the 2.5 tree lays them out")
    func findsNested() throws {
        try withTree { root in
            let nested = root.appendingPathComponent("ic-lora/pixel-spatial-upscaler")
            try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
            try write("ic-lora-pixel-spatial-upscaler-x2.safetensors", keys: loraKeys, in: nested)
            #expect(try AdapterCatalog.resolve(hint: "pixel-spatial", in: root).name
                == "ic-lora-pixel-spatial-upscaler-x2.safetensors")
        }
    }

    @Test("the model root is the checkpoint's grandparent")
    func modelRoot() {
        let dit = URL(fileURLWithPath: "/models/ltx/2.5/diffusion_models/dev.safetensors")
        #expect(AdapterCatalog.modelRoot(containing: dit).path == "/models/ltx/2.5")
    }
}

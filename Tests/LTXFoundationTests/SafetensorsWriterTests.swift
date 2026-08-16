// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXFoundation

/// Byte reproducibility, and that what is written is what this repo's own reader parses.
///
/// `sameTensorsSameBytesRegardlessOfInsertionOrder` is the test that matters. The bug it
/// pins is not hypothetical: two renders at the same seed produced bit-identical tensors in
/// files with different `data_offsets`, because `MLX.save` iterates a Swift dictionary and
/// Swift randomises that per process.
@Suite("Safetensors writer")
struct SafetensorsWriterTests {

    private func entry(_ name: String, _ dtype: SafetensorsHeader.DType,
                       _ shape: [Int]) -> SafetensorsWriter.Entry {
        SafetensorsWriter.Entry(name: name, dtype: dtype, shape: shape)
    }

    private func bytes(_ n: Int, fill: UInt8 = 0xAB) -> Data {
        Data(repeating: fill, count: n)
    }

    // MARK: - Reproducibility

    /// The whole point: order of assembly must not reach the file.
    @Test("the same tensors produce the same bytes whatever order they arrive in")
    func sameTensorsSameBytesRegardlessOfInsertionOrder() throws {
        let a = [entry("waveform", .f32, [1, 2, 4]), entry("mel", .f32, [1, 2, 3])]
        let b = [entry("mel", .f32, [1, 2, 3]), entry("waveform", .f32, [1, 2, 4])]
        let payload = ["waveform": bytes(8 * 4), "mel": bytes(6 * 4, fill: 0x11)]

        let fileA = try SafetensorsWriter.file(a, bytes: payload)
        let fileB = try SafetensorsWriter.file(b, bytes: payload)
        #expect(fileA == fileB)
    }

    @Test("data is laid out in sorted name order, contiguously from zero")
    func sortedContiguousLayout() throws {
        let plan = try SafetensorsWriter.layout([
            entry("zeta", .f32, [2]), entry("alpha", .f32, [3]), entry("mid", .u8, [5]),
        ])
        #expect(plan.order == ["alpha", "mid", "zeta"])
        #expect(plan.dataByteCount == 3 * 4 + 5 + 2 * 4)

        let header = try SafetensorsHeader.parse(json: plan.prefix.dropFirst(8),
                                                 dataStart: plan.prefix.count,
                                                 dataSize: plan.dataByteCount)
        #expect(header.tensors["alpha"]?.start == 0)
        #expect(header.tensors["mid"]?.start == 12)
        #expect(header.tensors["zeta"]?.start == 17)
    }

    @Test("metadata is included when present and omitted when empty")
    func metadataHandling() throws {
        let plain = try SafetensorsWriter.layout([entry("t", .f32, [1])])
        let text = String(decoding: plain.prefix.dropFirst(8), as: UTF8.self)
        #expect(!text.contains("__metadata__"))

        let tagged = try SafetensorsWriter.layout([entry("t", .f32, [1])],
                                                  metadata: ["model_version": "2.5.0"])
        let header = try SafetensorsHeader.parse(json: tagged.prefix.dropFirst(8),
                                                 dataStart: tagged.prefix.count,
                                                 dataSize: tagged.dataByteCount)
        #expect(header.metadata["model_version"] == "2.5.0")
    }

    // MARK: - Round trip through this repo's reader

    @Test("a written file parses back with the same shapes, dtypes and bytes")
    func roundTripThroughReader() throws {
        let entries = [entry("video", .f32, [1, 2, 3]), entry("flags", .u8, [4])]
        let videoBytes = Data((0 ..< 24).map { UInt8($0) })
        let flagBytes = Data([9, 8, 7, 6])
        let file = try SafetensorsWriter.file(entries,
                                              bytes: ["video": videoBytes, "flags": flagBytes],
                                              metadata: ["k": "v"])

        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("stw-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("t.safetensors")
        try file.write(to: url)

        let header = try SafetensorsHeader.read(from: url)
        #expect(header.tensors.count == 2)
        #expect(header.tensors["video"]?.shape == [1, 2, 3])
        #expect(header.tensors["video"]?.dtype == .f32)
        #expect(header.tensors["flags"]?.shape == [4])
        #expect(header.metadata["k"] == "v")

        // And the bytes land where the header says they do.
        let raw = try Data(contentsOf: url)
        let flags = header.tensors["flags"]!
        let start = header.dataStart + flags.start
        #expect(raw[start ..< start + flags.byteCount] == flagBytes)
    }

    @Test("a zero-element tensor is legal and occupies no bytes")
    func emptyTensor() throws {
        let file = try SafetensorsWriter.file([entry("empty", .f32, [0]),
                                               entry("real", .f32, [2])],
                                              bytes: ["empty": Data(), "real": bytes(8)])
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("stw-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("t.safetensors")
        try file.write(to: url)
        let header = try SafetensorsHeader.read(from: url)
        #expect(header.tensors["empty"]?.byteCount == 0)
    }

    // MARK: - Refusals

    @Test("a byte count that disagrees with the shape is refused")
    func byteCountMismatchRefused() {
        #expect(throws: SafetensorsWriter.Failure.self) {
            try SafetensorsWriter.file([entry("t", .f32, [4])], bytes: ["t": Data(count: 8)])
        }
    }

    @Test("a missing payload is refused rather than written as zeros")
    func missingPayloadRefused() {
        #expect(throws: SafetensorsWriter.Failure.self) {
            try SafetensorsWriter.file([entry("t", .f32, [4])], bytes: [:])
        }
    }

    @Test("duplicate names and the reserved name are refused")
    func nameRefusals() {
        #expect(throws: SafetensorsWriter.Failure.self) {
            try SafetensorsWriter.layout([entry("t", .f32, [1]), entry("t", .f32, [1])])
        }
        #expect(throws: SafetensorsWriter.Failure.self) {
            try SafetensorsWriter.layout([entry("__metadata__", .f32, [1])])
        }
    }

    @Test("a negative dimension is refused")
    func negativeDimensionRefused() {
        #expect(throws: SafetensorsWriter.Failure.self) {
            try SafetensorsWriter.layout([entry("t", .f32, [2, -1])])
        }
    }

    /// Widths come from `SafetensorsHeader.DType`, so the two cannot disagree about how many
    /// bytes a tensor occupies.
    @Test("byte counts follow the dtype width")
    func byteWidths() {
        #expect(entry("t", .f32, [10]).byteCount == 40)
        #expect(entry("t", .bf16, [10]).byteCount == 20)
        #expect(entry("t", .u8, [10]).byteCount == 10)
        #expect(entry("t", .f64, [10]).byteCount == 80)
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// These run in milliseconds with no GPU and no checkpoint, which is the point
/// of the MLX-free layer: header parsing is where a silent error stays silent.
///
/// Every malformed case here is something a real corrupt or hostile file does.
/// A parser that only ever sees well-formed input is no evidence that it
/// rejects anything.
@Suite("Safetensors header")
struct SafetensorsHeaderTests {

    /// Build a header blob the way safetensors does: 8-byte LE length, JSON,
    /// then payload.
    static func blob(_ entries: [String: Any], dataSize: Int) throws -> (Data, Int, Int) {
        let json = try JSONSerialization.data(withJSONObject: entries)
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        return (out, 8 + json.count, dataSize)
    }

    static func parse(_ entries: [String: Any], dataSize: Int) throws -> SafetensorsHeader {
        let json = try JSONSerialization.data(withJSONObject: entries)
        return try SafetensorsHeader.parse(json: json, dataStart: 8 + json.count,
                                           dataSize: dataSize)
    }

    @Test("a well-formed header parses, with shapes and offsets intact")
    func wellFormed() throws {
        let header = try Self.parse([
            "__metadata__": ["model_version": "2.5.0"],
            "a.weight": ["dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]],
            "a.bias": ["dtype": "F32", "shape": [3], "data_offsets": [12, 24]],
        ], dataSize: 24)

        #expect(header.tensors.count == 2)
        #expect(header.metadata["model_version"] == "2.5.0")
        let w = try #require(header.tensors["a.weight"])
        #expect(w.shape == [2, 3])
        #expect(w.dtype == .bf16)
        #expect(w.elementCount == 6)
        #expect(w.byteCount == 12)
        #expect(header.parameterCount == 9)
        #expect(header.declaredDataSize == 24)
    }

    @Test("prefix queries drive component detection")
    func prefixes() throws {
        let header = try Self.parse([
            "vision_model.patch_dense.weight": ["dtype": "BF16", "shape": [1], "data_offsets": [0, 2]],
            "text_embedding_projection.w": ["dtype": "BF16", "shape": [1], "data_offsets": [2, 4]],
        ], dataSize: 4)

        #expect(header.hasPrefix("vision_model."))
        #expect(header.hasPrefix("text_embedding_projection."))
        #expect(!header.hasPrefix("audio_projector."))
        #expect(header.names(withPrefix: "vision_model.") == ["vision_model.patch_dense.weight"])
    }

    @Test("a scalar has one element, not zero")
    func scalarShape() throws {
        let header = try Self.parse([
            "s": ["dtype": "F32", "shape": [], "data_offsets": [0, 4]],
        ], dataSize: 4)
        #expect(header.tensors["s"]?.elementCount == 1)
    }

    // MARK: - Rejections. Each is a real way a file goes wrong.

    @Test("a shape that disagrees with its byte range is refused")
    func sizeMismatch() throws {
        // [2,3] BF16 needs 12 bytes; the entry claims 8. Believing the range and
        // allocating from the shape is how a header overruns a buffer.
        #expect(throws: SafetensorsHeader.Failure.sizeMismatch(
            name: "t", declaredBytes: 8, shapeBytes: 12)) {
            try Self.parse(["t": ["dtype": "BF16", "shape": [2, 3],
                                  "data_offsets": [0, 8]]], dataSize: 8)
        }
    }

    @Test("a range past the end of the data section is refused")
    func pastEnd() throws {
        #expect(throws: SafetensorsHeader.Failure.rangeOutsideData(
            name: "t", end: 64, dataSize: 16)) {
            try Self.parse(["t": ["dtype": "F32", "shape": [16],
                                  "data_offsets": [0, 64]]], dataSize: 16)
        }
    }

    @Test("a shape that overflows Int is refused, not wrapped")
    func overflow() throws {
        let huge = Int.max / 2 + 1
        #expect(throws: SafetensorsHeader.Failure.elementCountOverflow(
            name: "t", shape: [huge, 4])) {
            try Self.parse(["t": ["dtype": "U8", "shape": [huge, 4],
                                  "data_offsets": [0, 1]]], dataSize: 1)
        }
    }

    @Test("a negative dimension is refused")
    func negativeDim() throws {
        #expect(throws: SafetensorsHeader.Failure.negativeDimension(name: "t", shape: [-1, 4])) {
            try Self.parse(["t": ["dtype": "U8", "shape": [-1, 4],
                                  "data_offsets": [0, 4]]], dataSize: 4)
        }
    }

    @Test("a decreasing byte range is refused")
    func badRange() throws {
        #expect(throws: SafetensorsHeader.Failure.badRange(name: "t", start: 16, end: 4)) {
            try Self.parse(["t": ["dtype": "U8", "shape": [4],
                                  "data_offsets": [16, 4]]], dataSize: 32)
        }
    }

    @Test("an unknown dtype is named, not swallowed")
    func unknownDType() throws {
        #expect(throws: SafetensorsHeader.Failure.unknownDType(name: "t", dtype: "FP4_MYSTERY")) {
            try Self.parse(["t": ["dtype": "FP4_MYSTERY", "shape": [4],
                                  "data_offsets": [0, 4]]], dataSize: 4)
        }
    }

    @Test("missing fields are reported by field name")
    func missingFields() throws {
        #expect(throws: SafetensorsHeader.Failure.missingField(name: "t", field: "dtype")) {
            try Self.parse(["t": ["shape": [4], "data_offsets": [0, 4]]], dataSize: 4)
        }
        #expect(throws: SafetensorsHeader.Failure.missingField(name: "t", field: "data_offsets")) {
            try Self.parse(["t": ["dtype": "U8", "shape": [4]]], dataSize: 4)
        }
    }

    // MARK: - File-level reading

    @Test("reading a real file on disk round-trips, and a truncated one throws")
    func fileReading() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ltx-sft-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let (prefix, _, _) = try Self.blob([
            "__metadata__": ["model_version": "2.5.0"],
            "w": ["dtype": "BF16", "shape": [4], "data_offsets": [0, 8]],
        ], dataSize: 8)

        let good = dir.appendingPathComponent("good.safetensors")
        try (prefix + Data(count: 8)).write(to: good)
        let header = try SafetensorsHeader.read(from: good)
        #expect(header.tensors["w"]?.elementCount == 4)
        #expect(header.metadata["model_version"] == "2.5.0")
        // The payload begins right after the JSON, and nothing above read it.
        #expect(header.tensors["w"]?.fileOffset(in: header) == header.dataStart)

        // Truncated: the declared JSON length runs past the end of the file. This
        // is what a half-finished download looks like, and it must not be read as
        // a valid header of whatever bytes happen to be present.
        let short = dir.appendingPathComponent("short.safetensors")
        try prefix.prefix(12).write(to: short)
        #expect(throws: (any Error).self) { try SafetensorsHeader.read(from: short) }

        let empty = dir.appendingPathComponent("empty.safetensors")
        try Data().write(to: empty)
        #expect(throws: SafetensorsHeader.Failure.tooSmall(fileSize: 0)) {
            try SafetensorsHeader.read(from: empty)
        }
    }

    @Test("every dtype has a plausible byte width")
    func dtypeWidths() {
        #expect(SafetensorsHeader.DType.bf16.byteWidth == 2)
        #expect(SafetensorsHeader.DType.f32.byteWidth == 4)
        #expect(SafetensorsHeader.DType.f8e4m3.byteWidth == 1)
        #expect(SafetensorsHeader.DType.i64.byteWidth == 8)
        #expect(SafetensorsHeader.DType.allCases.allSatisfy { $0.byteWidth > 0 })
    }
}

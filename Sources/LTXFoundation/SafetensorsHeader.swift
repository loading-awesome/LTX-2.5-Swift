// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// A safetensors file's header, read without mapping the payload.
///
/// The LTX 2.5 transformer is 39 GiB. Nothing here reads it. The header is two
/// ranged reads — an 8-byte little-endian length, then that many bytes of JSON —
/// which is the same thing `tools/ltx_probe.py` does over HTTP, and the reason
/// the probe can answer architecture questions without a download.
///
/// ## Checkpoints are untrusted input
///
/// A header is attacker-controlled data that tells us how much memory to
/// allocate and where to read from. So every field is validated *before* it is
/// believed, and the failures are typed rather than trapped:
///
/// * a header length that does not fit inside the file
/// * a tensor whose byte range runs past the end of the data section
/// * a shape whose element count times its dtype size disagrees with its byte
///   range — the mismatch that turns into a buffer overrun downstream
/// * a negative dimension, or an element count that overflows `Int`
///
/// `precondition` is deliberately not used for any of it: a malformed
/// third-party checkpoint must be a thrown error a caller can report, not a
/// crash.
///
/// ## Read dims from here, never declare them
///
/// The 2.5 PR's own headline fix was replacing a hardcoded `3840 * 49` with
/// `hidden_size * (num_hidden_layers + 1)`. That constant is still numerically
/// correct for this checkpoint, which is exactly why hardcoding it is
/// dangerous — it would keep being right until it silently was not. Every
/// consumer of this type should read the shape it needs.
public struct SafetensorsHeader: Sendable, Equatable {

    /// Element types safetensors can declare. The LTX 2.5 set uses `BF16`,
    /// `F32`, `I8`, `U8` and `F8_E4M3`; the rest are here so an unexpected
    /// dtype is *named* in an error rather than reported as unparseable.
    public enum DType: String, Sendable, CaseIterable {
        case bool = "BOOL"
        case u8 = "U8", i8 = "I8"
        case f8e4m3 = "F8_E4M3", f8e5m2 = "F8_E5M2"
        case u16 = "U16", i16 = "I16", f16 = "F16", bf16 = "BF16"
        case u32 = "U32", i32 = "I32", f32 = "F32"
        case u64 = "U64", i64 = "I64", f64 = "F64"

        /// Bytes per element.
        public var byteWidth: Int {
            switch self {
            case .bool, .u8, .i8, .f8e4m3, .f8e5m2: return 1
            case .u16, .i16, .f16, .bf16: return 2
            case .u32, .i32, .f32: return 4
            case .u64, .i64, .f64: return 8
            }
        }
    }

    public struct TensorEntry: Sendable, Equatable {
        public let name: String
        public let dtype: DType
        public let shape: [Int]
        /// Byte range within the data section, i.e. relative to `dataStart`.
        public let start: Int
        public let end: Int

        public var elementCount: Int { shape.reduce(1, *) }
        public var byteCount: Int { end - start }
        /// Absolute file offset of this tensor's first byte.
        public func fileOffset(in header: SafetensorsHeader) -> Int { header.dataStart + start }
    }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case tooSmall(fileSize: Int)
        case headerLengthOutOfRange(declared: Int, fileSize: Int)
        case headerNotJSONObject
        case entryNotObject(name: String)
        case missingField(name: String, field: String)
        case unknownDType(name: String, dtype: String)
        case negativeDimension(name: String, shape: [Int])
        case elementCountOverflow(name: String, shape: [Int])
        case badRange(name: String, start: Int, end: Int)
        case rangeOutsideData(name: String, end: Int, dataSize: Int)
        case sizeMismatch(name: String, declaredBytes: Int, shapeBytes: Int)

        public var description: String {
            switch self {
            case let .tooSmall(size):
                return "file is \(size) bytes; a safetensors header needs at least 8"
            case let .headerLengthOutOfRange(declared, fileSize):
                return "header declares \(declared) bytes of JSON but the file is \(fileSize)"
            case .headerNotJSONObject:
                return "header is not a JSON object"
            case let .entryNotObject(name):
                return "entry '\(name)' is not a JSON object"
            case let .missingField(name, field):
                return "entry '\(name)' has no '\(field)'"
            case let .unknownDType(name, dtype):
                return "entry '\(name)' has unknown dtype '\(dtype)'"
            case let .negativeDimension(name, shape):
                return "entry '\(name)' has a negative dimension: \(shape)"
            case let .elementCountOverflow(name, shape):
                return "entry '\(name)' shape \(shape) overflows Int"
            case let .badRange(name, start, end):
                return "entry '\(name)' has a non-increasing byte range \(start)..<\(end)"
            case let .rangeOutsideData(name, end, dataSize):
                return "entry '\(name)' ends at \(end), past the \(dataSize)-byte data section"
            case let .sizeMismatch(name, declaredBytes, shapeBytes):
                return "entry '\(name)' declares \(declaredBytes) bytes but its shape and "
                    + "dtype need \(shapeBytes) — refusing to allocate from a header that "
                    + "disagrees with itself"
            }
        }
    }

    public let tensors: [String: TensorEntry]
    /// The `__metadata__` block, if present. LTX 2.5 uses it for
    /// `model_version`, `gemma_config` and the embedded component configs.
    public let metadata: [String: String]
    /// Absolute file offset where tensor data begins: `8 + headerJSONLength`.
    public let dataStart: Int

    /// Total bytes of tensor payload the header accounts for.
    public var declaredDataSize: Int { tensors.values.map(\.end).max() ?? 0 }

    // MARK: - Reading

    /// Read and validate the header of the file at `url`.
    ///
    /// Two reads, no mapping: safe to call on a 39 GiB checkpoint.
    public static func read(from url: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let fileSize = Int(try handle.seekToEnd())
        guard fileSize >= 8 else { throw Failure.tooSmall(fileSize: fileSize) }
        try handle.seek(toOffset: 0)

        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw Failure.tooSmall(fileSize: fileSize)
        }
        let declared = lengthBytes.withUnsafeBytes { raw -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
        // Reject before allocating. A 64-bit length from an untrusted file can
        // ask for more memory than the machine has.
        guard declared > 0, declared <= UInt64(fileSize - 8), declared <= UInt64(Int.max) else {
            throw Failure.headerLengthOutOfRange(declared: Int(clamping: declared),
                                                 fileSize: fileSize)
        }
        let headerLength = Int(declared)
        guard let json = try handle.read(upToCount: headerLength),
              json.count == headerLength else {
            throw Failure.headerLengthOutOfRange(declared: headerLength, fileSize: fileSize)
        }

        let dataSize = fileSize - 8 - headerLength
        return try parse(json: json, dataStart: 8 + headerLength, dataSize: dataSize)
    }

    /// Parse a header JSON blob. Split out so it is testable without a file.
    public static func parse(json: Data, dataStart: Int, dataSize: Int) throws -> SafetensorsHeader {
        guard let root = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.headerNotJSONObject
        }

        var metadata: [String: String] = [:]
        if let raw = root["__metadata__"] as? [String: Any] {
            for (key, value) in raw {
                if let s = value as? String { metadata[key] = s }
                else if let data = try? JSONSerialization.data(withJSONObject: value),
                        let s = String(data: data, encoding: .utf8) { metadata[key] = s }
            }
        }

        var tensors: [String: TensorEntry] = [:]
        for (name, raw) in root where name != "__metadata__" {
            guard let entry = raw as? [String: Any] else { throw Failure.entryNotObject(name: name) }
            guard let dtypeString = entry["dtype"] as? String else {
                throw Failure.missingField(name: name, field: "dtype")
            }
            guard let dtype = DType(rawValue: dtypeString) else {
                throw Failure.unknownDType(name: name, dtype: dtypeString)
            }
            guard let shapeRaw = entry["shape"] as? [Any] else {
                throw Failure.missingField(name: name, field: "shape")
            }
            let shape = shapeRaw.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeRaw.count else {
                throw Failure.missingField(name: name, field: "shape")
            }
            guard shape.allSatisfy({ $0 >= 0 }) else {
                throw Failure.negativeDimension(name: name, shape: shape)
            }
            guard let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                  let start = (offsets[0] as? NSNumber)?.intValue,
                  let end = (offsets[1] as? NSNumber)?.intValue else {
                throw Failure.missingField(name: name, field: "data_offsets")
            }
            guard start >= 0, end >= start else {
                throw Failure.badRange(name: name, start: start, end: end)
            }
            guard end <= dataSize else {
                throw Failure.rangeOutsideData(name: name, end: end, dataSize: dataSize)
            }

            // Element count first, guarding the multiply: a crafted shape like
            // [2^40, 2^40] must be an error, not a wrap to something plausible.
            var count = 1
            for dim in shape {
                let (product, overflow) = count.multipliedReportingOverflow(by: dim)
                guard !overflow else { throw Failure.elementCountOverflow(name: name, shape: shape) }
                count = product
            }
            let (shapeBytes, widthOverflow) = count.multipliedReportingOverflow(by: dtype.byteWidth)
            guard !widthOverflow else { throw Failure.elementCountOverflow(name: name, shape: shape) }
            guard shapeBytes == end - start else {
                throw Failure.sizeMismatch(name: name, declaredBytes: end - start,
                                           shapeBytes: shapeBytes)
            }

            tensors[name] = TensorEntry(name: name, dtype: dtype, shape: shape,
                                        start: start, end: end)
        }

        return SafetensorsHeader(tensors: tensors, metadata: metadata, dataStart: dataStart)
    }

    // MARK: - Queries

    /// Tensor names sharing a prefix. The 2.5 files are organised by component
    /// prefix (`vision_model.`, `text_embedding_projection.`, …).
    public func names(withPrefix prefix: String) -> [String] {
        tensors.keys.filter { $0.hasPrefix(prefix) }.sorted()
    }

    /// Whether any tensor name begins with `prefix`. Component detection is
    /// key-presence based, exactly as the probe does it.
    public func hasPrefix(_ prefix: String) -> Bool {
        tensors.keys.contains { $0.hasPrefix(prefix) }
    }

    /// Total declared parameter count across every tensor.
    public var parameterCount: Int { tensors.values.reduce(0) { $0 + $1.elementCount } }

    /// A metadata value parsed as JSON, for the embedded component configs.
    public func metadataJSON(_ key: String) -> [String: Any]? {
        guard let raw = metadata[key], let data = raw.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}

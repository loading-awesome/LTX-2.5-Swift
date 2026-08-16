// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Byte-reproducible safetensors output: the same tensors always produce the same file.
///
/// ## Why this exists
///
/// `MLX.save(arrays:metadata:url:)` takes a Swift `Dictionary`, and Swift randomises
/// dictionary iteration order **per process**. The C++ writer sorts the JSON keys but lays
/// the data blobs down in iteration order, so two runs that produce bit-identical tensors
/// produce files with different `data_offsets` and therefore different checksums. Measured,
/// not theorised — two renders at the same seed differed like this:
///
/// ```
/// run A   waveform [0, 1539840]     mel [1539840, 1745152]
/// run B   mel      [0,  205312]     waveform [205312, 1745152]
/// ```
///
/// Nothing downstream reads by offset, so no result was ever wrong. What it cost is the
/// ability to **checksum an artifact**: the cheapest possible check, "are these two runs
/// the same file", silently answers no.
///
/// ## The rule
///
/// Tensors are laid out **in sorted name order**, contiguously from offset 0, and the header
/// JSON is emitted with sorted keys. Sorting rather than preserving insertion order because
/// insertion order is itself a property of the caller's data structure; a name sort is a
/// property of the content alone, and two callers that assembled the same tensors by
/// different routes should still write the same bytes.
public enum SafetensorsWriter {

    public struct Entry: Sendable, Equatable {
        public let name: String
        public let dtype: SafetensorsHeader.DType
        public let shape: [Int]

        public init(name: String, dtype: SafetensorsHeader.DType, shape: [Int]) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
        }

        public var elementCount: Int { shape.reduce(1, *) }
        public var byteCount: Int { elementCount * dtype.byteWidth }
    }

    public enum Failure: Error, CustomStringConvertible {
        case duplicateName(String)
        case negativeDimension(name: String, shape: [Int])
        case reservedName(String)
        case byteCountMismatch(name: String, declared: Int, supplied: Int)

        public var description: String {
            switch self {
            case let .duplicateName(name):
                return "two tensors are both named '\(name)'"
            case let .negativeDimension(name, shape):
                return "'\(name)' has a negative dimension in \(shape)"
            case let .reservedName(name):
                return "'\(name)' is reserved by the safetensors format"
            case let .byteCountMismatch(name, declared, supplied):
                return "'\(name)' declares \(declared) bytes from its shape and dtype but "
                    + "\(supplied) were supplied"
            }
        }
    }

    /// A header plus the order its data must be appended in.
    public struct Layout: Sendable, Equatable {
        /// The 8-byte length prefix and the JSON header, ready to write.
        public let prefix: Data
        /// Tensor names in the order their bytes follow `prefix`.
        public let order: [String]
        public let dataByteCount: Int
    }

    /// Plan the file. Deterministic in the entries alone.
    public static func layout(_ entries: [Entry],
                              metadata: [String: String] = [:]) throws -> Layout {
        var seen = Set<String>()
        for entry in entries {
            guard entry.name != "__metadata__" else { throw Failure.reservedName(entry.name) }
            guard seen.insert(entry.name).inserted else {
                throw Failure.duplicateName(entry.name)
            }
            guard entry.shape.allSatisfy({ $0 >= 0 }) else {
                throw Failure.negativeDimension(name: entry.name, shape: entry.shape)
            }
        }

        let sorted = entries.sorted { $0.name < $1.name }
        var root: [String: Any] = [:]
        if !metadata.isEmpty { root["__metadata__"] = metadata }

        var offset = 0
        for entry in sorted {
            let end = offset + entry.byteCount
            root[entry.name] = [
                "dtype": entry.dtype.rawValue,
                "shape": entry.shape,
                "data_offsets": [offset, end],
            ] as [String: Any]
            offset = end
        }

        // `.sortedKeys` is what makes the JSON itself reproducible; without it
        // `JSONSerialization` emits dictionary order, which is the same randomisation this
        // type exists to remove.
        let json = try JSONSerialization.data(withJSONObject: root,
                                              options: [.sortedKeys, .withoutEscapingSlashes])
        var prefix = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { prefix.append(contentsOf: $0) }
        prefix.append(json)

        return Layout(prefix: prefix, order: sorted.map(\.name), dataByteCount: offset)
    }

    /// Assemble a complete file from entries and their raw bytes.
    ///
    /// `bytes` must be contiguous row-major for each name — the layout the format defines.
    public static func file(_ entries: [Entry], bytes: [String: Data],
                            metadata: [String: String] = [:]) throws -> Data {
        let plan = try layout(entries, metadata: metadata)
        let byName = Dictionary(uniqueKeysWithValues: entries.map { ($0.name, $0) })

        var out = plan.prefix
        out.reserveCapacity(plan.prefix.count + plan.dataByteCount)
        for name in plan.order {
            let supplied = bytes[name] ?? Data()
            let declared = byName[name]!.byteCount
            guard supplied.count == declared else {
                throw Failure.byteCountMismatch(name: name, declared: declared,
                                                supplied: supplied.count)
            }
            out.append(supplied)
        }
        return out
    }
}

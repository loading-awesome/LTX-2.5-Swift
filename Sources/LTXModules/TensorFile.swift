// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// Writing `.safetensors` so the same tensors always produce the same bytes.
///
/// A drop-in for `MLX.save(arrays:metadata:url:)`, which cannot be made reproducible from
/// outside: it takes a Swift `Dictionary` and Swift randomises dictionary iteration per
/// process, so the C++ writer places the data blobs in a different order on each run.
/// ``LTXFoundation/SafetensorsWriter`` owns the layout rule and is tested without MLX; this
/// is only the part that needs `MLXArray` — the dtype mapping and the raw bytes.
public enum TensorFile {

    public enum Failure: Error, CustomStringConvertible {
        case unsupportedDType(name: String, dtype: String)

        public var description: String {
            switch self {
            case let .unsupportedDType(name, dtype):
                return "'\(name)' has dtype \(dtype), which this writer does not map to a "
                    + "safetensors dtype. Add it to TensorFile.dtype(_:) with its byte width "
                    + "checked, rather than guessing a width that happens to compile"
            }
        }
    }

    /// MLX's dtype to the format's name.
    ///
    /// Deliberately not exhaustive over `DType`: an unmapped case throws rather than falling
    /// through to a plausible default, because a wrong width writes a file that parses and
    /// decodes to garbage.
    static func dtype(_ dtype: DType, name: String) throws -> SafetensorsHeader.DType {
        switch dtype {
        case .bool: .bool
        case .uint8: .u8
        case .uint16: .u16
        case .uint32: .u32
        case .uint64: .u64
        case .int8: .i8
        case .int16: .i16
        case .int32: .i32
        case .int64: .i64
        case .float16: .f16
        case .bfloat16: .bf16
        case .float32: .f32
        case .float64: .f64
        default: throw Failure.unsupportedDType(name: name, dtype: "\(dtype)")
        }
    }

    /// Write `arrays` to `url`, byte-reproducibly.
    ///
    /// Arrays are evaluated and copied to host memory one at a time. `asData()` returns a
    /// contiguous row-major copy, which is exactly the layout the format wants.
    public static func save(arrays: [String: MLXArray], metadata: [String: String] = [:],
                            to url: URL) throws {
        var entries: [SafetensorsWriter.Entry] = []
        var bytes: [String: Data] = [:]
        entries.reserveCapacity(arrays.count)

        // Sorted here too, so the order tensors are evaluated and copied in does not depend
        // on dictionary iteration either — the peak host memory of a large bundle should not
        // vary run to run any more than its bytes do.
        for name in arrays.keys.sorted() {
            let array = arrays[name]!
            entries.append(SafetensorsWriter.Entry(
                name: name,
                dtype: try dtype(array.dtype, name: name),
                shape: array.shape))
            bytes[name] = array.asData()
        }

        let file = try SafetensorsWriter.file(entries, bytes: bytes, metadata: metadata)
        try file.write(to: url)
    }
}

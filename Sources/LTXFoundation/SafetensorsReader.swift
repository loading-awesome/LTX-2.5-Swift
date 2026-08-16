// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Reads tensor *payloads* from a safetensors file.
///
/// `SafetensorsHeader` deliberately reads only the header — two ranged reads, no
/// payload — because most questions about a checkpoint are answerable from names and
/// shapes. This is the other half: it memory-maps the file and materialises named
/// tensors as `[Float]`.
///
/// ## Why mmap rather than `Data(contentsOf:)`
///
/// The text encoder is 26 GiB and the DiT is 42 GiB. Reading either into memory to
/// pull out one 1.5 GiB tensor is not a small inefficiency, it is the difference
/// between working and not. `mmap` lets the kernel page in only the ranges actually
/// touched, and the pages are evictable under pressure rather than counting against
/// a resident allocation.
///
/// The mapping is read-only and the file is opened read-only, so a checkpoint cannot
/// be corrupted by a bug here.
///
/// ## Checkpoints are untrusted input
///
/// Every offset comes from a header that `SafetensorsHeader.parse` has already range
/// checked against the declared data size — but that check is against the *declared*
/// size, and this type maps the *actual* file. So the bounds are re-checked here
/// against the real mapping length before any pointer arithmetic. A truncated file
/// with an intact header is the exact case that would otherwise read past the end.
public final class SafetensorsReader {

    public let header: SafetensorsHeader
    public let url: URL
    private let descriptor: Int32
    private let base: UnsafeRawPointer
    private let length: Int

    public enum Failure: Error, CustomStringConvertible {
        case cannotOpen(String, errno: Int32)
        case cannotMap(String, errno: Int32)
        case unknownTensor(String)
        case outOfBounds(String, needs: Int, fileLength: Int)
        case unsupportedDType(String, dtype: String)
        case byteCountMismatch(String, declared: Int, computed: Int)

        public var description: String {
            switch self {
            case let .cannotOpen(path, e):
                return "cannot open \(path) (errno \(e))"
            case let .cannotMap(path, e):
                return "cannot mmap \(path) (errno \(e))"
            case let .unknownTensor(name):
                return "no tensor named \(name)"
            case let .outOfBounds(name, needs, fileLength):
                return "tensor \(name) ends at byte \(needs) but the file is "
                    + "\(fileLength) bytes. The header describes more data than exists — "
                    + "a truncated download with an intact header reads exactly like this"
            case let .unsupportedDType(name, dtype):
                return "tensor \(name) has dtype \(dtype), which this reader cannot "
                    + "convert. Add it explicitly rather than reinterpreting the bytes"
            case let .byteCountMismatch(name, declared, computed):
                return "tensor \(name) declares \(declared) bytes but its shape and dtype "
                    + "imply \(computed)"
            }
        }
    }

    public init(url: URL) throws {
        self.url = url
        self.header = try SafetensorsHeader.read(from: url)

        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw Failure.cannotOpen(url.path, errno: errno) }
        var stat = Foundation.stat()
        guard fstat(fd, &stat) == 0 else {
            close(fd)
            throw Failure.cannotOpen(url.path, errno: errno)
        }
        let size = Int(stat.st_size)
        guard let map = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              map != UnsafeMutableRawPointer(bitPattern: -1) else {
            close(fd)
            throw Failure.cannotMap(url.path, errno: errno)
        }
        self.descriptor = fd
        self.base = UnsafeRawPointer(map)
        self.length = size
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), length)
        close(descriptor)
    }

    /// Byte range of a tensor in the file, bounds-checked against the real mapping.
    private func range(_ name: String) throws -> (offset: Int, count: Int,
                                                  entry: SafetensorsHeader.TensorEntry) {
        guard let entry = header.tensors[name] else { throw Failure.unknownTensor(name) }
        let expected = entry.elementCount * entry.dtype.byteWidth
        guard entry.byteCount == expected else {
            throw Failure.byteCountMismatch(name, declared: entry.byteCount,
                                            computed: expected)
        }
        let start = entry.fileOffset(in: header)
        let end = start + entry.byteCount
        guard end <= length, start >= 0 else {
            throw Failure.outOfBounds(name, needs: end, fileLength: length)
        }
        return (start, entry.elementCount, entry)
    }

    /// Read a tensor as `[Float]`, converting from its stored dtype.
    ///
    /// bf16 conversion is a 16-bit left shift into the high half of an fp32 — exact,
    /// not an approximation, since bf16 *is* the top half of an fp32. fp16 needs a
    /// real conversion and uses `Float(Float16:)`.
    public func floats(_ name: String) throws -> [Float] {
        let (offset, count, entry) = try range(name)
        let raw = base.advanced(by: offset)

        switch entry.dtype {
        case .f32:
            return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                memcpy(buffer.baseAddress!, raw, count * 4)
                initialized = count
            }
        case .bf16:
            let words = raw.bindMemory(to: UInt16.self, capacity: count)
            return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                for i in 0..<count {
                    // bf16 -> fp32 is exact: shift into the high 16 bits.
                    buffer[i] = Float(bitPattern: UInt32(words[i]) << 16)
                }
                initialized = count
            }
        case .f16:
            let halves = raw.bindMemory(to: UInt16.self, capacity: count)
            return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                for i in 0..<count {
                    buffer[i] = Float(Float16(bitPattern: halves[i]))
                }
                initialized = count
            }
        case .f64:
            let doubles = raw.bindMemory(to: Double.self, capacity: count)
            return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                for i in 0..<count { buffer[i] = Float(doubles[i]) }
                initialized = count
            }
        default:
            throw Failure.unsupportedDType(name, dtype: entry.dtype.rawValue)
        }
    }

    /// Shape of a tensor, for callers that read the payload and need the geometry.
    public func shape(_ name: String) throws -> [Int] {
        guard let entry = header.tensors[name] else { throw Failure.unknownTensor(name) }
        return entry.shape
    }

    /// Read a UTF-8 blob stored as a tensor.
    ///
    /// The LTX text encoder ships its tokenizer this way: `tokenizer_json` is a 32 MB
    /// tensor of bytes, alongside the chat template and generation config. Extracting
    /// it is how a port gets a tokenizer without a second download.
    public func utf8String(_ name: String) throws -> String? {
        let (offset, count, entry) = try range(name)
        let bytes = count * entry.dtype.byteWidth
        let data = Data(bytes: base.advanced(by: offset), count: bytes)
        return String(data: data, encoding: .utf8)
    }
}

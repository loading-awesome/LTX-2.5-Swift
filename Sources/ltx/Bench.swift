// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Darwin
import Foundation
import LTX25
import MLX

/// `ltx bench` — the measurement side of `docs/PERFORMANCE.md`.
///
/// Two subcommands, in the order a sweep runs them:
///
/// - `gemm` is Step 0, the roofline. It decides whether the rest of the program is worth
///   running at all, so it is deliberately the only measurement that does not need a
///   render, a prompt, or a 42 GB load — it takes its shapes from a *ranged header read*
///   of the checkpoint and measures those shapes on synthetic operands.
/// - `forward` is the stage profile: where the wall clock of an actual render goes.
///
/// Both write the same machine-readable record shape (``BenchRecord``) so a sweep taken
/// months apart is still comparable.
///
/// ## What this harness is not
///
/// It measures nothing about correctness. `docs/PERFORMANCE.md` rule 2 — every
/// optimization is re-checked for quality — is not enforceable from here, and the record's
/// `quality` map is deliberately open and empty by default so that a caller fills it from
/// whatever check actually ran. **A bench record with an empty `quality` map is a speed
/// observation, not evidence about a build.**
struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Measure this machine against the model's own shapes (docs/PERFORMANCE.md).",
        discussion: """
            GPU work must be serialised. A cell measured beside another process is not a \
            slower cell, it is a wrong one. Serialise the GPU before trusting anything \
            these commands print.
            """,
        subcommands: [BenchGEMM.self, BenchForward.self])
}

// MARK: - JSON, with the non-finite rule built in

/// A JSON value that cannot throw on a non-finite `Double`.
///
/// `JSONEncoder` and `JSONSerialization` both *throw* on NaN and ±Inf unless a
/// non-conforming strategy is set, which means a single divide-by-zero anywhere in a
/// derived statistic destroys the entire record rather than one field of it. That has
/// already happened to the first control run of a sweep, which produced no artifact at
/// all — the cheapest possible way to lose a measurement.
///
/// So the rule is encoded at the one place it can bite: ``BenchJSON/double(_:)`` encodes a
/// non-finite value as JSON `null`. The CSV writer does the same with the empty string.
indirect enum BenchJSON: Encodable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([BenchJSON])
    case object([String: BenchJSON])

    static func opt(_ value: Double?) -> BenchJSON {
        guard let value else { return .null }
        return .double(value)
    }

    static func opt(_ value: Int?) -> BenchJSON {
        guard let value else { return .null }
        return .int(value)
    }

    static func opt(_ value: String?) -> BenchJSON {
        guard let value else { return .null }
        return .string(value)
    }

    static func strings(_ values: [String]) -> BenchJSON { .array(values.map(BenchJSON.string)) }
    static func doubles(_ values: [Double]) -> BenchJSON { .array(values.map(BenchJSON.double)) }
    static func ints(_ values: [Int]) -> BenchJSON { .array(values.map(BenchJSON.int)) }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case let .bool(value):
            try container.encode(value)
        case let .int(value):
            try container.encode(value)
        case let .double(value):
            // The whole reason this type exists.
            if value.isFinite { try container.encode(value) } else { try container.encodeNil() }
        case let .string(value):
            try container.encode(value)
        case let .array(values):
            try container.encode(values)
        case let .object(values):
            try container.encode(values)
        }
    }

    /// Pretty, key-sorted bytes. Sorted so two records diff cleanly.
    func serialized() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }
}

// MARK: - Statistics

/// Median, min, max and mean over a set of timed repeats.
///
/// ## Why the spread is a range and not a standard deviation
///
/// With two or three repeats a standard deviation is a statement about the sample size,
/// not about the machine. So the full range is reported instead — `(max - min) / median` —
/// whose own measured floor is 0.9–1.4%, which is what licenses the rule that
/// *any claimed gain below about 1.5% is not a gain*. That number is only meaningful if
/// the statistic that produced it is the one being compared against, so this type reports
/// the range and does not offer a stddev.
struct BenchStats {
    let samples: [Double]
    let count: Int
    let median: Double
    let minimum: Double
    let maximum: Double
    let mean: Double

    init(_ samples: [Double]) {
        self.samples = samples
        self.count = samples.count
        guard !samples.isEmpty else {
            median = .nan; minimum = .nan; maximum = .nan; mean = .nan
            return
        }
        let sorted = samples.sorted()
        median = sorted.count % 2 == 1
            ? sorted[sorted.count / 2]
            : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        minimum = sorted[0]
        maximum = sorted[sorted.count - 1]
        mean = samples.reduce(0, +) / Double(samples.count)
    }

    /// `(max - min) / median`. `nil` for fewer than two samples or a zero median — and a
    /// `nil` here becomes JSON `null`, never a fabricated 0.
    var spread: Double? {
        guard count >= 2, median != 0, median.isFinite else { return nil }
        return (maximum - minimum) / median
    }

    var json: BenchJSON {
        .object([
            "count": .int(count),
            "medianSeconds": .double(median),
            "minSeconds": .double(minimum),
            "maxSeconds": .double(maximum),
            "meanSeconds": .double(mean),
            "spreadOfRangeOverMedian": .opt(spread),
            "samplesSeconds": .doubles(samples),
        ])
    }
}

/// Median of an arbitrary sample, for the per-step arrays.
func benchMedian(_ samples: [Double]) -> Double {
    guard !samples.isEmpty else { return .nan }
    let sorted = samples.sorted()
    return sorted.count % 2 == 1
        ? sorted[sorted.count / 2]
        : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
}

// MARK: - Clock

/// Monotonic seconds. `Date` is wall-clock and can step; a bench must not.
@inline(__always)
func benchNow() -> Double {
    Double(clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)) / 1e9
}

// MARK: - Memory

/// Peak memory, and which peak it is.
///
/// **The two figures are not interchangeable and the difference is enormous.** MLX's
/// allocator peak counts what MLX allocated on the device; the process's peak resident
/// set counts everything, including a mmapped 42 GB checkpoint that the allocator never
/// sees. A record that says "peak memory" without saying which one is unreadable six
/// months later, which is why ``BenchPeakMemory/json`` always carries `source`.
///
/// The allocator figure is preferred as the headline because it is the one an optimization
/// moves: a residency change, a tiling change or a cache all show up there, and a mmap
/// does not.
enum BenchPeakMemory {

    /// Zero the allocator's high-water mark so the next phase measures its own peak.
    static func reset() {
        MLX.GPU.resetPeakMemory()
    }

    /// The allocator high-water mark since the last ``reset()``.
    ///
    /// Floored at the current active bytes: `reset` sets the mark to zero, so a phase that
    /// allocates nothing new would otherwise report a peak *below* the memory it is
    /// actually holding.
    static var allocatorPeakBytes: Int {
        max(MLX.Memory.peakMemory, MLX.Memory.activeMemory)
    }

    /// `getrusage(RUSAGE_SELF).ru_maxrss` — bytes on Darwin, and a true peak for the whole
    /// process lifetime (it cannot be reset, so it is reported once, globally).
    static var processMaxResidentBytes: Int {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
        return Int(usage.ru_maxrss)
    }

    static func json(allocatorPeak: Int) -> BenchJSON {
        .object([
            "source": .string("mlx_allocator"),
            "bytes": .int(allocatorPeak),
            "mlxAllocatorPeakBytes": .int(allocatorPeak),
            "mlxActiveBytes": .int(MLX.Memory.activeMemory),
            "mlxCacheBytes": .int(MLX.Memory.cacheMemory),
            "processMaxResidentBytes": .int(processMaxResidentBytes),
            "note": .string(
                "`bytes` is MLX's allocator high-water mark, NOT RSS. A mmapped "
                    + "checkpoint does not appear in the allocator figure, so "
                    + "processMaxResidentBytes is far larger on any phase that touches a "
                    + "checkpoint. The two are not interchangeable."),
        ])
    }
}

// MARK: - Runtime identity

/// Who measured this, on what, from which source tree.
///
/// A bench number without this is not reproducible and not comparable to the next one.
enum BenchRuntime {

    /// The package root, derived from this file's compile-time path.
    ///
    /// `#filePath` rather than a filesystem search: this harness must never scan for its
    /// own repository. `Sources/ltx/Bench.swift` → three levels up.
    static var repositoryRoot: URL? {
        let here = URL(fileURLWithPath: #filePath)
        let root = here.deletingLastPathComponent()   // Sources/ltx
            .deletingLastPathComponent()              // Sources
            .deletingLastPathComponent()              // <root>
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("Package.swift").path)
            ? root : nil
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }

    static func sysctlInt(_ name: String) -> Int? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return Int(value)
    }

    /// The pinned mlx-swift version, read out of `Package.resolved`.
    ///
    /// There is no runtime accessor for it: mlx-swift does not export `Cmlx`, so
    /// `mlx_version()` is unreachable from a package that only depends on the `MLX`
    /// product. The pin file is the next most authoritative thing and it is at least not a
    /// hand-copied constant that drifts.
    static func mlxSwiftVersion() -> String? {
        guard let root = repositoryRoot,
              let data = try? Data(contentsOf: root.appendingPathComponent("Package.resolved")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pins = json["pins"] as? [[String: Any]] else { return nil }
        for pin in pins where (pin["identity"] as? String) == "mlx-swift" {
            let state = pin["state"] as? [String: Any]
            let version = state?["version"] as? String
            let revision = state?["revision"] as? String
            switch (version, revision) {
            case let (v?, r?): return "\(v) (\(r.prefix(12)))"
            case let (v?, nil): return v
            case let (nil, r?): return String(r.prefix(12))
            default: return nil
            }
        }
        return nil
    }

    /// `git rev-parse HEAD` plus whether the worktree is dirty.
    ///
    /// The dirty flag is not decoration. This repository is benched from a working tree
    /// with uncommitted changes most of the time, and a record that names only a commit is
    /// naming a tree that was never built.
    static func git() -> (revision: String?, dirty: Bool?) {
        guard let root = repositoryRoot else { return (nil, nil) }

        func run(_ arguments: [String]) -> String? {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = ["git", "-C", root.path] + arguments
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            do { try process.run() } catch { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            return String(decoding: data, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let revision = run(["rev-parse", "HEAD"])
        let status = run(["status", "--porcelain"])
        return (revision, status.map { !$0.isEmpty })
    }

    static func json() -> BenchJSON {
        let device = MLX.GPU.deviceInfo()
        let (revision, dirty) = git()
        let formatter = ISO8601DateFormatter()
        return .object([
            "chip": .opt(sysctlString("machdep.cpu.brand_string")),
            "model": .opt(sysctlString("hw.model")),
            "memoryBytes": .opt(sysctlInt("hw.memsize")),
            "cpuCores": .opt(sysctlInt("hw.ncpu")),
            "gpuArchitecture": .string(device.architecture),
            "gpuMaxBufferBytes": .int(device.maxBufferSize),
            "gpuMaxRecommendedWorkingSetBytes": .int(Int(clamping: device.maxRecommendedWorkingSetSize)),
            "os": .string(ProcessInfo.processInfo.operatingSystemVersionString),
            "mlxSwiftVersion": .opt(mlxSwiftVersion()),
            "ltxVersion": .string(LTX25.version),
            "repoGitRevision": .opt(revision),
            "repoGitDirty": dirty.map(BenchJSON.bool) ?? .null,
            "hostname": .string(ProcessInfo.processInfo.hostName),
            "processIdentifier": .int(Int(ProcessInfo.processInfo.processIdentifier)),
            "timestamp": .string(formatter.string(from: Date())),
        ])
    }
}

// MARK: - The record

/// The artifact every bench run writes.
///
/// Field names are the contract; the `quality` map is deliberately open so a caller can
/// attach whatever the quality check of the day produced without this type learning
/// about it.
struct BenchRecord {
    /// `gemm` or `forward`.
    var kind: String
    /// The exact argv this run was invoked with, so a record can be re-run.
    var invocation: [String]
    /// Command-specific parameters — every sampling knob for `forward`, every shape and
    /// repeat count for `gemm`.
    var parameters: BenchJSON
    /// The measurements.
    var measurements: BenchJSON
    /// Free-form notes about what could *not* be measured. Never empty in practice.
    var caveats: [String]
    /// Open by design. Empty means "no quality evidence was attached", which is a
    /// statement, not an omission.
    var quality: BenchJSON = .object([:])
    var peakMemoryAllocatorBytes: Int

    func json() -> BenchJSON {
        .object([
            "schema": .string("ltx-bench/1"),
            "kind": .string(kind),
            "invocation": .strings(invocation),
            "runtime": BenchRuntime.json(),
            "parameters": parameters,
            "measurements": measurements,
            "peak_memory": BenchPeakMemory.json(allocatorPeak: peakMemoryAllocatorBytes),
            "quality": quality,
            "caveats": .strings(caveats),
        ])
    }

    /// Write the record, and say so on stderr. A bench that silently produced no artifact
    /// is the failure this whole type exists to prevent, so the write is loud.
    func write(to path: String?) throws {
        guard let path else {
            FileHandle.standardError.write(Data(
                "no --out given: this run left no machine-readable record\n".utf8))
            return
        }
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try json().serialized().write(to: url)
        FileHandle.standardError.write(Data("wrote \(url.path)\n".utf8))
    }
}

// MARK: - CSV

/// A flat sidecar for the record, for people who would rather sweep in a spreadsheet.
///
/// Non-finite values are the empty cell, matching the JSON `null` rule. A CSV that wrote
/// `nan` would sort and average as a number in every tool that reads it.
struct BenchCSV {
    let header: [String]
    private var rows: [[String]] = []

    init(header: [String]) { self.header = header }

    static func cell(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "" }
        return String(format: "%.9g", value)
    }

    static func cell(_ value: Int?) -> String {
        guard let value else { return "" }
        return String(value)
    }

    mutating func append(_ row: [String]) {
        precondition(row.count == header.count, "CSV row width must match the header")
        rows.append(row)
    }

    func write(to path: String?) throws {
        guard let path else { return }
        func escape(_ field: String) -> String {
            guard field.contains(",") || field.contains("\"") || field.contains("\n") else {
                return field
            }
            return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        var text = header.map(escape).joined(separator: ",") + "\n"
        for row in rows { text += row.map(escape).joined(separator: ",") + "\n" }
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
        FileHandle.standardError.write(Data("wrote \(url.path)\n".utf8))
    }
}

// MARK: - Deterministic bf16 operands

/// Host-side bf16 bit patterns, generated rather than drawn on device.
///
/// Two reasons this is not `MLXRandom.normal`:
///
/// 1. **The two vendors must be fed the same bytes.** A comparison between MLX and
///    MPSGraph is only a comparison if the operands are bit-identical, and the only way to
///    be sure of that is to own the bits and hand the same array to both.
/// 2. **The one-ULP control needs to flip exactly one bit.** That is a statement about a
///    bit pattern, not about a distribution.
enum BenchOperand {

    /// Round-to-nearest-even fp32 → bf16, as a bit pattern.
    static func bf16Bits(of value: Float) -> UInt16 {
        let bits = value.bitPattern
        // The standard RNE truncation: add half an ulp, biased by the retained low bit.
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    static func float(ofBf16 bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// `count` bf16 patterns in roughly `[-1, 1)`, from a splitmix64 stream.
    ///
    /// Deterministic in `seed` so a re-run measures the same operands. The range is kept
    /// modest so no product overflows bf16's range at K = 4096 and the numerical
    /// comparison between vendors is about rounding rather than about saturation.
    static func bits(count: Int, seed: UInt64) -> [UInt16] {
        var state = seed &+ 0x9E3779B97F4A7C15
        var out = [UInt16](repeating: 0, count: count)
        for index in 0..<count {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            // 24 bits of mantissa into [0, 1), then centred.
            let unit = Float(z >> 40) / Float(1 << 24)
            out[index] = bf16Bits(of: (unit - 0.5) * 2)
        }
        return out
    }

    /// The bits as an MLX bf16 array. `view` is a reinterpretation, not a conversion —
    /// `asType` would numerically convert the uint16 and produce entirely different values.
    static func mlx(_ bits: [UInt16], shape: [Int]) -> MLXArray {
        MLXArray(bits).reshaped(shape).view(dtype: .bfloat16)
    }

    /// Read an MLX bf16 array back as its raw bit patterns.
    static func bits(of array: MLXArray) -> [UInt16] {
        array.asType(.bfloat16).view(dtype: .uint16).flattened().asArray(UInt16.self)
    }
}

/// How two vendors' outputs compare, over the same bytes.
struct BenchComparison {
    var maxAbsoluteDifference: Double
    var relativeRMS: Double
    var bitIdenticalFraction: Double
    var elements: Int

    /// `reference` is the second vendor's output; `candidate` the first's.
    init(candidate: [UInt16], reference: [UInt16]) {
        let n = min(candidate.count, reference.count)
        elements = n
        var maxAbs = 0.0
        var sumSquaredDifference = 0.0
        var sumSquaredReference = 0.0
        var identical = 0
        for index in 0..<n {
            let a = Double(BenchOperand.float(ofBf16: candidate[index]))
            let b = Double(BenchOperand.float(ofBf16: reference[index]))
            let difference = a - b
            maxAbs = Swift.max(maxAbs, abs(difference))
            sumSquaredDifference += difference * difference
            sumSquaredReference += b * b
            if candidate[index] == reference[index] { identical += 1 }
        }
        maxAbsoluteDifference = maxAbs
        relativeRMS = sumSquaredReference > 0
            ? (sumSquaredDifference / sumSquaredReference).squareRoot()
            : (sumSquaredDifference > 0 ? .infinity : 0)
        bitIdenticalFraction = n > 0 ? Double(identical) / Double(n) : .nan
    }

    var json: BenchJSON {
        .object([
            "maxAbsoluteDifference": .double(maxAbsoluteDifference),
            "relativeRMS": .double(relativeRMS),
            "bitIdenticalFraction": .double(bitIdenticalFraction),
            "elementsCompared": .int(elements),
        ])
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Darwin
import Foundation

/// What this Mac is, measured at startup rather than assumed.
///
/// Every field comes from `sysctl` or `host_statistics64`. Nothing is inferred
/// from a marketing name: the same chip ships with memory configurations that
/// differ by a factor of eight.
public struct Machine: Sendable, Equatable {
    public let model: String
    public let chip: String
    public let memoryBytes: UInt64
    public let cores: Int

    public var memoryGB: Double { Double(memoryBytes) / 1e9 }

    /// Kept so existing call sites that branched on the placeholder compile.
    /// Admission is real now.
    public static let placeholder = false

    public init(model: String, chip: String, memoryBytes: UInt64, cores: Int) {
        self.model = model
        self.chip = chip
        self.memoryBytes = memoryBytes
        self.cores = cores
    }

    public static func detect() -> Machine {
        Machine(model: sysctlString("hw.model") ?? "unknown",
                chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
                memoryBytes: sysctlUInt64("hw.memsize") ?? 0,
                cores: Int(sysctlUInt64("hw.ncpu") ?? 0))
    }

    /// Memory the kernel can still hand over: free + inactive + speculative.
    ///
    /// A 64 GB machine with a browser open is not a 64 GB machine. Planning
    /// against `hw.memsize` would admit a render that then SIGKILLs.
    public static func availableBytes() -> UInt64 {
        var size = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size
                                          / MemoryLayout<integer_t>.size)
        var stats = vm_statistics64_data_t()
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &size)
            }
        }
        guard result == KERN_SUCCESS else { return sysctlUInt64("hw.memsize") ?? 0 }
        let page = sysctlUInt64("hw.pagesize") ?? 16384
        return (UInt64(stats.free_count) + UInt64(stats.inactive_count)
                + UInt64(stats.speculative_count)) * page
    }

    static func sysctlString(_ name: String) -> String? {
        var len = 0
        guard sysctlbyname(name, nil, &len, nil, 0) == 0, len > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: len)
        guard sysctlbyname(name, &buf, &len, nil, 0) == 0 else { return nil }
        let bytes = buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    static func sysctlUInt64(_ name: String) -> UInt64? {
        var value: UInt64 = 0
        var len = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &len, nil, 0) == 0 else { return nil }
        return value
    }

    public var summary: String {
        String(format: "%@ (%@), %.0f GB unified memory, %d cores",
               chip, model, memoryGB, cores)
    }
}

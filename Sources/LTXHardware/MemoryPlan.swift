// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Whether a render's working set fits this machine, from **measured** peaks.
///
/// Coefficients are fitted to allocator peaks measured on 2026-08-13 with fused
/// attention — the live render path. Explicit attention is quadratic and is
/// not what `Renderer` defaults to; do not plan against it.
///
/// The MLX allocator peak **includes** the loaded DiT weights (~42.06 GB).
/// mmap'd files that are not `MLX.loadArrays`'d do not appear in it. Phases
/// do not overlap: Gemma is dropped before the DiT arrives, the DiT is
/// dropped before decode.
public struct MemoryPlan: Sendable, Equatable {

    /// `42.06 GB of weights already resident`, measured on the attention sweep.
    public static let ditBytes: UInt64 = 42_060_000_000
    /// Gemma-4 12B encoder, on disk and in the allocator during text encode.
    public static let textEncoderBytes: UInt64 = 26_000_000_000
    /// Production 640×384×97 decode allocator peak (VAE + vocoder, DiT gone).
    public static let productionDecodeBytes: UInt64 = 47_700_000_000
    /// Production video token count: 13 × 12 × 20 = 3120.
    public static let productionVideoTokens = 3120

    /// Fused-path activation bytes above the DiT weights.
    ///
    /// Fitted to the two measured fused points:
    /// 9,216 tokens → 4.30 GB above weights; 17,856 tokens → 8.34 GB.
    /// Exponent 1.00. Linear in token count.
    public static func fusedActivationBytes(videoTokens: Int) -> UInt64 {
        UInt64((4.67e5 * Double(videoTokens)).rounded())
    }

    public enum Phase: String, Sendable, CaseIterable {
        case textEncode, sampling, decode
    }

    public let videoTokens: Int
    public let phaseBytes: [Phase: UInt64]
    public let peakBytes: UInt64
    public let availableBytes: UInt64
    public let headroomBytes: Int64
    /// The margin `headroomBytes` was computed with.
    ///
    /// Carried rather than assumed. The explanation used to print a hardcoded "15%", which
    /// was true only until the config file made the margin a setting — at which point the
    /// line would have described a calculation the plan had not performed.
    public let marginFraction: Double

    public var peakGB: Double { Double(peakBytes) / 1e9 }
    public var availableGB: Double { Double(availableBytes) / 1e9 }
    public var fits: Bool { headroomBytes > 0 }

    public static func plan(videoTokens: Int, availableBytes: UInt64,
                            marginFraction: Double = 0.15) -> MemoryPlan {
        let sampling = ditBytes + fusedActivationBytes(videoTokens: videoTokens)
        let phases: [Phase: UInt64] = [
            .textEncode: textEncoderBytes + 2_000_000_000,
            .sampling: sampling,
            // Tiled decode keeps the production figure from scaling with the
            // long side. Using the measured 47.70 GB as a floor, and sampling
            // as a ceiling when activations grow past it.
            .decode: max(productionDecodeBytes, sampling / 4 + 8_000_000_000),
        ]
        let peak = phases.values.max() ?? 0
        let withMargin = UInt64(Double(peak) * (1.0 + marginFraction))
        return MemoryPlan(videoTokens: videoTokens, phaseBytes: phases, peakBytes: peak,
                          availableBytes: availableBytes,
                          headroomBytes: Int64(availableBytes) - Int64(withMargin),
                          marginFraction: marginFraction)
    }

    public var explanation: String {
        var lines = [String]()
        lines.append(String(format: "  video tokens   %d (fused-path plan)", videoTokens))
        for phase in Phase.allCases {
            let b = phaseBytes[phase] ?? 0
            // Padded in Swift. Darwin's Foundation ignores width flags for `%@`, so the
            // `%-14@` this used to be did nothing and the three phase rows came out ragged
            // — each name a different length, each figure at a different column.
            let name = phase.rawValue.count >= 14
                ? phase.rawValue
                : phase.rawValue + String(repeating: " ", count: 14 - phase.rawValue.count)
            lines.append(String(format: "  %@ %6.1f GB%@", name, Double(b) / 1e9,
                                b == peakBytes ? "   <- peak" : ""))
        }
        lines.append(String(format: "  available      %6.1f GB", availableGB))
        lines.append(String(format: "  headroom       %+6.1f GB after a %.0f%% margin",
                            Double(headroomBytes) / 1e9, marginFraction * 100))
        return lines.joined(separator: "\n")
    }
}

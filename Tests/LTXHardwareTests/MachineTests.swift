// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
@testable import LTXHardware

@Suite("LTXHardware")
struct MachineTests {

    @Test("detect reads this Mac")
    func detectReadsSysctl() {
        let m = Machine.detect()
        #expect(m.memoryBytes > 0)
        #expect(m.cores > 0)
        #expect(!m.chip.isEmpty)
        #expect(Machine.placeholder == false)
        #expect(Machine.availableBytes() > 0)
    }

    @Test("production fused plan fits 256 GB and refuses 32 GB")
    func productionFitsAStudioNotA32GBBox() {
        let tokens = MemoryPlan.productionVideoTokens
        let studio = MemoryPlan.plan(videoTokens: tokens, availableBytes: 256_000_000_000)
        #expect(studio.fits)
        let small = MemoryPlan.plan(videoTokens: tokens, availableBytes: 32_000_000_000)
        #expect(!small.fits)
        // 64 GB empty: production decode floor 47.7 GB × 1.15 ≈ 54.9 GB. Fits.
        let sixtyFour = MemoryPlan.plan(videoTokens: tokens, availableBytes: 64_000_000_000)
        #expect(sixtyFour.fits)
        // 64 GB with 20 GB already spoken for does not.
        let contended = MemoryPlan.plan(videoTokens: tokens, availableBytes: 44_000_000_000)
        #expect(!contended.fits)
    }

    @Test("fused activation is linear in the measured token counts")
    func fusedActivationMatchesTheTable() {
        let at9216 = MemoryPlan.fusedActivationBytes(videoTokens: 9216)
        let at17856 = MemoryPlan.fusedActivationBytes(videoTokens: 17856)
        // RESULTS.md: 4.30 GB and 8.34 GB above weights. 4.67e5 B/token.
        #expect(abs(Double(at9216) / 1e9 - 4.30) < 0.05)
        #expect(abs(Double(at17856) / 1e9 - 8.34) < 0.05)
        #expect(at17856 > at9216)
    }
}

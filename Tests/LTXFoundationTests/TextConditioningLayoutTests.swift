// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// Each test here corresponds to a way the layout logic can be implemented
/// incorrectly while still producing a correctly-shaped tensor.
@Suite("Text conditioning layout")
struct TextConditioningLayoutTests {

    let layout = TextConditioningLayout()

    /// The encoded sequence is 1024 wide whatever the prompt length, so the padded
    /// length is the conditioning shape (contract 9).
    @Test("the padded sequence length is 1024, as measured")
    func measuredLength() {
        #expect(layout.sequenceLength == 1024)
    }

    @Test("left-padded input is normalised to right-padded")
    func leftPadded() throws {
        // Gemma tokenizers default to left padding, so this is the real case.
        var mask = [Bool](repeating: false, count: 1024)
        for i in 1019..<1024 { mask[i] = true }        // 5 valid tokens at the end

        let order = try layout.rightPadOrder(mask: mask)
        #expect(Array(order.prefix(5)) == [1019, 1020, 1021, 1022, 1023])
        #expect(try layout.reorderedMask(mask: mask).prefix(5).allSatisfy { $0 })
        #expect(try layout.reorderedMask(mask: mask).dropFirst(5).allSatisfy { !$0 })
    }

    /// The trap: an unstable sort gives a permutation with the right shape and the
    /// right valid count, and the prompt's words in the wrong order.
    @Test("valid tokens keep their relative order — the sort must be stable")
    func stability() throws {
        var mask = [Bool](repeating: false, count: 1024)
        for i in [3, 17, 400, 1001] { mask[i] = true }

        let order = try layout.rightPadOrder(mask: mask)
        #expect(Array(order.prefix(4)) == [3, 17, 400, 1001])   // ascending, not any order
    }

    /// The permutation is applied unconditionally, so it must be the identity on
    /// already-right-padded input. Skipping it as an optimisation agrees here and
    /// then diverges on left-padded input.
    @Test("idempotent on already-right-padded input")
    func idempotent() throws {
        var mask = [Bool](repeating: false, count: 1024)
        for i in 0..<7 { mask[i] = true }

        let order = try layout.rightPadOrder(mask: mask)
        #expect(order == Array(0..<1024))

        // And applying it twice changes nothing.
        let once = try layout.reorderedMask(mask: mask)
        #expect(try layout.reorderedMask(mask: once) == once)
    }

    @Test("padding is -dtypeMax, never -infinity")
    func additiveMask() {
        let bf16Max: Float = 3.3895314e38      // the largest finite bfloat16
        #expect(layout.additiveMaskValue(isValid: true, dtypeMax: bf16Max) == 0.0)

        let pad = layout.additiveMaskValue(isValid: false, dtypeMax: bf16Max)
        #expect(pad == -bf16Max)
        // The distinction that matters: -max is finite, so 0 * pad is 0 and not NaN.
        #expect(pad.isFinite)
        #expect((0.0 as Float * pad).isNaN == false)
    }

    /// The asymmetry is deliberate. If someone "fixes" it, this test is the
    /// record of what the intended behaviour is.
    @Test("video encoding is masked after the connector; audio is not")
    func maskAsymmetry() {
        #expect(layout.masksEncodingAfterConnector(stream: .video))
        #expect(!layout.masksEncodingAfterConnector(stream: .audio))
    }

    @Test("valid token count distinguishes the two guidance branches")
    func validCount() {
        var positive = [Bool](repeating: false, count: 1024)
        for i in 0..<7 { positive[i] = true }
        let negative = [Bool](repeating: false, count: 1024)

        #expect(layout.validTokenCount(mask: positive) == 7)
        #expect(layout.validTokenCount(mask: negative) == 0)
    }

    // MARK: - Rejections

    @Test("a mask of the wrong length is refused")
    func wrongLength() {
        #expect(throws: TextConditioningLayout.Failure.maskLengthMismatch(
            expected: 1024, got: 7)) {
            try layout.rightPadOrder(mask: [Bool](repeating: true, count: 7))
        }
    }

    @Test("an all-padding mask is refused")
    func allPadding() {
        #expect(throws: TextConditioningLayout.Failure.emptyMask) {
            try layout.rightPadOrder(mask: [Bool](repeating: false, count: 1024))
        }
    }
}

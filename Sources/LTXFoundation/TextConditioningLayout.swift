// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// How text conditioning is laid out before it reaches the connectors.
///
/// Pure index arithmetic — no MLX, no weights — because every rule here is a
/// silent failure in tensor form. Get the pad side wrong and the port produces a
/// correctly-shaped conditioning tensor with its tokens in the wrong positions.
/// No shape check catches that, and the render degrades rather than failing.
public struct TextConditioningLayout: Sendable, Equatable {

    /// Sequence length the tokenizer pads to.
    ///
    /// Contract 9: `ltxav_gemma4_tokenizer` raises `min_length` from 1 to 1024, so
    /// this is not incidental padding — it is the shape of the conditioning.
    /// Measured: `enc.attention_mask [1, 1024]` for a seven-word prompt.
    public let sequenceLength: Int

    public init(sequenceLength: Int = 1024) {
        self.sequenceLength = sequenceLength
    }

    public enum Failure: Error, Equatable, CustomStringConvertible {
        case maskLengthMismatch(expected: Int, got: Int)
        case emptyMask

        public var description: String {
            switch self {
            case let .maskLengthMismatch(expected, got):
                return "attention mask has \(got) entries, expected \(expected). The "
                    + "tokenizer pads to a fixed length (contract 9); a short mask means "
                    + "min_length was not applied"
            case .emptyMask:
                return "attention mask marks no valid tokens; conditioning would be "
                    + "entirely padding"
            }
        }
    }

    /// The permutation that moves valid tokens ahead of padding.
    ///
    /// The connectors require right-padded input. Rather than requiring the
    /// tokenizer to pad on a particular side, the layout is normalised here with a
    /// **stable descending sort of the binary mask**.
    ///
    /// Two properties are load-bearing and both are easy to lose:
    ///
    /// 1. **Stability.** Valid tokens must keep their relative order. An unstable
    ///    sort produces a valid-looking permutation that scrambles the prompt —
    ///    the tensor has the right shape, the right number of valid positions, and
    ///    the words in the wrong order.
    /// 2. **Idempotence.** Applying this to already-right-padded input must be the
    ///    identity, because it is applied unconditionally. A port that
    ///    "optimises" by skipping the permutation when input looks right-padded
    ///    agrees on those inputs and diverges on left-padded ones — which is the
    ///    default padding side Gemma tokenizers use.
    ///
    /// - Returns: `order[i]` is the source index that lands at position `i`.
    public func rightPadOrder(mask: [Bool]) throws -> [Int] {
        guard mask.count == sequenceLength else {
            throw Failure.maskLengthMismatch(expected: sequenceLength, got: mask.count)
        }
        guard mask.contains(true) else { throw Failure.emptyMask }

        // A stable partition, which is what a stable descending sort of a binary
        // key reduces to. Written as a partition rather than a sort so stability
        // is a property of the code and not of the sort implementation Swift
        // happens to use -- `sort(by:)` is documented as NOT stable.
        var valid: [Int] = []
        var pad: [Int] = []
        valid.reserveCapacity(mask.count)
        for (i, isValid) in mask.enumerated() {
            if isValid { valid.append(i) } else { pad.append(i) }
        }
        return valid + pad
    }

    /// The mask after reordering: `validCount` leading `true`s, then padding.
    ///
    /// Derived from the permutation rather than re-sorted, so the two can never
    /// disagree about where the boundary is.
    public func reorderedMask(mask: [Bool]) throws -> [Bool] {
        let order = try rightPadOrder(mask: mask)
        return order.map { mask[$0] }
    }

    /// Count of valid (non-padding) tokens.
    ///
    /// Also how two encode calls are told apart: text conditioning runs once per
    /// guidance branch, and the negative prompt is usually empty, so the valid-token
    /// count is what distinguishes one branch's encoding from the other's.
    public func validTokenCount(mask: [Bool]) -> Int {
        mask.reduce(into: 0) { $0 += $1 ? 1 : 0 }
    }

    /// Additive attention mask value for a position.
    ///
    /// The mask is `(binary - 1) * dtypeMax`, so valid is `0`, padding is `-max`,
    /// and **not** `-infinity`. The difference
    /// matters: `-inf` times a zero attention weight is `NaN`, while `-max` is
    /// finite and merely saturates. A port that reaches for `-.infinity` because it
    /// is the "obvious" mask value can produce NaNs on fully-masked rows.
    public func additiveMaskValue(isValid: Bool, dtypeMax: Float) -> Float {
        isValid ? 0.0 : -dtypeMax
    }

    /// Whether a stream's encoding is multiplied by the binary mask after the
    /// connector.
    ///
    /// **Video only.** The video encoding is multiplied by the binary mask; the
    /// audio connector's output is not.
    ///
    /// The asymmetry looks like an oversight and is not something to "fix" — masking
    /// both changes every padded position of the audio stream, which is most of them
    /// at 1024 tokens.
    public func masksEncodingAfterConnector(stream: Stream) -> Bool {
        switch stream {
        case .video: return true
        case .audio: return false
        }
    }

    public enum Stream: String, Sendable, CaseIterable {
        case video, audio
    }
}

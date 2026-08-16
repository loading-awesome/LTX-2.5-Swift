// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// What a training step actually optimises.
///
/// ## Three rules, all silent when wrong
///
/// 1. **The interpolation.** `noisy = (1 - σ)·clean + σ·noise`. Swap the coefficients and
///    the model trains against a time-reversed schedule. The loss still falls.
/// 2. **The target's sign.** `target = noise - clean`, not the other way round. A flipped
///    sign trains an adapter that subtracts what it should add, and the loss curve is
///    identical because squared error is symmetric in it.
/// 3. **The reduction.** The video loss divides by the *mask's own mean* rather than by a
///    token count, and the audio loss is **not masked at all**. Either mistake rescales the
///    gradient by a constant that nothing downstream reveals.
///
/// None of the three produces an error, a NaN, or a visibly wrong loss. They produce an
/// adapter that trained on the wrong objective and looks like a disappointing dataset.
///
/// ## Flat arrays, not tensors, and on purpose
///
/// This target links no MLX (`Package.swift`), so the rules live here as arithmetic over
/// flat row-major buffers with the shape passed alongside. That keeps them checkable in
/// milliseconds on a bare checkout. The tensor implementation the trainer will actually run
/// is a separate thing, and the way to keep the two honest is a cross-check test rather than
/// a second copy of the rule.
public enum TrainingObjective {

    /// Row-major `[batch, sequence, channels]`.
    public struct Shape: Sendable, Equatable {
        public let batch: Int
        public let sequence: Int
        public let channels: Int

        public init(batch: Int, sequence: Int, channels: Int) {
            self.batch = batch
            self.sequence = sequence
            self.channels = channels
        }

        public var count: Int { batch * sequence * channels }
        public var tokens: Int { batch * sequence }
        func index(_ b: Int, _ s: Int, _ c: Int) -> Int {
            (b * sequence + s) * channels + c
        }
    }

    /// `noisy = (1 - σ)·clean + σ·noise`, with conditioning tokens left clean.
    ///
    /// σ is per **example**, broadcast across the sequence. A port that broadcast per token
    /// instead would train every token of an example at a different
    /// noise level, which is a different objective wearing the same shape.
    public static func noised(clean: [Float], noise: [Float], sigmas: [Float],
                              conditioningMask: [Bool], shape: Shape) -> [Float] {
        precondition(clean.count == shape.count && noise.count == shape.count)
        precondition(sigmas.count == shape.batch)
        precondition(conditioningMask.count == shape.tokens)

        var out = [Float](repeating: 0, count: shape.count)
        for b in 0 ..< shape.batch {
            let sigma = sigmas[b]
            for s in 0 ..< shape.sequence {
                let conditioned = conditioningMask[b * shape.sequence + s]
                for c in 0 ..< shape.channels {
                    let i = shape.index(b, s, c)
                    out[i] = conditioned ? clean[i]
                                         : (1 - sigma) * clean[i] + sigma * noise[i]
                }
            }
        }
        return out
    }

    /// `target = noise - clean`. The velocity a flow-matching model predicts.
    public static func velocityTarget(clean: [Float], noise: [Float]) -> [Float] {
        precondition(clean.count == noise.count)
        return zip(noise, clean).map(-)
    }

    /// Per-token timesteps: a conditioning token reaches AdaLN at **0**, every other token
    /// at the example's sigma.
    ///
    /// The same zero that tells the transformer "this token is clean" during inference. Its
    /// absence is what makes a conditioned render condition on nothing.
    public static func perTokenTimesteps(sigmas: [Float], conditioningMask: [Bool],
                                         batch: Int, sequence: Int) -> [Float] {
        precondition(sigmas.count == batch)
        precondition(conditioningMask.count == batch * sequence)
        var out = [Float](repeating: 0, count: batch * sequence)
        for b in 0 ..< batch {
            for s in 0 ..< sequence {
                let i = b * sequence + s
                out[i] = conditioningMask[i] ? 0 : sigmas[b]
            }
        }
        return out
    }

    /// Masked mean squared error, per batch element.
    ///
    /// ```
    /// ((pred - target)² · mask).mean([-2,-1]) / mask.mean([-2,-1]).clamp(min: 1e-8)
    /// ```
    ///
    /// Note what the denominator is: the **mask's mean over `[sequence, 1]`**, i.e. the
    /// fraction of tokens that count, *not* the number of them. The numerator's mean is over
    /// `sequence × channels`. The two divisors do not cancel to "mean over unmasked
    /// elements" by accident — they cancel exactly, and only if both are written this way.
    ///
    /// The clamp is load-bearing at the degenerate end: an example whose every token is
    /// conditioning has a zero divisor, and the result must be **0**, not NaN.
    public static func videoLoss(prediction: [Float], target: [Float], lossMask: [Bool],
                                 shape: Shape) -> [Float] {
        precondition(prediction.count == shape.count && target.count == shape.count)
        precondition(lossMask.count == shape.tokens)

        var out = [Float](repeating: 0, count: shape.batch)
        let elements = Float(shape.sequence * shape.channels)
        for b in 0 ..< shape.batch {
            var maskedSum: Float = 0
            var maskSum: Float = 0
            for s in 0 ..< shape.sequence {
                guard lossMask[b * shape.sequence + s] else { continue }
                maskSum += 1
                for c in 0 ..< shape.channels {
                    let i = shape.index(b, s, c)
                    let error = prediction[i] - target[i]
                    maskedSum += error * error
                }
            }
            let numerator = maskedSum / elements
            let denominator = max(maskSum / Float(shape.sequence), 1e-8)
            out[b] = numerator / denominator
        }
        return out
    }

    /// Unmasked mean squared error, per batch element — the audio branch.
    ///
    /// **Unmasked is not an oversight.** The conditioning mask applies to the video stream
    /// only; the audio stream has no first-frame conditioning to exclude.
    /// Masking it "for symmetry" would change the objective.
    public static func audioLoss(prediction: [Float], target: [Float],
                                 shape: Shape) -> [Float] {
        precondition(prediction.count == shape.count && target.count == shape.count)
        var out = [Float](repeating: 0, count: shape.batch)
        let elements = Float(shape.sequence * shape.channels)
        for b in 0 ..< shape.batch {
            var sum: Float = 0
            for s in 0 ..< shape.sequence {
                for c in 0 ..< shape.channels {
                    let i = shape.index(b, s, c)
                    let error = prediction[i] - target[i]
                    sum += error * error
                }
            }
            out[b] = sum / elements
        }
        return out
    }

    /// `video_loss + audio_loss`, per batch element. The trainer reduces to a scalar; the
    /// unreduced form is what lets it track loss per sigma bucket.
    public static func combinedLoss(video: [Float], audio: [Float]?) -> [Float] {
        guard let audio else { return video }
        precondition(video.count == audio.count)
        return zip(video, audio).map(+)
    }
}

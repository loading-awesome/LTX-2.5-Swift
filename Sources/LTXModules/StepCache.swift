// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// Cross-step residual reuse — "first block cache".
///
/// Adjacent diffusion steps often produce nearly the same *change* to the
/// latent. Block 0 is a cheap probe: if its residual this step closely matches
/// last step's, skip blocks 1…47 and re-apply the previous **total** residual.
///
///     h1 = block0(h)
///     r  = h1 - h
///     if relative_change(r, r_prev) < threshold:
///         h_out = h + total_residual_prev
///     else:
///         h_out = blocks[1...](h1)
///         total_residual_prev = h_out - h
///
/// **Cache the residual, not the output.** Reusing an output pins the render
/// to a stale latent; reusing a delta applies the change wherever the
/// trajectory has actually reached.
///
/// ## One instance per guidance pass
///
/// LTX can run four forwards per step (conditional, unconditional, STG,
/// modality). Their residuals are not comparable. Sharing one cache across
/// them compares a conditional residual against an unconditional one and
/// reuses across the gap — every number stays finite. The sampler holds one
/// of these per `pass`.
///
/// LTX keeps video and audio as two tensors, not one packed sequence. The
/// whole-sequence probe is the element-count-weighted L1 over both; the audio
/// probe is audio alone and can veto.
public final class StepCache: @unchecked Sendable {

    public let pass: String
    public let policy: StepCachePolicy

    private var previousVideo: MLXArray?
    private var previousAudio: MLXArray?
    var cachedVideo: MLXArray?
    var cachedAudio: MLXArray?
    private var consecutiveSkips = 0

    public private(set) var stepsRun = 0
    public private(set) var stepsSkipped = 0
    public private(set) var trace: [Trace] = []

    public struct Trace: Sendable {
        public var step: Int
        public var wholeSequenceChange: Double
        public var videoChange: Double
        public var audioChange: Double
        public var decision: StepCachePolicy.Decision
        public var reason: StepCachePolicy.Reason
        public var constraints: [StepCachePolicy.Reason]
    }

    public enum Decision {
        case runFull
        case reuse(video: MLXArray, audio: MLXArray)
    }

    public init(pass: String, threshold: Double,
                maxConsecutiveSkips: Int = StepCachePolicy.measuredCap,
                perStream: Bool = true) {
        self.pass = pass
        self.policy = StepCachePolicy(threshold: threshold,
                                      maxConsecutiveSkips: maxConsecutiveSkips,
                                      perStream: perStream)
    }

    /// Decides after block 0 has run.
    ///
    /// - Parameters:
    ///   - videoProbe / audioProbe: block 0's residual, `h_after - h_in`, per stream.
    public func decide(videoProbe: MLXArray, audioProbe: MLXArray,
                       step: Int, totalSteps: Int) -> Decision {
        defer {
            previousVideo = videoProbe
            previousAudio = audioProbe
        }

        var whole = Double.infinity
        var audio = Double.infinity
        var video = Double.infinity
        if let pv = previousVideo, let pa = previousAudio,
           pv.shape == videoProbe.shape, pa.shape == audioProbe.shape {
            video = Self.relativeChange(videoProbe, pv)
            audio = Self.relativeChange(audioProbe, pa)
            whole = Self.wholeSequenceChange(currentVideo: videoProbe, currentAudio: audioProbe,
                                             previousVideo: pv, previousAudio: pa)
        }

        let cachedV = cachedVideo, cachedA = cachedAudio
        let have = cachedV != nil && cachedA != nil
            && cachedV?.shape == videoProbe.shape
            && cachedA?.shape == audioProbe.shape
        let verdict = policy.explain(
            wholeSequenceChange: whole,
            audioChange: policy.perStream ? audio : nil,
            videoChange: video,
            step: step, totalSteps: totalSteps,
            consecutiveSkips: consecutiveSkips,
            haveCachedResidual: have)

        trace.append(Trace(step: step, wholeSequenceChange: whole, videoChange: video,
                           audioChange: audio, decision: verdict.decision,
                           reason: verdict.reason, constraints: verdict.constraints))

        if verdict.decision == .reuse, let cachedV, let cachedA {
            consecutiveSkips += 1
            stepsSkipped += 1
            return .reuse(video: cachedV, audio: cachedA)
        }
        consecutiveSkips = 0
        stepsRun += 1
        return .runFull
    }

    public func record(videoResidual: MLXArray, audioResidual: MLXArray) {
        // Force the subtract. Each block output is already eval'd, so this is
        // one node, not the 48-block graph — but an uneval'd residual still
        // pins last step's in/out as parents, and a reuse then adds that node
        // instead of two buffers.
        MLX.eval(videoResidual, audioResidual)
        cachedVideo = videoResidual
        cachedAudio = audioResidual
    }

    public func release() {
        previousVideo = nil
        previousAudio = nil
        cachedVideo = nil
        cachedAudio = nil
    }

    public var summary: String {
        let reasons = Dictionary(grouping: trace, by: \.reason)
            .map { "\($0.key.rawValue)=\($0.value.count)" }
            .sorted()
            .joined(separator: ",")
        return "pass=\(pass) threshold=\(policy.threshold) cap=\(policy.maxConsecutiveSkips) "
            + "run=\(stepsRun) reused=\(stepsSkipped) reasons=[\(reasons)]"
    }

    /// Relative L1. Mean absolute rather than RMS: the probe is dominated by a
    /// few large rows, and RMS would let a big change in a small region hide
    /// behind a quiet majority.
    static func relativeChange(_ current: MLXArray, _ previous: MLXArray) -> Double {
        let denom = Double(MLX.mean(MLX.abs(previous)).item(Float.self))
        guard denom > 0 else { return .infinity }
        return Double(MLX.mean(MLX.abs(current - previous)).item(Float.self)) / denom
    }

    /// Concat-equivalent relative L1 over both streams, weighted by element count.
    static func wholeSequenceChange(currentVideo: MLXArray, currentAudio: MLXArray,
                                    previousVideo: MLXArray, previousAudio: MLXArray) -> Double {
        let nv = Double(previousVideo.size), na = Double(previousAudio.size)
        let den = nv * Double(MLX.mean(MLX.abs(previousVideo)).item(Float.self))
            + na * Double(MLX.mean(MLX.abs(previousAudio)).item(Float.self))
        guard den > 0 else { return .infinity }
        let num = nv * Double(MLX.mean(MLX.abs(currentVideo - previousVideo)).item(Float.self))
            + na * Double(MLX.mean(MLX.abs(currentAudio - previousAudio)).item(Float.self))
        return num / den
    }
}

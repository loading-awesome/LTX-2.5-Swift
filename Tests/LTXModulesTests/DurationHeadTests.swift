// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXFoundation
@testable import LTXModules

/// The duration head: connector token outputs → seconds → frames.
///
/// ## What this suite can and cannot assert
///
/// There is no ground-truth mapping from prompts to durations anywhere in this repo, so
/// **no test here asserts that a predicted duration is right**, or monotonic, or plausible.
/// What is left is structure, and it is worth more than it sounds:
///
/// * **Key accounting.** All fifteen checkpoint tensors are consumed at the shapes the
///   config predicts and in bf16, and nothing the module tree wants is absent. This is the
///   highest-value test here by some distance: a structurally wrong duration head runs to
///   completion and returns a perfectly plausible number of seconds, with no shape error
///   anywhere downstream.
/// * **Configuration.** The head count is the one number the checkpoint cannot record —
///   `"duration_head": {}` is empty — so it comes from the configurator's default, and that
///   default is pinned. Nothing else constrains it: no tensor shape changes when it is
///   wrong, and the prediction moves by seconds rather than failing.
/// * **Plumbing, on synthetic weights.** The forward runs end to end from either modality
///   or both and over ragged token counts, and the pooled vector stays fixed-width whatever
///   the sequence length — which is what a reversed query/key direction breaks. That the
///   features reach the output at all is asserted on `logSeconds` rather than `seconds`,
///   because bf16's spacing at these magnitudes swallows the difference; a head wired to
///   ignore its input and return a constant would pass everything else in the file.
/// * **Negative controls.** No modality, the two streams swapped, the wrong rank, an empty
///   sequence, a batch of two, and a head count that does not divide the pooler width are
///   each refused with their own error rather than coerced into an answer.
///
/// One thing no test here can see is the concatenation order. The pooler is one query
/// cross-attending over the concatenation of the two modalities, and attention is
/// permutation-invariant in its keys and values, so video-first and audio-first differ by
/// 1e-6 seconds.
///
/// The lattice half is the opposite case: `FRAGILE_CONTRACTS.md` #4 is a written-down
/// contract with exact arithmetic, and it is asserted hard.
@Suite("Duration head", .serialized)
struct DurationHeadTests {

    // MARK: - Fixtures

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root! + "/model_patches/"
        + "ltx-2.5-duration-head-bf16.safetensors"

    static var checkpointURL: URL? {
        FileManager.default.fileExists(atPath: checkpointPath)
            ? URL(fileURLWithPath: checkpointPath) : nil
    }

    /// The configuration this checkpoint yields, for the tests that only need the shape of
    /// the module tree and not its values.
    static let syntheticConfig = DurationHead.Configuration(
        videoWidth: 4096, audioWidth: 2048, poolerHidden: 256, queryCount: 1,
        poolerHeads: 4, mlpHidden: 256)

    /// A deterministic weight set, so the plumbing and negative-control tests run in
    /// milliseconds without the 3.8 MB file — and so "determinism" tests determinism rather
    /// than the stability of a random number generator.
    ///
    /// Values are a fixed low-amplitude sequence keyed on the tensor name. Not `MLXRandom`:
    /// a seeded global stream makes a test's result depend on how many tests ran before it.
    static func syntheticWeights(_ c: DurationHead.Configuration) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (name, shape) in DurationHead.parameterShapes(for: c) {
            let count = shape.reduce(1, *)
            var seed = UInt64(truncatingIfNeeded: name.utf8.reduce(1_469_598_103) {
                ($0 &* 1_099_511_628_211) ^ UInt64($1)
            })
            var values = [Float](repeating: 0, count: count)
            for i in 0 ..< count {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                values[i] = (Float(seed >> 40) / Float(1 << 24) - 0.5) * 0.05
            }
            // bf16, as the real file is: the dtype is what drives every cast in the forward.
            out[name] = MLXArray(values, shape).asType(.bfloat16)
        }
        return out
    }

    static func syntheticHead() throws -> DurationHead {
        try DurationHead(weights: syntheticWeights(syntheticConfig),
                         configuration: syntheticConfig)
    }

    static func features(_ c: DurationHead.Configuration, tokens: Int = 1024,
                         variant: Int = 0) -> (video: MLXArray, audio: MLXArray) {
        func fill(_ width: Int, _ salt: Int) -> MLXArray {
            var seed = UInt64(bitPattern: Int64(salt &* 7_919 &+ 7))
            var values = [Float](repeating: 0, count: tokens * width)
            let scale = Float(variant + 1)
            let shift = Float(variant) * 0.5
            for i in 0 ..< values.count {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                values[i] = (Float(seed >> 40) / Float(1 << 24) - 0.5) * scale + shift
            }
            return MLXArray(values, [1, tokens, width])
        }
        return (fill(c.videoWidth, 1 + variant * 2), fill(c.audioWidth, 2 + variant * 2))
    }

    // MARK: - Key accounting
    //
    // The highest-value test here by some distance. A structurally wrong duration head runs
    // to completion and returns a plausible number of seconds, with no shape error anywhere
    // downstream — so "every tensor is consumed at the shape the config predicts, and
    // nothing the tree wants is absent" is most of what can be checked.

    @Test("all fifteen checkpoint tensors are consumed at the shapes the config predicts")
    func keyAccounting() throws {
        guard let url = Self.checkpointURL else {
            print("SKIP keyAccounting: no checkpoint at \(Self.checkpointPath)")
            return
        }
        let header = try SafetensorsHeader.read(from: url)
        let config = try DurationHead.configuration(from: header)
        let head = try DurationHead(checkpoint: url)

        #expect(header.tensors.count == 15,
                "the patch file should hold exactly the head; got \(header.tensors.count)")
        #expect(head.weights.count == 15)

        let expected = DurationHead.parameterShapes(for: config)
        #expect(expected.count == 15)
        for (name, shape) in expected.sorted(by: { $0.key < $1.key }) {
            let stored = head.weights[name]
            #expect(stored != nil, "\(name) was not consumed")
            #expect(stored?.shape == shape, "\(name) is \(stored?.shape ?? []), want \(shape)")
            // The file is bf16 throughout and nothing here widens it — the matmuls are meant
            // to run at the weight's dtype.
            #expect(stored?.dtype == .bfloat16, "\(name) is \(stored?.dtype ?? .float32)")
        }
        print("keyAccounting: 15/15 consumed, all bf16, config \(config)")
    }

    @Test("an extra tensor and a missing tensor each fail loudly")
    func accountingFailsLoudly() throws {
        var weights = Self.syntheticWeights(Self.syntheticConfig)

        weights["duration_head.something_new"] = MLXArray([Float(0)], [1])
        #expect(throws: DurationHead.Failure.self) {
            _ = try DurationHead(weights: weights, configuration: Self.syntheticConfig)
        }
        weights.removeValue(forKey: "duration_head.something_new")

        weights.removeValue(forKey: "attention_pooler.cross_attn.in_proj_bias")
        #expect(throws: DurationHead.Failure.self) {
            _ = try DurationHead(weights: weights, configuration: Self.syntheticConfig)
        }
    }

    @Test("config is read from the checkpoint, including the head count it cannot record")
    func configuration() throws {
        guard let url = Self.checkpointURL else {
            print("SKIP configuration: no checkpoint at \(Self.checkpointPath)")
            return
        }
        let header = try SafetensorsHeader.read(from: url)
        let config = try DurationHead.configuration(from: header)

        // Widths come from `config.transformer.*`, cross-checked against the projections.
        #expect(config.videoWidth == 4096)
        #expect(config.audioWidth == 2048)
        #expect(config.poolerHidden == 256)
        #expect(config.queryCount == 1)
        #expect(config.mlpHidden == 256)
        // Not in the file — `"duration_head": {}` is empty, so this is the configurator's
        // default, which is what this checkpoint is meant to run with.
        #expect(config.poolerHeads == DurationHead.referencePoolerHeads)
        #expect(config.headDim == 64)

        #expect(header.metadata["model_version"] == "2.5.0")
        print("configuration: \(config) (poolerHeads from the reference default, "
            + "not from the file)")
    }

    // MARK: - Shape and plumbing

    @Test("full-shape features give a finite scalar")
    func plumbing() throws {
        let head = try Self.syntheticHead()
        let (video, audio) = Self.features(Self.syntheticConfig)
        #expect(video.shape == [1, 1024, 4096])
        #expect(audio.shape == [1, 1024, 2048])

        let seconds = try head.seconds(video: video, audio: audio)
        #expect(seconds.isFinite)
        #expect(seconds > 0, Comment(rawValue: "the head ends in exp(); a non-positive "
            + "result is impossible unless the exponential was dropped"))
        print("plumbing: synthetic weights predict \(seconds) s (no meaning — random "
            + "weights, printed to show the path runs end to end)")

        // Either modality alone, the head's other two entry points.
        #expect(try head.seconds(video: video, audio: nil).isFinite)
        #expect(try head.seconds(video: nil, audio: audio).isFinite)

        // Ragged token counts: the two streams are concatenated along the token axis, not
        // stacked, so they need not agree.
        let short = Self.features(Self.syntheticConfig, tokens: 128)
        #expect(try head.seconds(video: video, audio: short.audio).isFinite)
    }

    @Test("the pooler output is fixed-width regardless of sequence length")
    func poolerIsLengthInvariant() throws {
        let head = try Self.syntheticHead()
        for tokens in [8, 128, 1024] {
            let (video, audio) = Self.features(Self.syntheticConfig, tokens: tokens)
            let log = try head.logSeconds(video: video, audio: audio)
            // `[B]` after the trailing squeeze. Anything else means the query/key direction
            // was reversed and the "pooled" vector is still sequence-shaped.
            #expect(log.shape == [1], "tokens=\(tokens) gave \(log.shape)")
        }
    }

    // MARK: - Determinism

    @Test("the same input twice gives the identical scalar")
    func determinism() throws {
        let head = try Self.syntheticHead()
        let (video, audio) = Self.features(Self.syntheticConfig)
        let a = try head.seconds(video: video, audio: audio)
        let b = try head.seconds(video: video, audio: audio)
        let c = try head.seconds(video: video, audio: audio)
        #expect(a == b)
        #expect(b == c)
        print("determinism: \(a) three times")
    }

    @Test("the conditioning reaches the output")
    func differentInputsMove() throws {
        // Deliberately **not** a monotonicity or plausibility claim. There is no ground
        // truth mapping prompts to durations anywhere in this repo, so all that is asserted
        // is that the input reaches the output at all — a head wired to ignore its features
        // and return a constant would pass every other test in this file.
        //
        // Asserted on `logSeconds`, not on `seconds`, and the reason is itself worth
        // recording: with these deliberately small synthetic weights the two predictions are
        // 1.0201 s and 1.0206 s, and **bf16's spacing at 1.02 is 0.0078**, so both round to
        // the same 1.0234375. The forward is sensitive; the returned scalar's precision is
        // not.
        let head = try Self.syntheticHead()
        let first = Self.features(Self.syntheticConfig, variant: 0)
        let second = Self.features(Self.syntheticConfig, variant: 3)
        let a = try head.logSeconds(video: first.video, audio: first.audio).item(Float.self)
        let b = try head.logSeconds(video: second.video, audio: second.audio).item(Float.self)
        print("differentInputsMove: log-seconds \(a) vs \(b) -> "
            + "\(try head.seconds(video: first.video, audio: first.audio)) s vs "
            + "\(try head.seconds(video: second.video, audio: second.audio)) s "
            + "(synthetic weights; no meaning beyond the input being observable)")
        #expect(a != b)
    }

    // MARK: - The lattice
    //
    // This half IS a contract — FRAGILE_CONTRACTS.md #4 — so unlike the prediction it is
    // asserted hard. `FrameLattice` owns the arithmetic; what is checked here is that
    // `DurationHead` routes through it with the documented clamp defaults and does not
    // reimplement any part of it.

    @Test("snapped frame counts are on the 8k+1 lattice and snap down")
    func latticeSnapsDown() {
        let lattice = FrameLattice()
        let frameRate = 24.0
        let minSeconds = DurationHead.defaultMinSeconds     // 1.0
        let maxSeconds = DurationHead.defaultMaxSeconds     // 20.0
        let minFrames = Int((minSeconds * frameRate).rounded())     // 24
        let maxFrames = Int((maxSeconds * frameRate).rounded())     // 480

        var spread: [Double] = []
        for tenths in 1 ... 250 { spread.append(Double(tenths) / 10.0) }
        spread += [0.001, 0.4, 0.9999, 1.0, 1.0001, 19.999, 20.0, 20.0001, 100.0]

        for seconds in spread {
            let frames = DurationHead.snap(seconds: seconds, frameRate: frameRate,
                                           minSeconds: minSeconds, maxSeconds: maxSeconds)
            #expect(lattice.contains(frames),
                    "\(seconds)s gave \(frames), which is not 8k+1")
            #expect(frames >= minFrames && frames <= maxFrames,
                    "\(seconds)s gave \(frames), outside [\(minFrames), \(maxFrames)]")

            // The clamp is applied *before* the snap. A clamp applied afterwards gives
            // different counts near both bounds.
            let raw = max(minFrames, min(Int((seconds * frameRate).rounded()), maxFrames))
            if lattice.snapDown(raw) >= minFrames {
                // The ordinary case: snap down, never up.
                #expect(frames == lattice.snapDown(raw),
                        Comment(rawValue: "\(seconds)s: expected snapDown(\(raw)) = "
                            + "\(lattice.snapDown(raw)), got \(frames)"))
                #expect(frames <= raw, "\(seconds)s snapped UP from \(raw) to \(frames)")
            } else {
                // The one asymmetry: a floor that undershoots the minimum is raised to the
                // next lattice point instead of left short.
                #expect(frames == min(lattice.snapUp(minFrames), maxFrames))
                #expect(frames > raw || raw == frames)
            }
        }
        print("latticeSnapsDown: \(spread.count) durations, all on 8k+1 in "
            + "[\(minFrames), \(maxFrames)]")
    }

    @Test("the minimum bump is the only place snapping goes up")
    func latticeMinimumBump() {
        // 24 fps, min 1 s: minFrames = 24, and snapDown(24) = 17 < 24, so the bump fires and
        // the answer is snapUp(24) = 25. This is the minimum bump, and it is the only
        // upward move in the whole function.
        #expect(DurationHead.snap(seconds: 0.1, frameRate: 24, minSeconds: 1, maxSeconds: 20)
            == 25)
        #expect(DurationHead.snap(seconds: 1.0, frameRate: 24, minSeconds: 1, maxSeconds: 20)
            == 25)

        // A minimum that already sits on the lattice does not bump: minFrames = 25,
        // snapDown(25) = 25.
        #expect(DurationHead.snap(seconds: 0.1, frameRate: 25, minSeconds: 1, maxSeconds: 20)
            == 25)

        // Above the minimum it floors. 5 s at 24 fps is 120 frames -> 113, not 121.
        #expect(DurationHead.snap(seconds: 5.0, frameRate: 24, minSeconds: 1, maxSeconds: 20)
            == 113)
        // And 4.990764 s lands in the same place, which is why the fp32/bf16 spread in the
        // prediction — a few hundredths of a second — is invisible at the lattice.
        #expect(DurationHead.snap(seconds: 4.990764, frameRate: 24) == 113)
        #expect(DurationHead.snap(seconds: 5.0, frameRate: 24) == 113)

        // The maximum caps before the snap: 100 s clamps to 480 frames, which floors to 473.
        #expect(DurationHead.snap(seconds: 100, frameRate: 24, minSeconds: 1, maxSeconds: 20)
            == 473)
    }

    // MARK: - Negative controls

    @Test("no modality at all is refused")
    func refusesNoModality() throws {
        let head = try Self.syntheticHead()
        #expect(throws: DurationHead.Failure.self) {
            _ = try head.seconds(video: nil, audio: nil)
        }
        do {
            _ = try head.seconds(video: nil, audio: nil)
            Issue.record("expected .noModality")
        } catch let failure as DurationHead.Failure {
            guard case .noModality = failure else {
                Issue.record("expected .noModality, got \(failure)")
                return
            }
        }
    }

    @Test("the wrong feature width is refused, including the two streams swapped")
    func refusesWrongWidth() throws {
        let head = try Self.syntheticHead()
        let (video, audio) = Self.features(Self.syntheticConfig, tokens: 16)

        // The interesting case: both streams present, both the right rank, swapped.
        do {
            _ = try head.seconds(video: audio, audio: video)
            Issue.record("expected .featureWidth for swapped streams")
        } catch let failure as DurationHead.Failure {
            guard case let .featureWidth(name, got, expected) = failure else {
                Issue.record("expected .featureWidth, got \(failure)")
                return
            }
            #expect(name == "video")
            #expect(got == 2048)
            #expect(expected == 4096)
        }

        let narrow = MLXArray([Float](repeating: 0, count: 16 * 100), [1, 16, 100])
        do {
            _ = try head.seconds(video: narrow, audio: nil)
            Issue.record("expected .featureWidth")
        } catch let failure as DurationHead.Failure {
            guard case .featureWidth = failure else {
                Issue.record("expected .featureWidth, got \(failure)")
                return
            }
        }
    }

    @Test("the wrong rank is refused")
    func refusesWrongRank() throws {
        let head = try Self.syntheticHead()
        // A pooled vector rather than tokens — the shape a caller reaches for when they
        // think this head takes a sentence embedding.
        let pooled = MLXArray([Float](repeating: 0, count: 4096), [1, 4096])
        do {
            _ = try head.seconds(video: pooled, audio: nil)
            Issue.record("expected .rank")
        } catch let failure as DurationHead.Failure {
            guard case let .rank(name, got) = failure else {
                Issue.record("expected .rank, got \(failure)")
                return
            }
            #expect(name == "video")
            #expect(got == [1, 4096])
        }
    }

    @Test("an empty sequence is refused rather than returning NaN seconds")
    func refusesEmptySequence() throws {
        let head = try Self.syntheticHead()
        let empty = MLXArray([Float](), [1, 0, 4096])
        do {
            _ = try head.seconds(video: empty, audio: nil)
            Issue.record("expected .emptySequence")
        } catch let failure as DurationHead.Failure {
            guard case let .emptySequence(name) = failure else {
                Issue.record("expected .emptySequence, got \(failure)")
                return
            }
            #expect(name == "video")
        }
    }

    @Test("a batch other than one is refused")
    func refusesBatch() throws {
        let head = try Self.syntheticHead()
        // Both guidance branches stacked — the shape a caller gets by handing over the
        // pipeline's two encodings at once.
        let pair = MLXArray([Float](repeating: 0, count: 2 * 16 * 4096), [2, 16, 4096])
        do {
            _ = try head.seconds(video: pair, audio: nil)
            Issue.record("expected .batch")
        } catch let failure as DurationHead.Failure {
            guard case let .batch(name, got) = failure else {
                Issue.record("expected .batch, got \(failure)")
                return
            }
            #expect(name == "video")
            #expect(got == 2)
        }
    }

    @Test("a head count that does not divide the pooler width is refused at load")
    func refusesIndivisibleHeadCount() {
        let bad = DurationHead.Configuration(videoWidth: 4096, audioWidth: 2048,
                                             poolerHidden: 256, queryCount: 1,
                                             poolerHeads: 7, mlpHidden: 256)
        #expect(throws: DurationHead.Failure.self) {
            _ = try DurationHead(weights: Self.syntheticWeights(bad), configuration: bad)
        }
    }
}

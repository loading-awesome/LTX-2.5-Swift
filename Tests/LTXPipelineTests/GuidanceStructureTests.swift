// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXPipeline

/// The sampler's guidance structure, checked against the fixture's **call counts**.
///
/// No tensors and no GPU. The fixture records how many times each module fired, and that
/// count is a direct observation of the structure: how many forward passes a step costs,
/// which of them sever cross-modal attention, and how often text conditioning is encoded.
/// None of that is visible in any tensor's shape.
///
/// It says nothing about the sampler's numerics and must not be read as doing so.
@Suite("Guidance structure")
struct GuidanceStructureTests {

    /// The guiders as the fixture's manifest records them.
    ///
    /// Read from the manifest at run time rather than transcribed, so a fixture built with
    /// different guidance re-points this suite instead of silently invalidating it.
    struct Recorded {
        let steps: Int
        let calls: [String: Int]
    }

    static func recorded() -> Recorded? {
        let path = "Tests/Fixtures/goldens/tiny_v5/manifest.json"
        let candidates = [path, FileManager.default.currentDirectoryPath + "/" + path]
        for p in candidates {
            guard let data = FileManager.default.contents(atPath: p),
                  let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let params = root["params"] as? [String: Any],
                  let steps = params["steps"] as? Int,
                  let tensors = root["tensors"] as? [String: Any]
            else { continue }

            // `<family>.call<NNN>` counts monotonically across the whole stage, so the
            // highest index plus one is the number of times that module fired.
            var calls: [String: Int] = [:]
            for name in tensors.keys {
                guard let range = name.range(of: ".call", options: .backwards) else {
                    continue
                }
                let family = String(name[name.startIndex..<range.lowerBound])
                let suffix = name[range.upperBound...]
                guard let index = Int(suffix.prefix(3)) else { continue }
                calls[family] = max(calls[family] ?? 0, index + 1)
            }
            return Recorded(steps: steps, calls: calls)
        }
        return nil
    }

    /// The guidance settings this fixture was built with.
    ///
    /// cfg 3.0 / stg 1.0 over block 28 / rescale 0.7 / modality 3.0 / skip step 0 for
    /// video, and the same with cfg 7.0 for audio. Every one of those is non-neutral,
    /// which is why every step here pays for all four passes.
    static let video = GuidanceParams(cfgScale: 3.0, stgScale: 1.0, stgBlocks: [28],
                                      rescaleScale: 0.7, modalityScale: 3.0, skipStep: 0)
    static let audio = GuidanceParams(cfgScale: 7.0, stgScale: 1.0, stgBlocks: [28],
                                      rescaleScale: 0.7, modalityScale: 3.0, skipStep: 0)

    /// Four transformer calls per step.
    ///
    /// `adaln_single.in` fires **12** times over **3** steps, which is also why the call
    /// index counts monotonically across a whole stage rather than resetting per step: a
    /// per-step counter would have steps 0 and 1 both emit index 0, and the second would
    /// overwrite the first.
    @Test("the sampler plans exactly the number of passes the golden recorded")
    func passCountMatchesTheGolden() throws {
        guard let r = Self.recorded() else { return }
        let fired = try #require(r.calls["adaln_single.in"])

        let sampler = Sampler(video: Self.video, audio: Self.audio)
        let planned = sampler.passCount(steps: r.steps)
        print("golden fired adaln_single.in \(fired)x over \(r.steps) steps; "
            + "sampler plans \(planned)")
        #expect(planned == fired,
                Comment(rawValue: "sampler plans \(planned) transformer calls for "
                        + "\(r.steps) steps; the golden recorded \(fired)"))
        #expect(planned == r.steps * 4,
                Comment(rawValue: "every guidance term on this capture is non-neutral, "
                        + "so each step should cost all four passes"))
    }

    /// The call index is derived from this pair, not from a separate mutable counter:
    /// `step * 4 + pass.rawValue`. The branch meaning is part of the index — conditional is
    /// the positive context and unconditional the negative one — so changing the pass order
    /// would line a valid tensor up against the wrong branch.
    @Test("call indices retain the pass and branch order")
    func callIndicesRetainPassAndBranchOrder() {
        let sampler = Sampler(video: Self.video, audio: Self.audio)
        let expected: [Sampler.Pass] = [.conditional, .unconditional, .perturbed, .modality]
        for step in 0..<3 {
            let passes = sampler.plan(step: step).passes
            #expect(passes == expected,
                    Comment(rawValue: "step \(step) changed the "
                        + "conditional/unconditional/perturbed/modality order"))
            let calls = passes.map { step * expected.count + $0.rawValue }
            #expect(calls == Array((step * expected.count)..<((step + 1) * expected.count)),
                    Comment(rawValue: "step \(step) did not map pass identity to "
                        + "monotonic callNNN"))
        }
        #expect(1 * expected.count + Sampler.Pass.unconditional.rawValue == 5,
                Comment(rawValue: "call005 is the negative (unconditional) branch of step 1"))
    }

    /// Every module on the main path fires once per pass — except the cross-modal pair.
    ///
    /// **`audio_to_video_attn` and `video_to_audio_attn` fire 9 times where `attn1` and
    /// `ff` fire 12.** That difference is exactly one pass per step, and it is the
    /// `.modality` pass: it severs both cross-modal attentions, so they are never called
    /// on it. The count is therefore an independent observation that the modality pass
    /// exists, that it runs once per step, and that severing is how it is implemented —
    /// none of which is visible in any tensor's shape.
    ///
    /// A port that implemented modality guidance by *zeroing* the cross-modal output
    /// instead of skipping the call would produce identical numbers and 12 fires. This is
    /// the check that tells them apart.
    @Test("the modality pass severs cross-modal attention rather than zeroing it")
    func modalityPassSeversCrossModal() throws {
        guard let r = Self.recorded() else { return }
        let mainPath = try #require(r.calls["block_00.attn1.out"])
        let a2v = try #require(r.calls["block_00.audio_to_video_attn.out"])
        let v2a = try #require(r.calls["block_00.video_to_audio_attn.out"])

        print("block_00: attn1 \(mainPath)x, audio_to_video \(a2v)x, "
            + "video_to_audio \(v2a)x over \(r.steps) steps")
        #expect(a2v == v2a, "the two directions are severed together or not at all")
        #expect(mainPath - a2v == r.steps,
                Comment(rawValue: "cross-modal fired \(a2v) against \(mainPath) on the "
                        + "main path, a difference of \(mainPath - a2v); the modality "
                        + "pass runs once per step, so it should be \(r.steps)"))

        // And the sampler agrees that the modality pass is planned once per step.
        let sampler = Sampler(video: Self.video, audio: Self.audio)
        let modalityPasses = (0..<r.steps).reduce(0) {
            $0 + (sampler.plan(step: $1).passes.contains(.modality) ? 1 : 0)
        }
        #expect(modalityPasses == mainPath - a2v,
                Comment(rawValue: "sampler plans \(modalityPasses) modality passes; the "
                        + "golden's tap counts imply \(mainPath - a2v)"))
    }

    /// Text conditioning runs once per **guidance branch**, not once per pass.
    ///
    /// `enc.connector.video.in` fires **twice** — positive and negative prompt — against
    /// the sampler's 12 transformer calls. The encode happens before sampling and its
    /// result is reused across every step, so both of its calls see byte-identical inputs.
    @Test("text conditioning runs per guidance branch, not per pass")
    func encodeRunsPerBranch() throws {
        guard let r = Self.recorded() else { return }
        let encode = try #require(r.calls["enc.connector.video.in"])
        let passes = try #require(r.calls["adaln_single.in"])
        print("encode fired \(encode)x against \(passes) transformer calls")
        #expect(encode == 2,
                Comment(rawValue: "encode fired \(encode)x; two guidance branches were "
                        + "expected"))
        #expect(encode < passes, "encoding once per pass would be a wasted 12 encodes")
    }

    /// Turning off one guidance term costs exactly one pass, not a fraction of one.
    ///
    /// Guidance costs are additive, checked directly. It matters because an implementation
    /// that computes a disabled term and multiplies it by zero is byte-identical and twice
    /// the price — invisible to any comparison of the numbers.
    @Test("each guidance term costs exactly one pass")
    func guidanceCostsAreAdditive() {
        let steps = 3
        let full = Sampler(video: Self.video, audio: Self.audio).passCount(steps: steps)

        // Neutralise one term at a time, in BOTH streams — `plan` takes the union, so a
        // term left on in either stream still costs the pass.
        func cost(_ mutate: (inout GuidanceParams) -> Void) -> Int {
            var v = Self.video, a = Self.audio
            mutate(&v); mutate(&a)
            return Sampler(video: v, audio: a).passCount(steps: steps)
        }
        #expect(full - cost { $0.cfgScale = 1.0 } == steps, "CFG costs one pass per step")
        #expect(full - cost { $0.stgScale = 0.0 } == steps, "STG costs one pass per step")
        #expect(full - cost { $0.modalityScale = 1.0 } == steps,
                "modality guidance costs one pass per step")

        // **An empty STG block list does NOT disable the perturbed pass.** Whether that
        // pass runs is decided by `stgScale` alone; the block list only decides what the
        // perturbation touches, so an empty list runs the pass and perturbs nothing.
        //
        // The distinction is invisible in the output — a perturbation over no blocks
        // returns the conditional prediction, and `stgScale * (cond - ptb)` is zero either
        // way — and costs one full forward per step. Cost is the only thing that can see
        // it, which is why it is asserted here rather than left to the numbers.
        #expect(cost { $0.stgBlocks = [] } == full,
                Comment(rawValue: "an empty STG block list must still cost the perturbed "
                        + "pass on this pipeline, whose perturbation trigger "
                        + "tests stg_scale alone"))
        #expect(cost { $0.stgScale = 0.0 } < full,
                "a zero STG scale is the route that does disable the pass")
    }
}

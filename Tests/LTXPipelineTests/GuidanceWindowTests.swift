// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
@testable import LTXPipeline

@Suite("guidance windows")
struct GuidanceWindowTests {

    static let full = GuidanceParams(
        cfgScale: 3.0, stgScale: 1.0, stgBlocks: [28], rescaleScale: 0.7,
        modalityScale: 3.0)

    @Test("default [0, 1] window is every sampled sigma")
    func defaultWindowIsTheWholeSchedule() {
        let p = Self.full
        #expect(p.runsPerturbed(atSigma: 1.0))
        #expect(p.runsPerturbed(atSigma: 0.5))
        #expect(p.runsPerturbed(atSigma: 0.01))
        #expect(p.runsIsolatedModality(atSigma: 1.0))
        let sampler = Sampler(video: p, audio: p)
        #expect(sampler.passCount(steps: 4) == 16)
        #expect(sampler.plan(step: 0, sigma: 0.9).passes.contains(.perturbed))
        #expect(sampler.plan(step: 0, sigma: 0.9).passes.contains(.modality))
    }

    @Test("empty STG window (start=end=1) drops the perturbed pass")
    func emptySTGWindowDropsThePass() {
        var p = Self.full
        p.stgStartPercent = 1
        p.stgEndPercent = 1
        let sampler = Sampler(video: p, audio: p)
        #expect(sampler.passCount(steps: 4) == 12)
        #expect(!sampler.plan(step: 0, sigma: 0.9).passes.contains(.perturbed))
        #expect(sampler.plan(step: 0, sigma: 0.9).passes.contains(.modality))
        #expect(p.effective(atSigma: 0.9).stgScale == 0)
        #expect(p.effective(atSigma: 0.9).modalityScale == 3)
    }

    @Test("STG window [0, 0.5] is the high-sigma half")
    func stgWindowIsHighSigmaHalf() {
        var p = Self.full
        p.stgStartPercent = 0
        p.stgEndPercent = 0.5
        // p=0 → sigma 1; p=0.5 → sigma 0.5. Inside: 0.5 ≤ sigma ≤ 1.
        #expect(GuidanceParams.sigmaIsInsideWindow(0.9, start: 0, end: 0.5))
        #expect(GuidanceParams.sigmaIsInsideWindow(0.5, start: 0, end: 0.5))
        #expect(!GuidanceParams.sigmaIsInsideWindow(0.3, start: 0, end: 0.5))
        let sampler = Sampler(video: p, audio: p)
        #expect(sampler.plan(step: 0, sigma: 0.9).passes.contains(.perturbed))
        #expect(!sampler.plan(step: 1, sigma: 0.3).passes.contains(.perturbed))
        #expect(sampler.plan(step: 1, sigma: 0.3).passes.contains(.modality))
    }

    @Test("a pass that ran for the other stream does not apply this stream's term")
    func otherStreamDoesNotInheritTheWindow() {
        var video = Self.full
        video.stgStartPercent = 0
        video.stgEndPercent = 0.5
        var audio = Self.full
        audio.stgStartPercent = 1
        audio.stgEndPercent = 1
        let sampler = Sampler(video: video, audio: audio)
        let plan = sampler.plan(step: 0, sigma: 0.9)
        #expect(plan.passes.contains(.perturbed),
                "video is inside the window, so the pass still runs")
        #expect(audio.effective(atSigma: 0.9).stgScale == 0)
        #expect(video.effective(atSigma: 0.9).stgScale == 1)
    }
}

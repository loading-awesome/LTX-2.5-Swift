// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXRecipes

/// Which flags `--two-stage` accepts, and which it refuses **by name**.
///
/// The line these tests hold is drawn from what the two-stage pipeline can actually carry,
/// rather than from what is convenient to implement:
///
/// * a still image is the pipeline's **only** conditioning input. It is prepared twice —
///   at half size for stage 1 and at full size for stage 2 — so `--image` is supportable
///   and is supported.
/// * there is no video conditioning input, and neither stage's audio modality carries a
///   conditioning list, so `--video`, `--audio` and `--audio-mel` have nowhere to enter.
///   They are refused rather than silently dropped.
/// * both stages' sigmas are fixed literals and no guider is constructed, so `--steps` and
///   every guidance knob remain refused.
@Suite("Two-stage validation")
struct TwoStageValidationTests {

    private func request(image: URL? = nil, video: URL? = nil, audio: URL? = nil,
                         audioMel: URL? = nil, steps: Int? = nil,
                         videoCFG: Double? = nil, stgScale: Double? = nil,
                         negativePrompt: String? = nil,
                         strength: Double = 1.0,
                         cacheThreshold: Double = 0) -> RecipeRequest {
        RecipeRequest(
            prompt: "x",
            videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"),
            negativePrompt: negativePrompt,
            frames: 97,
            steps: steps,
            image: image,
            video: video,
            audio: audio,
            audioMel: audioMel,
            strength: strength,
            videoCFG: videoCFG,
            stgScale: stgScale,
            cacheThreshold: cacheThreshold,
            twoStage: true)
    }

    private static let png = URL(fileURLWithPath: "/tmp/anchor.png")
    private static let mp4 = URL(fileURLWithPath: "/tmp/clip.mp4")
    private static let wav = URL(fileURLWithPath: "/tmp/take.wav")
    private static let mel = URL(fileURLWithPath: "/tmp/mel.safetensors")

    @Test("--image is accepted on the two-stage route")
    func imageIsAccepted() throws {
        try request(image: Self.png).validate()
        try request(image: Self.png, strength: 0.85).validate()
    }

    /// The refusal must *name the flag*, so the message tells a caller which of four
    /// conditioning arguments it objected to.
    @Test("--video is refused by name, with the reason")
    func videoIsRefusedByName() {
        #expect {
            try request(video: Self.mp4).validate()
        } throws: { error in
            guard case let RecipeFailure.twoStageIgnores(detail) = error else { return false }
            return detail.contains("--video") && detail.contains("stills")
        }
    }

    @Test("--audio and --audio-mel are refused by name, with the reason")
    func audioIsRefusedByName() {
        for (flag, request) in [("--audio", request(audio: Self.wav)),
                                ("--audio-mel", request(audioMel: Self.mel))] {
            #expect {
                try request.validate()
            } throws: { error in
                guard case let RecipeFailure.twoStageIgnores(detail) = error else { return false }
                return detail.contains(flag) && detail.contains("conditioning slot")
            }
        }
    }

    /// Everything the previous refusal set covered, minus `--image`, still refused.
    @Test("--steps and the guidance knobs are still refused")
    func theOtherRefusalsAreUnchanged() {
        for request in [request(steps: 12), request(videoCFG: 3.0), request(stgScale: 1.0),
                        request(videoCFG: 3.0, negativePrompt: "blurry")] {
            #expect(throws: RecipeFailure.self) { try request.validate() }
        }
    }

    @Test("--cache-threshold is refused on the two-stage route")
    func cacheIsRefused() {
        #expect {
            try request(cacheThreshold: 0.10).validate()
        } throws: { error in
            guard case let RecipeFailure.twoStageIgnores(detail) = error else { return false }
            return detail.contains("--cache-threshold")
        }
    }

    @Test("a narrowed guidance window is refused on the two-stage route")
    func windowsAreRefused() {
        var r = request()
        r.recipeID = RecipeRegistry.distilled.id
        r.stgEndPercent = 0.5
        #expect {
            try r.validate()
        } throws: { error in
            guard case let RecipeFailure.twoStageIgnores(detail) = error else { return false }
            return detail.contains("--stg-start")
        }
    }

    @Test("start > end is refused")
    func invertedWindowIsRefused() {
        var r = request()
        r.recipeID = RecipeRegistry.production.id
        r.stgStartPercent = 0.8
        r.stgEndPercent = 0.2
        #expect {
            try r.validate()
        } throws: { error in
            guard case RecipeFailure.conflictingConditioning(let detail) = error else {
                return false
            }
            return detail.contains("stg-start")
        }
    }

    /// An image plus a video is still a conflict, and it is the *conditioning* conflict that
    /// fires rather than the two-stage one — the same message a single-stage caller gets.
    @Test("--image with --video is the ordinary conditioning conflict")
    func imageWithVideoIsTheOrdinaryConflict() {
        #expect {
            try request(image: Self.png, video: Self.mp4).validate()
        } throws: { error in
            if case RecipeFailure.conflictingConditioning = error { return true }
            return false
        }
    }
}

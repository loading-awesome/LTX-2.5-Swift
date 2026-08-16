// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXRecipes

/// Keyframe parsing and position resolution.
///
/// The load-bearing test is `midChunkFrameIsRefused`. The VAE is causal, so a still placed
/// inside a latent chunk would freeze all eight of that chunk's pixel frames to one image —
/// a render that completes, looks like a stutter, and has no error attached to it.
@Suite("Keyframes")
struct KeyframeTests {

    private func request(_ images: [ImageConditioning], frames: Int = 97) -> RecipeRequest {
        RecipeRequest(prompt: "x", videoOutput: URL(fileURLWithPath: "/tmp/o.mp4"),
                      frames: frames, images: images)
    }

    private func png(_ name: String) -> URL { URL(fileURLWithPath: "/tmp/\(name).png") }

    // MARK: - Parsing

    @Test("PATH@FRAME")
    func parsesFrame() throws {
        let parsed = try ImageConditioning.parse("/tmp/a.png@8")
        #expect(parsed.url.path == "/tmp/a.png")
        #expect(parsed.pixelFrame == 8)
        #expect(parsed.strength == nil)
    }

    @Test("PATH@FRAME:STRENGTH")
    func parsesStrength() throws {
        let parsed = try ImageConditioning.parse("/tmp/a.png@16:0.6")
        #expect(parsed.pixelFrame == 16)
        #expect(parsed.strength == 0.6)
    }

    @Test("'first' and 'last' are spelled out")
    func parsesNamedFrames() throws {
        #expect(try ImageConditioning.parse("/tmp/a.png@first").pixelFrame == 0)
        #expect(try ImageConditioning.parse("/tmp/a.png@last").pixelFrame
            == RecipeRequest.lastFramePlaceholder)
        #expect(try ImageConditioning.parse("/tmp/a.png@LAST").pixelFrame
            == RecipeRequest.lastFramePlaceholder)
    }

    /// Split on the last `@`, so a path containing one still parses.
    @Test("a path containing @ still parses")
    func pathWithAtSign() throws {
        let parsed = try ImageConditioning.parse("/tmp/user@host/a.png@0")
        #expect(parsed.url.path == "/tmp/user@host/a.png")
        #expect(parsed.pixelFrame == 0)
    }

    @Test("malformed specs are refused with the reason")
    func malformedRefused() {
        for spec in ["/tmp/a.png", "/tmp/a.png@", "@0", "/tmp/a.png@xyz",
                     "/tmp/a.png@-4", "/tmp/a.png@0:2.0", "/tmp/a.png@0:high"] {
            #expect(throws: RecipeFailure.self, "expected '\(spec)' to be refused") {
                try ImageConditioning.parse(spec)
            }
        }
    }

    // MARK: - Position resolution

    @Test("'last' resolves against the render's own duration")
    func lastResolves() throws {
        let images = [ImageConditioning(url: png("end"),
                                        pixelFrame: RecipeRequest.lastFramePlaceholder)]
        #expect(try request(images).resolvedImages(frames: 97)[0].pixelFrame == 96)
        #expect(try request(images, frames: 25).resolvedImages(frames: 25)[0].pixelFrame == 24)
    }

    /// The whole reason positions are validated rather than rounded.
    @Test("a frame inside a latent chunk is refused, not rounded")
    func midChunkFrameIsRefused() {
        for frame in [1, 5, 7, 9, 47] {
            #expect(throws: RecipeFailure.self, "frame \(frame) should be refused") {
                try request([ImageConditioning(url: png("m"), pixelFrame: frame)])
                    .resolvedImages(frames: 97)
            }
        }
    }

    @Test("latent boundaries and the closing frame are accepted")
    func boundariesAccepted() throws {
        for frame in [0, 8, 16, 48, 88, 96] {
            let resolved = try request([ImageConditioning(url: png("k"), pixelFrame: frame)])
                .resolvedImages(frames: 97)
            #expect(resolved[0].pixelFrame == frame)
        }
    }

    @Test("a frame past the end is refused")
    func pastTheEndRefused() {
        #expect(throws: RecipeFailure.self) {
            try request([ImageConditioning(url: png("k"), pixelFrame: 104)])
                .resolvedImages(frames: 97)
        }
    }

    @Test("two stills on one frame are refused rather than silently overwriting")
    func duplicateRefused() {
        #expect(throws: RecipeFailure.self) {
            try request([ImageConditioning(url: png("a"), pixelFrame: 8),
                         ImageConditioning(url: png("b"), pixelFrame: 8)])
                .resolvedImages(frames: 97)
        }
        // 'last' colliding with its own explicit index is the same mistake, spelled twice.
        #expect(throws: RecipeFailure.self) {
            try request([ImageConditioning(url: png("a"), pixelFrame: 96),
                         ImageConditioning(url: png("b"),
                                           pixelFrame: RecipeRequest.lastFramePlaceholder)])
                .resolvedImages(frames: 97)
        }
    }

    @Test("stills are returned in time order whatever order they arrived in")
    func sortedByFrame() throws {
        let resolved = try request([
            ImageConditioning(url: png("c"), pixelFrame: RecipeRequest.lastFramePlaceholder),
            ImageConditioning(url: png("a"), pixelFrame: 0),
            ImageConditioning(url: png("b"), pixelFrame: 48),
        ]).resolvedImages(frames: 97)
        #expect(resolved.map(\.pixelFrame) == [0, 48, 96])
    }

    @Test("a per-keyframe strength outside 0...1 is refused")
    func strengthRangeEnforced() {
        #expect(throws: RecipeFailure.self) {
            try request([ImageConditioning(url: png("a"), pixelFrame: 0, strength: 1.5)])
                .resolvedImages(frames: 97)
        }
    }

    // MARK: - Request integration

    @Test("the image: shorthand lands at frame 0 alongside an explicit last frame")
    func shorthandComposesWithKeyframes() throws {
        let request = RecipeRequest(
            prompt: "x", videoOutput: URL(fileURLWithPath: "/tmp/o.mp4"), frames: 97,
            image: png("start"),
            images: [ImageConditioning(url: png("end"),
                                       pixelFrame: RecipeRequest.lastFramePlaceholder)])
        let resolved = try request.resolvedImages(frames: 97)
        #expect(resolved.map(\.pixelFrame) == [0, 96])
        #expect(resolved[0].url.lastPathComponent == "start.png")
        #expect(resolved[1].url.lastPathComponent == "end.png")
    }

    @Test("validate() catches a bad keyframe before any checkpoint opens")
    func validateCatchesPositions() {
        #expect(throws: RecipeFailure.self) {
            try request([ImageConditioning(url: png("a"), pixelFrame: 5)]).validate()
        }
    }

    @Test("stills still conflict with a conditioning video")
    func imageVideoConflictSurvives() {
        var request = self.request([ImageConditioning(url: png("a"), pixelFrame: 0)])
        request.video = png("v")
        #expect(throws: RecipeFailure.self) { try request.validate() }
    }
}

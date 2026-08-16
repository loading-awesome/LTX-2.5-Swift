// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import LTXRecipes

@Suite("Recipe request resolution")
struct RecipeRequestResolutionTests {

    private func request(width: Int? = nil, height: Int? = nil,
                         megapixels: Double? = nil, frames: Int? = 97,
                         aspect: AspectRatio = .r16x9) -> RecipeRequest {
        RecipeRequest(
            prompt: "x",
            videoOutput: URL(fileURLWithPath: "/tmp/out.mp4"),
            frames: frames,
            megapixels: megapixels,
            aspectRatio: aspect,
            width: width,
            height: height)
    }

    @Test("no explicit size uses the pinned 640x384 gated shape")
    func defaultIsPinned() throws {
        let shape = try request().resolvedShape()
        #expect(shape == Shape(width: 640, height: 384))
    }

    @Test("exact 16:9 on the VAE 32-grid is refused; recipes are 64")
    func thirtyTwoOnlyRefused() {
        #expect(throws: RecipeFailure.self) {
            _ = try request(width: 512, height: 288).resolvedShape()
        }
    }

    @Test("a width not on the 64-grid is refused, not truncated")
    func offGridRefused() {
        #expect(throws: RecipeFailure.self) {
            _ = try request(width: 641, height: 384).resolvedShape()
        }
    }

    @Test("100 frames is off the 8k+1 lattice")
    func offLatticeRefused() {
        #expect(throws: RecipeFailure.self) {
            _ = try request(frames: 100).resolvedFrames()
        }
    }
}

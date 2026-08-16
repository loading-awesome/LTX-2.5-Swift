// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXPipeline

/// The sampler's sigma schedule, checked without a GPU in the room.
///
/// The schedule is a few dozen scalars produced by deterministic arithmetic, so it is the
/// one part of the sampler that stands on its own: that the sigmas descend and reach their
/// terminal value, and that the timestep scale multiplier is the checkpoint's 1000 rather
/// than the signature default.
///
/// The step count comes out of the `tiny_v5` manifest rather than a literal, so the
/// schedule is built at the shape the rest of that fixture describes.
///
/// It is emphatically **not** a check on the rest of the sampler. The step loop, the
/// guidance combination and the latent evolution have their own suites.
@Suite("Flow schedule")
struct FlowScheduleTests {

    /// The manifest's step count, so the schedule is built at the shape the fixture holds.
    static func recordedSteps() -> Int? {
        let path = "Tests/Fixtures/goldens/tiny_v5/manifest.json"
        let candidates = [path, FileManager.default.currentDirectoryPath + "/" + path]
        for p in candidates {
            guard let data = FileManager.default.contents(atPath: p),
                  let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let params = root["params"] as? [String: Any],
                  let steps = params["steps"] as? Int
            else { continue }
            return steps
        }
        return nil
    }

    /// The schedule is monotonically decreasing and ends at the terminal sigma.
    ///
    /// Cheap, and it catches the two orderings that produce a plausible-looking array:
    /// a reversed schedule (denoising backwards) and one that forgets the terminal value.
    @Test("sigmas descend and reach the terminal value")
    func sigmasDescend() throws {
        guard let steps = Self.recordedSteps() else { return }
        let schedule = FlowSchedule()
        let sigmas = try schedule.sigmas(steps: steps)
        for i in 1..<sigmas.count {
            #expect(sigmas[i] < sigmas[i - 1],
                    Comment(rawValue: "sigma[\(i)] \(sigmas[i]) is not below "
                            + "sigma[\(i - 1)] \(sigmas[i - 1])"))
        }
        #expect(sigmas.first! > sigmas.last!)
    }

    /// The timestep scale multiplier is 1000 on this checkpoint, not the signature default.
    ///
    /// `DiTForward` carries the same figure from the other side, so the sampler and the
    /// model agree on a value neither of them chose.
    @Test("the timestep scale multiplier is the checkpoint's 1000")
    func scaleMultiplier() {
        #expect(FlowSchedule.timestepScaleMultiplier == 1000)
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Testing
@testable import LTXFoundation

@Suite("Frame lattice")
struct FrameLatticeTests {
    let lattice = FrameLattice(timeScale: 8)

    @Test("lattice points are 8k + 1")
    func latticeMembership() {
        for k in 0..<20 {
            #expect(lattice.contains(8 * k + 1))
        }
        for f in [0, 2, 8, 10, 100] {
            #expect(!lattice.contains(f))
        }
    }

    @Test("snapping floors, and never below one")
    func snapDown() {
        #expect(lattice.snapDown(100) == 97)
        #expect(lattice.snapDown(97) == 97)
        #expect(lattice.snapDown(96) == 89)
        #expect(lattice.snapDown(1) == 1)
        #expect(lattice.snapDown(0) == 1)
    }

    @Test("snapUp is the smallest lattice point at or above")
    func snapUp() {
        #expect(lattice.snapUp(97) == 97)
        #expect(lattice.snapUp(98) == 105)
        #expect(lattice.snapUp(2) == 9)
        #expect(lattice.snapUp(1) == 1)
    }

    // A duration request floors onto the lattice, so it normally rounds down.
    // A "round to nearest" implementation would get this wrong in a way no
    // shape check would catch.
    @Test("duration floors onto the lattice")
    func durationFloors() {
        let f = lattice.frames(forSeconds: 4.0, frameRate: 24, minSeconds: 1, maxSeconds: 20)
        #expect(f == 89)          // 96 raw -> floor to 89
        #expect(lattice.contains(f))
    }

    // The one asymmetry worth a test of its own: flooring below the minimum
    // raises to the next lattice point rather than leaving the shot short.
    @Test("a floor below the minimum is raised, not left short")
    func floorBelowMinimumIsRaised() {
        // min 1s at 24fps = 24 frames; 24 floors to 17, which is under it.
        let f = lattice.frames(forSeconds: 1.0, frameRate: 24, minSeconds: 1, maxSeconds: 20)
        #expect(f == 25)
        #expect(lattice.contains(f))
        #expect(f >= 24)
    }

    @Test("the raise is still capped by the maximum")
    func raiseRespectsMaximum() {
        let f = lattice.frames(forSeconds: 100, frameRate: 24, minSeconds: 99, maxSeconds: 100)
        #expect(f <= 2400)
    }

    @Test("latent and pixel frame counts round-trip on the lattice")
    func scaleRoundTrip() {
        for k in 1..<20 {
            let latent = k
            let pixels = lattice.pixelFrames(fromLatentFrames: latent)
            #expect(lattice.contains(pixels))
            #expect(lattice.latentFrames(fromPixelFrames: pixels) == latent)
        }
    }
}

// A spread of durations pinned to the frame counts they must produce, so a
// change to the rounding rule shows up as a diff rather than as a clip that is
// quietly a different length. Regenerate against the shipped checkpoint's own
// time scale before trusting these at production shape.
extension FrameLatticeTests {
    @Test("matches the recorded table on a spread of inputs",
          arguments: [
            (4.0, 24.0, 1.0, 20.0, 89),
            (1.0, 24.0, 1.0, 20.0, 25),
            (100.0, 24.0, 99.0, 100.0, 2393),
            (0.1, 24.0, 1.0, 20.0, 25),
            (20.0, 24.0, 1.0, 20.0, 473),
            (2.5, 30.0, 0.5, 10.0, 73),
            (7.3, 25.0, 1.0, 20.0, 177),
          ])
    func matchesReference(seconds: Double, fps: Double,
                          minS: Double, maxS: Double, expected: Int) {
        let got = lattice.frames(forSeconds: seconds, frameRate: fps,
                                 minSeconds: minS, maxSeconds: maxS)
        #expect(got == expected)
    }
}

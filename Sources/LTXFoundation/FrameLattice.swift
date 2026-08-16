// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The frame-count lattice imposed by the VAE's causal temporal grid.
///
/// LTX's video VAE compresses time causally: one leading frame, then groups of
/// `timeScale`. So a renderable frame count is always `timeScale * k + 1`, and
/// the same `8` appears in the decoder's own scale contract — `upscale_ratio`
/// maps `a` latent frames to `a * 8 - 7` pixel frames.
///
/// This is arithmetic, not a model, which is why it lives in the MLX-free
/// layer and is tested without a GPU or a checkpoint. It is also exactly the
/// kind of rule that produces a confusing shape error three stages later when
/// it is applied loosely, so it is applied in one place.
///
/// Confirm `timeScale` against the checkpoint's VAE config before trusting this
/// at production shape — see `docs/FRAGILE_CONTRACTS.md` contract 4.
public struct FrameLattice: Sendable, Equatable {
    /// The temporal compression factor. 8 for LTX 2.3 and 2.5; verify per
    /// checkpoint.
    public let timeScale: Int

    public init(timeScale: Int = 8) {
        precondition(timeScale >= 1, "timeScale must be positive")
        self.timeScale = timeScale
    }

    /// Whether `frames` sits exactly on the lattice.
    public func contains(_ frames: Int) -> Bool {
        frames >= 1 && (frames - 1) % timeScale == 0
    }

    /// The largest lattice point at or below `frames`.
    ///
    /// Snapping is downward. A caller asking for 100 frames gets 97, not 105:
    /// rounding up would silently lengthen a shot.
    public func snapDown(_ frames: Int) -> Int {
        guard frames > 1 else { return 1 }
        return (frames - 1) / timeScale * timeScale + 1
    }

    /// The smallest lattice point at or above `frames`.
    public func snapUp(_ frames: Int) -> Int {
        guard frames > 1 else { return 1 }
        let k = (frames - 2) / timeScale + 1
        return k * timeScale + 1
    }

    /// Frame count for a duration, clamped then snapped.
    ///
    /// One asymmetry is deliberate: the result is floored onto the lattice, but
    /// a floor that falls below `minSeconds`
    /// is raised to the next lattice point rather than left short — and that
    /// raise is itself capped by `maxSeconds`. A clamp applied after snapping
    /// instead of before produces different frame counts near the bounds.
    public func frames(forSeconds seconds: Double,
                       frameRate: Double,
                       minSeconds: Double,
                       maxSeconds: Double) -> Int {
        let minFrames = max(1, Int((minSeconds * frameRate).rounded()))
        let maxFrames = Int((maxSeconds * frameRate).rounded())
        let raw = max(minFrames, min(Int((seconds * frameRate).rounded()), maxFrames))

        let snapped = snapDown(raw)
        guard snapped < minFrames else { return snapped }
        return min(snapUp(minFrames), maxFrames)
    }

    /// Pixel frames produced by decoding `latentFrames`: `a * scale - (scale - 1)`.
    public func pixelFrames(fromLatentFrames latentFrames: Int) -> Int {
        max(0, latentFrames * timeScale - (timeScale - 1))
    }

    /// Latent frames needed to encode `pixelFrames`.
    public func latentFrames(fromPixelFrames pixelFrames: Int) -> Int {
        max(0, Int((Double(pixelFrames) + Double(timeScale - 1)) / Double(timeScale)))
    }
}

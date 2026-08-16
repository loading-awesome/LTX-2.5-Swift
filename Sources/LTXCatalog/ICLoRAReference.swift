// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// How an IC-LoRA's reference clip relates to the render it conditions.
///
/// An in-context adapter carries its reference **in the same position space** as the
/// generated tokens, so the reference's patch coordinates have to be scaled back up into
/// the target's frame before they are used. The scale is a property of the adapter, and
/// every released adapter states it in its own `__metadata__`.
///
/// ## Why this is read and not assumed
///
/// It used to be the literal `2` in `VideoUpscaler`, written when the x2 pixel spatial
/// upscaler was the only adapter this port could run. Measured across the sixteen adapters
/// on disk, that constant is right for **three** of them:
///
/// ```
/// factor 1   ingredients, in-outpainting, colorization, deblur, decompression,
///            day-to-night, instant-shave, cross-eyed, water-simulation, lipdub
/// factor 2   pixel-spatial-upscaler-x2, union-control, motion-track
/// factor 4   pixel-spatial-upscaler-x4
/// ```
///
/// Using 2 for a factor-1 adapter is the silent failure `DiTForward.Geometry.Reference`
/// describes: the reference lands on a quarter of the target's pixel range, every token is
/// well-formed, and the model is shown a crop it was never trained to see. No error, and
/// output that looks like a bad render rather than a wrong one.
public struct ICLoRAReference: Sendable, Equatable {

    /// Target-over-reference spatial ratio. 1 means the reference is the same size as the
    /// render — the common case for editing adapters, which transform a clip in place.
    public let downscaleFactor: Int
    /// Target-over-reference temporal ratio. No released adapter declares this yet; 1 is the
    /// identity and the value every measured file behaves as.
    public let temporalScaleFactor: Int

    /// Whether the adapter actually declared a reference factor.
    ///
    /// The distinction between a **standard LoRA** and an **IC-LoRA**: a standard adapter
    /// (the distilled rank-450 one, or anything trained locally) has no reference clip at
    /// all and declares nothing, while an in-context adapter always states its factor.
    /// Reporting "ref x1" for a standard LoRA would invent a reference it does not have.
    public let isDeclared: Bool

    public init(downscaleFactor: Int, temporalScaleFactor: Int = 1, isDeclared: Bool = true) {
        self.downscaleFactor = downscaleFactor
        self.temporalScaleFactor = temporalScaleFactor
        self.isDeclared = isDeclared
    }

    public static let metadataKey = "reference_downscale_factor"
    public static let temporalMetadataKey = "reference_temporal_scale_factor"

    public enum Failure: Error, CustomStringConvertible {
        case unreadableFactor(key: String, value: String, path: String)

        public var description: String {
            switch self {
            case let .unreadableFactor(key, value, path):
                return "\(path) declares \(key) = '\(value)', which is not a positive integer. "
                    + "Refusing rather than substituting a default: the factor decides where "
                    + "the reference lands in the target's frame, and a wrong one produces a "
                    + "well-formed render of the wrong crop"
            }
        }
    }

    /// Read the factors an adapter declares.
    ///
    /// A **missing** key is the identity, 1 — which is what an adapter whose reference is
    /// the same size as its output would mean by saying nothing. A key that is present but
    /// unparseable is refused, because that is a file saying something this code failed to
    /// understand rather than a file saying nothing.
    public static func read(_ header: SafetensorsHeader,
                            path: String = "the adapter") throws -> ICLoRAReference {
        func factor(_ key: String) throws -> Int {
            guard let raw = header.metadata[key] else { return 1 }
            guard let value = Int(raw.trimmingCharacters(in: .whitespaces)), value >= 1 else {
                throw Failure.unreadableFactor(key: key, value: raw, path: path)
            }
            return value
        }
        return ICLoRAReference(downscaleFactor: try factor(metadataKey),
                               temporalScaleFactor: try factor(temporalMetadataKey),
                               isDeclared: header.metadata[metadataKey] != nil)
    }

    public static func read(contentsOf url: URL) throws -> ICLoRAReference {
        try read(try SafetensorsHeader.read(from: url), path: url.lastPathComponent)
    }
}

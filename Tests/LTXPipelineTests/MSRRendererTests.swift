// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import CoreGraphics
import Foundation
import ImageIO
import LTXFoundation
import LTXModules
import MLX
import Testing
import LTXRecipes
import UniformTypeIdentifiers
@testable import LTXPipeline

/// The parts of the MSR path that can be checked without the 42 GB transformer.
///
/// Which is most of what can go wrong. The reference preparation decides what the model is
/// shown, the slot assignment decides whether it can tell the cast apart, and the guards
/// decide whether a malformed request costs a message or twenty minutes — and none of the
/// three has a shape that disagrees when it is wrong.
@Suite("MSR renderer", .serialized)
struct MSRRendererTests {

    static let adapter =
        LTXConfiguration.resolved.checkpoints.root!
        + "/ic-lora/multiple-subject-reference/"
        + "LTX-2.5-Licon-MSR-V1.safetensors"

    /// A solid-colour PNG of a chosen size, written to a temporary file.
    static func image(width: Int, height: Int,
                      red: UInt8 = 20, green: UInt8 = 60, blue: UInt8 = 200) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("msr-\(width)x\(height)-\(UUID().uuidString).png")
        let info = CGImageAlphaInfo.noneSkipLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue
        let context = CGContext(data: nil, width: width, height: height,
                                bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: info)!
        context.setFillColor(red: Double(red) / 255, green: Double(green) / 255,
                             blue: Double(blue) / 255, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let cgImage = context.makeImage()!
        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, cgImage, nil)
        #expect(CGImageDestinationFinalize(destination))
        return url
    }

    /// A subject whose aspect ratio disagrees with the frame is **padded, not cropped**.
    ///
    /// This is the branch an implementation is likely to skip, because every other
    /// conditioning image in this codebase goes through aspect-fill-then-crop and the shape
    /// is identical either way. A 3:4 portrait centre-cropped into a 16:9 frame keeps a
    /// horizontal band of the source — for a character reference, the model is then
    /// conditioned on a subject it can only partly see, and there is nothing in the output
    /// to say so.
    ///
    /// The white fill is checked at the corners: pad and crop differ there and nowhere else
    /// on a flat source.
    @Test("a portrait subject in a landscape frame is padded with white, not cropped")
    func portraitSubjectIsPadded() throws {
        let url = try Self.image(width: 300, height: 400)
        let out = try MSRRenderer.prepared(image: url, width: 384, height: 128,
                                           isBackground: false, frames: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(out.shape == [1, 128, 384, 3])
        MLX.eval(out)

        // Corners are canvas: white, i.e. 1.0 in the [0, 1] range `MediaInput.image` returns.
        for x in [0, 383] {
            let corner = out[0, 0, x, 0...].asArray(Float.self)
            #expect(corner.allSatisfy { $0 > 0.95 },
                    "corner x=\(x) is \(corner), not white padding" as Comment)
        }
        // The centre is the source colour, so the image did land on the canvas.
        let centre = out[0, 64, 192, 0...].asArray(Float.self)
        #expect(centre[2] > centre[0], "centre \(centre) is not the blue source" as Comment)
    }

    /// A background is centre-cropped even when a subject at the same size would be padded.
    ///
    /// The two branches have the same output shape, so this is the only place the
    /// distinction is visible. Scenery filling the frame is the point of a plate; padding
    /// one would put white bars into the model's idea of the location.
    @Test("a background fills the frame regardless of aspect ratio")
    func backgroundIsCropped() throws {
        let url = try Self.image(width: 300, height: 400)
        let out = try MSRRenderer.prepared(image: url, width: 384, height: 128,
                                           isBackground: true, frames: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(out.shape == [1, 128, 384, 3])
        MLX.eval(out)
        for x in [0, 383] {
            let corner = out[0, 0, x, 0...].asArray(Float.self)
            #expect(corner[2] > corner[0],
                    "corner x=\(x) is \(corner) — a background should be filled, not padded"
                        as Comment)
        }
    }

    /// A same-family source that is not smaller is filled rather than padded.
    @Test("a landscape subject at or above the target size is centre-crop-filled")
    func matchingFamilyIsFilled() throws {
        let url = try Self.image(width: 800, height: 400)
        let out = try MSRRenderer.prepared(image: url, width: 384, height: 128,
                                           isBackground: false, frames: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(out.shape == [1, 128, 384, 3])
        MLX.eval(out)
        let corner = out[0, 0, 0, 0...].asArray(Float.self)
        #expect(corner[2] > corner[0], "corner \(corner) was padded, not filled" as Comment)
    }

    /// The still is repeated, not held for one frame.
    ///
    /// A single frame encodes to one latent frame. The slot offsets were chosen against a
    /// reference that occupies five, so a held still would sit in a fifth of the temporal
    /// extent its offset assumes — a difference no shape check sees, because the geometry is
    /// built from whatever the encode returned.
    @Test("a still is repeated onto the reference frame count")
    func stillIsRepeated() throws {
        let url = try Self.image(width: 128, height: 128)
        defer { try? FileManager.default.removeItem(at: url) }
        for count in [25, 33] {
            let out = try MSRRenderer.prepared(image: url, width: 128, height: 128,
                                               isBackground: false, frames: count)
            #expect(out.dim(0) == count)
        }
    }

    /// The background takes the **last** slot, and the offsets run pic1-furthest-back.
    @Test("slot order puts the background last and pic1 at the deepest offset")
    func slotOrder() throws {
        let options = try MSRRenderer.Options(
            references: [URL(fileURLWithPath: "/a.png"), URL(fileURLWithPath: "/b.png")],
            background: URL(fileURLWithPath: "/bg.png"),
            prompt: "", output: URL(fileURLWithPath: "/out.mp4"),
            plan: try RecipePlan.msr())
        let slots = options.slots
        #expect(slots.count == 3)
        #expect(slots.map(\.isBackground) == [false, false, true])
        #expect(slots[2].url.lastPathComponent == "bg.png")

        // The offsets these slots produce, as `render` computes them.
        let offsets = (0 ..< slots.count).map { -(slots.count - $0) }
        #expect(offsets == [-3, -2, -1])
    }

    /// Malformed requests must cost a message rather than a load.
    @Test("no references and too many references are both refused")
    func slotCountIsGuarded() throws {
        let checkpoints = VideoUpscaler.Checkpoints(
            videoVAE: URL(fileURLWithPath: "/nope/vae.safetensors"),
            upsampler: URL(fileURLWithPath: "/nope/up.safetensors"))
        func options(_ count: Int) throws -> MSRRenderer.Options {
            try MSRRenderer.Options(
                references: (0 ..< count).map { URL(fileURLWithPath: "/r\($0).png") },
                prompt: "", output: URL(fileURLWithPath: "/out.mp4"),
                plan: try RecipePlan.msr())
        }
        #expect(throws: MSRRenderer.Failure.self) {
            try MSRRenderer.render(options(0), checkpoints: checkpoints)
        }
        #expect(throws: MSRRenderer.Failure.self) {
            try MSRRenderer.render(options(6), checkpoints: checkpoints)
        }
    }

    /// The adapter must come apart into exactly the two pieces, losing nothing.
    ///
    /// `LoRAOverlay` throws on any tensor it cannot classify — deliberately, so a DoRA
    /// magnitude cannot be dropped in silence. The five slot tensors are exactly that case,
    /// so they are lifted out first. A split that took too much would quietly shrink the
    /// adapter; one that took too little would throw at load with the transformer already
    /// in memory.
    @Test("an MSR adapter splits into 960 LoRA tensors and 5 slot tensors")
    func adapterSplits() throws {
        guard FileManager.default.fileExists(atPath: Self.adapter) else {
            print("SKIP MSR split: adapter absent at \(Self.adapter)")
            return
        }
        let all = try MLX.loadArrays(url: URL(fileURLWithPath: Self.adapter))
        let (lora, slots) = ReferenceSlotEmbedding.split(all)
        #expect(lora.count + slots.count == all.count)
        #expect(slots.count == 5)
        #expect(lora.count == 960)
        #expect(!ReferenceSlotEmbedding.isPresent(in: lora))
        #expect(ReferenceSlotEmbedding.isPresent(in: slots))

        // The LoRA half loads through the overlay that would have refused the whole file.
        let overlay = try LoRAOverlay.compose(name: "msr-split", tensors: lora)
        #expect(overlay.reports.first?.moduleCount == 480)
        #expect(throws: LoRAOverlay.Failure.self) {
            try LoRAOverlay.compose(name: "msr-whole", tensors: all)
        }
    }
}

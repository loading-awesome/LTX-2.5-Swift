// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing
@testable import LTXCatalog
@testable import LTXFoundation

/// The factor that decides where an IC-LoRA's reference lands in the target's frame.
///
/// Read rather than assumed, because a wrong factor produces a well-formed render of the
/// wrong crop — no error, and output that reads as a bad generation instead of a bug.
@Suite("IC-LoRA reference")
struct ICLoRAReferenceTests {

    private static let root = LTXConfiguration.resolved.checkpoints.root!
    private static let modelsRoot = LTXConfiguration.resolvedModelsRoot

    private func header(_ metadata: [String: String]) throws -> SafetensorsHeader {
        let pairs = metadata.map { "\"\($0.key)\":\"\($0.value)\"" }.joined(separator: ",")
        let json = "{\"__metadata__\":{\(pairs)},"
            + "\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}"
        return try SafetensorsHeader.parse(json: Data(json.utf8), dataStart: 0, dataSize: 4)
    }

    @Test("the declared factor is used")
    func readsDeclared() throws {
        #expect(try ICLoRAReference.read(header(["reference_downscale_factor": "2"]))
            .downscaleFactor == 2)
        #expect(try ICLoRAReference.read(header(["reference_downscale_factor": "4"]))
            .downscaleFactor == 4)
        #expect(try ICLoRAReference.read(header(["reference_downscale_factor": "1"]))
            .downscaleFactor == 1)
    }

    /// An adapter whose reference is the same size as its output says so by saying nothing.
    @Test("an absent key is the identity, not the old hardcoded 2")
    func absentIsIdentity() throws {
        let reference = try ICLoRAReference.read(header([:]))
        #expect(reference.downscaleFactor == 1)
        #expect(reference.temporalScaleFactor == 1)
    }

    /// A standard LoRA has no reference clip at all. Reporting it as "ref x1" would invent
    /// one, which is the difference between the two adapter kinds.
    @Test("declaring nothing marks a standard LoRA, not an x1 IC-LoRA")
    func standardLoRAIsDistinguishable() throws {
        #expect(try ICLoRAReference.read(header([:])).isDeclared == false)
        #expect(try ICLoRAReference.read(header(["reference_downscale_factor": "1"]))
            .isDeclared == true)
    }

    @Test("the temporal factor is read independently and defaults to 1")
    func temporalFactor() throws {
        let both = try ICLoRAReference.read(header(["reference_downscale_factor": "2",
                                                    "reference_temporal_scale_factor": "3"]))
        #expect(both.downscaleFactor == 2)
        #expect(both.temporalScaleFactor == 3)
        #expect(try ICLoRAReference.read(header(["reference_downscale_factor": "2"]))
            .temporalScaleFactor == 1)
    }

    /// Present-but-unparseable is a file saying something this code failed to understand,
    /// which is a different thing from a file saying nothing.
    @Test("a malformed or non-positive factor is refused, not defaulted")
    func malformedRefused() {
        for bad in ["0", "-2", "two", "1.5", ""] {
            #expect(throws: ICLoRAReference.Failure.self, "expected '\(bad)' to be refused") {
                try ICLoRAReference.read(header(["reference_downscale_factor": bad]))
            }
        }
    }

    // MARK: - Against the real files

    /// The measurement that motivated reading rather than assuming. Skips where the tree is
    /// absent, in the suite's usual style.
    @Test("the adapters on disk disagree about the factor")
    func realAdaptersDisagree() throws {
        let cases: [(String, Int)] = [
            (Self.root + "/ic-lora/pixel-spatial-upscaler/"
                + "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors", 2),
            (Self.modelsRoot + "/ic-lora-2.3/creative-lab/ingredients/"
                + "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", 1),
        ]
        var checked = 0
        for (path, expected) in cases {
            guard FileManager.default.fileExists(atPath: path) else {
                print("SKIP ICLoRAReference: absent at \(path)")
                continue
            }
            let reference = try ICLoRAReference.read(
                contentsOf: URL(fileURLWithPath: path))
            #expect(reference.downscaleFactor == expected, "\(path)")
            checked += 1
        }
        if checked == 0 { print("SKIP ICLoRAReference: no adapters on disk") }
    }
}

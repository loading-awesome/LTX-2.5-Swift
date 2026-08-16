// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import Testing
@testable import LTXModules

/// The MSR slot embedding, pinned value by value against a fixture.
///
/// Pinned rather than eyeballed because every part of this is silent when wrong. A slot vector
/// that is off by one index, or scaled by the wrong divisor, still has a plausible norm and
/// still separates the references *somewhat* — it just separates them differently than the
/// weights were trained to, and the failure shows up as references bleeding into each other
/// in a render, which reads as "the adapter isn't very good".
@Suite("MSR slot embedding", .serialized)
struct ReferenceSlotEmbeddingTests {

    static let adapter =
        LTXConfiguration.resolved.checkpoints.root!
        + "/ic-lora/multiple-subject-reference/"
        + "LTX-2.5-Licon-MSR-V1.safetensors"

    struct Row: Decodable {
        let norm: Float
        let first8: [Float]
    }

    static func reference() throws -> [String: Row]? {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Tests/Fixtures/reference/msr_slot_embedding.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("SKIP MSR slot embedding: \(url.path) absent")
            return nil
        }
        return try JSONDecoder().decode([String: Row].self, from: Data(contentsOf: url))
    }

    static func weights() throws -> [String: MLXArray]? {
        guard FileManager.default.fileExists(atPath: adapter) else {
            print("SKIP MSR slot embedding: adapter absent at \(adapter)")
            return nil
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: adapter))
    }

    @Test("each slot's embedding matches the recorded table")
    func slotsMatch() throws {
        guard let table = try Self.reference(), let weights = try Self.weights() else { return }
        let embedding = try ReferenceSlotEmbedding.load(from: weights)
        #expect(embedding.latentChannels == 128)

        for index in 0 ..< 5 {
            guard let want = table[String(index)] else { continue }
            let got = embedding(slotIndex: index)
            MLX.eval(got)
            let values = got.asArray(Float.self)

            let norm = MLX.sqrt((got * got).sum()).item(Float.self)
            #expect(abs(norm - want.norm) <= 1e-4 * max(want.norm, 1),
                    "slot \(index) norm \(norm) vs \(want.norm)" as Comment)
            // Not just the norm — a sign flip or a permuted layer leaves it unchanged.
            for (i, expected) in want.first8.enumerated() {
                #expect(abs(values[i] - expected) <= 2e-5,
                        "slot \(index)[\(i)] \(values[i]) vs \(expected)" as Comment)
            }
        }
    }

    /// The slot index is 1-based inside the formula. Passing 0-based straight through shifts
    /// every reference by one slot, which nothing else would report.
    @Test("slots are distinct, and slot 0 is not the zero vector")
    func slotsAreDistinct() throws {
        guard let weights = try Self.weights() else { return }
        let embedding = try ReferenceSlotEmbedding.load(from: weights)

        var seen: [[Float]] = []
        for index in 0 ..< 5 {
            let value = embedding(slotIndex: index)
            MLX.eval(value)
            seen.append(value.asArray(Float.self))
        }
        // A zero first slot would mean the `+ 1` was dropped and slot 1 fell on scaled = 0.
        let firstNorm = seen[0].reduce(0) { $0 + $1 * $1 }
        #expect(firstNorm > 1e-3, "slot 0 is ~zero, which means the 1-based index was lost")

        for a in 0 ..< seen.count {
            for b in (a + 1) ..< seen.count {
                let distance = zip(seen[a], seen[b]).map { abs($0 - $1) }.max() ?? 0
                #expect(distance > 1e-3, "slots \(a) and \(b) are indistinguishable" as Comment)
            }
        }
    }

    /// A non-MSR adapter must be refused rather than silently loaded without slots.
    @Test("an adapter without slot tensors is refused by name")
    func nonMSRIsRefused() throws {
        let ingredients =
            LTXConfiguration.resolvedModelsRoot + "/ic-lora-2.3/creative-lab/ingredients/"
            + "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
        guard FileManager.default.fileExists(atPath: ingredients) else { return }
        let weights = try MLX.loadArrays(url: URL(fileURLWithPath: ingredients))
        #expect(!ReferenceSlotEmbedding.isPresent(in: weights))
        #expect(throws: ReferenceSlotEmbedding.Failure.self) {
            try ReferenceSlotEmbedding.load(from: weights)
        }
    }

    @Test("the embedding is added on the channel axis, broadcast over frames and space")
    func appliedBroadcasts() throws {
        guard let weights = try Self.weights() else { return }
        let embedding = try ReferenceSlotEmbedding.load(from: weights)
        let latent = MLXArray.zeros([1, 128, 5, 12, 20]).asType(.bfloat16)
        let out = try embedding.applied(to: latent, slotIndex: 2)
        #expect(out.shape == latent.shape)

        // Every spatial position of a channel carries that channel's value.
        let want = embedding(slotIndex: 2)
        MLX.eval(out, want)
        let expected = want.asArray(Float.self)[7]
        let got = out[0, 7, 3, 5, 9].item(Float.self)
        #expect(abs(got - expected) <= 5e-2, "\(got) vs \(expected)" as Comment)

        // A latent with the wrong channel count is refused, not broadcast into nonsense.
        #expect(throws: ReferenceSlotEmbedding.Failure.self) {
            try embedding.applied(to: MLXArray.zeros([1, 64, 2, 4, 4]), slotIndex: 0)
        }
    }
}

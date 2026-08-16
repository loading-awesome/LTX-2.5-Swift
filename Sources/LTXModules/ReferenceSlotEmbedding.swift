// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// The learned per-reference slot vector a Multiple-Subject-Reference adapter carries.
///
/// MSR conditions on several reference images at once. What keeps "image 1" from smearing
/// into "image 3" is a small learned network — five tensors that ship *inside* the adapter
/// alongside its 960 LoRA weights — turning a slot index into a 128-vector that is added to
/// that reference's latent channels.
///
/// ```
/// slot_id   = index + 1                                   1-based
/// scaled    = slot_id / 16
/// features  = [scaled, sin(scaled * f)x16, cos(scaled * f)x16]     -> 33
/// hidden    = silu(linear(features, net.0))                        -> 256
/// embedding = linear(hidden, net.2)                                -> 128
/// latent   += embedding                                    broadcast over C
/// ```
///
/// Three details are guessable-and-wrong, so none of them is inferred from the shapes: the
/// index is **1-based**, the divisor is **16** and unrelated to the frequency count that
/// happens to also be 16, and the raw `scaled` value is concatenated *ahead* of the sines —
/// which is why `net.0` takes 33 inputs rather than 32.
///
/// ## Why this is its own type
///
/// ``LoRAOverlay`` loads pairs. Handed an MSR checkpoint it would take the 960 it recognises
/// and skip these 5 without comment, producing a render that looks like it used the adapter
/// and has no slot separation at all. Loading them through a named type means the absence is
/// something ``load(from:)`` can refuse.
public struct ReferenceSlotEmbedding {

    /// `[frequencies]` — 16 on the V1 checkpoint.
    public let frequencies: MLXArray
    /// `[hidden, 1 + 2 * frequencies]`.
    public let inputWeight: MLXArray
    public let inputBias: MLXArray
    /// `[latentChannels, hidden]`.
    public let outputWeight: MLXArray
    public let outputBias: MLXArray

    /// The prefix these tensors carry inside an adapter file.
    public static let prefix = "diffusion_model.reference_slot_embedding."
    /// Also accepted without the `diffusion_model.` component: adapters spell it both ways.
    public static let bareProefix = "reference_slot_embedding."

    /// The divisor applied to the slot index before the Fourier features.
    ///
    /// Named because it is **not** the frequency count. The two are both 16 on this
    /// checkpoint, so conflating them works here and breaks on any adapter trained with a
    /// different number of frequencies.
    public static let slotScale: Float = 16

    public enum Failure: Error, CustomStringConvertible {
        case absent(sample: [String])
        case missing(String)
        case channelMismatch(embedding: Int, latent: Int)

        public var description: String {
            switch self {
            case let .absent(sample):
                return "this adapter carries no reference_slot_embedding tensors, so it is "
                    + "not an MSR adapter. Loading it as one would silently drop the slot "
                    + "separation. Sample keys: \(sample)"
            case let .missing(name):
                return "the adapter declares reference slot embeddings but \(name) is absent"
            case let .channelMismatch(embedding, latent):
                return "slot embedding is \(embedding)-dimensional, the latent has "
                    + "\(latent) channels"
            }
        }
    }

    /// Load from an already-read adapter weight map.
    public static func load(from weights: [String: MLXArray]) throws -> ReferenceSlotEmbedding {
        func tensor(_ suffix: String) throws -> MLXArray {
            if let value = weights[prefix + suffix] { return value }
            if let value = weights[bareProefix + suffix] { return value }
            throw Failure.missing(prefix + suffix)
        }
        guard weights.keys.contains(where: {
            $0.hasPrefix(prefix) || $0.hasPrefix(bareProefix)
        }) else {
            throw Failure.absent(sample: Array(weights.keys.sorted().prefix(3)))
        }
        return ReferenceSlotEmbedding(
            frequencies: try tensor("frequencies"),
            inputWeight: try tensor("net.0.weight"),
            inputBias: try tensor("net.0.bias"),
            outputWeight: try tensor("net.2.weight"),
            outputBias: try tensor("net.2.bias"))
    }

    /// Whether a weight map carries these tensors at all.
    public static func isPresent(in weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasPrefix(prefix) || $0.hasPrefix(bareProefix) }
    }

    /// Separate an MSR file into the part ``LoRAOverlay`` can load and the part it cannot.
    ///
    /// `LoRAOverlay.build` classifies every tensor and **throws** on anything that is
    /// neither an `A`/`B` factor nor a recognised prefix, on the stated grounds that a DoRA
    /// magnitude dropped in silence changes what the adapter does. These five tensors are
    /// exactly that case: they are not a factorisation, they are a small MLP. So they are
    /// lifted out before the overlay ever sees the map.
    ///
    /// Splitting rather than teaching the overlay to skip them keeps the overlay's refusal
    /// intact for every *other* unclassifiable tensor.
    public static func split(_ weights: [String: MLXArray])
        -> (lora: [String: MLXArray], slots: [String: MLXArray]) {
        var lora: [String: MLXArray] = [:]
        var slots: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix(prefix) || key.hasPrefix(bareProefix) {
                slots[key] = value
            } else {
                lora[key] = value
            }
        }
        return (lora: lora, slots: slots)
    }

    public var latentChannels: Int { outputWeight.dim(0) }

    /// The embedding for a reference's **slot index**, counted from 0.
    ///
    /// The `+ 1` is applied here rather than asked of the caller: slot ids are 1-based, and a
    /// caller passing an already-1-based index would land every reference one slot off with
    /// nothing to show for it.
    public func callAsFunction(slotIndex: Int) -> MLXArray {
        let scaled = MLXArray(Float(slotIndex + 1) / Self.slotScale)
        let phases = scaled * frequencies.asType(.float32)
        // `scaled` first, then the sines, then the cosines — 1 + 2F inputs.
        let features = MLX.concatenated(
            [scaled.reshaped([1]), MLX.sin(phases), MLX.cos(phases)])

        let hidden = MLX.matmul(inputWeight.asType(.float32), features)
            + inputBias.asType(.float32)
        let activated = hidden * MLX.sigmoid(hidden)   // SiLU
        return MLX.matmul(outputWeight.asType(.float32), activated)
            + outputBias.asType(.float32)
    }

    /// `latent + embedding`, broadcast over `[B, C, F, H, W]`'s channel axis.
    public func applied(to latent: MLXArray, slotIndex: Int) throws -> MLXArray {
        let embedding = callAsFunction(slotIndex: slotIndex)
        guard embedding.dim(0) == latent.dim(1) else {
            throw Failure.channelMismatch(embedding: embedding.dim(0), latent: latent.dim(1))
        }
        return latent + embedding.reshaped([1, -1, 1, 1, 1]).asType(latent.dtype)
    }
}

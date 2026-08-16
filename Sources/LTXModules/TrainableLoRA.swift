// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import LTXCatalog
import MLX
import MLXRandom

/// LoRA factors as **trainable parameters**, for the gradient to reach.
///
/// ``LoRAOverlay`` is the inference form: it loads `A` and `B` from a file, folds `strength`
/// into `B` once, and never differentiates anything. Training needs the opposite properties
/// — the factors must be live `MLXArray`s that `valueAndGrad` can treat as leaves, and the
/// strength must stay out of them so the optimiser sees the factor itself. Hence a separate
/// type rather than a mode on the old one.
///
/// ## The initialisation convention
///
/// Rank 64, alpha 64, targeting `["to_k", "to_q", "to_v", "to_out.0"]`, initialised the
/// standard way:
///
/// ```
/// A : [rank, in]     Kaiming-uniform(a = √5)  ->  U(-1/√in, +1/√in)
/// B : [out, rank]    zeros
/// scaling = alpha / rank                      ->  1.0 at the defaults
/// delta   = (B @ A) * scaling
/// ```
///
/// `B = 0` is what makes the adapter a no-op at step zero, so the first forward is the base
/// model exactly. Kaiming-uniform at `a = √5` collapses to `U(±1/√fan_in)` — the same bound
/// a linear layer's own default initialisation uses — which is worth writing out because
/// "Kaiming" suggests a different number than the one it actually produces here.
///
/// ## The short-gradient-list trap
///
/// A gradient list shorter than the parameter list is the first thing that goes wrong here,
/// and it surfaces as an index-out-of-range rather than as anything naming its cause. An
/// earlier implementation attributed it to `MLX.zeros` returning a shared buffer that
/// `valueAndGrad` then deduplicated, and worked around it by initialising `B` to tiny random
/// values — abandoning the exact-zero init to do so.
///
/// **That is not the cause on this mlx-swift.** Measured directly: two separately created
/// `MLXArray.zeros` of the same shape receive two distinct, correctly different gradients.
/// The real cause is `valueAndGrad`'s `argumentNumbers`, which **defaults to `[0]`** — it
/// differentiates the first array and silently ignores every other one. ``parameters()``
/// must therefore always be paired with `argumentNumbers: Array(params.indices)`.
///
/// Each `B` is still built from its own Swift array. Not as a workaround — it costs nothing,
/// and it keeps `B` exactly zero rather than a near-zero substitute.
public struct TrainableLoRA {

    /// One adapted linear layer's factors.
    public struct Factors {
        /// `[rank, in]`.
        public var a: MLXArray
        /// `[out, rank]`.
        public var b: MLXArray

        public init(a: MLXArray, b: MLXArray) {
            self.a = a
            self.b = b
        }
    }

    /// The defaults, named so a change to them is a diff rather than a surprise.
    public static let defaultRank = 64
    public static let defaultAlpha = 64
    public static let defaultTargetModules = ["to_k", "to_q", "to_v", "to_out.0"]

    /// The suffix set every **released** adapter occupies — the attention projections plus
    /// the feed-forward, which ``defaultTargetModules`` omits. Paired with
    /// ``DiTLoRATargets/Scope/videoStreamOnly`` it reproduces their 480 modules exactly.
    public static let releasedAdapterTargetModules =
        ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

    /// Base weight keys, in a fixed order. The order is the parameter order handed to
    /// `valueAndGrad`, so it must not depend on dictionary iteration.
    public let keys: [String]
    public private(set) var factors: [Factors]
    public let rank: Int
    public let alpha: Int

    /// `alpha / rank` — the LoRA scaling factor. 1.0 at the defaults, which is exactly
    /// why it must be carried explicitly: a hardcoded 1.0 would be right today and wrong for
    /// anyone who changed either number.
    public var scaling: Float { Float(alpha) / Float(rank) }

    /// Fresh factors at the standard initialisation.
    ///
    /// `shapes` maps a base weight key to `(in, out)` for that linear.
    public init(shapes: [(key: String, inFeatures: Int, outFeatures: Int)],
                rank: Int = TrainableLoRA.defaultRank,
                alpha: Int = TrainableLoRA.defaultAlpha) {
        precondition(rank >= 2, "LoraConfig constrains rank to >= 2")
        precondition(alpha >= 1, "LoraConfig constrains alpha to >= 1")
        self.rank = rank
        self.alpha = alpha
        self.keys = shapes.map(\.key)
        self.factors = shapes.map { shape in
            // Kaiming-uniform(a = √5) on a [rank, in] weight reduces to U(±1/√in).
            let bound = 1.0 / Foundation.sqrt(Float(shape.inFeatures))
            let a = MLXRandom.uniform(low: -bound, high: bound,
                                      [rank, shape.inFeatures]).asType(.float32)
            // Zeros with their own storage, and exactly zero — see the note above.
            let b = MLXArray(
                [Float](repeating: 0, count: shape.outFeatures * rank),
                [shape.outFeatures, rank])
            return Factors(a: a, b: b)
        }
    }

    public init(keys: [String], factors: [Factors], rank: Int, alpha: Int) {
        precondition(keys.count == factors.count)
        self.keys = keys
        self.factors = factors
        self.rank = rank
        self.alpha = alpha
    }

    // MARK: - The forward contribution

    /// `((x @ Aᵀ) @ Bᵀ) · scaling` — the delta added to one linear's output.
    ///
    /// Applied on the **input** side rather than by materialising `B @ A` as a full
    /// `[out, in]` weight, for the same reason `LoRAOverlay` does: the rank-decomposed form
    /// costs `2·rank/out` of the base linear instead of a second weight matrix. During
    /// training it also keeps the graph small, since the intermediate is `[..., rank]`.
    public static func delta(input: MLXArray, a: MLXArray, b: MLXArray,
                             scaling: Float) -> MLXArray {
        let projected = MLX.matmul(input, a.transposed())
        return MLX.matmul(projected, b.transposed()) * scaling
    }

    public func delta(input: MLXArray, at index: Int) -> MLXArray {
        Self.delta(input: input, a: factors[index].a, b: factors[index].b, scaling: scaling)
    }

    // MARK: - Autograd plumbing

    /// Every trainable array, flattened `[A₀, B₀, A₁, B₁, …]`.
    ///
    /// Interleaved rather than all-A-then-all-B so a caller reading a gradient list beside
    /// this one cannot silently pair the wrong halves.
    public func parameters() -> [MLXArray] {
        factors.flatMap { [$0.a, $0.b] }
    }

    /// Rebuild from a flat list in the same order ``parameters()`` produces.
    public mutating func update(from flat: [MLXArray]) {
        precondition(flat.count == factors.count * 2,
                     "expected \(factors.count * 2) arrays, got \(flat.count) — a short "
                        + "list almost always means valueAndGrad was called without "
                        + "argumentNumbers, which defaults to [0]")
        for index in factors.indices {
            factors[index] = Factors(a: flat[index * 2], b: flat[index * 2 + 1])
        }
    }

    /// The adapter as `LoRAOverlay` names things on disk, ready for ``TensorFile``.
    ///
    /// The key form is the one `LoRAOverlay.load` resolves, so an adapter trained here loads
    /// back through the inference path without a translation step — which is the only way
    /// the two halves stay honest about each other.
    public func stateDict(prefix: String = "diffusion_model.") -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (index, key) in keys.enumerated() {
            let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            out["\(Self.adapterName(for: base, prefix: prefix)).lora_A.weight"]
                = factors[index].a
            out["\(Self.adapterName(for: base, prefix: prefix)).lora_B.weight"]
                = factors[index].b
        }
        return out
    }

    /// A checkpoint weight's name in the released adapters' convention.
    ///
    /// The two namings differ by one component and it is easy to compose them wrongly. Base
    /// weights in this checkpoint carry the comfy prefix — `model.diffusion_model.attn1…` —
    /// while every released adapter names the same module `diffusion_model.attn1…`. So the
    /// existing prefix has to be **removed** before the adapter's is applied.
    ///
    /// Prepending instead produced `model.diffusion_model.model.diffusion_model.…` once
    /// `LoRAOverlay.load` added the base prefix back. Nothing failed at save time: the file
    /// was well formed, the shapes were right, and the adapter was simply unloadable by the
    /// inference path. It survived because the only test of `stateDict` compared it against
    /// itself; the fix came from asserting the documented contract instead — that a saved
    /// adapter loads through ``LoRAOverlay/load(url:strength:filter:versionPolicy:basePrefix:baseWeightShapes:)``
    /// and resolves to weights the checkpoint actually has.
    static func adapterName(for baseKey: String, prefix: String) -> String {
        var stem = baseKey
        for candidate in [TransformerTopology.comfyPrefix, prefix] where stem.hasPrefix(candidate) {
            stem = String(stem.dropFirst(candidate.count))
            break
        }
        return prefix + stem
    }
}

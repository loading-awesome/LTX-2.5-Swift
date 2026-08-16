// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import MLX

/// LoRA adapters applied as a **residual overlay**, never merged into the weights.
///
/// ## Overlay, not bake
///
/// Two application modes exist and they are not equivalent in cost.
///
/// **Bake** folds `W += strength · B·A` into the weight dictionary. It is the obvious
/// implementation and it is the wrong one here: baking a *single* rank-16 attention-only
/// adapter into a 22 B model costs a permanent 9.7 GB fp32 expansion, measured, because
/// the merged tensor can no longer be the mmap-backed bf16 buffer the
/// checkpoint shipped. It also destroys the base weights, so switching adapters between
/// render stages means re-reading and re-merging 42 GB.
///
/// **Overlay** leaves `W` untouched and keeps the rank-decomposed pair, adding
/// `out += ((x @ Aᵀ) @ Bᵀ)` inside each affected linear. The resident cost is the LoRA
/// file itself — 327 MB for the 2.5 spatial upscaler, ~72 MB for the rank-16 adapter
/// measured above — and the arithmetic cost is `2·rank/out` of the base linear's, about
/// 0.4 % at rank 16 against a 4096-wide projection, 11 % at the distilled adapter's rank 450.
/// A recipe can therefore swap adapters per stage, or stack two, without touching the
/// base weights at all.
///
/// That is why this type exists and why it does not offer a bake path.
///
/// ## What the guards establish
///
/// Every check below is **structural**: key resolution, pair completeness, shapes, ranks,
/// counts, and the linearity of `strength`. **None of it shows that an adapter produces
/// correct output.** A silently wrong *convention* — a transposed factor, an unapplied
/// alpha, the wrong stacking semantics — would pass all of it. The guards are built to
/// make the loud failures loud, not to certify the arithmetic.
///
/// ## The seam
///
/// ```swift
/// if let d = overlay.delta(forWeightKey: prefix + name + ".weight", input: x) {
///     y = y + d.asType(y.dtype)
/// }
/// ```
///
/// One dictionary lookup on a key the caller already has in hand, `nil` when no adapter
/// touches that linear. The lookup key is a **base checkpoint weight key**, in whatever
/// prefix form this repo's checkpoint uses (`TransformerTopology.prefix`), so a call
/// site does not have to know anything about LoRA naming.
///
/// ## Stacking
///
/// Multiple adapters compose by summation, and every delta is computed from the *base*
/// input `x` — no adapter sees another's output. So
///
/// ```
/// y = Wx + s₁·B₁A₁x + s₂·B₂A₂x
/// ```
///
/// which is **order-independent in exact arithmetic**, exactly as baking both would be.
/// Order survives only in floating-point accumulation: the deltas are summed in the
/// order the adapters were requested, deterministically, so two runs with the same
/// request list agree bit for bit and a reordered list may differ in the last bf16 ulp.
/// This is worth stating because the *other* plausible spelling — wrapping a
/// LoRA-modified linear in a second LoRA — is not order-independent, and is not what
/// happens here.
///
/// ## Statistics
///
/// This type computes no reductions. That is deliberate: contract 16 records that a
/// reduction over a bf16 tensor returns bf16, so any future diagnostic here that reports,
/// say, `‖ΔW‖` has to settle its working precision before the number means anything.
/// Counting modules and ranks needs no such decision, so the inventory below is entirely
/// integer.
public struct LoRAOverlay {

    // MARK: - Failures

    public enum Failure: Error, CustomStringConvertible {
        case unreadable(adapter: String, underlying: String)
        case unrecognisedPrefix(adapter: String, samples: [String], accepted: [String])
        case unclassifiedTensors(adapter: String, samples: [String], count: Int)
        case perModuleAlpha(adapter: String, samples: [String], count: Int)
        case orphanedFactor(adapter: String, present: String, missing: String)
        case notMatrix(adapter: String, key: String, shape: [Int])
        case rankDisagreement(adapter: String, module: String, a: [Int], b: [Int])
        case noApplicableModules(adapter: String, tensorCount: Int)
        case filterSelectedNothing(adapter: String, filter: String, candidates: Int)
        case missingModelVersion(adapter: String)
        case versionMismatch(adapter: String, adapterVersion: String, base: String)
        case unknownBaseWeight(adapter: String, key: String, sampleBaseKeys: [String])
        case widthMismatch(adapter: String, key: String, adapterWidths: (in: Int, out: Int),
                           baseShape: [Int])

        public var description: String {
            switch self {
            case let .unreadable(adapter, underlying):
                return "\(adapter): cannot read the adapter — \(underlying)"

            case let .unrecognisedPrefix(adapter, samples, accepted):
                return "\(adapter): \(samples.count)+ tensor name(s) start with a prefix this "
                    + "loader does not recognise, so there is no base weight they can be "
                    + "matched to. Samples: \(samples). Accepted prefixes: \(accepted). "
                    + "Refusing rather than guessing: the permissive version of this mapping "
                    + "silently merged nothing and shipped rainbow noise"

            case let .unclassifiedTensors(adapter, samples, count):
                return "\(adapter): \(count) tensor(s) are neither a lora_A/lora_down nor a "
                    + "lora_B/lora_up factor. Samples: \(samples). This loader implements "
                    + "plain LoRA only; a DoRA magnitude vector or a second factorisation "
                    + "would be silently dropped, which changes the adapter's effect with no "
                    + "shape error to catch it"

            case let .perModuleAlpha(adapter, samples, count):
                return "\(adapter): \(count) per-module `.alpha` tensor(s) present. Samples: "
                    + "\(samples). This loader does not implement alpha scaling "
                    + "(`scale = alpha / rank`), and ignoring it would apply the adapter at "
                    + "the wrong strength. No LTX adapter measured carries these; a kohya or "
                    + "Diffusers-trained one may"

            case let .orphanedFactor(adapter, present, missing):
                return "\(adapter): \(present) has no matching \(missing). A LoRA factor "
                    + "without its partner is half an update; applying the half that exists "
                    + "is not a smaller error than applying none"

            case let .notMatrix(adapter, key, shape):
                return "\(adapter): \(key) has shape \(shape); a LoRA factor must be 2-D "
                    + "([rank, in] for A, [out, rank] for B)"

            case let .rankDisagreement(adapter, module, a, b):
                return "\(adapter): \(module) has A \(a) and B \(b). A is [rank, in] and B is "
                    + "[out, rank], so A's first dimension must equal B's second — they "
                    + "disagree, which means one of the two is stored transposed"

            case let .noApplicableModules(adapter, tensorCount):
                return "\(adapter): \(tensorCount) tensors resolved to ZERO applicable "
                    + "modules. An empty overlay is not an overlay — it is a silent no-op "
                    + "that renders as if the adapter were never passed, and that is the "
                    + "failure this error exists to make loud"

            case let .filterSelectedNothing(adapter, filter, candidates):
                return "\(adapter): the module filter (\(filter)) excluded all \(candidates) "
                    + "resolved module(s), leaving an empty overlay. Either the filter is "
                    + "wrong for this adapter or the adapter is wrong for this bucket — e.g. "
                    + "every LTX IC-LoRA measured so far has no audio tensors at all"

            case let .missingModelVersion(adapter):
                return "\(adapter): no `model_version` in the adapter metadata, so its "
                    + "compatibility with the base checkpoint cannot be established. Pass "
                    + "`.acceptVersionMismatch` if you intend to apply it anyway"

            case let .versionMismatch(adapter, adapterVersion, base):
                return "\(adapter): adapter is model_version \(adapterVersion), base "
                    + "checkpoint is \(base). These are shape-compatible and semantically "
                    + "unvalidated — different base weights, different VAE latent space. "
                    + "Pass `.acceptVersionMismatch` to apply it anyway and own the result"

            case let .unknownBaseWeight(adapter, key, sampleBaseKeys):
                return "\(adapter): resolves to base weight \(key), which the checkpoint does "
                    + "not have. The prefix family or the module naming is wrong for this "
                    + "checkpoint. Checkpoint samples: \(sampleBaseKeys)"

            case let .widthMismatch(adapter, key, widths, baseShape):
                return "\(adapter): \(key) — the adapter's factors are [\(widths.out), "
                    + "\(widths.in)] once multiplied out, but the base weight is \(baseShape). "
                    + "The adapter was trained against a different module"
            }
        }
    }

    // MARK: - Key normalisation

    /// Factor suffixes, longest first so `.lora_A.weight` is stripped before `.lora_A`.
    ///
    /// Two naming families are in the wild for the same thing: `lora_A`/`lora_B`, which
    /// every LTX adapter uses, and `lora_down`/`lora_up`, which most ComfyUI-trained
    /// adapters use. `A` is `down`, `B` is `up`.
    private static let downSuffixes = [".lora_A.weight", ".lora_down.weight",
                                       ".lora_A", ".lora_down"]
    private static let upSuffixes = [".lora_B.weight", ".lora_up.weight",
                                     ".lora_B", ".lora_up"]

    /// Prefix families this loader will map, longest first.
    ///
    /// `model.diffusion_model.` **must** be tried before `model.`, or every 2.5-style key
    /// loses only `model.` and keeps a `diffusion_model.` that no base key has. That
    /// single ordering mistake is the whole class of bug this file guards against.
    public static let acceptedPrefixes = ["model.diffusion_model.", "diffusion_model.",
                                          "transformer.", "model."]

    /// Which half of a rank decomposition a tensor is.
    public enum Factor: String, Sendable { case down = "A", up = "B" }

    /// Split a LoRA tensor name into `(module stem, factor)`, or `nil` if it is not a
    /// LoRA factor at all.
    ///
    /// Suffix-anchored, never a substring replacement: a module legitimately named
    /// `…lora_Adapter…` would be corrupted by `replacingOccurrences`, and a silently
    /// corrupted key resolves to a base weight that does not exist rather than to the
    /// right one.
    public static func splitFactor(_ key: String) -> (stem: String, factor: Factor)? {
        for suffix in downSuffixes where key.hasSuffix(suffix) {
            return (String(key.dropLast(suffix.count)), .down)
        }
        for suffix in upSuffixes where key.hasSuffix(suffix) {
            return (String(key.dropLast(suffix.count)), .up)
        }
        return nil
    }

    /// Map a LoRA tensor name onto the base checkpoint weight key it modifies.
    ///
    /// `basePrefix` is whatever this repo's checkpoint uses —
    /// `TransformerTopology.prefix`, i.e. `model.diffusion_model.` on a ComfyUI export
    /// and `""` on a stripped one. All four accepted prefix families collapse onto it:
    ///
    /// ```
    /// model.diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight
    /// diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight
    /// transformer.transformer_blocks.0.attn1.to_q.lora_down.weight
    /// model.transformer_blocks.0.attn1.to_q.lora_B
    ///        -> <basePrefix>transformer_blocks.0.attn1.to_q.weight
    /// ```
    ///
    /// **Returns `nil` on an unrecognised prefix rather than guessing.** The permissive
    /// alternative — pass the name through unchanged and hope — merges zero tensors on
    /// every adapter while reporting success, and the first sign of it is the distilled
    /// fast tier rendering rainbow noise. A `nil` here becomes a thrown
    /// `.unrecognisedPrefix` at load, which is the entire point.
    public static func baseWeightKey(forLoRAKey key: String, basePrefix: String) -> String? {
        guard let (stem, _) = splitFactor(key) else { return nil }
        for prefix in acceptedPrefixes where stem.hasPrefix(prefix) {
            return basePrefix + String(stem.dropFirst(prefix.count)) + ".weight"
        }
        return nil
    }

    // MARK: - Module filters

    /// A declarative include/exclude filter over base weight keys.
    ///
    /// A recipe applies different strengths to different module groups — video, audio and
    /// bridge buckets, say — so the unit of selection is a substring of the
    /// resolved base key, not a regular expression and not a closure. Declarative because
    /// it has to be serialisable into a recipe and printable in an error.
    ///
    /// **Exclude wins.** A key is selected when it matches any `include` term (or
    /// `include` is empty, meaning "everything") and matches no `exclude` term.
    public struct ModuleFilter: Sendable, Equatable, CustomStringConvertible {
        public var include: [String]
        public var exclude: [String]

        public init(include: [String] = [], exclude: [String] = []) {
            self.include = include
            self.exclude = exclude
        }

        public func selects(_ baseKey: String) -> Bool {
            if exclude.contains(where: { baseKey.contains($0) }) { return false }
            if include.isEmpty { return true }
            return include.contains(where: { baseKey.contains($0) })
        }

        public var description: String {
            let inc = include.isEmpty ? "*" : include.joined(separator: "|")
            let exc = exclude.isEmpty ? "-" : exclude.joined(separator: "|")
            return "include \(inc), exclude \(exc)"
        }

        public static let all = ModuleFilter()

        /// Every attention projection on either stream, including both cross-modal
        /// bridges: `attn1`, `attn2`, `audio_attn1`, `audio_attn2`,
        /// `audio_to_video_attn`, `video_to_audio_attn`.
        public static let attentionOnly = ModuleFilter(include: ["attn"])

        /// Feed-forward only, both streams.
        public static let feedForwardOnly = ModuleFilter(include: [".ff.", "_ff."])

        /// The video stream alone.
        ///
        /// Excluding `audio_` is enough to drop both bridges as well as the audio tower,
        /// because both bridge modules carry it (`audio_to_video_attn`,
        /// `video_to_audio_attn`); `av_ca_` covers the bridges' own AdaLN modules, which
        /// do not.
        public static let videoStream = ModuleFilter(exclude: ["audio_", "av_ca_"])

        /// The audio stream alone, bridges excluded.
        public static let audioStream = ModuleFilter(
            include: ["audio_"],
            exclude: ["audio_to_video_attn", "video_to_audio_attn", "av_ca_"])

        /// The cross-modal bridge alone: both directions of AV cross-attention and the
        /// four `av_ca_*` AdaLN modules that gate and modulate them.
        public static let crossModalBridge = ModuleFilter(
            include: ["audio_to_video_attn", "video_to_audio_attn", "av_ca_"])
    }

    // MARK: - Version gating

    /// What to do about the adapter's `model_version` against the base checkpoint's.
    ///
    /// Every LTX adapter carries one: `"2.5"` on the 2.5 spatial upscaler, `"2.5.0"` on
    /// the 2.5 distilled LoRA, `"2.3"` on the 2.3 IC-LoRAs. The base 2.5 transformer
    /// reports `"2.5.0"`. Only the major and minor components are compared, so `2.5` and
    /// `2.5.0` agree.
    public enum VersionPolicy: Sendable, Equatable {
        /// Refuse an adapter whose `model_version` does not match, or is absent.
        case require(base: String)

        /// Apply the adapter anyway, and own the consequence.
        ///
        /// A 2.3 adapter on a 2.5 base **loads cleanly**: the block count, both stream
        /// widths and every attention and feed-forward shape are unchanged across the
        /// versions, so nothing in this file can object. What differs is everything the
        /// shapes do not carry — the base weights the adapter was trained as a delta
        /// against, and the VAE latent space its conditioning assumes. The result is
        /// unvalidated, and there is no way to tell a subtly degraded render from a
        /// correct one by looking.
        ///
        /// It is spelled out as its own case so that using it is a decision in the
        /// caller's source, not a default.
        case acceptVersionMismatch(base: String)

        /// No comparison at all — for a caller that has no base version to compare to.
        case unchecked

        var baseVersion: String? {
            switch self {
            case let .require(base), let .acceptVersionMismatch(base): return base
            case .unchecked: return nil
            }
        }
    }

    /// Whether two `model_version` strings agree on major and minor.
    ///
    /// `"2.5"` vs `"2.5.0"` agree; `"2.3"` vs `"2.5"` do not. Falls back to string
    /// equality for anything that does not parse as at least `major.minor`.
    public static func versionsAgree(_ lhs: String, _ rhs: String) -> Bool {
        func components(_ s: String) -> [Int] {
            s.split(separator: ".").compactMap { Int($0) }
        }
        let l = components(lhs), r = components(rhs)
        guard l.count >= 2, r.count >= 2 else { return lhs == rhs }
        return l[0] == r[0] && l[1] == r[1]
    }

    // MARK: - Requests

    /// One adapter to load, at one strength, through one filter.
    public struct Request: Sendable {
        public var url: URL
        /// Folded into `B` once at load. `0` produces an exactly zero delta.
        public var strength: Float
        public var filter: ModuleFilter
        public var versionPolicy: VersionPolicy

        public init(url: URL, strength: Float = 1.0,
                    filter: ModuleFilter = .all,
                    versionPolicy: VersionPolicy = .unchecked) {
            self.url = url
            self.strength = strength
            self.filter = filter
            self.versionPolicy = versionPolicy
        }
    }

    // MARK: - Inventory

    /// What one loaded adapter turned out to be. Integers and names only — see the note
    /// on statistics in the type comment.
    public struct AdapterReport: Sendable, CustomStringConvertible {
        public let name: String
        public let url: URL
        public let modelVersion: String?
        public let strength: Float
        public let filter: ModuleFilter
        /// Tensors in the file.
        public let tensorCount: Int
        /// Complete `(A, B)` pairs resolved, before the filter.
        public let candidateCount: Int
        /// Pairs that survived the filter and are in the overlay.
        public let moduleCount: Int
        /// rank → how many selected modules use it.
        public let ranksByModuleCount: [Int: Int]
        /// Transformer block indices touched.
        public let blockIndices: Set<Int>
        /// Selected modules that live on the audio tower or a cross-modal bridge.
        public let audioModuleCount: Int
        public let bridgeModuleCount: Int
        /// Distinct module stems, block index elided (`transformer_blocks.N.attn1.to_q`).
        public let moduleShapes: [String: Int]
        /// The adapter's metadata, minus the multi-kilobyte licence blob.
        public let metadata: [String: String]

        public var description: String {
            let ranks = ranksByModuleCount.sorted { $0.key < $1.key }
                .map { "rank \($0.key)×\($0.value)" }.joined(separator: ", ")
            return "\(name): model_version \(modelVersion ?? "absent"), \(tensorCount) tensors, "
                + "\(candidateCount) pairs, \(moduleCount) selected (\(ranks)), "
                + "\(blockIndices.count) blocks, audio \(audioModuleCount), "
                + "bridge \(bridgeModuleCount), strength \(strength), filter [\(filter)]"
        }
    }

    // MARK: - Stored overlay

    /// One module's rank decomposition, `strength` already folded into `b`.
    struct Pair {
        let adapter: String
        let baseKey: String
        /// `[rank, in]`, as stored.
        let a: MLXArray
        /// `[out, rank]`, pre-scaled by `strength`.
        let b: MLXArray
        let rank: Int
        let inputWidth: Int
        let outputWidth: Int
    }

    private let table: [String: [Pair]]
    public let reports: [AdapterReport]

    /// Base weight keys this overlay modifies.
    public var weightKeys: Set<String> { Set(table.keys) }
    /// Total `(base key, adapter)` overlays held, counting stacking.
    public var overlayCount: Int { table.values.reduce(0) { $0 + $1.count } }
    /// Distinct base weights touched.
    public var moduleCount: Int { table.count }
    public var isEmpty: Bool { table.isEmpty }

    // MARK: - Loading

    /// Load and compose a stack of adapters.
    ///
    /// - Parameters:
    ///   - requests: applied in order; see the stacking note on the type.
    ///   - basePrefix: the base checkpoint's key prefix, i.e. `TransformerTopology.prefix`.
    ///   - baseWeightShapes: optional `[key: shape]` from the loaded checkpoint. When
    ///     given, every resolved key is checked to exist *and* to have a shape the
    ///     factors can multiply into — which is the only check in this file that can
    ///     catch an adapter whose naming is self-consistent but wrong for this
    ///     checkpoint. Strongly recommended at the wiring site.
    public static func load(_ requests: [Request],
                            basePrefix: String = TransformerTopology.comfyPrefix,
                            baseWeightShapes: [String: [Int]]? = nil) throws -> LoRAOverlay {
        var table: [String: [Pair]] = [:]
        var reports: [AdapterReport] = []

        for request in requests {
            let (pairs, report) = try loadOne(request, basePrefix: basePrefix,
                                              baseWeightShapes: baseWeightShapes)
            for pair in pairs { table[pair.baseKey, default: []].append(pair) }
            reports.append(report)
        }
        return LoRAOverlay(table: table, reports: reports)
    }

    /// Single-adapter convenience.
    public static func load(url: URL,
                            strength: Float = 1.0,
                            filter: ModuleFilter = .all,
                            versionPolicy: VersionPolicy = .unchecked,
                            basePrefix: String = TransformerTopology.comfyPrefix,
                            baseWeightShapes: [String: [Int]]? = nil) throws -> LoRAOverlay {
        try load([Request(url: url, strength: strength, filter: filter,
                          versionPolicy: versionPolicy)],
                 basePrefix: basePrefix, baseWeightShapes: baseWeightShapes)
    }

    /// An overlay over **live** factors, for training.
    ///
    /// The loading path builds `Pair`s from a file and never differentiates them. This one
    /// builds them from arrays a caller still owns — the parameters `MLXNN.valueAndGrad` is
    /// tracing — so `delta(forWeightKey:input:)` becomes part of the traced graph and the
    /// gradient reaches the factors through the transformer's own forward.
    ///
    /// `scaling` is folded into `B` here, matching the loading path's treatment of
    /// `strength`. It is folded as a graph op rather than in place, so the multiply is
    /// differentiated too and the factor the optimiser sees stays unscaled — which is the
    /// parameter that is meant to be optimised.
    ///
    /// No validation against base weights: at training time the shapes come from the
    /// topology that produced them rather than from a file that might disagree.
    public static func trainable(
        _ factors: [(baseKey: String, a: MLXArray, b: MLXArray)],
        scaling: Float, adapter: String = "trainable"
    ) -> LoRAOverlay {
        var table: [String: [Pair]] = [:]
        for factor in factors {
            let rank = factor.a.dim(0)
            let pair = Pair(adapter: adapter, baseKey: factor.baseKey,
                            a: factor.a,
                            b: scaling == 1 ? factor.b : factor.b * scaling,
                            rank: rank,
                            inputWidth: factor.a.dim(1),
                            outputWidth: factor.b.dim(0))
            table[factor.baseKey, default: []].append(pair)
        }
        return LoRAOverlay(table: table, reports: [])
    }

    private init(table: [String: [Pair]], reports: [AdapterReport]) {
        self.table = table
        self.reports = reports
    }

    private static func loadOne(_ request: Request, basePrefix: String,
                                baseWeightShapes: [String: [Int]]?)
        throws -> ([Pair], AdapterReport) {
        let name = request.url.lastPathComponent
        let tensors: [String: MLXArray]
        var metadata: [String: String]
        do {
            (tensors, metadata) = try MLX.loadArraysAndMetadata(url: request.url)
        } catch {
            throw Failure.unreadable(adapter: name, underlying: "\(error)")
        }
        metadata.removeValue(forKey: "license")

        return try build(name: name, url: request.url, tensors: tensors, metadata: metadata,
                         request: request, basePrefix: basePrefix,
                         baseWeightShapes: baseWeightShapes)
    }

    /// Build from an already-materialised tensor map.
    ///
    /// Public so a test can construct a synthetic adapter — a key set that matches
    /// nothing, an orphaned factor, a hand-computed 2×3 example — without writing a
    /// safetensors file. Every guard below runs identically on a real file.
    public static func compose(name: String,
                               tensors: [String: MLXArray],
                               metadata: [String: String] = [:],
                               strength: Float = 1.0,
                               filter: ModuleFilter = .all,
                               versionPolicy: VersionPolicy = .unchecked,
                               basePrefix: String = TransformerTopology.comfyPrefix,
                               baseWeightShapes: [String: [Int]]? = nil)
        throws -> LoRAOverlay {
        let url = URL(fileURLWithPath: "/synthetic/" + name)
        let request = Request(url: url, strength: strength, filter: filter,
                              versionPolicy: versionPolicy)
        let (pairs, report) = try build(name: name, url: url, tensors: tensors,
                                        metadata: metadata, request: request,
                                        basePrefix: basePrefix,
                                        baseWeightShapes: baseWeightShapes)
        var table: [String: [Pair]] = [:]
        for pair in pairs { table[pair.baseKey, default: []].append(pair) }
        return LoRAOverlay(table: table, reports: [report])
    }

    private static func build(name: String, url: URL,
                              tensors: [String: MLXArray],
                              metadata: [String: String],
                              request: Request,
                              basePrefix: String,
                              baseWeightShapes: [String: [Int]]?)
        throws -> ([Pair], AdapterReport) {

        // 1. Version gate, before any tensor work. A refused adapter should cost a
        //    header read, not a walk over 3320 tensors.
        let adapterVersion = metadata["model_version"]
        switch request.versionPolicy {
        case let .require(base):
            guard let adapterVersion else { throw Failure.missingModelVersion(adapter: name) }
            guard versionsAgree(adapterVersion, base) else {
                throw Failure.versionMismatch(adapter: name, adapterVersion: adapterVersion,
                                              base: base)
            }
        case .acceptVersionMismatch, .unchecked:
            break
        }

        // 2. Classify every tensor. Anything we cannot name is an error: a DoRA
        //    magnitude vector or a per-module alpha dropped in silence changes what the
        //    adapter does with nothing to show for it.
        var downs: [String: MLXArray] = [:]      // stem -> A
        var ups: [String: MLXArray] = [:]        // stem -> B
        var unrecognisedPrefix: [String] = []
        var unclassified: [String] = []
        var alphas: [String] = []

        for key in tensors.keys.sorted() {
            if key.hasSuffix(".alpha") { alphas.append(key); continue }
            guard let (stem, factor) = splitFactor(key) else {
                unclassified.append(key); continue
            }
            guard acceptedPrefixes.contains(where: { stem.hasPrefix($0) }) else {
                unrecognisedPrefix.append(key); continue
            }
            switch factor {
            case .down: downs[stem] = tensors[key]
            case .up: ups[stem] = tensors[key]
            }
        }

        if !alphas.isEmpty {
            throw Failure.perModuleAlpha(adapter: name, samples: Array(alphas.prefix(3)),
                                         count: alphas.count)
        }
        if !unclassified.isEmpty {
            throw Failure.unclassifiedTensors(adapter: name,
                                              samples: Array(unclassified.prefix(3)),
                                              count: unclassified.count)
        }
        if !unrecognisedPrefix.isEmpty {
            throw Failure.unrecognisedPrefix(adapter: name,
                                             samples: Array(unrecognisedPrefix.prefix(3)),
                                             accepted: acceptedPrefixes)
        }

        // 3. Orphan check, over every stem in the file — not only the selected ones. A
        //    half-present factorisation is a malformed file wherever it sits.
        for stem in downs.keys where ups[stem] == nil {
            throw Failure.orphanedFactor(adapter: name, present: stem + ".lora_A",
                                         missing: stem + ".lora_B")
        }
        for stem in ups.keys where downs[stem] == nil {
            throw Failure.orphanedFactor(adapter: name, present: stem + ".lora_B",
                                         missing: stem + ".lora_A")
        }

        // 4. Resolve, shape-check, filter.
        var pairs: [Pair] = []
        var candidateCount = 0
        var ranks: [Int: Int] = [:]
        var blocks = Set<Int>()
        var moduleShapes: [String: Int] = [:]
        var audioCount = 0
        var bridgeCount = 0

        for stem in downs.keys.sorted() {
            guard let a = downs[stem], let b = ups[stem] else { continue }
            guard a.shape.count == 2 else {
                throw Failure.notMatrix(adapter: name, key: stem + ".lora_A", shape: a.shape)
            }
            guard b.shape.count == 2 else {
                throw Failure.notMatrix(adapter: name, key: stem + ".lora_B", shape: b.shape)
            }
            // A is [rank, in]; B is [out, rank].
            guard a.shape[0] == b.shape[1] else {
                throw Failure.rankDisagreement(adapter: name, module: stem,
                                               a: a.shape, b: b.shape)
            }
            candidateCount += 1

            guard let baseKey = baseWeightKey(forLoRAKey: stem + ".lora_A",
                                              basePrefix: basePrefix) else {
                // Unreachable: the prefix was checked in step 2. Kept as a throw rather
                // than a force-unwrap so a future edit to either side cannot crash.
                throw Failure.unrecognisedPrefix(adapter: name, samples: [stem],
                                                 accepted: acceptedPrefixes)
            }
            guard request.filter.selects(baseKey) else { continue }

            let rank = a.shape[0]
            let inputWidth = a.shape[1]
            let outputWidth = b.shape[0]

            if let shapes = baseWeightShapes {
                guard let baseShape = shapes[baseKey] else {
                    throw Failure.unknownBaseWeight(
                        adapter: name, key: baseKey,
                        sampleBaseKeys: Array(shapes.keys.sorted().prefix(3)))
                }
                guard baseShape == [outputWidth, inputWidth] else {
                    throw Failure.widthMismatch(adapter: name, key: baseKey,
                                                adapterWidths: (in: inputWidth,
                                                                out: outputWidth),
                                                baseShape: baseShape)
                }
            }

            // `strength` folded into B once, here, rather than multiplied into the
            // result on every call: this path runs 10 modules × 48 blocks × 4 passes ×
            // N steps. Scaled in fp32 and narrowed back, so the stored factor is the
            // single best representation of `strength · B` in the adapter's dtype
            // instead of a product rounded twice.
            let scaled: MLXArray = request.strength == 1.0
                ? b
                : (b.asType(.float32) * request.strength).asType(b.dtype)

            pairs.append(Pair(adapter: name, baseKey: baseKey, a: a, b: scaled,
                              rank: rank, inputWidth: inputWidth, outputWidth: outputWidth))

            ranks[rank, default: 0] += 1
            if let index = blockIndex(of: baseKey) { blocks.insert(index) }
            let shape = generalisedModule(baseKey)
            moduleShapes[shape, default: 0] += 1
            if isBridge(baseKey) { bridgeCount += 1 } else if isAudio(baseKey) { audioCount += 1 }
        }

        // 5. The tripwires. Zero applicable modules is an error in both of its forms,
        //    and they are separate errors because the fixes are different.
        if candidateCount == 0 {
            throw Failure.noApplicableModules(adapter: name, tensorCount: tensors.count)
        }
        if pairs.isEmpty {
            throw Failure.filterSelectedNothing(adapter: name,
                                                filter: request.filter.description,
                                                candidates: candidateCount)
        }

        let report = AdapterReport(
            name: name, url: url, modelVersion: adapterVersion, strength: request.strength,
            filter: request.filter, tensorCount: tensors.count,
            candidateCount: candidateCount, moduleCount: pairs.count,
            ranksByModuleCount: ranks, blockIndices: blocks,
            audioModuleCount: audioCount, bridgeModuleCount: bridgeCount,
            moduleShapes: moduleShapes, metadata: metadata)
        return (pairs, report)
    }

    // MARK: - Bucket classification (reporting only)

    /// The cross-modal bridge markers, as the 2.5 distilled adapter names them: the two
    /// attention directions and the four `av_ca_*` AdaLN modules that gate them.
    static func isBridge(_ key: String) -> Bool {
        key.contains("audio_to_video_attn") || key.contains("video_to_audio_attn")
            || key.contains("av_ca_")
    }

    static func isAudio(_ key: String) -> Bool { key.contains("audio_") }

    static func blockIndex(of key: String) -> Int? {
        guard let range = key.range(of: "transformer_blocks.") else { return nil }
        let rest = key[range.upperBound...]
        guard let dot = rest.firstIndex(of: ".") else { return nil }
        return Int(rest[rest.startIndex..<dot])
    }

    /// `…transformer_blocks.7.attn1.to_q.weight` → `transformer_blocks.N.attn1.to_q`, so
    /// 48 blocks' worth of the same module collapse to one inventory line.
    static func generalisedModule(_ key: String) -> String {
        var out = key
        for prefix in acceptedPrefixes where out.hasPrefix(prefix) {
            out = String(out.dropFirst(prefix.count))
            break
        }
        if out.hasSuffix(".weight") { out = String(out.dropLast(".weight".count)) }
        guard let range = out.range(of: "transformer_blocks.") else { return out }
        let rest = out[range.upperBound...]
        guard let dot = rest.firstIndex(of: "."), Int(rest[rest.startIndex..<dot]) != nil else {
            return out
        }
        return String(out[out.startIndex..<range.upperBound]) + "N" + String(rest[dot...])
    }

    // MARK: - The seam

    /// The residual a linear must add, or `nil` when no adapter touches it.
    ///
    /// ```
    /// delta = ((x @ Aᵀ) @ Bᵀ)          strength already folded into B
    /// ```
    ///
    /// **`x` is cast to the factor's dtype**, matching `DiTModules.linear` and for the
    /// same reason: the stack runs bf16 end to end, and letting MLX promote the matmul to
    /// fp32 would make this one linear compute at a different precision from every other
    /// one. The rank-sized intermediate stays in that dtype too.
    ///
    /// Non-throwing on purpose: it sits inside the per-linear hot path, and every
    /// condition it could throw on is settled at load. The one runtime disagreement it
    /// can see — an input whose width is not the width the adapter was trained for — is a
    /// `precondition`, because it means the call site passed the wrong tensor, and it is
    /// unreachable when `load` was given `baseWeightShapes`.
    ///
    /// Stacked adapters are summed in request order; see the note on the type.
    public func delta(forWeightKey key: String, input x: MLXArray) -> MLXArray? {
        guard let pairs = table[key] else { return nil }
        var accumulated: MLXArray?
        for pair in pairs {
            precondition(
                x.shape.last == pair.inputWidth,
                "LoRAOverlay: \(pair.adapter) overlays \(key) with a \(pair.inputWidth)-wide "
                + "input, but the tensor passed is \(x.shape). The call site is applying the "
                + "overlay to the wrong linear.")
            let hidden = MLX.matmul(x.asType(pair.a.dtype), pair.a.transposed(1, 0))
            let contribution = MLX.matmul(hidden.asType(pair.b.dtype),
                                          pair.b.transposed(1, 0))
            accumulated = accumulated.map { $0 + contribution.asType($0.dtype) } ?? contribution
        }
        return accumulated
    }

    /// Whether any adapter overlays this weight — a cheaper check than `delta` for a
    /// caller that wants to branch before it has an input.
    public func overlays(weightKey key: String) -> Bool { table[key] != nil }

    /// How many adapters overlay this weight.
    public func overlayCount(forWeightKey key: String) -> Int { table[key]?.count ?? 0 }

    // MARK: - Post-hoc validation

    /// Check this overlay against the checkpoint it will be applied to.
    ///
    /// The same check `load(_:basePrefix:baseWeightShapes:)` performs when given shapes,
    /// available separately for a caller that loads the overlay before the checkpoint.
    /// Throws if any resolved key is absent from the base, or if the widths disagree.
    public func validate(againstBaseWeights weights: [String: MLXArray]) throws {
        for (key, pairs) in table {
            guard let base = weights[key] else {
                throw Failure.unknownBaseWeight(
                    adapter: pairs.first?.adapter ?? "overlay", key: key,
                    sampleBaseKeys: Array(weights.keys.sorted().prefix(3)))
            }
            for pair in pairs where base.shape != [pair.outputWidth, pair.inputWidth] {
                throw Failure.widthMismatch(
                    adapter: pair.adapter, key: key,
                    adapterWidths: (in: pair.inputWidth, out: pair.outputWidth),
                    baseShape: base.shape)
            }
        }
    }
}

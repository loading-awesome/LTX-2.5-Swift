// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// Which linears a LoRA adapts, and how wide they are — read from the checkpoint.
///
/// The default target list — `["to_k", "to_q", "to_v", "to_out.0"]` — is a list of *suffixes*,
/// and turning it into concrete parameters needs the real `[out, in]` of every matching
/// weight, because `A` is `[rank, in]` and `B` is `[out, rank]`. Inferring those from the
/// model's nominal width would be wrong in this architecture: the cross-modal attentions are
/// deliberately asymmetric — `audio_to_video_attn.to_q` is `[audio, video]` while its
/// `to_out` is `[video, audio]` — so a single "hidden size" produces adapters that fail on
/// rank at the first matmul, or worse, adapt the wrong half of a bridge.
///
/// So the shapes come from the header, the same place `TransformerTopology` reads.
public enum DiTLoRATargets {

    /// Which parts of the model an adapter is allowed to touch.
    ///
    /// ## The default target list does not reproduce the released adapters
    ///
    /// On `ltx-2.5-22b-dev-transformer-bf16`, the default target list
    /// `["to_k", "to_q", "to_v", "to_out.0"]` matches **1216** weights:
    ///
    /// ```
    ///  384  video attn1/attn2      <- the released adapters have these
    ///  384  audio tower            <- released adapters report audio 0
    ///  384  cross-modal bridge     <- released adapters report bridge 0
    ///   64  connectors / other
    /// ```
    ///
    /// Every released adapter read here — the distilled rank-450 one and the 2.3
    /// IC-LoRAs — is **480 modules: 48 blocks x 10**, being the eight attention projections
    /// plus `ff.net.0.proj` and `ff.net.2`, on the video stream only. So the shipped default
    /// both *adds* the audio tower and the bridges and *omits* the feed-forward.
    ///
    /// The released adapters were plainly trained against an explicit target list rather than
    /// the default, so training with the defaults produces a structurally different adapter
    /// from any of them. Worth choosing deliberately rather than inheriting.
    public enum Scope: Sendable {
        /// Everything the suffixes match, including the audio tower, the cross-modal
        /// bridges and the embedding connectors. 1216 weights at the default suffixes.
        case referenceDefault
        /// The video stream's own transformer blocks only — the set every released adapter
        /// occupies. 480 weights at ``TrainableLoRA/releasedAdapterTargetModules``.
        case videoStreamOnly

        func admits(_ key: String) -> Bool {
            switch self {
            case .referenceDefault: return true
            case .videoStreamOnly:
                guard key.contains("transformer_blocks"), !key.contains("connector")
                else { return false }
                return !key.contains("audio_") && !key.contains("video_to_audio")
            }
        }
    }

    public struct Target: Sendable, Equatable {
        /// The base weight key, exactly as the checkpoint names it.
        public let key: String
        public let inFeatures: Int
        public let outFeatures: Int
        /// The transformer block this weight belongs to, or nil for a weight outside the
        /// stack. Kept so a caller can restrict training to a range of blocks.
        public let block: Int?
    }

    /// Every weight whose name ends in one of `suffixes`, with its measured shape.
    ///
    /// `suffixes` are matched against the key with `.weight` removed, so `to_out.0` matches
    /// `...attn1.to_out.0.weight` and does not also match `to_out.1`.
    ///
    /// **Cross-modal bridges are included by default and that is a decision, not an
    /// oversight.** They match the default suffix list, and every released adapter read here
    /// contains zero bridge modules — see `LoRAOverlay.AdapterReport`'s
    /// `bridgeModuleCount`, which exists to report exactly this. Excluding them here would
    /// silently disagree with the target list; reporting the count lets a caller decide.
    public static func discover(header: SafetensorsHeader,
                                prefix: String = TransformerTopology.comfyPrefix,
                                suffixes: [String] = TrainableLoRA.defaultTargetModules,
                                scope: Scope = .referenceDefault,
                                blocks: Range<Int>? = nil) -> [Target] {
        var out: [Target] = []
        for (key, entry) in header.tensors {
            guard key.hasSuffix(".weight"), entry.shape.count == 2 else { continue }
            let stem = String(key.dropLast(".weight".count))
            guard suffixes.contains(where: { stem.hasSuffix("." + $0) || stem == $0 })
            else { continue }
            guard scope.admits(key) else { continue }
            let block = blockIndex(in: key)
            if let blocks, let block, !blocks.contains(block) { continue }
            if let blocks, block == nil, !blocks.isEmpty { continue }
            _ = prefix
            out.append(Target(key: key, inFeatures: entry.shape[1],
                              outFeatures: entry.shape[0], block: block))
        }
        // Sorted so the parameter order is a property of the checkpoint rather than of
        // dictionary iteration — the same reason `TensorFile` sorts before writing.
        return out.sorted { $0.key < $1.key }
    }

    /// `...blocks.<n>...` -> n.
    static func blockIndex(in key: String) -> Int? {
        let parts = key.split(separator: ".")
        for (index, part) in parts.enumerated()
        where part.hasSuffix("blocks") || part == "blocks" {
            guard index + 1 < parts.count else { return nil }
            return Int(parts[index + 1])
        }
        return nil
    }

    /// Fresh trainable factors for the discovered targets.
    public static func makeLoRA(header: SafetensorsHeader,
                                rank: Int = TrainableLoRA.defaultRank,
                                alpha: Int = TrainableLoRA.defaultAlpha,
                                suffixes: [String] = TrainableLoRA.defaultTargetModules,
                                scope: Scope = .referenceDefault,
                                blocks: Range<Int>? = nil) -> TrainableLoRA {
        let targets = discover(header: header, suffixes: suffixes, scope: scope,
                               blocks: blocks)
        return TrainableLoRA(
            shapes: targets.map { (key: $0.key, inFeatures: $0.inFeatures,
                                   outFeatures: $0.outFeatures) },
            rank: rank, alpha: alpha)
    }

    /// Trainable parameter count, for a memory estimate before anything is allocated.
    public static func parameterCount(_ targets: [Target], rank: Int) -> Int {
        targets.reduce(0) { $0 + rank * ($1.inFeatures + $1.outFeatures) }
    }
}

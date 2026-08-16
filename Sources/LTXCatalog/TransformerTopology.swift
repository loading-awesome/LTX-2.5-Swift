// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// The DiT's structure, read from the checkpoint header rather than declared.
///
/// ## Why this type exists at all
///
/// A whole-render numerical comparison cannot catch an integration bug here: the
/// production-shape trajectory saturates at every step count, so end-to-end numbers
/// stop discriminating. Comparing individual tensors does not see *structure*
/// either — a port that wires the cross-modal attention backwards produces
/// correctly-shaped tensors everywhere.
///
/// This reads the topology out of the weights, where it is unambiguous, and can
/// therefore reject a structurally wrong port before a single tensor is compared.
/// It costs two ranged reads: no payload is touched.
///
/// ## Every value here is measured
///
/// From `ltx-2.5-22b-dev-transformer-bf16.safetensors` (4349 tensors, 21.00 B
/// parameters), read 2026-08-11. The named constants are defaults for *this*
/// checkpoint; the reader derives all of them and `verify()` compares.
public struct TransformerTopology: Sendable, Equatable {

    /// ComfyUI's export prefix. Present on this checkpoint and stripped by other
    /// tools, so a port must tolerate both spellings and assume neither.
    public static let comfyPrefix = "model.diffusion_model."

    public let blockCount: Int
    /// Video stream width — `inner_dim`.
    public let videoWidth: Int
    /// Audio stream width — `audio_inner_dim`. **Not equal to `videoWidth`**; see
    /// `LatentGeometry.streamTokenCounts` for why this is not one packed sequence.
    public let audioWidth: Int
    /// Latent channels in and out of the patchify projections.
    public let latentChannels: Int
    /// Feed-forward inner widths, per stream.
    public let videoFeedForwardWidth: Int
    public let audioFeedForwardWidth: Int
    /// Whether each stream's feed-forward carries biases. **These differ.**
    public let videoFeedForwardHasBias: Bool
    public let audioFeedForwardHasBias: Bool
    /// Whether the embeddings connectors ship inside the transformer file.
    public let carriesEmbeddingsConnectors: Bool
    /// Prefix the names actually used, if any.
    public let prefix: String
    /// Attention head count parsed from the checkpoint.
    public let numAttentionHeads: Int

    public enum Failure: Error, CustomStringConvertible {
        case noBlocks
        case missing(String)
        case nonContiguousBlocks(found: Int, highest: Int)
        case unexpectedShape(name: String, shape: [Int], expected: String)
        case streamsIdentical(width: Int)
        case crossModalMiswired(String)

        public var description: String {
            switch self {
            case .noBlocks:
                return "no transformer_blocks.<n>.* tensors found; this is not an LTX DiT "
                    + "checkpoint, or the prefix differs from both known forms"
            case let .missing(name):
                return "required tensor \(name) is absent"
            case let .nonContiguousBlocks(found, highest):
                return "found \(found) blocks but the highest index is \(highest); block "
                    + "indices must be contiguous from 0 or a block resolves to the wrong depth"
            case let .unexpectedShape(name, shape, expected):
                return "\(name) has shape \(shape), expected \(expected)"
            case let .streamsIdentical(width):
                return "video and audio streams are both \(width) wide. On LTX 2.5 they "
                    + "differ (4096 / 2048); equal widths mean this was read as one packed "
                    + "sequence, which is the structural mistake this check exists to catch"
            case let .crossModalMiswired(detail):
                return "cross-modal attention is not wired as this model requires: \(detail)"
            }
        }
    }

    // MARK: - Reading

    /// Derive the topology from a header.
    public static func read(_ header: SafetensorsHeader) throws -> TransformerTopology {
        // Accept the ComfyUI prefix or its absence. A port that hardcodes one and
        // meets the other reports "no tensors" and looks like a corrupt file.
        let prefix = header.hasPrefix(comfyPrefix + "transformer_blocks.")
            ? comfyPrefix
            : (header.hasPrefix("transformer_blocks.") ? "" : comfyPrefix)

        let blockNames = header.names(withPrefix: prefix + "transformer_blocks.")
        guard !blockNames.isEmpty else { throw Failure.noBlocks }

        var indices = Set<Int>()
        let stem = prefix + "transformer_blocks."
        for name in blockNames {
            let rest = name.dropFirst(stem.count)
            guard let dot = rest.firstIndex(of: "."),
                  let i = Int(rest[rest.startIndex..<dot]) else { continue }
            indices.insert(i)
        }
        guard let highest = indices.max() else { throw Failure.noBlocks }
        guard indices.count == highest + 1 else {
            throw Failure.nonContiguousBlocks(found: indices.count, highest: highest)
        }

        func entry(_ name: String) throws -> SafetensorsHeader.TensorEntry {
            guard let e = header.tensors[prefix + name] else { throw Failure.missing(prefix + name) }
            return e
        }

        // Widths come from the patchify projections, which are [width, channels].
        let vPatch = try entry("patchify_proj.weight")
        let aPatch = try entry("audio_patchify_proj.weight")
        guard vPatch.shape.count == 2 else {
            throw Failure.unexpectedShape(name: "patchify_proj.weight",
                                          shape: vPatch.shape, expected: "[width, channels]")
        }
        guard aPatch.shape.count == 2, aPatch.shape[1] == vPatch.shape[1] else {
            throw Failure.unexpectedShape(
                name: "audio_patchify_proj.weight", shape: aPatch.shape,
                expected: "[width, \(vPatch.shape[1])] — both streams patchify the same "
                    + "latent channel count")
        }
        let videoWidth = vPatch.shape[0]
        let audioWidth = aPatch.shape[0]
        guard videoWidth != audioWidth else { throw Failure.streamsIdentical(width: videoWidth) }

        let vFF = try entry("transformer_blocks.0.ff.net.0.proj.weight")
        let aFF = try entry("transformer_blocks.0.audio_ff.net.0.proj.weight")

        var numAttentionHeads = 32
        if let configString = header.metadata["config"],
           let configData = configString.data(using: .utf8),
           let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let transformerDict = configDict["transformer"] as? [String: Any] {
            if let heads = transformerDict["num_attention_heads"] as? Int {
                numAttentionHeads = heads
            } else if let headsString = transformerDict["num_attention_heads"] as? String,
                      let heads = Int(headsString) {
                numAttentionHeads = heads
            }
        }

        return TransformerTopology(
            blockCount: indices.count,
            videoWidth: videoWidth,
            audioWidth: audioWidth,
            latentChannels: vPatch.shape[1],
            videoFeedForwardWidth: vFF.shape[0],
            audioFeedForwardWidth: aFF.shape[0],
            // Measured asymmetry, and a trap for any shared FF code path: the video
            // feed-forward has NO biases and the audio one does. Build both from one
            // `bias: true` path and the video weights fail to load; build both with
            // `bias: false` and the audio biases are silently dropped, which is a
            // numerical divergence at every block with no shape error to catch it.
            videoFeedForwardHasBias:
                header.tensors[prefix + "transformer_blocks.0.ff.net.0.proj.bias"] != nil,
            audioFeedForwardHasBias:
                header.tensors[prefix + "transformer_blocks.0.audio_ff.net.0.proj.bias"] != nil,
            carriesEmbeddingsConnectors:
                header.hasPrefix(prefix + "video_embeddings_connector.")
                && header.hasPrefix(prefix + "audio_embeddings_connector."),
            prefix: prefix,
            numAttentionHeads: numAttentionHeads)
    }

    // MARK: - Structural verification

    /// Check the cross-modal attention is wired the way the weights say it is.
    ///
    /// This is the check a whole-render comparison cannot make. The two bridges are
    /// *not* symmetric, and swapping them is the natural mistake:
    ///
    /// ```
    /// audio_to_video_attn   to_q [audio, video]   query from VIDEO
    ///                       to_k [audio, audio]   key   from AUDIO
    ///                       to_out [video, audio] writes the VIDEO stream
    ///
    /// video_to_audio_attn   to_q [audio, audio]   query from AUDIO
    ///                       to_k [audio, video]   key   from VIDEO
    ///                       to_out [audio, audio] writes the AUDIO stream
    /// ```
    ///
    /// Both name their direction after the stream they *read*, and write the other
    /// one — the opposite of what the names suggest. Verified against weight shapes,
    /// where it cannot be misread.
    public func verifyCrossModalWiring(_ header: SafetensorsHeader) throws {
        func shape(_ name: String) throws -> [Int] {
            guard let e = header.tensors[prefix + name] else { throw Failure.missing(prefix + name) }
            return e.shape
        }
        for block in 0..<min(blockCount, 2) {
            let b = "transformer_blocks.\(block)."

            let a2vQ = try shape(b + "audio_to_video_attn.to_q.weight")
            guard a2vQ == [audioWidth, videoWidth] else {
                throw Failure.crossModalMiswired(
                    "block \(block) audio_to_video_attn.to_q is \(a2vQ), expected "
                    + "[\(audioWidth), \(videoWidth)] — its query must come from the VIDEO "
                    + "stream despite the name")
            }
            let a2vOut = try shape(b + "audio_to_video_attn.to_out.0.weight")
            guard a2vOut == [videoWidth, audioWidth] else {
                throw Failure.crossModalMiswired(
                    "block \(block) audio_to_video_attn.to_out is \(a2vOut), expected "
                    + "[\(videoWidth), \(audioWidth)] — it writes the VIDEO stream")
            }

            let v2aK = try shape(b + "video_to_audio_attn.to_k.weight")
            guard v2aK == [audioWidth, videoWidth] else {
                throw Failure.crossModalMiswired(
                    "block \(block) video_to_audio_attn.to_k is \(v2aK), expected "
                    + "[\(audioWidth), \(videoWidth)] — its key must come from the VIDEO stream")
            }
            let v2aOut = try shape(b + "video_to_audio_attn.to_out.0.weight")
            guard v2aOut == [audioWidth, audioWidth] else {
                throw Failure.crossModalMiswired(
                    "block \(block) video_to_audio_attn.to_out is \(v2aOut), expected "
                    + "[\(audioWidth), \(audioWidth)] — it writes the AUDIO stream")
            }
        }
    }

    /// Text cross-attention widths, per stream.
    ///
    /// `attn2` consumes the projected text conditioning, so these must equal the
    /// widths of the projected text features — 4096 and 2048 as measured. This is
    /// the one place the encode stage and the DiT have to agree,
    /// and they agree through the checkpoint rather than through a constant.
    public func textCrossAttentionWidths(_ header: SafetensorsHeader) throws
        -> (video: Int, audio: Int) {
        guard let v = header.tensors[prefix + "transformer_blocks.0.attn2.to_k.weight"],
              let a = header.tensors[prefix + "transformer_blocks.0.audio_attn2.to_k.weight"]
        else { throw Failure.missing(prefix + "transformer_blocks.0.attn2.to_k.weight") }
        return (video: v.shape[1], audio: a.shape[1])
    }
}

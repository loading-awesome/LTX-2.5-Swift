// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// Cross-validation of the header reader against the real checkpoint.
///
/// `tools/ltx_probe.py` reads the same files independently and its output is
/// committed at `docs/reference/probe-ltx-2.5.json`, so the Swift reader has
/// something to be wrong against — the only kind of check worth having on a
/// parser this load-bearing. Two independent readers of the same 39 GiB file
/// agreeing is a much stronger statement than either alone.
///
/// **Skipped when the weights are absent.** The MLX-free suites must stay
/// runnable on a bare checkout in seconds; these are the opt-in half, and they
/// still read only the header, not the payload.
@Suite("Real checkpoint headers")
struct RealCheckpointTests {

    static let root = URL(fileURLWithPath: LTXConfiguration.resolved.checkpoints.root!)

    static func url(_ relative: String) -> URL? {
        let u = root.appendingPathComponent(relative)
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }

    @Test("the DiT header agrees with the probe: 4349 tensors, 21.00B parameters")
    func transformer() throws {
        guard let u = Self.url("diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
        else { return }

        let header = try SafetensorsHeader.read(from: u)

        // Probe-reported facts. If these move, either the file changed or one of
        // the two readers is wrong -- both worth failing for.
        #expect(header.tensors.count == 4349)
        #expect(header.parameterCount == 21_004_025_600)

        // The checkpoint declares its own version, and the pipeline reads it to
        // set every default. It outranks anything inferred.
        let version = header.metadata["model_version"]
        #expect(version?.contains("2.5") == true)

        // dtypes: bf16 throughout with a little fp32, as probed.
        let widths = Set(header.tensors.values.map(\.dtype))
        #expect(widths.contains(.bf16))

        // The embedded config is authoritative for every dimension. Read it;
        // never declare these.
        let config = try #require(header.metadataJSON("config"))
        let transformer = try #require(config["transformer"] as? [String: Any])
        #expect(transformer["num_layers"] as? Int == 48)
        #expect(transformer["in_channels"] as? Int == 128)
        #expect(transformer["out_channels"] as? Int == 128)
        #expect(transformer["cross_attention_dim"] as? Int == 4096)
        #expect(transformer["num_attention_heads"] as? Int == 32)
        #expect(transformer["attention_head_dim"] as? Int == 128)
        // Measured on the checkpoint: the video FF carries no bias while the
        // audio FF does, and the keyframe embedding is on. That asymmetry is the
        // trap -- a port that reads one `ff_bias` for both streams is wrong on one.
        #expect(transformer["ff_bias"] as? Bool == false)
        #expect(transformer["use_keyframes_abs_pos_embedding"] as? Bool == true)

        // The keyframe marker is a learned [1, inner_dim] parameter, and the
        // checkpoint carries it at (1, 4096).
        if let kf = header.tensors.first(where: { $0.key.contains("keyframes_abs_pos") }) {
            #expect(kf.value.shape == [1, 4096])
        }
    }

    @Test("the text encoder is a unified Gemma-4 with vision and audio towers attached")
    func textEncoder() throws {
        guard let u = Self.url("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
        else { return }

        let header = try SafetensorsHeader.read(from: u)
        #expect(header.tensors.count == 686)

        // The claim this corrects: an earlier reading called this a text-only
        // extraction, after searching for Gemma-3/LLaVA-shaped names (`siglip`,
        // `vision_tower`, `mm_projector`). The unified Gemma-4 calls them these.
        #expect(header.hasPrefix("vision_model."))
        #expect(header.hasPrefix("multi_modal_projector."))
        #expect(header.hasPrefix("audio_projector."))
        // The vision path is SHALLOW -- a patch embedder, not a deep tower. The
        // unified model has no stack of hidden layers here, unlike the
        // non-unified 12B and 31B variants with 16 and 27.
        #expect(header.names(withPrefix: "vision_model.").count < 20)

        // LTX's dual-linear projection ships inside this file, which is why
        // caption_projection is None on the transformer.
        let projection = header.names(withPrefix: "text_embedding_projection.")
        #expect(!projection.isEmpty)
    }

    @Test("both video VAEs ship, and they are different decoders")
    func videoVAEs() throws {
        guard let conv = Self.url("vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
              let na = Self.url("vae/ltx-2.5-video-vae-bf16.safetensors")
        else { return }

        let convHeader = try SafetensorsHeader.read(from: conv)
        let naHeader = try SafetensorsHeader.read(from: na)

        // 170 vs 396 tensors: the causal conv decoder and the NA diffusion
        // decoder. They are not the same computation, so which of the two a run
        // loaded has to be recorded with its output.
        #expect(convHeader.tensors.count == 170)
        #expect(naHeader.tensors.count == 396)
        #expect(convHeader.tensors.count != naHeader.tensors.count)
    }

    @Test("headers are read without touching the payload")
    func headersAreCheap() throws {
        guard let u = Self.url("diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
        else { return }

        // 39 GiB of weights; this must complete in well under a second. If it
        // ever does not, something started mapping the file.
        let start = Date()
        let header = try SafetensorsHeader.read(from: u)
        let elapsed = Date().timeIntervalSince(start)
        #expect(elapsed < 2.0)

        // The data section is the overwhelming majority of the file, and the
        // header accounts for all of it without reading any.
        let fileSize = try #require(
            (try? FileManager.default.attributesOfItem(atPath: u.path)[.size]) as? Int)
        #expect(header.declaredDataSize + header.dataStart == fileSize)
    }
}

// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
import LTXFoundation
@testable import LTXCatalog

@Suite("Checkpoint identity bind")
struct CheckpointIdentityBindTests {

    @Test("a missing path is refused before any header is read")
    func missingFile() {
        let missing = URL(fileURLWithPath: "/tmp/ltx-bind-does-not-exist.safetensors")
        #expect(throws: CheckpointIdentity.Mismatch.self) {
            try CheckpointIdentity.bind(
                transformer: missing, textEncoder: missing,
                videoVAE: missing, audioVAE: missing)
        }
    }

    @Test("the real 2.5 set binds as causal video VAE")
    func realSet() throws {
        let root = URL(fileURLWithPath: LTXConfiguration.resolved.checkpoints.root!)
        let dit = root.appendingPathComponent(
            "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
        let text = root.appendingPathComponent(
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
        let video = root.appendingPathComponent(
            "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
        let audio = root.appendingPathComponent(
            "vae/ltx-2.5-audio-vae-bf16.safetensors")
        let fm = FileManager.default
        guard fm.fileExists(atPath: dit.path), fm.fileExists(atPath: text.path),
              fm.fileExists(atPath: video.path), fm.fileExists(atPath: audio.path)
        else { return }

        let bound = try CheckpointIdentity.bind(
            transformer: dit, textEncoder: text, videoVAE: video, audioVAE: audio)
        #expect(bound.videoDecoder == .videoVAECausal)
        #expect(bound.transformer.components.contains(.transformer))
        #expect(bound.textEncoder.components.contains(.textEncoder))
        #expect(bound.audioVAE.components.contains(.audioVAE))
    }
}

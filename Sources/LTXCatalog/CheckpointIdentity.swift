// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// Which component a checkpoint file holds.
///
/// LTX 2.3 shipped one combined 43 GiB file. **2.5 ships thirteen**, across
/// `diffusion_models/`, `text_encoders/`, `vae/`, `latent_upscale_models/`,
/// `loras/` and `model_patches/`. So "load the checkpoint" is no longer a single
/// path, and deciding what a file is has to be a rule rather than a filename.
public enum LTXComponent: String, Sendable, CaseIterable {
    case transformer
    case textEncoder
    case videoVAECausal
    case videoVAENADiffusion
    case audioVAE
    case vocoder
    case durationHead
    case latentUpscaler
    case lora
}

/// Component detection, by key presence — the same rule the probe uses.
///
/// Filenames are a convention and conventions get renamed; keys are the file's
/// own statement about what it contains. Every prefix below was **measured**
/// against the real 2.5 files rather than assumed:
///
/// ```
/// transformer          model.diffusion_model.          4349 tensors
/// text encoder         model.layers.  + vision_model.   686
/// video VAE (causal)   encoder. + decoder.              170
/// video VAE (NA diff)  ... + decoder.conv_in_x_t        396
/// audio VAE + vocoder  audio_vae. + vocoder.           1329
/// ```
///
/// The two video VAEs are the case that matters. They differ by **one key**:
/// `decoder.conv_in_x_t.weight` selects the neighborhood-attention diffusion
/// decoder; without it the file carries the causal conv decoder. They are
/// different decoders, so a render must record which one it loaded.
public enum CheckpointIdentity {

    /// The key that distinguishes the NA diffusion decoder from the causal one.
    public static let naDiffusionSelector = "decoder.conv_in_x_t.weight"

    public static func components(of header: SafetensorsHeader) -> Set<LTXComponent> {
        var found: Set<LTXComponent> = []

        if header.hasPrefix("model.diffusion_model.") { found.insert(.transformer) }
        if header.hasPrefix("duration_head.") { found.insert(.durationHead) }
        if header.hasPrefix("audio_vae.") { found.insert(.audioVAE) }
        if header.hasPrefix("vocoder.") { found.insert(.vocoder) }

        // A Gemma text stack. `text_embedding_projection.` is LTX's dual-linear
        // projection, which 2.5 ships *inside* the encoder file — which is in turn
        // why `caption_projection` is nil on the transformer.
        if header.hasPrefix("text_embedding_projection.")
            || (header.hasPrefix("model.layers.") && header.hasPrefix("vision_model.")) {
            found.insert(.textEncoder)
        }

        // Video VAE: encoder + decoder at the root, no component prefix.
        if header.hasPrefix("encoder.") && header.hasPrefix("decoder.") {
            found.insert(header.tensors[naDiffusionSelector] != nil
                         ? .videoVAENADiffusion : .videoVAECausal)
        }

        // LatentUpsampler has no LTX-shaped prefix at all; it is identified by its
        // own module names plus its declared class.
        if header.tensors["initial_conv.weight"] != nil,
           header.hasPrefix("post_upsample_res_blocks.") {
            found.insert(.latentUpscaler)
        }

        if header.tensors.keys.contains(where: { $0.contains(".lora_") || $0.hasSuffix(".lora_A.weight") }) {
            found.insert(.lora)
        }

        return found
    }

    /// One file, identified.
    public struct File: Sendable {
        public let url: URL
        public let components: Set<LTXComponent>
        /// The checkpoint's own declared version. Authoritative: it is what sets
        /// every pipeline default.
        public let modelVersion: String?
        public let tensorCount: Int
        public let parameterCount: Int

        public var name: String { url.lastPathComponent }
    }

    public static func identify(fileAt url: URL) throws -> File {
        let header = try SafetensorsHeader.read(from: url)
        return File(url: url,
                    components: components(of: header),
                    modelVersion: header.metadata["model_version"],
                    tensorCount: header.tensors.count,
                    parameterCount: header.parameterCount)
    }

    /// Walk a directory tree and identify every safetensors file in it.
    ///
    /// Header-only, so this is cheap even over ~70 GiB: the 39 GiB transformer's
    /// header reads in tens of milliseconds.
    public static func discover(in directory: URL) throws -> [File] {
        let fm = FileManager.default
        guard let walker = fm.enumerator(at: directory,
                                         includingPropertiesForKeys: [.isRegularFileKey],
                                         options: [.skipsHiddenFiles]) else { return [] }
        var out: [File] = []
        for case let url as URL in walker where url.pathExtension == "safetensors" {
            // A file that fails to parse is reported by skipping rather than
            // aborting the walk: one corrupt download should not make the whole
            // model directory unreadable.
            if let file = try? identify(fileAt: url) { out.append(file) }
        }
        return out.sorted { $0.name < $1.name }
    }

    // MARK: - Consistency

    public enum Mismatch: Error, Equatable, CustomStringConvertible {
        case missing(LTXComponent)
        case missingFile(String, String)
        case ambiguous(LTXComponent, [String])
        case unexpectedComponent(name: String, expected: LTXComponent, found: [String])
        /// The right component, the wrong build of it.
        ///
        /// Exists because ``unexpectedComponent(name:expected:found:)`` would say "is not a
        /// transformer checkpoint" about a file that plainly is one. The distinction is the
        /// whole point here: the dev and distilled transformers are byte-identical in
        /// structure, so this is a claim about the *filename*, and the message has to say so
        /// or it reads as a corrupt file.
        case wrongVariant(name: String, expectedFragment: String)
        case naDiffusionUnsupported(String)
        case versionDisagreement([String: String])

        public var description: String {
            switch self {
            case let .missing(component):
                return "no file provides \(component.rawValue)"
            case let .missingFile(name, path):
                return "the \(name) checkpoint is not at \(path)"
            case let .ambiguous(component, names):
                return "\(names.count) files provide \(component.rawValue): "
                    + names.joined(separator: ", ")
            case let .unexpectedComponent(name, expected, found):
                let got = found.isEmpty ? "none of the known LTX components" : found.joined(separator: ", ")
                return "\(name) is not a \(expected.rawValue) checkpoint (found \(got))"
            case let .wrongVariant(name, fragment):
                return "\(name) is not the '\(fragment)' build this pipeline expects. The dev "
                    + "and distilled transformers are byte-identical in structure — same "
                    + "header, 4349 tensors, same model_version — so the filename is the "
                    + "only signal, and running the wrong one produces a render rather than "
                    + "an error"
            case let .naDiffusionUnsupported(path):
                return "\(path) is the neighborhood-attention video decoder. v1 ships the "
                    + "causal conv decoder; MLX has no na3d kernel"
            case let .versionDisagreement(byFile):
                let pairs = byFile.sorted { $0.key < $1.key }
                    .map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
                return "files declare different model versions (\(pairs)) — mixing a 2.5 "
                    + "transformer with a VAE from another release is a shape error three "
                    + "stages later, not a load failure"
            }
        }
    }

    /// A resolved set for one render, plus the checks that make it trustworthy.
    public struct Resolved: Sendable {
        public let transformer: File
        public let textEncoder: File
        public let videoVAE: File
        public let audioVAE: File
        /// Which video decoder the chosen VAE file holds. Recorded because the
        /// two are different computations and a render must name the one it used.
        public let videoDecoder: LTXComponent
        public let modelVersion: String?
    }

    /// Resolve a render set from discovered files.
    ///
    /// `preferNADiffusion` picks between the two video VAEs when both are
    /// present, which for 2.5 is the normal case. Default false: v1 ships the
    /// causal conv decoder, because the NA decoder needs an `na3d` kernel that
    /// does not exist on Apple silicon.
    public static func resolve(_ files: [File],
                              preferNADiffusion: Bool = false) throws -> Resolved {
        func single(_ component: LTXComponent) throws -> File {
            let matches = files.filter { $0.components.contains(component) }
            guard let first = matches.first else { throw Mismatch.missing(component) }
            guard matches.count == 1 else {
                throw Mismatch.ambiguous(component, matches.map(\.name))
            }
            return first
        }

        let transformer = try single(.transformer)
        let textEncoder = try single(.textEncoder)
        let audioVAE = try single(.audioVAE)

        let wanted: LTXComponent = preferNADiffusion ? .videoVAENADiffusion : .videoVAECausal
        let videoVAE: File
        if let match = files.first(where: { $0.components.contains(wanted) }) {
            videoVAE = match
        } else {
            // Fall back to the other decoder rather than failing, but the caller
            // is told which one it got: silently substituting a different decoder
            // is precisely the kind of divergence no automated check can see,
            // because both produce plausible video.
            let other: LTXComponent = preferNADiffusion ? .videoVAECausal : .videoVAENADiffusion
            guard let match = files.first(where: { $0.components.contains(other) }) else {
                throw Mismatch.missing(wanted)
            }
            videoVAE = match
        }
        let decoder: LTXComponent = videoVAE.components.contains(.videoVAENADiffusion)
            ? .videoVAENADiffusion : .videoVAECausal

        // Every file states its version; they must agree.
        let set = [transformer, textEncoder, videoVAE, audioVAE]
        var byFile: [String: String] = [:]
        for file in set { if let v = file.modelVersion { byFile[file.name] = v } }
        if Set(byFile.values).count > 1 { throw Mismatch.versionDisagreement(byFile) }

        return Resolved(transformer: transformer, textEncoder: textEncoder,
                        videoVAE: videoVAE, audioVAE: audioVAE,
                        videoDecoder: decoder, modelVersion: byFile.values.first)
    }

    /// Bind four explicit paths, header-only, before any payload is mapped.
    ///
    /// Unlike ``resolve(_:)``, a named file that is the wrong component is refused rather
    /// than skipped, and the NA-diffusion VAE is refused unless `preferNADiffusion` is
    /// set — v1 has no `na3d` kernel, and silently substituting the causal file would
    /// hide that the caller pointed at the other decoder.
    public static func bind(transformer: URL, textEncoder: URL, videoVAE: URL, audioVAE: URL,
                            preferNADiffusion: Bool = false) throws -> Resolved {
        func require(_ url: URL, name: String, component: LTXComponent) throws -> File {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Mismatch.missingFile(name, url.path)
            }
            let file = try identify(fileAt: url)
            guard file.components.contains(component) else {
                throw Mismatch.unexpectedComponent(
                    name: name, expected: component,
                    found: file.components.map(\.rawValue).sorted())
            }
            return file
        }

        let transformerFile = try require(transformer, name: "DiT", component: .transformer)
        let textEncoderFile = try require(textEncoder, name: "text encoder", component: .textEncoder)
        let audioVAEFile = try require(audioVAE, name: "audio VAE", component: .audioVAE)

        guard FileManager.default.fileExists(atPath: videoVAE.path) else {
            throw Mismatch.missingFile("video VAE", videoVAE.path)
        }
        let videoFile = try identify(fileAt: videoVAE)
        let wanted: LTXComponent = preferNADiffusion ? .videoVAENADiffusion : .videoVAECausal
        if !preferNADiffusion, videoFile.components.contains(.videoVAENADiffusion) {
            throw Mismatch.naDiffusionUnsupported(videoVAE.path)
        }
        guard videoFile.components.contains(wanted) else {
            throw Mismatch.unexpectedComponent(
                name: "video VAE", expected: wanted,
                found: videoFile.components.map(\.rawValue).sorted())
        }

        let set = [transformerFile, textEncoderFile, videoFile, audioVAEFile]
        var byFile: [String: String] = [:]
        for file in set { if let v = file.modelVersion { byFile[file.name] = v } }
        if Set(byFile.values).count > 1 { throw Mismatch.versionDisagreement(byFile) }

        return Resolved(transformer: transformerFile, textEncoder: textEncoderFile,
                        videoVAE: videoFile, audioVAE: audioVAEFile,
                        videoDecoder: wanted, modelVersion: byFile.values.first)
    }
}

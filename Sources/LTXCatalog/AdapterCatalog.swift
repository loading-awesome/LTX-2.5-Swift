// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// Finding adapters on disk, and proving they are adapters.
///
/// A recipe names the adapters it wants by **filename hint** rather than by path
/// (`LTXRecipes.AdapterRequest`), because a recipe is a description of a pipeline and a path
/// is a property of one machine. Something has to turn one into the other, and that
/// something has to fail loudly in the two cases that matter: no match, and more than one.
///
/// ## Why matching is by key set and not by name
///
/// The 2.5 tree ships several files that read as adapters and are not. The clearest is the
/// pair in `latent_upscale_models/`: the spatial and temporal x2 upscalers differ by one
/// word in a 46-character filename, are both about the right size, and neither is a LoRA at
/// all. `DistilledRenderer` already had to grow a guard against loading the wrong one.
///
/// So candidacy here is decided by ``LTXCatalog/CheckpointIdentity/components(of:)`` finding
/// `.lora` in the file's own key set — its statement about what it contains — and the hint
/// only chooses among files that already passed that test. A hint that matches a non-adapter
/// therefore reports "no adapter matched", which is true, rather than resolving to a file
/// that fails much later inside `LoRAOverlay.validate`.
///
/// Header-only, so scanning a ~70 GiB tree costs tens of milliseconds.
public enum AdapterCatalog {

    public enum Failure: Error, CustomStringConvertible {
        case noMatch(hint: String, root: String, available: [String])
        case ambiguous(hint: String, matches: [String])
        case notAnAdapter(path: String, found: [String])

        public var description: String {
            switch self {
            case let .noMatch(hint, root, available):
                let list = available.isEmpty
                    ? "no LoRA adapters were found there at all"
                    : "adapters present: " + available.joined(separator: ", ")
                return "no adapter matching '\(hint)' under \(root) — \(list)"
            case let .ambiguous(hint, matches):
                return "'\(hint)' matches \(matches.count) adapters: "
                    + matches.joined(separator: ", ")
                    + ". Name it more precisely, or pass the path outright"
            case let .notAnAdapter(path, found):
                let got = found.isEmpty ? "none of the known LTX components"
                                        : found.joined(separator: ", ")
                return "\(path) holds no LoRA tensors (found \(got)). The spatial and "
                    + "temporal latent upscalers sit beside the adapters under "
                    + "near-identical names and are not adapters"
            }
        }
    }

    /// Every file under `root` whose own keys say it carries LoRA tensors.
    public static func discover(in root: URL) throws -> [CheckpointIdentity.File] {
        try CheckpointIdentity.discover(in: root)
            .filter { $0.components.contains(.lora) }
    }

    /// The one adapter matching `hint`, or a refusal naming what was there instead.
    ///
    /// Matching is a case-insensitive substring of the filename. An exact basename match
    /// wins outright, so a hint that happens to be a prefix of a longer sibling still
    /// resolves rather than reporting an ambiguity the caller cannot act on.
    public static func resolve(hint: String, in root: URL) throws -> CheckpointIdentity.File {
        let adapters = try discover(in: root)
        let needle = hint.lowercased()

        if let exact = adapters.first(where: {
            $0.name.lowercased() == needle
                || $0.url.deletingPathExtension().lastPathComponent.lowercased() == needle
        }) {
            return exact
        }

        let matches = adapters.filter { $0.name.lowercased().contains(needle) }
        guard let first = matches.first else {
            throw Failure.noMatch(hint: hint, root: root.path,
                                  available: adapters.map(\.name).sorted())
        }
        guard matches.count == 1 else {
            throw Failure.ambiguous(hint: hint, matches: matches.map(\.name).sorted())
        }
        return first
    }

    /// Verify a path the caller named outright.
    ///
    /// The hint path is checked for LoRA tensors for the same reason ``resolve(hint:in:)``
    /// filters by them: pointing `--lora` at the temporal upscaler otherwise fails deep
    /// inside a rank check, after the 42 GB transformer has been read.
    public static func verify(at url: URL) throws -> CheckpointIdentity.File {
        let file = try CheckpointIdentity.identify(fileAt: url)
        guard file.components.contains(.lora) else {
            throw Failure.notAnAdapter(path: url.path,
                                       found: file.components.map(\.rawValue).sorted())
        }
        return file
    }

    /// The model tree a checkpoint sits in — `<root>/diffusion_models/x.safetensors` → `<root>`.
    ///
    /// A convention rather than configuration, and it exists so a recipe's adapter hints
    /// resolve without a second flag naming a directory the caller already implied by
    /// pointing at a transformer. When the layout does not match, hint resolution simply
    /// finds nothing and says where it looked.
    public static func modelRoot(containing checkpoint: URL) -> URL {
        checkpoint.deletingLastPathComponent().deletingLastPathComponent()
    }
}

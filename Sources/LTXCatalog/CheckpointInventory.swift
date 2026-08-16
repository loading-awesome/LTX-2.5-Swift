// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation

/// Every configured checkpoint, identified, whether or not it is usable.
///
/// The report-shaped sibling of ``CheckpointIdentity/bind(transformer:textEncoder:videoVAE:audioVAE:preferNADiffusion:)``.
/// `bind` throws on the first thing wrong because a render cannot proceed past it; a
/// diagnostic must not, because **the value of a diagnostic is seeing every problem at
/// once** — one that stops at the first missing file makes you run it four times to find
/// four problems.
///
/// Header-only, so a full inventory of a ~70 GiB tree costs tens of milliseconds.
public enum CheckpointInventory {

    /// What a slot is expected to hold, and what it turned out to hold.
    public struct Row: Sendable {
        public let role: String
        public let path: String
        public let expected: LTXComponent?
        public let result: Result<CheckpointIdentity.File, CheckpointIdentity.Mismatch>
        /// True when the file is fine but is not the thing this slot is for.
        public var isWrongComponent: Bool {
            if case .failure(.unexpectedComponent) = result { return true }
            return false
        }
    }

    public struct Slot: Sendable {
        public let role: String
        public let url: URL
        /// `nil` skips the component cross-check and only proves the file parses.
        public let expected: LTXComponent?
        /// Optional filename fragment the file's name must contain.
        ///
        /// A blunt instrument, used for exactly one thing: the dev and distilled
        /// transformers are **byte-identical in structure** — same 677,616-byte header, same
        /// 4,349 tensors, same `__metadata__` including `model_version` — so nothing in the
        /// file says which one it is. Measured, not assumed. A name check is therefore the
        /// only discrimination available without reading 39 GiB of weights, and a diagnostic
        /// that stayed silent about which transformer is loaded would be silent about the
        /// most consequential thing on the list.
        public let nameMustContain: String?

        public init(role: String, url: URL, expected: LTXComponent? = nil,
                    nameMustContain: String? = nil) {
            self.role = role
            self.url = url
            self.expected = expected
            self.nameMustContain = nameMustContain
        }
    }

    public static func rows(_ slots: [Slot]) -> [Row] {
        slots.map { slot in
            func row(_ result: Result<CheckpointIdentity.File, CheckpointIdentity.Mismatch>) -> Row {
                Row(role: slot.role, path: slot.url.path, expected: slot.expected, result: result)
            }

            guard FileManager.default.fileExists(atPath: slot.url.path) else {
                return row(.failure(.missingFile(slot.role, slot.url.path)))
            }
            guard let file = try? CheckpointIdentity.identify(fileAt: slot.url) else {
                return row(.failure(.unexpectedComponent(name: slot.url.lastPathComponent,
                                                         expected: slot.expected ?? .transformer,
                                                         found: [])))
            }
            if let expected = slot.expected, !file.components.contains(expected) {
                return row(.failure(.unexpectedComponent(
                    name: file.name, expected: expected,
                    found: file.components.map(\.rawValue).sorted())))
            }
            if let fragment = slot.nameMustContain,
               !file.name.localizedCaseInsensitiveContains(fragment) {
                return row(.failure(.wrongVariant(name: file.name, expectedFragment: fragment)))
            }
            return row(.success(file))
        }
    }

    /// Every component present across a set of rows, for ``LTXRecipes/Capability``.
    public static func components(_ rows: [Row]) -> Set<LTXComponent> {
        var found: Set<LTXComponent> = []
        for row in rows { if case let .success(file) = row.result { found.formUnion(file.components) } }
        return found
    }

    /// Whether every file that declares a `model_version` declares the same one.
    ///
    /// Mixing a 2.5 transformer with a VAE from another release is a shape error three
    /// stages later rather than a load failure, which is why it is worth its own line.
    public static func versionDisagreement(_ rows: [Row]) -> [String: String]? {
        var byFile: [String: String] = [:]
        for row in rows {
            if case let .success(file) = row.result, let version = file.modelVersion {
                byFile[file.name] = version
            }
        }
        return Set(byFile.values).count > 1 ? byFile : nil
    }
}

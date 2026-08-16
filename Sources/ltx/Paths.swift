// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXCatalog
import LTXFoundation

/// Config plus flags, reconciled once, with every override said out loud.
///
/// Five subcommands open the same six files. Before there was a config they each carried
/// their own `String = Render.defaultX` defaults, which is five copies of one decision — and
/// the copy that drifts is the one nobody is looking at.
///
/// **Overrides are announced, not silent.** A flag that points somewhere other than the
/// configured path is a legitimate thing to do and a terrible thing to forget: the numbers
/// a benchmark reports only mean something if you know which file produced them, and
/// the dev and distilled transformers are byte-identical in structure, so nothing
/// downstream will notice the swap. One line on stderr costs nothing and closes that gap.
struct Paths {

    let config: LTXConfiguration
    /// Where the config came from, or nil when the built-in defaults are in use.
    let configURL: URL?
    private var announcements: [String] = []

    init(configPath: String? = nil) throws {
        do {
            (config, configURL) = try LTXConfiguration.load(
                from: configPath.map(URL.init(fileURLWithPath:)))
        } catch {
            throw ValidationError("\(error)")
        }
    }

    private init(config: LTXConfiguration, configURL: URL?) {
        self.config = config
        self.configURL = configURL
    }

    /// The built-in paths, for the one caller that must produce a `Paths` without throwing:
    /// `doctor` continuing past an unparseable config so the rest of its report still prints.
    static var builtIn: Paths { Paths(config: .builtIn, configURL: nil) }

    /// Resolve an adapter by hint across every configured adapter root.
    ///
    /// `AdapterCatalog.resolve(hint:in:)` searches one tree. The 2.3 IC-LoRAs live outside
    /// the 2.5 model root, which is exactly why `adapter_roots` exists — so a hint has to be
    /// tried against each of them rather than against the transformer's own directory alone.
    /// Roots are tried in configured order and the first match wins, which is stated here
    /// because "it found a different adapter with the same hint" is otherwise silent.
    mutating func resolveAdapter(hint: String) throws -> URL {
        var roots = config.checkpoints.adapterRoots.map(URL.init(fileURLWithPath:))
        if let dit = config.url(.ditDistilled) ?? config.url(.ditDev) {
            roots.append(AdapterCatalog.modelRoot(containing: dit))
        }
        var attempts: [String] = []
        for root in roots {
            do { return try AdapterCatalog.resolve(hint: hint, in: root).url }
            catch { attempts.append("\(root.path): \(error)") }
        }
        throw ValidationError(
            "no adapter matching '\(hint)' in any configured adapter root. Tried:\n  "
                + attempts.joined(separator: "\n  "))
    }

    /// Resolve one role, preferring an explicit flag and recording that it was used.
    mutating func url(_ role: LTXConfiguration.Role, override: String? = nil) throws -> URL {
        let configured = config.url(role)
        if let override, !override.isEmpty {
            let chosen = URL(fileURLWithPath: override)
            if chosen.standardizedFileURL != configured?.standardizedFileURL {
                announcements.append(
                    "OVERRIDE \(role.flag) \(chosen.path)"
                        + (configured.map { "  (config: \($0.path))" } ?? ""))
            }
            return chosen
        }
        guard let configured else {
            throw ValidationError(
                "no path configured for \(role.label) and none given on the command line. "
                    + "Run `ltx doctor` to write a config, then edit "
                    + "\(LTXConfiguration.defaultURL.path)")
        }
        return configured
    }

    /// Everything worth saying about where these paths came from. Empty when a run used the
    /// configured paths unmodified, so a clean run stays quiet.
    var notices: [String] {
        var out: [String] = []
        if let configURL {
            if !announcements.isEmpty { out.append("config: \(configURL.path)") }
        } else {
            out.append("no config file — using built-in defaults. "
                + "`ltx doctor` writes one to \(LTXConfiguration.defaultURL.path)")
        }
        out.append(contentsOf: announcements)
        return out
    }

    /// Print the notices to stderr, where they cannot contaminate piped output.
    func announce() {
        let lines = notices
        guard !lines.isEmpty else { return }
        FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }
}

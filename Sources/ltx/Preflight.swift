// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import ArgumentParser
import Foundation
import LTXCatalog
import LTXFoundation
import LTXHardware

/// What every command proves before it starts loading weights.
///
/// The checks are `doctor`'s, run silently: the Metal kernels are present, each checkpoint
/// this run needs exists and is the file it claims to be, and the checkpoints agree on a
/// model version. Nothing here reads a payload — `CheckpointInventory` works from
/// safetensors headers — so the whole pass costs a few ranged reads and finishes before
/// the first gigabyte is mapped.
///
/// ## Why this exists rather than letting the load fail
///
/// Both failures it catches are bad at explaining themselves. A missing metallib does not
/// surface as "missing metallib": every checkpoint resolves, the render starts, and the
/// first GPU operation dies inside MLX with an untyped C++ error carrying no path — and it
/// fires at *device resolution*, so it happens even asking for `Device(.cpu)`. A missing
/// checkpoint is friendlier but arrives late, after the text encoder's 26 GB has already
/// been read for a render that was never going to finish.
///
/// ## Every problem at once
///
/// The report lists all of them rather than throwing on the first. Someone setting this up
/// on a new machine usually has several — no config, no kernels, two checkpoints in the
/// wrong place — and finding them one re-run at a time is the difference between a minute
/// and an afternoon. Each carries its own remedy, because "not found" is a statement of
/// the problem and the next line is what the reader actually wants.
enum Preflight {

    /// One thing that is wrong, and what to do about it.
    struct Problem {
        let what: String
        let detail: String
        /// A command to run or a file to edit. Never "check your configuration".
        let remedy: String
    }

    /// Check, and throw a single report naming everything wrong.
    ///
    /// - Parameters:
    ///   - paths: resolved configuration, used to say where a path came from.
    ///   - slots: the checkpoints this particular run opens — not every role that exists.
    ///     A `plain` upscale needs no transformer and must not be told it is missing one.
    ///   - needsMetal: false for a command that only reads headers.
    static func check(paths: Paths,
                      slots: [CheckpointInventory.Slot],
                      needsMetal: Bool = true) throws {
        var problems: [Problem] = []

        if needsMetal, MetalLibrary.locate() == nil {
            problems.append(Problem(
                what: "the MLX Metal kernels are not built",
                detail: "Without them the first GPU operation fails with an untyped C++ "
                    + "error and no path. Looked in:\n      "
                    + MetalLibrary.searchPaths.map(\.path).joined(separator: "\n      "),
                remedy: "./tools/build_mlx_metallib.sh"))
        }

        let rows = CheckpointInventory.rows(slots)
        let failures = rows.compactMap { row -> (role: String, path: String,
                                                 mismatch: CheckpointIdentity.Mismatch)? in
            guard case let .failure(mismatch) = row.result else { return nil }
            return (row.role, row.path, mismatch)
        }

        // A machine that has not been set up yet fails every role for one reason, and
        // saying it five times with five identical remedies buries the one fact that
        // matters: the tree is not where the config says. Collapse that case.
        let allAbsent = !failures.isEmpty && failures.count == rows.count
            && failures.allSatisfy { if case .missingFile = $0.mismatch { true } else { false } }
        if allAbsent, let root = paths.config.checkpoints.root {
            problems.append(Problem(
                what: "no checkpoints found under \(root)",
                detail: "none of \(failures.map(\.role).joined(separator: ", ")) is there. "
                    + "Model weights are not part of this repository; the README lists the "
                    + "files and their sizes.",
                remedy: rootRemedy(paths)))
        } else {
            for failure in failures {
                problems.append(Problem(
                    what: "the \(failure.role) checkpoint is not usable",
                    detail: "\(failure.mismatch)",
                    remedy: configuredHere(paths, role: failure.role)))
            }
        }

        // Only when nothing above already failed: a version disagreement read off a set
        // that is half missing is noise on top of the real problem.
        if problems.isEmpty, let disagreement = CheckpointInventory.versionDisagreement(rows) {
            problems.append(Problem(
                what: "the checkpoints declare different model versions",
                detail: disagreement.sorted { $0.key < $1.key }
                    .map { "\($0.key) = \($0.value)" }.joined(separator: ", "),
                remedy: "point every role at one release; `ltx doctor` prints each file"))
        }

        guard problems.isEmpty else { throw report(problems, paths: paths) }
    }

    /// The remedy when the whole tree is elsewhere: one line naming one key in one file.
    private static func rootRemedy(_ paths: Paths) -> String {
        guard let configURL = paths.configURL else {
            return "run `ltx doctor`, then set `checkpoints.root` in "
                + LTXConfiguration.defaultURL.path
        }
        return "set `checkpoints.root` in \(configURL.path)"
    }

    /// Where a role's path came from, so the remedy names a file the reader can open.
    private static func configuredHere(_ paths: Paths, role: String) -> String {
        guard let configURL = paths.configURL else {
            return "run `ltx doctor`, then set the \(role) entry in "
                + LTXConfiguration.defaultURL.path
        }
        return "set the \(role) entry in \(configURL.path)"
    }

    /// A plain error rather than a `ValidationError`.
    ///
    /// ArgumentParser follows a `ValidationError` with a usage banner, which is right when
    /// the flags were wrong and wrong here: the command line was fine and the machine is
    /// not set up. Printing `Usage: ltx <subcommand>` under a missing-checkpoint report
    /// invites the reader to go looking for the flag they got wrong.
    struct Failed: Error, CustomStringConvertible {
        let description: String
    }

    private static func report(_ problems: [Problem], paths: Paths) -> Failed {
        var lines = [problems.count == 1
            ? "cannot start — 1 problem:"
            : "cannot start — \(problems.count) problems:"]
        for problem in problems {
            lines.append("")
            lines.append("  \(problem.what)")
            lines.append("      \(problem.detail)")
            lines.append("      fix: \(problem.remedy)")
        }
        lines.append("")
        lines.append(paths.configURL == nil
            ? "There is no config file yet. `ltx doctor` writes one and reports the whole "
                + "picture: every checkpoint, the adapters, and the memory plan."
            : "`ltx doctor` reports the whole picture: every checkpoint, the adapters, "
                + "and the memory plan.")
        return Failed(description: lines.joined(separator: "\n"))
    }
}

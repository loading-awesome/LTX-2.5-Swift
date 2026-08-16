// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX
import Testing
@testable import LTXModules

/// Which linears an adapter covers, read from the real checkpoint's header.
///
/// The counts below are the finding, not incidental detail: the **default** target list
/// and the module set of every **released** adapter are different sets, and a trainer that
/// inherits the default produces an adapter shaped unlike anything that shipped. Both
/// numbers are pinned so a change to either is a diff.
@Suite("DiT LoRA targets")
struct DiTLoRATargetsTests {

    static let devTransformer =
        LTXConfiguration.resolved.checkpoints.root! + "/diffusion_models/"
        + "ltx-2.5-22b-dev-transformer-bf16.safetensors"

    static func header() throws -> SafetensorsHeader? {
        guard FileManager.default.fileExists(atPath: devTransformer) else {
            print("SKIP DiTLoRATargets: transformer absent at \(devTransformer)")
            return nil
        }
        return try SafetensorsHeader.read(from: URL(fileURLWithPath: devTransformer))
    }

    /// Header-only, so this costs milliseconds against a 39 GiB file.
    @Test("the default target set matches 1216 weights on this checkpoint")
    func referenceDefaultCount() throws {
        guard let header = try Self.header() else { return }
        let targets = DiTLoRATargets.discover(header: header)
        #expect(targets.count == 1216)
    }

    /// The set every released adapter occupies — 48 blocks x 10 modules.
    @Test("the released-adapter preset reproduces their 480 modules exactly")
    func releasedAdapterCount() throws {
        guard let header = try Self.header() else { return }
        let targets = DiTLoRATargets.discover(
            header: header,
            suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly)
        #expect(targets.count == 480, "\(targets.count) — released adapters report 480")

        // And it really is 48 blocks of 10, not some other factorisation summing to 480.
        let blocks = Set(targets.compactMap(\.block))
        #expect(blocks.count == 48)
        #expect(blocks.min() == 0 && blocks.max() == 47)
        for block in blocks {
            #expect(targets.filter { $0.block == block }.count == 10, "block \(block)")
        }
    }

    /// The default list omits the feed-forward that every released adapter carries. Stated
    /// as a test so the asymmetry cannot be forgotten.
    @Test("the default list omits the feed-forward the released adapters include")
    func defaultOmitsFeedForward() throws {
        guard let header = try Self.header() else { return }
        let byDefault = DiTLoRATargets.discover(header: header, scope: .videoStreamOnly)
        #expect(byDefault.count == 384, "attention only: 48 x 8")
        #expect(!byDefault.contains { $0.key.contains("ff.net") })

        let released = DiTLoRATargets.discover(
            header: header, suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly)
        #expect(released.filter { $0.key.contains("ff.net") }.count == 96, "48 x 2")
    }

    /// The cross-modal bridges are asymmetric, which is why shapes are read rather than
    /// inferred from one hidden size.
    @Test("shapes come from the header, and the bridges are not square")
    func bridgeShapesAreAsymmetric() throws {
        guard let header = try Self.header() else { return }
        let targets = DiTLoRATargets.discover(header: header)
        let bridge = targets.filter { $0.key.contains("audio_to_video_attn") }
        #expect(!bridge.isEmpty)
        #expect(bridge.contains { $0.inFeatures != $0.outFeatures },
                "a bridge projection with equal widths would mean the asymmetry is gone")
    }

    @Test("the trainable parameter count is what the memory estimate is built on")
    func parameterCount() throws {
        guard let header = try Self.header() else { return }
        let released = DiTLoRATargets.discover(
            header: header, suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly)
        let params = DiTLoRATargets.parameterCount(released, rank: 64)
        // 327.2 M at rank 64 -> 1.31 GB fp32, 3.93 GB with AdamW's two moments.
        #expect(params > 320_000_000 && params < 335_000_000,
                "\(params) — was 327.2 M when measured" as Comment)
    }

    @Test("a block range narrows the target set")
    func blockRange() throws {
        guard let header = try Self.header() else { return }
        let firstFour = DiTLoRATargets.discover(
            header: header, suffixes: TrainableLoRA.releasedAdapterTargetModules,
            scope: .videoStreamOnly, blocks: 0 ..< 4)
        #expect(firstFour.count == 40, "4 blocks x 10")
        #expect(firstFour.allSatisfy { ($0.block ?? 99) < 4 })
    }

    @Test("targets are ordered by key, not by dictionary iteration")
    func stableOrder() throws {
        guard let header = try Self.header() else { return }
        let a = DiTLoRATargets.discover(header: header, scope: .videoStreamOnly)
        let b = DiTLoRATargets.discover(header: header, scope: .videoStreamOnly)
        #expect(a.map(\.key) == b.map(\.key))
        #expect(a.map(\.key) == a.map(\.key).sorted())
    }

    @Test("block indices are parsed out of the key")
    func blockParsing() {
        #expect(DiTLoRATargets.blockIndex(
            in: "diffusion_model.transformer_blocks.17.attn1.to_q.weight") == 17)
        #expect(DiTLoRATargets.blockIndex(in: "diffusion_model.patchify_proj.weight") == nil)
    }
}

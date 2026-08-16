// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import MLX
import Testing

@testable import LTXCatalog
@testable import LTXFoundation
@testable import LTXModules

/// LoRA overlay loading and arithmetic.
///
/// **What this suite can and cannot prove.** The overlay arithmetic is checked against a
/// hand-computed example, so `((x @ Aᵀ) @ Bᵀ)·s` is verified as *that expression*. Nothing
/// here shows what applying a real adapter to the real DiT does to an image: every
/// real-file test below is structural — counts, ranks, versions, key resolution — and a
/// wrong *convention* would pass all of them.
///
/// The tests that touch checkpoints print a loud `SKIP` and pass on a machine without
/// them, in the suite's usual style.
@Suite("LoRA overlay")
struct LoRAOverlayTests {

    // MARK: - Fixtures

    private static let root = LTXConfiguration.resolved.checkpoints.root!
    private static let modelsRoot = LTXConfiguration.resolvedModelsRoot

    static let upscaler25 =
        LoRAOverlayTests.root + "/ic-lora/pixel-spatial-upscaler/"
        + "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    static let ingredients23 =
        LoRAOverlayTests.modelsRoot + "/ic-lora-2.3/creative-lab/ingredients/"
        + "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"
    static let inOutpainting23 =
        LoRAOverlayTests.modelsRoot + "/ic-lora-2.3/in-outpainting/"
        + "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
    static let distilled25 =
        LoRAOverlayTests.root + "/loras/"
        + "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"

    /// The base 2.5 transformer's own `model_version`, read from its metadata.
    static let baseVersion = "2.5.0"
    static let basePrefix = TransformerTopology.comfyPrefix

    static func url(_ path: String) -> URL { URL(fileURLWithPath: path) }

    static func present(_ path: String, _ what: String) -> Bool {
        if FileManager.default.fileExists(atPath: path) { return true }
        print("SKIP LoRAOverlay/\(what): adapter absent at \(path)")
        return false
    }

    // MARK: - Synthetic factors

    /// `A = [[1,2,3],[4,5,6]]`, rank 2 over a 3-wide input.
    static func factorA() -> MLXArray { MLXArray([1, 2, 3, 4, 5, 6] as [Float], [2, 3]) }
    /// `B = [[1,-1],[2,3]]`, 2-wide output from rank 2.
    static func factorB() -> MLXArray { MLXArray([1, -1, 2, 3] as [Float], [2, 2]) }
    /// `x = [[1,0,-1],[0,1,2]]`.
    static func input() -> MLXArray { MLXArray([1, 0, -1, 0, 1, 2] as [Float], [2, 3]) }

    static let handModule = "diffusion_model.transformer_blocks.0.attn1.to_q"
    static var handBaseKey: String {
        basePrefix + "transformer_blocks.0.attn1.to_q.weight"
    }

    static func handTensors(module: String = handModule) -> [String: MLXArray] {
        [module + ".lora_A.weight": factorA(), module + ".lora_B.weight": factorB()]
    }

    static func values(_ array: MLXArray) -> [Float] {
        array.asType(.float32).flattened().asArray(Float.self)
    }

    // MARK: - The arithmetic

    /// The core test: everything else in the file is plumbing around this expression.
    ///
    /// ```
    /// A  = [[1,2,3],                B  = [[ 1,-1],        x = [[1, 0,-1],
    ///       [4,5,6]]                      [ 2, 3]]             [0, 1, 2]]
    ///
    /// h  = x @ Aᵀ   row0: [1·1+0·2−1·3, 1·4+0·5−1·6] = [−2, −2]
    ///               row1: [0·1+1·2+2·3, 0·4+1·5+2·6] = [ 8, 17]
    ///
    /// Δ  = h @ Bᵀ   row0: [−2·1 + −2·(−1), −2·2 + −2·3] = [ 0, −10]
    ///               row1: [ 8·1 + 17·(−1),  8·2 + 17·3] = [−9,  67]
    /// ```
    ///
    /// fp32 factors, so every intermediate is exact and the comparison can be equality.
    @Test("overlay delta matches the hand-computed value")
    func overlayMathMatchesHandComputation() throws {
        let overlay = try LoRAOverlay.compose(name: "hand.safetensors",
                                              tensors: Self.handTensors(),
                                              basePrefix: Self.basePrefix)
        let delta = try #require(overlay.delta(forWeightKey: Self.handBaseKey,
                                               input: Self.input()))
        #expect(delta.shape == [2, 2])
        #expect(Self.values(delta) == [0, -10, -9, 67])
    }

    @Test("no overlay for an untouched weight")
    func untouchedWeightReturnsNil() throws {
        let overlay = try LoRAOverlay.compose(name: "hand.safetensors",
                                              tensors: Self.handTensors(),
                                              basePrefix: Self.basePrefix)
        let other = Self.basePrefix + "transformer_blocks.0.attn1.to_k.weight"
        #expect(overlay.delta(forWeightKey: other, input: Self.input()) == nil)
        #expect(overlay.overlays(weightKey: other) == false)
        #expect(overlay.overlays(weightKey: Self.handBaseKey))
    }

    // MARK: - Strength

    /// `strength` is folded into `B` at load, so this is really a test that folding is
    /// linear and that `0` annihilates rather than merely shrinking.
    @Test("strength 0 is exactly zero and strength 2 is exactly twice strength 1")
    func strengthFoldingIsExactlyLinear() throws {
        func delta(_ strength: Float) throws -> [Float] {
            let overlay = try LoRAOverlay.compose(name: "hand.safetensors",
                                                  tensors: Self.handTensors(),
                                                  strength: strength,
                                                  basePrefix: Self.basePrefix)
            return Self.values(try #require(overlay.delta(forWeightKey: Self.handBaseKey,
                                                          input: Self.input())))
        }

        let zero = try delta(0)
        #expect(zero == [0, 0, 0, 0])

        let one = try delta(1)
        let two = try delta(2)
        let half = try delta(0.5)
        #expect(one == [0, -10, -9, 67])
        #expect(two == one.map { $0 * 2 })
        #expect(half == one.map { $0 * 0.5 })
    }

    // MARK: - Key normalisation

    /// All four accepted prefix families name the same base weight.
    ///
    /// The ordering trap is `model.diffusion_model.` versus `model.`: tried in the wrong
    /// order, every 2.5-style key keeps a `diffusion_model.` no base weight has, which is
    /// the mapping that merged nothing and shipped rainbow noise.
    @Test("every accepted prefix family resolves to the same base key")
    func prefixFamiliesCollapse() {
        let expected = Self.basePrefix + "transformer_blocks.0.attn1.to_q.weight"
        let keys = [
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight",
            "transformer.transformer_blocks.0.attn1.to_q.lora_down.weight",
            "model.transformer_blocks.0.attn1.to_q.lora_up",
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A",
        ]
        for key in keys {
            #expect(LoRAOverlay.baseWeightKey(forLoRAKey: key, basePrefix: Self.basePrefix)
                    == expected, "\(key)")
        }
    }

    @Test("an unrecognised prefix resolves to nil rather than a guess")
    func unknownPrefixIsNil() {
        let strangers = [
            "text_encoder.layers.0.self_attn.q_proj.lora_A.weight",
            "lora_unet_transformer_blocks_0_attn1_to_q.lora_down.weight",
            "unet.down_blocks.0.attn.to_q.lora_A.weight",
        ]
        for key in strangers {
            #expect(LoRAOverlay.baseWeightKey(forLoRAKey: key, basePrefix: Self.basePrefix)
                    == nil, "\(key)")
        }
        // Not a LoRA factor at all.
        #expect(LoRAOverlay.baseWeightKey(
            forLoRAKey: "diffusion_model.transformer_blocks.0.attn1.to_q.weight",
            basePrefix: Self.basePrefix) == nil)
        // Suffix-anchored, not substring-replaced: a module whose *name* contains the
        // token must not be mangled.
        #expect(LoRAOverlay.splitFactor("diffusion_model.lora_Adapter.to_q.weight") == nil)
    }

    @Test("the base prefix is whatever the checkpoint uses, including none")
    func strippedPrefixCheckpoint() {
        #expect(LoRAOverlay.baseWeightKey(
            forLoRAKey: "diffusion_model.transformer_blocks.3.ff.net.2.lora_A.weight",
            basePrefix: "") == "transformer_blocks.3.ff.net.2.weight")
    }

    // MARK: - Tripwires

    /// The highest-value guard in the file: an adapter that resolves to nothing must be
    /// an error, not an empty overlay that renders as if it were never passed.
    @Test("an adapter whose keys match nothing is an error, not an empty overlay")
    func zeroMergeTripwireFires() {
        let stranger: [String: MLXArray] = [
            "unet.down_blocks.0.attn.to_q.lora_A.weight": Self.factorA(),
            "unet.down_blocks.0.attn.to_q.lora_B.weight": Self.factorB(),
        ]
        #expect(throws: LoRAOverlay.Failure.self) {
            _ = try LoRAOverlay.compose(name: "stranger.safetensors", tensors: stranger,
                                        basePrefix: Self.basePrefix)
        }
        do {
            _ = try LoRAOverlay.compose(name: "stranger.safetensors", tensors: stranger,
                                        basePrefix: Self.basePrefix)
            Issue.record("expected a throw")
        } catch let failure as LoRAOverlay.Failure {
            guard case .unrecognisedPrefix = failure else {
                Issue.record("expected .unrecognisedPrefix, got \(failure)")
                return
            }
            print("TRIPWIRE unrecognised prefix: \(failure)")
        } catch { Issue.record("unexpected \(error)") }
    }

    /// The same tripwire in its other shape: names that *parse* but land on base weights
    /// this checkpoint does not have. This is the form that needs the base key set, and
    /// it is the one a wiring site should always pass.
    @Test("keys that parse but name no base weight are an error")
    func resolvedButAbsentBaseWeightThrows() {
        let shapes = [Self.basePrefix + "transformer_blocks.0.attn1.to_k.weight": [2, 3]]
        do {
            _ = try LoRAOverlay.compose(name: "wrongmodule.safetensors",
                                        tensors: Self.handTensors(),
                                        basePrefix: Self.basePrefix,
                                        baseWeightShapes: shapes)
            Issue.record("expected a throw")
        } catch let failure as LoRAOverlay.Failure {
            guard case .unknownBaseWeight = failure else {
                Issue.record("expected .unknownBaseWeight, got \(failure)")
                return
            }
            print("TRIPWIRE unknown base weight: \(failure)")
        } catch { Issue.record("unexpected \(error)") }
    }

    @Test("an empty adapter is an error")
    func emptyAdapterThrows() {
        do {
            _ = try LoRAOverlay.compose(name: "empty.safetensors", tensors: [:],
                                        basePrefix: Self.basePrefix)
            Issue.record("expected a throw")
        } catch let failure as LoRAOverlay.Failure {
            guard case .noApplicableModules = failure else {
                Issue.record("expected .noApplicableModules, got \(failure)")
                return
            }
            print("TRIPWIRE zero modules: \(failure)")
        } catch { Issue.record("unexpected \(error)") }
    }

    @Test("a lora_A with no lora_B throws")
    func orphanedFactorThrows() {
        let orphan = [Self.handModule + ".lora_A.weight": Self.factorA()]
        do {
            _ = try LoRAOverlay.compose(name: "orphan.safetensors", tensors: orphan,
                                        basePrefix: Self.basePrefix)
            Issue.record("expected a throw")
        } catch let failure as LoRAOverlay.Failure {
            guard case .orphanedFactor = failure else {
                Issue.record("expected .orphanedFactor, got \(failure)")
                return
            }
            print("TRIPWIRE orphan A: \(failure)")
        } catch { Issue.record("unexpected \(error)") }
    }

    @Test("a lora_B with no lora_A throws")
    func orphanedUpFactorThrows() {
        let orphan = [Self.handModule + ".lora_B.weight": Self.factorB()]
        #expect(throws: LoRAOverlay.Failure.self) {
            _ = try LoRAOverlay.compose(name: "orphan.safetensors", tensors: orphan,
                                        basePrefix: Self.basePrefix)
        }
    }

    @Test("a transposed factor pair throws rather than loading")
    func rankDisagreementThrows() {
        let bad = [
            Self.handModule + ".lora_A.weight": Self.factorA(),           // [2, 3]
            Self.handModule + ".lora_B.weight": MLXArray([1, 2, 3] as [Float], [1, 3]),
        ]
        do {
            _ = try LoRAOverlay.compose(name: "transposed.safetensors", tensors: bad,
                                        basePrefix: Self.basePrefix)
            Issue.record("expected a throw")
        } catch let failure as LoRAOverlay.Failure {
            guard case .rankDisagreement = failure else {
                Issue.record("expected .rankDisagreement, got \(failure)")
                return
            }
        } catch { Issue.record("unexpected \(error)") }
    }

    @Test("a DoRA magnitude vector is refused rather than silently dropped")
    func unclassifiedTensorThrows() {
        var tensors = Self.handTensors()
        tensors[Self.handModule + ".lora_magnitude_vector"] = MLXArray([1] as [Float], [1])
        #expect(throws: LoRAOverlay.Failure.self) {
            _ = try LoRAOverlay.compose(name: "dora.safetensors", tensors: tensors,
                                        basePrefix: Self.basePrefix)
        }
    }

    @Test("a per-module alpha is refused rather than ignored")
    func perModuleAlphaThrows() {
        var tensors = Self.handTensors()
        tensors[Self.handModule + ".alpha"] = MLXArray([8] as [Float], [1])
        #expect(throws: LoRAOverlay.Failure.self) {
            _ = try LoRAOverlay.compose(name: "kohya.safetensors", tensors: tensors,
                                        basePrefix: Self.basePrefix)
        }
    }

    // MARK: - Bucket filters

    /// A synthetic adapter that carries one module from each bucket, so the filters can
    /// be checked without a 300 MB file. Module names are the measured ones from the 2.5
    /// distilled adapter.
    static func bucketTensors() -> [String: MLXArray] {
        let modules = [
            "diffusion_model.transformer_blocks.0.attn1.to_q",            // video
            "diffusion_model.transformer_blocks.0.attn2.to_k",            // video
            "diffusion_model.transformer_blocks.0.ff.net.2",              // video
            "diffusion_model.transformer_blocks.0.audio_attn1.to_q",      // audio
            "diffusion_model.transformer_blocks.0.audio_ff.net.2",        // audio
            "diffusion_model.transformer_blocks.0.audio_to_video_attn.to_q",  // bridge
            "diffusion_model.transformer_blocks.0.video_to_audio_attn.to_k",  // bridge
            "diffusion_model.av_ca_v2a_gate_adaln_single.linear",         // bridge
            "diffusion_model.patchify_proj",                              // video
            "diffusion_model.audio_patchify_proj",                        // audio
        ]
        var tensors: [String: MLXArray] = [:]
        for module in modules {
            tensors[module + ".lora_A.weight"] = factorA()
            tensors[module + ".lora_B.weight"] = factorB()
        }
        return tensors
    }

    @Test("bucket filters select the expected module counts")
    func bucketFiltersSelect() throws {
        func count(_ filter: LoRAOverlay.ModuleFilter) throws -> Int {
            try LoRAOverlay.compose(name: "buckets.safetensors", tensors: Self.bucketTensors(),
                                    filter: filter, basePrefix: Self.basePrefix).moduleCount
        }
        #expect(try count(.all) == 10)
        #expect(try count(.videoStream) == 4)          // attn1, attn2, ff, patchify_proj
        #expect(try count(.audioStream) == 3)          // audio_attn1, audio_ff, audio_patchify
        #expect(try count(.crossModalBridge) == 3)     // a2v, v2a, av_ca_
        #expect(try count(.attentionOnly) == 5)        // attn1, attn2, audio_attn1, a2v, v2a
        #expect(try count(.feedForwardOnly) == 2)      // ff.net.2, audio_ff.net.2

        // The three buckets partition the adapter, which is the property a recipe
        // depends on when it assigns a strength per bucket.
        #expect(try count(.videoStream) + count(.audioStream) + count(.crossModalBridge)
                == count(.all))

        // A hand-written filter, since that is what a recipe will actually carry.
        let firstBlockSelfAttention = LoRAOverlay.ModuleFilter(
            include: ["transformer_blocks.0.attn1."], exclude: [])
        #expect(try count(firstBlockSelfAttention) == 1)

        // Exclude wins over include.
        let contradiction = LoRAOverlay.ModuleFilter(include: ["attn"], exclude: ["attn"])
        #expect(throws: LoRAOverlay.Failure.self) { _ = try count(contradiction) }
    }

    @Test("a filter that excludes everything is an error, not an empty overlay")
    func emptyFilterThrows() {
        do {
            _ = try LoRAOverlay.compose(name: "buckets.safetensors",
                                        tensors: Self.bucketTensors(),
                                        filter: .init(include: ["no_such_module"]),
                                        basePrefix: Self.basePrefix)
            Issue.record("expected a throw")
        } catch let failure as LoRAOverlay.Failure {
            guard case .filterSelectedNothing = failure else {
                Issue.record("expected .filterSelectedNothing, got \(failure)")
                return
            }
            print("TRIPWIRE empty filter: \(failure)")
        } catch { Issue.record("unexpected \(error)") }
    }

    // MARK: - Version gating

    @Test("model_version agreement is on major and minor")
    func versionComparison() {
        #expect(LoRAOverlay.versionsAgree("2.5", "2.5.0"))
        #expect(LoRAOverlay.versionsAgree("2.5.0", "2.5"))
        #expect(LoRAOverlay.versionsAgree("2.3", "2.3.0"))
        #expect(!LoRAOverlay.versionsAgree("2.3", "2.5.0"))
        #expect(!LoRAOverlay.versionsAgree("2.5", "3.5"))
    }

    @Test("a missing model_version is refused under .require")
    func missingVersionRefused() {
        #expect(throws: LoRAOverlay.Failure.self) {
            _ = try LoRAOverlay.compose(name: "anon.safetensors", tensors: Self.handTensors(),
                                        versionPolicy: .require(base: Self.baseVersion),
                                        basePrefix: Self.basePrefix)
        }
        // …and allowed once the caller says so in its own source.
        #expect(throws: Never.self) {
            _ = try LoRAOverlay.compose(
                name: "anon.safetensors", tensors: Self.handTensors(),
                versionPolicy: .acceptVersionMismatch(base: Self.baseVersion),
                basePrefix: Self.basePrefix)
        }
    }

    // MARK: - Stacking

    /// Two adapters on the same linear sum, and the sum is what a bake of both would be.
    @Test("stacked adapters sum, and the order does not change the value")
    func stackingSums() throws {
        let first = Self.handTensors()
        var second = Self.handTensors()
        second[Self.handModule + ".lora_B.weight"] = MLXArray([2, 0, 0, 1] as [Float], [2, 2])

        func delta(_ tensors: [[String: MLXArray]]) throws -> [Float] {
            var sum: [Float] = []
            for t in tensors {
                let overlay = try LoRAOverlay.compose(name: "stack.safetensors", tensors: t,
                                                      basePrefix: Self.basePrefix)
                let d = Self.values(try #require(
                    overlay.delta(forWeightKey: Self.handBaseKey, input: Self.input())))
                sum = sum.isEmpty ? d : zip(sum, d).map(+)
            }
            return sum
        }

        let single = try LoRAOverlay.compose(name: "a.safetensors", tensors: first,
                                             basePrefix: Self.basePrefix)
        #expect(single.overlayCount(forWeightKey: Self.handBaseKey) == 1)

        let separate = try delta([first, second])
        let reversed = try delta([second, first])
        #expect(separate == reversed)

        // Second adapter alone: h @ [[2,0],[0,1]]ᵀ = [[-4,-2],[16,17]].
        let secondOverlay = try LoRAOverlay.compose(name: "b.safetensors", tensors: second,
                                                    basePrefix: Self.basePrefix)
        let secondDelta = Self.values(try #require(
            secondOverlay.delta(forWeightKey: Self.handBaseKey, input: Self.input())))
        #expect(secondDelta == [-4, -2, 16, 17])
        let expectedSum: [Float] = [-4, -12, 7, 84]     // [0,-10,-9,67] + [-4,-2,16,17]
        #expect(separate == expectedSum)
    }

    /// The real stacking path, through `load`, using two copies of the same file at
    /// different strengths. Guarded because it needs an adapter on disk.
    @Test("load stacks two adapters onto one weight")
    func loadStacksAdapters() throws {
        guard Self.present(Self.upscaler25, "loadStacksAdapters") else { return }
        let overlay = try LoRAOverlay.load(
            [.init(url: Self.url(Self.upscaler25), strength: 1.0,
                   filter: .init(include: ["transformer_blocks.0.attn1.to_q"]),
                   versionPolicy: .require(base: Self.baseVersion)),
             .init(url: Self.url(Self.upscaler25), strength: 0.5,
                   filter: .init(include: ["transformer_blocks.0.attn1.to_q"]),
                   versionPolicy: .require(base: Self.baseVersion))],
            basePrefix: Self.basePrefix)

        let key = Self.basePrefix + "transformer_blocks.0.attn1.to_q.weight"
        #expect(overlay.moduleCount == 1)
        #expect(overlay.overlayCount == 2)
        #expect(overlay.overlayCount(forWeightKey: key) == 2)

        // 1.0 + 0.5 of the same adapter must be 1.5 of it, up to bf16 accumulation.
        let raw = (0..<4096).map { Float(sin(Double($0) * 0.017)) * 0.05 }
        let x = MLXArray(raw, [1, 4096]).asType(.bfloat16)
        let stacked = Self.values(try #require(overlay.delta(forWeightKey: key, input: x)))

        let single = try LoRAOverlay.load(
            url: Self.url(Self.upscaler25), strength: 1.5,
            filter: .init(include: ["transformer_blocks.0.attn1.to_q"]),
            versionPolicy: .require(base: Self.baseVersion), basePrefix: Self.basePrefix)
        let once = Self.values(try #require(single.delta(forWeightKey: key, input: x)))

        let worst = zip(stacked, once).map { abs($0 - $1) }.max() ?? 0
        let scale = once.map { abs($0) }.max() ?? 1
        print("STACK 1.0+0.5 vs 1.5: max_abs \(worst), scale \(scale)")
        #expect(worst <= scale * 0.02)
    }

    // MARK: - Real adapters

    /// The 2.5 spatial upscaler IC-LoRA: 480 module pairs across 48 blocks, rank 32.
    @Test("the 2.5 spatial upscaler IC-LoRA resolves to 480 modules across 48 blocks")
    func upscalerResolves() throws {
        guard Self.present(Self.upscaler25, "upscalerResolves") else { return }
        let overlay = try LoRAOverlay.load(url: Self.url(Self.upscaler25),
                                           versionPolicy: .require(base: Self.baseVersion),
                                           basePrefix: Self.basePrefix)
        let report = try #require(overlay.reports.first)
        print("ADAPTER \(report)")
        for (module, count) in report.moduleShapes.sorted(by: { $0.key < $1.key }) {
            print("   \(count)  \(module)")
        }
        #expect(report.modelVersion == "2.5")
        #expect(report.tensorCount == 960)
        #expect(report.candidateCount == 480)
        #expect(report.moduleCount == 480)
        #expect(report.blockIndices.count == 48)
        #expect(report.blockIndices == Set(0..<48))
        #expect(report.ranksByModuleCount == [32: 480])
        #expect(report.audioModuleCount == 0)
        #expect(report.bridgeModuleCount == 0)
        #expect(report.moduleShapes.count == 10)
        #expect(overlay.moduleCount == 480)
        // The established 10 modules per block.
        let expected = ["transformer_blocks.N.attn1.to_q", "transformer_blocks.N.attn1.to_k",
                        "transformer_blocks.N.attn1.to_v", "transformer_blocks.N.attn1.to_out.0",
                        "transformer_blocks.N.attn2.to_q", "transformer_blocks.N.attn2.to_k",
                        "transformer_blocks.N.attn2.to_v", "transformer_blocks.N.attn2.to_out.0",
                        "transformer_blocks.N.ff.net.0.proj", "transformer_blocks.N.ff.net.2"]
        for module in expected {
            #expect(report.moduleShapes[module] == 48, "\(module)")
        }
    }

    /// The 2.3 IC-LoRAs: same 480 modules, rank 128, and a version this base is not.
    @Test("the 2.3 IC-LoRAs resolve to 480 modules and are refused against a 2.5 base")
    func ingredientsResolvesAndIsVersionGated() throws {
        for path in [Self.ingredients23, Self.inOutpainting23] {
            guard Self.present(path, "ingredientsResolvesAndIsVersionGated") else { continue }

            // Refused without the opt-in.
            do {
                _ = try LoRAOverlay.load(url: Self.url(path),
                                         versionPolicy: .require(base: Self.baseVersion),
                                         basePrefix: Self.basePrefix)
                Issue.record("expected a version refusal for \(path)")
            } catch let failure as LoRAOverlay.Failure {
                guard case .versionMismatch = failure else {
                    Issue.record("expected .versionMismatch, got \(failure)")
                    continue
                }
                print("VERSION GATE: \(failure)")
            }

            // Accepted with it.
            let overlay = try LoRAOverlay.load(
                url: Self.url(path),
                versionPolicy: .acceptVersionMismatch(base: Self.baseVersion),
                basePrefix: Self.basePrefix)
            let report = try #require(overlay.reports.first)
            print("ADAPTER \(report)")
            #expect(report.modelVersion == "2.3")
            #expect(report.candidateCount == 480)
            #expect(report.moduleCount == 480)
            #expect(report.blockIndices == Set(0..<48))
            #expect(report.ranksByModuleCount == [128: 480])
            #expect(report.audioModuleCount == 0)
            #expect(report.bridgeModuleCount == 0)

            // And the audio bucket is empty on every IC-LoRA measured so far, which the
            // tripwire reports rather than returning an overlay that does nothing.
            #expect(throws: LoRAOverlay.Failure.self) {
                _ = try LoRAOverlay.load(
                    url: Self.url(path), filter: .audioStream,
                    versionPolicy: .acceptVersionMismatch(base: Self.baseVersion),
                    basePrefix: Self.basePrefix)
            }

            // Attention-only is 8 of the 10 modules per block.
            let attention = try LoRAOverlay.load(
                url: Self.url(path), filter: .attentionOnly,
                versionPolicy: .acceptVersionMismatch(base: Self.baseVersion),
                basePrefix: Self.basePrefix)
            #expect(attention.moduleCount == 8 * 48)
        }
    }

    /// The 8.9 GB distilled LoRA. Loading is lazy — mlx maps the file and materialises
    /// nothing until an array is used — and this test reads only names, shapes and
    /// metadata, so it costs a header parse rather than 8.9 GB of I/O.
    @Test("the 2.5 distilled LoRA's inventory")
    func distilledInventory() throws {
        guard Self.present(Self.distilled25, "distilledInventory") else { return }
        let started = Date()
        let overlay = try LoRAOverlay.load(url: Self.url(Self.distilled25),
                                           versionPolicy: .require(base: Self.baseVersion),
                                           basePrefix: Self.basePrefix)
        let report = try #require(overlay.reports.first)
        print("ADAPTER \(report)")
        print("   metadata: \(report.metadata)")
        print("   load took \(String(format: "%.2f", Date().timeIntervalSince(started))) s")
        for (module, count) in report.moduleShapes.sorted(by: { $0.key < $1.key }) {
            print("   \(count)  \(module)")
        }

        #expect(report.modelVersion == "2.5.0")
        #expect(report.tensorCount == 3320)
        #expect(report.candidateCount == 1660)
        #expect(report.blockIndices == Set(0..<48))
        // Unlike every IC-LoRA, this one covers the audio tower and both bridges.
        #expect(report.audioModuleCount > 0)
        #expect(report.bridgeModuleCount > 0)

        for filter in [LoRAOverlay.ModuleFilter.videoStream, .audioStream, .crossModalBridge,
                       .attentionOnly, .feedForwardOnly] {
            let bucket = try LoRAOverlay.load(
                url: Self.url(Self.distilled25), filter: filter,
                versionPolicy: .require(base: Self.baseVersion), basePrefix: Self.basePrefix)
            let r = try #require(bucket.reports.first)
            print("   BUCKET [\(filter)] -> \(r.moduleCount) modules, "
                  + "ranks \(r.ranksByModuleCount.sorted { $0.key < $1.key })")
        }
    }

    /// The overlay against the real transformer's key set — the only check in this file
    /// that can catch an adapter whose naming is self-consistent and wrong.
    ///
    /// It is the structural check with the most teeth: every resolved key must exist in
    /// the 4349-tensor checkpoint *and* its base weight must be exactly
    /// `[out, in]` for the adapter's factors. The distilled adapter is the interesting
    /// case, because it reaches 62 distinct module shapes including `to_gate_logits`,
    /// which no IC-LoRA touches.
    @Test("resolved keys all exist in the 2.5 transformer with matching shapes")
    func resolvedKeysExistInTheCheckpoint() throws {
        let ditPath = Self.root + "/diffusion_models/"
            + "ltx-2.5-22b-dev-transformer-bf16.safetensors"
        guard Self.present(ditPath, "resolvedKeysExistInTheCheckpoint") else { return }

        let header = try SafetensorsHeader.read(from: URL(fileURLWithPath: ditPath))
        #expect(header.metadata["model_version"] == Self.baseVersion)
        var shapes: [String: [Int]] = [:]
        for (name, entry) in header.tensors { shapes[name] = entry.shape }

        for (path, expected) in [(Self.upscaler25, 480), (Self.distilled25, 1660)] {
            guard Self.present(path, "resolvedKeysExistInTheCheckpoint") else { continue }
            let overlay = try LoRAOverlay.load(url: Self.url(path),
                                               versionPolicy: .require(base: Self.baseVersion),
                                               basePrefix: Self.basePrefix,
                                               baseWeightShapes: shapes)
            #expect(overlay.moduleCount == expected)
            print("BASE KEY CHECK \(URL(fileURLWithPath: path).lastPathComponent): "
                  + "\(overlay.moduleCount)/\(expected) resolved keys present, shapes match")
        }
    }
}

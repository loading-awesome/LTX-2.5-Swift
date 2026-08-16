// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation

// Capability-aware recipe resolution: the shape rule, the ladder and the request type.
// `Recipe` itself and the registry of named pipelines live in `RecipeRegistry.swift`.
//
// `docs/RECIPES.md` is the reader-facing version. The short version here:
//
// - Dimensions land on a grid of **64**, for every recipe. The VAE downsamples by 32, but
//   the latent upsampler doubles spatially, so a size that can be upscaled must already be
//   a multiple of 64 — and one rule for both kinds of recipe beats two.
// - The megapixel target is a **caller parameter**, not a baked-in preset size, so the
//   ladder is a rule rather than a table. The table is kept anyway, because a hand-pinned
//   rung (640x384, the one production-scale shape with measurements behind it) has to be
//   able to win over the derived one.
// - Evidence is **derived from the resolved shape**, not declared by the pipeline. The
//   measurements behind this port are at 640x384x97 @ 24 fps and 320x192x25 @ 24 fps.
//   Nothing else on any ladder has been measured, and the menu has to say so — which is
//   why `Recipe` carries `Gate`s and computes an `Evidence` per resolution rather than
//   holding one.
//
// No MLX here, and no hardware probe: `LTXHardware` owns memory detection
// (`Package.swift`), and a capability answer computed in this target would make that
// boundary decorative. `Capability` is therefore a value a caller supplies.

// MARK: - The grid

/// The pixel grid a render's dimensions must land on.
///
/// 64, not 32. The VAE still downsamples by 32, but the latent upsampler doubles
/// spatially, so a size that can be upscaled — and every recipe, including
/// single-stage, so the ladder is one rule — must already be a multiple of 64.
/// 512×288 is exact 16:9 on the VAE grid and is refused here.
public enum Grid: Sendable, CaseIterable {
    case single
    case twoStage

    public var value: Int { 64 }
}

public enum AspectRatio: String, Sendable, CaseIterable {
    case r16x9 = "16:9"
    case r9x16 = "9:16"
    case r1x1 = "1:1"
    case r21x9 = "21:9"
    case r4x3 = "4:3"
    case r3x4 = "3:4"

    /// Width over height.
    public var ratio: Double {
        switch self {
        case .r16x9: 16.0 / 9.0
        case .r9x16: 9.0 / 16.0
        case .r1x1: 1.0
        case .r21x9: 21.0 / 9.0
        case .r4x3: 4.0 / 3.0
        case .r3x4: 3.0 / 4.0
        }
    }

    /// The smallest integer `(w, h)` in exactly this ratio, in grid cells.
    ///
    /// Exact 16:9 on this 64-grid is `(1024k, 576k)`; exact 9:16 is `(576k, 1024k)`.
    /// The ladder is coarse, coarsest in portrait, which is why
    /// ``Ladder/nearestExact(aspect:grid:near:)`` exists.
    public var cells: (width: Int, height: Int) {
        switch self {
        case .r16x9: (16, 9)
        case .r9x16: (9, 16)
        case .r1x1: (1, 1)
        case .r21x9: (7, 3)
        case .r4x3: (4, 3)
        case .r3x4: (3, 4)
        }
    }
}

/// Integer pixel dimensions, and what they actually deliver.
public struct Shape: Sendable, Equatable, Hashable {
    public let width: Int
    public let height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public var pixels: Int { width * height }
    public var megapixels: Double { Double(pixels) / 1e6 }
    public var aspect: Double { Double(width) / Double(height) }

    public func isLegal(on grid: Grid) -> Bool {
        width % grid.value == 0 && height % grid.value == 0
            && width >= 2 * grid.value && height >= 2 * grid.value
    }

    /// The transpose. Portrait rungs are landscape rungs transposed, never re-derived —
    /// a portrait entry that disagreed with its landscape one would be a plausible size
    /// with nothing behind it.
    public var transposed: Shape { Shape(width: height, height: width) }

    public var description: String { "\(width)x\(height)" }
}

/// One rung: what was asked for, and what the grid could actually give.
public struct LadderRung: Sendable, Equatable {
    /// The nominal target this rung was derived from — **not what it delivers**.
    public let requestedMegapixels: Double
    public let requestedAspect: AspectRatio
    public let shape: Shape
    public let grid: Grid
    /// True when the shape was pinned by hand rather than derived, because it is a shape
    /// with evidence behind it. 640x384 is the only one today.
    public let isPinned: Bool

    public init(requestedMegapixels: Double, requestedAspect: AspectRatio,
                shape: Shape, grid: Grid, isPinned: Bool = false) {
        self.requestedMegapixels = requestedMegapixels
        self.requestedAspect = requestedAspect
        self.shape = shape
        self.grid = grid
        self.isPinned = isPinned
    }

    /// Signed relative error in delivered megapixels. Can exceed 10% at small sizes.
    public var megapixelError: Double {
        (shape.megapixels - requestedMegapixels) / requestedMegapixels
    }

    /// Signed relative error in delivered aspect ratio.
    public var aspectError: Double {
        (shape.aspect - requestedAspect.ratio) / requestedAspect.ratio
    }
}

public enum Ladder {

    /// Half-up snapping to the grid, floored at one cell.
    ///
    /// Half-up rather than Swift's `rounded()` or Python's banker's `round`: the committed
    /// shape table was built half-up, and its 1280x736 rung depends on `720 / 32 = 22.5`
    /// going to 23. A different tie rule silently produces a different ladder.
    public static func snap(_ value: Double, to grid: Grid) -> Int {
        let g = Double(grid.value)
        return max(grid.value, Int((value / g + 0.5).rounded(.down)) * grid.value)
    }

    /// Width first from the megapixel target, height derived from the aspect.
    ///
    /// Deriving both from the target independently lets the aspect drift much further than
    /// deriving one from the other; this is the rule the committed table was built with.
    /// A pinned shape at the same request wins — see ``pinned``.
    public static func rung(megapixels: Double, aspect: AspectRatio,
                            grid: Grid) throws -> LadderRung {
        guard megapixels > 0 else {
            throw RecipeFailure.belowGridFloor(width: 0, height: 0, grid: grid)
        }
        let width = snap((megapixels * 1e6 * aspect.ratio).squareRoot(), to: grid)
        let height = snap(Double(width) / aspect.ratio, to: grid)
        let shape = Shape(width: width, height: height)
        guard shape.isLegal(on: grid) else {
            throw RecipeFailure.belowGridFloor(width: width, height: height, grid: grid)
        }
        if let pin = pinned.first(where: { $0.aspectFamily == aspect
            && $0.shape.isLegal(on: grid)
            && abs($0.shape.megapixels - megapixels) <= abs(shape.megapixels - megapixels) }) {
            return LadderRung(requestedMegapixels: megapixels, requestedAspect: aspect,
                              shape: pin.shape, grid: grid, isPinned: true)
        }
        return LadderRung(requestedMegapixels: megapixels, requestedAspect: aspect,
                          shape: shape, grid: grid)
    }

    /// The nearest shape in *exactly* this aspect ratio, for the refusal message when the
    /// grid cannot express what was asked for.
    public static func nearestExact(aspect: AspectRatio, grid: Grid,
                                    near megapixels: Double) -> Shape? {
        let cells = aspect.cells
        let g = grid.value
        var best: Shape?
        var bestError = Double.greatestFiniteMagnitude
        for k in 1 ... 64 {
            let shape = Shape(width: cells.width * g * k, height: cells.height * g * k)
            guard shape.isLegal(on: grid) else { continue }
            let error = abs(shape.megapixels - megapixels)
            if error < bestError { bestError = error; best = shape }
        }
        return best
    }

    /// A shape chosen by hand because it has evidence, not because the rule produced it.
    public struct Pin: Sendable, Equatable {
        public let shape: Shape
        public let aspectFamily: AspectRatio
        public let why: String
    }

    /// Today: one shape and its transpose. Every measurement at production scale is at
    /// 640x384, and `Renderer.Spec.productionGeometry` defaults to it.
    ///
    /// Its aspect is 1.667, not 1.778. The 64-grid has nothing closer to 16:9 at that size,
    /// and the pin says so rather than rounding the claim.
    public static let pinned: [Pin] = [
        Pin(shape: Shape(width: 640, height: 384), aspectFamily: .r16x9,
            why: "the shape every measured gate in this repo is at; aspect 1.667, not 1.778"),
        Pin(shape: Shape(width: 384, height: 640), aspectFamily: .r9x16,
            why: "the transpose of the measured shape; a portrait draft bucket"),
    ]

    /// The committed shape ladder, filtered to conform to the 64-divisibility requirement.
    ///
    /// Only three rungs of the full table are multiples of 64 on both axes: 1152x640, 1344x768, and 1920x1088.
    public static let committedRungsOn64: [LadderRung] = [
        (0.7, 1152, 640),
        (0.98, 1344, 768),
        (2.0, 1920, 1088),
    ].map { LadderRung(requestedMegapixels: $0.0, requestedAspect: .r16x9,
                       shape: Shape(width: $0.1, height: $0.2), grid: .single) }
}

// MARK: - Guidance

/// One stream's guidance, mirrored as a plain value.
///
/// Not `LTXPipeline.GuidanceParams`: that target is MLX-bound and this one must stay free
/// of it. The neutral values are the trap worth restating — STG is neutral at **0.0**, not
/// 1.0, and modality at **1.0**, not 0.0.
public struct RecipeGuidance: Sendable, Equatable {
    public var cfgScale: Double
    public var stgScale: Double
    public var stgBlocks: [Int]
    public var rescaleScale: Double
    public var modalityScale: Double

    public init(cfgScale: Double, stgScale: Double = 0.0, stgBlocks: [Int] = [],
                rescaleScale: Double = 0.0, modalityScale: Double = 1.0) {
        self.cfgScale = cfgScale
        self.stgScale = stgScale
        self.stgBlocks = stgBlocks
        self.rescaleScale = rescaleScale
        self.modalityScale = modalityScale
    }

    /// `sampler_guidance.video` from the production manifest.
    public static let productionVideo = RecipeGuidance(
        cfgScale: 3.0, stgScale: 1.0, stgBlocks: [28],
        rescaleScale: 0.7, modalityScale: 3.0)

    /// The same shape, a different cfg scale. Dual CFG is two independent scales
    /// (contract 7), never one applied twice.
    public static let productionAudio = RecipeGuidance(
        cfgScale: 7.0, stgScale: 1.0, stgBlocks: [28],
        rescaleScale: 0.7, modalityScale: 3.0)

    /// The distilled pipeline builds no guider at all: CFG 1, nothing else engaged.
    public static let distilled = RecipeGuidance(cfgScale: 1.0)

    /// Production CFG with **nothing else engaged** — two forwards per step, not four.
    ///
    /// The pass count is the cost, and it is set by which terms are live: a step runs
    /// `conditional`, adds `unconditional` when CFG is not 1, adds `perturbed` when STG is
    /// not 0, and adds `modality` when the modality scale is not 1. Zeroing STG and
    /// returning modality to 1 removes two of the four.
    ///
    /// Worth having as its own named pair rather than as a flag because the halving is the
    /// whole reason to run it, and a name records what was run. CFG is the structural term —
    /// it is what makes the render follow the prompt — while STG and modality refine what
    /// CFG already placed. So this is the setting that buys back time without touching
    /// stage resolution, which is the knob that decides whether rigid structure survives.
    public static let cfgOnlyVideo = RecipeGuidance(cfgScale: 3.0)

    /// ``cfgOnlyVideo``'s audio half. Dual CFG is two independent scales (contract 7).
    public static let cfgOnlyAudio = RecipeGuidance(cfgScale: 7.0)
}

/// How a stage gets its sigmas.
public enum SigmaPlan: Sendable, Equatable {
    /// The continuous flow schedule at `steps`, anchored at the default 4096 tokens rather
    /// than the render's own token count.
    case continuous(steps: Int)
    /// A fixed table, `steps + 1` values, descending to 0.
    case fixed([Float])

    public var steps: Int {
        switch self {
        case let .continuous(steps): steps
        case let .fixed(values): max(0, values.count - 1)
        }
    }

    /// The distilled first stage — 8 steps. Identical in LTX 2.3 and 2.5.
    public static let distilledStage1 = SigmaPlan.fixed(
        [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0])

    /// The distilled second stage — 3 steps, starting at 0.909375. Variants of this table
    /// that start at 0.85 exist elsewhere and are deliberately not offered here.
    public static let distilledStage2 = SigmaPlan.fixed([0.909375, 0.725, 0.421875, 0.0])
}

public enum SamplerKind: String, Sendable {
    case euler
    /// Ancestral Euler at `eta = 1.0`, `s_noise = 1.0`. **Not ported.**
    case eulerAncestral
    /// `euler_cfg_pp` — deterministic Euler taking its step *direction* from the
    /// unconditional prediction while keeping the guided one as the destination.
    ///
    /// The guided x0 is an extrapolation away from the unconditional x0, and at a high CFG
    /// scale the plain step inherits the whole of that extrapolation as its direction.
    /// Keeping the destination and dropping the over-shoot is what CFG++ does, and the
    /// artifact it removes — the crushed, over-saturated look at high guidance — is what
    /// "CFG burn" names.
    ///
    /// Costs **nothing**: the unconditional prediction is already computed for CFG itself.
    case eulerCFGPP
    /// `euler_ancestral_cfg_pp` — the above with the ancestral renoise, which is the
    /// sampler the shipped Ingredients workflow actually names.
    case eulerAncestralCFGPP

    /// Whether this sampler takes its direction from the unconditional prediction.
    public var usesCFGPP: Bool {
        self == .eulerCFGPP || self == .eulerAncestralCFGPP
    }

    /// Whether this sampler renoises after the deterministic half.
    public var isAncestral: Bool {
        self == .eulerAncestral || self == .eulerAncestralCFGPP
    }
}

public enum TransformerRole: String, Sendable {
    case dev
    case distilled
}

/// An adapter a stage wants, by file hint rather than by path.
public struct AdapterRequest: Sendable, Equatable {
    public let filenameHint: String
    public let strength: Float
    /// Free-form, matching `LoRAOverlay.ModuleFilter`'s include list at the wiring site.
    public let include: [String]

    public init(filenameHint: String, strength: Float, include: [String] = []) {
        self.filenameHint = filenameHint
        self.strength = strength
        self.include = include
    }

    /// `HINT` or `HINT@STRENGTH`, where the hint is a path or a fragment of a filename.
    ///
    /// Split on the last `@`, and only when the tail parses as a number — an adapter living
    /// under a path containing `@` is more likely than a caller wanting a strength that
    /// isn't a number, so an unparseable tail is treated as part of the path rather than
    /// reported as a malformed strength.
    ///
    /// Strength is not clamped to `0...1`: `LoRAOverlay` folds it into `B` at load and
    /// values above 1 are a normal, if aggressive, thing to ask for. Negative values are
    /// refused because they are far more often a typo than an intent.
    public static func parse(_ spec: String) throws -> AdapterRequest {
        if let split = spec.lastIndex(of: "@"),
           let strength = Float(spec[spec.index(after: split)...]) {
            let hint = String(spec[spec.startIndex ..< split])
            guard !hint.isEmpty else {
                throw RecipeFailure.malformedAdapter(spec, reason: "the adapter name is empty")
            }
            guard strength >= 0 else {
                throw RecipeFailure.malformedAdapter(
                    spec, reason: "a negative strength (\(strength)) is almost always a typo")
            }
            return AdapterRequest(filenameHint: hint, strength: strength)
        }
        guard !spec.isEmpty else {
            throw RecipeFailure.malformedAdapter(spec, reason: "the adapter name is empty")
        }
        return AdapterRequest(filenameHint: spec, strength: 1.0)
    }
}

// MARK: - Stages

public struct RecipeStage: Sendable, Equatable {
    public let name: String
    /// 0.5 for a half-resolution first stage, 1.0 otherwise.
    public let scale: Double
    public let sigmas: SigmaPlan
    public let sampler: SamplerKind
    public let videoGuidance: RecipeGuidance
    public let audioGuidance: RecipeGuidance
    public let transformer: TransformerRole
    public let adapters: [AdapterRequest]
    /// False for a stage that hands a latent to the next one instead of decoding.
    public let decodes: Bool

    public init(name: String, scale: Double = 1.0, sigmas: SigmaPlan,
                sampler: SamplerKind = .euler,
                videoGuidance: RecipeGuidance, audioGuidance: RecipeGuidance,
                transformer: TransformerRole, adapters: [AdapterRequest] = [],
                decodes: Bool = true) {
        self.name = name
        self.scale = scale
        self.sigmas = sigmas
        self.sampler = sampler
        self.videoGuidance = videoGuidance
        self.audioGuidance = audioGuidance
        self.transformer = transformer
        self.adapters = adapters
        self.decodes = decodes
    }

    /// This stage's own shape, re-checked against the 32 grid.
    ///
    /// The check is the point. A half-resolution stage of an output that is only a multiple
    /// of 32 lands off the grid, and that failure is silent downstream.
    public func shape(forOutput output: Shape) throws -> Shape {
        let width = Int((Double(output.width) * scale).rounded())
        let height = Int((Double(output.height) * scale).rounded())
        let shape = Shape(width: width, height: height)
        guard shape.isLegal(on: .single) else {
            throw RecipeFailure.shapeOffGrid(width: width, height: height,
                                             grid: .single, stage: name)
        }
        return shape
    }
}

// MARK: - Evidence

/// What is actually known about a recipe at a shape.
///
/// A recipe is offered with its evidence attached rather than withheld until it has been
/// measured, because withholding would leave a menu of one entry: the measurements here are
/// at 640x384x97 @ 24 fps and 320x192x25 @ 24 fps, and a measurement at one shape says
/// nothing about another. None of these statuses is a claim about the picture either — how a
/// render looks is advisory at every shape.
public enum Evidence: Sendable, Equatable {
    /// This exact shape, step count and pipeline has been measured.
    case gated(shape: Shape, frames: Int, steps: Int, source: String)
    /// The same pipeline, a different shape. The arithmetic was measured; this shape was not.
    case sibling(of: Shape, note: String)
    /// The pipeline itself has never been measured here.
    case unmeasured(String)
    /// A stage this port has not implemented. Registered so the refusal can carry a reason.
    case noRoute(String)

    public var label: String {
        switch self {
        case .gated: "gated"
        case .sibling: "sibling"
        case .unmeasured: "unmeasured"
        case .noRoute: "no route"
        }
    }

    public var isRunnable: Bool {
        if case .noRoute = self { return false }
        return true
    }
}

// MARK: - Failures

public enum RecipeFailure: Error, Equatable, CustomStringConvertible {
    case unknownRecipe(String)
    case shapeOffGrid(width: Int, height: Int, grid: Grid, stage: String)
    case aspectUnachievable(requested: AspectRatio, resolved: Shape, nearestExact: Shape?)
    case belowGridFloor(width: Int, height: Int, grid: Grid)
    case framesOffLattice(Int, nearestBelow: Int, nearestAbove: Int)
    case conflictingShape(String)
    case conflictingConditioning(String)
    case negationWithoutGuidance(videoCFG: Double, audioCFG: Double)
    case stepsOutOfRange(Int, allowed: ClosedRange<Int>)
    case framesOutOfRange(Int, allowed: ClosedRange<Int>)
    case noRoute(recipe: String, reason: String)
    case missingComponent(LTXComponent, recipe: String)
    /// A knob that the two-stage distilled pipeline would silently ignore.
    case twoStageIgnores(String)
    /// A step count handed to a recipe whose every schedule is a literal table.
    ///
    /// The generalisation of ``twoStageIgnores(_:)``'s `--steps` clause. Derived from the
    /// recipe's own stages by ``Recipe/scheduleIsFixed``, so a fixed-schedule recipe added
    /// later inherits the refusal instead of needing someone to remember it.
    case scheduleIsFixed(recipe: String, detail: String)
    /// A transform recipe resolved against a generative request.
    case notGenerative(recipe: String, detail: String)
    case keyframeOffLattice(pixelFrame: Int, temporalScale: Int,
                            nearestBelow: Int, nearestAbove: Int)
    case keyframeOutOfRange(pixelFrame: Int, frames: Int)
    case duplicateKeyframe(pixelFrame: Int)
    case malformedKeyframe(String, reason: String)
    case malformedAdapter(String, reason: String)

    public var description: String {
        switch self {
        case let .unknownRecipe(id):
            return "no recipe registered as '\(id)'"
        case let .shapeOffGrid(width, height, grid, stage):
            return "stage '\(stage)' resolves to \(width)x\(height), which is not a multiple "
                + "of \(grid.value) on both axes (the upsampler's spatial factor)"
        case let .aspectUnachievable(requested, resolved, nearestExact):
            let exact = nearestExact.map { " The nearest exact \(requested.rawValue) is "
                + "\($0.description) at \(String(format: "%.3f", $0.megapixels)) MP." } ?? ""
            return "\(requested.rawValue) is not expressible on this grid near the requested "
                + "size; the nearest achievable is \(resolved.description) at aspect "
                + "\(String(format: "%.4f", resolved.aspect)).\(exact)"
        case let .belowGridFloor(width, height, grid):
            return "\(width)x\(height) is under the floor of two \(grid.value)-pixel cells "
                + "per axis; a shape that small is not one the model was trained on"
        case let .framesOffLattice(frames, below, above):
            return "\(frames) frames is not on the 8k+1 lattice (contract 4); the nearest "
                + "legal counts are \(below) and \(above)"
        case let .conflictingShape(detail):
            return "conflicting shape: \(detail)"
        case let .conflictingConditioning(detail):
            return "conflicting conditioning: \(detail)"
        case let .negationWithoutGuidance(videoCFG, audioCFG):
            return "a negative prompt has no effect at video cfg \(videoCFG) and audio cfg "
                + "\(audioCFG): at 1.0 the guidance term is exactly zero, and the pass still "
                + "runs (contract 7 sets disable_cfg1_optimization)"
        case let .stepsOutOfRange(steps, allowed):
            return "steps \(steps) is outside \(allowed) for this recipe"
        case let .framesOutOfRange(frames, allowed):
            return "\(frames) frames is outside \(allowed) for this recipe"
        case let .noRoute(recipe, reason):
            return "recipe '\(recipe)' has no route in this port: \(reason)"
        case let .missingComponent(component, recipe):
            return "recipe '\(recipe)' needs the \(component.rawValue) checkpoint, which was "
                + "not found"
        case let .twoStageIgnores(detail):
            return "the two-stage distilled pipeline ignores \(detail)"
        case let .scheduleIsFixed(recipe, detail):
            return "recipe '\(recipe)' has a fixed schedule and ignores \(detail)"
        case let .notGenerative(recipe, detail):
            return "recipe '\(recipe)' does not render from a prompt: \(detail)"
        case let .keyframeOffLattice(pixelFrame, scale, below, above):
            return "a keyframe at frame \(pixelFrame) does not land on a latent boundary. "
                + "The VAE is causal: frame 0 encodes alone, then every \(scale) frames "
                + "become one latent, so a still placed inside a chunk would freeze all "
                + "\(scale) of its frames to that image. The nearest boundaries are "
                + "\(below) and \(above)"
        case let .keyframeOutOfRange(pixelFrame, frames):
            return "a keyframe at frame \(pixelFrame) is outside this render's 0...\(frames - 1)"
        case let .duplicateKeyframe(pixelFrame):
            return "two keyframes both condition frame \(pixelFrame); the second would "
                + "overwrite the first"
        case let .malformedKeyframe(spec, reason):
            return "cannot read the keyframe '\(spec)': \(reason)"
        case let .malformedAdapter(spec, reason):
            return "cannot read the adapter '\(spec)': \(reason)"
        }
    }
}

// MARK: - Conditioning inputs

/// One still, and where in the clip it belongs.
///
/// The video stream's conditioning used to be a single `URL` that always meant frame 0,
/// because frame 0 was the only index the write in `RenderConditioning` could reach. It is
/// token-indexed now, so a first frame, a last frame and any set in between are the same
/// operation at different offsets.
///
/// `pixelFrame` must land on a latent boundary — 0, or a multiple of the VAE's temporal
/// scale. The VAE is causal: frame 0 encodes alone and every 8 pixel frames after it become
/// one latent, so a still requested mid-chunk would freeze all 8 of that chunk's frames to
/// one image. ``RecipeRequest/validate()`` refuses it rather than rounding, for the same
/// reason an explicit frame count is validated instead of snapped.
public struct ImageConditioning: Sendable, Equatable {
    public var url: URL
    /// 0 is the first frame. Use ``RecipeRequest/lastFramePlaceholder`` for the last one,
    /// which cannot be named until the duration is resolved.
    public var pixelFrame: Int
    /// `nil` takes the request's own ``RecipeRequest/strength``.
    public var strength: Double?

    public init(url: URL, pixelFrame: Int = 0, strength: Double? = nil) {
        self.url = url
        self.pixelFrame = pixelFrame
        self.strength = strength
    }

    /// `PATH@FRAME` or `PATH@FRAME:STRENGTH`, with `last` accepted for the closing frame.
    ///
    /// Split on the **last** `@` rather than the first: a path may contain one, a frame spec
    /// may not. `first` and `last` are spelled out because `-1` for "the end" is a
    /// convention somebody always reads as an error.
    ///
    ///     poster.png@0            the opening frame
    ///     poster.png@last         the closing frame
    ///     mid.png@48:0.6          frame 48, loosely held
    public static func parse(_ spec: String) throws -> ImageConditioning {
        guard let split = spec.lastIndex(of: "@") else {
            throw RecipeFailure.malformedKeyframe(
                spec, reason: "expected PATH@FRAME, e.g. 'poster.png@0' or 'end.png@last'")
        }
        let path = String(spec[spec.startIndex ..< split])
        guard !path.isEmpty else {
            throw RecipeFailure.malformedKeyframe(spec, reason: "the path is empty")
        }

        let tail = spec[spec.index(after: split)...].split(separator: ":", maxSplits: 1)
        guard let framePart = tail.first, !framePart.isEmpty else {
            throw RecipeFailure.malformedKeyframe(spec, reason: "no frame after '@'")
        }

        let frame: Int
        switch framePart.lowercased() {
        case "first": frame = 0
        case "last": frame = RecipeRequest.lastFramePlaceholder
        default:
            guard let parsed = Int(framePart), parsed >= 0 else {
                throw RecipeFailure.malformedKeyframe(
                    spec, reason: "'\(framePart)' is not a frame index; use a non-negative "
                        + "integer, 'first' or 'last'")
            }
            frame = parsed
        }

        var strength: Double?
        if tail.count == 2 {
            guard let parsed = Double(tail[1]), (0.0 ... 1.0).contains(parsed) else {
                throw RecipeFailure.malformedKeyframe(
                    spec, reason: "'\(tail[1])' is not a strength in 0...1")
            }
            strength = parsed
        }

        return ImageConditioning(url: URL(fileURLWithPath: path), pixelFrame: frame,
                                 strength: strength)
    }
}

// MARK: - The request

/// Everything one render is asked for, in one value.
///
/// The items a single-stream text-to-video request would not carry, and why each earns
/// its place: dual CFG is two scales and not one (contract 7); STG is a scale
/// **and** a block list; modality and rescale are two more scales; the audio stream's
/// latent count is `round(frames / fps * 25)` and so `frameRate` is load-bearing rather
/// than cosmetic (contract 3); and the frame lattice is `8k+1` (contract 4).
///
/// Deliberately absent: a quality-profile tier, a `resolution` enum (the megapixel
/// target replaces it), and an attention-backend selector — that last one is a memory
/// class rather than a preference, and `Renderer.Spec` owns it.
public struct RecipeRequest: Sendable {

    // what to render
    public var prompt: String
    public var negativePrompt: String?
    public var seconds: Double
    /// Overrides `seconds`. Must sit on the `8k+1` lattice.
    public var frames: Int?
    public var frameRate: Double
    /// `nil` takes the recipe's own step count.
    public var steps: Int?
    public var seed: UInt64

    // shape — megapixels selectable, per the ask
    public var megapixels: Double?
    public var aspectRatio: AspectRatio
    /// Explicit dimensions, which override `megapixels` and `aspectRatio`.
    public var width: Int?
    public var height: Int?
    public var recipeID: String?

    // conditioning; the mode is derived from these, never passed
    /// Stills pinned to latent frames — first, last, or any set in between.
    ///
    /// Empty for text-to-AV. Replaces the single frame-0 `image` this type used to carry;
    /// the `image:` initialiser parameter still exists and appends one at frame 0.
    public var images: [ImageConditioning]
    public var video: URL?
    public var audio: URL?
    public var audioMel: URL?
    public var strength: Double

    /// Adapters stacked on top of whatever the recipe's own stages ask for.
    ///
    /// `filenameHint` may be an absolute path or a fragment of a filename; the catalogue
    /// resolves the fragment against the model tree and refuses a hint that matches nothing
    /// or more than one thing. Overlays compose by summation and every delta is computed
    /// from the base input, so stacking two is well-defined — see `LoRAOverlay`.
    ///
    /// **Nothing here has been measured end to end.** What has been checked is that an
    /// adapter reaches the tensors it should, at the first step — not the trajectory that
    /// consumes it, and not the picture.
    public var adapters: [AdapterRequest]

    // numerics; nil takes the recipe's own
    public var videoCFG: Double?
    public var audioCFG: Double?
    public var stgScale: Double?
    public var stgBlocks: [Int]?
    public var modalityScale: Double?
    public var rescaleScale: Double?
    public var stgStartPercent: Double
    public var stgEndPercent: Double
    public var modalityStartPercent: Double
    public var modalityEndPercent: Double

    /// Relative L1 change in block 0's residual below which the rest of the
    /// stack is skipped. **0 disables** (the default). 0.10 is the measured
    /// knee; that is a starting point for a sweep, not a quality claim
    /// for this model. One cache per guidance pass; a shared cache fails silently.
    public var cacheThreshold: Double
    /// Consecutive-reuse ceiling. 5 is the cap that survived viewing; same caveat.
    public var cacheMaxSkips: Int

    // policy and output
    public var allowSuboptimal: Bool
    public var overwriteOutput: Bool
    public var videoOutput: URL
    public var audioOutput: URL?

    /// Whether the named recipe is the two-stage distilled pipeline.
    ///
    /// **Derived, not stored.** It used to be the field that chose the pipeline, which meant
    /// one request type described two pipelines and every knob belonging to one of them had
    /// to be refused by hand under the other. ``recipeID`` selects the pipeline now and this
    /// is a reading of it, so the two cannot disagree.
    public var twoStage: Bool { recipeID == RecipeRegistry.distilled.id }

    public init(prompt: String,
                videoOutput: URL,
                audioOutput: URL? = nil,
                negativePrompt: String? = nil,
                seconds: Double = 4.0,
                frames: Int? = nil,
                frameRate: Double = 24,
                steps: Int? = nil,
                seed: UInt64 = 0,
                megapixels: Double? = nil,
                aspectRatio: AspectRatio = .r16x9,
                width: Int? = nil,
                height: Int? = nil,
                recipeID: String? = nil,
                image: URL? = nil,
                images: [ImageConditioning] = [],
                video: URL? = nil,
                audio: URL? = nil,
                audioMel: URL? = nil,
                strength: Double = 1.0,
                adapters: [AdapterRequest] = [],
                videoCFG: Double? = nil,
                audioCFG: Double? = nil,
                stgScale: Double? = nil,
                stgBlocks: [Int]? = nil,
                modalityScale: Double? = nil,
                rescaleScale: Double? = nil,
                stgStartPercent: Double = 0,
                stgEndPercent: Double = 1,
                modalityStartPercent: Double = 0,
                modalityEndPercent: Double = 1,
                cacheThreshold: Double = 0,
                cacheMaxSkips: Int = 5,
                allowSuboptimal: Bool = false,
                overwriteOutput: Bool = false,
                twoStage: Bool = false) {
        self.prompt = prompt
        self.videoOutput = videoOutput
        self.audioOutput = audioOutput
        self.negativePrompt = negativePrompt
        self.seconds = seconds
        self.frames = frames
        self.frameRate = frameRate
        self.steps = steps
        self.seed = seed
        self.megapixels = megapixels
        self.aspectRatio = aspectRatio
        self.width = width
        self.height = height
        // `image:` is the frame-0 shorthand, prepended so an explicit `images:` list can
        // still add a last frame beside it without the caller ordering them by hand.
        self.images = (image.map { [ImageConditioning(url: $0)] } ?? []) + images
        self.video = video
        self.audio = audio
        self.audioMel = audioMel
        self.strength = strength
        self.adapters = adapters
        self.videoCFG = videoCFG
        self.audioCFG = audioCFG
        self.stgScale = stgScale
        self.stgBlocks = stgBlocks
        self.modalityScale = modalityScale
        self.rescaleScale = rescaleScale
        self.stgStartPercent = stgStartPercent
        self.stgEndPercent = stgEndPercent
        self.modalityStartPercent = modalityStartPercent
        self.modalityEndPercent = modalityEndPercent
        self.cacheThreshold = cacheThreshold
        self.cacheMaxSkips = cacheMaxSkips
        self.allowSuboptimal = allowSuboptimal
        self.overwriteOutput = overwriteOutput
        // `twoStage:` is kept as a bridge for callers that predate the registry. It selects
        // a recipe rather than setting a flag, so there is still only one selector.
        self.recipeID = recipeID ?? (twoStage ? RecipeRegistry.distilled.id : nil)
    }

    /// Names the last frame of a clip whose duration is not yet resolved.
    ///
    /// A caller working in seconds cannot write `frames - 1`, and making them resolve the
    /// lattice first just to ask for an end frame would put the snapping rule in two places.
    public static let lastFramePlaceholder = -1

    /// The stills, with ``lastFramePlaceholder`` resolved and every position checked.
    ///
    /// Sorted by frame, so a caller who listed the end frame first still gets a stable
    /// order in provenance and in the duplicate check.
    public func resolvedImages(frames: Int,
                               latent: LatentGeometry = LatentGeometry()) throws -> [ImageConditioning] {
        var seen = Set<Int>()
        var out: [ImageConditioning] = []
        for image in images {
            var resolved = image
            if image.pixelFrame == Self.lastFramePlaceholder {
                resolved.pixelFrame = frames - 1
            }
            guard resolved.pixelFrame >= 0, resolved.pixelFrame < frames else {
                throw RecipeFailure.keyframeOutOfRange(pixelFrame: resolved.pixelFrame,
                                                       frames: frames)
            }
            // The last pixel frame is the end of its own latent chunk, so it is a legal
            // boundary even though `frames - 1` is not a multiple of the temporal scale.
            let scale = latent.temporalScale
            let isLast = resolved.pixelFrame == frames - 1
            if !isLast, resolved.pixelFrame % scale != 0 {
                let below = (resolved.pixelFrame / scale) * scale
                throw RecipeFailure.keyframeOffLattice(pixelFrame: resolved.pixelFrame,
                                                       temporalScale: scale,
                                                       nearestBelow: below,
                                                       nearestAbove: min(below + scale, frames - 1))
            }
            guard seen.insert(resolved.pixelFrame).inserted else {
                throw RecipeFailure.duplicateKeyframe(pixelFrame: resolved.pixelFrame)
            }
            if let strength = resolved.strength, !(0.0 ... 1.0).contains(strength) {
                throw RecipeFailure.conflictingConditioning(
                    "keyframe strength must be in 0...1, got \(strength)")
            }
            out.append(resolved)
        }
        return out.sorted { $0.pixelFrame < $1.pixelFrame }
    }

    /// Every cheap, purely local contradiction — thrown before a byte of checkpoint opens.
    public func validate() throws {
        if (width == nil) != (height == nil) {
            throw RecipeFailure.conflictingShape(
                "width and height come as a pair; give both, or neither and a megapixel "
                    + "target with an aspect ratio")
        }
        if width != nil && megapixels != nil {
            throw RecipeFailure.conflictingShape(
                "explicit dimensions and a megapixel target both decide the shape; give one")
        }
        if !images.isEmpty && video != nil {
            throw RecipeFailure.conflictingConditioning(
                "an image and a video both condition the video stream")
        }
        // Positions are checked here, before a 26 GB text encode, rather than at the point
        // the latent is written.
        _ = try resolvedImages(frames: try resolvedFrames())
        if audio != nil && audioMel != nil {
            throw RecipeFailure.conflictingConditioning(
                "--audio and --audio-mel both condition the audio stream")
        }
        guard (0.0 ... 1.0).contains(strength) else {
            throw RecipeFailure.conflictingConditioning(
                "conditioning strength must be in 0...1, got \(strength)")
        }
        if let negativePrompt, !negativePrompt.isEmpty,
           (videoCFG ?? 1.0) <= 1.0, (audioCFG ?? 1.0) <= 1.0 {
            throw RecipeFailure.negationWithoutGuidance(videoCFG: videoCFG ?? 1.0,
                                                        audioCFG: audioCFG ?? 1.0)
        }
        guard cacheThreshold >= 0 else {
            throw RecipeFailure.conflictingConditioning(
                "cache threshold must be >= 0, got \(cacheThreshold)")
        }
        guard cacheMaxSkips >= 1 else {
            throw RecipeFailure.conflictingConditioning(
                "cache max skips must be >= 1, got \(cacheMaxSkips)")
        }
        try Self.validateWindow("--stg-start/--stg-end", start: stgStartPercent, end: stgEndPercent)
        try Self.validateWindow("--modality-start/--modality-end",
                                start: modalityStartPercent, end: modalityEndPercent)
        if twoStage { try validateTwoStage() }
    }

    private static func validateWindow(_ name: String, start: Double, end: Double) throws {
        guard (0...1).contains(start), (0...1).contains(end), start <= end else {
            throw RecipeFailure.conflictingConditioning(
                "\(name) must satisfy 0 ≤ start ≤ end ≤ 1, got [\(start), \(end)]")
        }
    }

    /// The knobs the two-stage distilled pipeline has no seam for, refused rather than
    /// accepted and dropped.
    ///
    /// Every one of these is a setting that changes a single-stage render and changes
    /// **nothing** here, which is the shape of mistake that survives a whole session: the
    /// render completes, looks plausible, and was not the render that was asked for.
    ///
    /// * The pipeline runs one plain denoise per step and builds **no** guider, so cfg, stg,
    ///   modality and rescale have nothing to act on — and with no unconditional pass, a
    ///   negative prompt is never encoded, let alone applied.
    /// * Both schedules are literal tables, 9 sigmas and 4. The step count is 8 + 3 and is
    ///   not a knob.
    /// * `--image` **is** supported: stage 1 conditions at half the output size and stage 2
    ///   rebuilds the conditioning at full size from the same file
    ///   (`DistilledRenderer.imageConditioning`, `MediaInput.resizeAndCenterCrop`).
    /// * `--video`, `--audio` and `--audio-mel` are not, and are refused **by name**. This
    ///   pipeline conditions the video stream with stills and nothing else: it has no seam
    ///   for a driving clip, and its audio stream takes no conditioning in either stage.
    ///   Accepting them and dropping them is the failure this whole function exists to
    ///   prevent.
    private func validateTwoStage() throws {
        if steps != nil {
            throw RecipeFailure.twoStageIgnores(
                "--steps: both stage schedules are fixed tables; the count is 8 + 3")
        }
        if let negativePrompt, !negativePrompt.isEmpty {
            throw RecipeFailure.twoStageIgnores(
                "the negative prompt: no unconditional pass runs, so only one text branch "
                    + "is encoded")
        }
        for (name, set) in [("--video-cfg", videoCFG != nil), ("--audio-cfg", audioCFG != nil),
                            ("--stg-scale", stgScale != nil),
                            ("--modality-scale", modalityScale != nil),
                            ("--rescale-scale", rescaleScale != nil)] where set {
            throw RecipeFailure.twoStageIgnores(
                "\(name): it constructs no guider at all (SimpleDenoiser, one transformer "
                    + "forward per step)")
        }
        if cacheThreshold > 0 {
            throw RecipeFailure.twoStageIgnores(
                "--cache-threshold: the two-stage path is already 11 forwards with no "
                    + "guider; the residual cache is a guided-path approximation and has "
                    + "not been measured on distilled schedules")
        }
        if stgStartPercent != 0 || stgEndPercent != 1
            || modalityStartPercent != 0 || modalityEndPercent != 1 {
            throw RecipeFailure.twoStageIgnores(
                "--stg-start/--stg-end/--modality-start/--modality-end: it constructs no "
                    + "guider at all")
        }
        if video != nil {
            throw RecipeFailure.twoStageIgnores(
                "--video: the two-stage pipeline takes stills only — it has no video "
                    + "conditioning parameter at all. Use the single-stage route, which "
                    + "does have a seam for it")
        }
        for (flag, url) in [("--audio", audio), ("--audio-mel", audioMel)] where url != nil {
            throw RecipeFailure.twoStageIgnores(
                "\(flag): the two-stage pipeline conditions the VIDEO stream only. Its "
                    + "audio stream carries no conditioning slot in either stage, so a "
                    + "frozen audio stream has nowhere to enter")
        }
    }

    /// The frame count this request actually renders, snapped onto the `8k+1` lattice.
    ///
    /// An explicit `frames` is validated rather than snapped: a caller who names a count is
    /// making a claim, and silently moving it is how anchors end up a fraction of a second
    /// off. A duration is snapped, because a duration was never an exact frame count.
    public func resolvedFrames(lattice: FrameLattice = FrameLattice()) throws -> Int {
        if let frames {
            guard lattice.contains(frames) else {
                throw RecipeFailure.framesOffLattice(frames,
                                                     nearestBelow: lattice.snapDown(frames),
                                                     nearestAbove: lattice.snapUp(frames))
            }
            return frames
        }
        return lattice.snapDown(max(1, Int((seconds * frameRate).rounded())))
    }

    /// Pixel size this request actually renders, on the 64-grid.
    ///
    /// Explicit width/height are validated, not snapped. No size at all takes the pinned
    /// shape for this aspect (640×384 for 16:9) rather than deriving a nearby rung that has
    /// nothing behind it. A megapixel target uses the ladder.
    public func resolvedShape(grid: Grid = .single) throws -> Shape {
        if let width, let height {
            let shape = Shape(width: width, height: height)
            guard width % grid.value == 0, height % grid.value == 0 else {
                throw RecipeFailure.shapeOffGrid(width: width, height: height, grid: grid,
                                                 stage: "output")
            }
            guard shape.isLegal(on: grid) else {
                throw RecipeFailure.belowGridFloor(width: width, height: height, grid: grid)
            }
            return shape
        }
        if let megapixels {
            return try Ladder.rung(megapixels: megapixels, aspect: aspectRatio, grid: grid).shape
        }
        if let pin = Ladder.pinned.first(where: {
            $0.aspectFamily == aspectRatio && $0.shape.isLegal(on: grid)
        }) {
            return pin.shape
        }
        return try Ladder.rung(megapixels: 0.25, aspect: aspectRatio, grid: grid).shape
    }
}

// MARK: - Resolution

/// A request and a recipe, reconciled — every shape decided, nothing loaded.
public struct ResolvedRecipe: Sendable {
    public let recipeID: String
    public let output: Shape
    /// One entry per stage, in order, each with the shape that stage samples at.
    public let stages: [(stage: RecipeStage, shape: Shape)]
    public let frames: Int
    public let frameRate: Double
    public let evidence: Evidence

    public init(recipeID: String, output: Shape,
                stages: [(stage: RecipeStage, shape: Shape)],
                frames: Int, frameRate: Double, evidence: Evidence) {
        self.recipeID = recipeID
        self.output = output
        self.stages = stages
        self.frames = frames
        self.frameRate = frameRate
        self.evidence = evidence
    }

    /// Video tokens per stage. Exact arithmetic, not an estimate — this port has no
    /// measured throughput, so it may not print a time and this is the honest stand-in.
    public func videoTokens(latent: LatentGeometry = LatentGeometry()) -> [Int] {
        let latentFrames = latent.latentFrames(forFrames: frames)
        return stages.map { _, shape in
            latentFrames * (shape.height / latent.spatialScale)
                * (shape.width / latent.spatialScale)
        }
    }

    /// `round(frames / fps * 25)` — contract 3, `round` and not `ceil`.
    public func audioTokens(latent: LatentGeometry = LatentGeometry()) -> Int {
        latent.audioLatentCount(frames: frames, frameRate: frameRate)
    }
}

// MARK: - Capability

/// What the machine and the checkpoint set can support.
///
/// A value the caller supplies, never probed here. `LTXHardware` owns chip and memory
/// detection and is currently a placeholder, so a memory budget computed in this target
/// would be both a boundary violation and a fabricated number.
public struct Capability: Sendable {
    public let unifiedMemoryBytes: UInt64?
    public let components: Set<LTXComponent>

    public init(unifiedMemoryBytes: UInt64? = nil, components: Set<LTXComponent> = []) {
        self.unifiedMemoryBytes = unifiedMemoryBytes
        self.components = components
    }

    /// Every reason this machine cannot run a recipe. Empty means nothing known is wrong —
    /// which is not the same as "it will fit", and must not be reported as such until
    /// `LTXHardware` can answer the memory question.
    public func reasonsCannotRun(_ recipeID: String, evidence: Evidence,
                                 requires: Set<LTXComponent>) -> [RecipeFailure] {
        var out: [RecipeFailure] = []
        if case let .noRoute(reason) = evidence {
            out.append(.noRoute(recipe: recipeID, reason: reason))
        }
        for component in requires.sorted(by: { $0.rawValue < $1.rawValue })
        where !components.contains(component) {
            out.append(.missingComponent(component, recipe: recipeID))
        }
        return out
    }
}

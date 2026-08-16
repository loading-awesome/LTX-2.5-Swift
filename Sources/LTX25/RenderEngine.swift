// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import LTXHardware
import LTXRecipes
import LTXPipeline

/// One in-flight render. A second caller is refused, not queued.
///
/// The Metal forward must not run on this actor: that would freeze admission,
/// progress and cancel for the length of the job. `startJob` only admits,
/// identifies checkpoints (header-only) and spawns a detached task.
public actor RenderEngine {

    public static let shared = RenderEngine()

    private var currentJob: RenderJob?

    private init() {}

    /// Start a rendering job. Throws `LTXError.engineBusy` if a job is already in progress.
    /// - Parameter memoryMarginFraction: held back when planning, from the caller's config.
    ///   A machine policy rather than a render parameter, which is why it rides on the call
    ///   and not on the request — but it must ride on *something*, or `doctor` reports one
    ///   margin while admission quietly applies another.
    public func startJob(request: LTX25.Request, checkpoints: LTX25.Checkpoints,
                         memoryMarginFraction: Double = 0.15) throws -> RenderJob {
        if let current = currentJob, !current.isFinished {
            throw LTX25.LTXError.engineBusy
        }

        // The recipe decides the pipeline, the schedule, the guidance and the shape. This is
        // the single place that resolution happens: `RenderJob` is handed the answer rather
        // than re-deriving it, so the job cannot render a shape the admission check did not
        // plan memory for.
        let resolved: ResolvedRecipe
        do {
            let recipe = try RecipeRegistry.recipe(request.recipeID ?? RecipeRegistry.defaultID)
            resolved = try recipe.resolve(request)
        } catch let error as LTX25.LTXError {
            throw error
        } catch {
            throw LTX25.LTXError.invalidRequest("\(error)")
        }
        let frames = resolved.frames
        let shape = resolved.output

        do {
            let tokens = try LatentGeometry().streamTokenCounts(
                width: shape.width, height: shape.height, frames: frames,
                frameRate: request.frameRate)
            let plan = MemoryPlan.plan(videoTokens: tokens.video,
                                       availableBytes: Machine.availableBytes(),
                                       marginFraction: memoryMarginFraction)
            if !plan.fits {
                throw LTX25.LTXError.insufficientMemory(plan.explanation)
            }
        } catch let error as LTX25.LTXError {
            throw error
        } catch {
            throw LTX25.LTXError.invalidRequest("\(error)")
        }

        let bound: CheckpointIdentity.Resolved
        do {
            bound = try CheckpointIdentity.bind(
                transformer: checkpoints.dit,
                textEncoder: checkpoints.textEncoder,
                videoVAE: checkpoints.videoVAE,
                audioVAE: checkpoints.audioVAE)
        } catch {
            throw LTX25.LTXError.invalidRequest("\(error)")
        }

        let pipeline = Renderer.Checkpoints(
            textEncoder: bound.textEncoder.url,
            dit: bound.transformer.url,
            videoVAE: bound.videoVAE.url,
            audioVAE: bound.audioVAE.url)

        // Admission, not execution: a two-stage render that names no upsampler cannot
        // start, and finding that out here costs a nil check rather than the 26 GB text
        // encode it would otherwise be discovered after.
        if resolved.stages.count > 1, checkpoints.upsampler == nil {
            throw LTX25.LTXError.invalidRequest(
                "recipe '\(resolved.recipeID)' has \(resolved.stages.count) stages and needs "
                    + "the x2 latent spatial upsampler checkpoint; pass --upsampler")
        }

        let job = RenderJob(request: request, resolved: resolved, checkpoints: pipeline,
                            upsampler: checkpoints.upsampler, engine: self)
        currentJob = job
        job.start()
        return job
    }

    /// Callback from `RenderJob` when it finishes or cancels.
    func jobDidFinish(_ job: RenderJob) {
        if currentJob === job {
            currentJob = nil
        }
    }
}

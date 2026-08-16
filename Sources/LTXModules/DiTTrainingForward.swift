// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import MLX

/// A ``LoRATrainingStep/Forward`` driven by the real transformer.
///
/// ``LoRATrainingStep`` takes its forward from the caller so the step can be exercised against
/// a small stand-in. This is the other implementation — the one an actual run uses. It holds
/// everything that is frozen for the step (the base weights, the encoded text, the audio
/// stream, the geometry) and turns the step's `(noisy, perTokenTimesteps)` into a full
/// `DiTForward` call.
///
/// Gradient checkpointing is **on by default**. At production shape it is the difference
/// between 100.8 GB and roughly 350 GB (`docs/TRAINING.md`), and there is no reason a training
/// forward would want it off except to measure the difference.
///
/// ## Batch 1
///
/// `DiTForward.Sampling` carries per-token timesteps as `[tokens]`, one stream, with no batch
/// axis — the sampler feeds it one sequence at a time. So this refuses a batch above 1 rather
/// than quietly training on the first example and reporting a loss for all of them.
/// Accumulating gradients over several single-example steps is the way to a larger effective
/// batch here.
public struct DiTTrainingForward {

    /// The frozen transformer. Its `checkpointing` is set per call from the traced module, so
    /// whatever is on this copy is ignored.
    public var base: DiTForward

    /// `norm_out` + modulation + `proj_out`.
    ///
    /// Required, not optional. ``DiTForward`` returns the stream at model width — 4096 here —
    /// and the objective's target is latent-shaped, 128 channels. Without the head the two
    /// do not even broadcast, which is how this was caught; but a head applied to the wrong
    /// stream, or an embedded timestep taken from the wrong place, would broadcast fine and
    /// train on a subtly wrong velocity.
    public var head: DiTOutputHead

    public var geometry: DiTForward.Geometry

    /// Encoded prompt features for each stream, from the text cache.
    public var videoContext: MLXArray
    public var audioContext: MLXArray

    /// The audio stream. Video-only training still has to supply one — the transformer's
    /// cross-modal bridges read it on every block — so this is the clean audio latent, left
    /// unnoised and excluded from the loss by the batch's conditioning mask.
    public var audioLatent: MLXArray

    /// Constant per-token timesteps for the audio stream, matching ``audioLatent``.
    public var audioTimestep: MLXArray

    public var gradientCheckpointing: Bool

    /// Blocks to run. Nil is all of them; a small number is for probes.
    public var blocks: Int?

    public init(base: DiTForward,
                head: DiTOutputHead,
                geometry: DiTForward.Geometry,
                videoContext: MLXArray,
                audioContext: MLXArray,
                audioLatent: MLXArray,
                audioTimestep: MLXArray,
                gradientCheckpointing: Bool = true,
                blocks: Int? = nil) {
        self.base = base
        self.head = head
        self.geometry = geometry
        self.videoContext = videoContext
        self.audioContext = audioContext
        self.audioLatent = audioLatent
        self.audioTimestep = audioTimestep
        self.gradientCheckpointing = gradientCheckpointing
        self.blocks = blocks
    }

    public enum Failure: Error, CustomStringConvertible {
        case batchUnsupported(Int)
        case videoTokenCount(got: Int, geometry: Int)

        public var description: String {
            switch self {
            case let .batchUnsupported(batch):
                return """
                    batch \(batch): DiTForward.Sampling has no batch axis, so a training \
                    forward runs one example at a time. Use gradient accumulation.
                    """
            case let .videoTokenCount(got, expected):
                return "latent carries \(got) video tokens, geometry implies \(expected)"
            }
        }
    }

    /// Checks what can be checked once, so the closure below can use `try!` honestly.
    ///
    /// Everything `DiTForward` throws is structural — a missing weight, a shape that does not
    /// match the topology, a token count that disagrees with the geometry. None of it depends
    /// on the step, so all of it either fails on the first call or never. The traced closure
    /// cannot throw (`MLXNN.valueAndGrad` takes a non-throwing body), and turning a structural
    /// error into a `fatalError` a thousand steps into a run is worse than refusing here.
    public func validate(shape: TrainingObjective.Shape) throws {
        guard shape.batch == 1 else { throw Failure.batchUnsupported(shape.batch) }
        guard shape.sequence == geometry.videoTokens else {
            throw Failure.videoTokenCount(got: shape.sequence,
                                          geometry: geometry.videoTokens)
        }
    }

    /// The step's forward, with `module` supplied by the gradient transform.
    ///
    /// Call ``validate(shape:)`` once before the loop.
    public func callAsFunction() -> LoRATrainingStep.Forward {
        { module, noisy, timesteps in
            var forward = base
            if gradientCheckpointing {
                // Per block, from the *traced* factors. Note there is no `attach` on this
                // path: attaching as well would capture the same arrays, and captured arrays
                // get no gradient without reporting anything.
                forward.checkpointing = module.blockScopedCheckpointing()
            } else {
                forward.attach(lora: LoRAOverlay.trainable(
                    zip(module.keys, zip(module.a, module.b)).map {
                        (baseKey: $0.0, a: $0.1.0, b: $0.1.1)
                    },
                    scaling: module.scaling))
            }

            // `[1, tokens] -> [tokens]`: validated to batch 1 above.
            let videoTimestep = timesteps.reshaped([-1])
            let sampling = DiTForward.Sampling(
                videoTimestep: videoTimestep,
                audioTimestep: audioTimestep,
                // The prompt and gate timesteps are scalars over the whole stream. Taking
                // element 0 of the per-token vector is exact here because every token in a
                // step shares one sigma — `TrainingObjective.perTokenTimesteps` broadcasts a
                // single draw across the sequence, varying only where conditioning holds a
                // token clean.
                videoPromptTimestep: videoTimestep[0 ..< 1],
                audioPromptTimestep: audioTimestep[0 ..< 1],
                a2vGateTimestep: audioTimestep[0 ..< 1],
                v2aGateTimestep: videoTimestep[0 ..< 1])

            // `try!` — see ``validate(shape:)``.
            //
            // The transformer takes its working precision from the latents it is handed
            // (`rotaries(geometry:dtype:)` casts the RoPE tables to match), so the two streams
            // must agree. The audio latent is the declaration: the noisy video arrives from
            // the objective in float32 and is cast to it.
            let out = try! forward(videoLatent: noisy.asType(audioLatent.dtype),
                                   audioLatent: audioLatent,
                                   videoContext: videoContext, audioContext: audioContext,
                                   sampling: sampling, geometry: geometry,
                                   blocks: blocks)
            // Back to latent space. The head's embedded timestep is the one `DiTForward`
            // derived for this stream, not a re-derivation — the two must be the same or the
            // modulation is applied at a timestep the blocks never saw.
            let projected = try! head(out.video,
                                      embeddedTimestep: out.videoEmbeddedTimestep,
                                      stream: .video)
            // The objective works in float32; the transformer runs in the checkpoint's dtype.
            return projected.latent.asType(.float32)
        }
    }
}

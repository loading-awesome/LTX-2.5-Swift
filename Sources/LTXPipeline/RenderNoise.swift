// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXFoundation
import LTXModules
import MLX

/// An initial latent drawn here, from a seed, for a render that is judged by eye.
///
/// ## Why this exists at all, next to a contract that forbids it
///
/// `Pipeline.recordedNoise` documents contract 1: the initial latent is a recorded
/// tensor, there is no seed, and there is deliberately no overload that takes one. That
/// is still true and this type does not weaken it. What it does is separate two questions
/// that were previously answered by the same absence:
///
/// * *Does this reproduce one specific trajectory?* — needs that trajectory's recorded
///   draw, and `recordedNoise` remains the only way to get one.
/// * *Does this produce a video at all, to look at?* — needs **a** latent, and no
///   recorded one exists for a prompt nobody has run.
///
/// The second question is perceptual, and a perceptual judgement does not need any
/// particular draw: two renders from two different draws are both legitimate.
///
/// ## Why it is a separate type rather than a `seed:` parameter
///
/// Because the failure it guards against is not a deliberate one. It is someone reaching
/// for the nearest constructor, getting a plausible latent, and then chasing a difference
/// that looks like a porting bug. The trajectory saturates within about two steps, so a
/// different starting point is a *different render*, not a slightly different one: the
/// resulting difference is large, real, and completely uninformative. An hour spent
/// bisecting it is the cheap version of that mistake.
///
/// So the drawn streams do not come out as a bare `Sampler.Streams`. They come out of
/// `streamsForRenderOnly()`, which is deliberately unpleasant to type and which reads as
/// a claim at the call site.
///
/// ## What is *not* claimed here
///
/// Nothing about matching any other draw bit-for-bit. A draw made directly in bf16 and
/// this one — fp32, then rounded — differ in the low bits of every element, and any
/// render from a different draw is a different render regardless. Rounding an fp32 draw
/// is the better-conditioned of the two, so the choice is made on numerical hygiene.
///
/// The one thing that *is* claimed, and tested: the same seed produces the same latent.
/// That is what makes a render reproducible.
public struct RenderNoise {

    public enum Failure: Error, CustomStringConvertible {
        case emptyStream(String, tokens: Int, channels: Int)

        public var description: String {
            switch self {
            case let .emptyStream(name, tokens, channels):
                return "\(name) noise would be [1, \(tokens), \(channels)] — a stream with "
                     + "no tokens or no channels is not a latent, and the geometry that "
                     + "produced it is wrong"
            }
        }
    }

    /// The seed this draw came from. Recorded, not decorative: a render that cannot be
    /// re-run is an anecdote.
    public let seed: UInt64

    /// How the draw was made, for a render manifest. Written out so a future reader can
    /// tell a drawn latent from a recorded one *after* the fact, when the only surviving
    /// record is the sidecar next to an mp4.
    public let provenance: String

    private let streams: Sampler.Streams

    /// Draw an initial latent for the geometry a render will actually run at.
    ///
    /// Shapes are token-space `[1, T, C]` — what the sampler works in and what
    /// `patchify_proj.in` records — taken from `geometry` rather than passed in, because
    /// the two must agree and a mismatch surfaces as a token-count throw deep inside
    /// `unpatchifyVideo` where it reads as a sampler bug. At 640x384x97 this is
    /// `[1, 3120, 128]` video and `[1, 101, 128]` audio.
    ///
    /// Video and audio are drawn from **one** key, split into two. Drawing them from two
    /// independently seeded keys would make the audio stream a function of a seed the
    /// caller did not choose, and the two streams would then not move together when the
    /// seed changed.
    public init(seed: UInt64, geometry: DiTForward.Geometry) throws {
        let videoTokens = geometry.videoTokens
        let audioTokens = geometry.audioTokens
        let videoChannels = geometry.latent.videoChannels
        // The audio stream is patchified as `C * F` per token — `unpatchifyAudio` reads it
        // straight back out as `[1, C, T, F]` — so the channel axis here is the product,
        // not `audioChannels`. Using `audioChannels` alone yields a correctly *ranked*
        // tensor that is 64x too narrow, and the first thing to complain would be a
        // matmul inside the audio patchify projection.
        let audioChannels = geometry.latent.audioChannels * geometry.latent.audioFeatures

        guard videoTokens > 0, videoChannels > 0 else {
            throw Failure.emptyStream("video", tokens: videoTokens, channels: videoChannels)
        }
        guard audioTokens > 0, audioChannels > 0 else {
            throw Failure.emptyStream("audio", tokens: audioTokens, channels: audioChannels)
        }

        // `MLX.key` / `MLX.split`, not `MLXRandom.*`: the MLXRandom spellings are
        // deprecated forwarders and compiling against them buys a warning per call for no
        // behavioural difference.
        let key = MLX.key(seed)
        let split = MLX.split(key: key, into: 2)

        // fp32 then round, for the reason in the type's doc comment. The cast is not
        // optional: the latent is bf16 and every rounding downstream depends on it being
        // so, so handing the sampler an fp32 latent would change the arithmetic of the
        // whole render.
        func draw(_ tokens: Int, _ channels: Int, _ k: MLXArray) -> MLXArray {
            MLX.normal([1, tokens, channels], dtype: .float32, key: k).asType(.bfloat16)
        }

        self.seed = seed
        self.streams = Sampler.Streams(video: draw(videoTokens, videoChannels, split[0]),
                                       audio: draw(audioTokens, audioChannels, split[1]))
        self.provenance = "drawn: seed=\(seed) fp32-normal->bf16 "
                        + "video=[1, \(videoTokens), \(videoChannels)] "
                        + "audio=[1, \(audioTokens), \(audioChannels)] "
                        + "drawn by MLX's own RNG"
    }

    /// The drawn streams, for a render.
    ///
    /// Named for the call site, not for the return value. Anything that needs one
    /// specific starting tensor rather than any legitimate one must take it from
    /// `Pipeline.recordedNoise` instead. See the type's doc comment for what a drawn
    /// latent costs you.
    public func streamsForRenderOnly() -> Sampler.Streams { streams }
}

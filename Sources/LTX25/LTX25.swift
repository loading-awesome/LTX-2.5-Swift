// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXRecipes

/// The supported product surface. One module, actor-owned runtime.
public enum LTX25 {
    public static let version = "2.5.0"

    public typealias Request = RecipeRequest
    public typealias Resolved = ResolvedRecipe
    public typealias Aspect = AspectRatio
    public typealias EvidenceKind = Evidence
    public typealias ImageShape = Shape
    /// A still pinned to a frame. Distinct from `RenderConditioning.Keyframe`, which is the
    /// same idea after the VAE has run — this one is a path and an index, that one a latent.
    public typealias Keyframe = ImageConditioning
    public typealias Recipes = RecipeRegistry
    public typealias Pipeline = Recipe
    public typealias Adapter = AdapterRequest

    /// Paths a render needs. Identified on admission, before any payload is mapped.
    public struct Checkpoints: Sendable {
        public var textEncoder: URL
        public var dit: URL
        public var videoVAE: URL
        public var audioVAE: URL
        /// The x2 latent spatial upsampler. Required only by a two-stage distilled render,
        /// which is the only path that has anything to upsample; `nil` on every other route.
        ///
        /// Not run through ``LTXCatalog/CheckpointIdentity/bind(transformer:textEncoder:videoVAE:audioVAE:)``
        /// with the other four: identity binding is about the four components that have to
        /// agree with each other, and the upsampler is a standalone file with its own
        /// `__metadata__.config`. `DistilledRenderer.Checkpoints.verifyPresent` parses that
        /// config, which is what rejects the temporal upscaler shipped beside it under a
        /// near-identical name.
        public var upsampler: URL?

        public init(textEncoder: URL, dit: URL, videoVAE: URL, audioVAE: URL,
                    upsampler: URL? = nil) {
            self.textEncoder = textEncoder
            self.dit = dit
            self.videoVAE = videoVAE
            self.audioVAE = audioVAE
            self.upsampler = upsampler
        }
    }

    /// What a finished job wrote. No MLX types — the tensors do not leave the job.
    public struct RenderResult: Sendable {
        public var video: URL
        public var audioWAV: URL
        public var waveform: URL
        public var sidecar: URL
        /// Set when sampling succeeded but the mp4 mux failed. The wav, waveform
        /// tensor and sidecar were still written.
        public var muxFailed: String?
        public var provenance: [String: String]
    }

    /// Stable error types reported by the engine.
    public enum LTXError: Error, CustomStringConvertible, LocalizedError {
        case engineBusy
        case cancelled
        case invalidRequest(String)
        case insufficientMemory(String)
        case executionFailed(Error)

        public var description: String {
            switch self {
            case .engineBusy:
                return "The render engine is already processing a job."
            case .cancelled:
                return "The render job was cancelled."
            case let .invalidRequest(reason):
                return "Invalid render request: \(reason)"
            case let .insufficientMemory(detail):
                return "This machine cannot finish the render:\n\(detail)"
            case let .executionFailed(underlying):
                return "Render execution failed: \(underlying.localizedDescription)"
            }
        }

        public var errorDescription: String? { description }
    }
}

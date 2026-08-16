// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import LTXCatalog
import LTXFoundation
import MLX

/// Prompt in, conditioning out: the text-conditioning path composed end to end.
///
/// Each stage below stands on its own. **The joins are where this path goes wrong**: the
/// tokenizer left-pads and the connectors need right-padding; an empty negative prompt is
/// not an empty string by the time it is tokenised; and the two guidance branches are
/// branches rather than steps, so a pipeline that runs once produces half the
/// conditioning and looks complete.
///
/// ```
/// prompt ──► GemmaTokenizer ──► GemmaTextEncoder ──► TextProjectionMLX ──┐
///                                   49 hidden states    video + audio    │
///                                                                        ▼
///                     features ◄── EmbeddingsConnector x2 ◄── right-pad permutation
/// ```
///
/// ## The five joins, and what each one is guarding against
///
/// **1. The negative prompt is substituted, not read.** An empty negative is replaced by
/// `PromptDefaults.negativePrompt` before tokenising, which is **223** real tokens where
/// a literal reading gives one BOS and 1023 pads. Nothing about the literal reading looks
/// wrong — same shape, same padding side — and the positive branch, the one anybody
/// checks first, is untouched by it. See `branchPrompts`.
///
/// **2. The pipeline runs once per guidance branch.** Branch 0 is the positive prompt
/// (11 valid tokens on a short one), branch 1 the negative (223). They are branches, not
/// sampler steps: both see the same weights and differ only in their tokens.
///
/// **3. A stable partition sits between the tokenizer and the connectors.** Contract 9
/// pads to 1024 on the **left** — the 11 real tokens of a short positive prompt land at
/// indices 1013…1023 — and the connector requires `[valid…, pad…]`. Skipping the
/// permutation hands the connector 1013 registers followed by 11 stranded embeddings, in a
/// tensor of exactly the right shape.
///
/// **4. The additive mask is `(binary - 1) * finfo(bf16).max`, shaped `[B, 1, 1, S]`.**
/// Not `-infinity`, and not `finfo(float32).max` — see `EmbeddingsProcessor.bf16Max`.
///
/// **5. One dtype carries the conditioning from the projection onward.** See
/// `conditioningDType`.
///
/// ## What this type does not own
///
/// No arithmetic. Every formula here belongs to one of the stages, and the composition is
/// deliberately written as calls into those stages rather than as a fused re-derivation —
/// a pipeline that reimplements the connector's masking rule "for clarity" is a second
/// place for that rule to be wrong. The one exception is the layout work either side of
/// the connectors, written out in `connect` so its intermediates are reachable;
/// `EmbeddingsProcessor.callAsFunction` packages the same steps and discards them, and the
/// two must stay in step.
public struct TextConditioningPipeline {

    public typealias Stream = TextConditioningLayout.Stream

    public let tokenizer: GemmaTokenizer
    public let encoder: GemmaTextEncoder
    public let projection: TextProjectionMLX
    public let processor: EmbeddingsProcessor
    public let layout: TextConditioningLayout

    public enum Failure: Error, CustomStringConvertible {
        case audioConnectorMissing
        case maskLengthMismatch(expected: Int, got: Int)

        public var description: String {
            switch self {
            case .audioConnectorMissing:
                return "the DiT checkpoint carries no audio_embeddings_connector, so this "
                    + "pipeline cannot produce enc.projected.audio's downstream taps. The "
                    + "connector weights live in the DiT checkpoint, not the text "
                    + "encoder's — a missing one means the wrong transformer file"
            case let .maskLengthMismatch(expected, got):
                return "the tokenizer produced \(got) positions, expected \(expected)"
            }
        }
    }

    /// Load every stage from the two checkpoints that carry them.
    ///
    /// **Two checkpoints, and which weight lives where is not guessable.** The tokenizer,
    /// Gemma and the projection are all in the text encoder; the two connectors that
    /// consume the projection's output are in the **DiT**. Every instinct says one file,
    /// and nothing in either file says otherwise — you get a `nil` lookup at a path you
    /// had no reason to doubt.
    public init(textEncoder textEncoderURL: URL, dit ditURL: URL) throws {
        let header = try SafetensorsHeader.read(from: textEncoderURL)
        let topology = try TextEncoderTopology.read(header)
        try topology.verify(header)

        self.tokenizer = try GemmaTokenizer.load(fromCheckpoint: textEncoderURL)
        // One map of the 26 GB encoder. The projection is four tensors in the same file;
        // constructing it with its own `MLX.loadArrays` would hold two maps for the
        // length of both Gemma forwards.
        let gemmaWeights = try MLX.loadArrays(url: textEncoderURL)
        self.encoder = GemmaTextEncoder(weights: gemmaWeights, topology: topology)
        self.projection = try TextProjectionMLX(weights: gemmaWeights, topology: topology,
                                                header: header)
        self.processor = try EmbeddingsProcessor(checkpoint: ditURL)
        self.layout = TextConditioningLayout(
            sequenceLength: GemmaTokenizer.conditioningLength)
    }

    /// For a caller that already has the stages loaded — each of the two checkpoints is
    /// tens of gigabytes, so one pipeline is usually shared across many uses.
    public init(tokenizer: GemmaTokenizer, encoder: GemmaTextEncoder,
                projection: TextProjectionMLX, processor: EmbeddingsProcessor,
                layout: TextConditioningLayout = TextConditioningLayout()) {
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.projection = projection
        self.processor = processor
        self.layout = layout
    }

    // MARK: - Prompts

    /// The prompts that are actually encoded, in branch order.
    ///
    /// **This is the trap, and it is a quiet one.** An empty negative prompt is not
    /// encoded as an empty string: `PromptDefaults.negativePrompt` is substituted before
    /// tokenising, and that is 223 real tokens. Taking a caller's `""` literally produces
    /// a correctly-shaped, correctly-padded, well-masked tensor that is wrong, and the
    /// damage only surfaces downstream of classifier-free guidance, where the cause is
    /// nowhere near the symptom.
    ///
    /// The constant is referenced, never retyped. A hand-transcribed copy of it agreed on
    /// only **805 of 1024** ids — close enough to look like a tokenizer bug and nowhere
    /// near right.
    public static func branchPrompts(prompt: String, negativePrompt: String = "") -> [String] {
        [prompt, negativePrompt.isEmpty ? PromptDefaults.negativePrompt : negativePrompt]
    }

    // MARK: - The result

    /// Everything one guidance branch produces, in the order it is computed.
    ///
    /// Held as a struct rather than returned piecemeal because the intermediates are the
    /// point: a pipeline that returned only `features` would be inspectable at exactly one
    /// place, and a wrong tensor there could have come from any of five stages.
    ///
    /// Not `Sendable`: `MLXArray` is a reference type into the MLX runtime and none of the
    /// module types here are either. Marking it would compile only by suppressing the
    /// check, and the pipeline is meant to be used from one context.
    public struct Encoding {

        /// Which guidance branch: 0 positive, 1 negative.
        public let branch: Int
        public let prompt: String

        /// The token ids, one row, left-padded. Integers: exact or wrong.
        public let inputIDs: [Int32]
        /// The per-token weights for this row — the same values as `attentionMask`.
        public let tokenWeights: [Int32]

        /// The attention mask for this branch, `[1, S]`.
        ///
        /// Shape matters here and is easy to get wrong in the *other* direction: this is
        /// per branch and therefore `[1, 1024]`, not the `[2, 1024]` token-weight tensor
        /// that carries both branches at once. Same construction, different tensors;
        /// using one where the other belongs gives a well-formed mask of the wrong rank.
        public let attentionMask: MLXArray

        /// The projection's two streams, `[1, S, width]`, pre-permutation.
        public let projected: [Stream: MLXArray]

        /// `projected` after the right-pad permutation. A pure gather — no arithmetic
        /// happens here, so the values are either the same ones or the wrong ones.
        public let connectorInput: [Stream: MLXArray]
        /// The reordered additive mask, `[1, 1, 1, S]`. One tensor, shared by both
        /// streams: it is the second argument of each connector call.
        public let connectorAdditiveMask: MLXArray

        /// Each connector's encoded output.
        public let connectorOutput: [Stream: MLXArray]
        /// The mask the connector returns, which is all zeros whenever registers are
        /// enabled.
        public let connectorReturnedMask: MLXArray

        /// The conditioning features the DiT consumes, and their mask.
        public let features: [Stream: MLXArray]
        public let featuresMask: MLXArray

        /// Valid (non-padding) tokens: 11 on a short positive prompt, 223 on the default
        /// negative. This is what tells the two branches apart, since they are otherwise
        /// identically shaped.
        public var validTokenCount: Int { tokenWeights.reduce(0) { $0 + Int($1) } }

        /// Qualify a name with this branch's index, e.g. `features.video.call001`.
        public func tap(_ family: String) -> String {
            family + String(format: ".call%03d", branch)
        }
    }

    /// The dtype the conditioning carries from the projection onward.
    ///
    /// fp32. The projection's normalisation is a per-token, per-layer scale, and in bf16
    /// that statistic is not a reproducible quantity — the same formula spelled three
    /// defensible ways disagrees with itself by more than the whole bf16-to-fp32 gap.
    /// Keeping the projection's fp32 result sidesteps the question entirely, and it stays
    /// fp32 through the connector stack. `TextProjectionMLX` documents the arithmetic
    /// this is protecting.
    public static let conditioningDType: DType = .float32

    // MARK: - The forward

    /// Both guidance branches, in branch order.
    ///
    /// Two full passes over a 12 B encoder, deliberately. Batching the two prompts into a
    /// single Gemma forward — `[2, 1024]` in, two `[1, 1024, D]` slices out — would give
    /// the same answer, because the sequences never interact: padding is masked for every
    /// query and attention is causal within a row. Running them one at a time is that same
    /// arithmetic with a different reduction order, and it keeps the encoder's batch-1
    /// shape assumptions intact.
    public func callAsFunction(prompt: String, negativePrompt: String = "") throws
        -> [Encoding] {
        try Self.branchPrompts(prompt: prompt, negativePrompt: negativePrompt)
            .enumerated()
            .map { try encode(prompt: $0.element, branch: $0.offset) }
    }

    /// One branch, from its prompt to everything it produces.
    public func encode(prompt: String, branch: Int) throws -> Encoding {
        let conditioning = try tokenizer.conditioning(for: prompt,
                                                      sequenceLength: layout.sequenceLength)
        guard conditioning.weights.count == layout.sequenceLength else {
            throw Failure.maskLengthMismatch(expected: layout.sequenceLength,
                                             got: conditioning.weights.count)
        }
        return try encode(inputIDs: conditioning.ids, tokenWeights: conditioning.weights,
                          prompt: prompt, branch: branch)
    }

    /// One branch from already-tokenised ids.
    ///
    /// Starting from ids keeps the tokenizer out of the picture: a tokenizer disagreement
    /// and an encoder bug both present as wrong features and they need different fixes.
    /// `GemmaTokenizerTests` owns the tokenizer half.
    public func encode(inputIDs: [Int32], tokenWeights: [Int32],
                       prompt: String = "", branch: Int) throws -> Encoding {
        // The tokenizer's weights column verbatim, not a mask re-derived from
        // `inputIDs != padID`. The two agree today and would stop agreeing the moment a
        // weighted prompt was used, at which point the derived version would silently
        // drop the weighting.
        let maskFloats = tokenWeights.map { Float($0) }
        let attentionMask = MLXArray(maskFloats, [1, tokenWeights.count])

        let hidden = try encoder.hiddenStates(inputIDs: inputIDs,
                                              tokenWeights: maskFloats)
        let (videoWide, audioWide) = try projection(hiddenStates: hidden,
                                                    mask: attentionMask)
        let projected: [Stream: MLXArray] = [
            .video: videoWide.asType(Self.conditioningDType),
            .audio: audioWide.asType(Self.conditioningDType),
        ]
        MLX.eval(projected[.video]!, projected[.audio]!)

        return try connect(projected: projected, attentionMask: attentionMask,
                           inputIDs: inputIDs, tokenWeights: tokenWeights,
                           prompt: prompt, branch: branch)
    }

    /// The layout work either side of the two connectors, written out so its
    /// intermediates are reachable.
    ///
    /// Step for step:
    ///
    /// ```
    /// additive            = (attentionMask - 1) * finfo(bf16).max
    /// order, mask         = rightPadOrder(additive)
    /// videoIn             = reorder(videoFeatures, order)
    /// videoEncoded, vmask = videoConnector(videoIn, mask)
    /// binary              = binaryMask(vmask)
    /// videoEncoded        = videoEncoded * binary
    /// audioIn             = reorder(audioFeatures, order)
    /// audioEncoded, _     = audioConnector(audioIn, mask)     // NOT masked
    /// ```
    ///
    /// The video/audio masking asymmetry is real and deliberate — audio's returned mask is
    /// discarded outright. It costs nothing today because the binary mask is all ones, and
    /// "simplifying" it away would look identical right up until the connector stopped
    /// zeroing its own mask.
    ///
    /// - Parameter projected: the projection's two streams, **pre-permutation**, in
    ///   `conditioningDType`.
    public func connect(projected: [Stream: MLXArray], attentionMask: MLXArray,
                        inputIDs: [Int32] = [], tokenWeights: [Int32] = [],
                        prompt: String = "", branch: Int) throws -> Encoding {
        try Self.connect(projected: projected, attentionMask: attentionMask,
                         processor: processor, inputIDs: inputIDs,
                         tokenWeights: tokenWeights, prompt: prompt, branch: branch)
    }

    /// ``connect(projected:attentionMask:inputIDs:tokenWeights:prompt:branch:)`` without the
    /// text encoder.
    ///
    /// The connector's weights live in the **DiT** checkpoint, so this stage needs nothing
    /// from Gemma. Training reads projected embeds from the cache and has to run the
    /// connector on them every step; going through the instance method would mean holding a
    /// 26 GB encoder for the length of a training run to use none of it.
    ///
    /// The instance method delegates here, so there is one implementation and both entry
    /// points get it.
    public static func connect(projected: [Stream: MLXArray], attentionMask: MLXArray,
                               processor: EmbeddingsProcessor,
                               inputIDs: [Int32] = [], tokenWeights: [Int32] = [],
                               prompt: String = "", branch: Int) throws -> Encoding {
        guard let audioConnector = processor.audio else { throw Failure.audioConnectorMissing }

        let tokens = attentionMask.dim(-1)
        // The additive mask: `(binary - 1)` times `finfo(dtype).max`, in the *features'*
        // dtype. Valid is 0, padding is `-3.3895e+38`. Not `-infinity` — `-inf` times a
        // zero attention weight is NaN, where `-max` is finite and merely saturates.
        let additive = ((attentionMask.asType(.float32) - Float(1))
            .reshaped([1, 1, 1, tokens]) * EmbeddingsProcessor.bf16Max)
            .asType(Self.conditioningDType)

        // The stable partition. Computed once and reused for both streams — two
        // independently computed permutations of the same mask are one refactor away
        // from disagreeing.
        let (order, reordered) = EmbeddingsProcessor.rightPadOrder(additiveMask: additive)

        var connectorInput: [Stream: MLXArray] = [:]
        for (stream, features) in projected {
            // The projection's own output stays fp32 — see `conditioningDType`. The
            // audio connector is a separate boundary, and its residual stream is bf16.
            // Narrowing only the value handed to that connector gives the audio stack
            // the dtype it is built around without throwing away the fp32 projection.
            // The matching seam is in `EmbeddingsProcessor.callAsFunction`; the two have
            // to stay identical.
            let connectorFeatures = stream == .audio ? features.asType(.bfloat16) : features
            connectorInput[stream] = EmbeddingsProcessor.reorder(connectorFeatures, order: order)
        }

        let (videoEncoded, videoMask) = try processor.video(connectorInput[.video]!,
                                                            additiveMask: reordered)
        let binary = EmbeddingsProcessor.binaryMask(connectorMask: videoMask)
        let (audioEncoded, _) = try audioConnector(connectorInput[.audio]!,
                                                   additiveMask: reordered)

        return Encoding(
            branch: branch, prompt: prompt,
            inputIDs: inputIDs, tokenWeights: tokenWeights,
            attentionMask: attentionMask,
            projected: projected,
            connectorInput: connectorInput,
            connectorAdditiveMask: reordered,
            connectorOutput: [.video: videoEncoded, .audio: audioEncoded],
            connectorReturnedMask: videoMask,
            features: [.video: videoEncoded * binary.asType(videoEncoded.dtype),
                       .audio: audioEncoded],
            featuresMask: binary.reshaped([binary.dim(0), binary.dim(1)]))
    }
}

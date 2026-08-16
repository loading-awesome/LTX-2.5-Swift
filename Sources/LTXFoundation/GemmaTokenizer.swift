// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// The Gemma-4 tokenizer, read out of the text-encoder checkpoint.
///
/// This is the first thing on the render path: nothing downstream runs until a prompt
/// becomes `input_ids`. Everything it needs is in
/// `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` — the HuggingFace `tokenizer.json`
/// is a 32 MB `tokenizer_json` byte tensor and the sidecar `tokenizer_config.json` is
/// an `hf_asset__` tensor beside it. Nothing external is downloaded, and no id here is
/// written down that the checkpoint could have been asked for.
///
/// ## What it is
///
/// A SentencePiece-style BPE: 262144 vocabulary entries, 514906 ranked merges, byte
/// fallback, `fuse_unk`. The merge loop is implemented rather than approximated,
/// because the loop is the only part that generalises — anything that reproduces a
/// handful of known strings is a lookup table, not a tokenizer.
///
/// ## Traps, each of which leaves every shape intact
///
/// 1. **The normalizer has no `Prepend`.** Gemma 3's tokenizer prepends `▁` so the
///    first word carries a word-boundary mark; this one only replaces `U+0020` with
///    `▁`. Measured: `"A red cube…"` starts `236776` (`"A"`), not the id of `"▁A"`. A
///    port that copies a Gemma 3 normalizer gets one id wrong in 1024 — an error no
///    length, mask or shape check can see.
/// 2. **BOS comes from the wrapper, not the post-processor.** Gemma 4's
///    `post_processor` is an identity `TemplateProcessing`; the layer above it
///    prepends `<bos>` itself. Trusting the post-processor loses BOS; doing both
///    duplicates it.
/// 3. **Padding is on the LEFT and the pad id is 0.** A short prompt's 11 real tokens
///    sit at indices 1013…1023 of 1024. Zero is also what an uninitialised buffer holds,
///    so a port that pads on the right produces a tensor of the correct shape, holding
///    the correct ids, in the wrong 1013 positions.
///    `TextConditioningLayout.rightPadOrder` normalises this downstream, and it is only
///    correct if the mask it is handed actually marks the left padding.
/// 4. **`enc.token_weights` is the attention mask, not prompt weighting.** The name
///    invites an A1111-style `(word:1.2)` implementation. It holds only 0.0 and 1.0 and
///    is the same row as `enc.attention_mask`, because both come from the same
///    `(token_id, attention)` pairs. A port that implements emphasis syntax produces a
///    plausible non-binary vector.
/// 5. **Symbols are Unicode scalars, not Swift `Character`s.** `"é"` written as
///    `e` + `U+0301` is two symbols to this algorithm and one grapheme cluster in Swift;
///    grouping it as one changes the tokenisation of every decomposed string. Measured:
///    precomposed gives `[…, 1559]`, decomposed `[…, 545, 238288]`.
/// 6. **Truncation happens before BOS is forced, then again after.** A prompt longer
///    than the sequence length keeps its first 1023 tokens after BOS is prepended — it
///    loses its *tail*, not its head.
///
/// ## Why the vocabulary is not parsed with `JSONSerialization`
///
/// Two measured facts, both of which corrupt a `[String: Int]` vocabulary silently:
///
/// * Swift `String` hashes and compares under **canonical equivalence**, so
///   `"▁là"` spelled `U+00E0` and `"▁là"` spelled `a` + `U+0300` are the same key. This
///   vocabulary contains 434 such pairs — Vietnamese, Bengali, Arabic, Devanagari —
///   with *different ids*. Bridged into a Swift dictionary, 262141 entries become
///   261707 and the survivor's id is whichever was inserted last.
/// * `String(data:encoding:.utf8)`, the path `JSONSerialization` decodes through,
///   **strips a leading `U+FEFF`**. Five vocabulary entries begin with one, and each
///   silently aliases onto another entry.
///
/// A tokenizer vocabulary is keyed by *bytes*, not by canonically-equivalent strings.
/// So the scanner below decodes the vocabulary and merge table straight from UTF-8 and
/// keys them by bytes, which is that same relation. Everything else in the file — the
/// normalizer, the pre-tokenizer, the added tokens — is small, plain ASCII, and goes
/// through `JSONSerialization` as usual.
///
/// Exercised by `GemmaTokenizerTests` over a corpus of byte-fallback, combining-mark,
/// CJK, embedded-added-token and over-length cases.
public struct GemmaTokenizer: Sendable {

    // MARK: - Failures

    public enum Failure: Error, CustomStringConvertible {
        case malformedJSON(String, atByte: Int)
        case notJSON(String)
        case missingField(String)
        case unsupportedModel(String)
        case unsupportedComponent(component: String, found: String)
        case unsupportedAddedToken(String)
        case unknownSpecialToken(role: String, content: String)
        case missingAsset(String)
        case badSequenceLength(Int)

        public var description: String {
            switch self {
            case let .malformedJSON(what, offset):
                return "tokenizer_json is malformed at byte \(offset): \(what)"
            case let .notJSON(what):
                return "\(what) is not valid JSON"
            case let .missingField(name):
                return "tokenizer.json has no \(name)"
            case let .unsupportedModel(type):
                return "tokenizer model type \(type) is not BPE. This port implements the "
                    + "merge loop only; a WordPiece or Unigram model needs a different "
                    + "algorithm, not a different table"
            case let .unsupportedComponent(component, found):
                return "\(component) is \(found), which this port does not implement. "
                    + "Refusing rather than ignoring it: an unapplied normalizer or "
                    + "pre-tokenizer still produces ids, just the wrong ones, and nothing "
                    + "downstream can tell"
            case let .unsupportedAddedToken(content):
                return "added token \(content) sets lstrip/rstrip/single_word or is "
                    + "normalized; this port matches added tokens literally against the "
                    + "raw text and would place it differently"
            case let .unknownSpecialToken(role, content):
                return "tokenizer_config names \(content) as the \(role) token but the "
                    + "vocabulary has no such entry"
            case let .missingAsset(name):
                return "the checkpoint has no \(name). The tokenizer ships inside the "
                    + "text-encoder file; without it there is nothing to substitute"
            case let .badSequenceLength(n):
                return "sequence length \(n) is not positive"
            }
        }
    }

    // MARK: - Model

    fileprivate struct Merge: Sendable {
        let rank: Int32
        let result: Int32
    }

    private struct AddedToken: Sendable {
        let content: String
        /// Matching happens over scalars, not `Character`s — see `encode`.
        let scalars: [Unicode.Scalar]
        let id: Int32
    }

    /// What the normalizer does, in order. Read from the file rather than assumed, so
    /// pointing this at a differently-normalised tokenizer fails instead of drifting.
    private enum NormalizerStep: Sendable {
        case prepend([Unicode.Scalar])
        case replace(pattern: [Unicode.Scalar], with: [Unicode.Scalar])
    }

    /// `Split` pre-tokenizer behaviours, named as `tokenizer.json` spells them.
    private enum SplitBehavior: String, Sendable {
        case removed = "Removed"
        case isolated = "Isolated"
        case mergedWithPrevious = "MergedWithPrevious"
        case mergedWithNext = "MergedWithNext"
    }

    /// Single-scalar vocabulary entries, which is all the merge loop ever looks up.
    ///
    /// The full byte-keyed vocabulary is needed to build the merge table and then
    /// discarded: every symbol enters the loop as one scalar or as byte-fallback
    /// tokens, and everything longer is reached by merging.
    private let scalarIDs: [UInt32: Int32]
    /// Packed `(left << 32) | right` to a merge, so the hot loop hashes integers.
    private let merges: [UInt64: Merge]
    /// Longest-first, which is how leftmost-longest matching is implemented below.
    private let addedTokens: [AddedToken]
    /// `<0x00>`…`<0xFF>`, present on this checkpoint; `nil` when the model declares no
    /// byte fallback.
    private let byteTokens: [Int32]?
    private let unknownID: Int32?
    private let fusesUnknown: Bool
    private let normalizerSteps: [NormalizerStep]
    private let splitPattern: [Unicode.Scalar]
    private let splitBehavior: SplitBehavior

    public let vocabularySize: Int
    public let bosID: Int32
    public let padID: Int32

    /// Contract 9's `min_length`, and the tokenizer's maximum length. The same 1024 does
    /// both jobs, so every encoding is exactly this long.
    public static let conditioningLength = 1024

    // MARK: - Loading

    /// Build from the two byte blobs the checkpoint carries.
    ///
    /// `tokenizerConfig` is required rather than optional: it is the only place that
    /// says which vocabulary entry is BOS and which is PAD. Those ids could be guessed
    /// from the strings `<bos>` and `<pad>`, but a constant the checkpoint can be asked
    /// for should be read from it, not written down here.
    public init(tokenizerJSON: Data, tokenizerConfig: Data) throws {
        let spec = try TokenizerSpec(json: tokenizerJSON)
        guard let config = try? JSONSerialization.jsonObject(with: tokenizerConfig)
            as? [String: Any] else { throw Failure.notJSON("tokenizer_config.json") }

        // Truncation and padding are null in this file; the wrapper supplies both. If a
        // checkpoint ever set them, encode() would silently apply a second, different
        // policy underneath the wrapper's.
        for key in ["truncation", "padding"] {
            if let node = spec.member(key), !(node is NSNull) {
                throw Failure.unsupportedComponent(component: key, found: "\(node)")
            }
        }

        guard (spec.member("model.type") as? String) == "BPE" else {
            throw Failure.unsupportedModel((spec.member("model.type") as? String) ?? "absent")
        }
        // Both would change which merges apply. `dropout` is stochastic and has no place
        // in an inference path at all.
        if let dropout = spec.member("model.dropout") as? Double, dropout != 0 {
            throw Failure.unsupportedComponent(component: "model.dropout", found: "\(dropout)")
        }
        if (spec.member("model.ignore_merges") as? Bool) == true {
            throw Failure.unsupportedComponent(component: "model.ignore_merges", found: "true")
        }
        for key in ["model.continuing_subword_prefix", "model.end_of_word_suffix"] {
            if let affix = spec.member(key) as? String, !affix.isEmpty {
                throw Failure.unsupportedComponent(component: key, found: affix)
            }
        }

        let vocabulary = spec.vocabulary
        guard !vocabulary.isEmpty else { throw Failure.missingField("model.vocab") }
        self.vocabularySize = vocabulary.count

        var scalars = [UInt32: Int32](minimumCapacity: 1 << 16)
        for (bytes, id) in vocabulary {
            guard let scalar = TokenizerSpec.singleScalar(bytes) else { continue }
            scalars[scalar] = id
        }
        self.scalarIDs = scalars
        self.merges = try spec.mergeTable(vocabulary: vocabulary)

        var added: [AddedToken] = []
        for entry in (spec.member("added_tokens") as? [[String: Any]]) ?? [] {
            guard let content = entry["content"] as? String,
                  let id = entry["id"] as? Int else { continue }
            let plain = (entry["lstrip"] as? Bool) != true
                && (entry["rstrip"] as? Bool) != true
                && (entry["single_word"] as? Bool) != true
                && (entry["normalized"] as? Bool) != true
            guard plain else { throw Failure.unsupportedAddedToken(content) }
            added.append(AddedToken(content: content,
                                    scalars: Array(content.unicodeScalars),
                                    id: Int32(id)))
        }
        // Leftmost-longest: at each position the longest candidate wins. Sorting once
        // here makes "first match in this array" mean "longest match".
        added.sort { $0.scalars.count > $1.scalars.count }
        self.addedTokens = added

        self.fusesUnknown = (spec.member("model.fuse_unk") as? Bool) == true
        self.unknownID = (spec.member("model.unk_token") as? String)
            .flatMap { vocabulary[Array($0.utf8)] }
        if (spec.member("model.byte_fallback") as? Bool) == true {
            var bytes: [Int32] = []
            bytes.reserveCapacity(256)
            for byte in 0..<256 {
                // Built by hand rather than with String(format:): the vocabulary spells
                // these `<0x0A>`, uppercase and zero-padded, and printf-style formatting
                // has already cost this repository one segfault.
                let hex = String(byte, radix: 16, uppercase: true)
                let name = "<0x" + (hex.count == 1 ? "0" + hex : hex) + ">"
                guard let id = vocabulary[Array(name.utf8)] else {
                    throw Failure.unsupportedComponent(
                        component: "model.byte_fallback",
                        found: "set, but the vocabulary is missing \(name)")
                }
                bytes.append(id)
            }
            self.byteTokens = bytes
        } else {
            self.byteTokens = nil
        }

        self.normalizerSteps = try GemmaTokenizer.normalizer(spec.member("normalizer"))
        let split = try GemmaTokenizer.splitRule(spec.member("pre_tokenizer"))
        self.splitPattern = split.pattern
        self.splitBehavior = split.behavior
        try GemmaTokenizer.requireIdentityPostProcessor(spec.member("post_processor"))

        // `bos_token` / `pad_token` may be a bare string or the serialised AddedToken
        // object; both spellings appear across tokenizer_config versions.
        func specialID(_ role: String, fallback: String?) throws -> Int32 {
            var content = (config[role] as? String)
                ?? ((config[role] as? [String: Any])?["content"] as? String)
            if content == nil { content = fallback }
            guard let content else { throw Failure.missingField("tokenizer_config.\(role)") }
            guard let id = vocabulary[Array(content.utf8)] else {
                throw Failure.unknownSpecialToken(role: role, content: content)
            }
            return id
        }
        self.bosID = try specialID("bos_token", fallback: nil)
        // When no pad token is declared the EOS token is used instead, which changes the
        // id that fills 1013 of 1024 positions.
        self.padID = try specialID("pad_token", fallback: config["eos_token"] as? String)
    }

    /// Read both assets straight out of the text-encoder checkpoint.
    public static func load(fromCheckpoint url: URL) throws -> GemmaTokenizer {
        let reader = try SafetensorsReader(url: url)
        guard let json = try reader.utf8String("tokenizer_json") else {
            throw Failure.missingAsset("tokenizer_json tensor")
        }
        guard let config = try reader.utf8String("hf_asset__tokenizer_config.json") else {
            throw Failure.missingAsset("hf_asset__tokenizer_config.json tensor")
        }
        return try GemmaTokenizer(tokenizerJSON: Data(json.utf8),
                                  tokenizerConfig: Data(config.utf8))
    }

    // MARK: - Component parsing

    private static func normalizer(_ node: Any?) throws -> [NormalizerStep] {
        guard let node = node as? [String: Any] else { return [] }
        switch node["type"] as? String {
        case "Sequence":
            var steps: [NormalizerStep] = []
            for child in (node["normalizers"] as? [Any]) ?? [] {
                steps.append(contentsOf: try normalizer(child))
            }
            return steps
        case "Prepend":
            return [.prepend(Array(((node["prepend"] as? String) ?? "").unicodeScalars))]
        case "Replace":
            guard let pattern = (node["pattern"] as? [String: Any])?["String"] as? String,
                  let content = node["content"] as? String else {
                throw Failure.unsupportedComponent(component: "normalizer.Replace",
                                                   found: "a non-literal pattern")
            }
            return [.replace(pattern: Array(pattern.unicodeScalars),
                             with: Array(content.unicodeScalars))]
        default:
            throw Failure.unsupportedComponent(component: "normalizer",
                                               found: (node["type"] as? String) ?? "unknown")
        }
    }

    private static func splitRule(_ node: Any?)
        throws -> (pattern: [Unicode.Scalar], behavior: SplitBehavior) {
        guard let node = node as? [String: Any] else { return ([], .isolated) }
        guard (node["type"] as? String) == "Split" else {
            throw Failure.unsupportedComponent(component: "pre_tokenizer",
                                               found: (node["type"] as? String) ?? "unknown")
        }
        guard let pattern = (node["pattern"] as? [String: Any])?["String"] as? String else {
            throw Failure.unsupportedComponent(component: "pre_tokenizer.Split",
                                               found: "a Regex pattern")
        }
        guard let behavior = SplitBehavior(rawValue: (node["behavior"] as? String) ?? "") else {
            throw Failure.unsupportedComponent(
                component: "pre_tokenizer.Split.behavior",
                found: (node["behavior"] as? String) ?? "absent")
        }
        if (node["invert"] as? Bool) == true {
            throw Failure.unsupportedComponent(component: "pre_tokenizer.Split.invert",
                                               found: "true")
        }
        return (Array(pattern.unicodeScalars), behavior)
    }

    /// Gemma 4's post-processor adds nothing — and that is the point.
    ///
    /// It is a `TemplateProcessing` whose single-sequence template is just the sequence
    /// and whose `special_tokens` map is empty. Gemma 3's emits `<bos>`. The wrapper
    /// compensates by prepending BOS unconditionally, so a port that also honoured a
    /// BOS-emitting template would double it.
    private static func requireIdentityPostProcessor(_ node: Any?) throws {
        guard let node = node as? [String: Any] else { return }
        let type = (node["type"] as? String) ?? "unknown"
        guard type == "TemplateProcessing" else {
            throw Failure.unsupportedComponent(component: "post_processor", found: type)
        }
        let specials = (node["special_tokens"] as? [String: Any]) ?? [:]
        let single = (node["single"] as? [Any]) ?? []
        let isIdentity = specials.isEmpty && single.count == 1
            && (single[0] as? [String: Any])?["Sequence"] != nil
        guard isIdentity else {
            throw Failure.unsupportedComponent(
                component: "post_processor",
                found: "a TemplateProcessing that inserts \(specials.count) special "
                    + "token(s); the wrapper already prepends BOS")
        }
    }

    // MARK: - Encoding

    @inline(__always)
    fileprivate static func key(_ left: Int32, _ right: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
    }

    /// Tokenise, with no BOS, truncation or padding — the model's own output.
    ///
    /// Everything below works on `Unicode.Scalar` arrays rather than on `String`. That
    /// is not a performance choice: `String`'s `hasPrefix`, `contains` and
    /// `replacingOccurrences` compare *grapheme clusters* under canonical equivalence,
    /// so `"<bos>"` would fail to match a `">"` carrying a combining mark, and `" "`
    /// would fail to match a space that begins a cluster. The algorithm is defined over
    /// scalars throughout, and the two views disagree on exactly the inputs a prompt box
    /// makes easy to type.
    public func encode(_ text: String) -> [Int32] {
        let scalars = Array(text.unicodeScalars)
        var out: [Int32] = []
        var pending: [Unicode.Scalar] = []

        func flush() {
            guard !pending.isEmpty else { return }
            for word in preTokenize(normalize(pending)) {
                appendMerged(word, into: &out)
            }
            pending.removeAll(keepingCapacity: true)
        }

        // Added tokens are matched against the *raw* text, before normalisation — all 24
        // of them are `normalized: false`. So `<bos>` typed into a prompt becomes id 2
        // rather than being spelled out, and the text on either side is normalised
        // separately, which is why a space before an added token survives as its own `▁`.
        var index = 0
        outer: while index < scalars.count {
            for token in addedTokens where matches(token.scalars, in: scalars, at: index) {
                flush()
                out.append(token.id)
                index += token.scalars.count
                continue outer
            }
            pending.append(scalars[index])
            index += 1
        }
        flush()
        return out
    }

    /// Encode into the fixed-length, left-padded form the text encoder consumes.
    ///
    /// - Parameter sequenceLength: contract 9's 1024. It serves as both `max_length` and
    ///   `min_length`, so one number truncates and pads.
    public func conditioning(for text: String,
                             sequenceLength: Int = GemmaTokenizer.conditioningLength)
        throws -> Conditioning {
        guard sequenceLength > 0 else { throw Failure.badSequenceLength(sequenceLength) }

        var ids = encode(GemmaTokenizer.strip(text))
        if ids.count > sequenceLength { ids.removeSubrange(sequenceLength...) }
        // Truncate, force BOS, truncate again: prepending BOS to an already-full
        // sequence drops the last token, not the first.
        if ids.first != bosID {
            ids.insert(bosID, at: 0)
            if ids.count > sequenceLength { ids.removeSubrange(sequenceLength...) }
        }

        let padCount = sequenceLength - ids.count
        return Conditioning(
            ids: [Int32](repeating: padID, count: padCount) + ids,
            weights: [Int32](repeating: 0, count: padCount)
                + [Int32](repeating: 1, count: ids.count))
    }

    /// One prompt's tokenizer output: the two tensors the encoder is called with.
    public struct Conditioning: Sendable, Equatable {
        /// `enc.input_ids`, one row. Left-padded.
        public let ids: [Int32]
        /// `enc.token_weights`, one row — and the same values as `enc.attention_mask`.
        ///
        /// Both come from the same `(token_id, attention)` pairs, split into `input_ids`
        /// and a mask, which is why the two rows are identical. It is *not* prompt
        /// emphasis.
        public let weights: [Int32]

        /// The mask in the form `TextConditioningLayout` wants.
        public var mask: [Bool] { weights.map { $0 != 0 } }
        public var validTokenCount: Int { weights.reduce(0) { $0 + Int($1) } }
    }

    // MARK: - Pipeline stages

    private func matches(_ needle: [Unicode.Scalar], in haystack: [Unicode.Scalar],
                         at index: Int) -> Bool {
        guard index + needle.count <= haystack.count else { return false }
        for offset in 0..<needle.count where haystack[index + offset] != needle[offset] {
            return false
        }
        return true
    }

    private func normalize(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var out = text
        for step in normalizerSteps {
            switch step {
            case let .prepend(prefix):
                out.insert(contentsOf: prefix, at: 0)
            case let .replace(pattern, replacement):
                guard !pattern.isEmpty else { continue }
                var replaced: [Unicode.Scalar] = []
                replaced.reserveCapacity(out.count)
                var index = 0
                while index < out.count {
                    if matches(pattern, in: out, at: index) {
                        replaced.append(contentsOf: replacement)
                        index += pattern.count
                    } else {
                        replaced.append(out[index])
                        index += 1
                    }
                }
                out = replaced
            }
        }
        return out
    }

    /// Split the normalised string on the pre-tokenizer's literal pattern.
    ///
    /// On this checkpoint the pattern is `" "` and the normalizer has already turned
    /// every space into `▁`, so this never fires and the whole prompt is one word — BPE
    /// runs across word boundaries, which is why `"double  space"` merges the two spaces
    /// into a single `▁▁` token. That is a property of the two components together, not
    /// an assumption, so the split is implemented rather than skipped: a checkpoint that
    /// dropped the `Replace` would need it.
    private func preTokenize(_ text: [Unicode.Scalar]) -> [[Unicode.Scalar]] {
        let pattern = splitPattern
        guard !pattern.isEmpty else { return text.isEmpty ? [] : [text] }

        var pieces: [[Unicode.Scalar]] = []
        var current: [Unicode.Scalar] = []
        var index = 0
        while index < text.count {
            if matches(pattern, in: text, at: index) {
                switch splitBehavior {
                case .removed:
                    if !current.isEmpty { pieces.append(current) }
                    current = []
                case .isolated:
                    if !current.isEmpty { pieces.append(current) }
                    pieces.append(pattern)
                    current = []
                case .mergedWithPrevious:
                    current.append(contentsOf: pattern)
                    pieces.append(current)
                    current = []
                case .mergedWithNext:
                    if !current.isEmpty { pieces.append(current) }
                    current = pattern
                }
                index += pattern.count
            } else {
                current.append(text[index])
                index += 1
            }
        }
        // Empty pieces are dropped rather than tokenised.
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }

    /// Resolve a word to symbols, then merge.
    private func appendMerged(_ word: [Unicode.Scalar], into out: inout [Int32]) {
        var ids: [Int32] = []
        ids.reserveCapacity(word.count)

        // One symbol per scalar. Grouping by `Character` instead would fuse combining
        // marks and emoji ZWJ sequences into single elements and push each through byte
        // fallback as one blob, rather than as the several vocabulary entries they are.
        for scalar in word {
            if let id = scalarIDs[scalar.value] {
                ids.append(id)
                continue
            }
            if let byteTokens {
                for byte in String(scalar).utf8 { ids.append(byteTokens[Int(byte)]) }
                continue
            }
            guard let unknownID else { continue }
            // `fuse_unk` collapses a run of unknown scalars into a single <unk>.
            if fusesUnknown, ids.last == unknownID { continue }
            ids.append(unknownID)
        }
        guard ids.count > 1 else {
            out.append(contentsOf: ids)
            return
        }
        merge(&ids)
        out.append(contentsOf: ids)
    }

    /// The merge loop: lowest rank first, ties to the leftmost position.
    ///
    /// A doubly-linked list over the symbol array plus a priority queue. The naive
    /// alternative — rescan for the best pair each
    /// round — is O(n²) on a 1024-token prompt and, worse, gets the tie-break wrong
    /// unless it also scans left to right, which is not observable on short prompts.
    /// Queue entries are never deleted; a stale one is recognised because the pair at
    /// that position no longer produces the id it was queued for.
    private func merge(_ ids: inout [Int32]) {
        let n = ids.count
        var next = Array(1...n)              // next[i] == n means "no successor"
        var previous = Array(-1..<(n - 1))
        var alive = [Bool](repeating: true, count: n)

        var queue = MergeQueue()
        queue.reserve(n)
        for i in 0..<(n - 1) {
            if let m = merges[GemmaTokenizer.key(ids[i], ids[i + 1])] {
                queue.push(rank: m.rank, position: i, result: m.result)
            }
        }

        var removed = 0
        while let top = queue.pop() {
            let position = top.position
            guard alive[position], next[position] < n, alive[next[position]] else { continue }
            let right = next[position]
            guard let m = merges[GemmaTokenizer.key(ids[position], ids[right])],
                  m.result == top.result else { continue }

            ids[position] = m.result
            alive[right] = false
            removed += 1
            let after = next[right]
            next[position] = after
            if after < n { previous[after] = position }

            let left = previous[position]
            if left >= 0, let m = merges[GemmaTokenizer.key(ids[left], ids[position])] {
                queue.push(rank: m.rank, position: left, result: m.result)
            }
            if after < n, let m = merges[GemmaTokenizer.key(ids[position], ids[after])] {
                queue.push(rank: m.rank, position: position, result: m.result)
            }
        }

        guard removed > 0 else { return }
        var compacted: [Int32] = []
        compacted.reserveCapacity(n - removed)
        for i in 0..<n where alive[i] { compacted.append(ids[i]) }
        ids = compacted
    }

    /// Leading and trailing whitespace removed, over Python's set rather than Swift's.
    ///
    /// A prompt is stripped before it is tokenised, and the set that has to be stripped
    /// is every character Python calls a space. That set is not Swift's
    /// `.whitespacesAndNewlines`: it includes the file/group/record/unit separators
    /// `U+001C`…`U+001F` and `U+0085`, which Swift's set omits. Leaving one of those on
    /// a prompt changes its first or last token.
    static func strip(_ text: String) -> String {
        func isPythonSpace(_ scalar: Unicode.Scalar) -> Bool {
            switch scalar.value {
            case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680,
                 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
                return true
            default:
                return false
            }
        }
        let scalars = Array(text.unicodeScalars)
        var start = 0
        var end = scalars.count
        while start < end, isPythonSpace(scalars[start]) { start += 1 }
        while end > start, isPythonSpace(scalars[end - 1]) { end -= 1 }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars[start..<end])
        return String(view)
    }
}

// MARK: - Byte-exact reading of tokenizer.json

/// Reads the parts of `tokenizer.json` that must be byte-exact, and hands the rest to
/// `JSONSerialization`.
///
/// The vocabulary and the merge table are decoded from UTF-8 into `[UInt8]` keys, which
/// is the relation a tokenizer vocabulary is actually defined over. See the type comment
/// on `GemmaTokenizer` for the two ways a `[String: Int]` vocabulary loses entries — both
/// measured on this file, both silent.
///
/// Every other member is captured as its raw byte range and parsed on demand, so the
/// 32 MB document is walked once and nothing but the vocabulary is retained.
private struct TokenizerSpec {

    /// Byte-keyed, which is how the vocabulary is defined.
    private(set) var vocabulary: [[UInt8]: Int32] = [:]
    private var members: [String: Data] = [:]
    private var mergeRange: Range<Int>?
    private let bytes: [UInt8]

    init(json: Data) throws {
        self.bytes = [UInt8](json)
        try scanRoot()
    }

    /// A small member, parsed with `JSONSerialization` on demand.
    ///
    /// Fragments are allowed because several of these are bare strings or `null`.
    func member(_ name: String) -> Any? {
        guard let data = members[name] else { return nil }
        return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    }

    /// The scalar value of a vocabulary entry that is exactly one scalar, else `nil`.
    static func singleScalar(_ bytes: [UInt8]) -> UInt32? {
        var decoder = UTF8()
        var iterator = bytes.makeIterator()
        guard case let .scalarValue(scalar) = decoder.decode(&iterator) else { return nil }
        guard case .emptyInput = decoder.decode(&iterator) else { return nil }
        return scalar.value
    }

    /// Resolve the merge list into `(pair) -> (rank, result)`.
    ///
    /// Re-walks the recorded byte range instead of holding 514906 decoded pairs: only
    /// the resolved integers survive.
    func mergeTable(vocabulary: [[UInt8]: Int32]) throws -> [UInt64: GemmaTokenizer.Merge] {
        guard let range = mergeRange else {
            throw GemmaTokenizer.Failure.missingField("model.merges")
        }
        var reader = TokenizerSpec.Reader(bytes: bytes, index: range.lowerBound)
        var table = [UInt64: GemmaTokenizer.Merge](minimumCapacity: 1 << 20)
        var rank: Int32 = 0

        try reader.expect(UInt8(ascii: "["))
        try reader.skipWhitespace()
        if reader.peek() == UInt8(ascii: "]") { return table }
        while true {
            // Both spellings exist in the wild: ["a", "b"] and "a b". This file uses the
            // pair form; the joined form splits on the first space.
            var left: [UInt8] = []
            var right: [UInt8] = []
            try reader.skipWhitespace()
            if reader.peek() == UInt8(ascii: "[") {
                try reader.expect(UInt8(ascii: "["))
                left = try reader.readString()
                try reader.expect(UInt8(ascii: ","))
                right = try reader.readString()
                try reader.expect(UInt8(ascii: "]"))
            } else {
                let joined = try reader.readString()
                if let space = joined.firstIndex(of: UInt8(ascii: " ")) {
                    left = Array(joined[..<space])
                    right = Array(joined[(space + 1)...])
                }
            }
            if let a = vocabulary[left], let b = vocabulary[right],
               let result = vocabulary[left + right] {
                // A duplicated pair keeps the *last* entry, which is what building the
                // map by insertion gives. Measured: this file has no duplicates.
                table[GemmaTokenizer.key(a, b)] = GemmaTokenizer.Merge(rank: rank,
                                                                       result: result)
            }
            rank += 1
            if try !reader.advancePastSeparator(closing: UInt8(ascii: "]")) { return table }
        }
    }

    // MARK: Scanning

    private mutating func scanRoot() throws {
        var reader = Reader(bytes: bytes, index: 0)
        try reader.expect(UInt8(ascii: "{"))
        try reader.skipWhitespace()
        if reader.peek() == UInt8(ascii: "}") { return }

        while true {
            let name = String(decoding: try reader.readString(), as: UTF8.self)
            try reader.expect(UInt8(ascii: ":"))
            if name == "model" {
                try scanModel(&reader)
            } else {
                members[name] = Data(bytes[try reader.skipValue()])
            }
            if try !reader.advancePastSeparator(closing: UInt8(ascii: "}")) { break }
        }
    }

    private mutating func scanModel(_ reader: inout Reader) throws {
        try reader.expect(UInt8(ascii: "{"))
        try reader.skipWhitespace()
        if reader.peek() == UInt8(ascii: "}") { reader.index += 1; return }

        while true {
            let field = String(decoding: try reader.readString(), as: UTF8.self)
            try reader.expect(UInt8(ascii: ":"))
            switch field {
            case "vocab":
                vocabulary = try reader.readVocabulary()
            case "merges":
                // Recorded, not decoded: resolving the pairs needs the vocabulary, and
                // the vocabulary may be written after the merges.
                mergeRange = try reader.skipValue()
            default:
                members["model." + field] = Data(bytes[try reader.skipValue()])
            }
            if try !reader.advancePastSeparator(closing: UInt8(ascii: "}")) { return }
        }
    }

    /// A byte-level JSON cursor. Only what this file needs: objects, arrays, strings,
    /// integers, and skipping anything else.
    private struct Reader {
        let bytes: [UInt8]
        var index: Int

        func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

        mutating func skipWhitespace() throws {
            while index < bytes.count {
                switch bytes[index] {
                case 0x20, 0x09, 0x0A, 0x0D: index += 1
                default: return
                }
            }
        }

        mutating func expect(_ byte: UInt8) throws {
            try skipWhitespace()
            guard index < bytes.count, bytes[index] == byte else {
                throw GemmaTokenizer.Failure.malformedJSON(
                    "expected \(Character(UnicodeScalar(byte)))", atByte: index)
            }
            index += 1
        }

        /// Consume the `,` or the closing bracket after a member. Returns false at the
        /// end of the container.
        ///
        /// Spelled as a step rather than as a `forEach`-style closure on purpose: a
        /// closure that also reads from this reader is two overlapping accesses to the
        /// same `inout` value, which Swift's exclusivity checking rejects outright.
        mutating func advancePastSeparator(closing: UInt8) throws -> Bool {
            try skipWhitespace()
            guard let byte = peek() else {
                throw GemmaTokenizer.Failure.malformedJSON("unterminated container",
                                                          atByte: index)
            }
            index += 1
            if byte == closing { return false }
            guard byte == UInt8(ascii: ",") else {
                throw GemmaTokenizer.Failure.malformedJSON("expected , or a closing bracket",
                                                          atByte: index - 1)
            }
            return true
        }

        mutating func readVocabulary() throws -> [[UInt8]: Int32] {
            try expect(UInt8(ascii: "{"))
            // 262144 entries on this checkpoint; reserving avoids ~18 rehashes.
            var vocab = [[UInt8]: Int32](minimumCapacity: 1 << 19)
            try skipWhitespace()
            if peek() == UInt8(ascii: "}") { index += 1; return vocab }
            while true {
                let token = try readString()
                try expect(UInt8(ascii: ":"))
                vocab[token] = try readInt()
                if try !advancePastSeparator(closing: UInt8(ascii: "}")) { return vocab }
            }
        }

        /// Decode a JSON string to its exact UTF-8 bytes.
        ///
        /// `\uXXXX` escapes are decoded here, surrogate pairs included, because the
        /// vocabulary's control-character entries are written that way and a byte-level
        /// copy would keep them as six literal ASCII characters.
        mutating func readString() throws -> [UInt8] {
            try skipWhitespace()
            try expect(UInt8(ascii: "\""))
            var out: [UInt8] = []
            var run = index
            while index < bytes.count {
                let byte = bytes[index]
                if byte == UInt8(ascii: "\"") {
                    out.append(contentsOf: bytes[run..<index])
                    index += 1
                    return out
                }
                if byte == UInt8(ascii: "\\") {
                    out.append(contentsOf: bytes[run..<index])
                    index += 1
                    try readEscape(into: &out)
                    run = index
                    continue
                }
                index += 1
            }
            throw GemmaTokenizer.Failure.malformedJSON("unterminated string", atByte: index)
        }

        private mutating func readEscape(into out: inout [UInt8]) throws {
            guard index < bytes.count else {
                throw GemmaTokenizer.Failure.malformedJSON("truncated escape", atByte: index)
            }
            let byte = bytes[index]
            index += 1
            switch byte {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(byte)
            case UInt8(ascii: "b"): out.append(0x08)
            case UInt8(ascii: "f"): out.append(0x0C)
            case UInt8(ascii: "n"): out.append(0x0A)
            case UInt8(ascii: "r"): out.append(0x0D)
            case UInt8(ascii: "t"): out.append(0x09)
            case UInt8(ascii: "u"):
                var value = UInt32(try readHex4())
                if value >= 0xD800, value <= 0xDBFF,
                   index + 1 < bytes.count, bytes[index] == UInt8(ascii: "\\"),
                   bytes[index + 1] == UInt8(ascii: "u") {
                    let mark = index
                    index += 2
                    let low = UInt32(try readHex4())
                    if low >= 0xDC00, low <= 0xDFFF {
                        value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00)
                    } else {
                        index = mark
                    }
                }
                // A lone surrogate cannot be encoded; the replacement character is what
                // every JSON reader in this pipeline produces for one.
                let scalar = Unicode.Scalar(value) ?? Unicode.Scalar(0xFFFD)!
                out.append(contentsOf: Array(String(scalar).utf8))
            default:
                throw GemmaTokenizer.Failure.malformedJSON("unknown escape", atByte: index - 1)
            }
        }

        private mutating func readHex4() throws -> UInt16 {
            guard index + 4 <= bytes.count else {
                throw GemmaTokenizer.Failure.malformedJSON("truncated \\u", atByte: index)
            }
            var value: UInt16 = 0
            for _ in 0..<4 {
                let byte = bytes[index]
                let digit: UInt16
                switch byte {
                case UInt8(ascii: "0")...UInt8(ascii: "9"):
                    digit = UInt16(byte - UInt8(ascii: "0"))
                case UInt8(ascii: "a")...UInt8(ascii: "f"):
                    digit = UInt16(byte - UInt8(ascii: "a")) + 10
                case UInt8(ascii: "A")...UInt8(ascii: "F"):
                    digit = UInt16(byte - UInt8(ascii: "A")) + 10
                default:
                    throw GemmaTokenizer.Failure.malformedJSON("bad hex digit", atByte: index)
                }
                value = value << 4 | digit
                index += 1
            }
            return value
        }

        mutating func readInt() throws -> Int32 {
            try skipWhitespace()
            var negative = false
            if peek() == UInt8(ascii: "-") { negative = true; index += 1 }
            var value: Int32 = 0
            var digits = 0
            while index < bytes.count, bytes[index] >= UInt8(ascii: "0"),
                  bytes[index] <= UInt8(ascii: "9") {
                value = value * 10 + Int32(bytes[index] - UInt8(ascii: "0"))
                index += 1
                digits += 1
            }
            guard digits > 0 else {
                throw GemmaTokenizer.Failure.malformedJSON("expected an integer", atByte: index)
            }
            return negative ? -value : value
        }

        /// Consume one value of any type, returning the byte range it occupied.
        mutating func skipValue() throws -> Range<Int> {
            try skipWhitespace()
            let start = index
            guard let byte = peek() else {
                throw GemmaTokenizer.Failure.malformedJSON("expected a value", atByte: index)
            }
            switch byte {
            case UInt8(ascii: "\""):
                _ = try readString()
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                var depth = 0
                while index < bytes.count {
                    let byte = bytes[index]
                    if byte == UInt8(ascii: "\"") {
                        _ = try readString()
                        continue
                    }
                    index += 1
                    if byte == UInt8(ascii: "{") || byte == UInt8(ascii: "[") { depth += 1 }
                    if byte == UInt8(ascii: "}") || byte == UInt8(ascii: "]") {
                        depth -= 1
                        if depth == 0 { break }
                    }
                }
            default:
                while index < bytes.count {
                    switch bytes[index] {
                    case UInt8(ascii: ","), UInt8(ascii: "}"), UInt8(ascii: "]"),
                         0x20, 0x09, 0x0A, 0x0D:
                        return start..<index
                    default: index += 1
                    }
                }
            }
            return start..<index
        }
    }
}

/// A binary min-heap on `(rank, position)`.
///
/// Written out because Swift has no priority queue and because the ordering is
/// load-bearing: lowest merge rank wins, and equal ranks resolve to the leftmost
/// position. Reversing either comparison produces a different, self-consistent
/// tokenisation that still looks right on most inputs.
private struct MergeQueue {
    struct Entry { let rank: Int32; let position: Int; let result: Int32 }

    private var storage: [Entry] = []

    mutating func reserve(_ n: Int) { storage.reserveCapacity(n) }

    @inline(__always)
    private func precedes(_ a: Entry, _ b: Entry) -> Bool {
        a.rank != b.rank ? a.rank < b.rank : a.position < b.position
    }

    mutating func push(rank: Int32, position: Int, result: Int32) {
        storage.append(Entry(rank: rank, position: position, result: result))
        var child = storage.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard precedes(storage[child], storage[parent]) else { break }
            storage.swapAt(child, parent)
            child = parent
        }
    }

    mutating func pop() -> Entry? {
        guard let first = storage.first else { return nil }
        storage.swapAt(0, storage.count - 1)
        storage.removeLast()
        var parent = 0
        while true {
            let left = 2 * parent + 1
            let right = left + 1
            var best = parent
            if left < storage.count, precedes(storage[left], storage[best]) { best = left }
            if right < storage.count, precedes(storage[right], storage[best]) { best = right }
            if best == parent { break }
            storage.swapAt(parent, best)
            parent = best
        }
        return first
    }
}

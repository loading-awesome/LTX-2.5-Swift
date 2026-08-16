// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation
import Testing

@testable import LTXFoundation

/// **Bit-exactness, not tolerance.** Token ids are integers: the encoder either
/// produces the expected ids position for position, or it is wrong. There is nothing to
/// calibrate and no floor to sit at.
///
/// The corpus below is the substance of it. The ids come from an independent tokenizer
/// implementation over this checkpoint's own vocabulary, and each case is chosen to fail
/// for a different structural reason — a normalizer that prepends a word mark, a
/// pre-tokenizer that splits on space runs, a String-keyed vocabulary that collapses
/// canonically-equivalent spellings. Around it sit the wrapper's own obligations:
/// truncation keeps the head, an empty prompt is one BOS, BOS is not doubled, and the
/// left-padded mask feeds `TextConditioningLayout`'s right-pad permutation.
///
/// Everything skips when the 26 GiB checkpoint is absent, because the tokenizer lives
/// inside it and the MLX-free suites must still run on a bare checkout.
@Suite("Gemma tokenizer")
struct GemmaTokenizerTests {

    static let checkpointPath =
        LTXConfiguration.resolved.checkpoints.root! + "/text_encoders/"
        + "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

    /// Loaded once for the suite: reading the 32 MB `tokenizer_json` tensor and
    /// resolving 514906 merges takes long enough to be worth not repeating.
    ///
    /// `nil` means "no checkpoint on this machine" and skips. A checkpoint that is
    /// present but unreadable surfaces as a thrown error instead, because silently
    /// skipping a broken tokenizer is how a suite goes green while the render path is
    /// dead.
    nonisolated(unsafe) static let loaded: Result<GemmaTokenizer, Error>? = {
        guard FileManager.default.fileExists(atPath: checkpointPath) else { return nil }
        do {
            return .success(try GemmaTokenizer.load(
                fromCheckpoint: URL(fileURLWithPath: checkpointPath)))
        } catch {
            return .failure(error)
        }
    }()

    static func tokenizer() throws -> GemmaTokenizer? {
        guard let loaded else { return nil }
        return try loaded.get()
    }

    /// The prompt is read from the manifest rather than written down here: a copy in
    /// this file and the manifest could drift apart, and a mismatch would give no way to
    /// tell which of the two moved.
    static func manifestPrompt() throws -> String? {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()      // LTXFoundationTests
            .deletingLastPathComponent()      // Tests
            .deletingLastPathComponent()      // repo root
        let url = root.appendingPathComponent("Tests/Fixtures/goldens/tiny_v5/manifest.json")
        guard let data = try? Data(contentsOf: url),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let params = json["params"] as? [String: Any] else { return nil }
        return params["prompt"] as? String
    }

    // MARK: - The layout contract

    /// The two halves of the layout contract meeting: the tokenizer left-pads, and
    /// `TextConditioningLayout` is what turns that into the right-padded order the
    /// connectors need. Neither is checkable from a shape.
    @Test("the mask feeds TextConditioningLayout's right-pad permutation")
    func layoutHandoff() throws {
        guard let tokenizer = try Self.tokenizer(),
              let prompt = try Self.manifestPrompt() else { return }

        let conditioning = try tokenizer.conditioning(for: prompt)
        let layout = TextConditioningLayout()
        let order = try layout.rightPadOrder(mask: conditioning.mask)

        #expect(Array(order.prefix(11)) == Array(1013..<1024))
        #expect(try layout.reorderedMask(mask: conditioning.mask).prefix(11).allSatisfy { $0 })
        #expect(layout.validTokenCount(mask: conditioning.mask) == 11)
    }

    // MARK: - The algorithm

    /// Each case is a different way to be wrong. The ids come from the HuggingFace
    /// backend over this checkpoint's own `tokenizer_json`.
    static let corpus: [(String, [Int32])] = [
        // The fixture's prompt, unpadded. Note the first token is "A", not "▁A": this
        // normalizer has no Prepend.
        ("A red cube rotating slowly on a white table.",
         [236776, 2604, 26365, 26644, 13198, 580, 496, 2173, 2633, 236761]),
        // Leading and trailing whitespace is stripped first, so the surrounding spaces
        // are gone and the first word again carries no word mark.
        ("  leading and trailing   ",
         [26016, 532, 45330]),
        // Runs of spaces become runs of ▁ and merge into multi-space tokens — only
        // possible because the pre-tokenizer never splits.
        ("double  space and triple   space",
         [7902, 138, 5780, 532, 22178, 139, 5780]),
        // Precomposed vs decomposed: one symbol or two, and different ids. Swift
        // considers these two Strings equal.
        ("combining: \u{E9}",
         [20024, 4049, 236787, 1559]),
        ("combining: e\u{301}",
         [20024, 4049, 236787, 545, 238288]),
        // The same distinction where the *vocabulary* holds both spellings with
        // different ids: 3244 vs 127866. A String-keyed vocabulary collapses these.
        ("\u{103}n l\u{E0} vui",
         [14094, 3244, 62344]),
        ("\u{103}n la\u{300} vui",
         [14094, 127866, 62344]),
        // A vocabulary entry that begins with U+FEFF. Foundation's UTF-8 String
        // decoding strips a leading BOM, which loses this entry entirely.
        ("\u{FEFF}// bom prefixed",
         [135260, 14597, 149989]),
        // Added tokens are matched against the raw text, and the spaces left behind
        // survive as bare ▁ (236743).
        ("<bos> literal <eos> and <pad>",
         [2, 37087, 236743, 1, 532, 236743, 0]),
        ("CJK: \u{4E2D}\u{6587}\u{6E2C}\u{8A66}",
         [44248, 236787, 17346, 237364, 154812]),
        // Byte fallback: NUL is not a vocabulary entry, so it resolves to <0x00>.
        ("control \u{0} byte",
         [8607, 236743, 238, 16812]),
        // Seven scalars, one grapheme cluster. Iterating Characters gets this wrong.
        ("flag \u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F}",
         [16213, 236743, 247292, 247968, 249179, 251878, 254231, 253972, 249258]),
        // U+00A0 counts as whitespace both to the stripper here and to Swift's
        // .whitespaces…
        ("\u{A0}nbsp padded\u{A0}",
         [11539, 71009]),
        // …but U+001C and U+001F count only to the stripper.
        ("\u{1C}unit separators\u{1F}",
         [6805, 69717]),
        // A literal ▁ in the prompt is indistinguishable from a normalised space.
        ("\u{2581}already a lower one eighth block",
         [3016, 496, 3718, 886, 35186, 3355]),
        ("1234567890 3.14159 -42",
         [236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819,
          236771, 236743, 236800, 236761, 236770, 236812, 236770, 236810, 236819,
          753, 236812, 236778]),
        ("URL https://example.com/path?query=1&other=2#frag",
         [6738, 4560, 1411, 8358, 236761, 854, 236786, 2337, 236881, 3278, 236784,
          236770, 236880, 1538, 236784, 236778, 236865, 45265]),
        ("trailing space then added token <bos>",
         [136697, 2557, 1299, 3742, 8369, 236743, 2]),
    ]

    @Test("the merge loop agrees with the HuggingFace backend across the corpus")
    func corpusAgreement() throws {
        guard let tokenizer = try Self.tokenizer() else { return }

        for (text, expected) in Self.corpus {
            let got = tokenizer.encode(GemmaTokenizer.strip(text))
            #expect(got == expected,
                    Comment(rawValue: "\(text.debugDescription): produced \(got), "
                        + "expected \(expected)"))
        }
    }

    /// The two Vietnamese cases above are the same `String` to Swift. Stating that
    /// here, rather than leaving it implicit in the corpus, so that a future
    /// "deduplicate the test cases" pass cannot quietly delete one of them.
    @Test("canonically-equivalent prompts are equal Strings and different token ids")
    func canonicalEquivalence() throws {
        guard let tokenizer = try Self.tokenizer() else { return }

        let precomposed = "\u{103}n l\u{E0} vui"
        let decomposed = "\u{103}n la\u{300} vui"
        #expect(precomposed == decomposed)                      // Swift says equal
        #expect(Array(precomposed.utf8) != Array(decomposed.utf8))   // the bytes are not
        #expect(tokenizer.encode(precomposed) != tokenizer.encode(decomposed))
        #expect(tokenizer.vocabularySize == 262_144)
    }

    // MARK: - The wrapper

    @Test("an over-length prompt loses its tail, not its head")
    func truncation() throws {
        guard let tokenizer = try Self.tokenizer() else { return }

        let text = String(repeating: "The quick brown fox jumps over the lazy dog. ",
                          count: 200)
        let full = tokenizer.encode(GemmaTokenizer.strip(text))
        #expect(full.count == 2000)

        let conditioning = try tokenizer.conditioning(for: text)
        #expect(conditioning.ids.count == 1024)
        #expect(conditioning.validTokenCount == 1024)          // no padding at all
        #expect(conditioning.ids[0] == tokenizer.bosID)
        // BOS is prepended *after* truncation and the result truncated again, so the
        // last kept token is full[1022] and full[1023] is dropped.
        #expect(Array(conditioning.ids[1...]) == Array(full[0..<1023]))
        #expect(conditioning.ids[1023] == full[1022])
    }

    @Test("an empty prompt is one BOS and 1023 pads")
    func emptyPrompt() throws {
        guard let tokenizer = try Self.tokenizer() else { return }

        let conditioning = try tokenizer.conditioning(for: "   \n  ")
        #expect(conditioning.ids.count == 1024)
        #expect(conditioning.validTokenCount == 1)
        #expect(conditioning.ids[1023] == tokenizer.bosID)
        #expect(conditioning.ids[..<1023].allSatisfy { $0 == tokenizer.padID })
    }

    @Test("BOS is prepended once, not twice, when the prompt already starts with it")
    func bosNotDoubled() throws {
        guard let tokenizer = try Self.tokenizer() else { return }

        let conditioning = try tokenizer.conditioning(for: "<bos>hello")
        let valid = Array(conditioning.ids.suffix(conditioning.validTokenCount))
        #expect(valid.first == tokenizer.bosID)
        #expect(valid.dropFirst().first != tokenizer.bosID)
    }

    @Test("the vocabulary is the size the checkpoint's own config declares")
    func vocabularyMatchesConfig() throws {
        guard let tokenizer = try Self.tokenizer() else { return }
        let header = try SafetensorsHeader.read(from: URL(fileURLWithPath: Self.checkpointPath))
        guard let config = header.metadataJSON("gemma_config") else { return }
        let text = (config["text_config"] as? [String: Any]) ?? config
        guard let declared = text["vocab_size"] as? Int else { return }

        // Not a formality: bridging this vocabulary through JSONSerialization's
        // String-keyed dictionary yields 261707, and every one of the 437 lost entries
        // would still tokenise *something*.
        #expect(tokenizer.vocabularySize == declared)
    }

    @Test("a sequence length of zero is refused rather than silently producing nothing")
    func rejectsEmptyLength() throws {
        guard let tokenizer = try Self.tokenizer() else { return }
        #expect(throws: GemmaTokenizer.Failure.self) {
            _ = try tokenizer.conditioning(for: "anything", sequenceLength: 0)
        }
    }
}

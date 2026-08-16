// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sean Kammerich

import Foundation

/// Prompt defaults substituted when a caller supplies nothing.
///
/// These are **not** tokenizer properties — they belong to the pipeline — but they have
/// to live somewhere a renderer and its callers can both reach, and they are inert data
/// with no dependencies.
public enum PromptDefaults {

    /// Substituted whenever the negative prompt is empty.
    ///
    /// **This is the trap, and it is a quiet one.** An empty negative prompt does not
    /// mean an empty negative branch: this string is substituted before tokenising, so
    /// the negative row holds **223 real tokens** where a port that encoded the caller's
    /// empty string literally produces a single BOS and 1023 pad ids.
    ///
    /// Nothing about that failure looks wrong: the tensor is `[2, 1024]` either way, the
    /// mask is well-formed, the padding is on the correct side, and the positive row —
    /// the one anybody checks first — is bit-exact. It surfaces only as a numerical
    /// divergence somewhere downstream of classifier-free guidance, by which point the
    /// cause is nowhere near the symptom.
    ///
    /// **Copy this string, never retype it.** A hand-transcribed version of it matched
    /// 805 of 1024 ids — close enough to look like a tokenizer bug and nowhere near
    /// right. It is one literal; there is no reason for a second one to exist anywhere
    /// in this repo.
    public static let negativePrompt = """
        has_subtitles, has_blurbox, transition from black, transition to black, \
        speech_ending_short, blurry, out of focus, overexposed, underexposed, low \
        contrast, washed out colors, excessive noise, grainy texture, poor lighting, \
        flickering, motion blur, distorted proportions, unnatural skin tones, deformed \
        facial features, asymmetrical face, missing facial features, extra limbs, \
        disfigured hands, wrong hand count, artifacts around text, inconsistent \
        perspective, camera shake, incorrect depth of field, background too sharp, \
        background clutter, distracting reflections, harsh shadows, inconsistent \
        lighting direction, color banding, cartoonish rendering, 3D CGI look, \
        unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong \
        gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, \
        silent or muted audio, distorted voice, robotic voice, echo, background noise, \
        off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery \
        movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent \
        framing, tilted camera, flat lighting, inconsistent tone, cinematic \
        oversaturation, stylized filters, or AI artifacts.
        """
}

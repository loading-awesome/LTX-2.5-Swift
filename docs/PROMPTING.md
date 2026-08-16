# Prompting LTX 2.5

Craft guidance for writing prompts, plus the handful of things this port has
actually measured about how prompts land.

**Provenance, because the two halves have very different standing.** The craft
guidance is condensed from Lightricks' own
[prompting guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide)
— it is the model authors' advice about their model, and this port has not
verified any of it. The sections marked **measured here** are ours, with the
evidence named. Do not let the two blur, and do not expect a number to settle
which prompt is better: seed-to-seed SSIM between two *correct* renders measured
**−0.0418**, so there is no prompt-quality metric here and judgement is by eye.

---

## The shape of a prompt

A prompt is a single flowing paragraph describing a complete visual and audio
narrative from start to finish, in **present tense**. Six things earn their place:

1. **Shot establishment** — the cinematography term that matches the genre.
2. **Scene** — lighting, colour, texture, atmosphere.
3. **Action** — chronological, beginning to end.
4. **Character** — age, appearance, clothing, and emotion expressed through
   *physical cues* rather than named feelings.
5. **Camera** — how and when it moves, and where the subject sits **after** the
   move.
6. **Audio** — ambient sound, music, dialogue in quotation marks, with accent or
   language when it matters.

Single shots work best at roughly **4–8 descriptive sentences**. Match structure
to complexity rather than padding to a target; a longer prompt earns its length
only if every sentence adds a concrete visual or audio detail.

**Include:** present-tense verbs, physical emotion cues, quoted dialogue,
consistent lighting logic, few clear characters.

**Avoid:** abstract emotional labels, conflicting geography between shots, mixed
light sources, on-screen text (rendered unreliably), chaotic physics.

### Camera and motion vocabulary

`follows`, `tracks`, `pans`, `circles`, `tilts`, `pushes in`, `pulls back`,
`overhead view`, `handheld`, `over-the-shoulder`, `establishing shot`, `static
frame`, `lens flare`, `motion blur`, `depth of field`, `slow motion`,
`time-lapse`.

---

## Multi-shot prompts

Multiple distinct shots go in **one chronological paragraph** with the cuts
described in prose. Not a numbered shot list, and not screenplay sluglines,
unless the cut itself is written out.

At every cut, four things must be restated:

- **Name the transition** — "hard cut", "dissolve", "match cut".
- **Re-establish the shot** — framing, scale, subjects, lighting. The new shot
  does not inherit the old one's setup.
- **Hold visual identity** — recurring characters and objects need their
  description carried across the cut, or they will drift.
- **State audio continuity** — whether music or dialogue continues, changes, or
  stops.

Prefer **2–4 shots** per generation, each with a clear narrative function.

**Use a single continuous take instead** for unbroken motion, intimate
performance, or lip-synced dialogue that needs one framing. A cut is not free:
it spends tokens on re-establishment that a single take spends on the subject.

---

## Audio and dialogue

Dialogue goes in quotation marks. Specify language, accent and delivery
(whisper, shout, robotic). Say at each moment whether ambient sound, music or
dialogue continues or changes.

> **What this port does with that today.** The audio stream is generated beside
> video through the DiT's cross-modal attention, decoded to a mel, and then
> synthesised to a 48 kHz stereo waveform by the vocoder. `ltx render` writes it
> as an AAC track in the mp4 and as a 32-bit float WAV beside it. A two-speaker
> dialogue clip came out clean and **lip-synced**.
>
> **Nothing in the suite consumes the waveform** — vocoder output cannot be
> compared numerically at all, for the reason contract 11 in
> [`FRAGILE_CONTRACTS.md`](FRAGILE_CONTRACTS.md) gives — so audio you can hear
> is evidence the port is plausible, never evidence it is correct.

---

## Frame counts, and why your duration moved

Frame counts live on the VAE's **`8k + 1` causal lattice** (contract 4). 97
frames is on it; 100 is not, and a request for 100 is refused rather than
quietly adjusted to fit. Adjusting it is not the harmless courtesy it looks
like: flooring the video to 97 does nothing to the audio stream, which keeps
counting from the number you asked for, so the two modalities end up disagreeing
about the length of the clip.

`--fps` moves the same geometry and is not container metadata: at 97 frames,
24 fps and 25 fps give different audio lengths and both look plausible. The
lattice and the arithmetic are in [`RECIPES.md`](RECIPES.md).

---

## Reference sheets — characters, wardrobe, props

The **Ingredients** IC-LoRA conditions a render on a *reference sheet*: one composite
image inventorying the scene's characters, outfits, props and location. You then
write a prompt describing the action, and the adapter's job is to keep those
elements visually consistent across the render.

Two prompting practices come with it, and the first is not obvious:

- **Repeat the colour token, glued to each clothing noun.** Write "BLACK fedora,
  white shirt, BLACK suit jacket, BLACK trousers" rather than "a black suit with
  a white shirt". A colour stated once, at the head of a list, drifts across the
  garments it was meant to govern.
- **Keep the sheet and the action separate.** The reference sheet describes what
  things *are*; the prompt describes what *happens*. Folding an inventory into the
  action sentence costs tokens in the wrong place, and multi-shot prompts already
  spend heavily on re-establishment at each cut.

Run it with `ltx ingredients --reference SHEET --prompt "..."`. Add `--upsample`
for the reference pipeline's second stage, which re-encodes the sheet at the
doubled size and refines through the *same* conditioning. Enlarging a finished
clip afterwards with `ltx upscale --mode refined` is a different thing entirely:
that route attaches no adapter, and it melts terrain.

> **Two things to keep in mind.** Every LTX IC-LoRA measured so far — including
> Lightricks' own 2.5 upscaler — carries **zero audio keys**, so an IC-LoRA
> adapts the video half of an audio-video model and leaves the cross-modal seam
> untouched. And **only the upscaler has a 2.5 release from Lightricks**:
> Ingredients, In-Outpainting and Union Control are 2.3-only, shape-compatible
> with the 2.5 transformer but trained against different base weights and a
> different VAE latent space. Treat any 2.3 adapter on 2.5 as an experiment to
> be looked at, not a supported path.

### A cast, not a sheet — `ltx msr`

Ingredients gives the model one image and lets it decide what on that image is
what. **Multiple-Subject-Reference** takes the subjects *separately* — up to
five stills, each in its own slot — and keeps them addressable:

```bash
ltx msr -r hero.png sidekick.png prop.png --background dunes.png \
        -p "The hero rides past the sidekick, the prop lashed to the bike." \
        --width 640 --height 384 --frames 121
```

What keeps the slots apart is not the prompt but the geometry the adapter was
trained with — [`CLI.md`](CLI.md) states it, contracts 18 and 19 in
[`FRAGILE_CONTRACTS.md`](FRAGILE_CONTRACTS.md) pin it down, and when it breaks
the render still completes and the cast blends. Three practices follow:

- **Name the slots in the prompt in the order you passed them.** Slot order is
  the only thing that ties a still to a description; the adapter learned
  `pic1 … picN` positions, not your nouns.
- **Use `--background` for a location or plate.** It takes the last slot and is
  centre-cropped to fill the frame, where a subject is padded with white when
  its aspect ratio disagrees with the output's — cropping a portrait into a
  16:9 frame turns a character reference into a reference to that character's
  chin.
- **Budget for the sequence.** Each still is repeated to 33 frames (or 25 with
  `--reference-frames 25`) and encoded at the *output's* resolution, so five
  references cost five clips' worth of tokens. On a short target they outweigh
  the picture.

## Negative prompts

`--negative-prompt` defaults to empty, and **empty does not mean none**. Contract
9 forces `min_length = 1024`, and the pipeline's default negative resolves to a
real sequence: measured on a live render, the two branches tokenise to
**11 and 223 valid tokens** for a short prompt and an empty negative. If you
supply your own negative prompt you are replacing that default, not adding to it.

---

## Measured here: the prompt is not the only thing deciding your shot

**Measured on** 30-step renders at 640×384×97, identical prompt and guidance,
differing only in the drawn initial latent.

Prompt: *"A red cube rotating slowly on a white table."*

| seed | what it rendered | `motion_fraction` |
|---|---|---|
| 0 | a **hand** entering frame and rotating the cube | 0.313 |
| 1 | the cube alone on a **turntable** | 0.087 |

Both are correct readings. The prompt says the cube rotates and never says *what
rotates it*, so the model supplied an agent at one seed and not at another — and
the resulting clips are not even in the same motion regime, differing by 3.6× in
the fraction of the frame that moves.

Two things follow, and the second is the one that costs people time:

1. **Name the agent of every motion**, or the seed will choose one. "A red cube
   rotating slowly on a white turntable, no people in frame" is a different
   instruction from the prompt above.
2. **Re-rolling the seed is not a small change.** This model's trajectory
   separates at production shape at *every* step count, so two seeds are two
   different renders, not two variations on one. If you liked a
   render, keep its seed — `ltx render` records it in the `.provenance.json`
   beside every mp4 for exactly this reason.

# The `ltx` command line

Seven subcommands. Two of them — `doctor` and `recipes` — answer "what can this
machine do" and "how do I ask for it" without starting a render, and are the ones
to run first.

```
ltx doctor        what this Mac can run, which checkpoints it found
ltx recipes       the pipelines, their shapes, and which command runs each
ltx render        generate a clip from a prompt
ltx ingredients   generate from a prompt plus a reference sheet
ltx msr           generate from a prompt plus up to five reference stills
ltx upscale       enlarge a clip that already exists
ltx bench         measure this machine
```

Every subcommand's `--help` carries the full per-flag prose and is authoritative.
This document is the map: what the commands are for, how they fit together, and
the settings whose defaults are not what you would guess.

## Finding the checkpoints

Model weights are not in this repository, and **the config file is how the CLI finds
them**. Three layers, each overriding the one before:

1. **built-in defaults** — a placeholder root at `~/models/ltx`, there so that a
   first run has something to write down rather than something to be right about;
2. **the config file**, `~/.config/ltx/config.json` by default and `--config` to
   point elsewhere. This is the real mechanism;
3. **a per-command override flag** (`--checkpoint`, `--text-encoder`, `--video-vae`,
   `--audio-vae`, `--upsampler`), for a one-off.

`ltx doctor` writes the file on a first run and reports what it resolved. Setting
`checkpoints.root` in it is normally the whole configuration, because the layout
below the root is the model's own. `ltx doctor --write-config` rewrites the file
from the defaults plus any overrides given on the same line — that is the explicit
opt-in, and without it an existing config is read and left alone, since it may carry
edits the command knows nothing about.

The library reads the same file: `LTXConfiguration.load(from:)` returns the config
and where it came from, and reports an unparseable one rather than falling back.

### Every command checks before it loads

Anything that opens a checkpoint runs `doctor`'s checks first, silently: the Metal
kernels are present, each file this run needs exists and is the file it claims to
be, and the checkpoints agree on a model version. It is header reads only, so it
finishes before the first gigabyte is mapped.

If something is wrong the command refuses to start and prints **every** problem it
found with a fix for each, rather than the first one and a stack of re-runs. Only
the roles a run actually opens are checked — `ltx upscale --mode plain` runs no
transformer and will not tell you yours is missing.

```
Error: cannot start — 2 problems:

  the video vae checkpoint is not usable
      ltx-2.5-video-vae-bf16.safetensors is not a videoVAECausal checkpoint
      (found videoVAENADiffusion)
      fix: set the video vae entry in ~/.config/ltx/config.json

  the audio vae checkpoint is not usable
      the audio vae checkpoint is not at …/vae/NOPE-audio.safetensors
      fix: set the audio vae entry in ~/.config/ltx/config.json
```

The two failures this exists to catch are the ones that explain themselves worst. A
missing metallib does not surface as "missing metallib": every path resolves, the
render starts, and the first GPU call dies inside MLX with an untyped C++ error and
no path — at device resolution, so it happens even on `Device(.cpu)`. And a missing
checkpoint would otherwise arrive after the 26 GB text-encoder pass that preceded
it.

**Overrides are announced on stderr, deliberately.** The dev and distilled
transformers are byte-identical in structure — same tensor count, same metadata,
same embedded `model_version` — so which one is loaded is knowable from the
filename and nothing else. Pointing `--checkpoint` at the wrong one produces a
render, not an error.

## Shape, duration and the two lattices

Every generating command shares one geometry model.

| | rule |
|---|---|
| width, height | multiples of **64** |
| frames | on the **8k+1** lattice — 1, 9, 17 … 97, 105 … |
| size | `--megapixels` with `--aspect-ratio`, or the recipe's own measured shape. `render` also takes `--width`/`--height`; `ingredients` and `msr` do not — their stage shapes come from the recipe |
| aspect ratios | `16:9`, `9:16`, `1:1`, `21:9`, `4:3`, `3:4` |

The frame lattice is the causal VAE's: frame 0 encodes alone and every 8 frames
after it become one latent. That is also why `--keyframe` refuses a frame inside a
chunk rather than rounding it — rounding would freeze all 8 of that chunk's frames
onto one still.

**`--fps` is not just container metadata.** The audio stream's length is
`round(frames / fps * 25)` latents. At 97 frames, 24 fps gives 101 audio latents
and 25 fps gives 97. Both render; they are different geometries. 24 is the default
and the shape everything here is measured at.

## `ltx doctor`

Reads safetensors **headers only** — no checkpoint is loaded, the GPU is not
touched, nothing is rendered. Safe and fast to run at any time.

```bash
ltx doctor
ltx doctor --tokens 7488          # plan memory for a larger shape
ltx doctor --recipe distilled     # check a different recipe's expectations
ltx doctor --write-config         # rewrite ~/.config/ltx/config.json
```

It prints, in order:

- **configuration** — which config file was used, the default recipe, the memory
  margin, the checkpoint root.
- **machine** — chip, unified memory, core count, and how much is actually free
  right now.
- **Metal kernels** — the path to the `mlx.metallib` the build produced. If this
  is missing, see the build note in the README; nothing will run without it.
- **MLX large-M GEMM patch** — applied or not. It lives in `.build/checkouts`,
  which is gitignored, so a `swift package reset` or a fresh clone silently drops
  it. Nothing fails when it is missing; renders are just slower on long clips.
- **checkpoints** — each resolved file with its size and tensor count.
- **adapters** — every LoRA and IC-LoRA found under the adapter roots, with its
  size, its declared model version, its kind, and a flag on anything that is not a
  2.5 checkpoint.
- **recipes** — the same table `ltx recipes` prints, plus the adapter each recipe
  resolves to and what it will cost resident.
- **memory plan** — per-phase peaks (text encode, sampling, decode) against what
  the machine has, with the margin applied. Adapter weights are counted separately
  and subtracted before planning.

It ends with `no problems found`, or with the problems.

## `ltx recipes`

The menu. Every registered pipeline, resolved at one duration and size target.

```bash
ltx recipes
ltx recipes --megapixels 0.6
ltx recipes --aspect-ratio 9:16 --seconds 8
```

With no `--megapixels`, each recipe reports its own gated shape — the shape `ltx
render` would pick if you named no size either.

The columns are id, **run with** (the subcommand that executes it), shape, stages,
tokens and status. See [RECIPES.md](RECIPES.md) for what a recipe is and what the
status words mean.

## `ltx render`

The single-stage and two-stage generation path.

```bash
ltx render --prompt "a slow pan across a quiet room, dust in the light" \
           --out room.mp4

ltx render --recipe distilled --prompt "…" --out fast.mp4

ltx render --prompt "…" --image open.png --last-frame close.png \
           --frames 97 --megapixels 0.2 --out shot.mp4
```

An `.mp4` is written, and a `.provenance.json` beside it recording every
parameter, the checkpoint identities, and how the latent was drawn.

### Choosing the pipeline

`--recipe` decides the stages, **the step count**, the sampler, the guidance and
the transformer. The guidance flags are overrides on top of it, and a recipe that
builds no guider refuses them by name rather than ignoring them. `--recipe
distilled` is the spelling to prefer over the older `--two-stage` flag, which means
the same thing.

**There is no `--steps`.** A step count is a property of the schedule a pipeline
was distilled or measured for, not a dial:

| job | steps | why that number |
|---|---|---|
| `prod` | **30** | the measured single-stage schedule |
| `distilled`, `ingredients`, `msr` | **8** | the recorded draft table, returned bit-identically at 8 |
| the refine and every `upscale` mode | **3** | the recorded stage-2 table, which is literally four sigmas long |

Below 8 the draft schedule is *thinned*, and it thins from the head — the first
five of its sigmas span 1.0 to 0.975 while the last four do the work. So a lower
count is not the same trajectory rendered faster; it is a different one with
`0.725` missing from the middle of the working range, which shows up as background
structure that never resolves.

### Guidance, and where neutral is

Each of these costs a transformer forward per step unless it is neutral, so the
pass count — and the render time — follows directly from them.

| flag | default | neutral at | note |
|---|---|---|---|
| `--video-cfg` | 3.0 | **1.0** | what makes `--negative-prompt` bite on the picture |
| `--audio-cfg` | 7.0 | **1.0** | independent of the video one |
| `--stg-scale` | 1.0 | **0.0**, not 1.0 | an empty `--stg-blocks` does *not* disable it — it runs a full-price pass that perturbs nothing |
| `--modality-scale` | 3.0 | **1.0** | one extra pass, shared across both directions |
| `--rescale-scale` | 0.7 | **0.0** | applied to the combination; costs no pass |

`--stg-start`/`--stg-end` and `--modality-start`/`--modality-end` restrict a term
to part of the schedule, as fractions from 0 (the first sampled sigma) to 1 (sigma
0). A window that excludes every sampled sigma is bit-identical to switching the
term off, and costs nothing.

All of these are refused under the distilled recipe, which builds no guider at all.

### Conditioning

| flag | conditions on |
|---|---|
| `--image` | a still at frame 0 — image-to-video |
| `--last-frame` | a still at the final frame; shorthand for `--keyframe PATH@last` |
| `--keyframe PATH@FRAME[:STRENGTH]` | a still pinned to a frame, repeatable |
| `--video` | an existing clip, matching frame count and dimensions |
| `--audio` | a waveform, through the measured mel front end |
| `--audio-mel` | a mel spectrogram directly |

`--strength` (0…1, default 1.0) sets how hard the conditioned tokens are held.
1.0 pins them exactly; the image is re-imposed at every step, so a high strength
holds the shot there rather than merely starting from it. A `--keyframe` may carry
its own strength after a colon, which overrides `--strength` for that keyframe
alone — an anchored opening and a loose closing frame are one command.

`FRAME` accepts an integer, `first` or `last`, and must land on a latent boundary.

Audio conditioning freezes the audio stream outright; `--strength` does not apply
to it. A waveform that is not already at the audio VAE's rate is resampled, and
that the resampler ran at all is recorded in the provenance sidecar.

Under the distilled recipe, `--image` and `--strength` work (the image is encoded
once per stage, at half size for the draft and full size for the refine); `--video`
and the audio flags are refused by name.

### Adapters

`--lora NAME[@STRENGTH]`, repeatable. `NAME` is either a path or a fragment of a
filename resolved against the model tree the transformer sits in. A fragment
matching nothing, or matching more than one adapter, is refused by name. Only
files whose own keys carry LoRA tensors are candidates, so pointing this at a
latent upscaler is caught before the 42 GB transformer is read.

Adapters are applied as a residual overlay, never baked, so stacking two is well
defined and the resident cost is the adapter file. `--lora` **stacks on top of**
whatever the recipe already asks for.

### Cross-step cache

`--cache-threshold` (default 0, dense) reuses the previous step's stack residual
when block 0 has barely moved, with `--cache-max-skips` (default 5) capping
consecutive reuse. It is off by default and is a speed/quality trade to sweep
yourself, not a free win. Refused under the distilled recipe.

## `ltx ingredients`

Conditions on a **reference sheet** — one composite image holding the characters,
objects and locations a shot should contain — through the Ingredients IC-LoRA.

```bash
ltx ingredients --reference sheet.png --prompt "…" --out shot.mp4
ltx ingredients -r sheet.png -p "…" --recipe ingredients-prod-cfg --megapixels 0.25
```

The sheet is encoded at each **stage's** resolution and its latent tokens ride
alongside the generated ones, held clean, then dropped before the decode.

**There is no `--width` or `--height`.** The output comes from `--megapixels` and
`--aspect-ratio` through the shape ladder, or from the recipe's own measured shape
when you name neither. What each **stage** samples at comes from that stage's
`scale`, so it is a property of the recipe rather than of your command line.

That is not tidiness. Draft resolution is the knob that decides whether rigid
structure survives: a subject occupying a 10×6 latent grid has nowhere to put
mechanical detail, and a refine that re-noises to 0.909 then invents it
independently per frame — which is what melting is. These flags briefly meant the
*output* size with stage 1 sampling at half, and the release that changed them
halved every existing caller's draft without their command line changing.

Three recipes run here. `ltx recipes` prints them with the shape each resolves to
and what it has been measured at.

| recipe | transformer | schedule | guidance | stages | forwards |
|---|---|---|---|---|---|
| `ingredients` | distilled | 8-step table + 3-step refine | none | 2 | **11** |
| `ingredients-prod` | dev | continuous, 30 | production | 1 | **120** |
| `ingredients-prod-cfg` | dev | continuous, 30 | CFG only | 1 | **60** |

**Two stages is not a speed setting.** It makes a *larger* output cheaper than
rendering it natively — measured at 1.95× — and it does not make a given output
faster, because the draft it economises on is the draft the picture is built from.
Comparing two-stage at 640×384 against single-stage at 640×384 compares a 320×192
draft against a 640×384 one. The prod recipes are single-stage at full resolution
for that reason and buy their time back on passes instead: `-cfg` drops STG and the
modality term, which are two of the four passes, for half the wall clock at the same
resolution. CFG is the structural term; the other two refine what it has placed.

`ingredients` still drafts and refines, because its distilled schedule is cheap
enough that the refine is most of its cost anyway. Its stage 2 starts at the
recorded 0.909375 re-noise level, so it continues stage 1 rather than re-rendering
it — the curve is scaled to that head rather than truncated, which keeps its shape.

Two machine knobs, neither of which changes what is rendered:
`--eval-cadence n` forces the latent every n-th step instead of every step, trading
a larger live working set for fewer GPU barriers; `--cache-threshold` reuses the
previous step's residual when block 0's has barely moved. Both default to the dense
behaviour every measurement in this repo was taken with. See `docs/PERFORMANCE.md`
for what each was measured at.

The shipped adapter is a 2.3 checkpoint. All 480 of its modules resolve against the
2.5 transformer with matching shapes, which is a statement about key sets and not
about whether its features transfer. Treat this route as an experiment.

## `ltx msr`

Multiple-Subject-Reference: up to five separate stills, each in its own slot, kept
distinguishable — which single-reference IC-LoRAs cannot do.

```bash
ltx msr -r alice.png -r bob.png -r prop.png --prompt "…" --out scene.mp4
ltx msr -r alice.png -r bob.png --background street.png --prompt "…"
```

Two mechanisms keep the slots apart: a learned slot vector added to each
reference's latent channels, and a distinct **negative** time offset per slot —
`-(n - i)` pixel frames — so no reference shares coordinates with another or with
the target's own frame 0.

References are given in slot order, 1–5 (or 1–4 with `--background`).
`--background` takes the last slot and is **centre-cropped** to the frame, where a
subject is **padded** rather than cropped when its aspect ratio disagrees with the
output's.

`--reference-frames` (25 or 33, default 33) is how many frames each still is
repeated to before encoding. This is the cost dial: five references at 33 frames
occupy five times what a 33-frame clip would. `ltx recipes` prints the in-context
token cost beside the generated one — budget memory from the total.

`--width` and `--height` are the **output** here too, and stage 1 samples at half
both. One difference from `ltx ingredients`: stage 2 re-prepares, re-encodes and
re-tags the **whole cast** at the output resolution, because
`reference_downscale_factor` is 1 and a reference must sit on its own stage's grid
— so a five-slot render pays five encodes twice. Seconds, against the minutes of
steps it saves. `--single-stage` opts out.

## `ltx upscale`

Enlarges a clip that already exists. Both axes of the input must be multiples of
32. The source's audio is carried over unchanged and trimmed to the new picture
duration; `--silent` drops it. Audio is never regenerated — every mode here changes
the picture and none of them has anything to say about the sound.

Three modes, in increasing cost:

| `--mode` | what runs | checkpoints |
|---|---|---|
| `plain` (default) | encode, x2 latent upsample, decode | 3.4 GB — no transformer, no text encoder |
| `refined` | plain, then 3 deterministic Euler steps | + the 42 GB DiT and a 26 GB text-encoder pass |
| `ic-lora` | the x2 pixel spatial upscaler IC-LoRA | + the DiT, the text encoder and the adapter |

**`refined` re-noises to 91% before re-denoising** (`--denoise`, default
0.909375). That value is the recipe's own, chosen for cleaning up an 8-step draft;
on a finished clip it re-renders rather than upscales. Lower it until the picture
stops changing identity.

`--prompt` is not decoration under `refined` and `ic-lora`: whatever the refine
re-derives, it re-derives toward that prompt, and an empty one points at the base
model's own prior, which for this checkpoint is photographic. Pass the film's style
string, or `--prompt-file` to keep it byte-identical across runs.

Every mode runs the recorded 3-step refine table. That is the schedule rather than
a sample of it, and it is the difference between cleaning up a draft and
re-rendering a finished clip — a decision `--mode` has already made.

`--ic-lora` is not `--upsampler`. The upsampler is the latent spatial upscaler that
sits between the two stages of a distilled render; the IC-LoRA is a rank-32 adapter
on the transformer and is the checkpoint actually built to upscale a clip. Both are
about 327 MB and both describe themselves as a spatial upscaler.

## `ltx bench`

Two subcommands, and **GPU work must be serialised for either to mean anything.** A
cell measured beside another process is not a slower cell, it is a wrong one.

```bash
ltx bench gemm    --out gemm.json    [--csv gemm.csv]
ltx bench forward --out stages.json  [--csv stages.csv]
```

`bench gemm` takes its shapes from a ranged header read of the transformer — no
payload, no 42 GB load — and measures them on synthetic operands. It measures the
machine, not a render. `--include-square` adds a square GEMM explicitly labelled a
best case and excluded from the ceiling summary, because a square is not the
model's shape.

`bench forward` runs a real render and reports per-phase wall clock and peak
memory. `--repeats` repeats the **sampling** phase only; the model load and the
decodes run once. At `--repeats 1` the spread statistic is null, not zero.
`--attention explicit` is a bench-only option and is a memory *class* change, not a
constant factor: it materialises a `[1, 32, T, T]` fp32 score matrix, so its peak
is quadratic in token count and it will be killed at long or large shapes.

Both write a machine-readable record carrying runtime identity (chip, memory, MLX
version, repo revision and whether the worktree was dirty), every sampling
parameter, per-phase and per-step wall clock, peak memory, and an open `quality`
map. **A record whose render was not looked at is a speed observation, not evidence
about a build.**

See [PERFORMANCE.md](PERFORMANCE.md) for what has been measured and the rules the
record format enforces.

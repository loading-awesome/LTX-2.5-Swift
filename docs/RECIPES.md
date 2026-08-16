# Recipes — the menu, and what its shapes and statuses mean

A **recipe** is a named pipeline: its stages, its sigma schedule, its sampler, its
guidance, which transformer it runs on, and which adapters it stacks. Naming one is
how you choose a pipeline, rather than assembling it out of flags and hoping the
combination is one the port has actually run.

```bash
ltx recipes                     # the menu, at each recipe's own shape
ltx recipes --megapixels 0.6 --aspect-ratio 9:16
```

`ltx render --recipe <id>` runs the generative ones. **A recipe is not the same
thing as a `render` flag**, though: three registry entries are executed by other
subcommands, and the menu's `run with` column says which. Keeping them out of the
menu because `render` cannot run them would hide the conditioning choice in the one
place you go to compare conditioning choices.

## Choosing one

| id | run with | what it is |
|---|---|---|
| **`prod`** | `ltx render` | Single-stage on the dev transformer with production guidance, 30 steps. The reference-quality route and the default. Four forwards per step. |
| **`distilled`** | `ltx render` | Two stages — an 8-step ancestral draft at half size, an x2 latent upsample of the video, then 3 deterministic Euler steps at full size. No guidance at all, so one forward per step: **11 forwards against `prod`'s 120**, and 16–17× faster. Start here unless you need `prod`. |
| **`dev-lora`** | `ltx render` | The dev transformer with the rank-450 distilled LoRA overlaid. Costs ~6.9 GB resident on top of the transformer. |
| **`ingredients`** | `ltx ingredients` | Prompt plus one reference sheet, through the Ingredients IC-LoRA. A 2.3 adapter on a 2.5 base — an experiment, not a supported path. |
| **`msr`** | `ltx msr` | Prompt plus up to five reference stills, each held apart by a learned slot vector and its own negative time offset. |
| **`upscale-plain`** | `ltx upscale` | Encode, x2 latent upsample, decode. No transformer, no text encoder, 3.4 GB of checkpoints. |
| **`upscale-refined`** | `ltx upscale` | The above, then the distilled stage-2 schedule. Re-noises to 91% first, so it is licensed to move away from the source. |
| **`upscale-ic-lora`** | `ltx upscale` | The x2 pixel spatial upscaler IC-LoRA — the only IC-LoRA released for 2.5. |

The reference-conditioned routes carry their reference tokens **in the video
sequence** beside the generated ones, held clean and dropped before the decode.
That costs real sequence length — five MSR slots at 640×384 add up to 6000 tokens
against 3120 generated — so budget memory from the total. `ltx recipes` prints both
numbers.

## How a shape is resolved

Ask for a size in megapixels and an aspect ratio, and the resolver derives width
from the megapixel target and then height from the aspect:

```
w = snap(sqrt(M · 1e6 · a), 64)
h = snap(w / a, 64)

snap(v, g) = max(g, floor(v/g + 0.5) · g)      half-up, never 0
```

Deriving both dimensions independently lets the aspect drift much further than
deriving one from the other, which is why height comes from width.

**The grid is 64, for every recipe including single-stage.** The VAE downsamples by
32, but the latent upsampler doubles spatially, so a size that can be upscaled must
already be a multiple of 64 — and one rule for both is better than two. 512×288 is
exact 16:9 on the VAE's own grid and is refused here.

Two consequences worth knowing before you type a number rather than after:

- **The megapixels you get are not the megapixels you asked for**, and can miss by
  around 10%. The request is a target, not a promise.
- **The aspect you get is not the aspect you asked for either.** On the ÷64 grid,
  exactly-16:9 sizes are 1024×576 and 2048×1152 and nothing else under 2.4 MP. The
  achievable ladder is coarse, and it is coarsest in portrait — the orientation
  most likely to be asked for by name.

So `ltx recipes` prints the shape each request actually resolves to. Read it there
before committing to a render.

Nothing silently snaps into something you did not ask for. Three cases, three
behaviours:

1. **Off-grid but resolvable** — the normal case. You get the resolved shape and
   the realised megapixels and aspect alongside it.
2. **Aspect unachievable within tolerance** — resolves to the nearest achievable
   shape and names the nearest *exact* one in the same breath, so you can choose.
   Exact 9:16 at 0.25 MP, for instance, has no close answer: the nearest achievable
   is 384×704, and the nearest exact 9:16 is 576×1024 — more than twice the pixels.
3. **Below the grid floor** — anything that would snap a dimension under two grid
   cells is **refused**, not clamped. A 64×32 render is a shape the model was never
   trained on, and clamping would hide that.

## Frames, and why `--fps` is not cosmetic

Frame counts live on the **8k+1** lattice — 1, 9, 17 … 97, 105 — because the VAE is
causal: frame 0 encodes alone, and every 8 frames after it become one latent. A
duration in seconds **snaps down** onto that lattice.

The audio stream's length is `round(frames / fps * 25)` latents, computed from the
duration rather than from the frame count. At 97 frames, **24 fps gives 101 audio
latents and 25 fps gives 97**. Both render. They are different geometries, and 24 is
the default and the shape everything here has been measured at.

## Guidance: where neutral actually is

Under `prod`, each non-neutral guidance term costs a transformer forward per step,
so the render time follows directly from these.

| term | default | neutral at |
|---|---|---|
| video CFG | 3.0 | 1.0 |
| audio CFG | 7.0 | 1.0 |
| STG | 1.0 | **0.0** — not 1.0 |
| modality | 3.0 | 1.0 |
| rescale | 0.7 | 0.0 (costs no pass either way) |

Two traps. STG is neutral at **zero**, not at one, unlike every other scale here.
And **an empty `--stg-blocks` does not switch STG off**: it runs a full-price pass
that perturbs nothing. To actually stop paying for it, set `--stg-scale 0`, or use a
window (`--stg-start`/`--stg-end`) that excludes every sampled sigma — which is
bit-identical to switching it off, and free.

The `distilled` recipe constructs no guider at all, so it refuses all of these by
name rather than ignoring them.

## What a status means

`ltx recipes` prints a status per row, and it is a claim about a narrow thing.

| status | meaning |
|---|---|
| `gated` | the arithmetic was checked at **this exact shape, duration, step count and pipeline** |
| `sibling` | the same pipeline, checked at a *different* shape. The arithmetic is checked; this shape is not |
| `unmeasured` | this pipeline has not been checked at any shape |
| `ok` | a transform recipe, whose shape comes from its input rather than from a request |
| `noRoute` | a stage this port does not implement |

The distinction exists because **a measurement at one shape says nothing about
another**, and a menu that offered only measured shapes would offer exactly one
shape, in one orientation, at one duration. Rather than relax that rule, the status
is carried per row so the menu can offer a shape and still be honest about what is
known of it. `gated` rows name the file their measurement is read from, so a claim
can be checked rather than trusted.

**One caveat travels with all of it.** A status is a claim about the arithmetic
under a pipeline — never about the picture. A finished render is advisory and is
not evidence: the sampling trajectory saturates at production shape at every step
count, so a recipe that renders successfully has demonstrated nothing about
whether it rendered *well*. Judge the picture by eye. That is what eyes are for,
and it is the only instrument that applies.

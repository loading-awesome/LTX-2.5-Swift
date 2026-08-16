# Performance — what has been measured, and how to measure it again

Measured on an M3 Ultra at 640×384×97 @ 24 fps, release build, GPU exclusive per
run. Every figure here comes from `ltx bench`; re-run it on your own machine before
planning against any of it.

## The four rules

1. **A speed claim needs four kinds of evidence**: the kernel, the render, the
   memory, and the picture. Three of them are numbers and the fourth is not.
2. **Re-run the tests after every optimization.** An optimization that changes the
   arithmetic is a correctness change wearing a performance disguise.
3. **A kernel N× is not a render N×.** Sampling is 97% of the wall clock here, but
   a kernel is a fraction of sampling, and Amdahl applies twice.
4. **Viewing is part of the gate.** Some quality failures are invisible to every
   tensor metric in this repository. The cache sweep below found one.

## The roofline

**Measured 2026-08-12.**

| | LTX 2.5 | how measured |
|---|---|---|
| FLOP per forward | **95.76 TFLOP** | per-case FLOPs × reconstructed multiplicities |
| Achieved during sampling | **13.19 TFLOP/s** | 7.261 s/pass, `bench forward` |
| MLX isolated, this shape mix | **16.05 TFLOP/s** | `bench gemm`, `x @ w.T` |
| MLX isolated, peak single shape | 18.05 TFLOP/s | `ff.net.2` — *not* the ceiling |
| Second vendor (MPSGraph), peak | 18.37 TFLOP/s | `bench gemm` |
| **Fraction of MLX's own rate** | **82.2%** | ratio — ~18% is outside the kernels |
| **Second vendor vs MLX** | **1.018×** at M=3120, **1.225×** at M=17856 | shape-dependent |
| Sampling, share of wall clock | **97.3%** | stage profile |
| Text encode, share of wall clock | 0.7% | stage profile |
| VAE video decode, share of wall clock | **0.5%** | stage profile |

One caveat on the headline: the FLOP count's per-block multiplicities are
reconstructed from module structure rather than measured. Per-case FLOPs and
per-case times *are* measured, so an error in the counting moves the 82.2%
proportionally.

Two things follow.

**There is implementation headroom, and it is not in the kernels.** 82.2% means
roughly a fifth of sampling is spent on elementwise work, norms, modulation, RoPE,
graph overhead and synchronisation. The Metal shaders for the norms are already in
the compile path — `rms_norm.metal` and `layer_norm.metal` are two of the nine
`tools/build_mlx_metallib.sh` compiles into `mlx.metallib` — and the DiT calls
`MLX.rmsNorm` / `MLX.layerNorm` rather than a decomposed mean-square graph. Gemma
keeps its `pow` path, and `rope.metal` is 1-D offset RoPE, which is the wrong
function for this model's split 3-D tables.

**MLX's large-K GEMM collapses above M ≈ 8192.** Swept across M, MLX falls from
18.87 to 15.50 TFLOP/s between M = 7,488 and M = 17,856 while MPSGraph holds around
19 — a ratio of 1.225. It is specific to large K: `ff.net.2` (K = 16384) collapses,
`ff.net.0.proj` (K = 4096) does not. The production shape sits below the M at which
it engages, so this costs nothing at 640×384×97 and 5–6% of the forward on long
clips or large frames — a 10-second 576×1024 render is M = 17,856 and is affected.

`patches/mlx-m3-ultra-large-m-gemm.patch` fixes it by routing those matmuls to a
Steel kernel MLX already ships — a host-side routing change, not a new kernel, and
bit-identical. **Applied 2026-08-13**, measured at 6.11% render-level gain at
M = 9216. The patch lives in `.build/checkouts`, which is gitignored, so a
`swift package reset` or a fresh clone drops it silently and nothing fails —
renders just get slower. `tools/check-mlx-patch.sh` is what makes that loud, and it
runs at the end of the metallib script so a development build cannot lose it
unnoticed.

## Cross-step caching — swept, and off by default

Adjacent diffusion steps often produce nearly the same *change* to the latent. Run
block 0 as a probe, compare its residual against the previous step's, and if it has
barely moved, reuse the previous total residual and skip the rest of the stack.

**Cache the residual, not the output.** Reusing an output pins the render to a
stale latent; reusing a delta applies the change to wherever the trajectory has
actually reached. That distinction is the whole technique.

**Swept 2026-08-13. The default stays off.** Speed and quality columns together,
with a control row:

| `--cache-threshold` | steps skipped | mean step s | speedup | detail | wav vs control | audio envelope | audio spectral |
|---|---|---|---|---|---|---|---|
| 0 (control) | 0/30 | 25.853 | — | control | — | 1.000 | 1.000 |
| 0.01 | **0/30** | 25.983 | 1.005× | probe is free | **identical** | 1.000 | 1.000 |
| 0.05 | 3/30 | 23.414 | 1.104× | same shot | no | 0.962 | 0.917 |
| 0.10 | 20/30 | 9.304 | 2.779× | grain, drifted | no | 0.665 | 0.654 |
| 0.10, cap 3 | 18/30 | 10.981 | 2.354× | ghosting | no | 0.679 | 0.717 |
| 0.15 | 22/30 | 7.648 | 3.380× | smear; flicker fails | no | 0.549 | 0.562 |

2.78× at 0.10 is real and it fails viewing. Nothing in the numeric columns says so
on its own — that is rule 4 earning its place.

**Keep a threshold row that skips nothing.** Here that is 0.01: zero steps reused,
waveform bit-identical to the control. It proves the probe itself costs nothing, so
any speedup at a higher threshold is attributable to the skipping rather than to
measurement drift.

Four invariants the implementation depends on, each with a silent failure mode:

- **One cache per conditioning stream.** The guidance passes run different
  conditioning and their residuals are not comparable. A shared cache compares a
  conditional residual against an unconditional one and reuses across the gap with
  every number staying finite. LTX runs up to four passes per step, so that is four
  caches.
- **Bound consecutive reuse.** The block-0 probe refreshes every step, skipped or
  not; what ages is the total stack residual, which only a full step rewrites. The
  cap is what bounds its age.
- **Never skip the first or last step.** The first has no history; the last lands
  the render.
- **Gate on `max(whole, audio)`.** A whole-sequence probe is a video decision — the
  packed sequence is overwhelmingly video rows — while the audio residual moves
  further per step. A separate audio probe leaves the waveform metrics flat and
  moves lip-sync materially.

One finding worth carrying into any future sweep: caching does not make a render
cleaner, it makes it **softer**, and those two are indistinguishable in every
metric here except detail.

## Measuring it yourself

```bash
ltx bench gemm    --out gemm.json    [--csv gemm.csv]
ltx bench forward --out stages.json  [--csv stages.csv]
```

`bench gemm` takes the model's own shapes from a ranged header read — no payload,
no 42 GB load — and measures them on synthetic operands, in the checkpoint's `[N,K]`
layout via `x @ w.T` rather than a contiguous `[K,N]`, because that is the multiply
a render actually performs. It carries a one-ULP control column so that a reported
`0.00e+00` difference is proven rather than assumed to be a live instrument.

`bench forward` runs a real render and reports per-phase wall clock and peak memory.

**GPU work must be serialised.** A cell measured beside another process is not a
slower cell, it is a wrong one.

### What a bench record enforces

Both commands write a machine-readable record carrying runtime identity (chip,
memory, MLX version, repo revision and whether the worktree was dirty), every
sampling parameter, the guidance schedule per step, the pass count, wall clock per
phase, the per-step array, peak memory, and an open `quality` map. Four rules are
built into that format, each from a specific way of being fooled:

- **The comparison statistic is `meanFullStepSeconds`, never the median.** Step
  times form multiple populations, because a step runs one to four forwards
  depending on guidance. A median once reported a cache that halved sampling time
  as 1.00×. The median is recorded separately, since a kernel win moves the
  per-step cost and a schedule change moves the pass count, and only the first
  composes.
- **Non-finite values encode as JSON `null`** and as an empty CSV cell.
  `JSONEncoder` throws on NaN and ±Inf, which once meant the control run of a sweep
  silently produced no record at all.
- **Peak memory is MLX's allocator peak, and the record says so.** A mmapped
  checkpoint does not appear in the allocator figure, so the allocator peak and the
  process RSS differ hugely and are not interchangeable. Both are recorded.
- **Repeat spread is the full range, `(max − min) / median`**, not a standard
  deviation — with two or three repeats a standard deviation is a statement about
  the sample size. The measured floor is 0.9–1.4%, which is what licenses "any
  claimed gain below about 1.5% is not a gain".

**A record whose render was never looked at is not admissible.** Speed measured on
a broken render is a number about nothing.

## Where the time actually goes on an M3 Ultra

Measured 2026-08-16 on a 256 GB M3 Ultra, against a 22 B bf16 transformer.

**The transformer runs at the machine's GEMM ceiling.** `ltx bench gemm` at the
model's own shapes reports **18.26 TFLOP/s** for MLX and **18.52** for MPSGraph —
two independent backends agreeing to 1.4%, which makes that a hardware number
rather than a library one. A 30-step guided render achieves ~18 TFLOP/s. There is
no implementation win of consequence sitting between this port and the metal.

**It is not memory-bandwidth bound, and the margin is not close.** A forward reads
44 GB of weights. At 4.91 s per forward that is **9 GB/s against the M3 Ultra's
819** — about 1%. Arithmetic intensity is high because a render is a large batch:
2,040 tokens means each weight read serves 2,040 multiplications. The bandwidth
intuition transfers from token-by-token LLM decode, where it is correct, and does
not survive here.

**Residency is not constrained either.** 73.5 GB peak against 256 GB installed and
a 192 GB `iogpu.wired_limit_mb`, with swap flat across the run.

**Per-forward cost is linear in tokens, not `tokens^0.75`.** Two points, same run:

| shape | tokens | s/forward |
|---|---|---|
| 320x192 | 2,040 | 4.91 |
| 640x384 | 8,160 | 18.50 |

4x the tokens costs 3.7x. Fitting `t = a + b·tokens` gives **b = 2.14 ms/token**
and **a = 0.4 s** of fixed per-forward cost. The older `tokens^0.75` law
understates what a resolution change buys and is not used for projections here.

**Fixed per-stage cost is negligible.** Solving across two runs that differed only
in stage-2 step count (`224.2 = X + 12c`, `592.3 = X + 32c`) gives c = 18.4 s and
**X = 3.3 s** — model load, IC-LoRA attach, the x2 upsample and a reference
re-encode, together, at 0.6% of the stage. Nothing is being reloaded between stages.

### MLX `compile`: assessed, not adopted

`mlx-swift` exposes `compile(inputs:outputs:shapeless:_:)` over
`([MLXArray]) -> [MLXArray]`. It is **not** an easy knob here, for three reasons,
and the upside does not justify the risk:

- `DiTForward` is a `struct` holding a `[String: MLXArray]` weight dictionary, not
  an MLXNN `Module`. `compile`'s state parameters take `any Updatable`, which a
  dictionary is not. Captured weights would be traced as constants — acceptable for
  inference, except the LoRA overlay attaches mutably.
- `callAsFunction` takes `Sampling`, `Geometry`, `Set<Int>` and several **closures**
  alongside its arrays. `adaLNInputObserver` and `tapObserver` have side effects,
  and a traced graph fires them once at trace time and never again — which would
  silently empty the L1 tap and the fixture comparisons rather than fail.
- Sequence length changes between stages, so a shape-specialised compile recompiles
  at every stage boundary.

The prize is the **0.4 s** fixed per-forward cost above, which compile's kernel
fusion is the right tool for. That is 8% of a forward at draft resolution and
**2.2% at 640x384** — and the recipes that cost real time are single-stage at full
resolution. Against a measured repeat spread of 0.9–1.4%, a 2% ceiling is barely
outside the noise floor this document already refuses to call a gain.

### The two machine knobs, measured

`ingredients` recipe, 49 frames at 640×384, seed 88, 11 forwards. Stage 1 is the
8-step draft; stage 2 is the 3-step refine.

| `--eval-cadence` | `--cache-threshold` | stage 1 | stage 2 | peak |
|---|---|---|---|---|
| 1 | 0 | **18.8 s** | 16.9 s | 70.4 GB |
| 3 | 0 | 17.7 s (−5.9%) | 16.9 s | 70.4 GB |
| 1 | 0.10 | 17.9 s (−4.8%) | 16.9 s | 70.4 GB |
| 3 | 0.10 | 17.8 s (−5.3%) | 17.0 s | 70.4 GB |

Both are real against the 0.9–1.4% repeat floor, both are small, and **they do not
compose** — 3 + 0.10 is no better than either alone, which is what you would expect
if both are recovering the same fixed per-forward cost.

**Peak memory did not move at cadence 3.** The extra live graph is two latent
streams per step, which is nothing beside 44 GB of resident weights.

Two caveats worth more than the numbers:

- **Stage 2 never moves**, because it runs 3 steps and the loop always forces the
  last one. A cadence only has room to work where there are steps to skip.
- **This is the step cache's worst case and should not be read as its ceiling.**
  The policy refuses reuse on the first step (no history) and on the last (it lands
  in the decoded pixels), and caps consecutive reuse at 5. On an 8-step schedule
  that leaves at most 6 candidates. A 30-step `prod` schedule has 28, and the cache
  has not been measured there. Its quality column is also still empty — the 0.10
  threshold is a measured knee **on a different model**, so the mechanism transfers
  and the number has not been checked against LTX.

### CFG++, and why the published formula could not be copied

Measured 2026-08-16. `ingredients-prod-cfg` against `ingredients-prod-cfgpp`, same seed,
same 30 steps, same CFG-only guidance, 97 frames at 640x384 from the speeder-bike sheet.

| | forwards | sampling | audio level range | audio centroid sd |
|---|---|---|---|---|
| plain CFG | 60 | 446.8 s | 4.5 dB | 140 Hz |
| **CFG++** | 60 | **442.6 s** | **11.8 dB** | **404 Hz** |
| full production guidance | 120 | 873.9 s | 6.0 dB | 122 Hz |

**CFG++ is implemented and correct, and no recipe uses it.** It measurably improves the
audio — 2.6x the level range and 3x the spectral movement of plain CFG on a clip whose
prompt asks for an engine that revs, and 9.6 dB against 6.0 on the full-guidance path — and
it measurably damages the picture. A viewer comparing the clips in motion called the
full-guidance CFG++ render "overly smoothed" and preferred plain full guidance.

That is the mechanism, not a surprise: `CFGPlusPlusTests.cfgPPReducesOvershoot` asserts the
CFG++ step lands *closer* to the guided prediction than the ordinary step does. A
systematically shorter step accumulates less high-frequency detail over thirty of them.
The test was written as though a smaller step were self-evidently a benefit; it is the cost.

The sampler support stays — ``SamplerKind/eulerCFGPP``, the rectified-flow derivation, and
the suite — because the arithmetic is right and the audio result is real. It is reachable
only by a recipe that declares it, and none does. Two recipes that did were registered and
withdrawn the same day.

**On measuring picture quality here.** An earlier version of this note claimed CFG++
carried the reference sheet's detail better, on the evidence of three still frames. Stills
cannot show melting, which is by definition temporal. A frame-to-frame incoherence proxy
computed afterwards separated none of the candidates (1.41-1.50) and also failed to
separate the render a viewer judged good from the ones they judged melted — so it measures
camera motion, not structural coherence. **Picture quality on this route rests on viewing,
and every numeric proxy tried so far has been worse than useless because it looked
authoritative.**

**It is free.** The unconditional prediction it steps along is already computed for CFG,
so the two runs differ by 4 s in 445 — inside the repeat floor.

**k-diffusion's `euler_cfg_pp` cannot be transcribed here.** It is written
`x = x0_guided + sigma_next·(x - x0_uncond)/sigma`, which is correct under the
variance-exploding convention `x = x0 + sigma·eps`, where a step's x0 coefficient is 1
whichever prediction supplies the direction. This schedule is **rectified flow**,
`x = (1 - sigma)·x0 + sigma·eps`, and its ordinary step carries an x0 coefficient of
`1 - sigma_next/sigma` — about **0.005** on the first step of 30 — because the
`-sigma_next·x0/sigma` inside the velocity cancels almost all of the leading term. Taking
the direction from a different prediction stops that cancellation and leaves x0_guided at
coefficient 1: a ~200x amplification of the roughest prediction in the trajectory, at
every step. It renders a uniform orange field.

That failure was measured at guidance 3.0 **and** at 1.5, which is what ruled out a scale
mismatch — the natural suspicion, since CFG++ was formulated as an interpolation and is
usually run at low scales.

The correct substitution recovers the unconditional **noise estimate** and steps with the
schedule's own coefficients:

    eps_u  = (x - (1 - sigma)·x0_uncond) / sigma
    x_next = (1 - sigma_target)·x0_guided + sigma_target·eps_u

which collapses to the ordinary step when the two predictions agree. It also predicts
something the transcribed version got backwards: at `sigma = 1` the `(1 - sigma)` factor
zeroes the unconditional term, so **CFG++ is inert at pure noise** — where the broken form
applied its largest correction. `CFGPlusPlusTests` pins that boundary, the reduction, and
bit-identity of the standard path.

**A note on provenance.** StoryForge's ingredients pipeline sets `cfgPP: 1.0`, which is
what prompted this. Its guidance config is `videoCFG=1.0` — no unconditional pass runs at
all, so CFG++ is inert in the arrangement they ship. A setting is not a measurement.

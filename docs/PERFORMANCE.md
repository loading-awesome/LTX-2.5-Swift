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

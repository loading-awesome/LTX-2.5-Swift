# LoRA training

A LoRA trainer for the 2.5 transformer: a preprocessed cache, a gated flow-matching objective, gradient
checkpointing through all 48 blocks, and a snapshot the render path loads directly.

## 1. Training an adapter

**Training is a library API.** `ltx` offers `render`, `ingredients`, `msr`, `upscale`, `recipes`, `doctor`
and `bench`; there is no `ltx train`. And **`LTXPipeline` is not a package product** — `Package.swift`
exports `.library(name: "LTX25")` and `.executable(name: "ltx")` only, and `LTX25` re-exports no training
type. `LoRATrainer` is in `Sources/LTXPipeline/LoRATrainer.swift`, so before any of this compiles you must
either add `.library(name: "LTXPipeline", targets: ["LTXPipeline"])` to `products`, or write your driver as
a target inside this package that depends on `LTXPipeline`.

```swift
import LTXCatalog; import LTXFoundation; import LTXModules; import LTXPipeline; import MLX
// transformer, textEncoderURL, vaeURL, clipsRoot, cacheRoot and outputURL are your own URLs.

// 1. Encode the dataset once — the VAE and Gemma run per clip, not per step.
let builder = TrainingCacheBuilder(
    options: .init(root: cacheRoot, mediaRoot: clipsRoot, width: 640, height: 384,
                   frames: 97, trigger: "sksstyle", overwrite: false),
    checkpoints: .init(textEncoder: textEncoderURL, dit: transformer, videoVAE: vaeURL))
_ = try await builder.build([.init(video: "a.mp4", caption: "a slow dolly through fog")],
                            note: { print($0) })

// 2. Open it. The fingerprint is checked here, before the run starts.
let dataset = try TrainingDataset(root: cacheRoot, dit: transformer,
                                  requiring: try builder.fingerprint())

// 3. Load the transformer once; `run` shares it across every step.
let header = try SafetensorsHeader.read(from: transformer)
let topology = try TransformerTopology.read(header)
let weights = try MLX.loadArrays(url: transformer)
let base = DiTForward(weights: weights, topology: topology, attentionPath: .fused)
let head = DiTOutputHead(weights: weights, topology: topology)

// 4. Train. Targets are not a parameter: `run` adapts the released 480-module set (§3).
let trainer = LoRATrainer(
    configuration: .init(steps: 600, learningRate: 1e-4, schedule: .cosine(), rank: 64,
                         alpha: 64, snapshotInterval: 100, outputDirectory: outputURL),
    dataset: dataset)
let adapter = try trainer.run(base: base, head: head, header: header) { r in
    print(r.step, r.loss, r.learningRate, r.gradientNorm, r.seconds, r.samples)
}
```

`run` writes `adapter-NNNNNN.safetensors` every `snapshotInterval` steps and again at the end, each beside
an `optimizer-NNNNNN.safetensors` holding `m`, `v` and the bias-correction counter;
`trainer.snapshot(adapter, optimizer: nil, step: n)` writes one by hand. Continue with `resumeFrom:
LoRATrainer.Resume(adapter:step:optimizerState:)`, and pass the moments — without them the first resumed
step takes AdamW's largest effective step and kicks a converged run.

`Configuration.init` (`LoRATrainer.swift:47`) requires `steps` and `outputDirectory`. The rest:

| field | default | decides |
|---|---|---|
| `learningRate` | `1e-4` | base rate; the schedule scales it |
| `schedule` | `.constant` | `.constant`, `.linear`, `.cosine`, `.polynomial` |
| `gradientAccumulationSteps` | `1` | samples per step — the only route to a batch above 1 |
| `maxGradientNorm` | `1.0` | gradient-norm clip; 0 or less disables it |
| `weightDecay` | `0.01` | AdamW's decoupled decay |
| `rank` / `alpha` | `64` / `64` | scaling is `alpha / rank`, carried explicitly |
| `blocks` | nil (all 48) | adapt a prefix of the stack, for a cheap smoke run |
| `gradientCheckpointing` | `true` | §5 |
| `seed` | `42` | sample order and every noise draw |
| `snapshotInterval` | `250` | nil disables intermediate snapshots |

The adapter is written in the released adapters' key convention, so the render path loads it directly:

```
ltx render --prompt "sksstyle, a slow dolly through fog" \
           --lora /path/to/out/adapter-000600.safetensors@0.8
```

`--lora` (`Sources/ltx/Render.swift:163`) is repeatable and takes `NAME` or `NAME@STRENGTH`. `NAME` is a
path, or a filename fragment resolved against the model tree `--checkpoint` sits in — an output directory
outside that tree needs the full path. Adapters apply as a residual overlay, never baked, so stacking two
is well defined.

## 2. What is gated, and against what

| piece | port | gate |
|---|---|---|
| timestep sampler | `LTXFoundation.TrainingTimestepSampler` | `Tests/Fixtures/reference/training_timestep_sampler.json` |
| noising, target, loss | `LTXFoundation.TrainingObjective` | `Tests/Fixtures/reference/training_objective.json` |
| LoRA factors and gradient | `LTXModules.TrainableLoRA` | central finite differences |
| the composed step | `LTXModules.LoRATrainingStep` | loss falls on a synthetic task |
| gradient and optimiser | `LTXModules.TrainableLoRA` + AdamW | `Tests/Fixtures/reference/training_backward.json` |

```
sigma   ShiftedLogitNormal, shift = lerp(1024 -> 0.95, 4096 -> 2.05), NOT clamped
noisy   (1 - sigma) * clean + sigma * noise;  conditioning tokens keep clean
target  noise - clean
video   ((pred - target)^2 * mask).mean([-2,-1]) / mask.mean([-2,-1]).clamp(1e-8)
audio   ((pred - target)^2).mean([-2,-1])          <- UNMASKED
loss    video + audio, per batch element
```

Four of those are silent when wrong and each has a test asserting the plausible *wrong* implementation
gives a different answer: the interpolation's coefficients, the target's sign, the mask on video but not
audio, and the `clamp` that makes an all-conditioning example score **0** rather than NaN. The sampler
agrees to **3e-07**, not bit-exactly — `torch.sigmoid` is not `1/(1+exp(-x))` and the percentile-span
division amplifies the last ulp; `Double` does not close it. Branch selection *is* exact.

## 3. Which modules to adapt

Measured on `ltx-2.5-22b-dev-transformer-bf16`. Every released adapter — the distilled rank-450 and the
2.3 IC-LoRAs — is **480 modules, 48 blocks x 10**: the eight attention projections plus `ff.net.0.proj`
and `ff.net.2`, **video stream only**, audio tower and cross-modal bridges untouched. The conventional
`["to_k", "to_q", "to_v", "to_out.0"]` matches 1216 weights here instead: it adds 384 in the audio tower
and 384 in the bridges, and omits the feed-forward. `TrainableLoRA.releasedAdapterTargetModules` +
`DiTLoRATargets.Scope.videoStreamOnly` reproduces the released set exactly, and is what `LoRATrainer`
uses. Trainable parameters at rank 64: **327.2 M** — 1.31 GB fp32, 3.93 GB with AdamW's two moments.

## 4. Memory — measured, with and without checkpointing

A real forward *and backward* through all 48 blocks with the adapter attached, batch 1, rank 64, fused
attention.

| video tokens | shape | plain peak | checkpointed peak | fwd+bwd (ckpt) |
|---|---|---|---|---|
| 32 | 128x128x9 | 41.92 GB | 40.66 GB | 2.5 s |
| 240 | 320x192x25 | 63.17 GB | 44.54 GB | 3.3 s |
| 780 | 320x192x97 | 120.16 GB | 54.86 GB | 5.9 s |
| 960 | 640x384x25 | 135.29 GB | 58.33 GB | 6.6 s |
| 3120 | 640x384x97 | *did not fit* | **100.78 GB** | 20.7 s |

Both series are **linear, not quadratic**, in the token count — plain `38.9 GB + 101.9 MB/token`,
checkpointed `39.9 GB + 19.5 MB/token`, every point within ~0.3 GB — because the fused SDPA path never
materialises a score matrix. The intercept is the weights; checkpointing does not touch it and cuts the
slope **5.2x**. The production shape fits a 256 GB machine with room to spare, so what binds is wall
clock: **20.7 s/step**, about 170 steps/hour.

## 5. Gradient checkpointing

`mlx_checkpoint` is in the mlx-c that mlx-swift vendors (`Source/Cmlx/include/mlx/c/transforms.h`) but is
not surfaced in the Swift layer; ``LTXModules.GradientCheckpoint`` binds it, `import Cmlx` resolving
transitively through the MLX product. ``DiTForward.checkpointing`` is nil by default, so inference is
untouched. Three rules, each of which fails silently or unrecognisably when broken, each with a test.

**It must be per segment.** One wrap around the whole forward saves nothing — the backward recomputes it
in a single trace. On a synthetic stack: plain 687.9 MB, per-segment 557.8 MB, whole-chain 813.7 MB,
*worse* than not checkpointing. mlx's own `nn.TransformerEncoder` applies it per layer.

**Only arrays passed across the boundary get gradients.** Captured ones are constants to the transform and
the gradient is structurally zero with nothing failing, which is why ``DiTForward.Checkpointing`` is two
closures rather than an overlay: `parameters` hands the block's factors in, `overlay` rebuilds the adapter
from them inside the recomputed trace. The captured conditioning's `x` is replaced by a placeholder for
the same reason, and because a captured `x` would chain all 48 closures into one retention graph.

**It needs a big stack.** `mlx::core::vjp` recurses, and `checkpoint`'s VJP is itself a call to `vjp`, so
N checkpointed blocks nest N traversals. Past ~16 blocks that overflows the 512 KB stack of a
`swift-testing` body or any `Task` on the cooperative pool and surfaces as `EXC_BAD_ACCESS`/SIGBUS in the
guard page, with a backtrace made entirely of `std::function` templates: nothing names this package and
nothing says "stack overflow". ``LTXModules.DeepStack`` runs the step on a thread with 256 MB of stack,
and `LoRATrainingStep.gradients` differentiates inside it unconditionally, so a caller cannot get this
wrong by scheduling training on the wrong executor.

## 6. The step

``LoRATrainingStep`` takes its forward from the caller; ``DiTTrainingForward`` is the one a real run uses,
holding what is frozen — base weights, output head, encoded text, audio stream, geometry.

**``DiTForward`` stops before the output head.** It returns the stream at model width, 4096, while the
objective's target is latent-shaped at 128 channels. ``DiTOutputHead`` closes the gap, driven by the
embedded timestep `DiTForward` already returns. Omitting it does not broadcast, so it fails loudly; using
the wrong stream or a re-derived timestep would not.

## 7. The cache

**The VAE and Gemma run once per clip, not once per step.** Not an optimisation: a 26 GB Gemma load and a
VAE pass per step would more than double a 20 s step for a result identical every time.

**Captions are cached pre-connector** — `video_prompt_embeds` is `enc.projected.video`, not
`enc.features.video`. The connector's weights live in the **DiT**, so caching after it would tie every
cached caption to one transformer. It runs per step from an `EmbeddingsProcessor` alone, not a resident
26 GB encoder.

**A stale cache is refused by fingerprint.** `conditions/` are not interchangeable across model versions —
2.5's Gemma 4 embeddings differ from 2.3's Gemma 3 — and a stale cache does not fail: it trains, and
converges, on features from the wrong encoder. `TrainingDataset.init` compares checkpoint identities,
bucket and trigger word and names every field that differs. Identity is header content, not path, so a
file swapped in place under the same name is caught.

**Frame rate is read from the file, never assumed.** `DiTForward.Geometry.frameRate` drives the RoPE time
axis and audio positions are wall-clock **seconds**, so a 30 fps clip cached as 24 is positioned wrong
along time with no shape error anywhere to say so. Scalars round-trip through `%.17g`.

**One bucket per cache.** `Options.bucket` is `"\(width)x\(height)x\(frames)"`, every clip is resized and
centre-cropped to it, and a clip decoding fewer frames is refused rather than padded.

## 8. The outer loop

`LoRATrainer` is sample, accumulate, clip, step, schedule, snapshot. Everything a resumed run has to
reproduce is recomputed from the step index rather than advanced: the rate from
`schedule.rate(base:step:steps:)`, the sigmas and noise from `LoRATrainer.stepKey(seed:step:)`, the order
from a seeded SplitMix64 shuffle keyed by seed **and** epoch. That shuffle deliberately does not use
`MLXRandom`, or a change in dataset size would change every noise draw. Accumulation sums and divides
once; the gradient norm is reported pre-clip, or a run that is entirely clip-limited is invisible. The
optimiser is `ResumableAdamW` because `MLXOptimizers`' moments cannot be written back — `stateStorage` and
`AdamState`'s initialisers are internal to it — and they go in their own file, since a deliverable adapter
whose key set varied with resumability would have to be stripped before shipping.

`cosine_with_restarts` and `step` are **not** ported: their defaults are derived from the step budget and
their resume behaviour needs care. `LearningRateSchedule.named` refuses them by name rather than silently
mapping them onto a constant rate.

`TrainableLoRA.stateDict` **normalises** the key prefix rather than prepending one. Checkpoint keys carry
the comfy prefix `model.diffusion_model.`; every released adapter names the same module `diffusion_model.`.
Prepend instead and `LoRAOverlay.load` re-adds the base prefix, resolving to
`model.diffusion_model.model.diffusion_model.…` — a well-formed file with correct shapes that nothing can
load. The test holding this asserts the contract end to end: a saved adapter loads through
`LoRAOverlay.load` and resolves against the checkpoint's actual weights.

## 9. The backward, gated

`Tests/Fixtures/reference/training_backward.json` records a torch run of this arithmetic: shapes and
scaling, the base weight and initial factors, the noisy input and target, the sigmas and conditioning
mask, analytic `grad_a`/`grad_b`, and a three-step `torch.optim.AdamW` trajectory of losses and factors at
lr 0.05, betas 0.9/0.999, eps 1e-8, weight decay 0.01. `TrainingBackwardGateTests` pins the port against
all of it. The forward there is a **single adapted linear**, identical on both sides by construction — the
transformer's forward has its own tests, and putting it in the middle would mean a failure could have come
from anywhere. The central differences in `TrainableLoRATests` do not cover it: they differentiate *this
port's own forward*, so they catch a wrong gradient but not one correct for the wrong parameterisation,
and the mask's divisor varies per example and survives into `dL/dpred` — treat it as a constant and the
loss comes out exactly right with the gradient wrong.

**mlx-swift's `AdamW` defaults to `biasCorrection: false`.** Its Adam follows the original paper and omits
bias correction; the torch AdamW this is gated against applies it. With loss and gradients matching to
2e-06 the trajectory is still 0.108 out on the first step and compounds from there — the loss falls the
whole way, along a different path with a much larger effective early step, which reads as a learning-rate
difference rather than a bug. `LoRATrainer` passes `biasCorrection: true` explicitly and takes
`weightDecay` as configuration rather than inheriting a default that differs between the two libraries.

## 10. How the gradient reaches the factors

`LoRAOverlay.trainable(_:scaling:)` builds an overlay from **live** `MLXArray`s instead of from a file.
`delta(forWeightKey:input:)` is already called inside `DiTAttention` and `DiTModules`, so when those
arrays are the ones `valueAndGrad` is tracing, the gradient flows through the transformer's own forward —
there is no separate training path. The overlay must be rebuilt **inside** the differentiated closure from
the model the transform passes in; built outside, it captures arrays the transform has already replaced,
and the gradient is structurally zero with nothing failing.

- **`MLX.valueAndGrad(_:argumentNumbers:)` defaults to `[0]`** — it differentiates the first array and
  silently ignores the rest. A gradient list shorter than the parameter list is almost always this.
  `MLXNN.valueAndGrad(model:)` avoids it entirely.
- **`MLX.eval` inside a gradient trace is fine here**, unlike MLX Python which raises. That matters
  because `DiTForward.runOneBlock` evals after every block on the plain path. The checkpointed path
  deliberately does not: forcing a block's output would materialise exactly what the checkpoint exists to
  drop.

## 11. Limits

- **One example per step.** `DiTForward.Sampling` has no batch axis, so ``DiTTrainingForward`` runs a
  single clip; `gradientAccumulationSteps` is the only route to a larger effective batch.
- **Audio is carried but not trained.** The cache holds no audio, `TrainingDataset` substitutes silence,
  and only the video stream's velocity is scored. The cross-modal bridges read the audio stream on every
  block whether or not the clip had a track, so one must be supplied either way.
- **Base weights are bf16, ~39 GB resident.** A sub-64 GB budget at production shape needs INT8
  quantization, which this port does not implement.
- **No validation renders during training.** The loop snapshots adapters; nothing samples from one mid-run.
- **The step cache must stay off.** `cache.record` pins a live graph node across the step. It is gated on
  `stepCount > 0`, so nothing in the training path reaches it, but a loop that set a step count would.

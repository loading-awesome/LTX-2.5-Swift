# Swift architecture

`Package.swift` is the architecture and is written to be read as one.

## The target graph

```
LTXFoundation   errors, geometry, frame lattice, safetensors, tokenizer   (no MLX)
LTXHardware     chip + memory detection, the memory planner               (no MLX)
LTXCatalog      checkpoint discovery, identification, topology            (no MLX)
LTXRecipes      capability-aware recipe resolution                        (no MLX)
LTXAttention    the attention backend seam, incl. STG passthrough         (MLX)
LTXModules      DiT, VAEs, vocoder, text encoder, LoRA training           (MLX)
LTXPipeline     conditioning, layout, sampler, guidance, decode, mux      (MLX)
LTX25           the public API and actor-owned runtime facade
ltx             a thin CLI over the public API
```

Dependencies point downward only.

### Why the MLX boundary sits there

The four lowest layers link no MLX, so `swift test` for the frame lattice,
latent geometry, safetensors headers, the conditioning layout, the text-feature
projection and the Gemma attention spec runs **in seconds, on any machine, with
no GPU and no checkpoint at all**, against hand-derived expectations written out
in the suite itself. Those are exactly the parts where a wrong answer stays
quiet: an off-lattice frame count or a mis-parsed header does not crash, it
produces a confusing shape error three stages later, so they are the parts that
must stay cheap enough to check on every edit. `TextFeatureProjection` is the
worked example — pure arithmetic with three distinct ways to be wrong that all
produce a correctly-shaped tensor (contract 8), none needing a GPU to catch.

The boundary is about *linkage*, not assets. The tokenizer is MLX-free and its
suite still wants the 26 GB encoder on disk, because the Gemma checkpoint
carries `tokenizer.json` as a 32 MB tensor — a safetensors problem, not an
MLX one.

## One public module

`LTX25` is the supported surface — `Request`, `Checkpoints`, `RenderResult` and
a closed `LTXError`. Everything else is an implementation module, and its
`public` declarations are not compatibility promises. The facade uses explicit
`typealias` rather than `@_exported import`, so those symbols do not leak into
its apparent API. **Not** API: MLX arrays, model layers, tap recorders,
packed-row internals, checkpoint readers, attention kernels, pipeline phases —
`RenderResult` carries URLs and strings, so no tensor type crosses the boundary.

`RenderEngine` is an actor holding one `RenderJob`; `startJob` refuses a second
with `LTXError.engineBusy` rather than queueing it. A model this large must
never end up with two renderers because two callers raced an async API.
Admission is the cheap half — recipe resolution, the memory plan, header-only
checkpoint identity binding, the missing-upsampler check — and all of it happens
before a byte of payload is mapped. The MLX forward that follows is a long
synchronous foreign-runtime call, and running it on the actor would stop the
actor rejecting concurrent admission or reporting state, so the job runs on
exactly one audited detached task: admission proves there is at most one, the
job owns its handle, every terminal path returns to the actor to release the
slot, and progress comes back over an `AsyncStream<Event>`. `RenderJob` writes
the `.provenance.json` sidecar.

## CLI

Thin by rule. Every subcommand is argument plumbing over a call into a library
target; logic that needs a test belongs where it can be tested without spawning
a process. The commands and their flags are in [`CLI.md`](CLI.md).

`render`, `ingredients` and `msr` are separate commands rather than modes of one,
because the reference-conditioned routes make the transformer's sequence longer
than the one that gets decoded, and `render` sizes its noise, masks and unpatchify
from a single token count. `Recipe.command` records which command runs each recipe
so `render` refuses the others by name instead of by guess.

The root command is `AsyncParsableCommand` **and must stay that way**: `render`
drives an actor so it is async, and ArgumentParser only awaits an async
subcommand when the root is async too. Declared synchronous, it prints a warning
and then never calls `run()` — the command parses its arguments, renders
nothing, and exits 0.

## MLX needs a bootstrap step: build its metallib first

**`swift build` does not compile `.metal` files at all.** Verified in isolation,
not inferred: a package containing one trivial kernel builds clean, produces no
`.metallib`, creates no resource bundle, and emits no "unhandled files" warning.
Metal compilation belongs to Xcode's build system, not to SwiftPM's command-line
build. mlx-swift depends on it anyway — `Package.swift` sets
`METAL_PATH="default.metallib"` and `SWIFTPM_BUNDLE="mlx-swift_Cmlx"`, and its
excludes reference a `PrepareMetalShaders` step that is not in the tree (0.31.6
ships only a `CudaBuild` plugin). So without the README's bootstrap step,
**every** MLX entry point dies with `Failed to load the default metallib`.
Three properties of that failure are worth knowing before debugging it:

- it happens at **device resolution during initialisation**, so it fires even when
  the caller asks for `Device(.cpu)` — there is no CPU fallback to retreat to;
- it **aborts the whole test process**, which is why the MLX-free suites must never
  gain an MLX dependency: one MLX test would take the geometry and identity suites
  down with it;
- the colocated lookup is for **`mlx.metallib`**, while the bundle lookup is for
  **`default.metallib`**. Dropping a correctly built `default.metallib` next to the
  binary achieves nothing, which cost a debugging round here. The full order is in
  `load_default_library` in `mlx/backend/metal/device.cpp`.

`tools/build_mlx_metallib.sh` compiles the nine kernels in mlx's prepared shader
tree and copies the result beside every built binary, at a path that differs per
product. Nine is the whole set: every other kernel is JIT-compiled at runtime
from source strings embedded in the library, which `MLXCapabilityTests` checks
by exercising binary, unary, reduction, softmax, matmul, bf16 and fused SDPA
paths separately, so a partial failure shows up as a partial failure.

## Targets

| target | state |
|---|---|
| `LTXFoundation` | geometry, lattice, safetensors reader/writer, Gemma tokenizer, text projection, conditioning layout, training objective and schedules — all MLX-free |
| `LTXCatalog` | checkpoint identity, adapter catalogue, transformer and text-encoder topology, read from the checkpoints themselves |
| `LTXModules` | Gemma encoder, projection, connectors, conditioning pipeline, DiT attention/blocks/48-block forward/output head, both VAE decoders, vocoder, LoRA training step and optimiser |
| `LTXPipeline` | conditioning, sampler, guidance, decode, mux, the renderers and the trainer's outer loop |
| `LTXAttention` | a placeholder holding the seam's STG note — fifteen lines and one constant, depended on by `LTXModules` and linking MLX but running nothing, since attention and STG are implemented in `LTXModules` |
| `LTXHardware` | machine description and the memory planner |
| `LTXRecipes` | the recipe registry, capability-aware resolution and grid alignment |
| `LTX25` | the single-actor facade and concurrency boundary |
| `ltx` | command-line executable built on the `LTX25` facade |

`swift test` runs the whole package. Suites whose checkpoints are absent return
early and report green, so read a test count alongside where it ran; the README
names which those are.

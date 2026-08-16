# Fragile contracts

Each entry is a constraint on this code and what it costs to break it. Most were
found by reading the reference implementation rather than by a failing test —
#15 is the entry that explains why a tolerance check structurally cannot find
them.

Paths like `ltx_core/...`, `transformers/models/gemma4/...` and `vocoder.py` are
citations into the upstream reference implementation, not files in this repository.

---

## 1. Never emulate CUDA's RNG

The LTX 2.3 port lost weeks to a bug that presented as kernel error and was
not. Production renders looked *soft* against the CUDA reference; per-block
cosine drift through the stack was read as accumulating bf16 error, and a
Metal flash-attention rewrite was planned to match CUDA's reduction order.

That investigation produced one definitive negative result: PyTorch SDPA on
Swift-captured Q/K/V showed MLX bf16 SDPA matching the fp32 reference **more**
accurately than every PyTorch CUDA backend variant. Attention was already
inside the equivalence class. A custom kernel would not have moved the metric.

The actual cause: `torch.randn` on CUDA does not run Philox4x32-10 per element
with a monotonically increasing subsequence. It runs a grid-stride loop,
`distribution_elementwise_grid_stride_kernel<unroll=4>(curand_normal4)`, whose
stride is `SMs × 8 blocks/SM × 256 threads`. On the A100 that is 221,184.
Swift's Philox matched for the first 221,184 elements and then silently
diverged. Short sequences fit inside one iteration and agreed; production
shapes did not. Swift was denoising a *different starting point*, and the
output looked like a soft variant of the reference.

The 2.3 port's fix was to emulate the grid-stride loop exactly. **This port does
not do that**, deliberately.

The emulation was load-bearing for a product that wanted to reproduce a CUDA
render from a seed. It bought that at the cost of a constant tied to one GPU
model, a hardcoded per-call offset table tied to one shape, and no self-test for
either. Move to a different card and the stride is wrong, which reproduces the
failure above exactly, with nothing to announce it.

**The rule that replaces it: this port uses MLX's own RNG, and a seed is
reproducible within this port only.** A seed is not portable across backends and
this port will not claim that it is. When a render has to start from a specific
latent, supply the latent — do not expect a seed to reconstruct one drawn
somewhere else.

**Do not revive the custom-kernel investigation** on the strength of a drifting
cosine. Check where the noise came from first.

### A pipeline can have more than one noise stream

`DistilledPipeline` steps with `EulerAncestralDiffusionStep(eta=1.0)`, and an
ancestral step **draws fresh noise on every step** — from a generator seeded
independently at `seed + 10000`. Only the initial latent comes from the noiser;
every step after the first adds a draw of its own, per modality:

```
initial draw          1 video + 1 audio per stage
ancestral (stage 1)   7 video + 7 audio   (one per step after the first)
```

A port that accounts only for the initial draw starts from the correct latent
and then diverges at step 1, and nothing about the shapes would look wrong. A
pipeline's RNG consumers are the noiser, the sampler's step function and any
stochastic scheduler, and they differ by path: the dev path's Euler step is
deterministic and draws nothing after the initial latent; the distilled path is
not.

## 2. Sharpness cannot be read from Laplacian variance

The 2.3 reference kept a Laplacian-variance ratio of 12.0 vs 19.0 against a
render that was visually equivalent. The metric reflected high-frequency
texture grain, not perceptual softness, and was discredited in three separate
experiments. Do not chase it.

More generally: **look at pixels.** Mean and standard deviation reported "low
contrast" for two rounds of an investigation while the frames were actually
structureless noise. A PIL grid of frames `[0, n/3, 2n/3, n-1]` costs nothing.

## 3. Audio latent count is `round`, not `ceil`

```python
num_of_latents_from_frames = round((frames / frame_rate) * latents_per_second)
```

Changed from `math.ceil` in the 2.5 PR. The same frame count can now yield one
fewer audio latent than 2.3. This surfaces as a **shape mismatch, not a wrong
value** — and a port that carries the 2.3 `ceil` will look fine until a frame
count lands near the boundary.

## 4. Frame count lives on the `8k + 1` lattice

`seconds_to_num_frames` snaps **down** to `8k + 1`, and bumps **up** to the
next grid point only when snapping undershoots the minimum:

```python
frames = (raw_frames - 1) // 8 * 8 + 1
if frames < min_frames:
    frames = min(-(-(min_frames - 1) // 8) * 8 + 1, max_frames)
```

This is the VAE's causal temporal grid, and it is the same `8` that appears in
`upscale_ratio = (a * 8 - 7, 32, 32)`. Off-lattice frame counts are not
renderable.

## 5. The AV latent is nested, ordered (video, audio)

The dual-CFG guider splits the packed latent at
`math.prod(latent.unbind()[0].shape[1:])` — element 0 is **video**. A port that
assumes a plain tensor, or the other order, will appear to work and be wrong.

## 6. STG skips more than the attention call

When a block is STG-perturbed, `out = v` — and `q_norm`, `k_norm` and the
entire RoPE application are skipped along with it. Only the value projection
survives. Applying RoPE and then discarding it is not equivalent, and gating
only the attention call is the obvious wrong implementation.

Per-head gating (`to_gate_logits`), where present, still applies afterwards.

## 7. Modality guidance and dual CFG compose additively

- Modality: `cfg_result + (cond_pred - mod_pred) * (modality_scale - 1.0)`,
  where `mod_pred` is a pass with **both** `a2v` and `v2a` cross-attention
  severed. Reference default 3.0; 1.0 disables and must skip the extra pass.
- Dual CFG: video scale over `[..., :v]`, audio scale over `[..., v:]`, with
  `disable_cfg1_optimization` set so the uncond pass runs even when one scale
  is 1.0.

With STG, modality guidance and dual CFG all engaged, **one sampler step is up
to four forward passes**. Budget compute for that, and never assume an
intermediate tensor pulled out of a step came from the conditional pass —
anything inspecting the stack mid-step has to know which branch it is looking
at.

## 8. Text projection input dim is derived, never hardcoded

```python
projection_in_dim = hidden_size * (num_hidden_layers + 1)
```

`3840 * 49 = 188160` — the value measured in the 2.3 checkpoint and the constant
the 2.5 PR deliberately replaced with the formula. Hardcoding it is exactly the
bug the PR fixed.

**Confirmed on 2.5, 2026-08-11.** The reference hands the feature extractor 49
hidden states, each `[1, 1024, 3840]`: 48 transformer layers **+ 1 embedding
output**. So the projection consumes every layer's hidden state plus the
embeddings — not the last layer — and the `1024` is contract 9 showing up in the
same reading.

The projection itself is two Linears, not one: `video_aggregate_embed`
(188160 → 4096) and `audio_aggregate_embed` (188160 → 2048), **both `bias=True`**
on V2 where V1's single `aggregate_embed` is `bias=False`. Their outputs are
`[1, 1024, 4096]` and `[1, 1024, 2048]`.

### The three ways to get `FeatureExtractorV2` wrong

Each of these produces a correctly-shaped tensor, so none is caught by a shape
check. `Sources/LTXFoundation/TextFeatureProjection.swift` implements the reference
math.

1. **Normalising over the wrong axis.** `variance = mean(encoded**2, dim=2)` on
   `[B, T, D, L]` is a mean of squares over the **hidden** dimension, computed
   separately for every `(token, layer)` pair. Every layer of every token gets its
   own scale. Checked by hand: token 0 layer 0 spans
   `[-1.0, -0.8, -0.6, -0.4]`, mean of squares `0.54`, `rsqrt(0.54 + 1e-6)` =
   `1.3608263…`; layer 1 of the same token has mean of squares `0.41` and a
   different scale.
2. **Flattening layer-major.** `reshape` on `[B, T, D, L]` is hidden-major and
   layer-minor — element `(d, l)` lands at `d * L + l`. "Concatenate the layers"
   is the intuitive reading and gives `l * D + d`: the same 188 160 numbers in a
   different order, after which the Linear reads every weight against the wrong
   input element. Confirmed by probing the reference: `encoded[0,0,1,0] = -0.8`
   normalises to `-1.0886610746383667` and lands at flat index **2**, which is
   `1 * L + 0`, not `0 * D + 1`.
3. **Rescaling once.** `_rescale_norm(x, target, source)` is
   `x * sqrt(target / source)`, and it is applied **per stream, before that
   stream's Linear**: `sqrt(4096/3840) = 1.0327955…` for video,
   `sqrt(2048/3840) = 0.7302967…` for audio. Computing it once and reusing it
   leaves one stream wrong by a constant factor — and a constant factor is exactly
   what `cos` cannot see (measured: a ×1.05 error leaves `cos` at
   `1.000000000`).

Note also that the denominator is `embedding_dim` from the **Gemma config**, while
the variance axis is `D` from the **tensor shape**. They are equal on every real
checkpoint and are read from different places, so they are kept separate in the
port rather than folded into one constant.

**And on this checkpoint it lives in the text encoder, not the transformer.**
`_build_caption_projections` returns `(None, None)` when
`caption_proj_before_connector` is set — "19B models: projection is in the
transformer. 22B models: projection is in the text encoder." Ours is the 22B, so
`LTXModel.caption_projection` is None by design and the port must place the
projection on the encoder side.

## 9. Gemma-4 LTXAV tokenizer forces `min_length = 1024`

`ltxav_gemma4_tokenizer` raises the wrapped tokenizer's `min_length` from 1 to
1024. Token count changes conditioning length, which changes shapes downstream.

**Measured:** everything on the encode path is 1024 wide — the attention mask is
`[1, 1024]` and the video features `[1, 1024, 4096]` — for a seven-word prompt.
So the padding is not incidental; it is the shape of the conditioning.

Two consequences the port must honour, both from
`embeddings_processor.create_embeddings`:

- **Pad side is normalised, not assumed.** `_compute_right_pad_order` computes a
  *stable* `argsort` of the binary mask and permutes the features so valid tokens
  precede pads. It is idempotent for already-right-padded input, and the same
  permutation is reused for audio. A port that pads on the other side and skips
  the permutation produces a correctly-shaped tensor with its tokens in the wrong
  positions — which no shape check catches, so the pad side has to be established
  and recorded rather than assumed.
- **The video encoding is masked, the audio encoding is not.**
  `video_encoded = video_encoded * binary_mask` has no audio counterpart in the
  reference. Do not "fix" that asymmetry.

## 9b. The audio mel is stereo, and its frame count is not the latent count

The audio decoder emits `[1, 2, 101, 64]` from a 26-latent audio tensor —
**2 channels**, 101 mel frames, 64 mel bins (`mel_bins` default in
`audio_vae.py`), in a single decoder call.

Neither the channel count nor the frame count follows from the audio latent shape
`(1, 8, 26, 16)`, so both are things to read off the reference rather than derive.
A port that assumes mono would produce half the tensor and pass a cosine check on
the channel it did compute.

## 10. Keyframe embedding masks a specific token set

The learned `[1, inner_dim]` marker is added to tokens whose temporal start is
0, **minus** the trailing `num_guide_tokens`, **plus** the explicit
generated-keyframe slots. The reference raises rather than guessing when
`keyframe_idxs` is not a whole number of latent frames, or when recorded
tokens-per-frame disagrees with the latent being sampled.

Reproduce the raise. Silently proceeding lands the embedding on the wrong
tokens.

## 11. Never compare vocoder waveforms numerically

Identical mel inputs can give different waveforms. Compare at the mel; evaluate
the waveform perceptually. Run the reference twice in separate processes and
every transformer intermediate, every sampled latent and the decoded video come
back bit-identical while the waveform alone moves, by `max_abs 1.911e-03`
(cos `0.999808058956`).

**The mechanism is cuDNN algorithm selection in the transposed convolutions**,
not an initial phase: the reference vocoder holds no stochastic element of any
kind, and the first tensor to differ across runs is `vocoder.ups[0]`, the first
`nn.ConvTranspose1d` — cuDNN's conv-backward-data path, the classic atomics /
split-reduction kernel. The tell is that forcing `cudnn.deterministic=True`
collapses the spread **and changes the answer**, by `max_abs 1.727e-02` on a
401-frame mel and `7.324e-04` on a 101-frame one; an algorithm choice behaves
that way and a random draw does not.

**And it is shape-dependent.** On `torch 2.13.0+cu132`, cuDNN 92101, RTX PRO
6000 Blackwell, a 401-frame mel is 20/20 bit-identical in-process and 3/3 across
processes, while a 101-frame mel gives 5 distinct outputs in 8 runs (worst pair
`3.468e-03`). Neither case licenses comparing one implementation's waveform
against another's: where the reference reproduces itself bit-for-bit, matching it
would demand being closer to it than an exact evaluation is, which only the same
kernel can do; where it does not, its own spread is larger than anything a
comparison would be trying to see.

**16 kHz is a far better boundary than 48 kHz.** The 16 kHz waveform's
run-to-run spread is `5.881e-05` (cos `0.9999997`); the log-mel re-analysis plus
the BWE generator amplifies that roughly **60×** to `3.468e-03` at 48 kHz. Check
the vocoder at the 16 kHz waveform, before the bandwidth-extension stage, rather
than at the final output.

**The mel is the audio boundary, and without it there is no numerical audio
check at all.** At the end of the pipeline the waveform is the only audio tensor
there is, and it is the one tensor that can never be compared — so an audio test
built on the final output looks like it covers audio and covers nothing. Every
fault in the audio VAE decoder (wrong upsample, wrong norm, wrong `tanh` gating)
lives in that gap. The seam is in `ltx_core/model/audio_vae/audio_vae.py`:

```python
def decode_audio(latent, audio_decoder, vocoder):
    decoded_audio = audio_decoder(latent)              # <- mel: COMPARE HERE
    waveform = vocoder(decoded_audio).squeeze(0).float()   # <- never compare
```

One thing follows from that shape: if the decoder fires more than once the
family is chunked, in which case the mel arrives per tile and one
implementation's tiling is not comparable to another's.

**The mel is bit-deterministic, which localises the nondeterminism to one
module.** Decoded twice in separate processes it is identical both times, so the
tensor immediately upstream of the vocoder is stable and the nondeterminism
enters **inside** the vocoder and nowhere earlier. That makes the whole audio
path up to and including the VAE decoder checkable by decoding the same latent
twice and comparing mel, and it makes the waveform the one stage where a
difference proves nothing — so claim nothing about the audio path from a check
that never reaches the mel.

## 12. Audio decode normalisation pins output std

An audio decoder ending in `wave / max(5 * std, 1.0)` pins output standard
deviation to exactly 0.2 for any raw std at or above 0.2. Two runs matching at
`rms 0.2031` vs `0.2029` is **not** evidence of prompt insensitivity or of
anything else — it is what the normaliser emits regardless.

This retracted a whole line of investigation. An audio path that ends in such
a normaliser carries no information at all in its summary statistics.

## 13. The Gemma-4 encoder has two layer kinds, and a per-layer scale

Read from `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (686 tensors) and
`transformers/models/gemma4/modeling_gemma4.py`, 2026-08-11. Verified layer by
layer against the weights by `TextEncoderTopology.verify` — all 48.

**The config and the tokenizer are inside the checkpoint.** There is no
`config.json`: the HuggingFace config is JSON in the safetensors metadata under
`gemma_config`, and the tokenizer is a 32 MB `tokenizer_json` tensor alongside the
chat template and generation config. So configuring this model from the checkpoint
is not a preference — it is the only available route,
and substituting defaults would be guessing about a model with two layer variants.

**48 layers in exactly two kinds**, selected by `layer_types`, with
`full_attention` at indices 5, 11, 17, 23, 29, 35, 41, 47 — every sixth:

| | sliding (40) | full / "global" (8) |
|---|---|---|
| `q_proj` | `[4096, 3840]` — 16 × 256 | `[8192, 3840]` — 16 × **512** |
| `k_proj` | `[2048, 3840]` — 8 KV heads | `[512, 3840]` — **1** KV head |
| `v_proj` | present | **absent** |
| `o_proj` | `[3840, 4096]` | `[3840, 8192]` |
| `q_norm`/`k_norm` | `[256]` | `[512]` |
| RoPE | `theta 1e4`, full rotation | `theta 1e6`, **partial 0.25** |

Four things there, each independently able to produce correctly-shaped, wrong
output:

1. **`global_head_dim` is 512, not `head_dim`'s 256**, and
   `num_global_key_value_heads` is 1 against `num_key_value_heads`' 8. Building 48
   identical layers fails loudly on the shapes, so that much is self-correcting.
2. **`attention_k_eq_v: true` — the global layers have no `v_proj` because V *is*
   K.** Filling V from K without knowing why is arithmetically right and leaves the
   port unable to explain itself; assuming a missing tensor means a broken
   checkpoint is worse. Both directions of this mismatch are rejected.
3. **RoPE is per layer kind.** `rope_parameters` is a dict keyed by layer type. One
   theta applied to all 48 layers is wrong in either 8 or 40 of them, and partial
   rotation means only 128 of the global layers' 512 head dims rotate at all.
4. **Text attention is causal.** `use_bidirectional_attention: "vision"` scopes
   bidirectionality to vision tokens. Making the text encoder bidirectional because
   "it is an encoder" changes every hidden state.

**`layer_scalar` is load-bearing, and it scales the residual stream.** One element
per layer, `hidden_states *= self.layer_scalar` as the *last* operation in
`Gemma4DecoderLayer.forward` — after both residual adds, so it rescales the whole
stream and not just the block's contribution. Registered as a buffer initialised to
ones, which is exactly how a port comes to ignore it. Measured on this checkpoint:

```
48 values, 41 distinct, min 0.004547  max 0.925781
layer 0: 0.052979    layer 47: 0.049805
```

Ignoring it means substituting 1.0, which overscales the worst layer by
`1/0.004547` = **220×**. (The spread *across* layers is 204× — a different claim,
and not the one that matters for a port that never reads the tensor.)

The partition is checkable against the weights alone, with no GPU: split the 48
layers by their **actual tensor signatures** rather than by what the config says,
and the layers with no `v_proj` come out as exactly the eight the config calls
`full_attention`. That is what `TextEncoderTopology.verify` reads off the
safetensors header, so a checkpoint whose weights and config disagree is rejected
rather than built from the config regardless.

This also **explains contract 8's per-layer RMS norm**. Because `layer_scalar`
rescales each layer's output by a different learned factor, the 49 hidden states
have wildly different magnitudes and are not comparable as-is — so
`norm_and_concat_per_token_rms` normalises each layer independently before the
projection consumes them. Two independent observations that account for each other.

**Four norms per layer, not two.** Gemma's sandwich layout normalises the attention
and feed-forward *outputs* before each residual add, on top of the usual pre-norms:

```python
h = input_layernorm(h); h = self_attn(h)
h = post_attention_layernorm(h); h = residual + h      # <- norm before the add
h = pre_feedforward_layernorm(h); h = mlp(h)
h = post_feedforward_layernorm(h); h = residual + h    # <- and again
h *= layer_scalar
```

A LLaMA-style two-norm layer silently drops half of them.

**One coincidence not to lean on.** `sliding_window` is 1024 and contract 9 forces
the tokenizer to exactly 1024 tokens, so at this length the sliding mask and the
full mask are the same mask. That makes the *masking* difference inert today. It
does **not** make the layer kinds inert — head geometry and RoPE differ at every
length — and if `min_length` ever rose above 1024, 40 of 48 layers would change
behaviour with nothing to announce it.

Two branches in `Gemma4DecoderLayer.forward` are inert here and are **refused**
rather than ignored if a checkpoint enables them:
`hidden_size_per_layer_input` (0) and `enable_moe_block` (false).

### The 49 hidden states are not the 49 you would build

Measured on the reference encoder's own per-layer hidden states, 2026-08-11. The
obvious reading — embeddings followed by all 48 layer outputs — is wrong in its last
element, and the error is a factor of ten on 1/49th of the projection's input.

The tuple is:

```
hidden[0]      = embed_tokens(input_ids) * sqrt(hidden_size)
hidden[1..47]  = the outputs of layers 0..46
hidden[48]     = final_norm(output of layer 47)      <- NOT the raw output
```

So **layer 47's raw output never appears**, and the last state is post-`model.norm`.
Established by signature rather than by reading the collection code (transformers 5.x
captures hidden states generically, outside the model's own `forward`): if a tensor is
`normed * W` then dividing by `W` leaves per-token RMS 1. Measured:

| hidden state | per-token RMS of `h / model.norm.weight` |
|---|---|
| `hidden.000` | 0.19, 1.21, 0.38, 0.32 — not normed |
| `hidden.047` | 54.1, 54.5, 53.4, 52.3 — not normed |
| `hidden.048` | **0.998, 1.002, 0.997, 1.000** — normed |

Consistent with the magnitudes: `hidden.047` has RMS 4.63 while `hidden.048` has 47.2,
because `model.norm.weight` reaches 600.

**`embed_scale` is `sqrt(hidden_size)`**, i.e. `sqrt(3840) = 61.9677`, applied inside
`Gemma4TextScaledWordEmbedding.forward`. Confirmed against the reference's own
`hidden.000`: `embed_tokens(ids) * sqrt(3840)` reproduces it to bf16 rounding, while
omitting the scale misses it by the size of the tensor itself.

### The arithmetic the weights cannot tell you

Everything above is checkable against the checkpoint. These are not, and each is a
silent error. They are specified in
`Sources/LTXFoundation/GemmaAttentionSpec.swift`, read off
`Gemma4TextAttention` and `Gemma4TextRotaryEmbedding` themselves.

**1. There is no `1/sqrt(head_dim)`.** `Gemma4TextAttention.__init__` sets
`self.scaling = 1.0` and passes it to the attention interface unchanged. The softmax
scale is folded into the learned `k_norm` weights, which is visible in the
checkpoint: they sit at ≈0.122 for the 256-dim heads and ≈0.061 for the 512-dim
ones, where a plain RMSNorm scale would sit near 1.0 — as `q_norm` does (a uniform
1.0312). Applying the conventional scale double-scales the logits by 16–22× and
saturates the softmax toward one-hot. Shapes are unaffected.

**2. `v_norm` exists, applies to every layer, and has no weights.** It is
`Gemma4RMSNorm(head_dim, with_scale=False)` — so V is RMS-normalised with no
learnable scale, and because `with_scale=False` allocates no parameter, **there is
no `v_norm` tensor in the checkpoint at all** (verified: zero matches against 48
`k_norm` and 40 `v_proj`). A port assembled by enumerating checkpoint tensors cannot
discover this operation, and omitting it leaves V unnormalised in all 48 layers.
This is the one finding on this page that no amount of checkpoint inspection would
have produced.

**3. On the K-shares-V layers, V aliases the *pre-norm* projection.**

```python
k = k_proj(h)
v = v_proj(h) if v_proj is not None else k   # <- binds the PRE-norm k
k = k_norm(k); k = rope(k)                   # rebinds the name, not the tensor
v = v_norm(v)                                # no rope on V, ever
```

So V is `v_norm(k_proj(h))` while K is `rope(k_norm(k_proj(h)))`: they share the
projection and nothing else. Writing `v = k` *after* computing K — the natural
reading of "K equals V" — gives V both the k_norm scale (≈0.061) and the rotation.

**4. RMSNorm is `normed * weight`, not `normed * (1 + weight)`.** Gemma-2 and
Gemma-3 use the `+1` convention, so this is inherited wrongly from the obvious
place. Confirmed twice: the reference multiplies by `self.weight` directly, and the
checkpoint's norm weights centre on 1.0 rather than 0.0. Under the `+1` reading
every norm in the model is roughly doubled. The norm is also computed in **fp32**
and cast back (`self._norm(hidden_states.float())`), and uses `pow(x, -0.5)` rather
than `rsqrt` — the reference comments that this is deliberate, for Torch/JAX
agreement.

**5. RoPE: two formulas, and they differ in the denominator.** Both compute
`1 / theta^(2i/denominator)`, but `default` divides by `dim = head_dim * factor`
while `proportional` divides by the **full `head_dim`** even though only `factor` of
it rotates. Measured:

```
sliding   128 freqs, all rotating,  base 1e4, /256   inv_freq[1] = 0.93057203
full      256 freqs,  64 rotating,  base 1e6, /512   inv_freq[1] = 0.94746351
```

Recomputed by hand: `10000^(-1/128) = 0.9305720` and `1000000^(-1/256) = 0.9474635`.
Applying the `default` formula to the global layers would divide by 128 instead of
512 — a completely different frequency set, in eight of the 48 layers.

**Partial rotation is expressed as zero frequencies, not a tensor split.** The
global layers' `inv_freq` is 256 long with the first 64 real and the remaining
**192 zero**. A zero frequency gives `cos = 1, sin = 0`, so the rotation is the
identity on those dimensions and the whole head passes through one uniform
operation.

`inv_freq` is built in **float32** in the reference, so a `Double` port agrees only
to ~1e-8 — which is why the Swift test compares relatively at fp32 precision rather
than asserting a tighter bound it cannot meet.

## 14. A tuned constant does not survive a model version

Distilled LoRA strength, an image-to-video strength cap, a stage-2 tail
refinement, temporal upsample settings — every one of these was fitted to 2.3, and
none of them transfers. They are the most tempting thing to carry over, because
each is a single number that looks like a fact about the pipeline rather than a
fact about the weights it was fitted against.

Carrying one produces a render, never an error. Derive the equivalent against 2.5
or leave the knob at its neutral value, which is the same rule contract 1 states
about a seed and for the same reason.

## 15. The keyframe marker is a real defect that no tolerance can see

`patchify_proj`'s output is not what block 0 receives. Between them,
`TransformerArgsPreprocessor.prepare` runs one line:

```python
x = self.patchify_proj(modality.latent)
x = apply_keyframes_absolute_embedding(x, modality.keyframes_mask,
                                       self.keyframes_embedding_provider)
```

and that function is `hidden_states + (keyframes_mask > 0).to(dtype) *
embedding.to(dtype)`, where `dtype` is `hidden_states.dtype` — the mask thresholded to
0/1 and both operands cast to the *stream's* precision, which is bf16. Adding in fp32
and rounding afterwards is a different tensor, measured: it reproduces 740,587 of the
983,040 elements of block 0's video input and differs on 242,453, essentially all of
the 245,760 that carry the marker.

Which tokens carry it is decided outside the model, in
`VideoLatentTools._first_frame_keyframes_mask`: the first `tokens_per_latent_frame`
entries of a zeroed `denoise_mask`, set **unconditionally** — not only when the caller
supplied keyframes. The video encoder is causal, so the target's first latent frame
covers one pixel frame where every later one covers eight, which makes it the same
token class as a generated keyframe slot. Contract 10 has the rest of the set.
**Video only:** `keyframes_abs_pos_embedding` is created in `_init_video`, and the
audio preprocessor is constructed with no provider, so the audio stream never receives
one.

The reference's own comment says the parameter is zero-initialised so that a
checkpoint predating it behaves identically. **This checkpoint's is trained**, and it
is small: `[1, 4096]` bf16, min `-3.433e-03`, max `2.579e-03`, std `8.00e-04`. It is
a weight rather than an activation, so nothing in a tensor inventory of the model
announces that it is applied at all.

**And that is the finding.** The marker lands on the first latent frame's tokens —
60 of a 240-token video grid — and the difference it makes to the block-0 input is
*smaller than bf16 rounding at that magnitude*, by a factor of about seven.
So a port that omits it produces a stream that no tolerance-based comparison at that
boundary could ever reject, and none at a later depth either: the stack starts from a
slightly wrong stream, all 48 blocks inherit it, and the render is quietly wrong. A
defect that is real, cheap to introduce, and smaller than the precision anything
downstream is stored in is the worst thing this file records.

**What catches it is equality.** Applying the marker reproduces the reference's
block-0 video input **bit for bit**, all 983,040 elements. That is stronger evidence
than any tolerance, and stronger for a structural reason rather than a lucky one: a
tolerance can never be tighter than the precision the tensor is stored in, while an
equality check has no such floor. **Where a port can reproduce a reference tensor
exactly, prefer equality — a tolerance is the fallback for boundaries where the
arithmetic genuinely differs, not the default.** The rotary reconstruction is the same
shape of evidence one seam over: bit-exact on both video grids, and only the audio
grid needs a tolerance at all.

The marker is applied where `DiTForward` builds the block-0 video input, and the
figures above are the reason the line is there rather than a note that it might be.

**The general lesson, which is not about keyframes.** A numerical tolerance bounds how
much *arithmetic* may differ and is silent about everything below its floor. "Close
enough" is therefore not the same as "correct": a term that is cheap to omit — an added
embedding, a mask, a permutation, one of 49 hidden states — can be missing outright and
still perturb the tensor less than bf16 rounding does. Structure needs a structural
check: an equality where one is reachable, an assertion that the parameter was read, or
a deliberate corruption that shows the check can see the thing at all. Where none of the
three is available, measure the omission and commit the number, so the next reader knows
what a green run did not prove.

## 16. A PyTorch reduction over bf16 returns bf16 — narrow where the reference does

*Discovered as "`PixelNorm`'s mean-square is bf16, not fp32", which is the first and
clearest instance. The general rule is at the end of this entry; read it too.*

`ltx_core/model/common/normalization.py` — **one class, shared by everything**:

```python
mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
rms = torch.sqrt(mean_sq + self.eps)
return x / rms
```

`torch.mean` over a bf16 tensor **returns bf16**. It accumulates in fp32 internally and
then narrows, so the value the square root sees has been rounded. `self.eps` is a Python
float, which does not promote a tensor, so the root runs in bf16 too.

Computing the mean in fp32 and keeping it there is the more accurate arithmetic and the
**less faithful** one — the same trap as contract 1's RNG and the sampler schedule's
`Float`-not-`Double`. It is invisible to every shape check and every key check, and it
compounds: `PixelNorm` is applied 45 times on the video decoder's path (22 resnets x 2,
plus `conv_norm_out`) and 37 on the encoder's.

**Measured, and found twice independently.** Residual drift in two different modules
traced back to this, in the same direction both times — distance from the reference
implementation's own output, fp32 mean-square versus bf16-rounded:

| module | fp32 mean-square | bf16-rounded | improvement |
|---|---|---|---|
| video encoder, `out` (small) | 1.747e-02 | **7.861e-03** | 2.2x |
| audio encoder, `out` (small) | 2.670e-02 | **1.914e-02** | 1.4x |
| audio encoder, `out` (production) | 1.127e-02 | **1.084e-02** | 1.04x |

That is the whole finding: the defensible, more-precise spelling of the norm is the
one that disagrees with the reference, and it disagrees by more than twice as much.

**Confirmed on the decoders too, 2026-08-12.** The prediction above was written before
the measurement and then tested rather than assumed. Release build, production shape:

| decoder output | fp32 mean-square | bf16-rounded | |
|---|---|---|---|
| video, relative error | 1.8501e-03 | **1.3649e-03** | 1.36x closer |
| video, cos deficit | 1.697e-06 | **9.31e-07** | halved |
| video, max_abs | 0.03125 | **0.015625** | **exactly halved** |
| audio mel, relative error | 2.9904e-03 | **2.8883e-03** | 1.04x closer |

**The `max_abs` halving is the tell.** It did not shrink by some arbitrary fraction; it
moved down exactly one bf16 representable step, which is the signature of a single
rounding being the last remaining difference at the worst element. Both decoders had
been shipping the fp32 form, at a difference small enough that nothing complained —
precisely the headroom this contract warns will hide a defect of this size.

**It is not about `mean`, and it is not about `PixelNorm`. Generalise it.** Two more
instances landed the same day, in different operations and different modules:

- **`softmax` over bf16 also returns bf16.** The duration head's attention pooler
  narrows its weights before the value matmul. On the head's own positive-branch
  features: narrowed **4.9997 s**, not narrowed **4.9608 s**, against the reference
  module's **5.0 s**. Computing in fp32 and staying there was off by *more than the
  entire bf16 quantisation it was avoiding* — the "more accurate" arm is further from
  the reference than the rounding it declined to do.
- **The tiled decode's seam rounds twice.** The reference's streamed `tiled_decode`
  accumulates each tile group's masked product into its own **bf16** buffer before
  `previous_chunk += buffer`, so the second seam term is narrowed as well. Keeping our
  second seam term in fp32 puts the decoded video 6.5% further from the reference than
  the double-rounded form does, over 25 of 97 frames.

So the rule is **any reduction PyTorch performs on a bf16 tensor returns bf16**, and the
faithful port narrows wherever the reference does. Four modules and one seam have now
been found wrong this way. When porting anything that reduces — `mean`, `sum`, `softmax`,
an accumulation buffer — check the return dtype in the reference before choosing a
working precision, and treat "we compute it in fp32 for accuracy" as a claim requiring
evidence rather than a safe default.

**One caution on epsilon, which is not the same question.** This decoder constructs
`PixelNorm()` with no arguments, so `eps` is the class default `1e-8`; the *audio* VAE
reaches the same class through `build_normalization_layer`, which passes `1e-6`. Contract
notes in `VideoVAEDecoder` already record that carrying the audio value across is wrong.
The video encoder measured eps 1e-6 vs 1e-8 at **1.0003x** — non-discriminating, and
recorded as such rather than dressed up as a trap. The dtype of the mean is the thing that
moves; the epsilon is not.

---

## 17. AVAssetWriter deadlocks when two inputs are drained by polling

**Never drive more than one `AVAssetWriterInput` by polling `isReadyForMoreMediaData`
from a single thread.** Use `requestMediaDataWhenReady(on:using:)`, one serial queue per
input, and let the writer schedule the interleave. With `expectsMediaDataInRealTime =
false` the pull model is the contract AVFoundation actually offers; polling is a shape
that happens to work for one input and deadlocks for two.

This is in the fragile list rather than the commit log because of *how* it fails. It is a
race, not a defect with a threshold: the video input simply stops being reported ready and
never recovers, after an unpredictable number of frames. It shipped green through an
entire AV render that came back "clean and lip synced", then wedged a render that had
already cost nine minutes of sampling — and it wedged **after** every step, both VAE
decodes and the vocoder had completed, at the cheapest stage in the pipeline.

### What was measured

A standalone AVFoundation probe sharing no code with this port, driving the same
interleave against the same two-track shape:

| variant | result |
|---|---|
| one input (video only), polling | all 97 frames, **every run** |
| two inputs, polling | stalls at frame 34, 37, 39, 40, 47, 85 or 88 |
| two inputs, pull model | all 97 frames, at 288x512 **and** 640x384 |

The stall frame moves with chunk size and timing, which is the tell that it is a race.
Everything else is a red herring, and each was tested alone rather than assumed:

- **Not the shape.** 640x384 — the shape of the render that previously succeeded —
  wedges exactly like 288x512.
- **Not the codec.** H.264, HEVC, AAC and LPCM all wedge. An LPCM audio track, which the
  writer copies rather than encodes, wedges too, so it is not the AAC encoder.
- **Not IOSurface backing, and not the compression properties.** Both were adopted from
  a working single-track writer on the theory that the video format was wrong. Neither
  changes the outcome; that writer survives because it writes **one** track.
- **Not the interleave policy.** Feeding all the audio first reaches frame 88 and
  giving audio a fixed 0.5 s lead reaches 85 — further, and still wedged.

### The trap for whoever reads this next

The failure presents as a video-encoder problem — the error names a video frame index and
the video input is the one that stops — so the instinct is to go looking at pixel formats,
dimensions, profile levels and colour tags. The video track is the *victim*. The variable
is the input **count**, and no amount of correctness in the video settings fixes it.

Two consequences worth keeping:

- **A stall detector must watch progress, not elapse a budget.** A 20-second render at
  1152x2048 legitimately takes minutes to encode, so a fixed deadline either fails honest
  work or waits out a real wedge for just as long. `RenderOutput` now fails only when
  *neither* track has advanced for the timeout.
- **The mux runs last and its failure does not discard the render.** Frames are the
  expensive thing in the room: sampling them takes minutes, muxing them takes seconds. A
  mux failure now dumps the decoded video as a tensor beside the WAV and the sidecar
  records `mux_failed`, so a container problem costs a remux and not a re-render.

---

## 18. MSR's metadata says `prepend`. Its code appends.

The Multiple-Subject-Reference adapter carries
`reference_token_order = prepend` in its safetensors metadata, and the only
implementation of it — `liconstudio/ComfyUI-LTX2.5-MSR` — reads that key,
**validates that it says `prepend`**, raises if it says anything else, and then
calls a function that concatenates the guide onto the end:

```python
if metadata.get("reference_token_order", "prepend") != "prepend":
    raise ValueError("Unsupported reference_token_order; expected 'prepend'.")
...
latent_image = torch.cat([latent_image, guiding_latent], dim=2)   # append_keyframe
```

ComfyUI's transformer then reads the guide block off the **tail**:

```python
pixel_coords[:, :, -keyframe_idxs.shape[2]:, :] = keyframe_idxs
mask[:, -num_guide_tokens:] = False        # keyframes_abs_pos_mask
```

So the metadata is a label nobody acted on. Following it instead of the code
would move the references in front of the target, which silently moves
something else with them: `applyKeyframeEmbedding` marks the **first**
`tokensPerLatentFrame` tokens, so a prepend would stamp the keyframe marker on
a reference and leave the target's own frame 0 bare. Contract #15 has the
measurement of what that marker is worth, and of why misplacing it is invisible
to any numerical check at that boundary.

**This port appends**, and `DiTForward.applyKeyframeEmbedding` says why in
place.

## 19. MSR's slot offsets are **pixel frames**, applied before the fps divide

`frame_offset = -(num_slots - slot_index)` — three references give -3, -2, -1 —
and it lands here:

```python
pixel_coords[:, 0] += frame_idx            # add_keyframe_index, pixel frames
...
fractional_coords[:, 0] *= (1.0 / frame_rate)   # much later, in the model
```

Three consequences, each of which produces a well-formed grid when missed:

* **Pixel frames, not latent frames.** At 24 fps an offset of -3 is an eighth of
  a *latent* frame, not three of them. A port reading it as latent frames moves
  the references eight times too far.
* **Before the divide, not after.** `(p + f) / fps` and `p / fps + f / fps` are
  the same number in exact arithmetic and different ones in float32, on a
  quantity that feeds a cosine scaled by up to `10000 * pi/2` (contract in
  `DiTForward.frequencyGrid`). `referencePositions` builds its grid at frame
  rate 1, adds the offset, and divides last.
* **They must be allowed to go negative.** This is the entire mechanism: it is
  what stops two references sharing coordinates with each other and with the
  target's frame 0. `ltx_core`'s own `VideoConditionByReferenceLatent` clamps
  time at zero — correctly, for the temporally-sparse case it clamps in — and
  that clamp would erase every offset here. The two do not compose, and
  `referencePositions` has a precondition rather than a silent winner.

None of the three has a symptom of its own. The render completes and the cast
blends, which reads as a mediocre adapter.

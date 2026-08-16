#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sean Kammerich
"""Read a safetensors header and say what the checkpoint actually is.

Drop-day question: is LTX 2.5 the LTX 2.3 architecture with new flags, or a
different model? The answer is in the checkpoint, not in the ComfyUI diff --
`comfy/model_detection.py` ends its LTX branch with

    dit_config.update(json.loads(metadata["config"]).get("transformer", {}))

so the checkpoint carries its own transformer config and overrides everything
ComfyUI probed. A resized or re-proportioned model needs zero ComfyUI code
changes and is therefore invisible in that PR. This tool reads the config.

A safetensors file starts with an 8-byte little-endian header length followed
by that many bytes of JSON: every tensor name, dtype and shape, plus a
`__metadata__` object. It sits at the front of the file, so for a remote
checkpoint two ranged GETs answer the question without downloading 46 GB.

Stdlib only, on purpose. It has to run on any machine with no venv and no install
step, at a moment when nobody wants to debug dependencies.

    ltx-probe ~/models/ltx/2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
    ltx-probe Lightricks/LTX-2.5 --json probe-2.5.json
    ltx-probe probe-2.5.json --baseline probe-2.3.json    # what changed, and what it costs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from collections import defaultdict

HF_API = "https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"

# A safetensors header is JSON and can be large on a 500-shard model, but LTX
# ships few, big files. 64 MiB is far past anything real and still bounded.
MAX_HEADER = 64 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


# --------------------------------------------------------------------------
# Reading headers
# --------------------------------------------------------------------------

def _request(url, headers=None):
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ltx-probe/1.0")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and "huggingface.co" in url:
        req.add_header("Authorization", "Bearer %s" % token)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    return req


def _fetch_range(url, start, length):
    """Ranged GET. Falls back to slicing if the server ignores Range."""
    end = start + length - 1
    req = _request(url, {"Range": "bytes=%d-%d" % (start, end)})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        if resp.status == 206:
            return data
    # 200: whole body came back. Slice it and warn -- this is a bandwidth bug,
    # not a correctness bug, but on a 46 GB file it is the difference between
    # one second and an hour.
    sys.stderr.write("warning: server ignored Range on %s (got %d bytes)\n"
                     % (url, len(data)))
    return data[start:start + length]


def read_header_local(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("%s: too short to be safetensors" % path)
        n = struct.unpack("<Q", raw)[0]
        if n == 0 or n > MAX_HEADER:
            raise ValueError("%s: implausible header length %d -- not safetensors?" % (path, n))
        blob = f.read(n)
        size = os.path.getsize(path)
    return json.loads(blob), size


def read_header_remote(url):
    n = struct.unpack("<Q", _fetch_range(url, 0, 8))[0]
    if n == 0 or n > MAX_HEADER:
        raise ValueError("%s: implausible header length %d -- not safetensors?" % (url, n))
    blob = _fetch_range(url, 8, n)
    return json.loads(blob), None


def read_header(source):
    if source.startswith("http://") or source.startswith("https://"):
        return read_header_remote(source)
    return read_header_local(source)


def hf_list_safetensors(repo, rev="main"):
    url = HF_API.format(repo=repo, rev=rev)
    try:
        with urllib.request.urlopen(_request(url), timeout=60) as resp:
            tree = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(
                "error: %s is gated or private (HTTP %d). Set HF_TOKEN and retry." % (repo, e.code))
        if e.code == 404:
            raise SystemExit("error: no such HF repo/revision: %s@%s" % (repo, rev))
        raise
    out = []
    for entry in tree:
        path = entry.get("path", "")
        if path.endswith(".safetensors"):
            out.append((path, entry.get("size")))
    return sorted(out)


def resolve_sources(spec, rev="main"):
    """A spec is a local file, a local directory, a URL, or an HF repo id."""
    if spec.startswith("http://") or spec.startswith("https://"):
        return [(spec, spec, None)]
    if os.path.isfile(spec):
        return [(spec, spec, os.path.getsize(spec))]
    if os.path.isdir(spec):
        found = []
        for root, _dirs, files in os.walk(spec):
            for name in sorted(files):
                if name.endswith(".safetensors"):
                    p = os.path.join(root, name)
                    found.append((os.path.relpath(p, spec), p, os.path.getsize(p)))
        if not found:
            raise SystemExit("error: no .safetensors under %s" % spec)
        return found
    if re.match(r"^[\w.-]+/[\w.-]+$", spec):
        return [(path, HF_RESOLVE.format(repo=spec, rev=rev, path=path), size)
                for path, size in hf_list_safetensors(spec, rev)]
    raise SystemExit("error: cannot interpret %r as a file, directory, URL or HF repo id" % spec)


# --------------------------------------------------------------------------
# Component detection
#
# Keys arrive with whatever prefix the packager chose -- a bare component file
# has none, a ComfyUI combined checkpoint uses `model.diffusion_model.`. Rather
# than maintain a prefix list that will be wrong for the next release, every
# signature is matched as a suffix and the prefix is whatever came before it.
# --------------------------------------------------------------------------

# Order matters: the NA diffusion decoder also has a `decoder.conv_in`, so it
# must be tried before the causal reading, which is suppressed at a prefix
# where the NA marker already matched.
#
# Both spellings of every conv are listed. LTX wraps its convolutions in a
# causal-padding module, so the shipped key is `decoder.conv_in.conv.weight`,
# not the `decoder.conv_in.weight` that ComfyUI's `sd.py` tests -- ComfyUI is
# matching against an already-remapped state dict. Probing the file directly
# means matching what the file actually contains.
SIGNATURES = [
    # (component, signature suffix, note)
    ("ltx_dit", "adaln_single.emb.timestep_embedder.linear_1.bias", "LTX DiT"),
    ("vae_na_diffusion", "decoder.conv_in_x_t.conv.weight", "NA diffusion video VAE decoder (2.4/2.5)"),
    ("vae_na_diffusion", "decoder.conv_in_x_t.weight", "NA diffusion video VAE decoder (2.4/2.5)"),
    ("vae_decoder", "decoder.conv_in.conv.weight", "VAE decoder"),
    ("vae_decoder", "decoder.conv_in.weight", "VAE decoder"),
    ("vae_encoder", "encoder.conv_in.conv.weight", "VAE encoder"),
    ("vae_encoder", "encoder.conv_in.weight", "VAE encoder"),
    ("vocoder", "bwe_generator.conv_pre.weight", "vocoder (mel -> waveform)"),
    ("text_encoder", "model.layers.0.post_feedforward_layernorm.weight", "Gemma text encoder"),
    ("duration_head", "attention_pooler.query_tokens", "duration head"),
]


def find_prefixes(keys, suffix):
    """Every prefix under which `suffix` appears."""
    out = set()
    tail = "." + suffix
    for k in keys:
        if k == suffix:
            out.add("")
        elif k.endswith(tail):
            out.add(k[:-len(suffix)])
    return sorted(out)


def subkeys(tensors, prefix):
    if prefix == "":
        return dict(tensors)
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


def shape_of(sub, key):
    entry = sub.get(key)
    return list(entry["shape"]) if entry else None


def count_indexed(sub, pattern):
    """Highest N+1 for keys matching `pattern` with one {} placeholder."""
    rx = re.compile("^" + re.escape(pattern).replace(r"\{\}", r"(\d+)"))
    best = -1
    for k in sub:
        m = rx.match(k)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


# --------------------------------------------------------------------------
# Analyzers
# --------------------------------------------------------------------------

def analyze_dit(sub):
    """Mirror comfy/model_detection.py's LTX branch, then go further.

    ComfyUI hardcodes 32 attention heads (`shape[0] // 32`). That is reported
    as `comfy_assumed_*` and cross-checked against patchify_proj, because a
    head-count change is exactly the kind of thing that would make ComfyUI
    load a 2.5 checkpoint into a silently mis-shaped model.
    """
    r = {}
    is_av = "audio_adaln_single.linear.weight" in sub
    r["variant"] = "ltxav" if is_av else "ltxv"
    r["num_layers"] = count_indexed(sub, "transformer_blocks.{}.")

    patchify = shape_of(sub, "patchify_proj.weight")
    if patchify:
        r["inner_dim"], r["in_channels"] = patchify[0], patchify[1]
    proj_out = shape_of(sub, "proj_out.weight")
    if proj_out:
        r["out_channels"] = proj_out[0]

    attn2_k = shape_of(sub, "transformer_blocks.0.attn2.to_k.weight")
    if attn2_k:
        r["cross_attention_dim"] = attn2_k[1]
        r["comfy_assumed_head_dim"] = attn2_k[0] // 32
        r["comfy_assumed_num_heads"] = 32
        if r.get("inner_dim") and attn2_k[0] != r["inner_dim"]:
            r["WARN_attn2_inner_dim_mismatch"] = [attn2_k[0], r["inner_dim"]]

    # AdaLayerNormSingle.linear is Linear(dim, coefficient * dim); the
    # coefficient is 6 normally and 9 when cross_attention_adaln is on.
    adaln = shape_of(sub, "adaln_single.linear.weight")
    if adaln and adaln[1]:
        coef = adaln[0] / adaln[1]
        r["adaln_embedding_coefficient"] = int(coef) if coef == int(coef) else coef
        r["cross_attention_adaln"] = (r["adaln_embedding_coefficient"] == 9)

    sst = shape_of(sub, "transformer_blocks.0.scale_shift_table")
    if sst:
        r["block_ada_params"] = sst[0]

    # The 2.5 flags, each read off key presence exactly as ComfyUI does.
    r["ff_bias"] = "transformer_blocks.0.ff.net.0.proj.bias" in sub
    r["ff_out_bias"] = "transformer_blocks.0.ff.net.2.bias" in sub
    r["use_prompt_adaln_single"] = "prompt_adaln_single.linear.weight" in sub
    r["use_keyframes_abs_pos_embedding"] = "keyframes_abs_pos_embedding" in sub
    r["apply_gated_attention"] = "transformer_blocks.0.attn1.to_gate_logits.weight" in sub
    r["caption_projection_first_linear"] = "caption_projection.linear_1.weight" in sub

    if is_av:
        a_patchify = shape_of(sub, "audio_patchify_proj.weight")
        if a_patchify:
            r["audio_inner_dim"], r["audio_in_channels"] = a_patchify[0], a_patchify[1]
        a_attn2_k = shape_of(sub, "transformer_blocks.0.audio_to_video_attn.to_k.weight")
        if a_attn2_k:
            r["audio_cross_attention_dim"] = a_attn2_k[1]
        r["audio_ff_bias"] = "transformer_blocks.0.audio_ff.net.0.proj.bias" in sub
        r["audio_use_prompt_adaln_single"] = "audio_prompt_adaln_single.linear.weight" in sub
        r["audio_connector_layers"] = count_indexed(
            sub, "audio_embeddings_connector.transformer_1d_blocks.{}.")
        r["video_connector_layers"] = count_indexed(
            sub, "video_embeddings_connector.transformer_1d_blocks.{}.")
    return r


def _first_shape(sub, *keys):
    for k in keys:
        s = shape_of(sub, k)
        if s:
            return s
    return None


def _modality_from_conv(shape):
    """A 5D conv weight is 3D convolution (video); 4D is 2D convolution (audio)."""
    if not shape:
        return "unknown"
    return {5: "video (3D conv)", 4: "audio (2D conv)"}.get(len(shape), "rank-%d" % len(shape))


def analyze_vae_na(sub):
    r = {"decoder_kind": "na_diffusion"}
    conv_in = _first_shape(sub, "decoder.conv_in.conv.weight", "decoder.conv_in.weight")
    if conv_in:
        r["latent_channels"] = conv_in[1]
        r["base_channels"] = conv_in[0]
        r["modality"] = _modality_from_conv(conv_in)
    r["stages"] = count_indexed(sub, "decoder.stages.{}.")
    r["diffusion_blocks"] = count_indexed(sub, "decoder.blocks.{}.")
    r["has_t_embedder"] = any(k.startswith("decoder.t_embedder.") for k in sub)
    r["has_shared_adaln"] = any("shared_adaln." in k for k in sub)
    r["fused_qkv"] = any(k.endswith("attn.qkv.weight") for k in sub)
    return r


def analyze_vae_decoder(sub):
    r = {"decoder_kind": "causal"}
    conv_in = _first_shape(sub, "decoder.conv_in.conv.weight", "decoder.conv_in.weight")
    conv_out = _first_shape(sub, "decoder.conv_out.conv.weight", "decoder.conv_out.weight")
    if conv_in:
        r["latent_channels"] = conv_in[1]
        r["base_channels"] = conv_in[0]
        r["modality"] = _modality_from_conv(conv_in)
    if conv_out:
        # The video decoder emits patchified pixels, so out_channels is
        # 3 * patch^2 rather than 3; report it raw and let the reader divide.
        r["out_channels"] = conv_out[0]
    r["up_blocks"] = count_indexed(sub, "decoder.up_blocks.{}.")
    r["has_per_channel_statistics"] = any(k.startswith("per_channel_statistics.") for k in sub)
    return r


def analyze_vae_encoder(sub):
    r = {}
    conv_in = _first_shape(sub, "encoder.conv_in.conv.weight", "encoder.conv_in.weight")
    conv_out = _first_shape(sub, "encoder.conv_out.conv.weight", "encoder.conv_out.weight")
    if conv_in:
        r["in_channels"] = conv_in[1]
        r["base_channels"] = conv_in[0]
        r["modality"] = _modality_from_conv(conv_in)
    if conv_out:
        # `latent_log_var: uniform` means the encoder emits 2x latent channels.
        r["encoder_out_channels"] = conv_out[0]
    r["down_blocks"] = count_indexed(sub, "encoder.down_blocks.{}.")
    return r


def analyze_vocoder(sub):
    r = {}
    pre = shape_of(sub, "bwe_generator.conv_pre.weight")
    post = shape_of(sub, "bwe_generator.conv_post.weight")
    if pre:
        r["bwe_in_channels"] = pre[1]
    if post:
        r["bwe_out_channels"] = post[0]
    r["bwe_resblocks"] = count_indexed(sub, "bwe_generator.resblocks.{}.")
    r["vocoder_resblocks"] = count_indexed(sub, "vocoder.resblocks.{}.")
    r["has_mel_stft"] = any(k.startswith("mel_stft.") for k in sub)
    return r


def analyze_text_encoder(sub):
    """Gemma variant, using ComfyUI's own discriminators.

    Gemma-3 12B and Gemma-4 12B both have 48 layers; the Unified Gemma-4 is
    told apart by dropping v_proj on its global layers.
    """
    r = {}
    n = count_indexed(sub, "model.layers.{}.")
    r["num_hidden_layers"] = n
    emb = shape_of(sub, "model.embed_tokens.weight")
    if emb:
        r["vocab_size"], r["hidden_size"] = emb[0], emb[1]

    has = lambda k: k in sub
    if has("model.layers.59.self_attn.q_norm.weight"):
        r["variant"] = "gemma4_31b"
    elif has("model.layers.47.self_attn.q_norm.weight") and not has("model.layers.5.self_attn.v_proj.weight"):
        r["variant"] = "gemma4_12b_unified"
    elif has("model.layers.41.self_attn.q_norm.weight") and not has("model.layers.47.self_attn.q_norm.weight"):
        r["variant"] = "gemma4_e4b"
    elif has("model.layers.34.self_attn.q_norm.weight") and not has("model.layers.41.self_attn.q_norm.weight"):
        r["variant"] = "gemma4_e2b"
    elif has("model.layers.47.self_attn.q_norm.weight"):
        r["variant"] = "gemma3_12b"
    elif has("model.layers.0.self_attn.q_norm.weight"):
        r["variant"] = "gemma3_4b_vision" if has("vision_model.embeddings.patch_embedding.weight") else "gemma3_4b"
    else:
        r["variant"] = "gemma2_2b"

    if r.get("hidden_size"):
        # comfy/text_encoders/lt.py: the projection consumes every layer's
        # hidden state plus the embedding output.
        r["expected_projection_in_dim"] = r["hidden_size"] * (n + 1)
    return r


def analyze_text_projection(tensors):
    """The LTXAV text projection, wherever it lives (its own file or bundled)."""
    for prefix in find_prefixes(tensors, "text_embedding_projection.video_aggregate_embed.weight"):
        sub = subkeys(tensors, prefix)
        v = shape_of(sub, "text_embedding_projection.video_aggregate_embed.weight")
        a = shape_of(sub, "text_embedding_projection.audio_aggregate_embed.weight")
        if v and a:
            return prefix, {
                "text_projection_type": "dual_linear",
                "projection_in_dim": v[1],
                "video_projection_dim": v[0],
                "audio_projection_dim": a[0],
                "video_projection_bias": "text_embedding_projection.video_aggregate_embed.bias" in sub,
                "audio_projection_bias": "text_embedding_projection.audio_aggregate_embed.bias" in sub,
            }
    for suffix in ("text_embedding_projection.weight",
                   "text_embedding_projection.aggregate_embed.weight"):
        for prefix in find_prefixes(tensors, suffix):
            sub = subkeys(tensors, prefix)
            s = shape_of(sub, suffix)
            if s:
                return prefix, {
                    "text_projection_type": "single_linear",
                    "projection_in_dim": s[1],
                    "video_projection_dim": s[0],
                    "video_projection_bias": suffix[:-len("weight")] + "bias" in sub,
                }
    return None, None


ANALYZERS = {
    "ltx_dit": analyze_dit,
    "vae_na_diffusion": analyze_vae_na,
    "vae_decoder": analyze_vae_decoder,
    "vae_encoder": analyze_vae_encoder,
    "vocoder": analyze_vocoder,
    "text_encoder": analyze_text_encoder,
}


# --------------------------------------------------------------------------
# Self-check: probed shapes vs the embedded config
#
# The two can disagree, and a disagreement is the loudest signal this tool can
# produce. ComfyUI hardcodes 32 attention heads for LTX; if 2.5 ships a
# different head count, ComfyUI would load the checkpoint into a mis-shaped
# model and this is the only place that would say so before a render does.
# --------------------------------------------------------------------------

def config_crosscheck(report):
    cfg = (report.get("embedded_config") or {}).get("transformer")
    if not isinstance(cfg, dict):
        return None
    dit = next((c for name, c in report["components"].items()
                if name.startswith("ltx_dit@")), None)
    if not dit:
        return None

    def prod(*names):
        vals = [cfg.get(n) for n in names]
        if any(v is None for v in vals):
            return None
        out = 1
        for v in vals:
            out *= v
        return out

    checks = [
        ("num_layers", dit.get("num_layers"), cfg.get("num_layers")),
        ("inner_dim", dit.get("inner_dim"), prod("num_attention_heads", "attention_head_dim")),
        ("cross_attention_dim", dit.get("cross_attention_dim"), cfg.get("cross_attention_dim")),
        ("in_channels", dit.get("in_channels"), cfg.get("in_channels")),
        ("out_channels", dit.get("out_channels"), cfg.get("out_channels")),
        ("cross_attention_adaln", dit.get("cross_attention_adaln"), cfg.get("cross_attention_adaln")),
        ("apply_gated_attention", dit.get("apply_gated_attention"), cfg.get("apply_gated_attention")),
        ("caption_projection_first_linear", dit.get("caption_projection_first_linear"),
         cfg.get("caption_projection_first_linear")),
        ("audio_inner_dim", dit.get("audio_inner_dim"),
         prod("audio_num_attention_heads", "audio_attention_head_dim")),
        ("audio_cross_attention_dim", dit.get("audio_cross_attention_dim"),
         cfg.get("audio_cross_attention_dim")),
        ("audio_in_channels", dit.get("audio_in_channels"), cfg.get("audio_in_channels")),
        # The one ComfyUI cannot get wrong quietly anywhere else.
        ("num_attention_heads (ComfyUI hardcodes 32)",
         dit.get("comfy_assumed_num_heads"), cfg.get("num_attention_heads")),
        ("connector layers", dit.get("video_connector_layers"), cfg.get("connector_num_layers")),
    ]
    out = []
    for name, probed, declared in checks:
        if probed is None or declared is None:
            continue
        out.append((name, probed, declared, probed == declared))
    return out


def is_lora(tensors):
    return any(re.search(r"\.lora_(A|B|down|up)\b|lora_down\.weight|lora_up\.weight", k)
               for k in tensors)


def tensor_stats(tensors):
    params = 0
    dtypes = defaultdict(int)
    for name, entry in tensors.items():
        shape = entry.get("shape") or []
        n = 1
        for d in shape:
            n *= d
        params += n
        dtypes[entry.get("dtype", "?")] += n
    return {
        "num_tensors": len(tensors),
        "num_parameters": params,
        "parameters_human": human_count(params),
        "dtype_parameter_counts": dict(sorted(dtypes.items(), key=lambda kv: -kv[1])),
    }


def human_count(n):
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return "%.2f%s" % (n / div, unit)
    return str(n)


def human_bytes(n):
    if n is None:
        return "unknown"
    for unit, div in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return "%.2f %s" % (n / div, unit)
    return "%d B" % n


def probe_one(name, source, size):
    header, local_size = read_header(source)
    metadata = header.pop("__metadata__", {}) or {}
    tensors = {k: v for k, v in header.items() if isinstance(v, dict)}

    report = {
        "name": name,
        "source": source,
        "file_size": size if size is not None else local_size,
        "metadata": metadata,
        "stats": tensor_stats(tensors),
        "components": {},
    }

    # The authoritative config, if the packager shipped one. This is the field
    # ComfyUI lets override every probed value, so it outranks everything below.
    for key in ("config", "modelspec.config"):
        if key in metadata:
            try:
                report["embedded_config"] = json.loads(metadata[key])
            except (ValueError, TypeError):
                report["embedded_config_raw"] = metadata[key]
            break

    if is_lora(tensors):
        report["components"]["lora"] = {
            "rank_keys": sum(1 for k in tensors if "lora" in k.lower()),
        }

    seen_prefixes = set()
    for component, suffix, note in SIGNATURES:
        for prefix in find_prefixes(tensors, suffix):
            # Two spellings of the same signature must not produce two entries.
            if (prefix, component) in seen_prefixes:
                continue
            # The NA diffusion decoder also has a `decoder.conv_in`, so the
            # plain decoder reading is suppressed where the NA marker matched.
            if component == "vae_decoder" and (prefix, "vae_na_diffusion") in seen_prefixes:
                continue
            key = "%s@%s" % (component, prefix or "<root>")
            sub = subkeys(tensors, prefix)
            entry = {"prefix": prefix, "note": note, "num_tensors": len(sub)}
            analyzer = ANALYZERS.get(component)
            if analyzer:
                try:
                    entry.update(analyzer(sub))
                except Exception as exc:  # a malformed header must not lose the rest
                    entry["ERROR"] = "%s: %s" % (type(exc).__name__, exc)
            report["components"][key] = entry
            seen_prefixes.add((prefix, component))

    prefix, proj = analyze_text_projection(tensors)
    if proj:
        proj["prefix"] = prefix
        report["components"]["text_projection@%s" % (prefix or "<root>")] = proj

    if not report["components"]:
        report["components"]["unknown"] = {
            "note": "no known LTX signature matched",
            "sample_keys": sorted(tensors)[:20],
        }

    crosscheck = config_crosscheck(report)
    if crosscheck:
        report["config_crosscheck"] = [
            {"field": n, "probed": p, "declared": d, "agree": ok} for n, p, d, ok in crosscheck]
    return report


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

# What a change to each field costs the Swift port. This is the whole point of
# the tool: turn a key diff into a work estimate before anyone starts typing.
PORT_IMPACT = {
    "variant": "MODEL CLASS -- a different variant is a different pipeline",
    "num_layers": "cheap: config value, no code change",
    "inner_dim": "cheap: config value, but re-check every hardcoded dim in the Metal kernels",
    "in_channels": "VAE latent contract changed -- check patchify and latent geometry",
    "out_channels": "VAE latent contract changed -- check unpatchify",
    "cross_attention_dim": "text conditioning contract -- check the connector and projection",
    "comfy_assumed_head_dim": "cheap if heads stay 32; ComfyUI hardcodes 32 heads",
    "adaln_embedding_coefficient": "STRUCTURAL -- 6 vs 9 changes the AdaLN modulation layout",
    "cross_attention_adaln": "STRUCTURAL -- adds per-block cross-attn modulation",
    "block_ada_params": "STRUCTURAL -- scale_shift_table layout",
    "ff_bias": "small: add/remove bias in the video FeedForward",
    "ff_out_bias": "small: add/remove bias in the FeedForward output linear",
    "audio_ff_bias": "small: add/remove bias in the audio FeedForward",
    "use_prompt_adaln_single": "MODERATE -- prompt AdaLN gating path on/off",
    "audio_use_prompt_adaln_single": "MODERATE -- audio prompt AdaLN gating path on/off",
    "use_keyframes_abs_pos_embedding": "MODERATE -- new learned embedding + token mask logic",
    "apply_gated_attention": "MODERATE -- per-head attention gating",
    "caption_projection_first_linear": "small: which caption projection module is used",
    "audio_inner_dim": "cheap: config value on the audio branch",
    "audio_in_channels": "audio VAE latent contract changed",
    "audio_cross_attention_dim": "audio text conditioning contract",
    "audio_connector_layers": "cheap: config value",
    "video_connector_layers": "cheap: config value",
    "decoder_kind": "MAJOR -- causal vs NA diffusion decoder is a different decoder to write",
    "latent_channels": "VAE latent contract changed",
    "variant_te": "MAJOR -- a different Gemma family is a text-encoder rewrite",
    "num_hidden_layers": "cheap if the family is unchanged",
    "hidden_size": "text encoder dims -- check the projection input dim",
    "text_projection_type": "MODERATE -- single vs dual linear projection",
    "projection_in_dim": "follows hidden_size x (layers+1); recompute, do not hardcode",
    "video_projection_dim": "conditioning width for the video branch",
    "audio_projection_dim": "conditioning width for the audio branch",
    "video_projection_bias": "small: bias on the projection",
    "audio_projection_bias": "small: bias on the projection",
}


def print_report(report, show_keys=False):
    w = sys.stdout.write
    w("\n" + "=" * 78 + "\n")
    w("%s\n" % report["name"])
    w("=" * 78 + "\n")
    st = report["stats"]
    w("  size        %s\n" % human_bytes(report.get("file_size")))
    w("  tensors     %d\n" % st["num_tensors"])
    w("  parameters  %s (%d)\n" % (st["parameters_human"], st["num_parameters"]))
    w("  dtypes      %s\n" % ", ".join(
        "%s:%s" % (d, human_count(c)) for d, c in st["dtype_parameter_counts"].items()))

    if report.get("metadata"):
        keys = [k for k in report["metadata"] if k != "config"]
        if keys:
            w("  metadata    %s\n" % ", ".join(sorted(keys)))

    if "embedded_config" in report:
        w("\n  -- embedded config (AUTHORITATIVE; overrides everything ComfyUI probes) --\n")
        for line in json.dumps(report["embedded_config"], indent=2, sort_keys=True).splitlines():
            w("  %s\n" % line)
    elif "embedded_config_raw" in report:
        w("\n  -- embedded config (unparseable) --\n  %s\n" % report["embedded_config_raw"][:2000])
    else:
        w("\n  -- no embedded config; every value below is probed from key shapes --\n")

    for name, comp in sorted(report["components"].items()):
        w("\n  [%s]\n" % name)
        for k, v in comp.items():
            if k in ("prefix", "note"):
                continue
            flag = "  <-- " + PORT_IMPACT[k] if k.startswith("WARN") else ""
            w("      %-34s %s%s\n" % (k, v, flag))
        if comp.get("note"):
            w("      %-34s %s\n" % ("note", comp["note"]))

    cc = report.get("config_crosscheck")
    if cc:
        bad = [c for c in cc if not c["agree"]]
        w("\n  -- self-check: probed shapes vs embedded config --\n")
        for c in cc:
            w("      %-42s %-8s %s %s\n" % (
                c["field"], c["probed"], "==" if c["agree"] else "!=", c["declared"]))
        if bad:
            w("\n      MISMATCH in %d field(s). The config is authoritative; a\n" % len(bad))
            w("      disagreement means the probe's key-shape assumptions no longer\n")
            w("      hold for this checkpoint -- and that ComfyUI's own hardcoded\n")
            w("      values may be wrong for it too. Resolve before porting.\n")
        else:
            w("\n      all probed fields agree with the declared config.\n")

    if show_keys:
        w("\n  -- all keys --\n")
        for k in sorted(report.get("_keys", [])):
            w("      %s\n" % k)
    w("\n")


def flatten(report):
    """component field -> value, for diffing."""
    out = {}
    for name, comp in report.get("components", {}).items():
        base = name.split("@")[0]
        for k, v in comp.items():
            if k in ("prefix", "note", "num_tensors"):
                continue
            out["%s.%s" % (base, k)] = v
    cfg = report.get("embedded_config")
    if isinstance(cfg, dict):
        for section, body in cfg.items():
            if isinstance(body, dict):
                for k, v in body.items():
                    out["config.%s.%s" % (section, k)] = v
    return out


def print_diff(baseline, current):
    w = sys.stdout.write
    a, b = flatten(baseline), flatten(current)
    w("\n" + "=" * 78 + "\n")
    w("DELTA  %s  ->  %s\n" % (baseline.get("name", "baseline"), current.get("name", "current")))
    w("=" * 78 + "\n")

    a_stats, b_stats = baseline.get("stats", {}), current.get("stats", {})
    if a_stats and b_stats:
        pa, pb = a_stats["num_parameters"], b_stats["num_parameters"]
        ratio = (" (%.2fx)" % (pb / pa)) if pa else ""
        w("  parameters  %s -> %s%s\n" % (human_count(pa), human_count(pb), ratio))

    changed, added, removed = [], [], []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            removed.append((k, a[k]))
        elif k not in a:
            added.append((k, b[k]))
        elif a[k] != b[k]:
            changed.append((k, a[k], b[k]))

    def impact(field):
        return PORT_IMPACT.get(field.split(".")[-1], "review")

    if changed:
        w("\n  CHANGED\n")
        for k, av, bv in changed:
            w("    %-44s %s -> %s\n" % (k, av, bv))
            w("    %-44s %s\n" % ("", impact(k)))
    if added:
        w("\n  ADDED (present in %s only)\n" % current.get("name", "current"))
        for k, v in added:
            w("    %-44s %s\n" % (k, v))
            w("    %-44s %s\n" % ("", impact(k)))
    if removed:
        w("\n  REMOVED (present in %s only)\n" % baseline.get("name", "baseline"))
        for k, v in removed:
            w("    %-44s %s\n" % (k, v))
            w("    %-44s %s\n" % ("", impact(k)))
    if not (changed or added or removed):
        w("\n  no differences in probed fields\n")

    majors = [k for k, *_ in changed if "MAJOR" in impact(k) or "STRUCTURAL" in impact(k)]
    majors += [k for k, _ in added if "MAJOR" in impact(k) or "STRUCTURAL" in impact(k)]
    w("\n  VERDICT: ")
    if majors:
        w("STRUCTURAL change. These do not follow the 2.3 description and\n")
        w("           must be re-derived from the reference, not adapted:\n")
        for k in majors:
            w("           %s\n" % k)
    else:
        w("no structural change in probed fields -- config and flag\n")
        w("           deltas only. The 2.3 reference is a sound guide to what\n")
        w("           to write; read dims from the checkpoint, never declare them.\n")
    w("\n  Reminder: this compares probed fields only. A field absent from both\n")
    w("  sides cannot be diffed, and the embedded config outranks all of it.\n\n")


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ltx-probe",
        description="Read safetensors headers and report LTX architecture, without downloading weights.")
    p.add_argument("sources", nargs="+",
                   help="local file/dir, URL, HF repo id (owner/name), or a saved probe JSON")
    p.add_argument("--rev", default="main", help="HF revision (default: main)")
    p.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    p.add_argument("--baseline", metavar="PATH",
                   help="a probe JSON to diff against; prints a port-impact delta")
    p.add_argument("--keys", action="store_true", help="dump every tensor key")
    p.add_argument("--quiet", action="store_true", help="suppress the human report")
    args = p.parse_args(argv)

    reports = []
    for spec in args.sources:
        # A saved report re-reads as itself, so `--baseline` works offline.
        if spec.endswith(".json") and os.path.isfile(spec):
            with open(spec) as f:
                saved = json.load(f)
            reports.extend(saved if isinstance(saved, list) else [saved])
            continue
        for name, source, size in resolve_sources(spec, args.rev):
            try:
                report = probe_one(name, source, size)
            except Exception as exc:
                sys.stderr.write("error: %s: %s: %s\n" % (name, type(exc).__name__, exc))
                continue
            if args.keys:
                header, _ = read_header(source)
                report["_keys"] = [k for k in header if k != "__metadata__"]
            reports.append(report)

    if not reports:
        raise SystemExit("error: nothing probed")

    if not args.quiet:
        for r in reports:
            print_report(r, show_keys=args.keys)

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        if isinstance(baseline, list):
            baseline = baseline[0]
        for r in reports:
            print_diff(baseline, r)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports if len(reports) > 1 else reports[0], f, indent=2, sort_keys=True)
        sys.stderr.write("wrote %s\n" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

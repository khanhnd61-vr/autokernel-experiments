#!/usr/bin/env python3
"""
load_pretrained.py -- Load Qwen3-VL-8B-Instruct pretrained text-LLM weights
                       into our custom Qwen3VLModel (models/qwen3_vl.py).

Scope:
    TEXT-LLM ONLY. The pretrained vision tower uses a native-resolution ViT
    with 2D RoPE + spatial merge that is architecturally incompatible with
    our fixed-448 ViT, so only text weights are transferable.

Name mapping (HF → custom):
    model.embed_tokens.weight                        → tok_embeddings.weight
    model.layers.i.input_layernorm.weight            → layers.i.attention_norm.weight
    model.layers.i.self_attn.q_proj.weight           → layers.i.attention.wq.weight
    model.layers.i.self_attn.k_proj.weight           → layers.i.attention.wk.weight
    model.layers.i.self_attn.v_proj.weight           → layers.i.attention.wv.weight
    model.layers.i.self_attn.o_proj.weight           → layers.i.attention.wo.weight
    model.layers.i.self_attn.q_norm.weight           → layers.i.attention.q_norm.weight
    model.layers.i.self_attn.k_norm.weight           → layers.i.attention.k_norm.weight
    model.layers.i.post_attention_layernorm.weight   → layers.i.ffn_norm.weight
    model.layers.i.mlp.gate_proj.weight              → layers.i.feed_forward.w1.weight
    model.layers.i.mlp.up_proj.weight                → layers.i.feed_forward.w3.weight
    model.layers.i.mlp.down_proj.weight              → layers.i.feed_forward.w2.weight
    model.norm.weight                                → norm.weight
    lm_head.weight                                   → output.weight

Usage (standalone test):
    uv run load_pretrained.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


REFERENCE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def _hf_download_all_weights(model_id: str):
    """Download the safetensors shards + index for the given HF model id.
    Returns (list_of_shard_paths, param_to_shard dict)."""
    from huggingface_hub import hf_hub_download

    # Qwen3-VL uses sharded safetensors
    index_path = hf_hub_download(
        repo_id=model_id, filename="model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    param_to_shard = index["weight_map"]     # {param_name: shard_filename}
    unique_shards = sorted(set(param_to_shard.values()))
    shard_paths = [
        hf_hub_download(repo_id=model_id, filename=shard)
        for shard in unique_shards
    ]
    return shard_paths, param_to_shard


def _build_hf_state_dict(shard_paths):
    """Open every safetensors shard and merge into one flat state_dict (CPU)."""
    from safetensors import safe_open
    sd = {}
    for p in shard_paths:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    return sd


def _map_hf_text_key(hf_key: str) -> str | None:
    """Map an HF Qwen3-VL key into a key on Qwen3VLModel (text LLM only).
    Skips vision and unknown keys."""
    if hf_key.startswith("model.visual.") or hf_key.startswith("visual."):
        return None
    if hf_key == "lm_head.weight":
        return "output.weight"
    k = hf_key
    if k.startswith("model.language_model."):
        k = k[len("model.language_model."):]
    elif k.startswith("model."):
        k = k[len("model."):]
    if k == "embed_tokens.weight":
        return "tok_embeddings.weight"
    if k == "norm.weight":
        return "norm.weight"
    if k.startswith("layers."):
        rest = k.split(".", 2)[2]
        idx = k.split(".")[1]
        mapping = {
            "self_attn.q_proj.weight":         f"layers.{idx}.attention.wq.weight",
            "self_attn.k_proj.weight":         f"layers.{idx}.attention.wk.weight",
            "self_attn.v_proj.weight":         f"layers.{idx}.attention.wv.weight",
            "self_attn.o_proj.weight":         f"layers.{idx}.attention.wo.weight",
            "self_attn.q_norm.weight":         f"layers.{idx}.attention.q_norm.weight",
            "self_attn.k_norm.weight":         f"layers.{idx}.attention.k_norm.weight",
            "input_layernorm.weight":          f"layers.{idx}.attention_norm.weight",
            "post_attention_layernorm.weight": f"layers.{idx}.ffn_norm.weight",
            "mlp.gate_proj.weight":            f"layers.{idx}.feed_forward.w1.weight",
            "mlp.up_proj.weight":              f"layers.{idx}.feed_forward.w3.weight",
            "mlp.down_proj.weight":            f"layers.{idx}.feed_forward.w2.weight",
        }
        return mapping.get(rest)
    return None


def _map_hf_vision_key(hf_key: str) -> str | None:
    """Map an HF Qwen3-VL vision key (`model.visual.*`) into a key on
    Qwen3VLVisionTower. Names line up 1:1 with our port (intentional)."""
    if hf_key.startswith("model.visual."):
        k = hf_key[len("model.visual."):]
    elif hf_key.startswith("visual."):
        k = hf_key[len("visual."):]
    else:
        return None
    # Skip rotary_pos_emb.inv_freq — buffer, not in state_dict (persistent=False)
    if k == "rotary_pos_emb.inv_freq":
        return None
    return k


# Backwards-compat alias used by load_pretrained_text_llm()
def _map_hf_key_to_custom(hf_key: str) -> str | None:
    return _map_hf_text_key(hf_key)


def load_pretrained_text_llm(custom_model, model_id: str = REFERENCE_MODEL_ID,
                              verbose: bool = True) -> dict:
    """Load pretrained text-LLM weights from HF into a custom Qwen3VLModel.

    The custom_model must already be instantiated with n_layers=36.
    Returns a stats dict: {"loaded": N, "skipped_vision": N, "missing": [...], ...}.
    """
    if verbose:
        print(f"[load_pretrained] Downloading shards for {model_id}...")
    shard_paths, _ = _hf_download_all_weights(model_id)
    if verbose:
        print(f"[load_pretrained] Got {len(shard_paths)} shards; "
              f"loading into memory (CPU)...")
    hf_sd = _build_hf_state_dict(shard_paths)
    if verbose:
        print(f"[load_pretrained] Total HF tensors: {len(hf_sd)}")

    # Build our custom state_dict by remapping HF keys
    custom_sd = {}
    skipped_vision = 0
    skipped_other = []
    for hf_key, tensor in hf_sd.items():
        dest = _map_hf_key_to_custom(hf_key)
        if dest is None:
            if hf_key.startswith("model.visual.") or hf_key.startswith("visual."):
                skipped_vision += 1
            else:
                skipped_other.append(hf_key)
            continue
        custom_sd[dest] = tensor

    # Load into the module, keeping freqs_cis / vision / projector params intact
    expected_keys = set(custom_model.state_dict().keys())
    provided_keys = set(custom_sd.keys())
    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)

    # shape check a few representative tensors
    for key in ("tok_embeddings.weight", "output.weight", "norm.weight"):
        if key in custom_sd:
            got = tuple(custom_sd[key].shape)
            want = tuple(custom_model.state_dict()[key].shape)
            assert got == want, f"{key}: got {got} want {want}"

    # Move tensors to the model's device + dtype
    target_param = next(custom_model.parameters())
    device = target_param.device
    dtype = target_param.dtype
    for k, t in custom_sd.items():
        custom_sd[k] = t.to(device=device, dtype=dtype)

    custom_model.load_state_dict(custom_sd, strict=False)

    if verbose:
        print(f"[load_pretrained] Loaded {len(custom_sd)} text-LLM tensors")
        print(f"[load_pretrained] Skipped {skipped_vision} vision tensors "
              "(architectural incompatibility)")
        if skipped_other:
            print(f"[load_pretrained] Skipped {len(skipped_other)} other HF keys")
        if missing:
            print(f"[load_pretrained] Missing {len(missing)} custom keys:")
            for m in missing[:8]:
                print(f"    - {m}")
            if len(missing) > 8:
                print(f"    ... and {len(missing) - 8} more")

    return {
        "loaded": len(custom_sd),
        "skipped_vision": skipped_vision,
        "skipped_other": len(skipped_other),
        "missing": missing,
        "unexpected": unexpected,
    }


def load_pretrained_vision_tower(vision_tower, model_id: str = REFERENCE_MODEL_ID,
                                  verbose: bool = True) -> dict:
    """Load pretrained vision-tower weights from HF into a Qwen3VLVisionTower.

    The vision_tower must be a `Qwen3VLVisionTower` instance (already on its
    target device/dtype).  Returns a stats dict.
    """
    if verbose:
        print(f"[load_pretrained] Downloading shards for {model_id}...")
    shard_paths, _ = _hf_download_all_weights(model_id)
    if verbose:
        print(f"[load_pretrained] Got {len(shard_paths)} shards; "
              f"loading into memory (CPU)...")
    hf_sd = _build_hf_state_dict(shard_paths)

    custom_sd = {}
    for hf_key, tensor in hf_sd.items():
        dest = _map_hf_vision_key(hf_key)
        if dest is None:
            continue
        custom_sd[dest] = tensor

    expected_keys = set(vision_tower.state_dict().keys())
    provided_keys = set(custom_sd.keys())
    missing = sorted(expected_keys - provided_keys)
    unexpected = sorted(provided_keys - expected_keys)

    if verbose and unexpected:
        print(f"[load_pretrained] WARNING: {len(unexpected)} unexpected vision keys, e.g.:")
        for u in unexpected[:8]:
            print(f"    + {u}")

    target_param = next(vision_tower.parameters())
    device, dtype = target_param.device, target_param.dtype
    for k, t in custom_sd.items():
        custom_sd[k] = t.to(device=device, dtype=dtype)

    vision_tower.load_state_dict(custom_sd, strict=False)

    if verbose:
        print(f"[load_pretrained] Loaded {len(custom_sd)} vision tensors")
        if missing:
            print(f"[load_pretrained] Missing {len(missing)} vision keys:")
            for m in missing[:8]:
                print(f"    - {m}")
            if len(missing) > 8:
                print(f"    ... and {len(missing) - 8} more")

    return {
        "loaded": len(custom_sd),
        "missing": missing,
        "unexpected": unexpected,
    }


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #

def main():
    import importlib.util
    print("=" * 68)
    print("  Loading Qwen3-VL-8B-Instruct text LLM into custom kernel")
    print("=" * 68)

    # Import models/qwen3_vl.py and instantiate with n_layers=36
    spec = importlib.util.spec_from_file_location(
        "user_model",
        os.path.abspath(str(Path(__file__).parent / "models" / "qwen3_vl.py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("  Instantiating Qwen3VLModel(n_layers=36)...")
    model = mod.Qwen3VLModel(n_layers=36).to(dtype=torch.bfloat16).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f} B params")

    stats = load_pretrained_text_llm(model)
    print(f"\n  Stats: {stats['loaded']} loaded, "
          f"{stats['skipped_vision']} vision-skipped, "
          f"{len(stats['missing'])} missing")

    # Sanity: run a forward pass
    print("\n  Sanity forward with 16 random token IDs...")
    ids = torch.randint(0, 151936, (1, 16), device="cuda", dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    print(f"  Output shape: {tuple(logits.shape)}  "
          f"(expected: (1, 16, 151936))")
    print(f"  Logit range: [{logits.min().item():.2f}, "
          f"{logits.max().item():.2f}]")
    print("=" * 68)


if __name__ == "__main__":
    main()

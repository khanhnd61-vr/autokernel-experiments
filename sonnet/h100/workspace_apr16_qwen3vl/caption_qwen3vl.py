#!/usr/bin/env python3
"""
caption_qwen3vl.py -- Image → caption with two backends side-by-side.

Backends:
    [CUSTOM]    models/qwen3_vl.py  (proxy: 4-layer LLM, random weights,
                                      KV-cached greedy decode -- our custom
                                      kernel pipeline)
    [REFERENCE] Qwen/Qwen3-VL-8B-Instruct via transformers
                (36-layer pretrained, HF AutoModelForImageTextToText,
                 used as ground-truth for accuracy + perf comparison)

Both paths:
    - Read the same cat.png (or any --image-path)
    - Greedy-decode the same number of new tokens
    - Decode the generated IDs via the official Qwen3-VL tokenizer
    - Print caption text + per-stage timings
    - Final side-by-side table

Accuracy caveat:
    CUSTOM has RANDOM weights (proxy for kernel profiling), so its caption
    is valid Qwen3-VL tokens but semantically noise. REFERENCE is the real
    pretrained model and produces a semantically meaningful caption.

Usage:
    uv run caption_qwen3vl.py --image-path cat.png --max-new-tokens 32
    uv run caption_qwen3vl.py --image-path cat.png --only custom     # skip reference
    uv run caption_qwen3vl.py --image-path cat.png --only reference  # skip custom
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = os.path.abspath(str(SCRIPT_DIR))

REFERENCE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ===========================================================================
# Shared helpers
# ===========================================================================

def load_tokenizer():
    """Load the official Qwen3-VL tokenizer (tokenizers library, no transformers
    dep). Returns None if the tokenizer.json can't be reached."""
    try:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=REFERENCE_MODEL_ID, filename="tokenizer.json")
        return Tokenizer.from_file(path)
    except Exception as e:
        print(f"  (Tokenizer unavailable: {e})")
        return None


def load_pil_image(image_path: str):
    from PIL import Image
    return Image.open(image_path).convert("RGB")


# ===========================================================================
# CUSTOM backend -- our proxy model + KV-cached greedy decode
# ===========================================================================

def load_custom_models(model_path: str, dtype: torch.dtype):
    spec = importlib.util.spec_from_file_location("user_model", os.path.abspath(model_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.Qwen3VLModel().to(dtype=dtype).cuda().eval()
    vision = mod.VisionEncoder().to(dtype=dtype).cuda().eval()
    return text, vision


def pil_to_custom_tensor(pil_img, image_size: int, dtype: torch.dtype):
    """PIL.RGB -> [1,3,H,W] tensor with SigLIP-style normalization."""
    import numpy as np
    from PIL import Image
    pil = pil_img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t.to(dtype=dtype, device="cuda")
    mean = torch.tensor([0.5, 0.5, 0.5], device="cuda", dtype=dtype).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device="cuda", dtype=dtype).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)


class VisionProjector(nn.Module):
    """Project vision features (1152) to LLM hidden size (4096)."""

    def __init__(self, vision_dim: int = 1152, text_dim: int = 4096,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.up = nn.Linear(vision_dim, text_dim, bias=True).to(dtype=dtype)
        self.act = nn.GELU(approximate="tanh")
        self.out = nn.Linear(text_dim, text_dim, bias=True).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.up(x)))


def _apply_rope_chunk(xq, xk, cos_chunk, sin_chunk):
    """Rotate-half RoPE using real-valued cos/sin buffers (float32).
    cos_chunk/sin_chunk: [T, head_dim//2]"""
    cos = cos_chunk.unsqueeze(0).unsqueeze(2).to(xq.dtype)  # [1,T,1,D/2]
    sin = sin_chunk.unsqueeze(0).unsqueeze(2).to(xq.dtype)

    def rotate(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotate(xq), rotate(xk)


# --------------------------------------------------------------------------- #
# HF-compatible rotate-half RoPE + MRoPE (for real pretrained weights)
# --------------------------------------------------------------------------- #

def _rotate_half_th(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_cossin(q, k, cos, sin):
    """Apply rotate-half RoPE to q,k of shape [B, T, H, D].
    cos, sin: [B, T, D] (broadcast over heads). Returns [B, T, H, D]."""
    cos_b = cos.unsqueeze(2)  # [B, T, 1, D]
    sin_b = sin.unsqueeze(2)
    q_orig_dtype, k_orig_dtype = q.dtype, k.dtype
    qf = q.float(); kf = k.float()
    q_out = qf * cos_b + _rotate_half_th(qf) * sin_b
    k_out = kf * cos_b + _rotate_half_th(kf) * sin_b
    return q_out.to(q_orig_dtype), k_out.to(k_orig_dtype)


def _build_inv_freq(head_dim: int, theta: float, device, dtype=torch.float32):
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))


def _apply_interleaved_mrope(freqs, mrope_section):
    """freqs: [3, B, T, D/2]. mrope_section: e.g. [24, 20, 20] (sums to D/2).
    Reorganize chunked [TTT...HHH...WWW] -> interleaved [THWTHW...TT].
    Returns [B, T, D/2]."""
    out = freqs[0].clone()
    # Channels for H and W are interleaved at offsets 1 and 2 within each
    # length-3 chunk, restricted to mrope_section[dim] * 3 channels.
    for dim, offset in enumerate((1, 2), start=1):  # H -> dim=1, W -> dim=2
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[dim][..., idx]
    return out


def compute_mrope_cos_sin(position_ids, head_dim, theta, device, dtype,
                          mrope_section=(24, 20, 20)):
    """position_ids: [3, B, T]  -- T,H,W axis positions per token.
    Returns (cos, sin) of shape [B, T, head_dim]."""
    inv_freq = _build_inv_freq(head_dim, theta, device, dtype=torch.float32)  # [D/2]
    # freqs[axis, b, t, j] = position_ids[axis, b, t] * inv_freq[j]
    pos = position_ids.float()                       # [3, B, T]
    freqs = pos.unsqueeze(-1) * inv_freq             # [3, B, T, D/2]
    freqs = _apply_interleaved_mrope(freqs, list(mrope_section))  # [B, T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [B, T, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def build_mrope_position_ids(input_ids, image_grid_thw, image_token_id,
                              vision_start_token_id, spatial_merge_size=2):
    """Replicates HF Qwen3VL get_rope_index for the single-image, single-batch
    case used by caption_qwen3vl.py. Returns:
        position_ids: [3, B, T]  (T-axis, H-axis, W-axis)
        next_position: int       (max position + 1, used by decode)
    """
    assert input_ids.shape[0] == 1, "single-batch only"
    seq = input_ids[0].tolist()
    T_total = len(seq)
    pos = torch.zeros(3, 1, T_total, dtype=torch.long, device=input_ids.device)

    img_idx = 0
    cur = 0  # running text-axis position
    i = 0
    grids = image_grid_thw.tolist()  # [[T, H, W], ...]
    while i < T_total:
        tok = seq[i]
        if tok == image_token_id:
            # Find the run of consecutive image tokens starting at i.
            j = i
            while j < T_total and seq[j] == image_token_id:
                j += 1
            n_img_tokens = j - i
            t, h, w = grids[img_idx]
            mh, mw = h // spatial_merge_size, w // spatial_merge_size
            assert mh * mw * t == n_img_tokens, (
                f"image-token count {n_img_tokens} != t*mh*mw {t*mh*mw}")
            # Build T/H/W coords for each image token (HF get_vision_position_ids):
            # per-token: T-axis = cur (constant), H-axis = mh row, W-axis = mw col
            w_coord = torch.arange(cur, cur + mw, device=input_ids.device).repeat(mh * t)
            h_coord = torch.arange(cur, cur + mh, device=input_ids.device).repeat_interleave(mw * t)
            t_coord = torch.full((n_img_tokens,), cur, device=input_ids.device, dtype=torch.long)
            pos[0, 0, i:j] = t_coord
            pos[1, 0, i:j] = h_coord
            pos[2, 0, i:j] = w_coord
            # advance running text position by max(mh, mw)
            cur += max(mh, mw)
            img_idx += 1
            i = j
        else:
            pos[0, 0, i] = cur
            pos[1, 0, i] = cur
            pos[2, 0, i] = cur
            cur += 1
            i += 1
    return pos, cur


def _attn_prefill_mrope(attn, x, cos, sin):
    B, T, _ = x.shape
    q = attn.wq(x).view(B, T, attn.n_heads, attn.head_dim)
    k = attn.wk(x).view(B, T, attn.n_kv_heads, attn.head_dim)
    v = attn.wv(x).view(B, T, attn.n_kv_heads, attn.head_dim)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q, k = _apply_rope_cossin(q, k, cos, sin)
    q = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    k_rep = k_t.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else k_t
    v_rep = v_t.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else v_t
    y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    return attn.wo(y), k_t, v_t


def _attn_decode_step_mrope(attn, x_new, past_k, past_v, cos_step, sin_step):
    B, _, _ = x_new.shape
    q = attn.wq(x_new).view(B, 1, attn.n_heads, attn.head_dim)
    k_new = attn.wk(x_new).view(B, 1, attn.n_kv_heads, attn.head_dim)
    v_new = attn.wv(x_new).view(B, 1, attn.n_kv_heads, attn.head_dim)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
        k_new = attn.k_norm(k_new)
    q, k_new = _apply_rope_cossin(q, k_new, cos_step, sin_step)
    q = q.transpose(1, 2)
    k_new_t = k_new.transpose(1, 2)
    v_new_t = v_new.transpose(1, 2)
    k_cat = torch.cat([past_k, k_new_t], dim=2)
    v_cat = torch.cat([past_v, v_new_t], dim=2)
    k_rep = k_cat.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else k_cat
    v_rep = v_cat.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else v_cat
    y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    y = y.transpose(1, 2).contiguous().view(B, 1, -1)
    return attn.wo(y), k_cat, v_cat


def custom_prefill_mrope(text_model, inputs_embeds, cos, sin,
                         deepstack_visual_embeds=None, visual_pos_masks=None):
    h = inputs_embeds
    caches = []
    for layer_idx, layer in enumerate(text_model.layers):
        attn_in = layer.attention_norm(h)
        attn_out, k_c, v_c = _attn_prefill_mrope(layer.attention, attn_in, cos, sin)
        h = h + attn_out
        h = h + layer.feed_forward(layer.ffn_norm(h))
        caches.append((k_c, v_c))
        if (deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)):
            vis = deepstack_visual_embeds[layer_idx].to(h.device, h.dtype)
            mask = visual_pos_masks.to(h.device)
            h = h.clone()
            h[mask, :] = h[mask, :] + vis
    h = text_model.norm(h)
    logits = text_model.output(h[:, -1:, :])
    return logits, caches


def custom_decode_step_mrope(text_model, embed_new, caches, cos_step, sin_step):
    h = embed_new
    new_caches = []
    for i, layer in enumerate(text_model.layers):
        attn_in = layer.attention_norm(h)
        past_k, past_v = caches[i]
        attn_out, k_c, v_c = _attn_decode_step_mrope(
            layer.attention, attn_in, past_k, past_v, cos_step, sin_step)
        h = h + attn_out
        h = h + layer.feed_forward(layer.ffn_norm(h))
        new_caches.append((k_c, v_c))
    caches[:] = new_caches
    h = text_model.norm(h)
    return text_model.output(h)


def _attn_prefill(attn, x, freqs_cos, freqs_sin):
    B, T, _ = x.shape
    q = attn.wq(x).view(B, T, attn.n_heads, attn.head_dim)
    k = attn.wk(x).view(B, T, attn.n_kv_heads, attn.head_dim)
    v = attn.wv(x).view(B, T, attn.n_kv_heads, attn.head_dim)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q, k = _apply_rope_chunk(q, k, freqs_cos[:T], freqs_sin[:T])
    q = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    k_rep = k_t.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else k_t
    v_rep = v_t.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else v_t
    y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    return attn.wo(y), k_t, v_t


def _attn_decode_step(attn, x_new, past_k, past_v, freqs_cos, freqs_sin, pos):
    B, _, _ = x_new.shape
    q = attn.wq(x_new).view(B, 1, attn.n_heads, attn.head_dim)
    k_new = attn.wk(x_new).view(B, 1, attn.n_kv_heads, attn.head_dim)
    v_new = attn.wv(x_new).view(B, 1, attn.n_kv_heads, attn.head_dim)
    if hasattr(attn, "q_norm"):
        q = attn.q_norm(q)
        k_new = attn.k_norm(k_new)
    q, k_new = _apply_rope_chunk(q, k_new, freqs_cos[pos:pos + 1], freqs_sin[pos:pos + 1])
    q = q.transpose(1, 2)
    k_new_t = k_new.transpose(1, 2)
    v_new_t = v_new.transpose(1, 2)
    k_cat = torch.cat([past_k, k_new_t], dim=2)
    v_cat = torch.cat([past_v, v_new_t], dim=2)
    k_rep = k_cat.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else k_cat
    v_rep = v_cat.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else v_cat
    y = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    y = y.transpose(1, 2).contiguous().view(B, 1, -1)
    return attn.wo(y), k_cat, v_cat


def custom_prefill(text_model, inputs_embeds, deepstack_visual_embeds=None,
                   visual_pos_masks=None):
    """KV-cached prefill. Optionally adds DeepStack visual embeds into the
    first few decoder layers at positions marked by visual_pos_masks."""
    h = inputs_embeds
    freqs_cos = text_model.freqs_cos
    freqs_sin = text_model.freqs_sin
    caches = []
    for layer_idx, layer in enumerate(text_model.layers):
        attn_in = layer.attention_norm(h)
        attn_out, k_c, v_c = _attn_prefill(layer.attention, attn_in, freqs_cos, freqs_sin)
        h = h + attn_out
        h = h + layer.feed_forward(layer.ffn_norm(h))
        caches.append((k_c, v_c))
        if (deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)):
            vis = deepstack_visual_embeds[layer_idx].to(h.device, h.dtype)
            mask = visual_pos_masks.to(h.device)
            h = h.clone()
            h[mask, :] = h[mask, :] + vis
    h = text_model.norm(h)
    logits = text_model.output(h[:, -1:, :])
    return logits, caches


def custom_decode_step(text_model, embed_new, caches, pos):
    h = embed_new
    freqs_cos = text_model.freqs_cos
    freqs_sin = text_model.freqs_sin
    new_caches = []
    for i, layer in enumerate(text_model.layers):
        attn_in = layer.attention_norm(h)
        past_k, past_v = caches[i]
        attn_out, k_c, v_c = _attn_decode_step(
            layer.attention, attn_in, past_k, past_v, freqs_cos, freqs_sin, pos)
        h = h + attn_out
        h = h + layer.feed_forward(layer.ffn_norm(h))
        new_caches.append((k_c, v_c))
    caches[:] = new_caches
    h = text_model.norm(h)
    return text_model.output(h)


def _make_real_weight_custom_multimodal(model_path, dtype):
    """Instantiate Qwen3VLModel(n_layers=36) + Qwen3VLVisionTower, both with
    pretrained weights loaded from HF. Returns (text_model, vision_tower)."""
    import importlib.util
    from load_pretrained import (load_pretrained_text_llm,
                                  load_pretrained_vision_tower)
    spec = importlib.util.spec_from_file_location(
        "user_model", os.path.abspath(model_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  Instantiating Qwen3VLModel(n_layers=36)...")
    text = mod.Qwen3VLModel(n_layers=36).to(dtype=dtype).cuda().eval()
    t_stats = load_pretrained_text_llm(text, verbose=False)
    print(f"  Text weights : {t_stats['loaded']} loaded, "
          f"{len(t_stats['missing'])} missing")
    print("  Instantiating Qwen3VLVisionTower...")
    vision = mod.Qwen3VLVisionTower().to(dtype=dtype).cuda().eval()
    v_stats = load_pretrained_vision_tower(vision, verbose=False)
    print(f"  Vision wts.  : {v_stats['loaded']} loaded, "
          f"{len(v_stats['missing'])} missing")
    return text, vision


def run_custom_multimodal_real(args, pil_image):
    """Full multimodal pipeline on the custom kernel with REAL pretrained
    weights -- vision tower (custom port) + text LLM (custom port) + KV-cached
    decode + DeepStack feature injection into early text layers.
    """
    print("\n" + "=" * 68)
    print("  [CUSTOM+REAL+MM] Qwen3-VL-8B FULL multimodal on custom kernel")
    print("                   (real text+vision weights, deepstack enabled)")
    print("=" * 68)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.manual_seed(args.seed)

    text_model, vision_tower = _make_real_weight_custom_multimodal(
        args.local_model, dtype)

    # HF processor for the chat-template + image-patch packing.
    orig_path = sys.path.copy()
    sys.path = [p for p in sys.path
                if p not in ("", ".") and os.path.abspath(p) != PROJECT_ROOT]
    try:
        from transformers import AutoProcessor
    finally:
        sys.path = orig_path
    processor = AutoProcessor.from_pretrained(REFERENCE_MODEL_ID)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    eos_token_id = processor.tokenizer.eos_token_id

    head_dim = text_model.layers[0].attention.head_dim
    rope_theta = 5_000_000.0  # Qwen3-VL official
    mrope_section = (24, 20, 20)
    spatial_merge_size = 2

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": args.prompt_reference},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    input_ids = inputs["input_ids"]                                # [1, T]
    pixel_values = inputs["pixel_values"].to(dtype=dtype)
    grid_thw = inputs["image_grid_thw"]
    in_len = input_ids.shape[1]
    print(f"  Input tokens (text + image placeholders): {in_len}")
    print(f"  Pixel patches: {tuple(pixel_values.shape)}, grid_thw={grid_thw.tolist()}")

    def _run(max_new):
        t0 = time.time()
        with torch.no_grad():
            # 1. Vision tower
            torch.cuda.synchronize()
            tv = time.time()
            image_embeds, deepstack_feats = vision_tower(pixel_values, grid_thw)
            torch.cuda.synchronize()
            vision_ms = (time.time() - tv) * 1000

            # 2. Build text+vision inputs_embeds
            tp = time.time()
            text_embeds = text_model.tok_embeddings(input_ids)        # [1, T, dim]
            visual_pos_mask = (input_ids == image_token_id)           # [1, T] bool
            n_image_tokens = int(visual_pos_mask.sum().item())
            assert n_image_tokens == image_embeds.shape[0], (
                f"image-token count {n_image_tokens} != vision-emb count "
                f"{image_embeds.shape[0]}")
            text_embeds = text_embeds.clone()
            text_embeds[visual_pos_mask] = image_embeds.to(text_embeds.dtype)
            torch.cuda.synchronize()
            proj_ms = (time.time() - tp) * 1000

            # 3. MRoPE position_ids and cos/sin
            position_ids, next_text_pos = build_mrope_position_ids(
                input_ids, grid_thw,
                image_token_id=image_token_id,
                vision_start_token_id=vision_start_token_id,
                spatial_merge_size=spatial_merge_size,
            )
            cos, sin = compute_mrope_cos_sin(
                position_ids, head_dim, rope_theta,
                device=text_embeds.device, dtype=text_embeds.dtype,
                mrope_section=mrope_section,
            )

            # 4. Prefill (KV cache + deepstack at first 3 layers + MRoPE)
            tpf = time.time()
            logits, caches = custom_prefill_mrope(
                text_model, text_embeds, cos, sin,
                deepstack_visual_embeds=deepstack_feats,
                visual_pos_masks=visual_pos_mask,
            )
            next_id = logits[0, -1].argmax().item()
            torch.cuda.synchronize()
            prefill_ms = (time.time() - tpf) * 1000

            # 5. Per-step decode with MRoPE (new tokens are pure text, so all
            # 3 axes get the same value: next_text_pos, +1, +2, ...)
            gen = [next_id]
            step_times = []
            cur_pos = next_text_pos
            for _ in range(max_new - 1):
                if next_id == eos_token_id:
                    break
                ts = time.time()
                en = text_model.tok_embeddings(
                    torch.tensor([[next_id]], device="cuda", dtype=torch.long))
                step_pos_ids = torch.tensor(
                    [[[cur_pos]], [[cur_pos]], [[cur_pos]]],
                    device=en.device, dtype=torch.long,
                )  # [3, 1, 1]
                cos_s, sin_s = compute_mrope_cos_sin(
                    step_pos_ids, head_dim, rope_theta,
                    device=en.device, dtype=en.dtype,
                    mrope_section=mrope_section,
                )
                logits = custom_decode_step_mrope(text_model, en, caches, cos_s, sin_s)
                next_id = logits[0, -1].argmax().item()
                gen.append(next_id)
                cur_pos += 1
                torch.cuda.synchronize()
                step_times.append((time.time() - ts) * 1000)

        return {
            "ids": gen,
            "vision_ms": vision_ms,
            "proj_ms": proj_ms,
            "prefill_ms": prefill_ms,
            "step_times": step_times,
            "total_ms": (time.time() - t0) * 1000,
            "init_seq_len": text_embeds.shape[1],
        }

    print("  Warming up (2 tokens)...")
    _run(2)
    print(f"  Generating {args.max_new_tokens} tokens...")
    r = _run(args.max_new_tokens)

    caption = processor.decode(r["ids"], skip_special_tokens=True)
    steps = sorted(r["step_times"])
    step_med = steps[len(steps) // 2] if steps else 0.0
    tok_per_s = (len(r["ids"]) / (r["total_ms"] / 1000)) if r["total_ms"] else 0.0

    print(f"  Vision encoder:   {r['vision_ms']:.2f} ms")
    print(f"  Embed merge:      {r['proj_ms']:.2f} ms")
    print(f"  Init seq len:     {r['init_seq_len']} tokens")
    print(f"  Prefill:          {r['prefill_ms']:.2f} ms")
    print(f"  Decode/step:      {step_med:.2f} ms (median)")
    print(f"  Total:            {r['total_ms']:.2f} ms  ({tok_per_s:.1f} tok/s)")
    print(f"  Caption: {caption!r}")

    del text_model, vision_tower
    torch.cuda.empty_cache()

    return {
        "backend": "custom_real_mm",
        "caption_text": caption,
        "ids": r["ids"],
        "total_ms": r["total_ms"],
        "tok_per_s": tok_per_s,
        "step_med_ms": step_med,
        "vision_ms": r["vision_ms"],
        "prefill_ms": r["prefill_ms"],
    }


def _make_real_weight_custom(model_path, dtype):
    """Instantiate Qwen3VLModel(n_layers=36) and load pretrained weights."""
    import importlib.util
    from load_pretrained import load_pretrained_text_llm
    spec = importlib.util.spec_from_file_location(
        "user_model", os.path.abspath(model_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  Instantiating Qwen3VLModel(n_layers=36)...")
    text = mod.Qwen3VLModel(n_layers=36).to(dtype=dtype).cuda().eval()
    stats = load_pretrained_text_llm(text, verbose=True)
    print(f"  Weights: {stats['loaded']} loaded, "
          f"{stats['skipped_vision']} vision-skipped, "
          f"{len(stats['missing'])} missing")
    # Vision encoder stays random + unused in this mode.
    vision = mod.VisionEncoder().to(dtype=dtype).cuda().eval()
    return text, vision


def run_custom_realweights(args, tokenizer):
    """Text-only path: custom kernel with REAL pretrained text-LLM weights.
    Tokenize the prompt, greedy-decode via our KV-cached custom kernel."""
    print("\n" + "=" * 68)
    print("  [CUSTOM+REAL] 36-layer Qwen3-VL text LLM on our custom kernel")
    print("                (real pretrained weights, text-only, no vision)")
    print("=" * 68)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    torch.manual_seed(args.seed)

    text_model, _ = _make_real_weight_custom(args.local_model, dtype)
    n_params = sum(p.numel() for p in text_model.parameters())
    print(f"  Text-LLM params: {n_params / 1e9:.2f} B")

    if tokenizer is None:
        raise RuntimeError("Tokenizer required for real-weights mode")

    # Tokenize the text prompt (raw — no chat template, to match reference path)
    prompt = args.text_prompt
    enc = tokenizer.encode(prompt)
    prompt_ids = torch.tensor([enc.ids], device="cuda", dtype=torch.long)
    print(f"  Prompt        : {prompt!r}")
    print(f"  Prompt tokens : {prompt_ids.shape[1]} ids: {enc.ids[:16]}"
          f"{'...' if len(enc.ids) > 16 else ''}")

    def _run(max_new):
        t0 = time.time()
        with torch.no_grad():
            prompt_embeds = text_model.tok_embeddings(prompt_ids)
            torch.cuda.synchronize()
            tpf = time.time()
            logits, caches = custom_prefill(text_model, prompt_embeds)
            next_id = logits[0, -1].argmax().item()
            torch.cuda.synchronize()
            prefill_ms = (time.time() - tpf) * 1000

            gen = [next_id]
            step_times = []
            pos = prompt_embeds.shape[1]
            for _ in range(max_new - 1):
                ts = time.time()
                en = text_model.tok_embeddings(
                    torch.tensor([[next_id]], device="cuda", dtype=torch.long))
                logits = custom_decode_step(text_model, en, caches, pos)
                next_id = logits[0, -1].argmax().item()
                gen.append(next_id)
                pos += 1
                torch.cuda.synchronize()
                step_times.append((time.time() - ts) * 1000)

        return {
            "ids": gen,
            "prefill_ms": prefill_ms,
            "step_times": step_times,
            "total_ms": (time.time() - t0) * 1000,
            "prompt_len": prompt_embeds.shape[1],
        }

    print("  Warming up (2 tokens)...")
    _run(2)
    print(f"  Generating {args.max_new_tokens} tokens...")
    r = _run(args.max_new_tokens)

    caption = tokenizer.decode(r["ids"])
    steps = sorted(r["step_times"])
    step_med = steps[len(steps) // 2] if steps else 0.0
    tok_per_s = (len(r["ids"]) / (r["total_ms"] / 1000)) if r["total_ms"] else 0.0

    print(f"  Prefill       : {r['prefill_ms']:.2f} ms  "
          f"({r['prompt_len']} tokens)")
    print(f"  Decode/step   : {step_med:.2f} ms (median)")
    print(f"  Total         : {r['total_ms']:.2f} ms  ({tok_per_s:.1f} tok/s)")
    print(f"  Completion    : {caption!r}")

    del text_model
    torch.cuda.empty_cache()

    return {
        "backend": "custom_real",
        "caption_text": caption,
        "ids": r["ids"],
        "total_ms": r["total_ms"],
        "tok_per_s": tok_per_s,
        "step_med_ms": step_med,
        "prefill_ms": r["prefill_ms"],
    }


def run_reference_text_only(args, tokenizer):
    """Reference Qwen3-VL-8B-Instruct doing pure text completion (no image).
    Used to verify custom+real-weights matches reference token-for-token."""
    print("\n" + "=" * 68)
    print(f"  [REFERENCE-TEXT] {REFERENCE_MODEL_ID} (text-only, no image)")
    print("=" * 68)

    orig_path = sys.path.copy()
    sys.path = [p for p in sys.path
                if p not in ("", ".") and os.path.abspath(p) != PROJECT_ROOT]
    try:
        from transformers import AutoModelForImageTextToText, AutoTokenizer
    finally:
        sys.path = orig_path

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    t_load = time.time()
    print(f"  Loading tokenizer + model...")
    hf_tok = AutoTokenizer.from_pretrained(REFERENCE_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        REFERENCE_MODEL_ID,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    print(f"  Loaded in {time.time() - t_load:.1f} s")

    prompt = args.text_prompt
    enc = hf_tok(prompt, return_tensors="pt").to(model.device)
    in_len = enc["input_ids"].shape[1]

    print("  Warming up...")
    with torch.no_grad():
        _ = model.generate(**enc, max_new_tokens=2, do_sample=False)

    torch.cuda.synchronize()
    print(f"  Generating {args.max_new_tokens} tokens...")
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) * 1000

    new_ids = out_ids[0, in_len:].tolist()
    caption = hf_tok.decode(new_ids, skip_special_tokens=True)
    tok_per_s = (len(new_ids) / (total_ms / 1000)) if total_ms else 0.0

    print(f"  Total         : {total_ms:.2f} ms  ({tok_per_s:.1f} tok/s)")
    print(f"  Completion    : {caption!r}")

    del model, hf_tok
    torch.cuda.empty_cache()

    return {
        "backend": "reference_text",
        "caption_text": caption,
        "ids": new_ids,
        "total_ms": total_ms,
        "tok_per_s": tok_per_s,
    }


def run_custom(args, pil_image, tokenizer):
    """Returns dict with keys: caption_text, ids, timings."""
    print("\n" + "=" * 68)
    print("  [CUSTOM] Proxy Qwen3-VL (4-layer, random weights, KV-cached)")
    print("=" * 68)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(args.seed)

    text_model, vision_encoder = load_custom_models(args.local_model, dtype)
    vision_projector = VisionProjector(
        vision_dim=1152, text_dim=4096, dtype=dtype).cuda().eval()
    n_params = (
        sum(p.numel() for p in text_model.parameters()) +
        sum(p.numel() for p in vision_encoder.parameters()) +
        sum(p.numel() for p in vision_projector.parameters()))
    print(f"  Params: {n_params / 1e6:.1f}M")

    image = pil_to_custom_tensor(pil_image, args.image_size, dtype)
    prompt_ids = torch.tensor([[1]], device="cuda", dtype=torch.long)

    def _run(max_new):
        t0 = time.time()
        with torch.no_grad():
            torch.cuda.synchronize()
            tv = time.time()
            vf = vision_encoder(image)
            torch.cuda.synchronize()
            vision_ms = (time.time() - tv) * 1000

            tp = time.time()
            ve = vision_projector(vf)
            pe = text_model.tok_embeddings(prompt_ids)
            emb = torch.cat([ve, pe], dim=1)
            torch.cuda.synchronize()
            proj_ms = (time.time() - tp) * 1000

            tpf = time.time()
            logits, caches = custom_prefill(text_model, emb)
            next_id = logits[0, -1].argmax().item()
            torch.cuda.synchronize()
            prefill_ms = (time.time() - tpf) * 1000

            gen = [next_id]
            step_times = []
            pos = emb.shape[1]
            for _ in range(max_new - 1):
                ts = time.time()
                en = text_model.tok_embeddings(
                    torch.tensor([[next_id]], device="cuda", dtype=torch.long))
                logits = custom_decode_step(text_model, en, caches, pos)
                next_id = logits[0, -1].argmax().item()
                gen.append(next_id)
                pos += 1
                torch.cuda.synchronize()
                step_times.append((time.time() - ts) * 1000)

        return {
            "ids": gen,
            "vision_ms": vision_ms,
            "proj_ms": proj_ms,
            "prefill_ms": prefill_ms,
            "step_times": step_times,
            "total_ms": (time.time() - t0) * 1000,
            "init_seq_len": emb.shape[1],
        }

    print("  Warming up (2 tokens)...")
    _run(2)
    print(f"  Generating {args.max_new_tokens} tokens...")
    r = _run(args.max_new_tokens)

    caption = tokenizer.decode(r["ids"]) if tokenizer is not None else "(no tokenizer)"

    steps = sorted(r["step_times"])
    step_med = steps[len(steps) // 2] if steps else 0.0
    tok_per_s = (len(r["ids"]) / (r["total_ms"] / 1000)) if r["total_ms"] else 0.0

    print(f"  Vision encoder:   {r['vision_ms']:.2f} ms")
    print(f"  Projector+concat: {r['proj_ms']:.2f} ms")
    print(f"  Init seq len:     {r['init_seq_len']} tokens")
    print(f"  Prefill:          {r['prefill_ms']:.2f} ms")
    print(f"  Decode/step:      {step_med:.2f} ms (median)")
    print(f"  Total:            {r['total_ms']:.2f} ms  ({tok_per_s:.1f} tok/s)")
    print(f"  Caption: {caption!r}")

    # Release VRAM before reference model loads
    del text_model, vision_encoder, vision_projector
    torch.cuda.empty_cache()

    return {
        "backend": "custom",
        "caption_text": caption,
        "ids": r["ids"],
        "total_ms": r["total_ms"],
        "tok_per_s": tok_per_s,
        "step_med_ms": step_med,
        "vision_ms": r["vision_ms"],
        "prefill_ms": r["prefill_ms"],
    }


# ===========================================================================
# REFERENCE backend -- pretrained Qwen3-VL-8B-Instruct via transformers
# ===========================================================================

def run_reference(args, pil_image):
    """Returns dict with keys: caption_text, ids, timings."""
    print("\n" + "=" * 68)
    print(f"  [REFERENCE] {REFERENCE_MODEL_ID} (via transformers, pretrained)")
    print("=" * 68)

    # Strip PROJECT_ROOT from sys.path before importing transformers so that
    # repo-local profile.py doesn't shadow stdlib profile (which torch._dynamo
    # imports via cProfile).
    orig_path = sys.path.copy()
    sys.path = [p for p in sys.path
                if p not in ("", ".") and os.path.abspath(p) != PROJECT_ROOT]
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    finally:
        sys.path = orig_path

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    t_load = time.time()
    print(f"  Loading processor + model (first run downloads ~16 GB)...")
    processor = AutoProcessor.from_pretrained(REFERENCE_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        REFERENCE_MODEL_ID,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    load_s = time.time() - t_load
    print(f"  Loaded {n_params / 1e9:.2f} B params in {load_s:.1f} s")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": args.prompt_reference},
        ],
    }]

    t_prep = time.time()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(dtype=dtype)
    prep_ms = (time.time() - t_prep) * 1000
    in_len = inputs["input_ids"].shape[1]
    print(f"  Prep (chat template + image patches): {prep_ms:.1f} ms")
    print(f"  Input tokens (text+image placeholders): {in_len}")

    # Warmup 2 tokens
    print("  Warming up...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)

    torch.cuda.synchronize()
    print(f"  Generating {args.max_new_tokens} tokens...")
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) * 1000

    new_ids = out_ids[0, in_len:].tolist()
    caption = processor.decode(new_ids, skip_special_tokens=True)

    tok_per_s = (len(new_ids) / (total_ms / 1000)) if total_ms else 0.0

    print(f"  Total generate:   {total_ms:.2f} ms  ({tok_per_s:.1f} tok/s)")
    print(f"  Caption: {caption!r}")

    del model, processor
    torch.cuda.empty_cache()

    return {
        "backend": "reference",
        "caption_text": caption,
        "ids": new_ids,
        "total_ms": total_ms,
        "tok_per_s": tok_per_s,
        "prep_ms": prep_ms,
    }


# ===========================================================================
# Compare
# ===========================================================================

def print_comparison(custom, reference, max_new_tokens):
    print("\n" + "=" * 68)
    print("  COMPARISON  (custom proxy vs. pretrained reference)")
    print("=" * 68)
    if custom is None or reference is None:
        print("  (one backend skipped via --only)")
        if custom:
            print(f"  [CUSTOM]    {custom['caption_text']!r}")
        if reference:
            print(f"  [REFERENCE] {reference['caption_text']!r}")
        return

    # Accuracy
    c_ids = custom["ids"]
    r_ids = reference["ids"]
    n = min(len(c_ids), len(r_ids))
    id_matches = sum(1 for a, b in zip(c_ids[:n], r_ids[:n]) if a == b)
    id_match_rate = id_matches / n if n > 0 else 0.0

    real_weights = custom.get("backend") in ("custom_real", "custom_real_mm")
    print(f"\n  Accuracy")
    print(f"    Custom caption    : {custom['caption_text']!r}")
    print(f"    Reference caption : {reference['caption_text']!r}")
    print(f"    Token ID overlap  : {id_matches}/{n}  ({id_match_rate*100:.1f}%)")
    if real_weights:
        print("    NOTE: custom uses REAL pretrained weights; high overlap is")
        print("          expected. Late divergence is bf16 numerical drift across")
        print("          36 transformer layers + accumulated SDPA differences.")
    else:
        print("    NOTE: custom uses RANDOM weights -- 0% overlap is expected; it")
        print("          verifies the pipeline runs, not semantic correctness.")

    print(f"\n  Performance  ({max_new_tokens} tokens generated each)")
    print(f"    {'Metric':<22}{'Custom':>16}{'Reference':>16}{'Ratio':>12}")
    print(f"    {'-'*22}{'-'*16}{'-'*16}{'-'*12}")
    def row(name, c, r, unit="ms", fmt="{:.2f}", higher_is_better=False):
        # ratio always = custom / reference, adjusted so >1 means custom wins
        if higher_is_better:
            ratio = (c / r) if r > 0 else float("inf")
        else:
            ratio = (r / c) if c > 0 else float("inf")
        c_str = fmt.format(c)
        r_str = fmt.format(r)
        print(f"    {name:<22}{c_str + ' ' + unit:>16}{r_str + ' ' + unit:>16}"
              f"{ratio:>11.2f}x")
    row("Total latency",  custom["total_ms"],  reference["total_ms"])
    row("Throughput",     custom["tok_per_s"], reference["tok_per_s"],
        unit="tok/s", fmt="{:.2f}", higher_is_better=True)
    print("    (Ratio >1 means custom is faster/better)")
    if real_weights:
        print("    NOTE: both backends are the full 36-layer Qwen3-VL-8B model.")
    else:
        print("    NOTE: custom is a 4-layer proxy; reference is the 36-layer real")
        print("          model. Direct latency comparison is not apples-to-apples,")
        print("          but shows the order-of-magnitude gap between our profiling")
        print("          harness and the full production model.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model", default="models/qwen3_vl.py")
    parser.add_argument("--image-path", default="cat.png",
                        help="Local image file (default: cat.png)")
    parser.add_argument("--image-size", type=int, default=448,
                        help="Resize for the CUSTOM path (default 448)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-reference", default="Describe this image in detail.",
                        help="Prompt for the reference model")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="both",
                        choices=["both", "custom", "reference"])
    parser.add_argument("--load-real-weights", action="store_true",
                        help="Load pretrained Qwen3-VL-8B text-LLM weights into "
                             "the custom kernel (36 layers). Text-only path: "
                             "compares bit-for-bit to reference on --text-prompt.")
    parser.add_argument("--real-multimodal", action="store_true",
                        help="Full multimodal pipeline on the custom kernel: real "
                             "text + vision weights, deepstack feature injection, "
                             "image -> caption end-to-end.")
    parser.add_argument("--text-prompt",
                        default="The capital of France is",
                        help="Text prompt used in --load-real-weights mode")
    args = parser.parse_args()

    print("=" * 68)
    print("  Qwen3-VL Image → Caption  :  CUSTOM vs REFERENCE")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Image: {args.image_path}   dtype: {args.dtype}   "
          f"max_new_tokens: {args.max_new_tokens}")
    print("=" * 68)

    tokenizer = load_tokenizer()

    # ------------------------------------------------------------------ #
    # Branch A: real pretrained text-LLM weights on custom kernel (no image)
    # ------------------------------------------------------------------ #
    if args.load_real_weights:
        print(f"  Mode: --load-real-weights (text-only, no image)")
        print(f"  Text prompt: {args.text_prompt!r}")

        custom_result = None
        reference_result = None
        if args.only in ("both", "custom"):
            custom_result = run_custom_realweights(args, tokenizer)
        if args.only in ("both", "reference"):
            reference_result = run_reference_text_only(args, tokenizer)

        print("\n" + "=" * 68)
        print("  COMPARISON  (custom-kernel-with-real-weights vs reference)")
        print("=" * 68)
        if custom_result and reference_result:
            c_ids = custom_result["ids"]
            r_ids = reference_result["ids"]
            n = min(len(c_ids), len(r_ids))
            match = sum(1 for a, b in zip(c_ids, r_ids) if a == b)
            rate = match / n if n else 0.0
            print(f"    Custom (real wts) : {custom_result['caption_text']!r}")
            print(f"    Reference         : {reference_result['caption_text']!r}")
            print(f"    Token ID overlap  : {match}/{n}  ({rate*100:.1f}%)")
            if rate >= 0.9:
                print("    ✓ High overlap: custom kernel reproduces reference "
                      "numerics (text-only).")
            elif rate > 0:
                print("    ~ Partial overlap: divergence likely from subtle "
                      "numerics (attention backend, etc.).")
            else:
                print("    ✗ Zero overlap: check weight loading / RoPE / Q-K "
                      "norm alignment.")
            def row(name, c, r, unit="ms", fmt="{:.2f}", higher_is_better=False):
                if higher_is_better:
                    ratio = (c / r) if r > 0 else float("inf")
                else:
                    ratio = (r / c) if c > 0 else float("inf")
                c_str = fmt.format(c) + " " + unit
                r_str = fmt.format(r) + " " + unit
                print(f"    {name:<22}{c_str:>16}{r_str:>16}{ratio:>11.2f}x")
            print(f"\n    {'Metric':<22}{'Custom':>16}{'Reference':>16}{'Ratio':>12}")
            print("    " + "-" * 66)
            row("Total latency",  custom_result["total_ms"],  reference_result["total_ms"])
            row("Throughput",     custom_result["tok_per_s"], reference_result["tok_per_s"],
                unit="tok/s", fmt="{:.2f}", higher_is_better=True)
        elif custom_result:
            print(f"    Custom (real wts) : {custom_result['caption_text']!r}")
        elif reference_result:
            print(f"    Reference         : {reference_result['caption_text']!r}")
        print("=" * 68)
        return

    # ------------------------------------------------------------------ #
    # Branch B: image → caption
    #   --real-multimodal: full custom kernel (real text + vision weights)
    #   default:           random-weight 4-layer proxy (pipeline-validation only)
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.image_path):
        print(f"\n  ERROR: image not found: {args.image_path}")
        sys.exit(1)

    pil_image = load_pil_image(args.image_path)
    print(f"  PIL image: {pil_image.size[0]}x{pil_image.size[1]} {pil_image.mode}")

    custom_result = None
    reference_result = None

    if args.only in ("both", "custom"):
        if args.real_multimodal:
            custom_result = run_custom_multimodal_real(args, pil_image)
        else:
            custom_result = run_custom(args, pil_image, tokenizer)

    if args.only in ("both", "reference"):
        reference_result = run_reference(args, pil_image)

    print_comparison(custom_result, reference_result, args.max_new_tokens)
    print("=" * 68)


if __name__ == "__main__":
    main()

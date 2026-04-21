#!/usr/bin/env python3
"""
infer_qwen3vl.py -- Single input → single output inference with Qwen3-VL.

Runs the optimized Qwen3-VL forward pass (fused projections + torch.compile)
on one input sequence and returns the next-token prediction.

Note: This uses the 4-layer proxy in models/qwen3_vl.py with random weights,
so the output tokens are NOT semantically meaningful. The goal is to verify:
  1. The full inference pipeline runs end-to-end
  2. Baseline (eager) and optimized (fused + compiled) outputs agree
  3. Latency numbers on real input → output decoding

Usage:
    uv run infer_qwen3vl.py                      # default input [1,2,3,...,32]
    uv run infer_qwen3vl.py --input-len 128      # longer prefix
    uv run infer_qwen3vl.py --top-k 10           # show top-k predictions
    uv run infer_qwen3vl.py --no-optimize        # eager only (skip fuse+compile)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).parent


def load_model(model_path: str, class_name: str, dtype: torch.dtype):
    spec = importlib.util.spec_from_file_location("user_model", os.path.abspath(model_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)().to(dtype=dtype).cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-model", default="models/qwen3_vl.py")
    parser.add_argument("--class-name", default="Qwen3VLModel")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--input-len", type=int, default=32,
                        help="Length of prefix to feed (default 32)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many top predictions to display")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-optimize", action="store_true",
                        help="Run eager only, skip fused + compiled path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    project_root = str(SCRIPT_DIR)

    print("=" * 64)
    print("  Qwen3-VL Single Input → Single Output Inference")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  dtype: {args.dtype}  input-len: {args.input_len}")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # 1. Load model
    # ------------------------------------------------------------------ #
    model = load_model(args.local_model, args.class_name, dtype)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 2. Prepare a single input
    #    (random token sequence -- this is a proxy model with random weights,
    #    so the exact token values don't matter for verifying the pipeline)
    # ------------------------------------------------------------------ #
    vocab_size = model.output.weight.shape[0]
    input_ids = torch.randint(0, min(32000, vocab_size), (1, args.input_len),
                              device="cuda", dtype=torch.long)
    print(f"\n  Input token_ids: shape={tuple(input_ids.shape)}, "
          f"first 16 = {input_ids[0, :16].tolist()}")

    # ------------------------------------------------------------------ #
    # 3. Eager (baseline) forward -- warmup first for fair comparison
    # ------------------------------------------------------------------ #
    print("\n--- Eager (baseline) inference ---")
    with torch.no_grad():
        for _ in range(5):
            model(input_ids)
        torch.cuda.synchronize()

    # Time median of 10 runs
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            eager_logits = model(input_ids)
            end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    eager_ms = sorted(times)[len(times) // 2]

    eager_next = eager_logits[0, -1]                # logits for next token
    eager_topk = torch.topk(eager_next.float(), args.top_k)
    eager_token = eager_topk.indices[0].item()
    print(f"  Forward latency:     {eager_ms:.3f} ms")
    print(f"  Output shape:        {tuple(eager_logits.shape)}")
    print(f"  Predicted next token (greedy): {eager_token}")
    print(f"  Top-{args.top_k} candidates:")
    for i in range(args.top_k):
        tid = eager_topk.indices[i].item()
        logit = eager_topk.values[i].item()
        prob = F.softmax(eager_next.float(), dim=-1)[tid].item()
        print(f"    #{i+1:<2} token_id={tid:<7} logit={logit:>+8.3f}  prob={prob:.4f}")

    if args.no_optimize:
        return

    # ------------------------------------------------------------------ #
    # 4. Apply optimizations (fused projections + torch.compile)
    # ------------------------------------------------------------------ #
    print("\n--- Optimized (fused + torch.compile) inference ---")

    # Fuse projections in-place
    sys.path.insert(0, project_root)
    from serve_qwen3vl import fuse_projections
    n_fused = fuse_projections(model, verbose=False)
    print(f"  Fused {n_fused} projection modules (QKV + gate/up)")

    # Compile (avoid profile.py stdlib shadow)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    orig_path = sys.path.copy()
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(project_root)]
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        # Warmup + CUDA graph capture
        with torch.no_grad():
            for _ in range(5):
                compiled_model(input_ids)
            torch.cuda.synchronize()

        # Time median of 10 runs
        times = []
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                opt_logits = compiled_model(input_ids)
                end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        opt_ms = sorted(times)[len(times) // 2]
    finally:
        sys.path = orig_path

    opt_next = opt_logits[0, -1]
    opt_topk = torch.topk(opt_next.float(), args.top_k)
    opt_token = opt_topk.indices[0].item()
    cos_sim = F.cosine_similarity(
        eager_logits.float().flatten().unsqueeze(0),
        opt_logits.float().flatten().unsqueeze(0),
    ).item()
    topk_match = sum(1 for a, b in zip(
        eager_topk.indices.tolist(), opt_topk.indices.tolist()) if a == b)

    print(f"  Forward latency:     {opt_ms:.3f} ms")
    print(f"  Predicted next token (greedy): {opt_token}")
    print(f"  Top-{args.top_k} candidates:")
    for i in range(args.top_k):
        tid = opt_topk.indices[i].item()
        logit = opt_topk.values[i].item()
        prob = F.softmax(opt_next.float(), dim=-1)[tid].item()
        print(f"    #{i+1:<2} token_id={tid:<7} logit={logit:>+8.3f}  prob={prob:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Agreement + speedup summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 64)
    print(f"  Eager latency:      {eager_ms:.3f} ms")
    print(f"  Optimized latency:  {opt_ms:.3f} ms")
    print(f"  Speedup:            {eager_ms/opt_ms:.3f}x")
    print(f"  Logit cosine sim:   {cos_sim:.8f}")
    print(f"  Greedy token match: {'YES' if eager_token == opt_token else 'NO '} "
          f"(eager={eager_token}, opt={opt_token})")
    print(f"  Top-{args.top_k} set overlap:   {topk_match}/{args.top_k}")
    status = "PASS" if cos_sim > 0.999 else "FAIL"
    print(f"  Correctness:        {status}")
    print("=" * 64)


if __name__ == "__main__":
    main()

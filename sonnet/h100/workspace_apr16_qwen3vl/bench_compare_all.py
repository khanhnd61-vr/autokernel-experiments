#!/usr/bin/env python3
"""Head-to-head decode timing: eager PyTorch (cuBLAS + flash-attn) vs our custom kernels.

Both paths use:
  * cuBLAS for the lm_head and wo projections
  * F.scaled_dot_product_attention → flash-attn on H100

The only difference is whether our 4 fused Triton kernels are active:
  * fused_rmsnorm      (replaces 4-op RMSNorm)
  * fused_gate_up_silu + gemv_row_reduce  (replaces gate/up/silu*mul/down)
  * fused_qkv          (replaces wq/wk/wv split-K GEMV)
  * fused_add_rmsnorm  (fuses residual-add + next RMSNorm)

Usage:
    uv run bench_compare_all.py
    uv run bench_compare_all.py --n-layers 36     # full Qwen3-VL-8B
    uv run bench_compare_all.py --iters 200
"""

import argparse
import importlib.util
import sys
import time

import torch


def load_model(n_layers):
    sys.path.insert(0, "..")
    spec = importlib.util.spec_from_file_location("qwen3_vl", "../models/qwen3_vl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Qwen3VLModel(n_layers=n_layers).to(dtype=torch.float16).cuda().eval()
    return mod, m


# Save the original getters before any toggling
_ORIG_GETTERS = {}


def set_fusions(mod, enabled: bool):
    """Toggle all 4 fused kernel paths."""
    names = ["_get_fused_swiglu", "_get_fused_rmsnorm",
             "_get_fused_qkv", "_get_fused_add_rmsnorm"]
    if not _ORIG_GETTERS:
        for n in names:
            _ORIG_GETTERS[n] = getattr(mod, n)
    if enabled:
        for n in names:
            setattr(mod, n, _ORIG_GETTERS[n])
    else:
        for n in names:
            setattr(mod, n, lambda: None)


def bench_decode(m, iters, warmup=50):
    """Median GPU-time per decode step using CUDA events."""
    ids = torch.randint(0, 151936, (1, 1), device="cuda")
    with torch.no_grad():
        for _ in range(warmup):
            m(ids)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            m(ids)
            ends[i].record()
        torch.cuda.synchronize()
        ts = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))  # µs
        return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=4,
                    help="4 = proxy profile, 36 = full Qwen3-VL-8B")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--mode", choices=["both", "eager", "custom"], default="both",
                    help="run only one path (use this with separate processes for clean state)")
    args = ap.parse_args()

    print(f"Qwen3-VL decode benchmark  —  n_layers={args.n_layers}, iters={args.iters}")
    print("Hardware: H100 SXM  |  dtype: fp16  |  seq=1 (decode step)")
    print("=" * 72)

    mod, m = load_model(args.n_layers)

    if args.mode == "eager":
        set_fusions(mod, enabled=False)
        t = bench_decode(m, args.iters)
        print(f"PyTorch eager (cuBLAS + flash-attn)             {t:>10.1f} µs/step")
        return
    if args.mode == "custom":
        set_fusions(mod, enabled=True)
        t = bench_decode(m, args.iters)
        print(f"Custom kernels (4 fused) + cuBLAS + flash-attn  {t:>10.1f} µs/step")
        return

    # Both — note: state may leak between paths.  For fully clean numbers
    # run separately with --mode eager and --mode custom in fresh processes.
    set_fusions(mod, enabled=False)
    t_eager = bench_decode(m, args.iters)
    set_fusions(mod, enabled=True)
    t_custom = bench_decode(m, args.iters)

    print()
    print(f"{'Path':<50} {'µs/step':>10} {'speedup':>10}")
    print("-" * 72)
    print(f"{'PyTorch eager (cuBLAS GEMM + flash-attn SDPA)':<50} "
          f"{t_eager:>10.1f} {1.0:>9.2f}×")
    print(f"{'Custom kernels (4 fused Triton) + cuBLAS + flash-attn':<50} "
          f"{t_custom:>10.1f} {t_eager/t_custom:>9.2f}×")
    print()
    saved = t_eager - t_custom
    pct = saved / t_eager * 100
    print(f"  Saved: {saved:.1f} µs/step  ({pct:+.1f}%)")

    if args.n_layers == 4:
        print()
        print(f"  Extrapolated to 36 layers (full Qwen3-VL-8B):")
        print(f"    eager  : ~{t_eager*9/1000:.2f} ms/token")
        print(f"    custom : ~{t_custom*9/1000:.2f} ms/token")
        print(f"    → ~{(t_eager - t_custom)*9/1000:.2f} ms/token saved")


if __name__ == "__main__":
    main()

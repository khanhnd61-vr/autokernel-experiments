#!/usr/bin/env python3
"""Bench fused_rmsnorm vs PyTorch naive RMSNorm (M=1, D=4096, fp16)."""

import torch
import torch.nn.functional as F
from fused_rmsnorm import fused_rmsnorm

torch.manual_seed(0)
device = "cuda"
dt = torch.float16

D = 4096
EPS = 1e-6


def pytorch_rmsnorm(x, w, eps=EPS):
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return w * x


def bench(fn, *args, warmup=50, timed=500, flush_mb=0):
    if flush_mb:
        flush = torch.empty(flush_mb * 1024 * 1024 // 4, device=device, dtype=torch.float32)
    for _ in range(warmup):
        if flush_mb:
            flush.zero_()
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        if flush_mb:
            flush.zero_()
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


x = torch.randn(1, D, device=device, dtype=dt)
w = torch.randn(D, device=device, dtype=dt)

# Correctness
y_ref = pytorch_rmsnorm(x, w)
y_tri = fused_rmsnorm(x, w, EPS)
cos = F.cosine_similarity(y_ref.float().flatten().unsqueeze(0),
                          y_tri.float().flatten().unsqueeze(0)).item()
mad = (y_ref - y_tri).abs().max().item()
print(f"=== Correctness (M=1, D={D}) ===")
print(f"  cosine sim      = {cos:.6f}")
print(f"  max abs diff    = {mad:.5f}")
print()

# L2-warm (typical in decode — activations stay in cache)
t_torch_warm = bench(pytorch_rmsnorm, x, w)
t_triton_warm = bench(fused_rmsnorm, x, w, EPS)

# L2-cold (worst case)
t_torch_cold = bench(pytorch_rmsnorm, x, w, flush_mb=64)
t_triton_cold = bench(fused_rmsnorm, x, w, EPS, flush_mb=64)

print(f"=== Latency (M=1, D={D}) ===")
print(f"{'path':<28} {'warm µs':>10} {'cold µs':>10}")
print("-" * 52)
print(f"{'PyTorch (4 kernels)':<28} {t_torch_warm*1000:>10.2f} {t_torch_cold*1000:>10.2f}")
print(f"{'Triton fused (1 kernel)':<28} {t_triton_warm*1000:>10.2f} {t_triton_cold*1000:>10.2f}")
print(f"{'speedup':<28} {t_torch_warm/t_triton_warm:>9.2f}× {t_torch_cold/t_triton_cold:>9.2f}×")

# Per-token extrapolation
per_step_warm = (t_torch_warm - t_triton_warm) * 73 * 1000  # 36*2 + 1 = 73
print()
print(f"=== Per-token decode savings (73 RMSNorms/token) ===")
print(f"  warm  : {per_step_warm:.1f} µs/token")

# Also test with prefill shape (M=512)
print()
print("=== Prefill shape (M=512) ===")
x_pref = torch.randn(512, D, device=device, dtype=dt)
y_ref = pytorch_rmsnorm(x_pref, w)
y_tri = fused_rmsnorm(x_pref, w, EPS)
cos = F.cosine_similarity(y_ref.float().flatten().unsqueeze(0),
                          y_tri.float().flatten().unsqueeze(0)).item()
print(f"  cosine sim = {cos:.6f}")
t_torch = bench(pytorch_rmsnorm, x_pref, w)
t_triton = bench(fused_rmsnorm, x_pref, w, EPS)
print(f"  PyTorch: {t_torch*1000:.2f} µs    Triton: {t_triton*1000:.2f} µs    "
      f"speedup: {t_torch/t_triton:.2f}×")

#!/usr/bin/env python3
"""Bench fused_qkv vs eager 3-kernel baseline and cuBLAS pre-concat."""

import torch
import torch.nn.functional as F
from fused_qkv import fused_qkv

torch.manual_seed(0)
device = "cuda"
dt = torch.float16

# Qwen3-VL attention dims (GQA: 32 Q heads, 8 KV heads, head_dim=128)
K = 4096          # hidden_size
N_Q = 4096        # 32 * 128
N_KV = 1024       # 8 * 128

x = torch.randn(K, device=device, dtype=dt)
Wq = torch.randn(N_Q, K, device=device, dtype=dt) * 0.02
Wk = torch.randn(N_KV, K, device=device, dtype=dt) * 0.02
Wv = torch.randn(N_KV, K, device=device, dtype=dt) * 0.02
Wqkv_cat = torch.cat([Wq, Wk, Wv], dim=0).contiguous()   # [6144, 4096]


def eager_qkv(x, Wq, Wk, Wv):
    q = F.linear(x, Wq)
    k = F.linear(x, Wk)
    v = F.linear(x, Wv)
    return q, k, v


def cublas_cat_qkv(x, Wqkv):
    return F.linear(x, Wqkv)  # single cuBLAS GEMV


def bench(fn, *args, warmup=30, timed=300, flush_mb=0):
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


# Correctness
qkv_tri = fused_qkv(x, Wq, Wk, Wv)
q_tri = qkv_tri[:N_Q]
k_tri = qkv_tri[N_Q:N_Q + N_KV]
v_tri = qkv_tri[N_Q + N_KV:]

q_ref, k_ref, v_ref = eager_qkv(x, Wq, Wk, Wv)

cos = lambda a, b: F.cosine_similarity(a.float().flatten().unsqueeze(0),
                                        b.float().flatten().unsqueeze(0)).item()
print("=== Correctness ===")
print(f"  q: cos={cos(q_tri, q_ref):.6f}  max_diff={(q_tri - q_ref).abs().max().item():.4f}")
print(f"  k: cos={cos(k_tri, k_ref):.6f}  max_diff={(k_tri - k_ref).abs().max().item():.4f}")
print(f"  v: cos={cos(v_tri, v_ref):.6f}  max_diff={(v_tri - v_ref).abs().max().item():.4f}")
print()

# Bench
t_eager_warm = bench(eager_qkv, x, Wq, Wk, Wv)
t_cublas_warm = bench(cublas_cat_qkv, x, Wqkv_cat)
t_triton_warm = bench(fused_qkv, x, Wq, Wk, Wv)

t_eager_cold = bench(eager_qkv, x, Wq, Wk, Wv, flush_mb=64)
t_cublas_cold = bench(cublas_cat_qkv, x, Wqkv_cat, flush_mb=64)
t_triton_cold = bench(fused_qkv, x, Wq, Wk, Wv, flush_mb=64)

print(f"=== Latency (M=1) ===")
print(f"{'path':<32} {'warm µs':>10} {'cold µs':>10}")
print("-" * 56)
print(f"{'eager (3 kernels)':<32} {t_eager_warm*1000:>10.2f} {t_eager_cold*1000:>10.2f}")
print(f"{'cuBLAS pre-cat QKV':<32} {t_cublas_warm*1000:>10.2f} {t_cublas_cold*1000:>10.2f}")
print(f"{'Triton fused_qkv':<32} {t_triton_warm*1000:>10.2f} {t_triton_cold*1000:>10.2f}")
print()
print(f"  vs eager  (warm): {t_eager_warm/t_triton_warm:.2f}×   (cold): {t_eager_cold/t_triton_cold:.2f}×")
print(f"  vs cuBLAS (warm): {t_cublas_warm/t_triton_warm:.2f}×   (cold): {t_cublas_cold/t_triton_cold:.2f}×")

print()
print(f"=== Per-token decode savings (36 layers × 1 QKV/layer) ===")
saved_vs_eager = (t_eager_warm - t_triton_warm) * 36 * 1000
saved_vs_cublas = (t_cublas_warm - t_triton_warm) * 36 * 1000
print(f"  vs eager : {saved_vs_eager:+.0f} µs/token")
print(f"  vs cuBLAS: {saved_vs_cublas:+.0f} µs/token")

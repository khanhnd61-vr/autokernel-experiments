#!/usr/bin/env python3
"""Bench fused_norm_qkv vs (fused_rmsnorm + fused_qkv) vs eager."""

import torch
import torch.nn.functional as F
from fused_rmsnorm import fused_rmsnorm
from fused_qkv import fused_qkv
from fused_norm_qkv import fused_norm_qkv

torch.manual_seed(0)
device = "cuda"
dt = torch.float16

K, N_Q, N_KV = 4096, 4096, 1024
EPS = 1e-6

x = torch.randn(K, device=device, dtype=dt)
w_norm = torch.randn(K, device=device, dtype=dt).abs() * 0.5 + 0.5  # RMSNorm-ish
Wq = torch.randn(N_Q, K, device=device, dtype=dt) * 0.02
Wk = torch.randn(N_KV, K, device=device, dtype=dt) * 0.02
Wv = torch.randn(N_KV, K, device=device, dtype=dt) * 0.02


def eager_rmsnorm(x, w, eps):
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * w


def eager_flow(x, w_norm, Wq, Wk, Wv):
    h = eager_rmsnorm(x, w_norm, EPS)
    return F.linear(h, Wq), F.linear(h, Wk), F.linear(h, Wv)


def fused_parts_flow(x, w_norm, Wq, Wk, Wv):
    h = fused_rmsnorm(x, w_norm, EPS)
    return fused_qkv(h, Wq, Wk, Wv)


def fused_all_flow(x, w_norm, Wq, Wk, Wv):
    return fused_norm_qkv(x, w_norm, Wq, Wk, Wv, EPS)


# Correctness
qkv_all = fused_all_flow(x, w_norm, Wq, Wk, Wv)
qkv_parts = fused_parts_flow(x, w_norm, Wq, Wk, Wv)
q_ref, k_ref, v_ref = eager_flow(x, w_norm, Wq, Wk, Wv)

q_all = qkv_all[:N_Q]
k_all = qkv_all[N_Q:N_Q + N_KV]
v_all = qkv_all[N_Q + N_KV:]

cos = lambda a, b: F.cosine_similarity(a.float().flatten().unsqueeze(0),
                                        b.float().flatten().unsqueeze(0)).item()
print("=== Correctness (fused_norm_qkv vs eager) ===")
print(f"  q: cos={cos(q_all, q_ref):.6f}  max_diff={(q_all - q_ref).abs().max().item():.4f}")
print(f"  k: cos={cos(k_all, k_ref):.6f}  max_diff={(k_all - k_ref).abs().max().item():.4f}")
print(f"  v: cos={cos(v_all, v_ref):.6f}  max_diff={(v_all - v_ref).abs().max().item():.4f}")
print()


def bench(fn, *args, warmup=30, timed=300, flush_mb=0):
    if flush_mb:
        flush = torch.empty(flush_mb * 1024 * 1024 // 4, device=device, dtype=torch.float32)
    for _ in range(warmup):
        if flush_mb: flush.zero_()
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        if flush_mb: flush.zero_()
        starts[i].record(); fn(*args); ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts)//2]


t_eager_w = bench(eager_flow, x, w_norm, Wq, Wk, Wv)
t_parts_w = bench(fused_parts_flow, x, w_norm, Wq, Wk, Wv)
t_all_w = bench(fused_all_flow, x, w_norm, Wq, Wk, Wv)
t_eager_c = bench(eager_flow, x, w_norm, Wq, Wk, Wv, flush_mb=64)
t_parts_c = bench(fused_parts_flow, x, w_norm, Wq, Wk, Wv, flush_mb=64)
t_all_c = bench(fused_all_flow, x, w_norm, Wq, Wk, Wv, flush_mb=64)

print(f"=== Latency (M=1, RMSNorm + QKV proj) ===")
print(f"{'path':<40} {'warm µs':>10} {'cold µs':>10}")
print("-" * 64)
print(f"{'eager: rmsnorm (4) + F.linear×3':<40} {t_eager_w*1000:>10.2f} {t_eager_c*1000:>10.2f}")
print(f"{'fused_rmsnorm + fused_qkv (2 kernels)':<40} {t_parts_w*1000:>10.2f} {t_parts_c*1000:>10.2f}")
print(f"{'fused_norm_qkv (1 kernel)':<40} {t_all_w*1000:>10.2f} {t_all_c*1000:>10.2f}")
print()
print(f"  vs eager (warm): {t_eager_w/t_all_w:.2f}×   (cold): {t_eager_c/t_all_c:.2f}×")
print(f"  vs parts (warm): {t_parts_w/t_all_w:.2f}×   (cold): {t_parts_c/t_all_c:.2f}×")

print()
print(f"=== Per-token savings (36 layers × 1 per attention pre-norm) ===")
s_vs_eager = (t_eager_w - t_all_w) * 36 * 1000
s_vs_parts = (t_parts_w - t_all_w) * 36 * 1000
print(f"  vs eager: {s_vs_eager:+.0f} µs/token")
print(f"  vs parts: {s_vs_parts:+.0f} µs/token")

#!/usr/bin/env python3
"""Bench fused_add_rmsnorm vs (add + rmsnorm) baselines."""

import torch
import torch.nn.functional as F
from fused_rmsnorm import fused_rmsnorm
from fused_add_rmsnorm import fused_add_rmsnorm

torch.manual_seed(0)
device = "cuda"
dt = torch.float16
D = 4096
EPS = 1e-6


def eager_add_norm(x, d, w):
    s = x + d
    var = s.pow(2).mean(-1, keepdim=True)
    return s * torch.rsqrt(var + EPS) * w


def parts_add_norm(x, d, w):
    s = x + d  # PyTorch elementwise
    return fused_rmsnorm(s, w, EPS)


def fused_flow(x, d, w):
    # Note: mutates x in-place (matches in-model use)
    return fused_add_rmsnorm(x, d, w, EPS)


def bench(make_inputs, fn_name, fn, warmup=30, timed=300, flush_mb=0):
    """Recreate inputs every call so in-place mutation doesn't accumulate."""
    if flush_mb:
        flush = torch.empty(flush_mb * 1024 * 1024 // 4, device=device, dtype=torch.float32)
    for _ in range(warmup):
        if flush_mb: flush.zero_()
        args = make_inputs()
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    args_list = [make_inputs() for _ in range(timed)]
    torch.cuda.synchronize()
    for i in range(timed):
        if flush_mb: flush.zero_()
        starts[i].record()
        fn(*args_list[i])
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


def make_inputs():
    x = torch.randn(1, D, device=device, dtype=dt)
    d = torch.randn(1, D, device=device, dtype=dt)
    w = torch.randn(D, device=device, dtype=dt)
    return x, d, w


# Correctness — make sure to clone to handle in-place
x, d, w = make_inputs()
y_ref = eager_add_norm(x.clone(), d.clone(), w)
y_parts = parts_add_norm(x.clone(), d.clone(), w)
y_fused = fused_add_rmsnorm(x.clone(), d.clone(), w, EPS)

cos = lambda a, b: F.cosine_similarity(a.float().flatten().unsqueeze(0),
                                        b.float().flatten().unsqueeze(0)).item()
print(f"=== Correctness ===")
print(f"  parts vs eager : cos={cos(y_parts, y_ref):.6f}  diff={(y_parts - y_ref).abs().max().item():.4f}")
print(f"  fused vs eager : cos={cos(y_fused, y_ref):.6f}  diff={(y_fused - y_ref).abs().max().item():.4f}")

# Verify in-place mutation actually happened
x_chk = torch.zeros(1, D, device=device, dtype=dt)
d_chk = torch.ones(1, D, device=device, dtype=dt) * 0.5
w_chk = torch.ones(D, device=device, dtype=dt)
fused_add_rmsnorm(x_chk, d_chk, w_chk, EPS)
mutated = (x_chk == 0.5).all().item()
print(f"  in-place mutation works: {mutated}")
print()

# Bench
t_eager = bench(make_inputs, "eager", eager_add_norm)
t_parts = bench(make_inputs, "parts", parts_add_norm)
t_fused = bench(make_inputs, "fused", fused_flow)

print(f"=== Latency (M=1, D={D}, L2-warm) ===")
print(f"{'path':<32} {'us':>10}")
print("-" * 44)
print(f"{'eager (add + 4-op rmsnorm)':<32} {t_eager*1000:>10.2f}")
print(f"{'add + fused_rmsnorm':<32} {t_parts*1000:>10.2f}")
print(f"{'fused_add_rmsnorm':<32} {t_fused*1000:>10.2f}")
print()
print(f"  fused vs parts: {t_parts/t_fused:.2f}×")
print(f"  fused vs eager: {t_eager/t_fused:.2f}×")
print()
# Per token: 36 layers × 2 fusions per layer = 72 fusions
saved = (t_parts - t_fused) * 72 * 1000
print(f"  Per-token savings (72 fusions): {saved:+.0f} µs/token")

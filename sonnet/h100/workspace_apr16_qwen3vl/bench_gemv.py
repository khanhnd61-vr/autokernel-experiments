#!/usr/bin/env python3
"""Bench M=1 lm_head GEMV — Triton vs cuBLAS — with proper L2 flush.

Without an L2 flush between calls, the 1.245 GB lm_head weight matrix
appears partly cached and we see fictional >100% peak DRAM bandwidth.
We flush a 64 MB scratch tensor to defeat the L2 cache before each timed call.
"""

import torch
import importlib

# Reload kernel.py
spec = importlib.util.spec_from_file_location("kernel", "../kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
triton_mm = mod.kernel_fn

torch.manual_seed(0)
device = "cuda"
dt = torch.float16
dt_bytes = 2

# H100 HBM3 peak bandwidth = 2039 GB/s
PEAK_BW_GBPS = 2039.0


def bench_with_flush(fn, A, B, warmup=20, timed=200, l2_mb=64):
    """Time fn(A, B) with an L2 flush before every call.

    This means cold weights → real DRAM bandwidth, not L2-cached numbers.
    """
    flush = torch.empty(l2_mb * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    for _ in range(warmup):
        flush.zero_()  # writes 64 MB → evicts L2
        fn(A, B)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        flush.zero_()
        starts[i].record()
        fn(A, B)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def bench_no_flush(fn, A, B, warmup=20, timed=200):
    for _ in range(warmup):
        fn(A, B)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        starts[i].record()
        fn(A, B)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


# Decode-time matmul shapes (M=1)
shapes = [
    ("lm_head      M=1 K=4096  N=151936", 1, 4096, 151936),
    ("gate+up_fused M=1 K=4096 N=24576",   1, 4096, 24576),
    ("down         M=1 K=12288 N=4096",    1, 12288, 4096),
    ("qkv_fused    M=1 K=4096  N=6144",    1, 4096, 6144),
    ("kv_fused     M=1 K=4096  N=2048",    1, 4096, 2048),
]


def correctness_check():
    print("=== Correctness ===")
    M, K, N = 1, 4096, 151936
    A = torch.randn(M, K, device=device, dtype=dt)
    B = torch.randn(K, N, device=device, dtype=dt)
    ref = torch.mm(A, B)
    out = triton_mm(A, B)
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp(min=1e-3)).max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.float().flatten().unsqueeze(0),
        out.float().flatten().unsqueeze(0)
    ).item()
    print(f"  lm_head shape   max_abs={max_abs:.4f}  max_rel={max_rel:.4f}  "
          f"cos_sim={cos_sim:.6f}")
    assert cos_sim > 0.999, f"cos sim too low: {cos_sim}"
    print("  PASS")


def show(label, fn):
    print(f"\n=== {label} ===")
    print(f"{'Shape':<42} {'Triton us':>10} {'cuBLAS us':>10} "
          f"{'Speedup':>9} {'Triton GB/s':>11} {'cuBLAS GB/s':>11}")
    print("-" * 95)
    for name, M, K, N in shapes:
        A = torch.randn(M, K, device=device, dtype=dt)
        B = torch.randn(K, N, device=device, dtype=dt)
        ms_t = fn(triton_mm, A, B)
        ms_c = fn(torch.mm, A, B)
        bytes_w = K * N * dt_bytes
        gbs_t = bytes_w / (ms_t / 1000.0) / 1e9
        gbs_c = bytes_w / (ms_c / 1000.0) / 1e9
        speedup = ms_c / ms_t
        winner = "T" if speedup > 1.0 else "C"
        print(f"{name:<42} {ms_t*1000:>10.1f} {ms_c*1000:>10.1f} "
              f"{speedup:>8.3f}× {gbs_t:>11.0f} {gbs_c:>11.0f}  [{winner}]")


correctness_check()

# Two regimes: L2-warm (back-to-back, like a synthetic bench)
#              L2-cold (flush between calls, like real decode loop where the
#                       only thing keeping the weight warm is the kernel itself).
show("L2-WARM (back-to-back, no flush — overstates BW)", bench_no_flush)
show("L2-COLD (flush 64MB between calls — realistic decode)", bench_with_flush)

print(f"\nH100 HBM3 peak bandwidth: {PEAK_BW_GBPS} GB/s")

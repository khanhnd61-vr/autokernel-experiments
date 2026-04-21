#!/usr/bin/env python3
"""Benchmark decode-time matmul shapes (M=1) — the regime where Qwen3-VL spends
its time during token generation.

cuBLAS is well-tuned for M>=64 GEMM but uses less-optimized GEMV paths for M=1.
This is where handwritten Triton/CUDA can plausibly beat cuBLAS.
"""

import torch

# Decode-step shapes for Qwen3-VL: M=1 (single token), 36 layers
# Per token, the per-layer hot weights are:
#   wq:    [4096, 4096]
#   wk:    [4096, 1024]   GQA
#   wv:    [4096, 1024]   GQA
#   wo:    [4096, 4096]
#   w1+w3: [4096, 12288]  gate + up (often fused, 1 matmul into 24576)
#   w2:    [12288, 4096]  down
# Plus a single lm_head per token: [4096, 151936]
shapes = [
    ("lm_head      M=1 K=4096  N=151936", 1, 4096, 151936),
    ("gate+up      M=1 K=4096  N=12288",   1, 4096, 12288),
    ("gate+up_fused M=1 K=4096 N=24576",   1, 4096, 24576),
    ("down         M=1 K=12288 N=4096",    1, 12288, 4096),
    ("q_proj       M=1 K=4096  N=4096",    1, 4096, 4096),
    ("kv_proj      M=1 K=4096  N=1024",    1, 4096, 1024),
    ("kv_fused     M=1 K=4096  N=2048",    1, 4096, 2048),
    ("qkv_fused    M=1 K=4096  N=6144",    1, 4096, 6144),
]


def bench(fn, A, B, warmup=20, timed=200):
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
    return times[len(times) // 2]  # median ms


print(f"H100 peak fp16 = 756 TFLOPS, peak bw = 2039 GB/s")
print()
print(f"{'Shape':<42} {'cuBLAS us':>10} {'TFLOPS':>8} {'GB/s':>8} {'%peak BW':>10}")
print("-" * 85)

dt = torch.float16
dt_bytes = 2

# Native [M,K] @ [K,N] (B is contiguous)
for name, M, K, N in shapes:
    A = torch.randn(M, K, device="cuda", dtype=dt)
    B = torch.randn(K, N, device="cuda", dtype=dt)
    ms = bench(torch.mm, A, B)
    us = ms * 1000.0
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * dt_bytes  # almost all is weight
    tflops = flops / (ms / 1000.0) / 1e12
    gbs = bytes_moved / (ms / 1000.0) / 1e9
    pct_bw = gbs / 2039.0 * 100
    print(f"{name:<42} {us:>10.1f} {tflops:>8.1f} {gbs:>8.0f} {pct_bw:>9.1f}%")

print()
print("=== With weight.t() (nn.Linear pattern: W is [N,K], we use W.t()) ===")
print(f"{'Shape':<42} {'cuBLAS us':>10} {'TFLOPS':>8} {'GB/s':>8} {'%peak BW':>10}")
print("-" * 85)
for name, M, K, N in shapes:
    A = torch.randn(M, K, device="cuda", dtype=dt)
    W = torch.randn(N, K, device="cuda", dtype=dt)  # [out, in]
    Bt = W.t()
    ms = bench(torch.mm, A, Bt)
    us = ms * 1000.0
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * dt_bytes
    tflops = flops / (ms / 1000.0) / 1e12
    gbs = bytes_moved / (ms / 1000.0) / 1e9
    pct_bw = gbs / 2039.0 * 100
    print(f"{name:<42} {us:>10.1f} {tflops:>8.1f} {gbs:>8.0f} {pct_bw:>9.1f}%")

# Also benchmark F.linear which is the actual call path
print()
print("=== F.linear(x, W) — actual nn.Linear forward call ===")
import torch.nn.functional as F
print(f"{'Shape':<42} {'F.linear us':>12} {'TFLOPS':>8} {'GB/s':>8} {'%peak BW':>10}")
print("-" * 85)
for name, M, K, N in shapes:
    x = torch.randn(M, K, device="cuda", dtype=dt)
    W = torch.randn(N, K, device="cuda", dtype=dt)  # nn.Linear stores as [out, in]
    def f(x, W):
        return F.linear(x, W)
    ms = bench(f, x, W)
    us = ms * 1000.0
    flops = 2 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * dt_bytes
    tflops = flops / (ms / 1000.0) / 1e12
    gbs = bytes_moved / (ms / 1000.0) / 1e9
    pct_bw = gbs / 2039.0 * 100
    print(f"{name:<42} {us:>12.1f} {tflops:>8.1f} {gbs:>8.0f} {pct_bw:>9.1f}%")

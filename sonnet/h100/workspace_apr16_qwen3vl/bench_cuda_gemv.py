#!/usr/bin/env python3
"""Bench CUDA C++ GEMV vs Triton GEMV vs cuBLAS for the lm_head shape."""

import torch
import importlib

# Triton GEMV (current kernel.py)
spec = importlib.util.spec_from_file_location("kernel", "../kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
triton_mm = mod.kernel_fn

# CUDA GEMV
from cuda_gemv import gemv_cuda

torch.manual_seed(0)
device = "cuda"
dt = torch.float16


def bench_with_flush(fn, *args, warmup=20, timed=200, l2_mb=64):
    flush = torch.empty(l2_mb * 1024 * 1024 // 4, device=device, dtype=torch.float32)
    for _ in range(warmup):
        flush.zero_()
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        flush.zero_()
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


# Force compile of CUDA kernel
print("Compiling CUDA kernel...")
K, N = 4096, 151936
A = torch.randn(1, K, device=device, dtype=dt)
W = torch.randn(K, N, device=device, dtype=dt)
x = A.view(K)

ref = torch.mm(A, W).view(N)
out_cuda = gemv_cuda(x, W)
diff = (out_cuda.float() - ref.float()).abs()
cos = torch.nn.functional.cosine_similarity(
    out_cuda.float().flatten().unsqueeze(0),
    ref.float().flatten().unsqueeze(0)).item()
print(f"  CUDA correctness: max_abs={diff.max().item():.4f}  cos={cos:.6f}")
assert cos > 0.999, f"correctness fail: {cos}"

print()
print(f"{'shape':<28} {'cuBLAS us':>10} {'Triton us':>10} {'CUDA us':>10}  "
      f"{'cuBLAS GB/s':>11} {'Triton GB/s':>11} {'CUDA GB/s':>11}")
print("-" * 110)

# Just the lm_head shape — that's where the wall is
shapes = [
    ("lm_head K=4096 N=151936", 4096, 151936),
    ("gate_up K=4096 N=24576", 4096, 24576),
    ("down    K=12288 N=4096", 12288, 4096),
    ("qkv     K=4096 N=6144",  4096, 6144),
]

for name, K, N in shapes:
    A = torch.randn(1, K, device=device, dtype=dt)
    W = torch.randn(K, N, device=device, dtype=dt)
    x = A.view(K)

    cublas_ms = bench_with_flush(torch.mm, A, W)
    triton_ms = bench_with_flush(triton_mm, A, W)
    cuda_ms = bench_with_flush(gemv_cuda, x, W)

    bytes_w = K * N * 2  # fp16
    g_c = bytes_w / (cublas_ms / 1000.0) / 1e9
    g_t = bytes_w / (triton_ms / 1000.0) / 1e9
    g_d = bytes_w / (cuda_ms   / 1000.0) / 1e9

    print(f"{name:<28} {cublas_ms*1000:>10.1f} {triton_ms*1000:>10.1f} {cuda_ms*1000:>10.1f}  "
          f"{g_c:>11.0f} {g_t:>11.0f} {g_d:>11.0f}")

#!/usr/bin/env python3
"""Grid search over key hyperparameters for fused_gate_up_silu and gemv_row_reduce."""

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
device = "cuda"
dt = torch.float16
K_IN, N_MID, N_OUT = 4096, 12288, 4096

x = torch.randn(K_IN, device=device, dtype=dt)
Wg = torch.randn(N_MID, K_IN, device=device, dtype=dt) * 0.02
Wu = torch.randn(N_MID, K_IN, device=device, dtype=dt) * 0.02
m0 = torch.randn(N_MID, device=device, dtype=dt)
Wd = torch.randn(N_OUT, N_MID, device=device, dtype=dt) * 0.02

flush = torch.empty(64 * 1024 * 1024 // 4, device=device, dtype=torch.float32)


def bench_fn(fn, *args, warmup=15, timed=100):
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


# cuBLAS reference for gate+up (using fused cat call = best cuBLAS can do)
import torch.nn.functional as F
Wgu = torch.cat([Wg, Wu], dim=0).contiguous()
t_cublas_gu = bench_fn(lambda: F.linear(x, Wgu))
t_cublas_dn = bench_fn(lambda: F.linear(m0, Wd))
print(f"cuBLAS gate_up_fused: {t_cublas_gu*1000:.1f} µs")
print(f"cuBLAS down:          {t_cublas_dn*1000:.1f} µs")
print()

# ---------------------------------------------------------------
# Kernel 1: fused_gate_up_silu — sweep BLOCK_K and num_warps
# ---------------------------------------------------------------
from fused_swiglu import fused_gate_up_silu_kernel

print("=== gate+up+silu sweep ===")
print(f"{'BN':>4} {'BK':>4} {'WPS':>4} {'STG':>4} {'us':>8} {'vs cuBLAS':>10}")
print("-" * 50)

best_gu, best_gu_cfg = 1e9, None
for BLOCK_N in [64, 128, 256]:
    for BLOCK_K in [64, 128, 256]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                BLOCK_M = 16
                try:
                    def run_gu():
                        mid = torch.empty((N_MID,), device=device, dtype=dt)
                        grid = (triton.cdiv(N_MID, BLOCK_N),)
                        fused_gate_up_silu_kernel[grid](
                            x, Wg, Wu, mid, K_IN, N_MID,
                            Wg.stride(0), Wg.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            EVEN_K=(K_IN % BLOCK_K == 0),
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        return mid
                    run_gu()  # compile + warmup
                    t = bench_fn(run_gu)
                    if t < best_gu:
                        best_gu, best_gu_cfg = t, (BLOCK_N, BLOCK_K, num_warps, num_stages)
                    vs = t_cublas_gu / t
                    marker = " <-- best" if t == best_gu else ""
                    if vs > 0.95:
                        print(f"{BLOCK_N:>4} {BLOCK_K:>4} {num_warps:>4} {num_stages:>4} "
                              f"{t*1000:>8.1f} {vs:>9.3f}×{marker}")
                except Exception as e:
                    pass

print(f"\nBEST gate+up cfg: {best_gu_cfg}  → {best_gu*1000:.1f} µs  "
      f"({t_cublas_gu/best_gu:.3f}× cuBLAS)")

# ---------------------------------------------------------------
# Kernel 2: gemv_row_reduce — sweep BLOCK_ROWS and BLOCK_K
# ---------------------------------------------------------------
from fused_swiglu import gemv_row_kernel

print()
print("=== down (gemv_row_reduce) sweep ===")
print(f"{'BR':>4} {'BK':>4} {'WPS':>4} {'STG':>4} {'us':>8} {'vs cuBLAS':>10}")
print("-" * 50)

best_dn, best_dn_cfg = 1e9, None
for BLOCK_ROWS in [8, 16, 32, 64]:
    for BLOCK_K in [128, 256, 512]:
        for num_warps in [2, 4, 8, 16]:
            for num_stages in [2, 3, 4]:
                try:
                    def run_dn():
                        out = torch.empty((N_OUT,), device=device, dtype=dt)
                        grid = (triton.cdiv(N_OUT, BLOCK_ROWS),)
                        gemv_row_kernel[grid](
                            m0, Wd, out, N_MID, N_OUT,
                            Wd.stride(0), Wd.stride(1),
                            BLOCK_ROWS=BLOCK_ROWS, BLOCK_K=BLOCK_K,
                            EVEN_K=(N_MID % BLOCK_K == 0),
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        return out
                    run_dn()  # compile + warmup
                    t = bench_fn(run_dn)
                    if t < best_dn:
                        best_dn, best_dn_cfg = t, (BLOCK_ROWS, BLOCK_K, num_warps, num_stages)
                    vs = t_cublas_dn / t
                    marker = " <-- best" if t == best_dn else ""
                    if vs > 0.90:
                        print(f"{BLOCK_ROWS:>4} {BLOCK_K:>4} {num_warps:>4} {num_stages:>4} "
                              f"{t*1000:>8.1f} {vs:>9.3f}×{marker}")
                except Exception as e:
                    pass

print(f"\nBEST down cfg: {best_dn_cfg}  → {best_dn*1000:.1f} µs  "
      f"({t_cublas_dn/best_dn:.3f}× cuBLAS)")

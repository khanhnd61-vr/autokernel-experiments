#!/usr/bin/env python3
"""Grid search fused_qkv hyperparameters."""

import torch
import triton
import torch.nn.functional as F
from fused_qkv import fused_qkv_kernel

torch.manual_seed(0)
device = "cuda"
dt = torch.float16
K, N_Q, N_KV = 4096, 4096, 1024

x = torch.randn(K, device=device, dtype=dt)
Wq = torch.randn(N_Q, K, device=device, dtype=dt) * 0.02
Wk = torch.randn(N_KV, K, device=device, dtype=dt) * 0.02
Wv = torch.randn(N_KV, K, device=device, dtype=dt) * 0.02
Wqkv_cat = torch.cat([Wq, Wk, Wv], dim=0).contiguous()


def bench(fn, *args, warmup=15, timed=200):
    for _ in range(warmup): fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        starts[i].record(); fn(*args); ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts)//2]


t_cublas = bench(lambda: F.linear(x, Wqkv_cat))
print(f"cuBLAS pre-cat QKV: {t_cublas*1000:.2f} µs")
print()
print(f"{'BN':>4} {'BK':>4} {'WPS':>4} {'STG':>4} {'us':>8} {'vs cuBLAS':>10}")
print("-" * 50)

best, best_cfg = 1e9, None
N_total = N_Q + 2 * N_KV
for BLOCK_N in [32, 64, 128, 256]:
    if N_Q % BLOCK_N or N_KV % BLOCK_N: continue
    for BLOCK_K in [64, 128, 256]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                try:
                    def run():
                        out = torch.empty((N_total,), device=device, dtype=dt)
                        grid = (triton.cdiv(N_total, BLOCK_N),)
                        fused_qkv_kernel[grid](
                            x, Wq, Wk, Wv, out,
                            K, N_Q, N_KV,
                            Wq.stride(0), Wq.stride(1),
                            Wk.stride(0), Wk.stride(1),
                            Wv.stride(0), Wv.stride(1),
                            BLOCK_M=16, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            EVEN_K=(K % BLOCK_K == 0),
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        return out
                    run()
                    t = bench(run)
                    if t < best:
                        best, best_cfg = t, (BLOCK_N, BLOCK_K, num_warps, num_stages)
                    vs = t_cublas / t
                    if vs > 0.85:
                        mark = " <-- best" if t == best else ""
                        print(f"{BLOCK_N:>4} {BLOCK_K:>4} {num_warps:>4} {num_stages:>4} "
                              f"{t*1000:>8.2f} {vs:>9.3f}×{mark}")
                except Exception:
                    pass

print(f"\nBEST: {best_cfg} → {best*1000:.2f} µs ({t_cublas/best:.3f}× cuBLAS)")

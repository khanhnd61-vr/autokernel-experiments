#!/usr/bin/env python3
"""Bench fused SwiGLU (M=1) vs the F.linear/silu/F.linear baseline.

Qwen3-VL FFN at decode:  x[4096] → Wg/Wu[12288,4096] → silu*mul → Wd[4096,12288] → out[4096]

Baselines:
  * eager  : 4 separate ops (gate, up, silu*mul, down) — 4 kernel launches
  * fused2 : pre-cat Wgu = [Wg;Wu] → gate_up = F.linear(x, Wgu) (1 GEMV) + silu*mul + down (3 launches)
  * triton : fused_gate_up_silu (1 launch) + gemv_nn (1 launch) — 2 launches
"""

import torch
import torch.nn.functional as F
from fused_swiglu import fused_gate_up_silu, gemv_row_reduce, fused_swiglu_ffn

torch.manual_seed(0)
device = "cuda"
dt = torch.float16

# Qwen3-VL dims
K_IN, N_MID, N_OUT = 4096, 12288, 4096


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


# Make weights
x = torch.randn(K_IN, device=device, dtype=dt) * 0.1
Wg = torch.randn(N_MID, K_IN, device=device, dtype=dt) * 0.02
Wu = torch.randn(N_MID, K_IN, device=device, dtype=dt) * 0.02
Wd = torch.randn(N_OUT, N_MID, device=device, dtype=dt) * 0.02

# Pre-cat for fused2
Wgu = torch.cat([Wg, Wu], dim=0).contiguous()
assert Wgu.shape == (2 * N_MID, K_IN)


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def eager_ffn(x, Wg, Wu, Wd):
    gate = F.linear(x, Wg)
    up = F.linear(x, Wu)
    mid = F.silu(gate) * up
    return F.linear(mid, Wd)


def fused2_ffn(x, Wgu, Wd):
    """Pre-concatenated gate+up matmul, then silu*mul, then down."""
    gu = F.linear(x, Wgu)        # [24576]
    gate, up = gu.chunk(2, dim=-1)
    mid = F.silu(gate) * up      # [12288]
    return F.linear(mid, Wd)     # [4096]


def triton_ffn(x, Wg, Wu, Wd):
    return fused_swiglu_ffn(x, Wg, Wu, Wd)


# ----------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------
print("=== Correctness ===")
out_eager = eager_ffn(x, Wg, Wu, Wd)
out_fused2 = fused2_ffn(x, Wgu, Wd)
out_triton = triton_ffn(x, Wg, Wu, Wd)

cos = lambda a, b: torch.nn.functional.cosine_similarity(
    a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)).item()
print(f"  eager vs fused2  cos = {cos(out_eager, out_fused2):.6f}")
print(f"  eager vs triton  cos = {cos(out_eager, out_triton):.6f}")
print(f"  triton max abs diff = {(out_triton - out_eager).abs().max().item():.4f}")

# ----------------------------------------------------------------------
# Per-step timing
# ----------------------------------------------------------------------
print()
print("=== Per-call latency (L2-cold, with 64MB flush) ===")
print()
print(f"{'Path':<32} {'Total us':>10} {'vs eager':>10} {'vs fused2':>10}")
print("-" * 70)

t_eager   = bench_with_flush(eager_ffn,  x, Wg, Wu, Wd)
t_fused2  = bench_with_flush(fused2_ffn, x, Wgu, Wd)
t_triton  = bench_with_flush(triton_ffn, x, Wg, Wu, Wd)

print(f"{'eager (4 ops)':<32} {t_eager*1000:>10.1f} {1.0:>9.2f}× {t_eager/t_fused2:>9.2f}×")
print(f"{'fused2 (gate+up cat, 3 ops)':<32} {t_fused2*1000:>10.1f} {t_eager/t_fused2:>9.2f}× {1.0:>9.2f}×")
print(f"{'triton (2 ops, fused silu)':<32} {t_triton*1000:>10.1f} {t_eager/t_triton:>9.2f}× {t_fused2/t_triton:>9.2f}×")

# ----------------------------------------------------------------------
# Per-component
# ----------------------------------------------------------------------
print()
print("=== Per-component breakdown (L2-cold) ===")
print()
print(f"{'Op':<40} {'us':>8}")
print("-" * 50)

# eager components
def gate_only(x, Wg): return F.linear(x, Wg)
def silu_mul(g, u):  return F.silu(g) * u
def down_only(m, Wd): return F.linear(m, Wd)

g0 = F.linear(x, Wg)
u0 = F.linear(x, Wu)
m0 = F.silu(g0) * u0

t_gate    = bench_with_flush(gate_only, x, Wg)
t_up      = bench_with_flush(gate_only, x, Wu)
t_silumul = bench_with_flush(silu_mul, g0, u0)
t_down    = bench_with_flush(down_only, m0, Wd)
t_gateup_fused = bench_with_flush(gate_only, x, Wgu)
t_fused_silu  = bench_with_flush(fused_gate_up_silu, x, Wg, Wu)
t_gemv_down   = bench_with_flush(gemv_row_reduce, m0, Wd)

print(f"{'gate (cuBLAS GEMV)':<40} {t_gate*1000:>8.1f}")
print(f"{'up   (cuBLAS GEMV)':<40} {t_up*1000:>8.1f}")
print(f"{'silu*mul (elementwise)':<40} {t_silumul*1000:>8.1f}")
print(f"{'down (cuBLAS GEMV)':<40} {t_down*1000:>8.1f}")
print(f"{'gate_up_fused (cat+1 cuBLAS GEMV)':<40} {t_gateup_fused*1000:>8.1f}")
print(f"{'triton fused_gate_up_silu':<40} {t_fused_silu*1000:>8.1f}")
print(f"{'triton gemv_row_reduce (down)':<40} {t_gemv_down*1000:>8.1f}")
print()
sum_eager = (t_gate + t_up + t_silumul + t_down) * 1000
sum_fused2 = (t_gateup_fused + t_silumul + t_down) * 1000
sum_triton = (t_fused_silu + t_gemv_down) * 1000
print(f"  Sum eager (4 ops):  {sum_eager:.1f} us")
print(f"  Sum fused2 (3 ops): {sum_fused2:.1f} us")
print(f"  Sum triton (2 ops): {sum_triton:.1f} us")

# ----------------------------------------------------------------------
# Per-token total impact (36 layers)
# ----------------------------------------------------------------------
print()
print("=== Per-token decode impact (×36 layers, FFN only) ===")
saved_vs_eager  = (t_eager - t_triton) * 36 * 1000  # us
saved_vs_fused2 = (t_fused2 - t_triton) * 36 * 1000
print(f"  Triton FFN saves vs eager  : {saved_vs_eager:.0f} us/token  "
      f"({(t_eager - t_triton) / t_eager * 100:+.1f}%)")
print(f"  Triton FFN saves vs fused2 : {saved_vs_fused2:.0f} us/token  "
      f"({(t_fused2 - t_triton) / t_fused2 * 100:+.1f}%)")

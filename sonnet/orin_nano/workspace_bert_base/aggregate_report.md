# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-18 13:07:06 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 4.51 | 8.74 | 1.94x | 11 | 4 | 36% | 42 |
| 2 | kernel_gemm_2.py | gemm | DONE | 4.57 | 9.10 | 1.99x | 14 | 4 | 29% | 115 |
| 3 | kernel_gemm_3.py | gemm | DONE | 4.47 | 9.95 | 2.23x | 9 | 3 | 33% | 135 |
| 4 | kernel_fused_mlp_4.py | fused_mlp | DONE | 0.49 | 0.53 | 1.08x | 8 | 3 | 38% | 25 |
| 5 | kernel_fused_mlp_5.py | fused_mlp | DONE | 6.80 | 8.35 | 1.23x | 10 | 4 | 40% | 35 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.44x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 21.6% of GPU time, 1.94x speedup (10.5% time saved)
- **kernel_gemm_2.py**: 18.0% of GPU time, 1.99x speedup (9.0% time saved)
- **kernel_gemm_3.py**: 16.6% of GPU time, 2.23x speedup (9.1% time saved)
- **kernel_fused_mlp_4.py**: 7.4% of GPU time, 1.08x speedup (0.6% time saved)
- **kernel_fused_mlp_5.py**: 7.4% of GPU time, 1.23x speedup (1.4% time saved)

## Time Allocation

Total optimization time: 352 minutes (5.9 hours)

- kernel_gemm_1.py: 42 min (12%)
- kernel_gemm_2.py: 115 min (33%)
- kernel_gemm_3.py: 135 min (38%)
- kernel_fused_mlp_4.py: 25 min (7%)
- kernel_fused_mlp_5.py: 35 min (10%)

## Keep Rates

- kernel_gemm_1.py: 4/11 (36%)
- kernel_gemm_2.py: 4/14 (29%)
- kernel_gemm_3.py: 3/9 (33%)
- kernel_fused_mlp_4.py: 3/8 (38%)
- kernel_fused_mlp_5.py: 4/10 (40%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.94x (target: 2.0x)
- **kernel_gemm_2.py** (rank 2): speedup only 1.99x (target: 2.0x)
- **kernel_fused_mlp_4.py** (rank 4): speedup only 1.08x (target: 2.0x); only 6.4% of peak (headroom to 90%)
- **kernel_fused_mlp_5.py** (rank 5): speedup only 1.23x (target: 2.0x)

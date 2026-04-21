# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-20 16:40:07 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 27.58 | 34.19 | 1.24x | 19 | 6 | 32% | 33 |
| 2 | kernel_gemm_2.py | gemm | DONE | 38.26 | 38.26 | 1.00x | 6 | 1 | 17% | 36 |
| 3 | kernel_gemm_3.py | gemm | DONE | 51.87 | 51.87 | 1.00x | 6 | 1 | 17% | 39 |
| 4 | kernel_softmax_4.py | softmax | DONE | 1.22 | 1.22 | 1.00x | 6 | 1 | 17% | 41 |
| 5 | kernel_softmax_5.py | softmax | DONE | 1.22 | 1.22 | 1.00x | 6 | 1 | 17% | 41 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.03x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 13.1% of GPU time, 1.24x speedup (2.5% time saved)
- **kernel_gemm_2.py**: 7.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_gemm_3.py**: 7.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_softmax_4.py**: 6.9% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_softmax_5.py**: 6.9% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 190 minutes (3.2 hours)

- kernel_gemm_1.py: 33 min (17%)
- kernel_gemm_2.py: 36 min (19%)
- kernel_gemm_3.py: 39 min (21%)
- kernel_softmax_4.py: 41 min (22%)
- kernel_softmax_5.py: 41 min (22%)

## Keep Rates

- kernel_gemm_1.py: 6/19 (32%)
- kernel_gemm_2.py: 1/6 (17%)
- kernel_gemm_3.py: 1/6 (17%)
- kernel_softmax_4.py: 1/6 (17%)
- kernel_softmax_5.py: 1/6 (17%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.24x (target: 2.0x); high impact (13.1% of GPU time) with low speedup
- **kernel_gemm_2.py** (rank 2): speedup only 1.00x (target: 2.0x)
- **kernel_gemm_3.py** (rank 3): speedup only 1.00x (target: 2.0x)
- **kernel_softmax_4.py** (rank 4): speedup only 1.00x (target: 2.0x)
- **kernel_softmax_5.py** (rank 5): speedup only 1.00x (target: 2.0x)

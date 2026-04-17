# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-17 04:28:33 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 39.13 | 39.63 | 1.01x | 7 | 2 | 29% | 16 |
| 2 | kernel_gemm_2.py | gemm | DONE | 31.71 | 34.64 | 1.09x | 19 | 6 | 32% | 29 |
| 3 | kernel_gemm_3.py | gemm | DONE | 31.87 | 40.43 | 1.27x | 11 | 1 | 9% | 32 |
| 4 | kernel_gemm_4.py | gemm | DONE | 29.43 | 31.51 | 1.07x | 7 | 1 | 14% | 0 |
| 5 | kernel_gemm_5.py | gemm | DONE | 25.62 | 26.92 | 1.05x | 8 | 2 | 25% | 0 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.04x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 33.2% of GPU time, 1.01x speedup (0.4% time saved)
- **kernel_gemm_2.py**: 13.7% of GPU time, 1.09x speedup (1.2% time saved)
- **kernel_gemm_3.py**: 7.3% of GPU time, 1.27x speedup (1.5% time saved)
- **kernel_gemm_4.py**: 7.0% of GPU time, 1.07x speedup (0.5% time saved)
- **kernel_gemm_5.py**: 5.3% of GPU time, 1.05x speedup (0.3% time saved)

## Time Allocation

Total optimization time: 77 minutes (1.3 hours)

- kernel_gemm_1.py: 16 min (21%)
- kernel_gemm_2.py: 29 min (38%)
- kernel_gemm_3.py: 32 min (42%)
- kernel_gemm_4.py: 0 min (0%)
- kernel_gemm_5.py: 0 min (0%)

## Keep Rates

- kernel_gemm_1.py: 2/7 (29%)
- kernel_gemm_2.py: 6/19 (32%)
- kernel_gemm_3.py: 1/11 (9%)
- kernel_gemm_4.py: 1/7 (14%)
- kernel_gemm_5.py: 2/8 (25%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.01x (target: 2.0x); high impact (33.2% of GPU time) with low speedup
- **kernel_gemm_2.py** (rank 2): speedup only 1.09x (target: 2.0x); high impact (13.7% of GPU time) with low speedup
- **kernel_gemm_3.py** (rank 3): speedup only 1.27x (target: 2.0x)
- **kernel_gemm_4.py** (rank 4): speedup only 1.07x (target: 2.0x)
- **kernel_gemm_5.py** (rank 5): speedup only 1.05x (target: 2.0x)

# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-19 06:16:24 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 4.96 | 7.43 | 1.50x | 2 | 2 | 100% | 625 |
| 2 | kernel_gemm_2.py | gemm | DONE | 4.81 | 5.78 | 1.20x | 2 | 2 | 100% | 635 |
| 3 | kernel_gemm_3.py | gemm | DONE | 4.69 | 6.56 | 1.40x | 2 | 2 | 100% | 640 |
| 4 | kernel_gemm_4.py | gemm | DONE | 0.30 | 0.31 | 1.05x | 4 | 4 | 100% | 671 |
| 5 | kernel_gemm_5.py | gemm | OPTIMIZING | 0.30 | 0.30 | 1.00x | 1 | 1 | 100% | 677 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.13x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 17.3% of GPU time, 1.50x speedup (5.8% time saved)
- **kernel_gemm_2.py**: 13.9% of GPU time, 1.20x speedup (2.3% time saved)
- **kernel_gemm_3.py**: 12.0% of GPU time, 1.40x speedup (3.4% time saved)
- **kernel_gemm_4.py**: 8.8% of GPU time, 1.05x speedup (0.4% time saved)
- **kernel_gemm_5.py**: 5.1% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 3248 minutes (54.1 hours)

- kernel_gemm_1.py: 625 min (19%)
- kernel_gemm_2.py: 635 min (20%)
- kernel_gemm_3.py: 640 min (20%)
- kernel_gemm_4.py: 671 min (21%)
- kernel_gemm_5.py: 677 min (21%)

## Keep Rates

- kernel_gemm_1.py: 2/2 (100%)
- kernel_gemm_2.py: 2/2 (100%)
- kernel_gemm_3.py: 2/2 (100%)
- kernel_gemm_4.py: 4/4 (100%)
- kernel_gemm_5.py: 1/1 (100%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.50x (target: 2.0x); high impact (17.3% of GPU time) with low speedup
- **kernel_gemm_2.py** (rank 2): speedup only 1.20x (target: 2.0x); high impact (13.9% of GPU time) with low speedup
- **kernel_gemm_3.py** (rank 3): speedup only 1.40x (target: 2.0x); high impact (12.0% of GPU time) with low speedup
- **kernel_gemm_4.py** (rank 4): speedup only 1.05x (target: 2.0x)
- **kernel_gemm_5.py** (rank 5): speedup only 1.00x (target: 2.0x)

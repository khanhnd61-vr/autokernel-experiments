# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-17 16:50:57 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 11.99 | 16.13 | 1.35x | 11 | 3 | 27% | 46 |
| 2 | kernel_gemm_2.py | gemm | DONE | 21.43 | 22.81 | 1.06x | 14 | 3 | 21% | 90 |
| 3 | kernel_gemm_3.py | gemm | DONE | 17.37 | 19.33 | 1.11x | 7 | 2 | 29% | 45 |
| 4 | kernel_gemm_4.py | gemm | DONE | 10.73 | 14.01 | 1.31x | 5 | 1 | 20% | 30 |
| 5 | kernel_gemm_5.py | gemm | DONE | 0.74 | 0.82 | 1.11x | 2 | 1 | 50% | 10 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.13x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 21.5% of GPU time, 1.35x speedup (5.5% time saved)
- **kernel_gemm_2.py**: 18.8% of GPU time, 1.06x speedup (1.1% time saved)
- **kernel_gemm_3.py**: 14.6% of GPU time, 1.11x speedup (1.5% time saved)
- **kernel_gemm_4.py**: 13.1% of GPU time, 1.31x speedup (3.1% time saved)
- **kernel_gemm_5.py**: 4.6% of GPU time, 1.11x speedup (0.5% time saved)

## Time Allocation

Total optimization time: 221 minutes (3.7 hours)

- kernel_gemm_1.py: 46 min (21%)
- kernel_gemm_2.py: 90 min (41%)
- kernel_gemm_3.py: 45 min (20%)
- kernel_gemm_4.py: 30 min (14%)
- kernel_gemm_5.py: 10 min (5%)

## Keep Rates

- kernel_gemm_1.py: 3/11 (27%)
- kernel_gemm_2.py: 3/14 (21%)
- kernel_gemm_3.py: 2/7 (29%)
- kernel_gemm_4.py: 1/5 (20%)
- kernel_gemm_5.py: 1/2 (50%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.35x (target: 2.0x); high impact (21.5% of GPU time) with low speedup
- **kernel_gemm_2.py** (rank 2): speedup only 1.06x (target: 2.0x); high impact (18.8% of GPU time) with low speedup
- **kernel_gemm_3.py** (rank 3): speedup only 1.11x (target: 2.0x); high impact (14.6% of GPU time) with low speedup
- **kernel_gemm_4.py** (rank 4): speedup only 1.31x (target: 2.0x); high impact (13.1% of GPU time) with low speedup
- **kernel_gemm_5.py** (rank 5): speedup only 1.11x (target: 2.0x)

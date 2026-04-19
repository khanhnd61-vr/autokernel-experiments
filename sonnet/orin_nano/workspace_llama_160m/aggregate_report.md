# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-18 18:31:34 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 6.42 | 6.42 | 1.00x | 6 | 1 | 17% | 42 |
| 2 | kernel_gemm_2.py | gemm | DONE | 6.13 | 6.13 | 1.00x | 4 | 1 | 25% | 80 |
| 3 | kernel_gemm_3.py | gemm | DONE | 6.16 | 7.56 | 1.23x | 3 | 2 | 67% | 30 |
| 4 | kernel_gemm_4.py | gemm | DONE | 5.66 | 6.00 | 1.06x | 5 | 2 | 40% | 25 |
| 7 | kernel_gemm_7.py | gemm | DONE | 4.31 | 4.38 | 1.02x | 3 | 2 | 67% | 15 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.02x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 14.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_gemm_2.py**: 10.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_gemm_3.py**: 7.5% of GPU time, 1.23x speedup (1.4% time saved)
- **kernel_gemm_4.py**: 6.5% of GPU time, 1.06x speedup (0.4% time saved)
- **kernel_gemm_7.py**: 3.6% of GPU time, 1.02x speedup (0.1% time saved)

## Time Allocation

Total optimization time: 192 minutes (3.2 hours)

- kernel_gemm_1.py: 42 min (22%)
- kernel_gemm_2.py: 80 min (42%)
- kernel_gemm_3.py: 30 min (16%)
- kernel_gemm_4.py: 25 min (13%)
- kernel_gemm_7.py: 15 min (8%)

## Keep Rates

- kernel_gemm_1.py: 1/6 (17%)
- kernel_gemm_2.py: 1/4 (25%)
- kernel_gemm_3.py: 2/3 (67%)
- kernel_gemm_4.py: 2/5 (40%)
- kernel_gemm_7.py: 2/3 (67%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.00x (target: 2.0x); only 76.7% of peak (headroom to 90%); high impact (14.1% of GPU time) with low speedup
- **kernel_gemm_2.py** (rank 2): speedup only 1.00x (target: 2.0x); only 73.4% of peak (headroom to 90%); high impact (10.4% of GPU time) with low speedup
- **kernel_gemm_3.py** (rank 3): speedup only 1.23x (target: 2.0x)
- **kernel_gemm_4.py** (rank 4): speedup only 1.06x (target: 2.0x); only 71.8% of peak (headroom to 90%)
- **kernel_gemm_7.py** (rank 7): speedup only 1.02x (target: 2.0x); only 52.4% of peak (headroom to 90%)

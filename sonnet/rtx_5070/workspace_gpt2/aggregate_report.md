# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-16 12:12:01 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 35.86 | 39.98 | 1.11x | 8 | 3 | 38% | 62 |
| 2 | kernel_gemm_2.py | gemm | DONE | 24.65 | 30.36 | 1.23x | 7 | 2 | 29% | 70 |
| 3 | kernel_gemm_3.py | gemm | DONE | 36.80 | 39.58 | 1.08x | 8 | 4 | 50% | 95 |
| 16 | kernel_gemm_16.py | gemm | DONE | 36.06 | 39.20 | 1.09x | 3 | 1 | 33% | 20 |
| 18 | kernel_gemm_18.py | gemm | DONE | 36.06 | 39.30 | 1.09x | 1 | 1 | 100% | 10 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.05x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 13.3% of GPU time, 1.11x speedup (1.4% time saved)
- **kernel_gemm_2.py**: 11.2% of GPU time, 1.23x speedup (2.1% time saved)
- **kernel_gemm_3.py**: 11.2% of GPU time, 1.08x speedup (0.8% time saved)
- **kernel_gemm_16.py**: 3.0% of GPU time, 1.09x speedup (0.2% time saved)
- **kernel_gemm_18.py**: 2.3% of GPU time, 1.09x speedup (0.2% time saved)

## Time Allocation

Total optimization time: 257 minutes (4.3 hours)

- kernel_gemm_1.py: 62 min (24%)
- kernel_gemm_2.py: 70 min (27%)
- kernel_gemm_3.py: 95 min (37%)
- kernel_gemm_16.py: 20 min (8%)
- kernel_gemm_18.py: 10 min (4%)

## Keep Rates

- kernel_gemm_1.py: 3/8 (38%)
- kernel_gemm_2.py: 2/7 (29%)
- kernel_gemm_3.py: 4/8 (50%)
- kernel_gemm_16.py: 1/3 (33%)
- kernel_gemm_18.py: 1/1 (100%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.11x (target: 2.0x); only 70.2% of peak (headroom to 90%); high impact (13.3% of GPU time) with low speedup
- **kernel_gemm_2.py** (rank 2): speedup only 1.23x (target: 2.0x); only 53.3% of peak (headroom to 90%); high impact (11.2% of GPU time) with low speedup
- **kernel_gemm_3.py** (rank 3): speedup only 1.08x (target: 2.0x); only 69.5% of peak (headroom to 90%); high impact (11.2% of GPU time) with low speedup
- **kernel_gemm_16.py** (rank 16): speedup only 1.09x (target: 2.0x); only 68.8% of peak (headroom to 90%)
- **kernel_gemm_18.py** (rank 18): speedup only 1.09x (target: 2.0x); only 69.0% of peak (headroom to 90%)

# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-18 07:49:36 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 4.15 | 5.78 | 1.51x | 11 | 3 | 27% | 38 |
| 4 | kernel_gemm_4.py | gemm | DONE | 2.62 | 2.49 | 1.07x | 3 | 1 | 33% | 20 |
| 5 | kernel_softmax_5.py | softmax | DONE | 0.02 | 0.07 | 3.31x | 11 | 5 | 45% | 45 |
| 6 | kernel_softmax_6.py | softmax | DONE | 0.02 | 0.07 | 3.07x | 1 | 1 | 100% | 5 |
| 11 | kernel_gemm_11.py | gemm | DONE | 5.00 | 10.36 | 0.96x | 3 | 1 | 33% | 15 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.17x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 16.1% of GPU time, 1.51x speedup (5.4% time saved)
- **kernel_gemm_4.py**: 7.5% of GPU time, 1.07x speedup (0.5% time saved)
- **kernel_softmax_5.py**: 6.5% of GPU time, 3.31x speedup (4.5% time saved)
- **kernel_softmax_6.py**: 6.5% of GPU time, 3.07x speedup (4.4% time saved)
- **kernel_gemm_11.py**: 4.0% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 123 minutes (2.0 hours)

- kernel_gemm_1.py: 38 min (31%)
- kernel_gemm_4.py: 20 min (16%)
- kernel_softmax_5.py: 45 min (37%)
- kernel_softmax_6.py: 5 min (4%)
- kernel_gemm_11.py: 15 min (12%)

## Keep Rates

- kernel_gemm_1.py: 3/11 (27%)
- kernel_gemm_4.py: 1/3 (33%)
- kernel_softmax_5.py: 5/11 (45%)
- kernel_softmax_6.py: 1/1 (100%)
- kernel_gemm_11.py: 1/3 (33%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.51x (target: 2.0x); only 69.1% of peak (headroom to 90%)
- **kernel_gemm_4.py** (rank 4): speedup only 1.07x (target: 2.0x)
- **kernel_softmax_5.py** (rank 5): only 0.9% of peak (headroom to 90%)
- **kernel_softmax_6.py** (rank 6): only 0.9% of peak (headroom to 90%)
- **kernel_gemm_11.py** (rank 11): speedup only 0.96x (target: 2.0x)

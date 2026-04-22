# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-22 07:24:28 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_flash_attention_1.py | flash_attention | OPTIMIZING | 4.89 | 19.26 | 3.93x | 4 | 4 | 100% | 140 |
| 2 | kernel_gemm_2.py | gemm | PENDING | 46.67 | 46.67 | 1.00x | 1 | 1 | 100% | 82 |
| 3 | kernel_gemm_3.py | gemm | PENDING | 50.60 | 50.60 | 1.00x | 1 | 1 | 100% | 86 |
| 4 | kernel_gemm_4.py | gemm | PENDING | -- | -- | -- | 0 | 0 | -- | 0 |
| 5 | kernel_gemm_5.py | gemm | PENDING | -- | -- | -- | 0 | 0 | -- | 0 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.15x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_flash_attention_1.py**: 17.6% of GPU time, 3.93x speedup (13.1% time saved)
- **kernel_gemm_2.py**: 13.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_gemm_3.py**: 9.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_gemm_4.py**: 7.7% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_gemm_5.py**: 6.7% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 308 minutes (5.1 hours)

- kernel_flash_attention_1.py: 140 min (45%)
- kernel_gemm_2.py: 82 min (27%)
- kernel_gemm_3.py: 86 min (28%)
- kernel_gemm_4.py: 0 min (0%)
- kernel_gemm_5.py: 0 min (0%)

## Keep Rates

- kernel_flash_attention_1.py: 4/4 (100%)
- kernel_gemm_2.py: 1/1 (100%)
- kernel_gemm_3.py: 1/1 (100%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_2.py** (rank 2): not yet optimized
- **kernel_gemm_3.py** (rank 3): not yet optimized
- **kernel_gemm_4.py** (rank 4): not yet optimized
- **kernel_gemm_5.py** (rank 5): not yet optimized

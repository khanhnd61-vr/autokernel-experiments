# AutoKernel -- Aggregate Optimization Report

Generated: 2026-04-17 04:23:50 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 1 | kernel_gemm_1.py | gemm | DONE | 35.91 | 38.67 | 1.08x | 10 | 3 | 30% | 18 |
| 4 | kernel_gemm_4.py | gemm | DONE | 33.58 | 41.53 | 1.24x | 12 | 3 | 25% | 18 |
| 6 | kernel_flash_attention_6.py | flash_attention | DONE | 37.77 | 52.40 | 1.39x | 9 | 2 | 22% | 15 |
| 7 | kernel_flash_attention_7.py | flash_attention | DONE | 37.77 | 53.97 | 1.43x | 1 | 1 | 100% | 5 |
| 8 | kernel_fused_mlp_8.py | fused_mlp | DONE | 31.52 | 32.84 | 1.04x | 8 | 2 | 25% | 15 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.08x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_gemm_1.py**: 26.1% of GPU time, 1.08x speedup (1.9% time saved)
- **kernel_gemm_4.py**: 13.3% of GPU time, 1.24x speedup (2.6% time saved)
- **kernel_flash_attention_6.py**: 4.4% of GPU time, 1.39x speedup (1.2% time saved)
- **kernel_flash_attention_7.py**: 4.4% of GPU time, 1.43x speedup (1.3% time saved)
- **kernel_fused_mlp_8.py**: 2.9% of GPU time, 1.04x speedup (0.1% time saved)

## Time Allocation

Total optimization time: 71 minutes (1.2 hours)

- kernel_gemm_1.py: 18 min (25%)
- kernel_gemm_4.py: 18 min (25%)
- kernel_flash_attention_6.py: 15 min (21%)
- kernel_flash_attention_7.py: 5 min (7%)
- kernel_fused_mlp_8.py: 15 min (21%)

## Keep Rates

- kernel_gemm_1.py: 3/10 (30%)
- kernel_gemm_4.py: 3/12 (25%)
- kernel_flash_attention_6.py: 2/9 (22%)
- kernel_flash_attention_7.py: 1/1 (100%)
- kernel_fused_mlp_8.py: 2/8 (25%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_gemm_1.py** (rank 1): speedup only 1.08x (target: 2.0x); high impact (26.1% of GPU time) with low speedup
- **kernel_gemm_4.py** (rank 4): speedup only 1.24x (target: 2.0x); only 72.9% of peak (headroom to 90%); high impact (13.3% of GPU time) with low speedup
- **kernel_flash_attention_6.py** (rank 6): speedup only 1.39x (target: 2.0x)
- **kernel_flash_attention_7.py** (rank 7): speedup only 1.43x (target: 2.0x)
- **kernel_fused_mlp_8.py** (rank 8): speedup only 1.04x (target: 2.0x); only 57.7% of peak (headroom to 90%)

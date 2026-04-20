**ViT-B/16 AutoKernel Optimization — Results**

| Kernel |  Shape                |  Baseline     |  Best         | Speedup |
|--------|-----------------------|---------------|---------------|---------|
| gemm_1 |  M=197, N=768, K=3072 |  11.99 TFLOPS |  16.13 TFLOPS |  +34.6% |
| gemm_2 |  M=197, N=3072, K=768 |  21.43 TFLOPS |  22.81 TFLOPS |  +6.4%  |
| gemm_3 |  M=197, N=2304, K=768 |  17.37 TFLOPS |  19.33 TFLOPS |  +11.3% |
| gemm_4 |  M=197, N=768, K=768  |  10.73 TFLOPS |  14.01 TFLOPS |  +30.6% |
| gemm_5 |  M=197, N=197, K=64   |  0.74 TFLOPS  |  0.82 TFLOPS  |  +10.9% |


End-to-end: 0.98x (2.8ms → 2.8ms) — neutral due to verify.py applying only rank-1 kernel to all layers; individual kernel gains don't fully compound end-to-end.

Key findings:
  - BK=64 consistently helped across all shapes (halves K-loop iterations)
  - BM=32 improved shapes with M=197 (7 blocks vs 4, better SM utilization)
  - BN=128 helped for wide-N shapes (N=2304, N=3072)
  - allow_tf32=True breaks float32 primary; num_stages>1 breaks correctness with masked loads

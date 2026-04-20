**Final summary**

Kernel optimization results (48 experiments total):

| Kernel |  Shape               |  Baseline    |  Best |  Speedup |  Key change                |
|--------|----------------------|--------------|-------|----------|----------------------------|
| gemm_1 |  M=512 N=32000 K=768 |  39.1 TFLOPS |  39.6 |  1.01x   |  128×128 tiles + stages=4  |
| gemm_2 |  M=512 N=2048 K=768  |  31.7 TFLOPS |  34.6 |  1.09x   |  tf32 + swizzle + hints    |
| gemm_3 |  M=512 N=32000 K=768 |  31.9 TFLOPS |  40.4 |  1.27x   |  Full validated pack       |
| gemm_4 |  M=512 N=768 K=2048  |  29.4 TFLOPS |  31.5 |  1.07x   |  2D grid + stages=4 + tf32 |
| gemm_5 |  M=512 N=768 K=768   |  25.6 TFLOPS |  26.9 |  1.05x   |  2D grid + stages=5 + tf32 |

End-to-end: 
  - Correctness PASS (max_abs_err 3.28e-06), latency 22.0 ms vs 20.2 ms reference → 0.92x.
  - The Triton dispatch overhead exceeds the kernel-level gains when cuBLAS is already handling these shapes well.
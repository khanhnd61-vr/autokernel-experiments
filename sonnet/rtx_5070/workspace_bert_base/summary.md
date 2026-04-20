**AutoKernel BERT-base Optimization — Final Report**

**Model**: BertModel (109.5M params), input (8, 512), fp16 

**GPU**: RTX 5070 Laptop (57 TFLOPS FP16, 36 SMs) 

**Branch**: autokernel/apr17-bert-base 



**Phase B: Kernel Optimization Results**


|        Kernel       |        Shape        |   Baseline  |    Best     | Speedup | % Peak | Key Technique                      |
|---------------------|---------------------|-------------|-------------|---------|--------|------------------------------------|
| gemm_1 (26.1%)      | 4096×768×768        | 35.9 TFLOPS | 38.7 TFLOPS | +7.7%   | 67.9%  | BLOCK 128×64 (384 CTAs)            |
| gemm_4 (13.3%)      | 4096×3072×768       | 33.6 TFLOPS | 41.5 TFLOPS | +23.7%  | 72.9%  | BLOCK 128×128 + L2 swizzle         |
| flash_attn_6 (4.4%) | B=8,H=12,S=512,D=64 | 37.8 TFLOPS | 52.4 TFLOPS | +38.8%  | 92.0%  | fp16 QK tensor cores + BLOCK_N=32  |
| flash_attn_7 (4.4%) | same                | 37.8 TFLOPS | 54.0 TFLOPS | +42.9%  | 94.8%  | same config                        |
| fused_mlp_8 (2.9%)  | 4096×768×3072       | 31.5 TFLOPS | 32.8 TFLOPS | +4.2%   | 57.7%  | BLOCK 128×64 + swizzle             |

**Phase C: End-to-End Verification**

|       Metric       |            Value          |
|--------------------|---------------------------|
| Original latency   | 138.4 ms                  |
| Optimized latency  | 123.8 ms                  |
| End-to-end speedup | 1.12x                     |
| Correctness        | PASS (max_err = 1.25e-06) |

**Key insight:** 
The flash_attention fp16 QK tensor cores optimization (removing the explicit .to(tl.float32) cast before tl.dot) was the biggest single win — 33.8% throughput gain, reaching 92%+ of hardware peak.
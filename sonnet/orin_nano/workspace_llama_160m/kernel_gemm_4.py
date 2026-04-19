"""
AutoKernel -- Extracted kernel from model profiling.
Op type: gemm
Rank: 4 (6.5% of GPU time)
Model shape: M=512, N=768, K=768

This kernel was extracted from profiling models/llama_7b.py.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "gemm"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'M': 512, 'N': 768, 'K': 768}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'M': 512, 'N': 768, 'K': 768}),
    # Also test nearby sizes for robustness
    ("model_half", {'M': 256, 'N': 384, 'K': 384}),
    ("model_double", {'M': 1024, 'N': 1536, 'K': 1536}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.02, 'rtol': 0.02}, 'float32': {'atol': 0.0001, 'rtol': 0.0001}}


def FLOPS_FN(s):
    return 2 * s["M"] * s["N"] * s["K"]


def BYTES_FN(s, dt_bytes):
    return (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/gemm.py)
# ======================================================================

import torch
import triton
import triton.language as tl

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32


@triton.jit
def triton_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def gemm_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper function to call the triton kernel.

    Args:
        data: tuple of (A, B)
            A: (M, K)
            B: (K, N)
    Returns:
        Output tensor of shape (M, N)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    triton_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py.
        Must match reference.matmul_ref signature.
    """
    assert A.is_cuda and B.is_cuda
    return gemm_kernel(A, B)
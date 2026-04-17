"""
AutoKernel -- Extracted kernel from model profiling.
Op type: gemm
Rank: 3 (7.3% of GPU time)
Model shape: M=512, N=32000, K=768

This kernel was extracted from profiling models/llama_7b.py.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "gemm"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'M': 512, 'N': 32000, 'K': 768}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'M': 512, 'N': 32000, 'K': 768}),
    # Also test nearby sizes for robustness
    ("model_half", {'M': 256, 'N': 16000, 'K': 384}),
    ("model_double", {'M': 1024, 'N': 64000, 'K': 1536}),
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

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
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
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    M = tl.multiple_of(M, BLOCK_SIZE_M)
    N = tl.multiple_of(N, BLOCK_SIZE_N)
    K = tl.multiple_of(K, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)

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

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    triton_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_M=4,
        num_stages=4,
    )
    return C


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py.
        Must match reference.matmul_ref signature.
    """
    assert A.is_cuda and B.is_cuda
    return gemm_kernel(A, B)
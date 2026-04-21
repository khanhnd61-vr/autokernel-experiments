"""
Fused RMSNorm for decode-time (M=1, fp16).

Naive PyTorch RMSNorm (4 kernel launches):
    var = x.pow(2).mean(-1, keepdim=True)     # reduce
    x = x * torch.rsqrt(var + eps)             # elementwise x2 + rsqrt
    return weight * x                          # elementwise

Fused (1 launch):
    one Triton kernel: load x, square+sum, rsqrt, multiply by weight, store y

For Qwen3-VL decode (dim=4096) this fuses 4 kernels per norm × 73 norms/token
into 73 launches — saving ~3× the raw kernel-launch overhead plus register-level
fusion of the rsqrt/multiply chain.

Universal: works for any RMSNorm (LLaMA, Qwen, Mistral, Gemma, Yi, ...).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr,           # [M, D] fp16 (contiguous)
    w_ptr,           # [D]     fp16
    y_ptr,           # [M, D] fp16
    stride_xm,
    M, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x_row_ptr = x_ptr + row * stride_xm + offs
    x = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)

    # Variance = mean(x^2)
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(y_ptr + row * stride_xm + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    x:      [..., D]  fp16 contiguous
    weight: [D]       fp16
    Returns y: same shape as x
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    x2 = x.reshape(-1, D)
    M = x2.shape[0]

    # BLOCK_D must be pow-of-2 >= D (single-block reduction along D)
    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_D <= 1024:
        num_warps = 4
    elif BLOCK_D <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    y = torch.empty_like(x2)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x2, weight, y,
        x2.stride(0),
        M, D, eps,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return y.reshape(orig_shape)

"""
Fused residual-add + RMSNorm for pre-norm transformers.

Pre-norm pattern in every modern decoder LLM:
    x = x + attn_out                       # residual add (1 elementwise launch)
    h = rmsnorm(x, ffn_norm.weight, eps)   # 1 fused (or 4 eager) launches

Fused (1 launch):
    x_new, h = fused_add_rmsnorm(x, attn_out, weight, eps)

Why this fusion is safe (unlike fused_norm_qkv):
  Each row is owned by a SINGLE program — the rstd reduction is local to that
  program, so there is NO redundant work across programs.  Compared to a plain
  RMSNorm kernel, this kernel just adds a 1-fma elementwise sum into the existing
  load → essentially free GPU work.

Returns:
  x_new:  x + delta (also written back to HBM for the next residual)
  normed: rmsnorm(x_new, weight, eps)

Universal: every pre-norm decoder (LLaMA, Qwen, Mistral, Gemma, Yi, Phi, DeepSeek).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_fwd_kernel(
    x_ptr,           # [M, D] fp16 — residual stream (read & written)
    delta_ptr,       # [M, D] fp16 — to be added
    w_ptr,           # [D]    fp16 — RMSNorm weight
    out_ptr,         # [M, D] fp16 — normalized output
    stride_m,
    M, D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(delta_ptr + row * stride_m + offs, mask=mask, other=0.0).to(tl.float32)

    s = x + d  # new residual

    # Write back updated residual (next residual-add will read this)
    tl.store(x_ptr + row * stride_m + offs, s.to(x_ptr.dtype.element_ty), mask=mask)

    # RMSNorm on the new residual
    var = tl.sum(s * s, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = s * rstd * w

    tl.store(out_ptr + row * stride_m + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor,         # [..., D] fp16  contiguous (modified in place)
    delta: torch.Tensor,     # [..., D] fp16  contiguous, same shape
    weight: torch.Tensor,    # [D]      fp16
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    In-place adds delta into x and returns the normalized output.
    Returns: normed [..., D]
    """
    assert x.shape == delta.shape
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    d2 = delta.reshape(-1, D)
    M = x2.shape[0]

    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_D <= 1024:
        num_warps = 4
    elif BLOCK_D <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    out = torch.empty_like(x2)
    grid = (M,)
    add_rmsnorm_fwd_kernel[grid](
        x2, d2, weight, out,
        x2.stride(0),
        M, D, eps,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)

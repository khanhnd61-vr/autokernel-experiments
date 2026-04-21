"""
Fused RMSNorm + QKV projection for decode-time (M=1, fp16) pre-norm LLMs.

Standard pre-norm path (5 kernel launches):
    h = rmsnorm(x, w_norm, eps)   # 1 fused kernel (ours) or 4 eager kernels
    q = F.linear(h, Wq)           # cuBLAS
    k = F.linear(h, Wk)           # cuBLAS
    v = F.linear(h, Wv)           # cuBLAS

Fused (1 kernel):
    q, k, v = fused_norm_qkv(x, w_norm, Wq, Wk, Wv, eps)

Benefits:
  * Normalized x never materializes in HBM — stays in registers / L2
  * 2 kernel launches → 1 (RMSNorm + QKV-fused both go away)
  * x and w_norm are loaded once per program (vs twice: once for RMSNorm, once for matmul)

Universal: every pre-norm decoder (LLaMA, Qwen, Mistral, Gemma, Yi, Phi, DeepSeek).

Kernel structure (one program owns BLOCK_N rows in one of {Wq, Wk, Wv}):
    Phase 1: stream x in BLOCK_K chunks, accumulate sum(x²) to get rstd
    Phase 2: stream x + w_norm + W in BLOCK_K chunks, multiply x_raw*rstd*w_norm
             on-the-fly, tl.dot against W with M-pad tensor cores.

Redundant work cost: every program does Phase 1 over full K.  K=4096, BLOCK_K=128
→ 32 cheap vector-load/square-reduce iterations per program — negligible next to
the matmul it does in Phase 2.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_norm_qkv_kernel(
    x_ptr,                           # [K]      fp16
    wn_ptr,                          # [K]      fp16 RMSNorm weight
    Wq_ptr, Wk_ptr, Wv_ptr,          # weights
    out_ptr,                         # [N_q + 2*N_kv]
    K, N_q, N_kv,
    eps,
    stride_wqn, stride_wqk,
    stride_wkn, stride_wkk,
    stride_wvn, stride_wvk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n_out = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    N_total = N_q + 2 * N_kv
    out_mask = offs_n_out < N_total

    # ----- Pick matrix (Q/K/V) for this program -----
    pid_start = pid * BLOCK_N
    if pid_start < N_q:
        local_n = offs_n_out
        n_mask = local_n < N_q
        w_base = Wq_ptr
        sn, sk = stride_wqn, stride_wqk
    elif pid_start < N_q + N_kv:
        local_n = offs_n_out - N_q
        n_mask = local_n < N_kv
        w_base = Wk_ptr
        sn, sk = stride_wkn, stride_wkk
    else:
        local_n = offs_n_out - (N_q + N_kv)
        n_mask = local_n < N_kv
        w_base = Wv_ptr
        sn, sk = stride_wvn, stride_wvk

    # ======================================================================
    # Phase 1: compute rstd = rsqrt(mean(x²) + eps)
    # ======================================================================
    sum_sq = tl.zeros((1,), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            xv = tl.load(x_ptr + k * BLOCK_K + offs_k).to(tl.float32)
        else:
            k_mask = k * BLOCK_K + offs_k < K
            xv = tl.load(x_ptr + k * BLOCK_K + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(xv * xv, axis=0)

    rstd = 1.0 / tl.sqrt(sum_sq / K + eps)
    rstd_scalar = tl.sum(rstd, axis=0)  # collapse (1,) → scalar

    # ======================================================================
    # Phase 2: streaming matmul with x_norm = x * rstd * w_norm
    # ======================================================================
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    w_ptrs = w_base + local_n[None, :] * sn + offs_k[:, None] * sk

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            xv_raw = tl.load(x_ptr + k * BLOCK_K + offs_k)
            wn = tl.load(wn_ptr + k * BLOCK_K + offs_k)
            wv = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_mask = k * BLOCK_K + offs_k < K
            xv_raw = tl.load(x_ptr + k * BLOCK_K + offs_k, mask=k_mask, other=0.0)
            wn = tl.load(wn_ptr + k * BLOCK_K + offs_k, mask=k_mask, other=0.0)
            wv = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Fused normalize: x_norm = x * rstd * w_norm
        x_norm = (xv_raw.to(tl.float32) * rstd_scalar * wn.to(tl.float32)).to(xv_raw.dtype)

        # M-pad → tensor-core dot
        x_bcast = tl.zeros((BLOCK_M, BLOCK_K), dtype=x_norm.dtype)
        x_bcast = tl.where(offs_m[:, None] == 0, x_norm[None, :], x_bcast)
        acc += tl.dot(x_bcast, wv, allow_tf32=True)

        w_ptrs += BLOCK_K * sk

    y = tl.sum(tl.where(offs_m[:, None] == 0, acc, 0.0), axis=0)
    tl.store(out_ptr + offs_n_out, y.to(out_ptr.dtype.element_ty), mask=out_mask)


def fused_norm_qkv(
    x: torch.Tensor,                 # [K]       fp16
    w_norm: torch.Tensor,            # [K]       fp16 RMSNorm weight
    Wq: torch.Tensor,                # [N_q, K]  fp16
    Wk: torch.Tensor,                # [N_kv, K] fp16
    Wv: torch.Tensor,                # [N_kv, K] fp16
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Returns concatenated qkv: [N_q + 2*N_kv] fp16
    """
    K = x.shape[0]
    N_q = Wq.shape[0]
    N_kv = Wk.shape[0]

    # Tuned from fused_qkv (same shape family)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 128
    num_warps, num_stages = 2, 4

    assert N_q % BLOCK_N == 0 and N_kv % BLOCK_N == 0

    N_total = N_q + 2 * N_kv
    out = torch.empty((N_total,), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N_total, BLOCK_N),)
    fused_norm_qkv_kernel[grid](
        x, w_norm, Wq, Wk, Wv, out,
        K, N_q, N_kv, eps,
        Wq.stride(0), Wq.stride(1),
        Wk.stride(0), Wk.stride(1),
        Wv.stride(0), Wv.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        EVEN_K=(K % BLOCK_K == 0),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out

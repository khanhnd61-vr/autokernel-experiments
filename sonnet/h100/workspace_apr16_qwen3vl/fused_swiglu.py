"""
Fused SwiGLU for Qwen3-VL decode (M=1, fp16).

Standard path (4 kernels per FFN layer):
    gate = F.linear(x, Wg)      # GEMV
    up   = F.linear(x, Wu)      # GEMV
    mid  = F.silu(gate) * up    # elementwise
    out  = F.linear(mid, Wd)    # GEMV

Fused path (2 kernels per FFN layer):
    mid  = fused_gate_up_silu(x, Wg, Wu)    kernel 1: 2 GEMVs + silu*mul epilogue
    out  = gemv_row_reduce(mid, Wd)          kernel 2: row-dot-product GEMV

Memory traffic is identical either way (weight matrices dominate).
The win is kernel-launch overhead: 4 launches → 2 saves ~540 µs / token
across 36 layers, and the silu*mul kernel is absorbed for free.

Layout notes:
  fused_gate_up_silu: Wg, Wu are [N_mid, K_in] in nn.Linear convention.
      M-pad trick (pad M=1 → BLOCK_M=16) + tl.dot → tensor cores.
      Access: wg[k, n] with k as first dim, n as second; n-axis has stride K_in
      → non-coalesced but L2 / prefetch compensates (matches cuBLAS speed).

  gemv_row_reduce: Wd is [N_out, K_mid] in nn.Linear convention.
      Each program owns one output row: reads Wd[n, :] which is CONTIGUOUS in K.
      No M-pad / tensor cores — pure dot product, max L1 bandwidth.
      Matches cuBLAS on this kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# ======================================================================
# Kernel 1: fused gate + up + silu*mul → mid
# ======================================================================
@triton.jit
def fused_gate_up_silu_kernel(
    x_ptr,                         # [K]
    Wg_ptr,                        # [N_mid, K]  nn.Linear layout
    Wu_ptr,                        # [N_mid, K]
    mid_ptr,                       # [N_mid] output
    K, N_mid,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N_mid

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_k
    # W[offs_n, offs_k] for nn.Linear layout [N, K]: stride_wn=K, stride_wk=1
    wg_ptrs = Wg_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
    wu_ptrs = Wu_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            xv = tl.load(x_ptrs)
            wg = tl.load(wg_ptrs, mask=n_mask[None, :], other=0.0)
            wu = tl.load(wu_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            k_mask = offs_k < k_remaining
            xv = tl.load(x_ptrs, mask=k_mask, other=0.0)
            wg = tl.load(wg_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            wu = tl.load(wu_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # M-pad to 16 → tensor-core dot
        x_bcast = tl.zeros((BLOCK_M, BLOCK_K), dtype=xv.dtype)
        x_bcast = tl.where(offs_m[:, None] == 0, xv[None, :], x_bcast)

        acc_g += tl.dot(x_bcast, wg, allow_tf32=True)
        acc_u += tl.dot(x_bcast, wu, allow_tf32=True)

        x_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    # Row 0 only (rest are M-pad zeros)
    g0 = tl.sum(tl.where(offs_m[:, None] == 0, acc_g, 0.0), axis=0)
    u0 = tl.sum(tl.where(offs_m[:, None] == 0, acc_u, 0.0), axis=0)

    # SwiGLU epilogue: silu(gate) * up
    sig_g = 1.0 / (1.0 + tl.exp(-g0))
    mid = g0 * sig_g * u0

    tl.store(mid_ptr + offs_n, mid.to(mid_ptr.dtype.element_ty), mask=n_mask)


def fused_gate_up_silu(
    x: torch.Tensor, Wg: torch.Tensor, Wu: torch.Tensor
) -> torch.Tensor:
    """
    x:  [K]       fp16
    Wg: [N_mid,K] fp16  nn.Linear layout
    Wu: [N_mid,K] fp16  nn.Linear layout
    Returns mid: [N_mid] fp16  (= silu(Wg@x) * (Wu@x))
    """
    K = x.shape[0]
    N_mid = Wg.shape[0]

    mid = torch.empty((N_mid,), device=x.device, dtype=x.dtype)

    # Tuned on H100: BN=64, BK=256, 4 warps, 3 stages → 1.009× cuBLAS
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 256
    num_warps, num_stages = 4, 3

    grid = (triton.cdiv(N_mid, BLOCK_N),)
    fused_gate_up_silu_kernel[grid](
        x, Wg, Wu, mid,
        K, N_mid,
        Wg.stride(0), Wg.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        EVEN_K=(K % BLOCK_K == 0),
        num_warps=num_warps, num_stages=num_stages,
    )
    return mid


# ======================================================================
# Kernel 2: down-projection GEMV — row-dot-product style
#
# Wd is [N_out, K_mid] nn.Linear layout.
# One program owns BLOCK_ROWS output rows.
# Each row's K-strip Wd[n, :] is contiguous → coalesced loads.
# Reduction uses scalar fp32 accumulator (no tensor cores needed).
# ======================================================================
@triton.jit
def gemv_row_kernel(
    mid_ptr,                       # [K_mid] fp16 input
    Wd_ptr,                        # [N_out, K_mid] fp16 nn.Linear layout
    out_ptr,                       # [N_out] fp16 output
    K_mid, N_out,
    stride_wn, stride_wk,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_base = pid * BLOCK_ROWS
    rows = row_base + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < N_out

    # acc[r] = dot(Wd[rows[r], :], mid[:])
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    mid_ptrs = mid_ptr + offs_k

    for k in range(0, tl.cdiv(K_mid, BLOCK_K)):
        if EVEN_K:
            xv = tl.load(mid_ptrs)                    # [BLOCK_K]
        else:
            k_mask = offs_k < K_mid - k * BLOCK_K
            xv = tl.load(mid_ptrs, mask=k_mask, other=0.0)

        # Load BLOCK_ROWS rows × BLOCK_K cols of Wd (contiguous in k-direction)
        # w[r, kk] = Wd[rows[r], offs_k[kk]]  address = rows[r]*stride_wn + offs_k[kk]*stride_wk
        w_ptrs = Wd_ptr + rows[:, None] * stride_wn + offs_k[None, :] * stride_wk
        if EVEN_K:
            wv = tl.load(w_ptrs, mask=row_mask[:, None], other=0.0)  # [BLOCK_ROWS, BLOCK_K]
        else:
            k_mask = offs_k < K_mid - k * BLOCK_K
            wv = tl.load(w_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Dot product: acc[r] += sum_k wv[r, k] * xv[k]
        acc += tl.sum(wv.to(tl.float32) * xv[None, :].to(tl.float32), axis=1)

        mid_ptrs += BLOCK_K
        offs_k += BLOCK_K

    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + rows, out, mask=row_mask)


def gemv_row_reduce(mid: torch.Tensor, Wd: torch.Tensor) -> torch.Tensor:
    """
    mid: [K_mid]    fp16
    Wd:  [N_out, K_mid] fp16 nn.Linear layout
    Returns: [N_out] fp16
    """
    K_mid = mid.shape[0]
    N_out = Wd.shape[0]

    out = torch.empty((N_out,), device=mid.device, dtype=mid.dtype)

    # One program per BLOCK_ROWS output rows.
    # N_out=4096: BLOCK_ROWS=16 → 256 programs (fills SMs well).
    # Tuned on H100: BR=16, BK=256, 16 warps, 2 stages → 1.081× cuBLAS
    BLOCK_ROWS = 16
    BLOCK_K = 256

    grid = (triton.cdiv(N_out, BLOCK_ROWS),)
    gemv_row_kernel[grid](
        mid, Wd, out,
        K_mid, N_out,
        Wd.stride(0), Wd.stride(1),
        BLOCK_ROWS=BLOCK_ROWS, BLOCK_K=BLOCK_K,
        EVEN_K=(K_mid % BLOCK_K == 0),
        num_warps=16, num_stages=2,
    )
    return out


# ======================================================================
# Public API: full fused FFN, decode-time
# ======================================================================
def fused_swiglu_ffn(
    x: torch.Tensor,                   # [K_in=4096]
    Wg: torch.Tensor,                  # [N_mid=12288, K_in]  nn.Linear
    Wu: torch.Tensor,                  # [N_mid, K_in]        nn.Linear
    Wd: torch.Tensor,                  # [N_out=4096, N_mid]  nn.Linear
) -> torch.Tensor:
    """Fused decode-time SwiGLU FFN.  x → out[N_out] in 2 kernel launches."""
    mid = fused_gate_up_silu(x, Wg, Wu)
    return gemv_row_reduce(mid, Wd)

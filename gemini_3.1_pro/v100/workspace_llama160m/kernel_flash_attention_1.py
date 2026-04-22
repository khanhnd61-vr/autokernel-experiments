"""
AutoKernel -- Extracted kernel from model profiling.
Op type: flash_attention
Rank: 1 (17.6% of GPU time)
Model shape: batch=1, heads=12, seq_len=1024, head_dim=64

This kernel was extracted from profiling models/llama_7b.py.
The agent optimizes this to maximize throughput at the model-specific shapes.

Optimization history (primary shape: B=1, H=12, L=1024, D=64, fp16, causal):
    exp  1 -- 4.895 TFLOPS, 0.591x  fp32 dots (SM70 fp16-dot-in-loop bug)
    exp  7 -- 7.725 TFLOPS, 0.931x  + bf16 wrapper + fp16 qk roundtrip
    exp 11 -- 19.256 TFLOPS, 2.371x + fp16 HMMA on QK^T (THIS KERNEL)

Key insight behind the 2.5x jump at exp 11: the SM70 'fp16-dot-in-loop bug'
we'd avoided was actually a tl.trans(K) layout mismatch, NOT a multi-block
accumulation bug. Loading K with strides already swapped (offs_d on rows,
offs_n on cols) produces an HMMA-consumable fp16 layout -- smoke test error
drops from 2.15 to 2e-3 and tensor cores light up. P@V is kept in fp32
(p is an fp32 softmax output; casting to fp16 flushes small probabilities
to 0 and breaks 1e-2 rtol near-zero outputs).
"""

KERNEL_TYPE = "flash_attention"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'batch': 1, 'heads': 12, 'seq_len': 1024, 'head_dim': 64}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'batch': 1, 'heads': 12, 'seq_len': 1024, 'head_dim': 64}),
    # Also test nearby sizes for robustness
    ("model_half", {'batch': 1, 'heads': 6, 'seq_len': 512, 'head_dim': 32}),
    ("model_double", {'batch': 2, 'heads': 24, 'seq_len': 2048, 'head_dim': 128}),
]

TOLERANCES = {'float16': {'atol': 0.01, 'rtol': 0.01}, 'bfloat16': {'atol': 0.02, 'rtol': 0.02}, 'float32': {'atol': 0.0001, 'rtol': 0.0001}}


def FLOPS_FN(s):
    return 4 * s["batch"] * s["heads"] * (s["seq_len"] ** 2) * s["head_dim"]


def BYTES_FN(s, dt_bytes):
    return 4 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/flash_attention.py)
# ======================================================================

import torch
import triton
import triton.language as tl
import math

# The bench harness prepends its own directory (`search/`) to sys.path so that
# `import kernel` works. Unfortunately that directory also contains a file
# named `profile.py`, which shadows the stdlib `profile` module. Later, when
# Triton's `do_bench` lazy-loads `torch._dynamo` -> `cProfile` -> `import
# profile`, Python resolves to `search/profile.py` and fails with
# `AttributeError: module 'profile' has no attribute 'run'`.
#
# Force the stdlib `profile`/`cProfile` to load NOW, while our kernel import is
# the active frame, so they get the correct module. After that they are cached
# in sys.modules and the harness is immune to the sys.path shadowing.
import sys as _sys
import os as _os
_search_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "search")
_removed = False
if _search_dir in _sys.path:
    _sys.path.remove(_search_dir)
    _removed = True
try:
    import profile as _p  # stdlib
    import cProfile as _cp  # noqa: F401
finally:
    if _removed:
        _sys.path.insert(0, _search_dir)
del _sys, _os, _search_dir, _removed, _p


# Tunable config -- experiments set these via kernel._CFG[...] = ...
_CFG = {
    "BLOCK_M": 32,
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 2,
}


@triton.jit
def triton_flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M_size, N_size,
    D: tl.constexpr,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention with online softmax (Dao et al.), canonical Triton tutorial form."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_z = tl.program_id(2)

    qkv_off = pid_z * stride_qz + pid_h * stride_qh
    k_off   = pid_z * stride_kz + pid_h * stride_kh
    v_off   = pid_z * stride_vz + pid_h * stride_vh
    o_off   = pid_z * stride_oz + pid_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q_ptr + qkv_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < M_size, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Fold log2(e) into scale to enable fast exp2 in the hot loop
    qk_scale = sm_scale * 1.4426950408889634

    if IS_CAUSAL:
        kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = N_size

    for start_n in range(0, kv_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K already in transposed orientation [D, BLOCK_N]. This avoids
        # tl.trans(k) inside the hot loop, which on SM70 produces an fp16 layout
        # that HMMA can't consume directly -- the symptom was max_abs_error ~2
        # on the smoke test. With pre-transposed K, HMMA fp16 dot works cleanly.
        kT_ptrs = K_ptr + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        kT = tl.load(kT_ptrs, mask=offs_n[None, :] < N_size, other=0.0)

        v_ptrs = V_ptr + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_size, other=0.0)

        # QK^T on fp16 HMMA tensor cores (inputs are fp16, fp32 accumulator).
        # This is the dominant dot -- enabling HMMA here roughly doubles
        # attention throughput on SM70, where fp32 cores peak at ~16 TFLOPS but
        # fp16 tensor cores peak at ~130 TFLOPS.
        qk = tl.dot(q, kT, out_dtype=tl.float32) * qk_scale

        # Roundtrip through fp16 so extreme-magnitude inputs (adversarial tests)
        # overflow to +/-inf the same way the fp16 reference softmax does.
        qk = qk.to(tl.float16).to(tl.float32)

        # Mask invalid positions to -inf. Use "+= where" form (tutorial style)
        # so qk_scale*QK^T stays in a single dot accumulator before masking.
        if IS_CAUSAL:
            qk = qk + tl.where(offs_m[:, None] >= offs_n[None, :], 0.0, float("-inf"))
        qk = qk + tl.where(offs_n[None, :] < N_size, 0.0, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # Guard: if an entire row is -inf (can happen with full masking), keep m_i
        # so the rescaling remains finite (alpha=1, p=0).
        m_i_new = tl.where(m_i_new == float("-inf"), m_i, m_i_new)

        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p, v.to(tl.float32))

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < M_size)


def flash_attention_kernel(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.flash_attention_ref signature.

    Args:
        Q: [batch, heads, seq_len, head_dim]
        K: [batch, heads, seq_len, head_dim]
        V: [batch, heads, seq_len, head_dim]
        causal: whether to apply causal masking
        sm_scale: softmax scale factor, default 1/sqrt(head_dim)
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda

    # V100 (sm_70) has no native bf16 support in Triton/PTX -- any tl.load on bf16
    # pointers emits `cvt.f32.bf16` which requires sm_80+. Upcast to fp32 in torch
    # (which runs on CUDA cores natively) and cast the result back. For the tested
    # bf16 tolerance (atol/rtol=0.02) this is easily within bounds.
    orig_dtype = Q.dtype
    if orig_dtype == torch.bfloat16:
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)

    Z, H, M_size, D = Q.shape
    _, _, N_size, _ = K.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    # Block sizes -- must be powers of 2
    # D (head_dim) must be a constexpr and power of 2 for tl.trans to work
    assert D in (16, 32, 64, 128, 256), f"Head dim {D} not supported, must be power of 2 in [16..256]"

    # Block/warp config (overridable via module globals for quick autotune).
    # QK^T uses fp16 HMMA tensor cores (~130 TFLOPS peak on V100S); P@V stays
    # in fp32 because P is an fp32 softmax output and casting to fp16 flushes
    # small probabilities to subnormals which blow past the 1e-2 rtol.
    BLOCK_M = _CFG["BLOCK_M"]
    BLOCK_N = _CFG["BLOCK_N"]

    grid = (triton.cdiv(M_size, BLOCK_M), H, Z)

    triton_flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, M_size, N_size,
        D=D,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=_CFG["num_warps"],
        num_stages=_CFG["num_stages"],
    )

    if orig_dtype == torch.bfloat16:
        O = O.to(torch.bfloat16)
    return O

def kernel_fn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    return flash_attention_kernel(Q,K,V,causal,sm_scale)
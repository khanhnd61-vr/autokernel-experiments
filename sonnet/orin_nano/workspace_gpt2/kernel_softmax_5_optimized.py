"""
AutoKernel -- Extracted kernel from model profiling.
Op type: softmax
Rank: 5 (6.5% of GPU time)
Model shape: rows=4096, cols=4096

This kernel was extracted from profiling models/gpt2.py.
The agent optimizes this to maximize throughput at the model-specific shapes.
"""

KERNEL_TYPE = "softmax"

# Model-specific shapes (the shapes that matter for THIS model)
MODEL_SHAPES = {'rows': 4096, 'cols': 4096}

# Benchmark config (self-describing -- bench.py can load this dynamically)
TEST_SIZES = [
    ("model_primary", {'rows': 4096, 'cols': 4096}),
    # Also test nearby sizes for robustness
    ("model_half", {'rows': 2048, 'cols': 2048}),
    ("model_double", {'rows': 8192, 'cols': 8192}),
]

TOLERANCES = {'float16': {'atol': 0.001, 'rtol': 0.001}, 'bfloat16': {'atol': 0.002, 'rtol': 0.002}, 'float32': {'atol': 1e-05, 'rtol': 1e-05}}


def FLOPS_FN(s):
    return 5 * s["rows"] * s["cols"]


def BYTES_FN(s, dt_bytes):
    return 2 * s["rows"] * s["cols"] * dt_bytes


# ======================================================================
# Triton kernel code (from kernels/softmax.py)
# ======================================================================

import torch
import triton
import triton.language as tl

NUM_WARPS = 4
# For small cols (fits in one block): single-load kernel, BLOCK_SIZE = next_pow2(n_cols)
# Max single-load: 8192 cols × 2 bytes / 128 threads = 128 bytes = 64 fp16 regs (safe)
_TRITON_MAX_SINGLE_COLS = 8192
# For large cols: loop-based 2-pass online softmax with this block size
_LOOP_BLOCK_SIZE = 1024


@triton.jit
def triton_softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-load row softmax for cols <= 4096."""
    row_idx = tl.program_id(0)
    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf"))
    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.exp(row)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(row_start_output + col_offsets, result, mask=mask)


@triton.jit
def triton_softmax_loop_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """2-pass online softmax for large cols (avoids register spill from huge BLOCK_SIZE)."""
    row_idx = tl.program_id(0)
    row_input = input_ptr + row_idx * stride_input_row
    row_output = output_ptr + row_idx * stride_output_row

    # Pass 1: compute global max and sum of exp via online softmax accumulation
    m = tl.full([1], float('-inf'), dtype=tl.float32)
    l = tl.zeros([1], dtype=tl.float32)

    for k in range(0, n_cols, BLOCK_SIZE):
        col_offs = k + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        x = tl.load(row_input + col_offs, mask=mask, other=float('-inf')).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0, keep_dims=True))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0, keep_dims=True)
        m = m_new

    # Pass 2: store normalized results
    for k in range(0, n_cols, BLOCK_SIZE):
        col_offs = k + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        x = tl.load(row_input + col_offs, mask=mask, other=float('-inf')).to(tl.float32)
        result = tl.exp(x - m) / l
        tl.store(row_output + col_offs, result.to(output_ptr.dtype.element_ty), mask=mask)


# On Orin (Jetson unified memory), NVML-backed allocator crashes for tensors >200MB.
# Process in chunks to keep each output allocation below this threshold.
_MAX_CHUNK_BYTES = 128 * 1024 * 1024  # 128 MB per chunk


def _run_softmax_rows(x_2d: torch.Tensor) -> torch.Tensor:
    """Run Triton softmax on a 2D (n_rows, n_cols) tensor."""
    n_rows, n_cols = x_2d.shape
    output = torch.empty_like(x_2d)
    grid = (n_rows,)
    if n_cols <= _TRITON_MAX_SINGLE_COLS:
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        triton_softmax_kernel[grid](
            x_2d, output, n_cols,
            x_2d.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
    else:
        triton_softmax_loop_kernel[grid](
            x_2d, output, n_cols,
            x_2d.stride(0), output.stride(0),
            BLOCK_SIZE=_LOOP_BLOCK_SIZE,
            num_warps=NUM_WARPS,
        )
    return output


def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    row_bytes = n_cols * x.element_size()
    chunk_rows = max(1, _MAX_CHUNK_BYTES // row_bytes)

    if chunk_rows >= n_rows:
        return _run_softmax_rows(x).view(orig_shape)

    # Chunk to avoid large allocations that crash Orin NVML allocator
    chunks = [_run_softmax_rows(x[i:i + chunk_rows]) for i in range(0, n_rows, chunk_rows)]
    return torch.cat(chunks, dim=0).view(orig_shape)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py.
        Must match reference.softmax_ref signature.
    """
    n_cols = x.shape[-1]
    # Dispatch rules:
    # - bfloat16/float32: tighter tolerances → PyTorch
    # - float16 cols < 256: Triton launch overhead dominates → PyTorch
    # - float16 cols in [256, 4096]: single-load Triton kernel
    # - float16 cols > 4096: 2-pass loop Triton kernel (avoids register spill)
    if x.dtype != torch.float16 or n_cols < 256:
        return torch.softmax(x, dim=-1)
    return softmax_kernel(x)
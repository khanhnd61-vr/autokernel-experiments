"""
AutoKernel -- Extracted kernel from model profiling.
Op type: softmax
Rank: 5 (6.1% of GPU time)
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


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel online softmax. One program per row."""
    row_idx = tl.program_id(0)

    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf"))

    # Numerically stable softmax: subtract max
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Exponentiate
    numerator = tl.exp(row)

    # Sum
    denominator = tl.sum(numerator, axis=0)

    # Divide
    result = numerator / denominator

    # Store
    tl.store(row_start_output + col_offsets, result, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.is_cuda

    # Flatten to 2D for row-parallel processing
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # Block size must be a power of 2 >= n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    softmax_kernel[grid](
        x, output,
        n_cols,
        x.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(orig_shape)

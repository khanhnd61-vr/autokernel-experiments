"""
CUDA C++ GEMV for Qwen3-VL lm_head decode (M=1, K=4096, N=151936).

Strategy: vectorized fp16x8 (uint4) loads, warp-reduced dot product per output.
  - Each block computes BLOCK_N output columns (one warp per column tile).
  - Each warp loads x[k] and W[k, n0:n0+32] in fp16x8 chunks.
  - Reduction across K via FMA, accumulator in fp32.
  - Final reduction via __shfl_xor_sync within the warp.

This is the canonical "memory-streaming GEMV" pattern.  We expect to match
cuBLAS's bandwidth (it does roughly the same thing) and prove the ceiling.
"""

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// One warp produces ONE output column.
// Blocks are sized so each block has WARPS_PER_BLOCK warps and produces that
// many output columns.  Each warp's 32 lanes share the K-reduction.

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARPS_PER_BLOCK * 32;
constexpr int VEC = 8;  // fp16x8 = uint4 (16 B)

__global__ void __launch_bounds__(THREADS)
gemv_lmhead_kernel(
    const __half* __restrict__ x,    // [K]
    const __half* __restrict__ W,    // [K, N] row-major
    __half* __restrict__ y,          // [N]
    int K, int N
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (n >= N) return;

    float acc = 0.f;

    // Each lane processes K/(32*VEC) chunks of 8 fp16 elements
    const int per_lane = (K + 32 * VEC - 1) / (32 * VEC);

    #pragma unroll 4
    for (int i = 0; i < per_lane; i++) {
        int k = (i * 32 + lane) * VEC;
        if (k + VEC <= K) {
            // Vectorized load: 8 fp16 = 16 bytes = uint4
            uint4 xv = *reinterpret_cast<const uint4*>(x + k);
            uint4 wv = *reinterpret_cast<const uint4*>(W + (size_t)k * N + n * 1);
            // Wait — W is row-major [K, N], so W[k, n] = W[k*N + n].
            // For VEC consecutive k's, the n offset is fixed → strided in memory.
            // We need a different layout.  Re-do below.
            // (This kernel is replaced by the corrected version.)
            (void)xv; (void)wv;
        }
    }

    // Placeholder; correct kernel below.
    if (lane == 0) y[n] = __float2half(acc);
}

// ====================================================================
// Corrected GEMV: each warp owns ONE output n; lanes coop on K-reduction.
// W[k, n] memory layout: contiguous in n, strided in k.
// For a fixed n, walking k means strided loads (stride=N).  Bad.
//
// Better layout: assign one warp to BLOCK_N=32 outputs, lane=k offset.
// Each lane reads x[k] (broadcast across n) and W[k, n0+lane] (stride 1 in n)
// and accumulates into a per-lane fp32 partial that maps to its n.
// Then we have to sum across lanes → wrong, lanes own different n's.
//
// Cleanest pattern: one warp owns 32 consecutive n's (BLOCK_N=32 per warp).
// Lane = output column offset within the warp's tile.
// All 32 lanes co-walk K, each lane accumulating into its own n.
// Loads: x[k] broadcast (one fp16 per k for all lanes via shfl), W[k, n] coalesced.
// ====================================================================

constexpr int BLOCK_N = 128;          // outputs per block
constexpr int WARPS   = 4;            // 32-wide warps
// Each warp owns BLOCK_N / WARPS = 32 outputs (lane = output offset)
constexpr int N_PER_WARP = BLOCK_N / WARPS;  // 32

__global__ void __launch_bounds__(THREADS)
gemv_kernel(
    const __half* __restrict__ x,    // [K]
    const __half* __restrict__ W,    // [K, N] row-major
    __half* __restrict__ y,          // [N]
    int K, int N
) {
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int n_base  = blockIdx.x * BLOCK_N + warp_id * N_PER_WARP;
    const int n       = n_base + lane;
    if (n >= N) return;

    float acc = 0.f;

    // Walk K — every lane in the warp reads its own W[k, n_base+lane],
    // and they co-load x[k] (single broadcast).
    // Load x in fp16 chunks of 8 via shared-memory broadcast.
    extern __shared__ __half x_smem[];

    // Cooperative load of x into shared memory (once per block)
    for (int k = threadIdx.x; k < K; k += THREADS) {
        x_smem[k] = x[k];
    }
    __syncthreads();

    // Co-walk K. Each lane reads W[k, n_base+lane] every step.
    // To get vectorization, we load x_smem in VEC=8 chunks per lane.
    #pragma unroll 1
    for (int k = 0; k < K; k += 8) {
        // Each lane loads its W[k+0..7, n] — 8 strided loads
        // (these can't be vectorized in n direction since each lane has different n)
        // but they CAN be vectorized in k direction: W[k..k+7, n] is contiguous? NO,
        // for fixed n, increasing k jumps by N (151936) each time.  So 8 strided loads.
        // The compiler may emit ld.global.nc.u16 (cached) for each.

        // x slice (broadcast — same value for all lanes within same k):
        // Load from shared memory.
        __half x0 = x_smem[k + 0];
        __half x1 = x_smem[k + 1];
        __half x2 = x_smem[k + 2];
        __half x3 = x_smem[k + 3];
        __half x4 = x_smem[k + 4];
        __half x5 = x_smem[k + 5];
        __half x6 = x_smem[k + 6];
        __half x7 = x_smem[k + 7];

        const __half* wp = W + (size_t)(k) * N + n;
        __half w0 = wp[0 * N];
        __half w1 = wp[1 * N];
        __half w2 = wp[2 * N];
        __half w3 = wp[3 * N];
        __half w4 = wp[4 * N];
        __half w5 = wp[5 * N];
        __half w6 = wp[6 * N];
        __half w7 = wp[7 * N];

        acc += __half2float(x0) * __half2float(w0);
        acc += __half2float(x1) * __half2float(w1);
        acc += __half2float(x2) * __half2float(w2);
        acc += __half2float(x3) * __half2float(w3);
        acc += __half2float(x4) * __half2float(w4);
        acc += __half2float(x5) * __half2float(w5);
        acc += __half2float(x6) * __half2float(w6);
        acc += __half2float(x7) * __half2float(w7);
    }

    y[n] = __float2half(acc);
}

torch::Tensor gemv_cuda(torch::Tensor x, torch::Tensor W) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(W.dtype() == torch::kFloat16, "W must be fp16");
    TORCH_CHECK(x.dim() == 1 && W.dim() == 2, "x must be 1D, W must be 2D");
    int K = x.size(0);
    int N = W.size(1);
    TORCH_CHECK(W.size(0) == K, "shape mismatch");

    auto y = torch::empty({N}, x.options());

    int n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    size_t smem_bytes = K * sizeof(__half);

    gemv_kernel<<<n_blocks, THREADS, smem_bytes>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        K, N);

    return y;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "gemv_cuda", verbose=False)
    return _module


def gemv_cuda(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """x: [K] fp16; W: [K, N] fp16 row-major contiguous → [N] fp16."""
    return _get_module().gemv_cuda(x, W)

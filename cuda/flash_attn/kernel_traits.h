/*
 * kernel_traits.h — Compile-time kernel configuration.
 *
 * With WMMA, this is much simpler than the CuTe version — just block sizes,
 * thread counts, and shared memory math. No layout algebra needed.
 *
 * Block size choices (from Tri Dao, A100/SM80):
 *   hdim64:  BLOCK_M=128, BLOCK_N=64,  4 warps → 128 threads
 *   hdim128: BLOCK_M=128, BLOCK_N=64,  4 warps → 128 threads
 *
 * WMMA fragment size is fixed at 16×16×16 (fp16 inputs, fp32 accumulator).
 * We tile the BLOCK_M×BLOCK_N matmul into (BLOCK_M/16) × (BLOCK_N/16) WMMA tiles,
 * each accumulating over head_dim in chunks of 16.
 */

#pragma once

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits {
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;     // Q rows per CTA
    static constexpr int kBlockN = kBlockN_;     // KV columns per CTA (inner loop tile)
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    // WMMA tile dimensions (fixed by the API)
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;

    // Number of WMMA tiles needed to cover each block dimension
    static constexpr int kWmmaTilesM = kBlockM / WMMA_M;  // e.g. 128/16 = 8
    static constexpr int kWmmaTilesN = kBlockN / WMMA_N;  // e.g. 64/16 = 4
    static constexpr int kWmmaTilesK = kHeadDim / WMMA_K; // e.g. 64/16 = 4

    // Shared memory sizes (in bytes)
    // Q tile: BLOCK_M × head_dim × sizeof(half)
    // K tile: BLOCK_N × head_dim × sizeof(half) — ×2 for double buffer
    // V tile: BLOCK_N × head_dim × sizeof(half) — ×2 for double buffer
    static constexpr int kSmemQ = kBlockM * kHeadDim * sizeof(half);
    static constexpr int kSmemK = kBlockN * kHeadDim * sizeof(half);
    static constexpr int kSmemV = kBlockN * kHeadDim * sizeof(half);

    // Without double buffering (Phase 1):
    static constexpr int kSmemSize = kSmemQ + kSmemK + kSmemV;

    // With double buffering (Phase 2):
    // static constexpr int kSmemSize = kSmemQ + 2 * kSmemK + 2 * kSmemV;

    // Register pressure check:
    // O accumulator: BLOCK_M × head_dim / kNThreads floats per thread
    // For hdim128, BLOCK_M=128, 128 threads: 128 × 128 / 128 = 128 fp32 regs
    // Plus WMMA fragments, softmax state, loop vars → ~180-200 total
    // A100 max: 65536 / 128 threads = 512 regs/thread — plenty of headroom
};

// Concrete configs
using Traits_hdim64  = Flash_fwd_kernel_traits<64,  128, 64, 4>;
using Traits_hdim128 = Flash_fwd_kernel_traits<128, 128, 64, 4>;

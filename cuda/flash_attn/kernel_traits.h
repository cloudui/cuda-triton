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

    // Padded strides (must match flash_fwd_kernel.h) to break smem bank-conflict
    // alignment. fp16 buffers padded by 8 halves (16 bytes); fp32 buffers padded
    // by 4 floats (16 bytes). Both maintain the 16-byte alignment required by
    // ldmatrix / store_matrix_sync.
    static constexpr int kQStride     = kHeadDim + 8;   // halves
    static constexpr int kKVStride    = kHeadDim + 8;   // halves
    static constexpr int kScoreStride = kBlockN  + 4;   // floats
    static constexpr int kPStride     = kBlockN  + 8;   // halves
    static constexpr int kOStride     = kHeadDim + 4;   // floats

    // Shared memory sizes (in bytes), accounting for padded strides.
    // Q tile:    BLOCK_M × Q_STRIDE × sizeof(half)
    // K/V tile:  BLOCK_N × KV_STRIDE × sizeof(half) — reused (K then V, not alive together)
    // scores/O:  BLOCK_M × max(SCORE_STRIDE, O_STRIDE) × sizeof(float) — same buffer, two roles
    // P tile:    BLOCK_M × P_STRIDE × sizeof(half)
    static constexpr int kSmemQ      = kBlockM * kQStride     * sizeof(half);
    static constexpr int kSmemKV     = kBlockN * kKVStride    * sizeof(half);
    static constexpr int kSmemScores = kBlockM * kScoreStride * sizeof(float);
    static constexpr int kSmemO      = kBlockM * kOStride     * sizeof(float);
    static constexpr int kSmemP      = kBlockM * kPStride     * sizeof(half);

    // scores and O alias the same buffer, so we need max of the two
    static constexpr int kSmemScoresO = kSmemScores > kSmemO ? kSmemScores : kSmemO;

    // Without double buffering (Phase 1):
    static constexpr int kSmemSize = kSmemQ + kSmemKV + kSmemScoresO + kSmemP;

    // With double buffering (Phase 2):
    // static constexpr int kSmemSize = kSmemQ + 2 * kSmemKV + kSmemScoresO + kSmemP;

    // Register pressure check:
    // O accumulator: BLOCK_M × head_dim / kNThreads floats per thread
    // For hdim128, BLOCK_M=128, 128 threads: 128 × 128 / 128 = 128 fp32 regs
    // Plus WMMA fragments, softmax state, loop vars → ~180-200 total
    // A100 max: 65536 / 128 threads = 512 regs/thread — plenty of headroom

    // Smem totals with padding (for reference):
    //   hdim64:   18K + 9K  + 34K + 18K ≈  79 KB
    //   hdim128:  35K + 17K + 66K + 18K ≈ 136 KB  (still fits A100's 164 KB max)
};

// Concrete configs
using Traits_hdim64  = Flash_fwd_kernel_traits<64,  128, 64, 4>;
using Traits_hdim128 = Flash_fwd_kernel_traits<128, 128, 64, 4>;

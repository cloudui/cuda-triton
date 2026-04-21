/*
 * softmax.h — Online softmax for FlashAttention.
 *
 * Same algorithm as your Triton kernel:
 *   m_new = max(m_old, max(scores))
 *   correction = exp2(m_old - m_new)
 *   p = exp2(scores - m_new)
 *   l_new = l_old * correction + sum(p)
 *   o_new = o_old * correction + p @ V
 *
 * The main new thing in CUDA vs Triton: you need to manually reduce across
 * threads that share the same output row. In Triton, tl.max/tl.sum just work.
 * Here, scores from one Q row are spread across multiple threads after the
 * WMMA matmul, so you need warp shuffles to collect them.
 *
 * With WMMA 16×16×16 on fp16, the accumulator fragment (fp32) has 8 elements
 * per thread. The thread-to-element mapping within a warp is:
 *
 *   Each thread holds elements from 2 rows (non-contiguous).
 *   Threads {t, t+1, t+2, t+3} for t ∈ {0,4,8,...,28} share no rows.
 *   But across WMMA tiles covering BLOCK_N, the same row's scores end up
 *   in different WMMA accumulator fragments.
 *
 * Simpler approach for Phase 1: after the WMMA matmul, store the scores
 * from accumulators back to shared memory in row-major layout, then each
 * thread reads its own row and does softmax in registers. This is slower
 * (extra shared memory round-trip) but much easier to get right. Optimize
 * to warp-shuffle reductions in Phase 3.
 *
 * What you need to implement:
 *   - Thread-level m, l state (one per Q row this thread is responsible for)
 *   - row_max(): find max across a row (shared memory or warp shuffle)
 *   - row_sum(): sum across a row (shared memory or warp shuffle)
 *   - rescale_o(): multiply O accumulator by exp2(m_old - m_new)
 *   - finalize(): O = O / l after all KV tiles processed
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// ============================================================================
// Phase 1: shared-memory-based softmax (simpler, easier to debug)
// ============================================================================

// TODO: After WMMA computes S = Q @ K^T into accumulator fragments,
// store S to shared memory in row-major layout so each thread can
// read full rows for softmax.
//
// Workflow:
//   1. wmma::store_matrix_sync(smem_scores, acc_frag, BLOCK_N, mem_row_major)
//   2. __syncthreads()
//   3. Each thread handles (BLOCK_M / kNThreads) rows:
//      - Find row max
//      - exp2(score - max) for each element
//      - Sum the row
//      - Normalize (or defer normalization to final step)
//   4. Store P (attention weights) back to shared memory for P @ V matmul
//   5. wmma::load_matrix_sync(p_frag, smem_scores, BLOCK_N)

struct RowSoftmaxState {
  float m;
  float l;

  __device__ RowSoftmaxState() : m(-INFINITY), l(0.0f) {}

  __device__ void update(float *scores, int n_cols, float log2e_scale) {
    float row_max = -INFINITY;

    for (int i = 0; i < n_cols; i++) {
      row_max = fmaxf(row_max, scores[i]);
    }

    float m_new = fmaxf(m, row_max);
    float correction = exp2f(m - m_new);

    float row_sum = 0.0f;
    for (int i = 0; i < n_cols; i++) {
      scores[i] = exp2f(log2e_scale * (scores[i] - m_new));
      row_sum += scores[i];
    }

    m = m_new;
    l = correction * l + row_sum;
  }

  __device__ float get_correction(float m_old) { return exp2f(m_old - m); }
};

// ============================================================================
// Phase 3: warp-shuffle-based softmax (faster, no shared memory round-trip)
// ============================================================================

// TODO: Once Phase 1 works, optimize by doing the max/sum reductions
// directly on the WMMA accumulator fragments using __shfl_xor_sync.
// This avoids the store→sync→load round-trip through shared memory.
//
// The key challenge is figuring out which threads hold elements from
// the same row in the WMMA accumulator layout. For SM80 16×16×16:
//   - Thread t holds rows: (t / 4) and (t / 4 + 8)  [within the 16-row tile]
//   - Columns depend on which WMMA tile along BLOCK_N
//   - Threads (t % 4) within a group of 4 hold different columns of the same
//   row
//
// So to reduce across columns, you shuffle among threads {t, t^1, t^2, t^3}
// (XOR with 1 and 2).

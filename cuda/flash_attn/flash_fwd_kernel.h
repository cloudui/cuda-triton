/*
 * flash_fwd_kernel.h — FlashAttention-2 forward kernel using WMMA.
 *
 * One CTA handles BLOCK_M rows of Q against ALL of K/V for one (batch, head).
 * Uses nvcuda::wmma for the two matmuls (Q@K^T and P@V).
 *
 * Algorithm (same as your Triton kernel):
 *   1. Load Q tile (BLOCK_M × head_dim) into shared memory
 *   2. For each KV tile of BLOCK_N rows:
 *      a. Load K tile (BLOCK_N × head_dim) into shared memory
 *      b. Compute S = Q @ K^T via WMMA (BLOCK_M × BLOCK_N, fp32 accum)
 *      c. Apply causal mask
 *      d. Online softmax: update m, l, rescale O
 *      e. Load V tile (BLOCK_N × head_dim) into shared memory
 *      f. Compute O += P @ V via WMMA (BLOCK_M × head_dim, fp32 accum)
 *   3. Normalize O = O / l
 *   4. Write O to global memory
 *
 * Implementation phases:
 *
 *   PHASE 1 — Basic correctness (synchronous loads, smem softmax):
 *     [ ] Shared memory: allocate Q, K, V, scores tiles
 *     [ ] Load Q into shared memory (cooperative, synchronous)
 *     [ ] KV loop: load K, WMMA Q@K^T, store scores to smem
 *     [ ] Softmax via shared memory (store scores, read rows, compute)
 *     [ ] Convert P to fp16 in smem, WMMA P@V, accumulate O
 *     [ ] Rescale O accumulator when max changes
 *     [ ] Final O/l normalization, write to global
 *
 *   PHASE 2 — Async memory pipeline:
 *     [ ] Replace synchronous loads with cp.async (cp_async_16B)
 *     [ ] cp_async_commit + cp_async_wait around loads
 *     [ ] Double buffer K and V tiles in shared memory
 *     [ ] Prefetch next K tile while computing current tile
 *
 *   PHASE 3 — Performance:
 *     [ ] Causal early exit (skip KV tiles above diagonal)
 *     [ ] Warp-shuffle softmax (avoid smem round-trip for scores)
 *     [ ] Bank conflict reduction via smem padding or swizzling
 *     [ ] Tune BLOCK_M/BLOCK_N for occupancy
 *
 * WMMA matmul tiling:
 *   To compute S[BLOCK_M × BLOCK_N] = Q[BLOCK_M × D] @ K^T[D × BLOCK_N]:
 *   - Outer loops over WMMA tiles: i ∈ [0, BLOCK_M/16), j ∈ [0, BLOCK_N/16)
 *   - Inner K-loop: k ∈ [0, D/16)
 *   - Each iteration: load 16×16 fragment of Q and K, mma_sync, accumulate
 *
 *   Same pattern for O[BLOCK_M × D] += P[BLOCK_M × BLOCK_N] @ V[BLOCK_N × D]:
 *   - Outer loops: i ∈ [0, BLOCK_M/16), j ∈ [0, D/16)
 *   - Inner K-loop: k ∈ [0, BLOCK_N/16)
 *
 * Grid launch:
 *   grid.x = ceil(seqlen_q / BLOCK_M)
 *   grid.y = batch_size * num_heads
 *   block  = kNThreads (128 for 4 warps)
 *
 * Shared memory (hdim128, BLOCK_M=128, BLOCK_N=64, no double buffer):
 *   Q:      128 × 128 × 2 = 32 KB
 *   K:       64 × 128 × 2 = 16 KB
 *   V:       64 × 128 × 2 = 16 KB
 *   Scores: 128 ×  64 × 4 = 32 KB (fp32, for softmax)
 *   Total: ~96 KB → need cudaFuncSetAttribute for extended smem
 *
 *   For hdim64 it's half the Q/K/V sizes → ~64 KB.
 *   Phase 1 can reuse K's smem for V (they're not needed simultaneously)
 *   to save space: Q + K_or_V + Scores = 16 + 8 + 32 = 56 KB for hdim64.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "flash.h"
#include "kernel_traits.h"
#include "softmax.h"
#include "utils.h"

using namespace nvcuda;

template <typename Traits, bool Is_causal>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
  // Compile-time constants
  constexpr int BLOCK_M = Traits::kBlockM;
  constexpr int BLOCK_N = Traits::kBlockN;
  constexpr int HEAD_DIM = Traits::kHeadDim;
  constexpr int NTHREADS = Traits::kNThreads;

  // Thread / block indices
  const int m_block = blockIdx.x; // which chunk of Q rows
  const int bh_idx = blockIdx.y;  // which (batch, head)
  const int batch_idx = bh_idx / params.num_heads;
  const int head_idx = bh_idx % params.num_heads;
  const int tid = threadIdx.x;

  // For GQA/MQA: kv_head = head_idx / (num_heads / num_heads_k)
  // const int kv_head_idx = head_idx / (params.num_heads / params.num_heads_k);

  // ========================================================================
  // Step 1: Shared memory setup
  // ========================================================================
  extern __shared__ char smem_raw[];

  half *smem_q = reinterpret_cast<half *>(smem_raw);
  half *smem_kv = smem_q + BLOCK_M * HEAD_DIM;
  float *smem_scores = reinterpret_cast<float *>(smem_kv + HEAD_DIM * BLOCK_N);
  //
  // Partition shared memory:
  //   half *smem_q      = (half *)smem_raw;                         //
  //   BLOCK_M × HEAD_DIM half *smem_k      = smem_q + BLOCK_M * HEAD_DIM;
  //   // BLOCK_N × HEAD_DIM half *smem_v      = smem_k + BLOCK_N *
  //   HEAD_DIM;              // BLOCK_N × HEAD_DIM float *smem_scores =
  //   (float *)(smem_v + BLOCK_N * HEAD_DIM);  // BLOCK_M × BLOCK_N
  //
  // Or reuse K/V space (they're not live simultaneously):
  //   half *smem_kv     = smem_q + BLOCK_M * HEAD_DIM;              //
  //   BLOCK_N × HEAD_DIM float *smem_scores = (float *)(smem_kv + BLOCK_N *
  //   HEAD_DIM);
  //   // BLOCK_M × BLOCK_N

  // float scale = rsqrtf(HEAD_DIM);

  const half *q_ptr = reinterpret_cast<const half *>(params.q_ptr) +
                      BLOCK_M * m_block * params.q_row_stride +
                      batch_idx * params.q_batch_stride +
                      head_idx * params.q_head_stride;
  const half *k_ptr = reinterpret_cast<const half *>(params.k_ptr) +
                      batch_idx * params.k_batch_stride +
                      head_idx * params.k_head_stride;
  const half *v_ptr = reinterpret_cast<const half *>(params.v_ptr) +
                      batch_idx * params.v_batch_stride +
                      head_idx * params.v_head_stride;
  const half *o_ptr = reinterpret_cast<const half *>(params.o_ptr) +
                      batch_idx * params.o_batch_stride +
                      head_idx * params.o_head_stride;

  // ========================================================================
  // Step 3: Load Q tile into shared memory (one-time, reused across KV tiles)
  // ========================================================================

  int q_rows_valid = min(BLOCK_M, params.seqlen_q - m_block * BLOCK_M);
  load_tile_sync<BLOCK_M, HEAD_DIM, NTHREADS>(
      smem_q, q_ptr, params.q_row_stride, q_rows_valid, tid);

  // ========================================================================
  // Step 4: Initialize O accumulator (fp32, in registers) and softmax state
  // ========================================================================
  // O accumulator: each thread manages a subset of the BLOCK_M × HEAD_DIM
  // output.
  //
  // With 128 threads and BLOCK_M=128, each thread handles 1 full row (HEAD_DIM
  // floats). Or with WMMA store/load, it's distributed across warp threads.
  //
  // Simplest approach (Phase 1): keep O in shared memory as fp32, accumulate
  // there. Better approach: keep O as an array of WMMA accumulator fragments.
  //
  // For Phase 1, allocate per-thread:
  const int rows_per_thread = BLOCK_M / NTHREADS;
  float o_acc[rows_per_thread][HEAD_DIM];
  float m_state[rows_per_thread];
  float l_state[rows_per_thread];
  //   const int rows_per_thread = BLOCK_M / NTHREADS;  // e.g., 128/128 = 1
  //   float o_acc[rows_per_thread][HEAD_DIM];           // zero-initialized
  //   float m_state[rows_per_thread];                   // = -INFINITY
  //   float l_state[rows_per_thread];                   // = 0

  // ========================================================================
  // Step 5: KV tile loop
  // ========================================================================
  // Causal bound:
  int kv_end = (Is_causal) ? min((m_block + 1) * BLOCK_M, params.seqlen_k)
                           : params.seqlen_k;
  //
  for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N) {
    int kv_rows_valid = min(BLOCK_M, params.seqlen_k - kv_start);
    load_tile_sync<BLOCK_N, HEAD_DIM, NTHREADS>(
        smem_kv, k_ptr, params.k_row_stride, kv_rows_valid, tid);

    // Q @ K^T using WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

    for (int wi = 0; wi < BLOCK_M; wi += 16) {
      wmma::fill_fragment(s_frag, 0.0f);
      for (int wj = 0; wj < HEAD_DIM; wj += 16) {
        wmma::load_matrix_sync(q_frag, &smem_q[wi * HEAD_DIM + wk], HEAD_DIM);
        wmma::load_matrix_sync(k_frag, &smem_k[wi * HEAD_DIM + wk], HEAD_DIM);
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      wmma::store_matrix_sync(&smem_scores[wi * BLOCK_N + wj], s_frag, BLOCK_N,
                              wmma::mem_row_major);
    }
  }
  //     // --- 5b. Compute S = Q @ K^T using WMMA ---
  //     // S is BLOCK_M × BLOCK_N, stored in fp32 accumulators.
  //     //

  //     // for (int wi = 0; wi < BLOCK_M; wi += 16) {
  //     //     for (int wj = 0; wj < BLOCK_N; wj += 16) {
  //     //         wmma::fill_fragment(s_frag, 0.0f);
  //     //         for (int wk = 0; wk < HEAD_DIM; wk += 16) {
  //     //             wmma::load_matrix_sync(q_frag, &smem_q[wi * HEAD_DIM +
  //     wk], HEAD_DIM);
  //     //             wmma::load_matrix_sync(k_frag, &smem_k[wj * HEAD_DIM +
  //     wk], HEAD_DIM);
  //     //             // K is row-major in smem but we want K^T, so load as
  //     col_major
  //     //             wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
  //     //         }
  //     //         // Apply softmax scale (log2e-adjusted)
  //     //         // for (int i = 0; i < s_frag.num_elements; i++)
  //     //         //     s_frag.x[i] *= params.scale_softmax_log2;
  //     //
  //     //         // Store S tile to shared memory for softmax
  //     //         // wmma::store_matrix_sync(&smem_scores[wi * BLOCK_N + wj],
  //     s_frag, BLOCK_N, wmma::mem_row_major);
  //     //     }
  //     // }
  //     // __syncthreads();
  //
  //     // --- 5c. Causal masking ---
  //     // if constexpr (Is_causal) {
  //     //     // Each thread handles some rows of smem_scores
  //     //     int q_row_start = m_block * BLOCK_M;
  //     //     for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += NTHREADS) {
  //     //         int qi = idx / BLOCK_N;
  //     //         int kj = idx % BLOCK_N;
  //     //         if ((q_row_start + qi) < (kv_start + kj)) {
  //     //             smem_scores[qi * BLOCK_N + kj] = -INFINITY;
  //     //         }
  //     //     }
  //     //     __syncthreads();
  //     // }
  //
  //     // --- 5d. Online softmax on scores in shared memory ---
  //     // Each thread processes its assigned rows:
  //     // for (int r = 0; r < rows_per_thread; r++) {
  //     //     int row = tid * rows_per_thread + r;  // or however rows map to
  //     threads
  //     //     float *row_scores = &smem_scores[row * BLOCK_N];
  //     //     float m_old = m_state[r];
  //     //
  //     //     softmax_state[r].update(row_scores, kv_rows_valid,
  //     params.scale_softmax_log2);
  //     //
  //     //     // Rescale O accumulator
  //     //     float correction = softmax_state[r].get_correction(m_old);
  //     //     for (int d = 0; d < HEAD_DIM; d++)
  //     //         o_acc[r][d] *= correction;
  //     // }
  //     // __syncthreads();
  //
  //     // --- 5e. Convert P to fp16 in shared memory ---
  //     // P is currently fp32 in smem_scores. Convert to fp16 for WMMA input.
  //     // Reinterpret smem_scores as half* or use a separate buffer.
  //     // half *smem_p = (half *)smem_scores;  // careful: fp32 is 4 bytes,
  //     half is 2
  //     // for (int idx = tid; idx < BLOCK_M * BLOCK_N; idx += NTHREADS) {
  //     //     smem_p[idx] = __float2half(smem_scores_copy[idx]);
  //     // }
  //     // __syncthreads();
  //
  //     // --- 5f. Load V tile into shared memory ---
  //     // load_tile_sync<BLOCK_N, HEAD_DIM, NTHREADS>(
  //     //     smem_v, v_ptr + kv_start * params.v_row_stride,
  //     //     params.v_row_stride, kv_rows_valid, tid);
  //
  //     // --- 5g. Compute O += P @ V using WMMA ---
  //     // O is BLOCK_M × HEAD_DIM
  //     // P is BLOCK_M × BLOCK_N (fp16 in smem)
  //     // V is BLOCK_N × HEAD_DIM (fp16 in smem)
  //     //
  //     // This accumulates into o_acc (or into WMMA fragments that map to
  //     o_acc).
  //     // Phase 1 approach: store P@V result to smem, then each thread adds to
  //     o_acc.
  //     //
  //     // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
  //     p_frag;
  //     // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
  //     v_frag;
  //     // wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
  //     //
  //     // for (int wi = 0; wi < BLOCK_M; wi += 16) {
  //     //     for (int wj = 0; wj < HEAD_DIM; wj += 16) {
  //     //         wmma::fill_fragment(o_frag, 0.0f);
  //     //         for (int wk = 0; wk < BLOCK_N; wk += 16) {
  //     //             wmma::load_matrix_sync(p_frag, &smem_p[wi * BLOCK_N +
  //     wk], BLOCK_N);
  //     //             wmma::load_matrix_sync(v_frag, &smem_v[wk * HEAD_DIM +
  //     wj], HEAD_DIM);
  //     //             wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
  //     //         }
  //     //         // Store o_frag to smem, then add to o_acc
  //     //         // wmma::store_matrix_sync(&smem_o_tmp[wi * HEAD_DIM + wj],
  //     o_frag, HEAD_DIM, wmma::mem_row_major);
  //     //     }
  //     // }
  //     // __syncthreads();
  //     //
  //     // // Add this tile's contribution to the running O
  //     // for (int r = 0; r < rows_per_thread; r++) {
  //     //     int row = tid * rows_per_thread + r;
  //     //     for (int d = 0; d < HEAD_DIM; d++)
  //     //         o_acc[r][d] += smem_o_tmp[row * HEAD_DIM + d];
  //     // }
  // }

  // ========================================================================
  // Step 6: Final normalization and writeback
  // ========================================================================
  // for (int r = 0; r < rows_per_thread; r++) {
  //     int row = tid * rows_per_thread + r;
  //     float inv_l = 1.0f / l_state[r];
  //     for (int d = 0; d < HEAD_DIM; d++) {
  //         o_ptr[row * params.o_row_stride + d] = __float2half(o_acc[r][d] *
  //         inv_l);
  //     }
  // }
  //
  // // Optionally store log-sum-exp for backward pass:
  // // softmax_lse[batch][head][q_row] = m + log2f(l) / log2(e)
  // // if (params.softmax_lse_ptr != nullptr) {
  // //     for (int r = 0; r < rows_per_thread; r++) {
  // //         int row = m_block * BLOCK_M + tid * rows_per_thread + r;
  // //         if (row < params.seqlen_q) {
  // //             int lse_idx = batch_idx * params.num_heads * params.seqlen_q
  // //                         + head_idx * params.seqlen_q + row;
  // //             params.softmax_lse_ptr[lse_idx] = m_state[r] / M_LOG2E +
  // logf(l_state[r]);
  // //         }
  // //     }
  // // }
}

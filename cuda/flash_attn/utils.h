/*
 * utils.h — Memory movement and synchronization utilities.
 *
 * No CuTe dependency — just raw CUDA intrinsics for cp.async and helpers.
 *
 * cp.async overview:
 *   SM80 introduced cp.async: DMA from global to shared memory that bypasses
 *   registers entirely. The GPU can overlap these loads with compute on
 *   previously-loaded tiles — this is how we hide memory latency.
 *
 *   Flow:
 *     cp_async_copy(src_global, dst_shared, bytes);  // non-blocking
 *     cp_async_commit();                              // group outstanding
 * copies
 *     // ... do compute ...
 *     cp_async_wait<N>();                             // wait until ≤N groups
 * in flight
 *     __syncthreads();
 *
 * What you need to implement:
 *   - cp_async_copy(): issue a single 16-byte async copy
 *   - Cooperative tile loaders: threads divide up a (BLOCK × HEAD_DIM) tile
 *   - Predicated copies for boundary tiles (seqlen % BLOCK != 0)
 *   - Double-buffer swap helpers (Phase 2)
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// cp.async intrinsics (SM80+)
// ============================================================================

// Copy 16 bytes from global to shared memory asynchronously.
// This bypasses registers — the data goes directly global → L2 → shared.
__device__ __forceinline__ void cp_async_16B(void *dst_shared,
                                             const void *src_global) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(dst), "l"(src_global));
}

// Commit all outstanding cp.async copies as a group.
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

// Wait until at most N groups are still in flight.
// cp_async_wait<0>() = wait for ALL copies to finish.
// cp_async_wait<1>() = allow 1 group in flight (for double buffering).
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// ============================================================================
// Cooperative tile loading helpers
// ============================================================================

// TODO: Load a (rows × cols) half tile from global to shared memory.
// Each thread loads 8 halfs (16 bytes) per cp.async call.
// Total elements = rows * cols, total 16B loads = rows * cols / 8.
// Distribute across kNThreads.
//
// template <int rows, int cols, int kNThreads>
// __device__ void load_tile_global_to_shared(
//     half *smem,                    // destination in shared memory
//     const half *gmem,              // source in global memory
//     int row_stride,                // stride between rows in global memory
//     int valid_rows,                // for boundary masking
//     int tid
// ) {
//     // Strategy: treat the tile as a flat array of 16-byte chunks.
//     // Each thread handles (rows * cols / 8) / kNThreads chunks.
//     //
//     // For contiguous tiles (stride == cols), this is straightforward.
//     // For strided tiles (stride != cols), you need to compute the
//     // global address per-row. The shared memory destination is always
//     // packed contiguously (so the WMMA loads work).
//     //
//     // Boundary handling: for the last tile in the sequence, some rows
//     // may be out-of-bounds. Use predicated copies or write zeros to
//     // shared memory for invalid rows.
//     constexpr int total_loads = (rows * cols) / 8;  // number of 16B loads
//     constexpr int loads_per_thread = total_loads / kNThreads;
//
//     #pragma unroll
//     for (int i = 0; i < loads_per_thread; i++) {
//         int load_idx = tid + i * kNThreads;
//         int row = (load_idx * 8) / cols;
//         int col = (load_idx * 8) % cols;
//
//         if (row < valid_rows) {
//             cp_async_16B(&smem[row * cols + col], &gmem[row * row_stride +
//             col]);
//         } else {
//             // Zero out shared memory for out-of-bounds rows
//             // (so softmax sees -inf after Q@K^T, contributing nothing)
//             // *reinterpret_cast<uint4*>(&smem[row * cols + col]) =
//             make_uint4(0,0,0,0);
//         }
//     }
// }

// ============================================================================
// Simple synchronous tile load (Phase 1 — no cp.async)
// ============================================================================

// TODO: Start with this, replace with cp.async in Phase 2.
// Each thread cooperatively loads part of a (rows × cols) tile.
//
template <int rows, int cols, int smem_stride, int kNThreads>
__device__ void load_tile_sync(half *smem, const half *gmem, int row_stride,
                               int valid_rows, int tid) {
  constexpr int n = rows * cols;
  constexpr int n_per_thread = n / (8 * kNThreads); // vector load later
  for (int i = 0; i < n_per_thread; i++) {
    int index = 8 * (tid + i * kNThreads);
    int row = index / cols;
    int col = index % cols;
    int smem_idx = row * smem_stride + col;
    if (row < valid_rows) {
      // grab float 4 chunk from gmem
      float4 chunk =
          reinterpret_cast<const float4 *>(&gmem[row * row_stride + col])[0];
      half2 *c_ptr = reinterpret_cast<half2 *>(&chunk);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        half2 val = c_ptr[j];
        smem[smem_idx + 2 * j] = val.x;
        smem[smem_idx + 2 * j + 1] = val.y;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        smem[smem_idx + j] = __float2half(0.0f);
      }
    }
  }
  __syncthreads();
}

template <int rows, int cols, int kNThreads>
__device__ void load_tile_sync_half(half *smem, const half *gmem,
                                    int row_stride, int valid_rows, int tid) {
  constexpr int total_elems = rows * cols;
  constexpr int elems_per_thread = total_elems / kNThreads;
#pragma unroll
  for (int i = 0; i < elems_per_thread; i++) {
    int idx = tid + i * kNThreads;
    int row = idx / cols;
    int col = idx % cols;
    if (row < valid_rows) {
      smem[idx] = gmem[row * row_stride + col];
    } else {
      smem[idx] = __float2half(0.0f);
    }
  }
  __syncthreads();
}

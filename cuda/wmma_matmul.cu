/*
 * Standalone tiled matmul using WMMA to test tensor core usage.
 *
 * Computes C[M×N] = A[M×K] @ B[K×N] using fp16 inputs, fp32 accumulator.
 *
 * Grid:  (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
 * Block: 128 threads (4 warps)
 *
 * Each CTA computes a BLOCK_M × BLOCK_N tile of C:
 *   - Load A and B tiles into shared memory
 *   - Warps partition the BLOCK_M rows
 *   - Inner k-loop accumulates 16×16 WMMA tiles
 *   - Store result to global memory
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;

constexpr int TILE_SIZE = 16;
constexpr int NWARPS = 1;
constexpr int NTHREADS = NWARPS * 32;

__global__ void wmma_matmul_kernel(const half *A, const half *B, float *C,
                                   int M, int N, int K) {
  int block_row = blockIdx.x * TILE_SIZE;
  int block_col = blockIdx.y * TILE_SIZE;

  wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float>
      c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  const half *a_tile = A + block_row * K;
  const half *b_tile = B + block_col;

  for (int i = 0; i < K; i += TILE_SIZE) {
    wmma::load_matrix_sync(a_frag, a_tile + i, K);
    wmma::load_matrix_sync(b_frag, b_tile + i * N, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  float *c_tile = C + block_row * N + block_col;
  wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}

torch::Tensor wmma_matmul_cuda(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);

  auto C =
      torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
  dim3 block(NTHREADS);

  //   int smem_size = (TILE_SIZE * TILE_SIZE + BLOCK_K * BLOCK_N) *
  //   sizeof(half);

  wmma_matmul_kernel<<<grid, block>>>(
      reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
      reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
      C.data_ptr<float>(), M, N, K);

  return C;
}

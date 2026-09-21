#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(const float *__restrict__ X,
                                 float *__restrict__ output, int M, int N) {
  __shared__ float stage[TILE][TILE + 1];

  int col = blockIdx.x * TILE + threadIdx.x;
  int row = blockIdx.y * TILE + threadIdx.y;

  // Load: each thread handles TILE/BLOCK_ROWS = 4 rows of the tile.
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (row + j < M && col < N) {
      stage[threadIdx.y + j][threadIdx.x] = X[(row + j) * N + col];
    }
  }

  __syncthreads();

  int out_col = blockIdx.y * TILE + threadIdx.x;
  int out_row = blockIdx.x * TILE + threadIdx.y;

  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (out_row + j < N && out_col < M) {
      output[(out_row + j) * M + out_col] = stage[threadIdx.x][threadIdx.y + j];
    }
  }
}

torch::Tensor transpose_cuda(torch::Tensor X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(X.dim() == 2, "X must be 2D");

  X = X.contiguous();
  int M = X.size(0), N = X.size(1);
  auto output = torch::empty({N, M}, X.options());

  dim3 block(TILE, BLOCK_ROWS);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  transpose_kernel<<<grid, block>>>(X.data_ptr<float>(),
                                    output.data_ptr<float>(), M, N);

  return output;
}
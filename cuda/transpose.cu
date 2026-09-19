// FP32
//
// Matrix transpose CUDA kernel — naive implementation (no shared-memory
// tiling, so reads or writes end up uncoalesced depending on direction).
//
// Build with: python cuda/setup.py install
// Or use torch.utils.cpp_extension.load() for JIT compilation

#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#define TILE 16

__global__ void transpose_kernel(const float *__restrict__ X,
                                 float *__restrict__ output, int M, int N) {
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  __shared__ float stage[TILE][TILE + 1];

  if (row < M && col < N) {
    stage[threadIdx.y][threadIdx.x] = X[row * N + col];
  }

  __syncthreads();

  // update row & col to tile's transpose perspectivew
  // for coalescing indexing
  int out_row = blockIdx.x * TILE + threadIdx.y;
  int out_col = blockIdx.y * TILE + threadIdx.x;

  if (out_row < N && out_col < M) {
    output[out_row * M + out_col] = stage[threadIdx.x][threadIdx.y];
  }
}

torch::Tensor transpose_cuda(torch::Tensor X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(X.dim() == 2, "X must be 2D");

  X = X.contiguous();
  int M = X.size(0), N = X.size(1);
  auto output = torch::empty({N, M}, X.options());

  const int BX = TILE, BY = TILE;
  dim3 block(BX, BY);
  dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);

  transpose_kernel<<<grid, block>>>(X.data_ptr<float>(),
                                    output.data_ptr<float>(), M, N);

  return output;
}
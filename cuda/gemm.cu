#include <cuda_runtime.h>
#include <torch/extension.h>

#define TILE 16

__global__ void gemm_kernel(const float *__restrict__ A,
                            const float *__restrict__ B,
                            float *__restrict__ output, int M, int N, int K) {
  int tidx = threadIdx.x, tidy = threadIdx.y;

  int row = blockIdx.y * blockDim.y + tidy;
  int col = blockIdx.x * blockDim.x + tidx;

  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int tile = 0, idx = 0; tile < K; tile += TILE, idx++) {
      As[row][idx] = A[row * K + tile + tidx];
      Bs[idx][col] = B[(tile + idx) * N + col];
      __syncthreads();

      for (int i = 0; i < TILE; i++) {
        sum += As[row][i] * Bs[i][col];
      }
      __syncthreads();
    }

    output[row * N + col] = sum;
  }
}

torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {

  A = A.contiguous();
  B = B.contiguous();

  int M = A.size(0), K = A.size(1);
  int N = B.size(1);
  auto output = torch::empty({M, N}, A.options());

  //   dim3 block(TILE, BLOCK_ROWS);
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                               output.data_ptr<float>(), M, N, K);

  return output;
}
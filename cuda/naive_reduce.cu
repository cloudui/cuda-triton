

#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void naive_reduce_kernel(const float *__restrict__ X,
                                    float *__restrict__ output,
                                    int n_elements) {
  int tid = threadIdx.x;
  int idx = blockDim.x * blockIdx.x + tid;

  extern __shared__ float shared[];

  shared[tid] = (idx < n_elements) ? X[idx] : 0.0f;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride >= 32; stride /= 2) {
    if (tid < stride) {
      shared[tid] = shared[tid] + shared[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      shared[tid] += __shfl_down_sync(0xffffffff, shared[tid], offset);
    }
  }

  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

torch::Tensor naive_reduce_cuda(torch::Tensor X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");

  X = X.contiguous();
  int n_elements = X.numel();
  int threads = 256;

  torch::Tensor input = X;

  while (n_elements > 1) {
    int grid = (n_elements + threads - 1) / threads;
    auto output = torch::empty({grid}, X.options());
    int smem_size = threads * sizeof(float);

    naive_reduce_kernel<<<grid, threads, smem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n_elements);

    input = output;
    n_elements = grid;
  }

  return input; // a 1-element tensor holding the scalar result
}
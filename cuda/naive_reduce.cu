
#include <algorithm>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void naive_reduce_kernel(const float *__restrict__ X,
                                    float *__restrict__ output,
                                    int n_elements) {
  int tid = threadIdx.x;
  int warpid = tid / 32;
  constexpr int nWarps = blockDim.x / 32;

  extern __shared__ float shared[];

  // cascade add
  float sum = 0.0f;
  for (int idx = blockDim.x * blockIdx.x + tid; idx < n_elements;
       idx += blockDim.x * gridDim.x) {
    sum += X[idx];
  }

  // pre tree reduce warp reduction
  float preval = shared[tid];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    preval += __shfl_down_sync(0xffffffff, preval, offset);
  }
  if (tid % 32 == 0) {
    shared[warpid] = preval;
  }

  __syncthreads();

  if (warpid == 0) {
    float val = (tid < blockDim.x / 32) ? shared[tid] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid == 0) {
      atomicAdd(output, val);
    }
  }
}

torch::Tensor naive_reduce_cuda(torch::Tensor X) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");

  X = X.contiguous();
  int n_elements = X.numel();
  int threads = 256;

  int grid = (n_elements + threads - 1) / threads;
  grid = std::min(grid, 82 * 32);
  auto output = torch::zeros({1}, X.options());
  int smem_size = threads / 32 * sizeof(float);

  naive_reduce_kernel<<<grid, threads, smem_size>>>(
      X.data_ptr<float>(), output.data_ptr<float>(), n_elements);

  return output; // a 1-element tensor holding the scalar result
}
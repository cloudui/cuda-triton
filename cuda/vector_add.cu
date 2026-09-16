// FP32
//
// Vector Add CUDA kernel — barebones NAIVE implementation.
//
// Compare to kernels/vector_add.py to see what Triton abstracts away: the
// grid-stride index math (blockIdx/blockDim/threadIdx vs tl.program_id +
// tl.arange), and the explicit bounds check (vs Triton's mask=).
//
// Build with: python cuda/setup.py install
// Or use torch.utils.cpp_extension.load() for JIT compilation

#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float *__restrict__ X,
                                  const float *__restrict__ Y,
                                  float *__restrict__ output, int n_elements) {
  int block = blockIdx.x;
  int tid = threadIdx.x;

  int idx = block * blockDim.x + tid;

  if (idx < n_elements) {
    output[idx] = X[idx] + Y[idx];
  }
}

torch::Tensor vector_add_cuda(torch::Tensor X, torch::Tensor Y) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(Y.is_cuda(), "Y must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
  TORCH_CHECK(Y.scalar_type() == torch::kFloat32, "Y must be float32");
  TORCH_CHECK(X.sizes() == Y.sizes(), "X and Y must have the same shape");

  X = X.contiguous();
  Y = Y.contiguous();

  auto output = torch::empty_like(X);
  int n_elements = X.numel();

  int threads = 256;
  int grid = (n_elements + threads - 1) / threads;

  vector_add_kernel<<<grid, threads>>>(X.data_ptr<float>(), Y.data_ptr<float>(),
                                       output.data_ptr<float>(), n_elements);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return output;
}
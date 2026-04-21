#include <torch/extension.h>

// Forward declarations — implemented in their own .cu files
torch::Tensor softmax_cuda(torch::Tensor input);
torch::Tensor softmax_triton_cuda(torch::Tensor input);
torch::Tensor fused_rmsnorm_swiglu_cuda(torch::Tensor input,
                                        torch::Tensor weight,
                                        torch::Tensor gate);
torch::Tensor wmma_matmul_cuda(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &softmax_cuda, "Naive Softmax (CUDA)");
  m.def("softmax_triton", &softmax_triton_cuda, "Softmax Triton-style (CUDA) ");
  m.def("fused_rmsnorm_swiglu", &fused_rmsnorm_swiglu_cuda,
        "Fused RMSNorm + SwiGLU (CUDA)");
  m.def("wmma_matmul", &wmma_matmul_cuda, "WMMA Tiled Matmul (1 warp)");
}

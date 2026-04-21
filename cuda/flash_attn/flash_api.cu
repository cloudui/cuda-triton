/*
 * flash_api.cu — PyTorch C++ extension entry point.
 *
 * Bridges PyTorch tensors → Flash_fwd_params → kernel launch.
 * Handles:
 *   - Input validation (shapes, dtypes, contiguity)
 *   - Extracting strides from PyTorch tensors
 *   - Allocating output and LSE tensors
 *   - Dispatching to the right head_dim instantiation
 *
 * What you need to implement:
 *   - mha_fwd(): main entry point, accepts Q/K/V tensors + options
 *   - PYBIND11_MODULE for the Python binding
 */

#include <torch/extension.h>

#include "flash.h"
#include "flash_fwd_launch_template.h"

std::vector<torch::Tensor> mha_fwd(
    torch::Tensor &q,   // (batch, seqlen_q, num_heads, head_dim)
    torch::Tensor &k,   // (batch, seqlen_k, num_heads_k, head_dim)
    torch::Tensor &v,   // (batch, seqlen_k, num_heads_k, head_dim)
    bool is_causal
) {
    // --- Input validation ---
    // TORCH_CHECK(q.dtype() == torch::kHalf, "Only fp16 supported");
    // TORCH_CHECK(q.is_cuda(), "Input must be on CUDA");
    // TORCH_CHECK(q.stride(-1) == 1, "Head dim must be contiguous");

    const int batch_size = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_dim = q.size(3);
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    // --- Allocate outputs ---
    auto output = torch::empty_like(q);
    auto softmax_lse = torch::empty(
        {batch_size, num_heads, seqlen_q},
        torch::dtype(torch::kFloat32).device(q.device())
    );

    // --- Fill params struct ---
    Flash_fwd_params params;
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = output.data_ptr();
    params.softmax_lse_ptr = softmax_lse.data_ptr<float>();

    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.num_heads = num_heads;
    params.num_heads_k = num_heads_k;
    params.head_dim = head_dim;

    // Strides (PyTorch gives strides in elements)
    params.q_batch_stride = q.stride(0);
    params.q_row_stride = q.stride(1);
    params.q_head_stride = q.stride(2);

    params.k_batch_stride = k.stride(0);
    params.k_row_stride = k.stride(1);
    params.k_head_stride = k.stride(2);

    params.v_batch_stride = v.stride(0);
    params.v_row_stride = v.stride(1);
    params.v_head_stride = v.stride(2);

    params.o_batch_stride = output.stride(0);
    params.o_row_stride = output.stride(1);
    params.o_head_stride = output.stride(2);

    params.scale_softmax = 1.0f / sqrtf(static_cast<float>(head_dim));
    params.scale_softmax_log2 = params.scale_softmax * M_LOG2E;

    params.is_causal = is_causal;

    // --- Dispatch by head_dim ---
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 64) {
        run_mha_fwd_hdim64(params, stream);
    } else if (head_dim == 128) {
        run_mha_fwd_hdim128(params, stream);
    } else {
        TORCH_CHECK(false, "FlashAttention only supports head_dim 64 and 128, got ", head_dim);
    }

    return {output, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_fwd", &mha_fwd, "FlashAttention-2 forward (CUDA)");
}

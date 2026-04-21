/*
 * flash_fwd_launch_template.h — Kernel launch configuration and dispatch.
 *
 * Bridges runtime head_dim → compile-time template parameters.
 * Each .cu instantiation file includes this and calls run_flash_fwd<Traits>().
 */

#pragma once

#include <cuda_runtime.h>

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"

template <typename Traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int smem_size = Traits::kSmemSize;

    const int num_m_blocks = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dim3 grid(num_m_blocks, params.batch_size * params.num_heads);
    dim3 block(Traits::kNThreads);

    auto kernel = &flash_fwd_kernel<Traits, Is_causal>;

    // A100 default shared memory is 48 KB. If we need more, request it.
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

// Dispatch by head_dim — called from flash_api.cu

inline void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_causal) {
        run_flash_fwd<Traits_hdim64, true>(params, stream);
    } else {
        run_flash_fwd<Traits_hdim64, false>(params, stream);
    }
}

inline void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.is_causal) {
        run_flash_fwd<Traits_hdim128, true>(params, stream);
    } else {
        run_flash_fwd<Traits_hdim128, false>(params, stream);
    }
}

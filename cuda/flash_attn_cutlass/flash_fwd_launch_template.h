/*
 * flash_fwd_launch_template.h — Kernel launch dispatch.
 *
 * Same structure as the WMMA version's launch template.
 */

#pragma once

#include <cassert>
#include <cuda_runtime.h>

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "kernel_traits.cuh"

template <typename Traits>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int kBlockM = Traits::kBlockM;
    constexpr int smem_size = Traits::kSmemSize;

    const int num_m_blocks = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dim3 grid(num_m_blocks, params.batch_size * params.num_heads);
    dim3 block(Traits::kNThreads);

    auto kernel = &FLASH::flash_fwd_kernel<Traits>;

    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

inline void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<Traits_hdim32>(params, stream);
}

inline void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<Traits_hdim64>(params, stream);
}

inline void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    run_flash_fwd<Traits_hdim128>(params, stream);
}

/*
 * Instantiation: forward, hdim=128, fp16, non-causal, SM80 (A100)
 */

#include "flash_fwd_launch_template.h"

// template void run_flash_fwd<Flash_fwd_kernel_traits<128, 128, 64, 4, cutlass::half_t>, false>(
//     Flash_fwd_params &params, cudaStream_t stream);

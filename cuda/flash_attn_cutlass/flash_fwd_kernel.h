/*
 * flash_fwd_kernel.h — CuTe-based FlashAttention-2 forward kernel.
 *
 * STUB. To be implemented during the CuTe rewrite.
 *
 * Algorithm (same as WMMA version, but using CuTe primitives):
 *
 *   1. Make CuTe tensor views of Q, K, V, O in gmem (with strides from Flash_fwd_params)
 *   2. Slice to this CTA's tile via local_tile()
 *   3. cp.async load Q tile into smem (cute::copy with TiledCopy)
 *   4. KV loop:
 *      a. cp.async load K, V tiles
 *      b. cute::gemm(tiled_mma, S_frag, Q_frag, K_frag) — fragment-resident accumulator
 *      c. Online softmax in fragment registers (warp-shuffle reductions)
 *      d. convert_layout_acc_Aregs(S_frag) → P_frag (rebind accumulator → matrix_a)
 *      e. cute::gemm(tiled_mma, O_frag, P_frag, V_frag, O_frag) — accumulate into O_frag
 *      f. Rescale O_frag by `correction` per iteration (fragment-aware traversal)
 *   5. Epilogue: normalize O_frag by l_state, store to gmem coalesced via TiledCopy
 *
 * Pipelining (Phase 2):
 *   Wrap the KV loop with cute::cp_async_pipeline for 3-stage overlap of loads + compute.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash.h"
#include "kernel_traits.h"

// TODO: include CuTe headers
// #include <cute/tensor.hpp>
// #include <cutlass/cutlass.h>

template <typename Traits, bool Is_causal>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
    // TODO: implement using CuTe primitives
    //
    // Phase 1 — Get a correct unpipelined version (kStages=1):
    //
    //   using namespace cute;
    //   using Element = typename Traits::Element;
    //   using TiledMma = typename Traits::TiledMma;
    //
    //   extern __shared__ char smem_raw[];
    //   auto smem_q  = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_raw)),
    //                              typename Traits::SmemLayoutQ{});
    //   ...
    //
    //   // gmem tensors with proper strides
    //   Tensor mQ = make_tensor(
    //       make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + ...),
    //       make_shape(params.seqlen_q, params.h, params.d),
    //       make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    //
    //   Tensor gQ = local_tile(mQ(_, head_idx, _),
    //                          Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                          make_coord(m_block, 0));
    //
    //   // load Q, KV loop, epilogue ...
    //
    // Phase 2 — Add cp_async_pipeline (kStages=3)
}

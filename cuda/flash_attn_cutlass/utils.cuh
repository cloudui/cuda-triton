#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace FLASH {

// default gemm
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThrCopyA,
          typename ThrCopyB>
__forceinline__ __device__ void
gemm(Tensor0 &acc,        // (MMA, MMA_M, MMA_N)        — fp32 accumulator
     Tensor1 &tCrA,       // (MMA, MMA_M, MMA_K)        — A regs (working set)
     Tensor2 &tCrB,       // (MMA, MMA_N, MMA_K)        — B regs (working set)
     Tensor3 const &tCsA, // (CPY, MMA_M, MMA_K)        — A in smem
     Tensor4 const &tCsB, // (CPY, MMA_N, MMA_K)        — B in smem
     TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
     TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
     ThrCopyB smem_thr_copy_B) {

  // retile register to match Copy Atom
  Tensor tXrA = smem_thr_copy_A.retile_D(tCrA);
  Tensor tXrB = smem_thr_copy_B.retile_D(tCrB);

  cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tXrA(_, _, _0{}));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tXrB(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); i++) {
    // prefetch next block
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tXrA(_, _, i + 1));
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tXrB(_, _, i + 1));
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// O += P · V   (A is already in registers; only B comes from smem)
// Used after softmax — P lives in registers, never in smem.
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void
gemm_rs(Tensor0 &acc,        // (MMA, MMA_M, MMA_N)      — fp32 accumulator
        Tensor1 &tCrA,       // (MMA, MMA_M, MMA_K)      — A already in regs
        Tensor2 &tCrB,       // (MMA, MMA_N, MMA_K)      — B regs (working set)
        Tensor3 const &tCsB, // (CPY, MMA_N, MMA_K)      — B in smem
        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
        ThrCopy smem_thr_copy_B) {}

} // namespace FLASH

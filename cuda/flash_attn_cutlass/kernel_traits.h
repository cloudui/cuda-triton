/*
 * kernel_traits.h — CuTe-based kernel configuration.
 *
 * STUB. To be filled in during the CuTe rewrite.
 *
 * Mirrors the structure of Tri Dao's kernel_traits.h:
 *   https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h
 *
 * Sketch of what this file will contain (using CuTe types):
 *
 *   - Element types:       cutlass::half_t, float (accumulator)
 *   - WarpShape:           e.g. Shape<_64, _64, _16>
 *   - InstructionShape:    Shape<_16, _8, _16> (m16n8k16 MMA)
 *   - SmemLayoutQ:         composition(Swizzle<3,3,3>{}, Layout<Shape<_BLOCK_M, _HEAD_DIM>, Stride<_HEAD_DIM, _1>>{})
 *   - SmemLayoutK:         same shape, swizzled
 *   - SmemLayoutV:         transposed for efficient matrix_b access in the second matmul
 *   - TiledMma:            decltype(make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, ...))
 *   - GmemTiledCopyQKV:    decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<...>, Element>{}, ...))
 *   - SmemCopyAtom:        Copy_Atom<SM75_U32x4_LDSM_N, Element>  (ldmatrix.x4)
 *   - kSmemSize:           sum of all SmemLayout sizes via cosize_v<Layout> * sizeof(Element)
 *   - Pipeline depth:      kStages = 1 (initial), bumped to 3 once cp.async pipeline is in
 */

#pragma once

#include <cuda_fp16.h>

// TODO: include CUTLASS / CuTe headers
// #include <cutlass/cutlass.h>
// #include <cutlass/numeric_types.h>
// #include <cute/tensor.hpp>
// #include <cute/atom/mma_atom.hpp>
// #include <cute/atom/copy_atom.hpp>

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits {
    static constexpr int kHeadDim  = kHeadDim_;
    static constexpr int kBlockM   = kBlockM_;
    static constexpr int kBlockN   = kBlockN_;
    static constexpr int kNWarps   = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kStages   = 1;  // bump to 3 when adding cp.async pipelining

    // TODO: define CuTe types
    // using Element       = cutlass::half_t;
    // using ElementAccum  = float;
    // using TileShape     = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>, cute::Int<kHeadDim>>;
    // using SmemLayoutQ   = decltype(cute::composition(
    //                          cute::Swizzle<3, 3, 3>{},
    //                          cute::Layout<cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>,
    //                                       cute::Stride<cute::Int<kHeadDim>, cute::_1>>{}));
    // using SmemLayoutKV  = ...
    // using SmemLayoutVtransposed = ...
    // using TiledMma      = decltype(cute::make_tiled_mma(
    //                          cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>{},
    //                          cute::Layout<cute::Shape<cute::Int<kNWarps>, cute::_1, cute::_1>>{},
    //                          cute::Tile<cute::Int<16 * kNWarps>, cute::_16, cute::_16>{}));
    // using GmemTiledCopyQKV = ...
    // using SmemCopyAtom  = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

    // Placeholder smem size to keep launch template compiling.
    // Real value: cosize_v<SmemLayoutQ> * sizeof(Element) + ... (CuTe gives us this for free)
    static constexpr int kSmemSize = 0;
};

// Concrete configs (mirror the WMMA version's choices)
using Traits_hdim64  = Flash_fwd_kernel_traits<64,  128, 64, 4>;
using Traits_hdim128 = Flash_fwd_kernel_traits<128, 128, 64, 4>;

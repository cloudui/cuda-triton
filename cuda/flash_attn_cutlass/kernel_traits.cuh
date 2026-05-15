#pragma once

// TODO: include CUTLASS / CuTe headers
// #include <cutlass/cutlass.h>
// #include <cutlass/numeric_types.h>
#include <cuda_fp16.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits {
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * 32;

  // sm80: 64 col fp16 follows <3,3,3> swizzle pattern
  // bank conflict each 4 byte * 32 = 128-byte boundary
  // 128 byte / 2 byte/half = 64 bit boundary
  // 2B swizzle for
  static constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
  static constexpr int kSwizzle = (kBlockKSmem == 64) ? 3 : 2;
  // static constexpr int kBlockKGmem = 0;

  // smem copy defs
  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  // raw tensor shape for fragments
  // https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle_layout.hpp
  using SmemLayoutVtNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVt{}));
  // O, same as Layout Q
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>;
  // explicit 16-bit half x 8, transposed
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t>;
  // for our implementation, same as Universal Copy for uint32_t
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::half_t>;

  // gmem intermediary static ints
  static constexpr int kGmemElementsPerLoad =
      sizeof(cute::uint128_t) / sizeof(cute::half_t);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElementsPerLoad;

  // gmem copy defs
  using GmemCopyAtom =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>;
  using GmemLayout = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemLayoutO = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      GmemCopyAtom{}, GmemLayout{}, Layout<Shape<_1, _8>>{}));
  using GmemTiledCopyO = decltype(make_tiled_copy(
      SmemCopyAtomO{}, GmemLayoutO{}, Layout<Shape<_1, _8>>{}));

  // tiled mma defs
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<Int<kNWarps>, _1, _1>>,
                            Tile<Int<16 * kNWarps>, _16, _16>>;

  // Smem footprint per stage: sQ + sK + sV (sO reuses sQ's space).
  static constexpr int kSmemSize =
      sizeof(cute::half_t) * (kBlockM * kHeadDim + 2 * kBlockN * kHeadDim);
};

// Concrete configs (mirror the WMMA version's choices)
using Traits_hdim32 = Flash_fwd_kernel_traits<32, 128, 64, 4>;
using Traits_hdim64 = Flash_fwd_kernel_traits<64, 128, 64, 4>;
using Traits_hdim128 = Flash_fwd_kernel_traits<128, 128, 64, 4>;

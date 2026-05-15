// Minimal repro: does building the V fragment from the SWIZZLED sVt
// (instead of sVtNoSwizzle) actually break, and where?
//
// Build:
//   make run-cuda fn=scratch/v_fragment_test.cu
//
// Toggle USE_SWIZZLED_FRAGMENT to 1 to see the failure mode.

#include <cstdio>
#include <cuda_fp16.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

#ifndef USE_SWIZZLED_FRAGMENT
#define USE_SWIZZLED_FRAGMENT 0
#endif

template <int kHeadDim, int kBlockM, int kBlockN, int kNWarps> void run() {
  constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
  // constexpr int kSwizzle = (kBlockKSmem == 64) ? 3 : 2;
  constexpr int kSwizzle = 2;
  printf("\n========== kHeadDim=%d kBlockM=%d kBlockN=%d kNWarps=%d "
         "(kBlockKSmem=%d kSwizzle=%d) ==========\n",
         kHeadDim, kBlockM, kBlockN, kNWarps, kBlockKSmem, kSwizzle);

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
  using SmemLayoutVtNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVt{}));

  printf("SmemLayoutKV         : ");
  print(SmemLayoutKV{});
  printf("\n");
  printf("SmemLayoutVt         : ");
  print(SmemLayoutVt{});
  printf("\n");
  printf("SmemLayoutVtNoSwizzle: ");
  print(SmemLayoutVtNoSwizzle{});
  printf("\n\n");

  // Fake smem tensors (data ptr is unused for shape/partition inspection).
  cute::half_t *dummy = nullptr;
  using SmemLayoutQNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutQ{}));
  Tensor sQ = make_tensor(make_smem_ptr(dummy), SmemLayoutQ{});
  Tensor sQNS = make_tensor(make_smem_ptr(dummy), SmemLayoutQNoSwizzle{});
  Tensor sVt = make_tensor(make_smem_ptr(dummy), SmemLayoutVt{});
  Tensor sVtNS = make_tensor(make_smem_ptr(dummy), SmemLayoutVtNoSwizzle{});

  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<Int<kNWarps>, _1, _1>>,
                            Tile<Int<16 * kNWarps>, _16, _16>>;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(0);

  // ---- (1) partition_fragment_B on both ------------------------------------
  // Both should compile and give the same shape — partition_fragment only
  // looks at the logical shape/strides.
  Tensor tOrV_ns = thr_mma.partition_fragment_B(sVtNS);
  Tensor tOrV_sw = thr_mma.partition_fragment_B(sVt);
  printf("partition_fragment_B(sVtNoSwizzle) : ");
  print(tOrV_ns);
  printf("\n");
  printf("partition_fragment_B(sVt)          : ");
  print(tOrV_sw);
  printf("\n");

  Tensor tSrQ_ns = thr_mma.partition_fragment_A(sQNS);
  Tensor tSrQ_sw = thr_mma.partition_fragment_A(sQ);
  printf("partition_fragment_A(sQNoSwizzle)  : ");
  print(tSrQ_ns);
  printf("\n");
  printf("partition_fragment_A(sQ)           : ");
  print(tSrQ_sw);
  printf("\n");
  bool same_q = cute::is_same_v<decltype(tSrQ_ns.layout()),
                                 decltype(tSrQ_sw.layout())>;
  printf("--> Q fragment layout types match  : %s\n\n", same_q ? "YES" : "NO");

  // ---- (2) retile_D against the LDSM.trans copy atom -----------------------
  // This is where I expect the swizzled fragment to break.
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, cute::half_t>;
  auto smem_tiled_copy_V =
      make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(0);

  auto tOsVt = smem_thr_copy_V.partition_S(sVt);
  printf("partition_S(sVt) : ");
  print(tOsVt);
  printf("\n");

#if USE_SWIZZLED_FRAGMENT
  // Build the fragment from the SWIZZLED layout, then retile_D.
  auto tOrVt_view = smem_thr_copy_V.retile_D(tOrV_sw);
  printf("retile_D(tOrV from sVt)          : ");
  print(tOrVt_view);
  printf("\n");
#else
  auto tOrVt_view = smem_thr_copy_V.retile_D(tOrV_ns);
  printf("retile_D(tOrV from sVtNoSwizzle) : ");
  print(tOrVt_view);
  printf("\n");
  // Diff check: also retile_D the swizzled fragment and compare layouts.
  auto tOrVt_view_sw = smem_thr_copy_V.retile_D(tOrV_sw);
  printf("retile_D(tOrV from sVt)          : ");
  print(tOrVt_view_sw);
  printf("\n");
  bool same_frag =
      cute::is_same_v<decltype(tOrV_ns.layout()), decltype(tOrV_sw.layout())>;
  bool same_retile = cute::is_same_v<decltype(tOrVt_view.layout()),
                                     decltype(tOrVt_view_sw.layout())>;
  printf("--> fragment layout types match : %s\n", same_frag ? "YES" : "NO");
  printf("--> retile_D layout types match : %s\n", same_retile ? "YES" : "NO");
#endif
}

int main() {
  // (kHeadDim, kBlockM, kBlockN, kNWarps)
  // run<64, 128, 64, 4>();  // Traits_hdim64
  // run<128, 128, 64, 4>(); // Traits_hdim128
  // run<64, 128, 128, 4>(); // bigger N
  run<32, 128, 64, 4>(); // forces kBlockKSmem=32, kSwizzle=2
  run<96, 128, 64, 4>(); // not a multiple of 64 -> kBlockKSmem=32, kSwizzle=2
  return 0;
}

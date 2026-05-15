// Print the FA2 SMEM layouts (Q/K, Vt, VtNoSwizzle) so you can stare at the
// swizzle pattern without running the full kernel. Useful when reading the
// blog's "Swizzling FA2" and sVtNoSwizzle sections.
//
// To see the LaTeX/visual layout, pipe `print_latex` output into Overleaf or
// any LaTeX renderer.
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 -I<path-to-cutlass>/include \
//     scratch/swizzle_layouts.cu -o scratch/swizzle_layouts && ./scratch/swizzle_layouts
//
// Tweak HDIM below to compare hdim=64 (Swizzle<3,3,3>) vs hdim=32 (default
// Swizzle<2,3,3>). The hdim=32 case is what motivates the sVtNoSwizzle finding.

#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

template <int kHeadDim, int kBlockM, int kBlockN>
void print_one() {
  constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
  constexpr int kSwizzle    = (kBlockKSmem == 64) ? 3 : 2;

  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutQ  = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
                  Stride<Int<kBlockN>, _1>{})));
  using SmemLayoutVtNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVt{}));

  printf("===== kHeadDim=%d kBlockM=%d kBlockN=%d "
         "(kBlockKSmem=%d kSwizzle=%d) =====\n",
         kHeadDim, kBlockM, kBlockN, kBlockKSmem, kSwizzle);
  printf("SmemLayoutQ          : "); print(SmemLayoutQ{});           printf("\n");
  printf("SmemLayoutKV         : "); print(SmemLayoutKV{});          printf("\n");
  printf("SmemLayoutVt         : "); print(SmemLayoutVt{});          printf("\n");
  printf("SmemLayoutVtNoSwizzle: "); print(SmemLayoutVtNoSwizzle{}); printf("\n");
  printf("\n");
}

int main() {
  // hdim=64 path: kBlockKSmem=64, Swizzle<3,3,3>. Vt and VtNoSwizzle produce
  // identical fragment shapes; sVtNoSwizzle is a no-op.
  print_one<64, 128, 64>();

  // hdim=32 path: kBlockKSmem=32, Swizzle<2,3,3>. The non-adjacent row/col
  // bits make Vt's fragment shape diverge from VtNoSwizzle. See
  // print_latex(SmemLayoutKV{}) to see why -- the swizzled stride alternates
  // 32/40 instead of being a clean 72.
  print_one<32, 128, 64>();
  return 0;
}

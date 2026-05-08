#include <cstdio>
#include <cuda_fp16.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  constexpr int kBlockKSmem = 64;
  constexpr int kBlockM = 256;
  constexpr int kBlockN = 128;
  constexpr int kHeadDim = 64;
  constexpr int kSwizzle = 3;

  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      // This has to be kBlockKSmem, using kHeadDim gives wrong results for
      // d=128
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
                                  Stride<Int<kBlockN>, _1>{})));
  using SmemLayoutVtTD = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  // ));
  // SmemLayoutAtomVt{}, Shape<Int<kHeadDim>, Int<kBlockN>>{}));

  // // raw tensor shape for fragments
  // //
  // https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle_layout.hpp
  // using SmemLayoutVNoSwizzle =
  // decltype(get_nonswizzle_portion(SmemLayoutVt{}));
  print(SmemLayoutKV{});
  printf("\n");
  print(SmemLayoutVt{});
  printf("\n");
  print(SmemLayoutVtTD{});
  printf("\n");

  // printf("=== row-major 8x4 === \n");
  // auto l_row = make_layout(make_shape(_8{}, _4{}), make_stride(_4{}, _1{}));
  // print_layout(l_row);
  // printf("\n");

  // auto ln = composition(
  //     l_row, make_layout(make_shape(_4{}, _8{}), make_stride(_1{}, _8{})));
  // print_layout(ln);
  // printf("\n");

  return 0;
}

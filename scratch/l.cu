#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
using namespace cute;

template <typename Layout>
__forceinline__ auto convert_layout_rowcol(Layout const &in) {
  // (MMA, MMA_M, MMA_N), MMA=4 -> (2,2)
  auto sl = logical_divide(in, Shape<_2>{}); // ((2, MMA/2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(sl), get<1>(sl)),
                     make_layout(get<0, 0>(sl), get<2>(sl)));
}

static int x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

int main() {
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kHeadDim = 64;
  constexpr int kNWarps = 1;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<Int<kNWarps>, _1, _1>>,
                            Tile<Int<16 * kNWarps>, _16, _16>>;
  TiledMma tiled_mma;
  print_layout(tiled_mma.get_layoutC_TV());
  printf("\n");

  Tensor sQ = make_tensor(reinterpret_cast<cute::half_t *>(x),
                          Layout<Shape<Int<kBlockM>, Int<kBlockN>>>{});
  auto thr_mma = tiled_mma.get_thread_slice(0);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

  Tensor acc_s = partition_fragment_C(
      tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

  Tensor acc_o = partition_fragment_C(
      tiled_mma, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));

  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::half_t>;
  using SmemCopyAtomO1 =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<8>, cute::half_t>;
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_tiled_copy_O1 = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);

  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(0);
  auto trO = smem_thr_copy_O.retile_S(acc_o);

  print(smem_tiled_copy_O);
  printf("\n");
  print(smem_tiled_copy_O1);
  printf("\n");
  // print(acc_o);
  // printf("\n");
  // print(trO);
  // printf("\n");
  // auto p = make_layout(make_shape(_8{}, _2{}, _4{}));
  // auto l = logical_divide(p, Shape<_2>{}); // ((2, MMA/2), MMA_M, MMA_N)
  // auto l1 = convert_layout_rowcol(acc_s.layout());

  // print(acc_s);
  // printf("\n");
  // print(l);
  // printf("\n");
  // print(l1);
  // printf("\n");
  // print(tSrQ);
  // printf("\n");

  // int *data = x;
  // printf("=== row-major 8x4 === \n");
  // auto l_row =
  //     make_layout(make_shape(_4{}, _4{}, _2{}), make_stride(_8{}, _2{},
  //     _1{}));
  // auto l = logical_divide(l_row, Shape<_2>{});
  // auto nl = make_layout(make_layout(get<0, 1>(l), get<1>(l)),
  //                       make_layout(get<0, 0>(l), get<2>(l)));
  // auto t_row = make_tensor(data, l_row);
  // auto t_new = make_tensor(data, nl);
  // auto b = make_tensor(data, make_layout(make_shape(_8{}, _4{}, _1{})));
  // auto c =
  //     logical_divide(make_layout(make_shape(_4{}, _16{}, _4{})),
  //     Shape<_2>{});

  // print(size<0>(t_new));
  // printf("\n");
  //   print(c2);

  //   // // print_layout(l_row);
  //   // print(l);
  //   // printf("\n=====================\n");
  //   // print_layout(nl);
  //   // printf("\n");
  //   // print_tensor(t_row);
  //   // printf("\n");
  //   // print_tensor(t_new);
  //   // printf("\n");

  return 0;
}

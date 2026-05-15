// Demonstrates `convert_layout_rowcol`: reshapes a tiled-MMA accumulator
// fragment from `((2,2), MMA_M, MMA_N)` into a row-major `(2*MMA_M, 2*MMA_N)`
// view. This is the trick the FA2 source uses so that the per-row max/sum
// reductions in `Softmax::softmax_rescale_o` can be written as a normal 2D
// loop instead of indexing the hierarchical MMA shape.
//
// See the blog's "Fragment Reshape" section for the derivation.
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 -I<path-to-cutlass>/include \
//     scratch/fragment_reshape.cu -o scratch/fragment_reshape && ./scratch/fragment_reshape

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

int main() {
  constexpr int kBlockM = 128;
  constexpr int kBlockN = 128;
  constexpr int kNWarps = 1;
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<Int<kNWarps>, _1, _1>>,
                            Tile<Int<16 * kNWarps>, _16, _16>>;
  TiledMma tiled_mma;

  // Show the C-fragment thread/value layout so you can see where each thread's
  // 4 values live within a 16x8 atom.
  printf("===== C fragment thread-value layout =====\n");
  print_layout(tiled_mma.get_layoutC_TV());
  printf("\n");

  // The actual reshape demo. acc_s is the partial S = QK^T fragment,
  // shaped ((2,2), MMA_M, MMA_N).
  Tensor acc_s = partition_fragment_C(
      tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

  printf("acc_s layout (raw)        : "); print(acc_s.layout()); printf("\n");
  printf("acc_s layout (row-col view): ");
  print(convert_layout_rowcol(acc_s.layout()));
  printf("\n");

  // Verify both views address the same underlying register: index ((0,1), 4, 3)
  // in the raw layout should map to the same offset as (9, 6) in the row-col
  // view (m_tile=4 -> rows 8-9, n_tile=3 -> cols 6-7; (col=0, row=1) picks
  // row 9 col 6).
  auto raw = acc_s.layout();
  auto rc  = convert_layout_rowcol(acc_s.layout());
  printf("\nraw((0,1), 4, 3) = %d\n", raw(make_coord(make_coord(0,1), 4, 3)));
  printf("rc(9, 6)         = %d\n", rc(9, 6));
  return 0;
}

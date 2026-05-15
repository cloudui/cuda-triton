// Demonstrates `convert_layout_acc_Aregs`: rebinds an MMA accumulator (C)
// fragment layout into the layout the next MMA expects for its A operand.
// Used in FA2 between QK^T and SV: we need to feed the softmaxed scores
// (the C-layout output of QK^T) as the A-layout input of SV without moving
// data, since both views point at the same physical registers.
//
// The key idea: each thread's 4-element C tile (shape (2,2)) becomes part of
// an A tile whose value layout has 8 elements per thread; we just have to
// stitch two N-adjacent C tiles together to form one A tile. logical_divide
// + make_layout does this purely as a layout rewrite.
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 -I<path-to-cutlass>/include \
//     scratch/convert_acc_to_a.cu -o scratch/convert_acc_to_a && ./scratch/convert_acc_to_a

#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

template <typename Layout>
auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  // Split the N-dim into pairs of adjacent N-tiles (each 16x16 A tile is two
  // 16x8 C tiles stitched along N).
  auto l = logical_divide(acc_layout,
                          Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N/2))
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                     get<1>(l), get<2, 1>(l));
}

int main() {
  // Mock C-fragment layout: ((2,2)=MMA, MMA_M=4, MMA_N=8).
  auto c_layout = make_layout(make_shape(make_shape(_2{}, _2{}), _4{}, _8{}));
  auto a_layout = convert_layout_acc_Aregs(c_layout);

  printf("C fragment layout: "); print(c_layout); printf("\n");
  printf("A fragment layout: "); print(a_layout); printf("\n");
  return 0;
}

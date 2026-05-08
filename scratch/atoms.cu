#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

template <typename Layout> auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  // constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  // static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  // if constexpr (mma_shape_K == 8) {
  // return acc_layout;
  // } else {
  auto l = logical_divide(acc_layout,
                          Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                     get<2, 1>(l));
  // }
};

int main() {
  // printf("=== row-major 8x4 === \n");
  // auto l_row = make_layout(make_shape(_8{}, _4{}), make_stride(_4{}, _1{}));
  // print_layout(l_row);
  // printf("\n");

  // printf("=== col-major 4x8 === \n");
  // auto l_col = make_layout(make_shape(_8{}, _4{}), make_stride(_1{}, _4{}));
  // print_layout(l_col);
  // printf("\n");
  auto l = make_layout(make_shape(make_shape(_2{}, _2{}), _4{}, _8{}));
  auto s = l.shape();
  auto stride = l.stride();
  auto t = convert_layout_acc_Aregs(l);

  auto shape_n =
      make_shape(make_shape(get<0>(s), _2{}), get<1>(s), get<2>(s) / _2{});
  auto stride_n = make_stride(make_stride(get<0>(stride), get<2>(stride)),
                              get<1>(stride), get<2>(stride) * _2{});
  auto l_n = make_layout(shape_n, stride_n);

  print(l);
  printf("\n");
  print(t);
  printf("\n");
  print(l_n);
  printf("\n");

  return 0;
}

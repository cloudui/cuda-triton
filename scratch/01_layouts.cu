#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  printf("=== row-major 8x4 === \n");
  auto l_row = make_layout(make_shape(_8{}, _4{}), make_stride(_4{}, _1{}));
  print_layout(l_row);
  printf("\n");

  printf("=== col-major 4x8 === \n");
  auto l_col = make_layout(make_shape(_8{}, _4{}), make_stride(_1{}, _4{}));
  print_layout(l_col);
  printf("\n");

  return 0;
}

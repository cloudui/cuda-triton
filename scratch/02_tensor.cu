// Wrap a raw pointer in row-major and column-major CuTe layouts and slice
// rows/columns. Used in the blog's "Tensors" section.
//
// Build & run:
//   nvcc -std=c++17 -I<cutlass>/include scratch/02_tensor.cu -o scratch/02_tensor && ./scratch/02_tensor

#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

static int x[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

int main() {

  int *data = x;
  printf("=== row-major 8x4 === \n");
  auto l_row = make_layout(make_shape(_8{}, _4{}), make_stride(_4{}, _1{}));
  auto t_row = make_tensor(data, l_row);
  print_layout(l_row);
  printf("\n");
  print_tensor(t_row);
  printf("\n");

  auto row = t_row(_0{}, _);
  print_tensor(row);
  printf("\n");

  auto col = t_row(_, _0{});
  print_tensor(col);
  printf("\n");

  printf("=== col-major 4x8 === \n");
  auto l_col = make_layout(make_shape(_8{}, _4{}), make_stride(_1{}, _8{}));
  auto t_col = make_tensor(data, l_col);
  print_layout(l_col);
  printf("\n");

  auto row1 = t_col(_0{}, _);
  print_tensor(row1);
  printf("\n");

  auto col1 = t_col(_, _0{});
  print_tensor(col1);
  printf("\n");

  return 0;
}

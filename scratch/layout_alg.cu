#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

static int x[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

int main() {

  // // (_8):(_1)
  // int *x = data;
  // auto l = make_layout(make_shape(_8{}, _4{}), make_stride(_4{}, _1{}));
  // auto l1 = make_layout(make_shape(_2{}, _4{}));
  // auto l2 = make_layout(make_shape(_4{}, _2{}));

  // auto t = make_tensor(data, l);
  // auto t1 = make_tensor(data, l1);
  // auto t2 = make_tensor(data, l2);

  auto lx = make_layout(make_shape(make_shape(_2{}, _4{}), _4{}));

  // print(t(2));
  // printf("\n");
  // print(t1(2));
  // printf("\n");
  // print(t2(2));
  // printf("\n");
  print(lx);
  printf("\n");
  print(lx(make_coord(1, 1), 1));
  print(lx(1, 2));

  return 0;
}

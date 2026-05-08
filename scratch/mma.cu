#include <cstdio>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
using namespace cute;

static int x[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

int main() {
  // using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
  //                           Layout<Shape<Int<kNWarps>, _1, _1>>,
  //                           Tile<Int<16 * kNWarps>, _16, _16>>;

  auto atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{};
  print_latex(atom);

  return 0;
}

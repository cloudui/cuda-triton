#include <cstdio>
#include <cute/tensor.hpp>
using namespace cute;

int main() {
  TiledMMA mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                Layout<Shape<_2, _2>>{}, Tile<_32, _32, _16>{});
  print_latex(mma);

  return 0;
}

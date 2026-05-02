#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace FLASH {
using namespace cute;

// default gemm
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThrCopyA,
          typename ThrCopyB>
__forceinline__ __device__ void
gemm(Tensor0 &acc,        // (MMA, MMA_M, MMA_N)        — fp32 accumulator
     Tensor1 &tCrA,       // (MMA, MMA_M, MMA_K)        — A regs (working set)
     Tensor2 &tCrB,       // (MMA, MMA_N, MMA_K)        — B regs (working set)
     Tensor3 const &tCsA, // (CPY, MMA_M, MMA_K)        — A in smem
     Tensor4 const &tCsB, // (CPY, MMA_N, MMA_K)        — B in smem
     TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
     TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
     ThrCopyB smem_thr_copy_B) {

  // retile register to match Copy Atom
  Tensor tXrA = smem_thr_copy_A.retile_D(tCrA);
  Tensor tXrB = smem_thr_copy_B.retile_D(tCrB);

  cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tXrA(_, _, _0{}));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tXrB(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); i++) {
    // prefetch next block
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tXrA(_, _, i + 1));
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tXrB(_, _, i + 1));
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// O += P · V   (A is already in registers; only B comes from smem)
// Used after softmax — P lives in registers, never in smem.
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void
gemm_rs(Tensor0 &acc,        // (MMA, MMA_M, MMA_N)      — fp32 accumulator
        Tensor1 &tCrA,       // (MMA, MMA_M, MMA_K)      — A already in regs
        Tensor2 &tCrB,       // (MMA, MMA_N, MMA_K)      — B regs (working set)
        Tensor3 const &tCsB, // (CPY, MMA_N, MMA_K)      — B in smem
        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
        ThrCopy smem_thr_copy_B) {

  Tensor tXrB = smem_thr_copy_B.retile_D(tCrB);
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tXrB(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); i++) {
    // prefetch next block
    if (i < size<2>(tCrB) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tXrB(_, _, i + 1));
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// retile to row/column layout -> convert (MMA, MMA_M, MMA_N) tile to be
// indexed like a normal array for simplicity -> (ROW, COL) <-> ((2,MMA_N), (2,
// MMA_N)) Compatible currently with SM80
template <typename Layout>
__forceinline__ __device__ auto convert_layout_rowcol(Layout const &in) {
  // (MMA, MMA_M, MMA_N), MMA=4 -> (2,2)
  auto sl = logical_divide(in, Shape<_2>{}); // ((2, MMA/2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(sl), get<1>(sl)),
                     make_layout(get<0, 0>(sl), get<2>(sl)));
}

// SM80: 16x8x16, C->A reshape for P
// P = Q@K.T, P is C frag
// O = P @ V, P is A frag
template <typename Layout>
__forceinline__ __device__ auto
convert_c_frag_to_a_frag(Layout const &acc_layout) {
  using _ = Underscore;
  // SM80 16x8x16 MMA Atom
  // A fragment layout: ((_2,_2,_2),_1,_1):((_1,_2,_4),_0,_0)
  // C fragment layout: ((_2,_2),_1,_1):((_1,_2),_0,_0)

  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout,
                          Shape<_, _, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                     get<2, 1>(l));

  // equivalent code, but less dogmatic
  // auto shape_n =
  //     make_shape(make_shape(get<0>(s), _2{}), get<1>(s), get<2>(s) / _2{});
  // auto stride_n = make_stride(make_stride(get<0>(stride), get<2>(stride)),
  //                             get<1>(stride), get<2>(stride) * _2{});
}

// copied from tridao - tensor type conversion
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto
convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

} // namespace FLASH

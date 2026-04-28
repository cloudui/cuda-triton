#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace FLASH {

////////////////////////////////////////////////////////////////////////////////
// Reduction op functors. Plug into Allreduce<N>::run(value, op).

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a > b ? a : b;
  }
};

struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Warp-lane butterfly reduction across N adjacent lanes.
// N must be a power of 2 in [2, 32]. After the call, every participating
// lane holds the same reduced value over its N-lane group.
//
// Used for row-wise reductions where one row is held by N lanes
// (e.g. for SM80_16x8x16, each MMA row is split across 4 lanes → N = 4).
// N: threads
template <int N> struct Allreduce {
  static_assert(N == 2 || N == 4 || N == 8 || N == 16 || N == 32);

  template <typename T, typename Op>
  __device__ __forceinline__ static T run(T x, Op op) {
    // every lane in the group holds the reduction.
    return Allreduce<N / 2>::run(op(x, __shfl_xor_sync(0xffffffff, x, N / 2)),
                                 op);
  }
};

template <> struct Allreduce<1> {
  template <typename T, typename Op>
  __device__ __forceinline__ static T run(T x, Op) {
    return x;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Per-row reduction over the N-mode of a (MMA, MMA_M, MMA_N) accumulator.
//
// Each thread owns kNRows partial values (one per row it covers across all
// MMA_M tiles). For each row, this:
//   1. Reduces locally over the thread's columns
//   2. Allreduces across the N lanes that share the row
// On return, every thread holding row r has the full row's reduced value
// in dst(r).

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Op>
__device__ __forceinline__ void thread_reduce_(
    Tensor<Engine0, Layout0> const &tensor, // (MMA, MMA_M, MMA_N) acc_s, fp32
    Tensor<Engine1, Layout1> &dst,          // (kNRows,)  per-row scratch
    Op op) {
#pragma unroll
  for (int row = 0; row < size<0>(dst); row++) {
    dst(row) = (zero_init) ? tensor(row, 0) : op(tensor(row, 0), dst(row));
#pragma unroll
    for (int col = 0; col < size<1>(dst); col++) {
      dst(row) = op(tensor(row, col), dst(row));
    }
  }
  // TODO:
  // 1. For each row r in dst:
  //      acc = zero_init ? -INFINITY (or 0) : dst(r)
  //      for each (col_atom, col_in_atom) covering row r:
  //          acc = op(acc, tensor(...,r,...))
  //      dst(r) = acc
}

template <int kNCols, typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Op>
__device__ __forceinline__ void
quad_allreduce_(Tensor<Engine0, Layout0> &dst, // (kNRows,) per-row reduced
                Tensor<Engine1, Layout1> &src, // (kNRows,) per-row local
                Op op) {
  // TODO: for each row r: dst(r) = Allreduce<kNCols>::run(src(r), op)
  // kNCols is the number of lanes that share a row (4 for SM80_16x8x16).
}

////////////////////////////////////////////////////////////////////////////////
// Apply per-row scale: out(r, ...) *= scale(r). Used to rescale acc_o by
// exp(m_old - m_new) when the running max changes.

template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__device__ __forceinline__ void
scale_apply_(Tensor<Engine0, Layout0> &out, // (MMA, MMA_M, MMA_K) acc_o, fp32
             Tensor<Engine1, Layout1> const &scale) { // (kNRows,) per-row
  // TODO: for each row r in out: multiply every element of that row by scale(r)
}

////////////////////////////////////////////////////////////////////////////////
// The Softmax struct: holds running (m, l) per row, owns the rescale logic.

template <int kNRows> struct Softmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));

  TensorT row_max; // running per-row max (m)
  TensorT row_sum; // running per-row sum (l)

  __device__ __forceinline__ Softmax() {};

  // The online softmax recurrence.
  // Mutates acc_s in place (S → P).
  // Mutates acc_o in place (rescale by exp(m_old - m_new)).
  // Updates row_max and row_sum.
  //
  // Is_first: skip the rescale step (no prior state to correct)
  // Check_inf: clamp -inf to a finite value (needed when masking can produce
  // all-masked rows)
  template <bool Is_first, bool Check_inf, typename Tensor0, typename Tensor1>
  __device__ __forceinline__ void
  softmax_rescale_o(Tensor0 &acc_s, // (MMA, MMA_M, MMA_N) score block, fp32
                    Tensor1 &acc_o, // (MMA, MMA_M, MMA_K) output acc, fp32
                    float softmax_scale_log2) {
    // TODO:
    // 1. row_max_prev = row_max  (save for correction)
    // 2. thread_reduce_<MaxOp>(acc_s, row_max, max_op)        // local max per
    // row
    // 3. quad_allreduce_<4>(row_max, row_max, max_op)         // warp-share row
    // max
    //    (handle Check_inf: if row_max == -INFINITY, treat as 0)
    // 4. if (!Is_first):
    //      scale = exp2((row_max_prev - row_max) * softmax_scale_log2)
    //      scale_apply_(acc_o, scale)                         // rescale prior
    //      O row_sum *= scale                                   // rescale
    //      prior l
    // 5. acc_s = exp2(acc_s * softmax_scale_log2 - row_max *
    // softmax_scale_log2)
    //    (acc_s now holds P)
    // 6. row_sum_block = thread_reduce_<SumOp>(acc_s, ...)    // local sum of
    // new P
    //    quad_allreduce_<4>(row_sum_block, ..., sum_op)
    //    row_sum = (Is_first ? row_sum_block : row_sum + row_sum_block)
  }

  // Final epilogue: divide acc_o by row_sum, compute LSE.
  // Returns log-sum-exp = m + log(l) per row.
  template <typename Tensor0>
  __device__ __forceinline__ TensorT
  normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale) {
    TensorT lse = make_tensor<float>(Shape<Int<kNRows>>{});
    // TODO:
    // 1. for each row r:
    //      inv_sum = 1.f / row_sum(r)
    //      lse(r)  = row_max(r) * softmax_scale + logf(row_sum(r))
    // 2. scale_apply_(acc_o, inv_sum)                         // O /= l
    return lse;
  }
};

} // namespace FLASH

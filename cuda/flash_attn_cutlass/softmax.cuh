#pragma once

#include "utils.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace FLASH {
using namespace cute;

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

// try cleaned up
template <int N, typename T, typename Op>
__device__ __forceinline__ T allreduce(T x, Op op) {
#pragma unroll
  for (int stride = N / 2; stride > 0; stride /= 2) {
    x = op(x, __shfl_xor_sync(0xffffffff, x, stride));
  }
  return x;
}

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
  CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(dst));
#pragma unroll
  for (int row = 0; row < size<0>(tensor); row++) {
    dst(row) = (zero_init) ? tensor(row, 0) : op(dst(row), tensor(row, 0));
#pragma unroll
    for (int col = 1; col < size<1>(tensor); col++) {
      dst(row) = op(tensor(row, col), dst(row));
    }
  }
}

// 4 threads reduce each row among each other, post per-thread reduction
// sm80 layout: https://leimao.github.io/blog/CuTe-Thread-Value-Layout/
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Op>
__device__ __forceinline__ void
quad_allreduce_(Tensor<Engine0, Layout0> &dst, // (kNRows,) per-row reduced
                Tensor<Engine1, Layout1> &src, // (kNRows,) per-row local
                Op op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
  for (int row = 0; row < size(src); row++) {
    dst(row) = allreduce<4>(src(row), op);
  }
}

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Op>
__forceinline__ __device__ void reduce_(Tensor<Engine0, Layout0> const &tensor,
                                        Tensor<Engine1, Layout1> &dst, Op op) {
  thread_reduce_<zero_init>(tensor, dst, op);
  quad_allreduce_(dst, dst, op);
}

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__forceinline__ __device__ void
reduce_sum(Tensor<Engine0, Layout0> const &tensor,
           Tensor<Engine1, Layout1> &sum) {
  // we defer allreduce only after all iterations are completed
  // so we only do curr_sum * correction + new_sum until final
  // iteration. Saves unneccessary aggregation/registers
  thread_reduce_<zero_init>(tensor, sum, SumOp{});
}

template <bool zero_init = true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1>
__forceinline__ __device__ void
reduce_max(Tensor<Engine0, Layout0> const &tensor,
           Tensor<Engine1, Layout1> &max) {
  reduce_<zero_init>(tensor, max, MaxOp{});
}

// exp
template <typename Engine0, typename Layout0, typename Engine1,
          typename Layout1>
__forceinline__ __device__ void
scale_apply_exp2(Tensor<Engine0, Layout0> &tensor,
                 Tensor<Engine1, Layout1> const &max,
                 const float softmax_scale_log2) {
#pragma unroll
  for (int r = 0; r < size<0>(tensor); r++) {
    float adj_max = max(r) * softmax_scale_log2;
#pragma unroll
    for (int c = 0; c < size<1>(tensor); c++) {
      tensor(r, c) = exp2f(tensor(r, c) * softmax_scale_log2 - adj_max);
    }
  }
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
  template <bool Is_first, typename Tensor0, typename Tensor1>
  __device__ __forceinline__ void
  softmax_rescale_o(Tensor0 &acc_s, // (MMA, MMA_M, MMA_N) score block, fp32
                    Tensor1 &acc_o, // (MMA, MMA_M, MMA_K) output acc, fp32
                    float softmax_scale_log2) {
    // reshape acc_s to ((2, MMA_M), (2, MMA_N)) for helper methods
    // purely for code interpretability
    Tensor scores =
        make_tensor(acc_s.data(), convert_layout_rowcol(acc_s.layout()));

    static_assert(decltype(size<0>(scores))::value == kNRows);
    if (Is_first) { // first block, no prevs
      FLASH::reduce_max<true>(scores, row_max);
      FLASH::scale_apply_exp2(scores, row_max, softmax_scale_log2);
      FLASH::reduce_sum<true>(scores, row_sum);
    } else {
      // save old max
      Tensor row_max_old = make_fragment_like(row_max);
      cute::copy(row_max, row_max_old);
      // max_new
      FLASH::reduce_max<false>(scores, row_max);

      // retile to linear matrix format, same reason as acc_s
      // (M_o, N_o)
      Tensor output = make_tensor(acc_o.data(),
                                  FLASH::convert_layout_rowcol(acc_o.layout()));
// apply correction to output
#pragma unroll
      for (int r = 0; r < size<0>(output); r++) {
        // exp(m_old-m_new)
        float correction =
            exp2f((row_max_old(r) - row_max(r)) * softmax_scale_log2);

        row_sum(r) *= correction;
#pragma unroll
        for (int c = 0; c < size<1>(output); c++) {
          output(r, c) *= correction;
        }
      }
      // exp2(scores-m_new)
      FLASH::scale_apply_exp2(scores, row_max, softmax_scale_log2);
      // sum(scores_exp), per thread, full reduce at end of kernel main
      FLASH::reduce_sum<false>(scores, row_sum);
    }
  }

  // Final epilogue: divide acc_o by row_sum, compute LSE.
  // Returns log-sum-exp = m + log(l) per row.
  template <typename Tensor0>
  __device__ __forceinline__ void normalize_softmax(Tensor0 &acc_o) {
    // final sum denom
    quad_allreduce_(row_sum, row_sum, SumOp{});
    Tensor output =
        make_tensor(acc_o.data(), FLASH::convert_layout_rowcol(acc_o.layout()));
    for (int r = 0; r < size<0>(output); r++) {
      float row_sum_i = 1.f / row_sum(r);
      for (int c = 0; c < size<1>(output); c++) {
        output(r, c) *= row_sum_i;
      }
    }
  }
};

} // namespace FLASH

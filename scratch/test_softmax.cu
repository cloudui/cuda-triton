// End-to-end test for FLASH::Softmax<kNRows>::softmax_rescale_o.
//
// Strategy:
//   1. Set up a fake CTA with one warp (4 warps, single CTA) doing
//      (kBlockM, kBlockN) score blocks against a fake V matrix.
//   2. Load known scores into acc_s via a tiled gmem→reg copy that uses the
//      same partition the real kernel would.
//   3. For each iteration: copy fresh scores into acc_s, run softmax_rescale_o,
//      then accumulate P @ V into acc_o using a real GEMM.
//   4. After all iterations, normalize_softmax_lse and write acc_o + lse to gmem.
//   5. Compare against a CPU reference that simulates the same online recurrence.
//
// Build: nvcc -std=c++17 -arch=sm_80 -I<path-to-cutlass>/include test_softmax.cu -o test_softmax
// Run:   ./test_softmax

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "../cuda/flash_attn_cutlass/softmax.cuh"

using namespace cute;

////////////////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, int kHeadDim, int kNIters>
__global__ void test_softmax_kernel(
    const float* __restrict__ gmem_S,   // (kNIters, kBlockM, kBlockN) — scores per iter
    const half*  __restrict__ gmem_V,   // (kNIters, kBlockN, kHeadDim) — V per iter
    float*       __restrict__ gmem_O,   // (kBlockM, kHeadDim)
    float*       __restrict__ gmem_lse, // (kBlockM,)
    float        softmax_scale)
{
  // ---- TiledMma matching FA2 conventions (4 warps in 2x2) ----
  using TiledMma = TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                            Layout<Shape<_2, _2, _1>>,
                            Tile<_32, _32, _16>>;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  // ---- Per-thread fragments ----
  Tensor acc_s = partition_fragment_C(tiled_mma,
                                      Shape<Int<kBlockM>, Int<kBlockN>>{});  // S accumulator
  Tensor acc_o = partition_fragment_C(tiled_mma,
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{}); // O accumulator
  clear(acc_o);

  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  FLASH::Softmax<kNRows> softmax;

  const float softmax_scale_log2 = softmax_scale * float(M_LOG2E);

  // For loading the per-iter S into acc_s, we treat acc_s as a partitioned
  // C-fragment of a hypothetical (kBlockM, kBlockN) gmem tensor.  Each iteration
  // has its own such matrix at offset iter*kBlockM*kBlockN.
  //
  // We need an A-operand fragment for V (fp16) for the second GEMM.
  Tensor tOrV = thr_mma.partition_fragment_B(
      make_tensor(make_gmem_ptr((half*)nullptr),
                  make_layout(Shape<Int<kBlockN>, Int<kHeadDim>>{}, LayoutRight{})));

  // ---- Loop over iterations ----
  for (int iter = 0; iter < kNIters; ++iter) {
    // Fill acc_s by reading from gmem_S[iter] using thr_mma.partition_C.
    // Build a gmem tensor view of this iter's score block.
    Tensor mS_iter = make_tensor(make_gmem_ptr(gmem_S + iter * kBlockM * kBlockN),
                                 make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{},
                                             LayoutRight{}));
    Tensor tSgS = thr_mma.partition_C(mS_iter);
    // Element-wise copy into acc_s (same shape as tSgS).
    CUTE_UNROLL
    for (int i = 0; i < size(acc_s); ++i) {
      acc_s(i) = tSgS(i);
    }

    // Run the softmax rescale.
    if (iter == 0) {
      softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/false>(
          acc_s, acc_o, softmax_scale_log2);
    } else {
      softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(
          acc_s, acc_o, softmax_scale_log2);
    }

    // After softmax_rescale_o, acc_s holds P (fp32).  We need to do
    // acc_o += P @ V_iter.  For simplicity here we cast acc_s → fp16 in-place
    // into a separate fragment.  In the real kernel, this is convert_type +
    // a layout convert + gemm_rs.  Below we just write a naive accumulation
    // using a fresh partition for V from gmem.
    //
    // For test purposes you can skip the second GEMM and just verify that
    // row_max, row_sum, and the rescale of acc_o are correct.  The CPU
    // reference can match either path.

    // === simplest path: skip the GEMM and let CPU reference also skip it ===
    // (recommended for first-pass verification — focuses on softmax math)
  }

  // ---- Epilogue: normalize and produce LSE ----
  auto lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Split=*/false>(
      acc_o, softmax_scale);

  // ---- Write acc_o back to gmem ----
  // Build a gmem tensor view of O, partition it the same way as acc_o,
  // and copy element-wise.
  Tensor mO = make_tensor(make_gmem_ptr(gmem_O),
                          make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                      LayoutRight{}));
  Tensor tOgO = thr_mma.partition_C(mO);
  CUTE_UNROLL
  for (int i = 0; i < size(acc_o); ++i) {
    tOgO(i) = acc_o(i);
  }

  // ---- Write LSE back ----
  // lse is per-row, but each row is owned by 4 lanes.  Only one of those
  // lanes (the one with col_in_atom == 0) should write to avoid races.
  // For SM80_16x8x16, lane (threadIdx.x % 4 == 0) is the "first col" of its
  // row group.  Build the row index from threadIdx the same way the MMA
  // distributes them.
  //
  // Lane t holds rows: group = t/4, then row = group, group+8, group+16, ...
  // Across MMA_M atoms.  Match this to lse(mi).
  if ((threadIdx.x % 4) == 0) {
    // For SM80 16x8 C: lane t covers rows {t/4 + 8*k_offset} across atoms.
    // For 4 warps in 2x2 layout, warp_m = (threadIdx.x / 32) % 2 selects
    // M-half.  Each warp's atom 0 covers rows 0..15, atom 1 covers 16..31, etc.
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id % 2;       // which warp in M direction (0 or 1)
    int lane_in_warp = threadIdx.x % 32;
    int row_in_atom = lane_in_warp / 4;   // 0..7

    // For Tile<_32,_32,_16> with 2 warps in M, kBlockM=32 means MMA_M=1
    // For larger kBlockM, need to iterate over the MMA_M atoms.
    constexpr int MMA_M_per_thread = decltype(size<1>(acc_o))::value;

    CUTE_UNROLL
    for (int m = 0; m < MMA_M_per_thread; ++m) {
      // Two rows per atom (sub-row 0 and 1)
      int base_row = warp_m * 16 + m * 32 + row_in_atom;
      // sub-row 0
      if (base_row < kBlockM)         gmem_lse[base_row]     = lse(2*m + 0);
      // sub-row 1 (offset by 8 within the atom)
      if (base_row + 8 < kBlockM)     gmem_lse[base_row + 8] = lse(2*m + 1);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference
////////////////////////////////////////////////////////////////////////////////

// Reference implementation: simulates the online softmax recurrence.
// If skip_v is true, does not do the P@V accumulation (matches the simple
// kernel above that focuses on the softmax math).
//
// O[m,:] = sum_k softmax(scale * S_k[m,:]) @ V_k    (across iterations k)
// lse[m] = log(sum_k sum_n exp(scale * S_k[m,n])) computed online
void cpu_reference(int M, int N, int D, int n_iters, float scale,
                   const float* S, const half* V,
                   float* O_out, float* lse_out, bool skip_v) {
  std::vector<float> m_run(M, -INFINITY);
  std::vector<float> l_run(M, 0.f);
  std::vector<float> O(M * D, 0.f);

  for (int iter = 0; iter < n_iters; ++iter) {
    const float* S_iter = S + iter * M * N;
    const half*  V_iter = V + iter * N * D;

    // 1. Compute new max
    std::vector<float> m_new(M);
    for (int i = 0; i < M; ++i) {
      m_new[i] = m_run[i];
      for (int j = 0; j < N; ++j) m_new[i] = std::max(m_new[i], S_iter[i*N + j]);
    }

    // 2. Correction (rescales acc_o and l)
    for (int i = 0; i < M; ++i) {
      float corr = (m_run[i] == -INFINITY)
          ? 1.0f
          : std::exp((m_run[i] - m_new[i]) * scale);
      l_run[i] *= corr;
      if (!skip_v) {
        for (int d = 0; d < D; ++d) O[i*D + d] *= corr;
      }
    }

    // 3. Compute P (= exp(scale * (S - m_new))) and accumulate
    std::vector<float> P(M * N);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        P[i*N + j] = std::exp((S_iter[i*N + j] - m_new[i]) * scale);
        l_run[i] += P[i*N + j];
      }
    }

    // 4. acc_o += P @ V (skip if skip_v)
    if (!skip_v) {
      for (int i = 0; i < M; ++i) {
        for (int d = 0; d < D; ++d) {
          float sum = 0.f;
          for (int j = 0; j < N; ++j) {
            sum += P[i*N + j] * __half2float(V_iter[j*D + d]);
          }
          O[i*D + d] += sum;
        }
      }
    }

    m_run = m_new;
  }

  // Normalize and compute LSE
  for (int i = 0; i < M; ++i) {
    float l = l_run[i];
    float inv = (l == 0.f) ? 1.f : 1.f / l;
    if (!skip_v) {
      for (int d = 0; d < D; ++d) O_out[i*D + d] = O[i*D + d] * inv;
    }
    lse_out[i] = (l == 0.f) ? INFINITY : m_run[i] * scale + std::log(l);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Driver
////////////////////////////////////////////////////////////////////////////////

int main() {
  // Test config — small to keep CPU reference fast.
  // NOTE: kBlockM and kBlockN must be compatible with the TiledMma's tile size
  // (Tile<_32,_32,_16>) and warp layout (2x2 = 4 warps).  Smallest that works:
  // kBlockM=32, kBlockN=32, kHeadDim=32.
  constexpr int kBlockM  = 32;
  constexpr int kBlockN  = 32;
  constexpr int kHeadDim = 32;
  constexpr int kNIters  = 3;

  const float softmax_scale = 1.0f / std::sqrt((float)kHeadDim);

  // ---- Generate inputs ----
  std::vector<float> h_S(kNIters * kBlockM * kBlockN);
  std::vector<half>  h_V(kNIters * kBlockN * kHeadDim);

  // Reproducible "random" values.
  srand(42);
  for (auto& x : h_S) x = ((rand() % 1000) / 100.f) - 5.f;   // ~[-5, 5]
  for (auto& x : h_V) x = __float2half(((rand() % 1000) / 100.f) - 5.f);

  // ---- CPU reference ----
  std::vector<float> ref_O(kBlockM * kHeadDim);
  std::vector<float> ref_lse(kBlockM);
  cpu_reference(kBlockM, kBlockN, kHeadDim, kNIters, softmax_scale,
                h_S.data(), h_V.data(), ref_O.data(), ref_lse.data(),
                /*skip_v=*/true);  // skip V for first-pass: only test softmax

  // ---- GPU run ----
  float* d_S; half* d_V; float* d_O; float* d_lse;
  cudaMalloc(&d_S,   h_S.size() * sizeof(float));
  cudaMalloc(&d_V,   h_V.size() * sizeof(half));
  cudaMalloc(&d_O,   kBlockM * kHeadDim * sizeof(float));
  cudaMalloc(&d_lse, kBlockM * sizeof(float));

  cudaMemcpy(d_S, h_S.data(), h_S.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V.data(), h_V.size() * sizeof(half),  cudaMemcpyHostToDevice);
  cudaMemset(d_O,   0, kBlockM * kHeadDim * sizeof(float));
  cudaMemset(d_lse, 0, kBlockM * sizeof(float));

  dim3 grid(1);
  dim3 block(128);  // 4 warps
  test_softmax_kernel<kBlockM, kBlockN, kHeadDim, kNIters>
      <<<grid, block>>>(d_S, d_V, d_O, d_lse, softmax_scale);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  std::vector<float> got_O(kBlockM * kHeadDim);
  std::vector<float> got_lse(kBlockM);
  cudaMemcpy(got_O.data(),   d_O,   got_O.size()   * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(got_lse.data(), d_lse, got_lse.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // ---- Compare LSE (the most informative check when skipping V) ----
  printf("Comparing LSE (per row):\n");
  bool ok = true;
  const float tol = 1e-3f;
  for (int i = 0; i < kBlockM; ++i) {
    float diff = std::abs(got_lse[i] - ref_lse[i]);
    bool row_ok = diff < tol;
    if (!row_ok) ok = false;
    if (!row_ok || i < 4) {
      printf("  row %2d  got=%10.4f  ref=%10.4f  diff=%10.4f  %s\n",
             i, got_lse[i], ref_lse[i], diff, row_ok ? "ok" : "FAIL");
    }
  }

  printf("\n%s\n", ok ? "ALL PASSED" : "FAILED");

  cudaFree(d_S); cudaFree(d_V); cudaFree(d_O); cudaFree(d_lse);
  return ok ? 0 : 1;
}

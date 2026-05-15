// Full single-warp 16x8x16 MMA with the proper gmem -> smem -> regs -> MMA
// pipeline (cp.async-style copy via TiledCopy + ldmatrix), then verifies
// against a CPU reference. Used in the blog's "Tiled MMA" + "Tiled Copy
// A, B, C" sections to show the end-to-end staging.
//
// Build & run:
//   nvcc -std=c++17 -arch=sm_80 -I<cutlass>/include scratch/04_mma.cu -o scratch/04_mma && ./scratch/04_mma

#include <cstdio>
#include <cuda_fp16.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void single_mma(half const *A, half const *B, float *C) {
  extern __shared__ half smem[];
  half *sA = smem;
  half *sB = sA + 16 * 16;

  using MMA = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

  TiledMMA<MMA, Layout<Shape<_1, _1, _1>>> tiled_mma;

  Tensor mA = make_tensor(make_gmem_ptr(A),
                          make_layout(Shape<_16, _16>{}, LayoutRight{}));
  Tensor mB = make_tensor(make_gmem_ptr(B),
                          make_layout(Shape<_8, _16>{}, LayoutRight{}));
  Tensor mC = make_tensor(make_gmem_ptr(C),
                          make_layout(Shape<_16, _8>{}, LayoutRight{}));

  TiledCopy gmem_copy_A = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
      Layout<Shape<_16, _2>, Stride<_2, _1>>{}, Layout<Shape<_1, _8>>{});
  TiledCopy gmem_copy_B = make_tiled_copy(
      Copy_Atom<UniversalCopy<uint64_t>, half_t>{},
      Layout<Shape<_8, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});

  auto smem_copy_A =
      make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, tiled_mma);
  auto smem_copy_B =
      make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, tiled_mma);

  // smem tensors (wrap raw pointers so CuTe knows they're in smem)
  Tensor smA = make_tensor(make_smem_ptr(sA),
                           make_layout(Shape<_16, _16>{}, LayoutRight{}));
  Tensor smB = make_tensor(make_smem_ptr(sB),
                           make_layout(Shape<_8, _16>{}, LayoutRight{}));

  // MMA thread slice — for fragment allocation and C partition
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor trA = thr_mma.partition_fragment_A(mA);
  Tensor trB = thr_mma.partition_fragment_B(mB);
  Tensor trC = thr_mma.partition_fragment_C(mC);
  Tensor tgC = thr_mma.partition_C(mC);

  // gmem copy thread slices — for gmem->smem
  auto gmem_thr_A = gmem_copy_A.get_thread_slice(threadIdx.x);
  auto gmem_thr_B = gmem_copy_B.get_thread_slice(threadIdx.x);
  Tensor tgA = gmem_thr_A.partition_S(mA);
  Tensor tsA = gmem_thr_A.partition_D(smA);
  Tensor tgB = gmem_thr_B.partition_S(mB);
  Tensor tsB = gmem_thr_B.partition_D(smB);

  // smem copy thread slices — for smem->regs (ldmatrix)
  auto smem_thr_A = smem_copy_A.get_thread_slice(threadIdx.x);
  auto smem_thr_B = smem_copy_B.get_thread_slice(threadIdx.x);
  Tensor tsA_r = smem_thr_A.partition_S(smA);
  Tensor tsB_r = smem_thr_B.partition_S(smB);

  // gmem->smem
  copy(gmem_copy_A, tgA, tsA);
  copy(gmem_copy_B, tgB, tsB);
  __syncthreads();

  // smem->regs (ldmatrix)
  copy(smem_copy_A, tsA_r, trA);
  copy(smem_copy_B, tsB_r, trB);

  clear(trC);
  gemm(tiled_mma, trA, trB, trC);
  copy(trC, tgC);

  if (threadIdx.x == 0) {
    printf("thread 0's A fragment layout: ");
    print(trA.layout());
    printf("\n");
    printf("thread 0's C fragment layout: ");
    print(trC.layout());
    printf("\n");
  }
}

int main() {
  constexpr int M = 16, N = 8, K = 16;
  half hA[M * K], hB[N * K];
  float hC[M * N], ref[M * N] = {};

  for (int i = 0; i < M * K; i++) {
    hA[i] = __float2half(i);
  }

  for (int i = 0; i < M * N; i++) {
    hB[i] = __float2half(i * 0.5f - 3.2f);
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float dot = 0.f;
      for (int k = 0; k < K; k++) {
        dot += __half2float(hA[m * K + k]) * __half2float(hB[n * K + k]);
      }
      ref[m * N + n] = dot;
    }
  }

  half *dA, *dB;
  float *dC;
  cudaMalloc(&dA, sizeof(hA));
  cudaMalloc(&dB, sizeof(hB));
  cudaMalloc(&dC, sizeof(hC));

  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

  constexpr int smem_size = M * K + N * K;
  single_mma<<<1, 32, smem_size * sizeof(half)>>>(dA, dB, dC);
  cudaDeviceSynchronize();
  cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

  int errors = 0;
  for (int i = 0; i < M * N; i++) {
    printf("%f %f\n", hC[i], ref[i]);
    if (fabsf(hC[i] - ref[i]) > 1e-2f) {
      errors++;
    }
  }

  printf("errors: %d/%d\n", errors, M * N);
}

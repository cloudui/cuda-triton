// Single-warp `SM80_16x8x16` MMA, no SMEM staging: A/B come straight from
// gmem partitioned via the MMA fragment layout. The simplest possible
// "I issued one tensor-core instruction" demo. Used in the blog's "Tiled MMA"
// section.
//
// Build & run:
//   nvcc -std=c++17 -arch=sm_80 -I<cutlass>/include scratch/03_mma.cu -o scratch/03_mma && ./scratch/03_mma

#include <cstdio>
#include <cuda_fp16.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void single_mma(half const *A, half const *B, float *C) {
  using MMA = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
  TiledMMA<MMA, Layout<Shape<_1, _1, _1>>> tiled_mma;

  Tensor mA = make_tensor(make_gmem_ptr(A),
                          make_layout(Shape<_16, _16>{}, LayoutRight{}));
  Tensor mB = make_tensor(make_gmem_ptr(B),
                          make_layout(Shape<_8, _16>{}, LayoutRight{}));
  Tensor mC = make_tensor(make_gmem_ptr(C),
                          make_layout(Shape<_16, _8>{}, LayoutRight{}));

  //   print_latex(tiled_mma);
  auto t_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCrA = t_mma.partition_fragment_A(mA);
  Tensor tCrB = t_mma.partition_fragment_B(mB);
  Tensor tCrC = t_mma.partition_fragment_C(mC);

  Tensor tCgA = t_mma.partition_A(mA);
  Tensor tCgB = t_mma.partition_B(mB);
  Tensor tCgC = t_mma.partition_C(mC);
  copy(tCgA, tCrA);
  copy(tCgB, tCrB);
  clear(tCrC);

  gemm(tiled_mma, tCrA, tCrB, tCrC);
  copy(tCrC, tCgC);

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

  single_mma<<<1, 32>>>(dA, dB, dC);
  cudaDeviceSynchronize();
  cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);

  int errors = 0;
  for (int i = 0; i < M * N; i++) {
    if (fabsf(hC[i] - ref[i]) > 1e-2f) {
      errors++;
    }
  }

  printf("errors: %f/%f\n", errors, M * N);
}

// Runnable end-to-end test for FLASH::flash_fwd_kernel.
//
// Compares against a CPU reference (naive softmax(Q·K^T)·V) on a small input.
//
// Build:
//   nvcc -std=c++17 -arch=sm_80 \
//       --expt-relaxed-constexpr --expt-extended-lambda \
//       -I/data/users/echen314/eric/triton/cuda/flash_attn_cutlass \
//       -I/data/users/echen314/eric/triton/third_party/cutlass/include \
//       scratch/fa_test.cu -o /tmp/fa_test
//
// Run:
//   /tmp/fa_test

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "kernel_traits.cuh"

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU reference: naive softmax(Q · K^T / sqrt(d)) · V
//
// Shapes (host, simple row-major):
//   Q: (B, H, Sq, D)
//   K: (B, H, Sk, D)
//   V: (B, H, Sk, D)
//   O: (B, H, Sq, D)
////////////////////////////////////////////////////////////////////////////////
void cpu_reference(const std::vector<half> &Q, const std::vector<half> &K,
                   const std::vector<half> &V, std::vector<half> &O, int B,
                   int H, int Sq, int Sk, int D) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  std::vector<float> S(Sq * Sk);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      // Compute S = Q · K^T * scale, shape (Sq, Sk)
      for (int i = 0; i < Sq; ++i) {
        for (int j = 0; j < Sk; ++j) {
          float dot = 0.f;
          for (int d = 0; d < D; ++d) {
            float q = __half2float(Q[((b * H + h) * Sq + i) * D + d]);
            float k = __half2float(K[((b * H + h) * Sk + j) * D + d]);
            dot += q * k;
          }
          S[i * Sk + j] = dot * scale;
        }
      }

      // Softmax row-wise
      for (int i = 0; i < Sq; ++i) {
        float maxv = -INFINITY;
        for (int j = 0; j < Sk; ++j)
          maxv = std::max(maxv, S[i * Sk + j]);
        float sumv = 0.f;
        for (int j = 0; j < Sk; ++j) {
          S[i * Sk + j] = std::exp(S[i * Sk + j] - maxv);
          sumv += S[i * Sk + j];
        }
        float inv = 1.f / sumv;
        for (int j = 0; j < Sk; ++j)
          S[i * Sk + j] *= inv;
      }

      // O = P · V, shape (Sq, D)
      for (int i = 0; i < Sq; ++i) {
        for (int d = 0; d < D; ++d) {
          float sum = 0.f;
          for (int j = 0; j < Sk; ++j) {
            float v = __half2float(V[((b * H + h) * Sk + j) * D + d]);
            sum += S[i * Sk + j] * v;
          }
          O[((b * H + h) * Sq + i) * D + d] = __float2half(sum);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Driver
////////////////////////////////////////////////////////////////////////////////
int main() {
  // ---- Test config (head_dim = 64 → use Traits_hdim64) ----
  using Traits = Traits_hdim64; // kBlockM=128, kBlockN=64, kHeadDim=64, kNWarps=4

  constexpr int B = 1;
  constexpr int H = 1;
  constexpr int Sq = Traits::kBlockM; // 128
  constexpr int Sk = Traits::kBlockN; // 64
  constexpr int D = Traits::kHeadDim; // 64

  // ---- Generate random inputs ----
  std::vector<half> hQ(B * H * Sq * D);
  std::vector<half> hK(B * H * Sk * D);
  std::vector<half> hV(B * H * Sk * D);
  std::vector<half> hO(B * H * Sq * D, __float2half(0.f));

  std::srand(42);
  auto rand_half = []() {
    return __float2half(((std::rand() % 1000) / 500.0f - 1.0f) * 0.5f); // [-0.5, 0.5)
  };
  for (auto &x : hQ)
    x = rand_half();
  for (auto &x : hK)
    x = rand_half();
  for (auto &x : hV)
    x = rand_half();

  // ---- CPU reference ----
  std::vector<half> hO_ref(B * H * Sq * D);
  cpu_reference(hQ, hK, hV, hO_ref, B, H, Sq, Sk, D);

  // ---- GPU buffers ----
  half *dQ, *dK, *dV, *dO;
  float *dLSE;
  CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dO, hO.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dLSE, B * H * Sq * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(dK, hK.data(), hK.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(dV, hV.data(), hV.size() * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hO.size() * sizeof(half)));

  // ---- Set up Flash_fwd_params ----
  Flash_fwd_params params{};
  params.q_ptr = dQ;
  params.k_ptr = dK;
  params.v_ptr = dV;
  params.o_ptr = dO;
  params.softmax_lse_ptr = dLSE;

  params.batch_size = B;
  params.seqlen_q = Sq;
  params.seqlen_k = Sk;
  params.num_heads = H;
  params.num_heads_k = H;
  params.head_dim = D;

  // Strides assuming row-major (B, H, S, D) → (H*S*D, S*D, D, 1) flat layout:
  // The kernel indexes Q as Q[batch_idx*q_batch_stride + head_idx*q_head_stride
  //                          + seq_idx*q_row_stride + d]
  // For (B, H, S, D) row-major: q_batch = H*Sq*D, q_head = Sq*D, q_row = D
  params.q_batch_stride = H * Sq * D;
  params.q_head_stride = Sq * D;
  params.q_row_stride = D;

  params.k_batch_stride = H * Sk * D;
  params.k_head_stride = Sk * D;
  params.k_row_stride = D;

  params.v_batch_stride = H * Sk * D;
  params.v_head_stride = Sk * D;
  params.v_row_stride = D;

  params.o_batch_stride = H * Sq * D;
  params.o_head_stride = Sq * D;
  params.o_row_stride = D;

  params.scale_softmax = 1.0f / std::sqrt(static_cast<float>(D));
  params.scale_softmax_log2 =
      params.scale_softmax * static_cast<float>(M_LOG2E);
  params.is_causal = false;

  // ---- Launch ----
  dim3 grid(/*m_blocks=*/(Sq + Traits::kBlockM - 1) / Traits::kBlockM,
            /*batch*heads=*/B * H,
            /*split=*/1);
  dim3 block(Traits::kNThreads);

  // Smem footprint: sQ + sK + sV (sO reuses sQ)
  // sQ: kBlockM × kHeadDim halfs
  // sK, sV: kBlockN × kHeadDim halfs each
  int smem_size =
      sizeof(half) * (Traits::kBlockM * Traits::kHeadDim     // sQ
                      + 2 * Traits::kBlockN * Traits::kHeadDim); // sK, sV
  printf("Smem size: %d bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);

  // Opt in to extra dynamic smem if needed
  CUDA_CHECK(cudaFuncSetAttribute(
      FLASH::flash_fwd_kernel<Traits, /*Is_causal=*/false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  printf("Launching kernel: grid (%d, %d, %d), block (%d), smem %d\n", grid.x,
         grid.y, grid.z, block.x, smem_size);

  FLASH::flash_fwd_kernel<Traits, /*Is_causal=*/false>
      <<<grid, block, smem_size>>>(params);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  // ---- Copy back and compare ----
  CUDA_CHECK(
      cudaMemcpy(hO.data(), dO, hO.size() * sizeof(half), cudaMemcpyDeviceToHost));

  // Compare element-by-element
  printf("\nComparing GPU output to CPU reference:\n");
  int mismatches = 0;
  float max_abs_err = 0.f;
  float total_abs_err = 0.f;
  const float tol = 5e-2f; // fp16 accumulation through fp32 acc, generous tol

  for (size_t i = 0; i < hO.size(); ++i) {
    float gpu = __half2float(hO[i]);
    float ref = __half2float(hO_ref[i]);
    float err = std::abs(gpu - ref);
    max_abs_err = std::max(max_abs_err, err);
    total_abs_err += err;
    if (err > tol) {
      if (mismatches < 10) {
        printf("  idx %zu: gpu=%+.4f ref=%+.4f err=%.4f\n", i, gpu, ref, err);
      }
      mismatches++;
    }
  }

  float avg_abs_err = total_abs_err / hO.size();
  printf("\nMax abs err:   %.6f\n", max_abs_err);
  printf("Mean abs err:  %.6f\n", avg_abs_err);
  printf("Mismatches:    %d / %zu (tol=%.4f)\n", mismatches, hO.size(), tol);

  // Print first few outputs for spot check
  printf("\nFirst 8 GPU outputs vs CPU ref:\n");
  for (int i = 0; i < 8 && i < (int)hO.size(); ++i) {
    printf("  [%d]  gpu=%+.4f  ref=%+.4f\n", i, __half2float(hO[i]),
           __half2float(hO_ref[i]));
  }

  bool pass = (mismatches == 0) && (max_abs_err < tol);
  printf("\n%s\n", pass ? "PASSED" : "FAILED");

  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dO);
  cudaFree(dLSE);

  return pass ? 0 : 1;
}

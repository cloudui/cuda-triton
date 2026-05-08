// Standalone test for FLASH::Allreduce<N>::run.
//
// Each lane starts with its own value (its lane id). After the butterfly
// reduction, every lane in the N-lane group should hold the same combined
// value.
//
// Build: nvcc -std=c++17 -arch=sm_80 -I<path-to-cutlass>/include test_allreduce.cu -o test_allreduce
// Run:   ./test_allreduce

#include <cstdio>
#include <cuda_runtime.h>

#include "../cuda/flash_attn_cutlass/softmax.cuh"

// One block, one warp.  Each lane grabs threadIdx.x as its starting value.
template <int N>
__global__ void allreduce_max_kernel(float* out) {
  float v = static_cast<float>(threadIdx.x);
  FLASH::MaxOp max_op;
  float r = FLASH::Allreduce<N>::run(v, max_op);
  out[threadIdx.x] = r;
}

template <int N>
__global__ void allreduce_sum_kernel(float* out) {
  float v = static_cast<float>(threadIdx.x);
  FLASH::SumOp sum_op;
  float r = FLASH::Allreduce<N>::run(v, sum_op);
  out[threadIdx.x] = r;
}

template <int N>
bool check_max(const float* h_out) {
  // Lanes group as [0..N-1], [N..2N-1], ...
  // Group g's max = highest lane id in group = g*N + (N-1)
  for (int lane = 0; lane < 32; ++lane) {
    int group = lane / N;
    float expected = static_cast<float>(group * N + (N - 1));
    if (h_out[lane] != expected) {
      printf("  Allreduce<%d> max FAIL  lane=%d  got=%.1f  expected=%.1f\n",
             N, lane, h_out[lane], expected);
      return false;
    }
  }
  printf("  Allreduce<%d> max OK\n", N);
  return true;
}

template <int N>
bool check_sum(const float* h_out) {
  // Group g's sum = sum_{i=0..N-1} (g*N + i) = g*N*N + N*(N-1)/2
  for (int lane = 0; lane < 32; ++lane) {
    int group = lane / N;
    float expected = static_cast<float>(group * N * N + N * (N - 1) / 2);
    if (h_out[lane] != expected) {
      printf("  Allreduce<%d> sum FAIL  lane=%d  got=%.1f  expected=%.1f\n",
             N, lane, h_out[lane], expected);
      return false;
    }
  }
  printf("  Allreduce<%d> sum OK\n", N);
  return true;
}

template <int N>
bool run_one() {
  float* d_out;
  cudaMalloc(&d_out, 32 * sizeof(float));
  float h_out[32];

  allreduce_max_kernel<N><<<1, 32>>>(d_out);
  cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok_max = check_max<N>(h_out);

  allreduce_sum_kernel<N><<<1, 32>>>(d_out);
  cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok_sum = check_sum<N>(h_out);

  cudaFree(d_out);
  return ok_max && ok_sum;
}

int main() {
  bool all_ok = true;
  printf("Testing FLASH::Allreduce<N>::run\n");
  all_ok &= run_one<2>();
  all_ok &= run_one<4>();
  all_ok &= run_one<8>();
  all_ok &= run_one<16>();
  all_ok &= run_one<32>();
  printf("\n%s\n", all_ok ? "ALL PASSED" : "FAILED");
  return all_ok ? 0 : 1;
}

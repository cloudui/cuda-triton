// Tiled GEMM kernel using TN layout -> A: (M x K) B: (N x K)
// K is contiguous dim. Used in the blog's "MMA Loop: QK^T GEMM" section as
// the standalone reference for the inner GEMM pattern (without the surrounding
// FA2 softmax/online-rescale machinery).
//
// Build & run:
//   nvcc -std=c++17 -arch=sm_80 -I<cutlass>/include scratch/05_gemm.cu -o scratch/05_gemm && ./scratch/05_gemm
#include <cstdio>
#include <cuda_fp16.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// Block tile sizes — each thread block computes a BLK_M x BLK_N tile of C
// BLK_K for K dim
constexpr int BLK_M = 64, BLK_N = 64, BLK_K = 16;
constexpr int THREADS = 128; // 4 warps

auto bM = _128{};
auto bN = _128{};
auto bK = _64{};

__global__ void tiled_gemm(half const *A, half const *B, float *C, int m, int n,
                           int k) {
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto shape = make_shape(M, N, K);

  // extern __shared__ half smem[];
  // half *sA = smem;
  // half *sB = sA + BLK_M * BLK_K;

  auto strideA = make_stride(K, _1{});
  auto strideB = make_stride(K, _1{});
  auto strideC = make_stride(_1{}, N);

  TiledCopy copyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<_1, _8>{});
  TiledCopy copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<_1, _8>{});

  // 128 threads, 4 warps, 16
  TiledMMA mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                Layout<Shape<_2, _2>>{}, Tile<_32, _32, _16>{});

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_B;

  Tensor mA = make_tensor(make_gmem_ptr(A), Shape<M, K>, dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), Shape<N, K>, dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), Shape<M, N>, dC); // (M,N)

  auto cta_tiler = make_shape(bM, bN, bK);
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

  // gA = mA + blockIdx.x * K * BLOCK_M
  Tensor gA = local_tile(mA, cta_tiler, cta_coord,
                         Step<_1, _, _1>{}); // (BLK_M,BLK_K,k)
  // B: gB = mB + blockIdx.y * K * BLOCK_N
  Tensor gB = local_tile(mB, cta_tiler, cta_coord,
                         Step<_, _1, _1>{}); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, _>{}); // (BLK_M,BLK_N)

  // A type, B type
  using SharedStorage = SharedStorage<half_t, half_t, ASmemLayout, BSmemLayout>;
  SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()),
                          sA_layout); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()),
                          sB_layout); // (BLK_N,BLK_K,PIPE)

  // yeah this just works, not sure why
  auto swizzle_atom = composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{});
}

int main() {
  constexpr int M = 256, N = 256, K = 256;
  // TODO: allocate hA, hB (half), hC, ref (float)
  // TODO: fill hA and hB with test values
  // TODO: compute ref on CPU (triple loop)
  // TODO: cudaMalloc + cudaMemcpy H2Ddoes
  // TODO: launch tiled_gemm — figure out grid (M/BLK_M, N/BLK_N), block
  // (THREADS,)
  //       smem = (BLK_M*BLK_K + BLK_K*BLK_N) * sizeof(half)
  // TODO: cudaMemcpy D2H, count errors (threshold 1e-1f for fp16 accumulation
  // noise)
  // TODO: printf errors
}

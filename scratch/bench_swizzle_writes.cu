// Single-kernel-per-invocation benchmark for measuring whether Swizzle<3,3,3>
// affects bank conflicts on cp.async (LDGSTS) writes and on STS.128 writes,
// under FA2's hdim=64 thread layout (16 thread-rows × 8 thread-cols, 8 halfs
// per thread).
//
// Usage:
//   bench_swz <mode>
//     0 = cp.async swizzled
//     1 = cp.async unswizzled
//     2 = STS.128 swizzled
//     3 = STS.128 unswizzled

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

constexpr int kBlockM = 64;
constexpr int kBlockK = 64;
constexpr int kNThreads = 128;
constexpr int kThreadsPerRow = 8;
constexpr int kThreadRows = kNThreads / kThreadsPerRow;

using AtomNoSwz =
    Layout<Shape<_8, Int<kBlockK>>, Stride<Int<kBlockK>, _1>>;
using AtomSwz =
    decltype(composition(Swizzle<3, 3, 3>{}, AtomNoSwz{}));

using LayoutNoSwz = decltype(tile_to_shape(
    AtomNoSwz{}, Shape<Int<kBlockM>, Int<kBlockK>>{}));
using LayoutSwz = decltype(tile_to_shape(
    AtomSwz{}, Shape<Int<kBlockM>, Int<kBlockK>>{}));

using GmemThrLayout = Layout<
    Shape<Int<kThreadRows>, Int<kThreadsPerRow>>,
    Stride<Int<kThreadsPerRow>, _1>>;

template <typename SmemLayout, int ITERS>
__global__ void bench_cp_async(const half *__restrict__ gQ_ptr,
                               half *__restrict__ sink) {
  __shared__ __align__(128) half smem[cosize_v<SmemLayout>];

  auto sQ = make_tensor(make_smem_ptr(smem), SmemLayout{});
  auto mQ = make_tensor(make_gmem_ptr(gQ_ptr),
                        Shape<Int<kBlockM>, Int<kBlockK>>{},
                        Stride<Int<kBlockK>, _1>{});

  using CpAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
  TiledCopy g2s = make_tiled_copy(CpAtom{}, GmemThrLayout{},
                                  Layout<Shape<_1, _8>>{});
  ThrCopy thr = g2s.get_thread_slice(threadIdx.x);
  auto tQgQ = thr.partition_S(mQ);
  auto tQsQ = thr.partition_D(sQ);

#pragma unroll 1
  for (int i = 0; i < ITERS; ++i) {
    copy(g2s, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
  }
  if (threadIdx.x == 0xffff) sink[blockIdx.x] = smem[0];
}

template <typename SmemLayout, int ITERS>
__global__ void bench_sts(const half *__restrict__ gQ_ptr,
                          half *__restrict__ sink) {
  __shared__ __align__(128) half smem[cosize_v<SmemLayout>];

  auto sQ = make_tensor(make_smem_ptr(smem), SmemLayout{});
  auto mQ = make_tensor(make_gmem_ptr(gQ_ptr),
                        Shape<Int<kBlockM>, Int<kBlockK>>{},
                        Stride<Int<kBlockK>, _1>{});

  using GAtom =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half_t>;
  using SAtom = Copy_Atom<UniversalCopy<uint128_t>, half_t>;

  TiledCopy g2r = make_tiled_copy(GAtom{}, GmemThrLayout{},
                                  Layout<Shape<_1, _8>>{});
  TiledCopy r2s = make_tiled_copy(SAtom{}, GmemThrLayout{},
                                  Layout<Shape<_1, _8>>{});
  ThrCopy g2r_thr = g2r.get_thread_slice(threadIdx.x);
  ThrCopy r2s_thr = r2s.get_thread_slice(threadIdx.x);

  auto tQgQ = g2r_thr.partition_S(mQ);
  auto tQrQ = make_fragment_like(g2r_thr.partition_D(sQ));
  auto tQsQ = r2s_thr.partition_D(sQ);

  copy(g2r, tQgQ, tQrQ);
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < ITERS; ++i) {
    copy(r2s, tQrQ, tQsQ);
    __syncthreads();
  }
  if (threadIdx.x == 0xffff) sink[blockIdx.x] = smem[0];
}

template <typename Kernel>
float run(Kernel kernel, int blocks, half *in, half *out, int reps) {
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (int i = 0; i < 3; ++i) kernel<<<blocks, kNThreads>>>(in, out);
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i = 0; i < reps; ++i) kernel<<<blocks, kNThreads>>>(in, out);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms;
  cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  return ms / reps;
}

int main(int argc, char **argv) {
  constexpr int ITERS = 4000;
  constexpr int REPS = 20;
  constexpr int BLOCKS = 108;
  constexpr int N = kBlockM * kBlockK;

  int mode = (argc > 1) ? atoi(argv[1]) : 0;

  half *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(half));
  cudaMalloc(&d_out, BLOCKS * sizeof(half));
  cudaMemset(d_in, 0, N * sizeof(half));

  const char *names[] = {"cp.async swizzled", "cp.async unswizzled",
                         "STS.128  swizzled", "STS.128  unswizzled"};
  float ms = 0;
  switch (mode) {
    case 0: ms = run(bench_cp_async<LayoutSwz,   ITERS>, BLOCKS, d_in, d_out, REPS); break;
    case 1: ms = run(bench_cp_async<LayoutNoSwz, ITERS>, BLOCKS, d_in, d_out, REPS); break;
    case 2: ms = run(bench_sts     <LayoutSwz,   ITERS>, BLOCKS, d_in, d_out, REPS); break;
    case 3: ms = run(bench_sts     <LayoutNoSwz, ITERS>, BLOCKS, d_in, d_out, REPS); break;
    default: fprintf(stderr, "mode 0..3\n"); return 1;
  }
  printf("mode=%d (%s)  ITERS/blk=%d REPS=%d BLOCKS=%d  -> %7.3f ms/launch\n",
         mode, names[mode], ITERS, REPS, BLOCKS, ms);

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}

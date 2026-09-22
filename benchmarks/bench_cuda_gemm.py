"""
Benchmark: PyTorch vs CUDA gemm

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_gemm.py

NOTE: cuda/gemm.cu indexes B with row-stride N instead of K, so it only
produces valid results for square (M, K, N) with K == N, and what it
computes in that case is `A @ B.T` (not `A @ B`) — see the comment on
TestCUDAGemm in tests/test_kernels.py. This benchmark sticks to square
shapes and benchmarks against `A @ B.T` to match.
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")

try:
    import cuda_kernels
except ImportError:
    print("CUDA extension not found. Build it first with: make build-cuda")
    sys.exit(1)


def benchmark():
    sizes = [128, 256, 512, 1024, 2048, 4096]

    print(f"{'NxN':<12} {'PyTorch (ms)':<15} {'CUDA (ms)':<14} {'CUDA vs PyT'}")
    print("-" * 55)

    for n in sizes:
        A = torch.randn(n, n, device="cuda", dtype=torch.float32)
        B = torch.randn(n, n, device="cuda", dtype=torch.float32)

        ms_pytorch = do_bench(lambda: A @ B.T)
        ms_cuda = do_bench(lambda: cuda_kernels.gemm(A, B))

        speedup = ms_pytorch / ms_cuda

        print(f"{n:<12} {ms_pytorch:<15.4f} {ms_cuda:<14.4f} {speedup:.2f}x")

    print()
    print(
        "Expect CUDA to lag PyTorch by a wide margin here: the naive kernel\n"
        "does one global-memory read per multiply-add with no shared-memory\n"
        "tiling and no register blocking, vs. cuBLAS's tiled/blocked kernels."
    )


if __name__ == "__main__":
    benchmark()

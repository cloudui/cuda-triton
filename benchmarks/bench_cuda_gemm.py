"""
Benchmark: PyTorch vs CUDA gemm

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_gemm.py

A: MxK, B: KxN, output: MxN — standard row-major GEMM.

NOTE: cuda/gemm.cu still writes `output[row * N + col]` outside the
`row < M && col < N` bounds check, so this sticks to M/N that are
multiples of the 16x16 thread block to avoid an out-of-bounds write.
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
    # (M, K, N) — square sizes plus a couple of rectangular shapes.
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 1024),
        (4096, 1024, 4096),
    ]

    print(f"{'MxKxN':<18} {'PyTorch (ms)':<15} {'CUDA (ms)':<14} {'CUDA vs PyT'}")
    print("-" * 62)

    for M, K, N in sizes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)

        ms_pytorch = do_bench(lambda: A @ B)
        ms_cuda = do_bench(lambda: cuda_kernels.gemm(A, B))

        speedup = ms_pytorch / ms_cuda

        print(f"{f'{M}x{K}x{N}':<18} {ms_pytorch:<15.4f} {ms_cuda:<14.4f} {speedup:.2f}x")

    print()
    print(
        "Expect CUDA to lag PyTorch by a wide margin here: the naive kernel\n"
        "does one global-memory read per multiply-add with no shared-memory\n"
        "tiling and no register blocking, vs. cuBLAS's tiled/blocked kernels."
    )


if __name__ == "__main__":
    benchmark()

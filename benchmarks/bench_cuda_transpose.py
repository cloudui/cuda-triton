"""
Benchmark: PyTorch vs CUDA transpose

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_transpose.py
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
    sizes = [(128, 128), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096), (1024, 4096), (16384, 16384)]

    print(f"{'MxN':<16} {'PyTorch (ms)':<15} {'CUDA (ms)':<14} {'CUDA vs PyT'}")
    print("-" * 60)

    for M, N in sizes:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)

        ms_pytorch = do_bench(lambda: x.t().contiguous())
        ms_cuda = do_bench(lambda: cuda_kernels.transpose(x))

        speedup = ms_pytorch / ms_cuda

        print(f"{f'{M}x{N}':<16} {ms_pytorch:<15.4f} {ms_cuda:<14.4f} {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()

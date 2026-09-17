"""
Benchmark: PyTorch vs CUDA naive_reduce (full-tensor sum)

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_naive_reduce.py
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
    sizes = [1024, 65536, 1048576, 8388608, 67108864]

    print(f"{'N':<12} {'PyTorch (ms)':<15} {'CUDA (ms)':<14} {'CUDA vs PyT'}")
    print("-" * 55)

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        ms_pytorch = do_bench(lambda: x.sum())
        ms_cuda = do_bench(lambda: cuda_kernels.naive_reduce(x))

        speedup = ms_pytorch / ms_cuda

        print(f"{n:<12} {ms_pytorch:<15.4f} {ms_cuda:<14.4f} {speedup:.2f}x")

    print()
    print(
        "Expect CUDA to lag PyTorch here: naive_reduce does one kernel launch per\n"
        "tree level (log_256(N) launches) with no grid-stride loop or warp shuffle,\n"
        "vs. PyTorch's single-pass, vectorized, warp-shuffle reduction."
    )


if __name__ == "__main__":
    benchmark()

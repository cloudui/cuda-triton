"""
Benchmark: PyTorch vs Triton vs CUDA vector_add

Run on a GPU machine (requires `make build-cuda` first):
  python benchmarks/bench_cuda_vector_add.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.vector_add import vector_add_pytorch, vector_add

try:
    import cuda_kernels
except ImportError:
    print("CUDA extension not found. Build it first with: make build-cuda")
    sys.exit(1)


def benchmark():
    sizes = [1024, 65536, 1048576, 8388608, 67108864]

    print(f"{'N':<12} {'PyTorch (ms)':<15} {'Triton (ms)':<14} {'CUDA (ms)':<14} {'CUDA vs PyT':<12} {'CUDA vs Tri'}")
    print("-" * 80)

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        ms_pytorch = do_bench(lambda: vector_add_pytorch(x, y))
        ms_triton = do_bench(lambda: vector_add(x, y))
        ms_cuda = do_bench(lambda: cuda_kernels.vector_add(x, y))

        speedup_pytorch = ms_pytorch / ms_cuda
        speedup_triton = ms_triton / ms_cuda

        print(
            f"{n:<12} {ms_pytorch:<15.4f} {ms_triton:<14.4f} {ms_cuda:<14.4f} {speedup_pytorch:.2f}x{'':<7} {speedup_triton:.2f}x"
        )


if __name__ == "__main__":
    benchmark()

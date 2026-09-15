"""
Benchmark: PyTorch vs Triton vector_add

Run on a GPU machine:
  python benchmarks/bench_vector_add.py
"""

import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, ".")
from kernels.vector_add import vector_add_pytorch, vector_add


def benchmark():
    sizes = [1024, 65536, 1048576, 8388608, 67108864]

    print(f"{'N':<12} {'PyTorch (ms)':<15} {'Triton (ms)':<14} {'Speedup'}")
    print("-" * 55)

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        ms_pytorch = do_bench(lambda: vector_add_pytorch(x, y))
        ms_triton = do_bench(lambda: vector_add(x, y))

        speedup = ms_pytorch / ms_triton

        print(f"{n:<12} {ms_pytorch:<15.4f} {ms_triton:<14.4f} {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()

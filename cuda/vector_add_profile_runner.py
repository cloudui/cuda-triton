"""
Single-launch profile harness for the vector_add CUDA kernel.

Used by `make prof-vector-add` / `make prof-nsys-vector-add` to give Nsight
Compute / Nsight Systems a focused, deterministic workload (one size, multiple
launches so ncu can skip warmup and profile a steady-state run).

Usage:
    python cuda/vector_add_profile_runner.py [--n N] [--iters N]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_kernels  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1 << 26)  # ~64M elements, 256MB/tensor
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.n, device="cuda", dtype=torch.float32)
    y = torch.randn(args.n, device="cuda", dtype=torch.float32)

    for _ in range(args.iters):
        cuda_kernels.vector_add(x, y)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

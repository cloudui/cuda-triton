"""
Single-launch profile harness for the naive_reduce CUDA kernel.

Used by `make prof-naive-reduce` / `make prof-nsys-naive-reduce` to give
Nsight Compute / Nsight Systems a focused, deterministic workload (one size,
multiple launches so ncu can skip warmup and profile a steady-state run).

Note: each call to naive_reduce() issues multiple kernel launches internally
(one per tree level), so --launch-count 1 in the ncu Makefile target profiles
just the first-level (largest) launch, which dominates runtime.

Usage:
    python cuda/naive_reduce_profile_runner.py [--n N] [--iters N]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda_kernels  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1 << 24)  # ~16M elements, 64MB
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.n, device="cuda", dtype=torch.float32)

    for _ in range(args.iters):
        cuda_kernels.naive_reduce(x)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

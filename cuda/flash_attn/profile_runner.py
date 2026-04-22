"""
Single-launch profile harness for the WMMA FlashAttention kernel.

Used by `make prof-fac` to give Nsight Compute a focused, deterministic workload
(one shape, multiple launches so ncu can skip warmup and profile a steady-state run).

Usage:
    python profile_runner.py [--seq N] [--batch B] [--heads H] [--head-dim D] [--causal] [--iters N]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flash_attn_cuda  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64, choices=[64, 128])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    shape = (args.batch, args.seq, args.heads, args.head_dim)
    q = torch.randn(*shape, device="cuda", dtype=torch.float16)
    k = torch.randn(*shape, device="cuda", dtype=torch.float16)
    v = torch.randn(*shape, device="cuda", dtype=torch.float16)

    for _ in range(args.iters):
        flash_attn_cuda.mha_fwd(q, k, v, args.causal)

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

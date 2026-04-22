"""
Benchmark: WMMA CUDA FlashAttention vs PyTorch SDPA (Tri Dao's CUDA FA)

Run on a GPU machine (requires `make build-fa` first):
  make bench-fa
  # or directly:
  LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python benchmarks/bench_cuda_flash_attention.py
"""

import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

sys.path.insert(0, ".")
sys.path.insert(0, "cuda/flash_attn")

try:
    import flash_attn_cuda
except ImportError:
    print("flash_attn_cuda extension not built. Build with: make build-fa")
    sys.exit(1)


def benchmark():
    batch = 4
    n_heads = 8
    sizes = [128, 256, 512, 1024, 2048]

    for head_dim in [64, 128]:
        print(f"\n{'=' * 90}")
        print(f"  WMMA CUDA FlashAttention  (batch={batch}, heads={n_heads}, head_dim={head_dim})")
        print(f"{'=' * 90}")
        print(f"{'Seq Len':<10} {'CUDA WMMA (ms)':<18} {'Native (ms)':<15} {'WMMA vs Native'}")
        print("-" * 80)

        for seq_len in sizes:
            # Layout: (batch, seqlen, num_heads, head_dim)
            q = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn(batch, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float16)

            # Native uses (batch, heads, seqlen, head_dim)
            q_nat = q.transpose(1, 2).contiguous()
            k_nat = k.transpose(1, 2).contiguous()
            v_nat = v.transpose(1, 2).contiguous()

            ms_cuda = do_bench(lambda: flash_attn_cuda.mha_fwd(q, k, v, False))
            ms_native = do_bench(lambda: F.scaled_dot_product_attention(q_nat, k_nat, v_nat, is_causal=False))

            ratio = ms_cuda / ms_native
            print(f"{seq_len:<10} {ms_cuda:<18.4f} {ms_native:<15.4f} {ratio:.2f}x slower")


if __name__ == "__main__":
    benchmark()

"""
Benchmark: CUTLASS-based FlashAttention vs PyTorch SDPA (and optionally WMMA).

Build first:
  cd cuda/flash_attn_cutlass && python setup.py build_ext --inplace

Run:
  LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
    python benchmarks/bench_cuda_flash_attention_cutlass.py
"""

import sys

import torch
import torch.nn.functional as F
from triton.testing import do_bench

sys.path.insert(0, ".")
sys.path.insert(0, "cuda/flash_attn_cutlass")

try:
    import flash_attn_cutlass
except ImportError:
    print("flash_attn_cutlass not built. Build with:")
    print("  cd cuda/flash_attn_cutlass && python setup.py build_ext --inplace")
    sys.exit(1)

# Optional: existing wmma version for 3-way comparison
HAS_WMMA = False
try:
    sys.path.insert(0, "cuda/flash_attn")
    import flash_attn_cuda  # noqa: F401
    HAS_WMMA = True
except ImportError:
    pass


def correctness_check(head_dim, seq_len=128):
    """Quick sanity: CUTLASS output matches PyTorch SDPA within fp16 tolerance."""
    B, H, S, D = 2, 4, seq_len, head_dim
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)

    out, _lse = flash_attn_cutlass.mha_fwd(q, k, v, False)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2)

    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    ok = max_err < 0.1  # generous fp16 tolerance
    status = "✓" if ok else "✗"
    print(f"  [{status}] head_dim={head_dim}, seq={seq_len}: "
          f"max_err={max_err:.4f}, mean_err={mean_err:.4f}")
    return ok


def benchmark():
    batch = 4
    n_heads = 8
    seq_lens = [128, 256, 512, 1024, 2048, 4096]

    print("Correctness check:")
    for hd in [64, 128]:
        correctness_check(hd)

    for head_dim in [64, 128]:
        print(f"\n{'=' * 100}")
        print(f"  head_dim={head_dim}, batch={batch}, heads={n_heads}")
        print(f"{'=' * 100}")

        # Header
        cols = ["Seq Len", "CUTLASS (ms)"]
        if HAS_WMMA:
            cols.append("WMMA (ms)")
        cols.extend(["SDPA (ms)", "CUTLASS/SDPA"])
        print("  ".join(f"{c:<16}" for c in cols))
        print("-" * 100)

        for seq_len in seq_lens:
            # Layout: (batch, seqlen, num_heads, head_dim)
            q = torch.randn(batch, seq_len, n_heads, head_dim,
                            device="cuda", dtype=torch.float16)
            k = torch.randn(batch, seq_len, n_heads, head_dim,
                            device="cuda", dtype=torch.float16)
            v = torch.randn(batch, seq_len, n_heads, head_dim,
                            device="cuda", dtype=torch.float16)

            # PyTorch SDPA wants (batch, heads, seqlen, head_dim)
            q_nat = q.transpose(1, 2).contiguous()
            k_nat = k.transpose(1, 2).contiguous()
            v_nat = v.transpose(1, 2).contiguous()

            ms_cutlass = do_bench(
                lambda: flash_attn_cutlass.mha_fwd(q, k, v, False)
            )
            ms_native = do_bench(
                lambda: F.scaled_dot_product_attention(
                    q_nat, k_nat, v_nat, is_causal=False
                )
            )

            row = [str(seq_len), f"{ms_cutlass:.4f}"]
            if HAS_WMMA:
                ms_wmma = do_bench(
                    lambda: flash_attn_cuda.mha_fwd(q, k, v, False)
                )
                row.append(f"{ms_wmma:.4f}")
            row.append(f"{ms_native:.4f}")
            row.append(f"{ms_cutlass / ms_native:.2f}x")
            print("  ".join(f"{c:<16}" for c in row))


if __name__ == "__main__":
    benchmark()

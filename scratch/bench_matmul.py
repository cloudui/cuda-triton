import argparse
import math
import time

import torch


def benchmark_matmul(M, K, N, dtype=torch.float16, warmup=10, iters=100, device="cuda"):
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    for _ in range(warmup):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = elapsed / iters * 1000
    tflops = 2 * M * N * K / (elapsed / iters) / 1e12
    print(
        f"[matmul]   M={M:6d} K={K:6d} N={N:6d} | dtype={str(dtype).split('.')[-1]:8s} | {ms:.3f} ms | {tflops:.2f} TFLOPS"
    )
    return ms, tflops


def benchmark_softmax(M, N, dtype=torch.float16, warmup=10, iters=100, device="cuda"):
    x = torch.randn(M, N, dtype=dtype, device=device)

    for _ in range(warmup):
        y = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        y = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = elapsed / iters * 1000
    gb = 2 * M * N * x.element_size() / 1e9
    gb_s = gb / (elapsed / iters)
    # per element: subtract max (1), exp (1), sum (1), divide (1), plus max reduction (1) = 5 ops
    tflops = 5 * M * N / (elapsed / iters) / 1e12
    print(
        f"[softmax]  M={M:6d} N={N:6d}          | dtype={str(dtype).split('.')[-1]:8s} | {ms:.3f} ms | {gb_s:.2f} GB/s | {tflops:.2f} TFLOPS"
    )
    return ms, gb_s


def benchmark_attention(
    seq_len, d_head, dtype=torch.float16, warmup=10, iters=100, device="cuda"
):
    Q = torch.randn(seq_len, d_head, dtype=dtype, device=device)
    K = torch.randn(seq_len, d_head, dtype=dtype, device=device)
    V = torch.randn(seq_len, d_head, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(d_head)

    def attn(Q, K, V, scale):
        S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (seq_len, seq_len)
        P = torch.softmax(S, dim=-1)
        return torch.matmul(P, V)  # (seq_len, d_head)

    for _ in range(warmup):
        out = attn(Q, K, V, scale)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = attn(Q, K, V, scale)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = elapsed / iters * 1000
    # Two matmuls dominate: QK^T is seq^2*d_head, PV is seq^2*d_head
    tflops = (4 * seq_len * seq_len * d_head) / (elapsed / iters) / 1e12
    print(
        f"[attn]     seq={seq_len:6d} d={d_head:4d}          | dtype={str(dtype).split('.')[-1]:8s} | {ms:.3f} ms | {tflops:.2f} TFLOPS"
    )
    return ms, tflops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=32768)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--N", type=int, default=32768)
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--d_head", type=int, default=128)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep over common square sizes (matmul only)",
    )
    parser.add_argument(
        "--op", type=str, default="all", choices=["all", "matmul", "softmax", "attn"]
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.sweep:
        sizes = [512, 1024, 2048, 4096, 8192]
        for s in sizes:
            benchmark_matmul(s, s, s, dtype=dtype, warmup=args.warmup, iters=args.iters)
    elif args.op == "all":
        benchmark_matmul(
            args.M, args.K, args.N, dtype=dtype, warmup=args.warmup, iters=args.iters
        )
        benchmark_softmax(
            args.M, args.K, dtype=dtype, warmup=args.warmup, iters=args.iters
        )
        benchmark_attention(
            args.seq_len, args.d_head, dtype=dtype, warmup=args.warmup, iters=args.iters
        )
    elif args.op == "matmul":
        benchmark_matmul(
            args.M, args.K, args.N, dtype=dtype, warmup=args.warmup, iters=args.iters
        )
    elif args.op == "softmax":
        benchmark_softmax(
            args.M, args.K, dtype=dtype, warmup=args.warmup, iters=args.iters
        )
    elif args.op == "attn":
        benchmark_attention(
            args.seq_len, args.d_head, dtype=dtype, warmup=args.warmup, iters=args.iters
        )

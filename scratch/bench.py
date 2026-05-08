import time

import torch


def benchmark_matmul(
    m,
    n,
    k,
    iterations=50,
    dtype=torch.float16,
    ta=False,
    tb=False,
):
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not found. This benchmark requires an NVIDIA GPU.")
        return

    device = "cuda"

    # 1. Setup Matrices
    # Size (N, N) @ (N, N)
    if ta:
        a = torch.randn(k, m, device=device, dtype=dtype)
        a = a.T
    else:
        a = torch.randn(m, k, device=device, dtype=dtype)

    if tb:
        b = torch.randn(n, k, device=device, dtype=dtype)
        b = b.T
    else:
        b = torch.randn(k, n, device=device, dtype=dtype)

    # 2. Warm-up
    # GPU kernels are lazy-loaded; the first few runs are always slow.
    print(f"Warming up with {m}x{n}x{k} matrices...")
    for _ in range(100):
        torch.matmul(a, b)

    # Synchronize to ensure warm-up is finished
    torch.cuda.synchronize()

    # 3. Benchmark Loop
    print(f"Running benchmark ({iterations} iterations)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        # Using A @ B (shorthand for torch.matmul)
        res = a @ b
    end_event.record()

    # Wait for GPU to finish all recorded events
    torch.cuda.synchronize()

    # 4. Results Calculation
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_s = (elapsed_time_ms / 1000) / iterations

    # Standard GEMM formula for FLOPS: 2 * M * N * K
    # For square matrices: 2 * size^3
    tflops = (2 * (m * n * k)) / (avg_time_s * 1e12)

    print("-" * 30)
    # print(f"Matrix Size: {m} x {n} x {k}")
    print(f"ta: {ta}, tb: {tb}")
    print(f"Avg Time:    {avg_time_s * 1000:.3f} ms")
    # print(f"Performance: {tflops:.2f} TFLOPS")
    print("-" * 30)


if __name__ == "__main__":
    # FP16 is standard for modern Tensor Core benchmarking
    for ta in [False, True]:
        for tb in [False, True]:
            benchmark_matmul(
                m=8192,
                n=8192,
                k=4096,
                iterations=100,
                dtype=torch.float16,
                ta=ta,
                tb=tb,
            )

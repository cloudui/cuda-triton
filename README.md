# Triton + CUDA GPU Kernels

Custom Triton and CUDA kernels for core transformer operations — from elementwise activations to FlashAttention-2 and quantized matmul — with PyTorch reference implementations and benchmarks. Built from scratch to understand GPU kernel programming, memory-aware algorithm design, and tensor core utilization.

## Kernels

### RMSNorm
Row-wise normalization used in Llama, Mistral, etc. Each Triton program handles one row — loads the full row, computes `sqrt(mean(x^2) + eps)`, normalizes, and scales by a learned weight. **3.3-4.9x faster than PyTorch.**

### SwiGLU
Gated activation: `SwiGLU(x, gate) = x * silu(gate)`. Elementwise operation — each Triton program processes a flat block of elements. **1.5-2.3x faster than PyTorch.**

### Fused RMSNorm + SwiGLU
Combines normalization and activation into a single kernel to avoid an extra round trip to HBM. In a standard transformer FFN, these run back-to-back, so fusing them halves global memory accesses. **3-6x faster than PyTorch, 1.3-6x faster than torch.compile.**

### Softmax
Numerically stable softmax with row-wise max subtraction for fp16 precision. **3.5-8x faster than PyTorch, 1.0-1.8x faster than native.**

### Naive Attention
Single-head scaled dot-product attention: `softmax(Q @ K^T / sqrt(d_k)) @ V`. All three operations fused into one kernel. Block size scales with sequence length, but performance degrades at longer sequences since the entire attention row must fit in one block. **1.0-2.5x faster than PyTorch at short sequences.** FlashAttention removes this limitation via tiling.

### FlashAttention-2 (Full)
Batched multi-head FlashAttention-2 with optional causal masking. Tiled attention with online softmax, `tl.dot` for tensor core acceleration, causal early exit optimization, and `exp2` hardware intrinsics. O(N) memory instead of O(N²). **Achieves 84-115% of PyTorch's production FlashAttention (Tri Dao's CUDA implementation) on A100.**

### Tiled fp16 Matmul
Triton tiled matmul using the same 2D grid + K-loop accumulation pattern as the quantized kernels, but with no dequantization. Serves as a baseline to isolate how much of the quantized kernel slowdown is due to dequant overhead vs cuBLAS being more optimized (software pipelining, L2 swizzling, warp specialization, etc.). **~0.74-1.03x of cuBLAS.**

### Quantized Matmul (int8 / int4)
Tiled matmul kernels that load quantized weights (int8 or int4) and dequantize on the fly to fp16 for tensor core computation. Int4 kernel handles bit-packed weights (two values per byte) with group-wise scale/zero_point. **2x / 3.8x weight memory savings.** Demonstrates the tiled matmul programming pattern (2D grid + K-loop accumulation) and quantization fundamentals.

### CUDA Softmax
Hand-written CUDA softmax kernels for comparison with Triton. Two versions: **v1** uses a three-pass approach (max, exp+sum, normalize) with float4 vectorized loads and shared memory tree reductions. **v2** caches all values in registers on the first pass (Triton-style single-pass caching), so exp/sum/normalize become register-only work with no redundant global memory reads. Uses templated `NUM_ITERS` for compile-time loop unrolling and a generic `block_reduce` helper. Currently supports rows up to N=8192 (256 threads × 8 values/thread × 4 max iterations). **v2 is 1.0-1.3x faster than v1, competitive with Triton.**

## Benchmarks

All benchmarks on NVIDIA A100 80GB with fp16.

### RMSNorm (batch=128)
```
Hidden Size     PyTorch (ms)    Native (ms)     Triton (ms)     Speedup
----------------------------------------------------------------------
1024            0.033           0.039           0.007           5.18x
2048            0.032           0.041           0.007           5.50x
4096            0.039           0.049           0.008           5.84x
8192            0.036           0.051           0.011           4.74x
```

### SwiGLU
```
n=     1,024 | PyTorch: 0.0152ms | Triton: 0.0104ms | Speedup: 1.46x
n=    65,536 | PyTorch: 0.0162ms | Triton: 0.0074ms | Speedup: 2.18x
n= 1,048,576 | PyTorch: 0.0200ms | Triton: 0.0128ms | Speedup: 1.56x
n= 8,388,608 | PyTorch: 0.1015ms | Triton: 0.0438ms | Speedup: 2.32x
```

### Fused RMSNorm+SwiGLU (batch=128)
```
Hidden     PyTorch (ms)    Compiled (ms)    Fused (ms)     Fused vs PyT   Fused vs Comp
----------------------------------------------------------------------------------------
128        0.0451          0.0097           0.0075         5.98           1.29
512        0.0396          0.0470           0.0075         5.28           6.26
1024       0.0417          0.0372           0.0072         5.83           5.21
2048       0.0425          0.0346           0.0164         2.58           2.10
4096       0.0498          0.0328           0.0114         4.35           2.86
8192       0.0519          0.0447           0.0146         3.55           3.06
```

### Softmax (batch=32)
```
Hidden     PyTorch (ms)    Native (ms)     Triton (ms)    vs PyTorch   vs Native
---------------------------------------------------------------------------
128        0.0275          0.0074          0.0064         4.27x       1.15x
512        0.0306          0.0092          0.0070         4.40x       1.32x
1024       0.0353          0.0121          0.0066         5.34x       1.83x
2048       0.0446          0.0083          0.0072         6.16x       1.15x
4096       0.0628          0.0088          0.0078         8.09x       1.14x
8192       0.0336          0.0100          0.0096         3.49x       1.03x
```

### Naive Attention (single-head, d_k=64)
```
Seq Len      PyTorch (ms)    Native (ms)     Triton (ms)    vs PyTorch   vs Native
--------------------------------------------------------------------------------
64           0.0217          0.0151          0.0087         2.49x       1.72x
128          0.0193          0.0118          0.0113         1.72x       1.05x
256          0.0202          0.0154          0.0159         1.27x       0.97x
512          0.0369          0.0162          0.0374         0.99x       0.43x
```

### FlashAttention-2 — Non-Causal (batch=4, heads=8, d_k=64)
```
Seq Len    Naive (ms)     Flash (ms)     Native (ms)    Flash vs Naive   Flash vs Native
--------------------------------------------------------------------------------
128        0.0788         0.0124         0.0130         6.34x            1.04x
256        0.0755         0.0155         0.0170         4.88x            1.10x
512        0.0866         0.0285         0.0311         3.04x            1.09x
1024       0.3437         0.0695         0.0740         4.94x            1.06x
2048       1.7805         0.2482         0.2172         7.17x            0.88x
4096       5.6174         0.9358         0.8457         6.00x            0.90x
```

### FlashAttention-2 — Causal (batch=4, heads=8, d_k=64)
```
Seq Len    Naive (ms)     Flash (ms)     Native (ms)    Flash vs Naive   Flash vs Native
--------------------------------------------------------------------------------
128        0.2040         0.0110         0.0126         18.56x           1.15x
256        0.1470         0.0168         0.0179         8.74x            1.07x
512        0.1489         0.0306         0.0337         4.87x            1.10x
1024       0.6240         0.0604         0.0608         10.33x           1.01x
2048       3.1640         0.1730         0.1555         18.29x           0.90x
4096       10.3571        0.5845         0.4911         17.72x           0.84x
```

### CUDA Softmax (batch=32)
```
Hidden     PyTorch (ms)    Triton (ms)     CUDA (ms)       CUDA-v2 (ms)    v2 vs Tri    v2 vs v1
----------------------------------------------------------------------------------------------------
128        0.0285          0.0066          0.0087          0.0069          0.95x        1.25x
512        0.0313          0.0066          0.0086          0.0075          0.88x        1.14x
1024       0.0359          0.0096          0.0081          0.0072          1.34x        1.13x
2048       0.0454          0.0149          0.0084          0.0081          1.85x        1.04x
4096       0.0635          0.0084          0.0100          0.0086          0.97x        1.16x
8192       0.0342          0.0100          0.0130          0.0100          1.00x        1.30x
```

Native uses PyTorch's built-in `scaled_dot_product_attention` (Tri Dao's production FlashAttention CUDA implementation). Our Triton implementation achieves 84-115% of production performance at small-to-medium sequence lengths, narrowing to ~84% at seq_len=4096.

### Quantized Matmul (M=128)
```
K×N            cuBLAS (ms)   TT fp16 (ms)   int8 TT (ms)   int4 TT (ms)
-------------------------------------------------------------------------------------
1024×1024      0.0129        0.0159         0.0224         0.0299
2048×2048      0.0194        0.0261         0.0309         0.0542
4096×4096      0.0480        0.0578         0.0694         0.1054
4096×11008     0.1116        0.1083         0.1503         0.1366
8192×8192      0.1230        0.1525         0.1934         0.2380

K×N            TT vs cuBLAS   int8 vs cuBLAS   int4 vs cuBLAS   int8 vs TT fp16   int4 vs TT fp16
----------------------------------------------------------------------------------------------------
1024×1024      0.81x          0.57x            0.43x            0.71x             0.53x
2048×2048      0.74x          0.63x            0.36x            0.84x             0.48x
4096×4096      0.83x          0.69x            0.46x            0.83x             0.55x
4096×11008     1.03x          0.74x            0.82x            0.72x             0.79x
8192×8192      0.81x          0.64x            0.52x            0.79x             0.64x

Weight Memory Savings:  int8 = 2.0x less,  int4 = 3.8x less
```

The quantized kernels are slower than cuBLAS fp16 for two reasons. First, a naive Triton tiled matmul is already ~0.74-1.03x of cuBLAS, which has deep optimizations (software pipelining, L2 swizzling, warp specialization) that this kernel doesn't use. Second, dequantization adds per-tile overhead (cast + subtract + multiply) inside the K-loop. Comparing int8/int4 against the Triton fp16 baseline (same kernel structure, no dequant) isolates the dequant cost at ~0.71-0.84x / ~0.48-0.79x.

Despite loading less data from HBM (int8 = half, int4 = quarter), the bandwidth savings don't compensate for the dequant compute at these sizes — the kernels aren't purely memory-bandwidth bound. Production systems avoid this tradeoff entirely by using integer tensor core instructions (int8×int8→int32) or FP8 tensor cores (H100+), which compute directly on quantized data without dequantizing. The value of this dequantize-on-the-fly approach is memory savings (fitting larger models on fewer GPUs), not latency.

## Project Structure

```
cuda/
  softmax.cu                  # CUDA softmax: vectorized float4, shared memory reductions
  softmax_triton.cu           # CUDA softmax v2: register caching, templated unrolling
  reduce.cuh                  # Generic block_reduce helper (max, sum)
  bindings.cu                 # PyTorch C++ extension bindings
  setup.py                    # Build script for CUDA extension
kernels/
  rmsnorm.py                  # RMSNorm: row-wise normalization
  swiglu.py                   # SwiGLU: gated FFN activation
  softmax.py                  # Softmax: numerically stable row normalization
  attention.py                # Naive attention: fused scaled dot-product
  flash_attention.py          # FlashAttention-2: single-head, tiled with online softmax
  flash_attention_full.py     # FlashAttention-2: batched, multi-head, causal
  fused_rmsnorm_swiglu.py     # Fused RMSNorm + SwiGLU into one kernel
  quantized_matmul.py         # fp16/int8/int4 tiled matmul with dequantize-on-the-fly
benchmarks/
  bench_rmsnorm.py            # RMSNorm performance
  bench_swiglu.py             # SwiGLU performance
  bench_softmax.py            # Softmax performance
  bench_attention.py          # Naive attention performance (small seqs)
  bench_flash_attention.py    # FlashAttention single-head vs naive vs native
  bench_flash_attention_full.py  # FlashAttention batched/multi-head/causal
  bench_fused.py              # Fused kernel vs PyTorch vs torch.compile
  bench_quantized_matmul.py   # Quantized matmul latency + memory savings
tests/
  test_kernels.py             # Correctness tests for all kernels
```

## Usage

```bash
# Install
pip install torch triton

# Run tests
python -m pytest tests/test_kernels.py -v

# Run benchmarks
python benchmarks/bench_rmsnorm.py
python benchmarks/bench_swiglu.py
python benchmarks/bench_softmax.py
python benchmarks/bench_attention.py
python benchmarks/bench_flash_attention.py
python benchmarks/bench_flash_attention_full.py
python benchmarks/bench_fused.py
python benchmarks/bench_quantized_matmul.py

# Build and benchmark CUDA kernels
make build-cuda
python benchmarks/bench_cuda_softmax.py
```

## Learning Path

This repo represents a progression through Triton kernel development:

1. **SwiGLU** — Elementwise ops. Learn the Triton programming model: programs, blocks, masks.
2. **RMSNorm** — Row-wise reduction. Learn per-row reductions with `tl.sum`.
3. **Fused RMSNorm+SwiGLU** — Kernel fusion. Learn why combining ops saves HBM bandwidth.
4. **Softmax** — Numerically stable reduction. Learn the max-subtract trick for fp16.
5. **Naive Attention** — 2D block loads. Learn broadcasting and multi-dimensional indexing.
6. **FlashAttention-2 (single-head)** — Tiled attention with online softmax. Learn looping within kernels and incremental algorithms.
7. **FlashAttention-2 (full)** — Batched multi-head with causal masking. Learn 2D grids, stride-based pointer arithmetic, tensor core utilization via `tl.dot`, and causal early exit.
8. **Tiled fp16 Matmul** — Bare tiled matmul as a cuBLAS comparison baseline. Learn the 2D grid + K-loop accumulation pattern.
9. **Quantized Matmul** — int8/int4 dequantize-on-the-fly. Learn bit packing/unpacking, group quantization, and the tradeoffs between memory savings and compute overhead.

## Testing Quantized Matmul

The quantization tests are split into four classes, separating concerns so failures are isolated:

- **`TestInt8Quantization` / `TestInt4Quantization`** — test the quantize/dequantize utilities with no Triton dependency. If the kernel crashes, these still run.
- **`TestFp16Matmul`** — test Triton tiled fp16 matmul against cuBLAS (`x @ W`).
- **`TestInt8Matmul` / `TestInt4Matmul`** — test matmul correctness (PyTorch reference vs fp16 baseline, Triton vs PyTorch reference).

All tests are parametrized over multiple matrix sizes to catch dimension-dependent bugs.

### Error bounds

Tolerances are derived from quantization theory, not guesswork:

**Roundtrip error** (quantize → dequantize): each value is rounded to the nearest integer bin, so max per-element error = `scale / 2`. For `randn` weights with range ≈ 6:
- int8: `scale = 6/255 ≈ 0.024`, mean error ≈ `scale/4 ≈ 0.006`, max ≈ `scale/2 ≈ 0.012`
- int4: `scale = 6/15 ≈ 0.4`, mean error ≈ `0.1`, max ≈ `0.2`

**Matmul error** (quantized vs fp16): quantization error accumulates over the K-dimension dot product. Each output element sums K independent error terms, so the standard deviation grows as `scale * sqrt(K/12)`:
- int8, K=256: `std ≈ 0.024 * sqrt(256/12) ≈ 0.11`
- int4, K=256: `std ≈ 0.4 * sqrt(256/12) ≈ 1.85`

**Triton vs PyTorch** (same math, different execution): tight tolerance (`atol=0.1`) since the only difference is floating point accumulation order on GPU vs CPU.

### What else the tests check
- Quantized weight dtypes and value ranges (int8 in [-128, 127], int4 nibbles in [0, 15])
- Scale is positive for every column
- Memory savings are exact (int8 = 2x, int4 packed = 4x vs fp16 weight bytes)
- Output shapes match expected (M, N)

Run just the quantization utility tests (no Triton needed):
```bash
pytest tests/test_kernels.py -k "Quantization" -v
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA)
- Triton 2.0+
- NVIDIA GPU (benchmarked on A100 80GB)

# Triton + CUDA GPU Kernels

Triton and CUDA kernels for transformer operations — RMSNorm, SwiGLU, softmax, FlashAttention-2, fp16 / int8 / int4 matmul — with PyTorch reference implementations, parametrized correctness tests, and A100 benchmarks.

> **Blog post:** [**FlashAttention, but the Actual Details**](https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/) — line-by-line walkthrough of the CuTe FA2 implementation in [`cuda/flash_attn_cutlass/`](cuda/flash_attn_cutlass/), covering swizzling, tiled MMAs, online softmax, the V-copy + transpose, and at least one finding about an unused line in Tri Dao's source. The companion `scratch/` directory has standalone runnable demos for each concept ([scratch README](scratch/README.md)).

## Kernels

### Triton

| Kernel | Notes | vs PyTorch |
|---|---|---|
| RMSNorm | One row per program, `tl.sum` reduction | 3.3-4.9× |
| SwiGLU | Elementwise, flat blocks | 1.5-2.3× |
| Fused RMSNorm + SwiGLU | Single kernel; halves HBM accesses vs the two-kernel version | 3-6× (1.3-6× vs `torch.compile`) |
| Softmax | Numerically stable max-subtract | 3.5-8× (1.0-1.8× vs SDPA) |
| Naive Attention | Single-head fused `softmax(QK^T/√d)V` for short sequences | 1.0-2.5× |
| FlashAttention-2 | Batched, multi-head, optional causal masking; tiled attention with online softmax, `tl.dot`, causal early exit | **84-115% of native FA-2** (Tri Dao's CUDA, via PyTorch SDPA) |
| Tiled fp16 Matmul | 2D grid + K-loop accumulation | 0.74-1.03× of cuBLAS |
| Quantized Matmul (int8 / int4) | On-the-fly dequant; int4 with bit-packed weights and group-wise scale | 2× / 3.8× weight memory savings |

### CUDA

| Kernel | Notes |
|---|---|
| Softmax v1 | Three-pass (max → exp+sum → normalize), `float4` loads, shared-memory tree reductions |
| Softmax v2 | Single-pass register caching (Triton-style), templated unrolling. 1.0-1.3× faster than v1 |
| WMMA FlashAttention-2 | Hand-written using `nvcuda::wmma`. 4 profile-driven optimization iterations from 1.4% → 10.4% of A100 fp16 peak. See [`cuda/flash_attn/`](cuda/flash_attn/). |
| CuTe FlashAttention-2 | Production-style rewrite using CUTLASS 3.x's CuTe layout algebra. **Within ~1-15% of native FA-2** across hdim=64/128 on A100. See [`cuda/flash_attn_cutlass/`](cuda/flash_attn_cutlass/). |

## Benchmarks

A100 80GB, fp16.

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

`Native` is PyTorch's `scaled_dot_product_attention`, which dispatches to Tri Dao's CUDA FlashAttention.

### CuTe FlashAttention-2 (CUDA, batch=4, heads=8)

A100 40GB, fp16. CUTLASS 3.x CuTe-based rewrite vs. production FA-2 (via SDPA). `% peak` is fraction of A100 fp16 tensor-core peak (312 TFLOP/s).

```
hdim=64
Seq Len    CuTe (ms)    Native (ms)   Ratio   TFLOP/s   % peak
----------------------------------------------------------------
128          0.0149       0.0156      0.96x      9.0     2.9%
256          0.0186       0.0177      1.05x     28.9     9.3%
512          0.0334       0.0332      1.01x     64.3    20.6%
1024         0.0872       0.0764      1.14x     98.5    31.6%
2048         0.2379       0.2186      1.09x    144.4    46.3%
4096         0.8786       0.8501      1.03x    156.4    50.1%
8192         3.3720       3.2199      1.05x    163.0    52.3%
16384       13.4770      12.8500      1.05x    163.2    52.3%
32768       53.4953      51.1262      1.05x    164.4    52.7%
65536      213.6306     204.8852      1.04x    164.7    52.8%

hdim=128
Seq Len    CuTe (ms)    Native (ms)   Ratio   TFLOP/s   % peak
----------------------------------------------------------------
128          0.0159       0.0167      0.95x     16.9     5.4%
256          0.0227       0.0226      1.01x     47.3    15.2%
512          0.0485       0.0499      0.97x     88.5    28.4%
1024         0.1309       0.1245      1.05x    131.3    42.1%
2048         0.4339       0.3808      1.14x    158.4    50.8%
4096         1.4727       1.4979      0.98x    186.6    59.8%
8192         5.8427       5.6883      1.03x    188.2    60.3%
16384       22.5305      22.7326      0.99x    195.2    62.6%
32768       89.4886      90.9835      0.98x    196.6    63.0%
65536      361.0303     365.8720      0.99x    194.9    62.5%
```

Within ~1-14% of native FA-2 across all configurations, **at parity (and occasionally faster) for long context** out to 64K-token sequences. Sustains **~53% of A100 fp16 peak at hdim=64** and **~63% at hdim=128** — comparable to production FA-2's reported utilization. See [`cuda/flash_attn_cutlass/CUTE_NOTES.md`](cuda/flash_attn_cutlass/CUTE_NOTES.md) for the experimental finding that several "defensive" CuTe idioms inherited from CUTLASS examples (`SmemLayoutVtNoSwizzle`, the variable `kSwizzle` formula) are no-ops on SM80.

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

The quantized kernels are slower than cuBLAS fp16 for two reasons. First, the Triton fp16 baseline is already 0.74-1.03× of cuBLAS, which has software pipelining, L2 swizzling, and warp specialization that this kernel doesn't. Second, dequantization adds per-tile overhead inside the K-loop. Comparing int8/int4 to the Triton fp16 baseline isolates the dequant cost at ~0.71-0.84× / ~0.48-0.79×.

The bandwidth savings from loading less data (int8 = half, int4 = quarter) don't compensate for the dequant compute at these sizes — the kernels aren't purely memory-bandwidth bound. Production avoids the tradeoff entirely with integer tensor core instructions (int8×int8→int32) or FP8 tensor cores (H100+), which compute on quantized data without dequantizing. The value of dequantize-on-the-fly is memory savings (fitting larger models on fewer GPUs), not latency.

## Project structure

```
cuda/
  softmax.cu                  CUDA softmax: vectorized float4, smem tree reductions
  softmax_triton.cu           CUDA softmax v2: register caching, templated unrolling
  reduce.cuh                  Generic block_reduce helper
  bindings.cu                 PyTorch C++ extension bindings
  setup.py                    Build script
  flash_attn/                 WMMA FlashAttention-2 (see directory README)
  flash_attn_cutlass/         CuTe FlashAttention-2 (see directory README)
kernels/                      Triton kernels (one .py per kernel)
benchmarks/                   One bench_*.py per kernel
tests/test_kernels.py         Parametrized correctness tests
```

## Setup

```bash
# CUDA 13.0 path (current setup)
conda create -n torch_cuda13 python=3.12 -y
conda activate torch_cuda13
conda install -c nvidia/label/cuda-13.0.0 cuda-toolkit cuda-nvcc -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt   # triton, ninja, pytest, numpy

# CUDA 12.x: drop the --index-url and use the default PyTorch wheels.
```

## Usage

```bash
# Tests
python -m pytest tests/test_kernels.py -v

# Triton benchmarks
python benchmarks/bench_rmsnorm.py
python benchmarks/bench_swiglu.py
python benchmarks/bench_softmax.py
python benchmarks/bench_attention.py
python benchmarks/bench_flash_attention.py
python benchmarks/bench_flash_attention_full.py
python benchmarks/bench_fused.py
python benchmarks/bench_quantized_matmul.py

# CUDA softmax
make build-cuda
make bench-cuda-softmax

# WMMA FlashAttention
make build-fac
make test-fac
make bench-fac

# CuTe FlashAttention (CUTLASS 3.x). Requires CUTLASS headers — fetch once.
make fetch-cutlass         # clones v3.6.0 into third_party/cutlass
make build-fac-cutlass
make test-fac-cutlass
make bench-fac-cutlass
```

## Quantized matmul correctness

Test tolerances are derived from quantization theory rather than empirical fudge factors.

**Roundtrip error** (quantize → dequantize): each value is rounded to the nearest integer bin, so max per-element error = `scale / 2`. For `randn` weights with range ≈ 6:
- int8: `scale = 6/255 ≈ 0.024`, mean error ≈ 0.006, max ≈ 0.012
- int4: `scale = 6/15 ≈ 0.4`, mean error ≈ 0.1, max ≈ 0.2

**Matmul error** (quantized vs fp16): quantization error accumulates over the K dot product. Each output sums K independent error terms, so the standard deviation grows as `scale * sqrt(K/12)`:
- int8, K=256: `std ≈ 0.024 × √(256/12) ≈ 0.11`
- int4, K=256: `std ≈ 0.4 × √(256/12) ≈ 1.85`

**Triton vs PyTorch** (same math, different execution): `atol=0.1` since the only difference is fp accumulation order on GPU vs CPU.

The tests also check quantized weight dtypes and value ranges, scale positivity, exact memory savings, and output shapes. Run only the quantization utility tests (no Triton needed):

```bash
pytest tests/test_kernels.py -k "Quantization" -v
```

## Requirements

- Python 3.10+ (3.12 recommended)
- PyTorch 2.0+ with CUDA (currently 2.11+cu130)
- Triton 2.0+ (currently 3.6)
- Ninja (incremental builds with header dependency tracking)
- pytest
- NVIDIA GPU (benchmarked on A100 80GB)
- CUDA Toolkit 12.x or 13.x (currently 13.0)

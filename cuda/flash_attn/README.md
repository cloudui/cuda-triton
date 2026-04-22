# WMMA FlashAttention-2

Hand-written CUDA forward kernel for FlashAttention-2 using `nvcuda::wmma` (no CUTLASS or CuTe dependencies). Implements the FA-2 algorithm with online softmax, fp16 inputs, fp32 accumulators, and per-thread row decomposition.

Reference baseline is PyTorch's `scaled_dot_product_attention` (Tri Dao's CUDA FlashAttention).

**Hardware:** NVIDIA A100 80GB. **Configs:** `batch=4, heads=8, fp16`.

---

## v1 — Baseline (single-warp WMMA, synchronous loads)

**Implementation:**
- One warp performs both WMMAs; remaining 3 warps idle through the matmul blocks
- Cooperative loads use all 128 threads with synchronous `__syncthreads()` (no `cp.async`)
- P round-trips through shared memory as fp16 in a separate `smem_p` buffer
- O accumulator in registers (`o_acc[HEAD_DIM]` per thread, one thread per row)
- Online softmax updates `m_state` and `l_state` in registers and rescales `o_acc` per iteration
- Scores buffer aliased with O temp buffer (saves 32 KB on hdim64)

**Shared memory:** 72 KB (hdim64) / 128 KB (hdim128), allocated via `cudaFuncSetAttribute`.

**Correctness:** all parametrized tests pass; max error ~5e-4 (fp16 rounding).

| Seq  | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native |
|------|-------------|--------------|------------------|
| 128  | 0.16        | 0.21         | 11.4× slower     |
| 256  | 0.31        | 0.38         | 17.4× slower     |
| 512  | 0.68        | 1.42         | 21.6× slower     |
| 1024 | 2.25        | 4.12         | 30.1× slower     |
| 2048 | 7.80        | 13.53        | 36.0× slower     |

**Identified bottlenecks** (priority order):
1. Three of four warps idle during both WMMAs — caps utilization at 25%
2. Synchronous `__syncthreads()` loads — no overlap of memory and compute
3. No double-buffering or `cp.async` — every K/V tile load fully blocks the SM
4. P round-trips through shared memory rather than staying in fragment registers
5. No shared memory swizzling — bank conflicts on every WMMA load
6. No causal early-exit (relevant once causal masking is implemented)

---

## v2 — Multi-warp WMMA distribution

**Change:** Distribute WMMA tile work across all 4 warps. Each warp owns a 32-row band of the BLOCK_M output (warp 0 → rows 0-31, warp 1 → rows 32-63, etc.). Both Q@K^T and P@V loops parallelize across warps; the single-warp gate is removed.

**Correctness:** all tests pass.

| Seq  | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native | Speedup vs v1 |
|------|-------------|--------------|------------------|---------------|
| 128  | 0.12        | 0.15         | 8.5× slower      | 1.34×         |
| 256  | 0.19        | 0.26         | 11.3× slower     | 1.58×         |
| 512  | 0.65        | 0.95         | 20.3× slower     | 1.05×         |
| 1024 | 1.91        | 2.72         | 25.7× slower     | 1.18×         |
| 2048 | 6.23        | 8.84         | 29.1× slower     | 1.25×         |

**Analysis:**
- Speedup ranges from 1.05× to 1.58× — well below the theoretical 4× ceiling from warp utilization alone.
- The bottleneck has shifted to memory: with 4× more compute throughput, the SM now idles waiting on synchronous K/V loads instead of waiting on a single warp.
- Smaller sequences see larger improvements (warp-idle cost was a larger fraction of total time); long sequences are dominated by memory transfer.
- hdim128 improvements are more uniform (~1.4-1.5×) due to higher arithmetic intensity per memory load.

**Bottlenecks after v2:**
1. Synchronous K/V loads — `load_tile_sync` blocks all 128 threads at `__syncthreads()`. Addressed by v3.
2. P round-trips through shared memory
3. No double buffering
4. No shared memory swizzling

---

## Roadmap

The remaining optimizations are mostly mechanical implementations of techniques the Triton compiler emits automatically. Expected speedups are based on roofline analysis given the v2 baseline.

| Version | Optimization | Expected impact |
|---------|--------------|-----------------|
| v3 | Async copies (`cp.async` + `cp.async.commit_group` + `cp.async.wait_group`) | Largest remaining win; overlaps HBM loads with WMMA compute |
| v4 | Double-buffered shared memory for K/V tiles | Compounds with v3 to fully hide HBM latency at long sequences |
| v5 | Shared memory swizzling (XOR layouts) | Eliminates bank conflicts visible in `ncu` memory analysis |
| v6 | Causal masking with diagonal early-exit | ~2× on causal workloads at long sequences |
| v7+ | Q-fragment hoisting (4 accumulators per warp), warp-shuffle softmax, fragment-layout-aware O rescale, BLOCK_M / BLOCK_N tuning | Diminishing returns; required for parity with production |

Hopper-specific optimizations (TMA, WGMMA, warp specialization) — i.e. the techniques that move FA-2 to FA-3 — are out of scope for this A100-targeted implementation.

---

## Build, test, benchmark

From the repository root:

```bash
make build-fac     # compile the extension
make test-fac      # parametrized correctness tests vs scaled_dot_product_attention
make bench-fac     # benchmark vs PyTorch SDPA
```

## Profiling

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
    ncu --set basic --target-processes all \
    --kernel-name flash_fwd_kernel \
    --launch-skip 5 --launch-count 1 \
    python benchmarks/bench_cuda_flash_attention.py
```

Note: shared dev hosts often restrict GPU performance counters. If `ncu` reports "driver resource unavailable", `nsys profile --stats=true` works as a fallback for kernel timing without requiring the same exclusive counter access.

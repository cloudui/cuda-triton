# WMMA FlashAttention-2 — Iteration Log

A from-scratch CUDA FlashAttention-2 forward kernel using `nvcuda::wmma` (no CUTLASS / CuTe). Built primarily as a learning exercise to understand tensor cores, online softmax, and the production-FA optimization stack.

Reference baseline is PyTorch's `scaled_dot_product_attention` (Tri Dao's CUDA FlashAttention).

Hardware: NVIDIA A100 80GB. Configs: `batch=4, heads=8, fp16`.

---

## v1 — Baseline (single-warp WMMA, sync loads)

**Implementation choices:**
- 1 warp does both WMMAs; other 3 warps idle through the WMMA blocks
- Cooperative loads use all 128 threads, but plain `__syncthreads()` (no `cp.async`)
- P round-trips through smem as fp16 (separate `smem_p` buffer)
- O accumulator lives in registers (`o_acc[HEAD_DIM]` per thread, 1 thread per row)
- Online softmax updates `m_state`, `l_state` in registers, rescales `o_acc` per iter
- Scores buffer is reused for O temp (saves 32 KB on hdim64)
- No causal masking yet
- No async copies, no double buffer, no smem swizzling

**Smem footprint:** 72 KB (hdim64) / 128 KB (hdim128) — extended via `cudaFuncSetAttribute`.

**Numerical correctness:** ✅ All tests pass (max diff ~5e-4, fp16 rounding noise).

| Seq | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native |
|---|---|---|---|
| 128  | 0.16  | 0.21  | 11.4× slower |
| 256  | 0.31  | 0.38  | 17.4× slower |
| 512  | 0.68  | 1.42  | 21.6× slower |
| 1024 | 2.25  | 4.12  | 30.1× slower |
| 2048 | 7.80  | 13.53 | 36.0× slower |

**Bottleneck inventory** (in rough priority order for future passes):
1. 3 of 4 warps idle during both WMMAs → hard ceiling of ~4× from warp utilization
2. Synchronous `__syncthreads()` loads — no overlap of memory and compute
3. No double-buffering / `cp.async` — every K/V tile load fully blocks
4. P round-trips through smem instead of staying in fragment registers
5. No smem swizzling — bank conflicts on every WMMA load
6. Causal early exit not implemented (will help once causal added)

---

## v2 — TBD (multi-warp WMMA distribution)

**Plan:** Distribute the WMMA tile loops across all 4 warps so each warp owns a different output sub-tile region. Expected ~3-4× speedup from warp utilization alone.

| Seq | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native |
|---|---|---|---|
| 128  | TBD | TBD | TBD |
| 256  | TBD | TBD | TBD |
| 512  | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD |
| 2048 | TBD | TBD | TBD |

---

## v3 — TBD (async copies via `cp.async`)

**Plan:** Replace `load_tile_sync` with `cp_async_16B`-based loads. Issue commits + waits to overlap K/V tile loads with WMMA compute.

| Seq | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native |
|---|---|---|---|
| 128  | TBD | TBD | TBD |
| ...  |     |     |     |

---

## v4 — TBD (double-buffered smem)

**Plan:** Allocate two K and two V tiles in smem. Prefetch next tile while computing current. Should fully hide K/V load latency at long sequence lengths.

---

## v5 — TBD (smem swizzling for bank conflict avoidance)

**Plan:** Pad smem rows or use XOR-swizzled layouts so consecutive 128-bit WMMA loads don't hit the same bank. Should eliminate 4-way / 8-way bank conflicts visible in `ncu`'s memory analysis.

---

## v6 — TBD (causal early exit + masking)

**Plan:** Skip KV blocks fully above the diagonal. For partial blocks, mask scores to `-INFINITY` before softmax. Should give ~2× on causal workloads at long sequences.

---

## v7+ — Possible advanced optimizations

- Keep P in fragment registers across iterations (avoid smem round-trip entirely)
- Warp-shuffle softmax (avoid smem for max/sum reductions)
- Fragment-layout-aware `o_acc` rescale (avoid smem round-trip for O)
- Tune BLOCK_M / BLOCK_N for occupancy and L2 reuse
- Hopper-only: TMA + WGMMA + warp specialization (FA-3 territory; would need cloud H100)

---

## How to reproduce

From the repo root:

```bash
make build-fa     # compile the extension
make test-fa      # 11 parametrized correctness tests
make bench-fa     # run benchmark vs PyTorch SDPA
```

Profiling with Nsight Compute:

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
    ncu --set full --target-processes all \
    python benchmarks/bench_cuda_flash_attention.py
```

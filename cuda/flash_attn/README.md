# WMMA FlashAttention-2

Hand-written CUDA forward kernel for FlashAttention-2 using `nvcuda::wmma` (no CUTLASS or CuTe dependencies). Implements the FA-2 algorithm with online softmax, fp16 inputs, fp32 accumulators, and per-thread row decomposition.

Reference baseline is PyTorch's `scaled_dot_product_attention` (Tri Dao's CUDA FlashAttention).

**Hardware:** NVIDIA A100 80GB. **Configs:** `batch=4, heads=8, fp16`.

## Performance progression at-a-glance

Latency at `batch=4, heads=8, head_dim=64, seq=2048` and ratio vs PyTorch's `scaled_dot_product_attention` (Tri Dao's CUDA FlashAttention).

| Version | ms | vs Native | % of A100 fp16 peak |
|---|---|---|---|
| v1 (single-warp WMMA, sync loads) | 7.80 | 36.0× slower | 1.4% |
| v2 (multi-warp WMMA distribution) | 6.23 | 29.1× slower | 1.8% |
| **v3 (smem padding for bank conflicts)** | **1.14** | **5.3× slower** | **10.0%** |

Profile-level metric trends at the same workload:

| Version | L1/TEX % | Compute % | DRAM % | Bank conflicts (loads) | Top stall reason |
|---|---|---|---|---|---|
| v2 | 98.5% | 3.5% | 0.19% | 32-way, 92% wasted | `stall_short_scoreboard` 84.9% |
| v3 | 72.7% | 16.1% | 1.04% | 4-way, 21% wasted | `stall_short_scoreboard` 40.8% |

Native FA on A100 sits at ~50% of fp16 peak (~156 TFLOPs). Realistic ceiling for a WMMA-based kernel is ~25-30% of peak; further gains beyond that require dropping to MMA PTX (CUTLASS-style).

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

**Profiling (Nsight Compute, `--set full`, seq=2048, hdim64):**

| Metric | Value | Interpretation |
|---|---|---|
| L1/TEX Cache Throughput | 98.5% | Smem subsystem fully saturated |
| Compute (SM) Throughput | 3.5% | Tensor cores idle, waiting on smem |
| DRAM Throughput | 0.2% | HBM essentially unused |
| Shared load bank conflicts | **32-way average, 92% of wavefronts wasted** | Maximum-severity conflicts on every WMMA load |
| Shared store bank conflicts | 23-way average, 94% wasted | Same root cause |
| Top stall reason | `stall_short_scoreboard` (84.9%) | Confirms smem ops are the wait |
| Achieved warps per SM | 7.36 / 8 (smem-limited) | Occupancy is essentially full given smem footprint |

The 32-way conflict has a clean explanation: every smem buffer uses stride 64 halves = 128 bytes = exactly one full row of 32 banks, so all 16 rows of a WMMA fragment hit the same bank columns. Eliminating this is the single largest available optimization. The kernel is **bank-conflict-bound, not memory-bound**, which significantly reorders the planned optimizations.

**Bottlenecks after v2 (empirically validated, in priority order):**
1. **Bank conflicts on every smem access** — addressed by v3 (swizzling)
2. **Smem footprint limits occupancy to 2 blocks/SM (1 at hdim128)** — addressed by v4
3. Synchronous K/V loads — addressed later by v5 (cp.async)
4. No double buffering — v6
5. Tail wave (33% at hdim64, 20% at hdim128) — partially addressed by v4

---

## v3 — Shared memory padding to eliminate bank conflicts

**Change:** Pad the row stride of every shared memory buffer so that consecutive rows start in different banks. fp16 buffers (smem_q, smem_kv, smem_p) padded by 8 halves; fp32 buffers (smem_scores, smem_o) padded by 4 floats. Both maintain the 16-byte alignment required by `ldmatrix` / `store_matrix_sync`.

Implementation: ~25 lines across `utils.h` (added `smem_stride` template parameter to `load_tile_sync`), `flash_fwd_kernel.h` (introduced `Q_STRIDE / KV_STRIDE / SCORE_STRIDE / P_STRIDE / O_STRIDE` constants and used them at every smem access site), and `kernel_traits.h` (updated `kSmemSize`).

**Smem footprint** grew from 72 KB → ~79 KB (hdim64) and 128 KB → ~136 KB (hdim128). Still within the A100's 164 KB max.

**Correctness:** all tests pass.

| Seq | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native | Speedup vs v2 (hdim64) |
|---|---|---|---|---|
| 128  | 0.04 | 0.07 | 3.2× slower | 3.0× |
| 256  | 0.06 | 0.10 | 3.7× slower | 3.2× |
| 512  | 0.13 | 0.32 | 4.1× slower | 5.0× |
| 1024 | 0.38 | 0.82 | 5.2× slower | 5.0× |
| 2048 | 1.14 | 2.51 | 5.3× slower | 5.5× |

**Analysis:**
- 3-5× speedup at hdim64, 2.1-3.5× at hdim128 — the largest single optimization in the roadmap so far.
- The kernel now lands at ~19-31% of native FlashAttention performance (up from ~3-9% in v2).
- Larger gains at long sequences: bank conflicts compounded multiplicatively over the KV loop count, so removing them benefits long-seq more than short-seq.
- hdim128 sees smaller gains than hdim64 because hdim128's per-iteration compute work was already partially hiding the smem stalls; the absolute gain is similar but the ratio is smaller.

**Profiling (Nsight Compute, `--set full`, seq=2048, hdim64):**

| Metric | v2 | v3 | Change |
|---|---|---|---|
| Duration | 7.71 ms | 1.40 ms | **5.5× faster** |
| L1/TEX Throughput | 98.5% | **72.7%** | no longer saturated |
| Compute (SM) Throughput | 3.5% | **16.1%** | 4.6× more compute issued |
| DRAM Throughput | 0.19% | 1.04% | still tiny but rising |
| Shared load bank conflicts | **32-way, 92% wasted** | **4-way, 21% wasted** | matches predicted padding outcome |
| Shared store bank conflicts | 23-way, 94% wasted | **4-way, 53% wasted** | same |
| `stall_short_scoreboard` | 84.9% | 40.8% | halved |
| Eligible warps/scheduler | 0.03 / 16 | 0.17 / 16 | 5× more issuance |

The 4-way residual conflicts are exactly the unavoidable consequence of padding (vs full XOR swizzling). The smem bottleneck is gone but not zero.

**New highest-leverage bottleneck identified:** uncoalesced global stores in the final O writeback (60% potential per `ncu`). Each thread writes its row to gmem, but consecutive threads' rows are `o_row_stride` bytes apart → 30 of 32 bytes per cache line wasted on every write. This is a cheap fix with warp-cooperative writes and overtakes cp.async / double-buffering as the immediate next priority.

**Bottlenecks after v3 (priority order, empirically updated):**
1. **Uncoalesced global O writeback** (~60% potential) — easy fix, addressed by v4
2. cp.async (modest gain since DRAM still at 1%)
3. Residual 4-way smem bank conflicts (XOR swizzling could take to 0-way) — ~30% potential
4. Smem footprint limits occupancy to 2 blocks/SM
5. Tail wave (33% potential)

---

## Roadmap

Reordered after the v3 profile flagged uncoalesced global writes as the new highest-leverage bottleneck.

| Version | Optimization | Expected impact |
|---------|--------------|-----------------|
| **v4** | **Coalesce final O writeback to gmem** via warp-cooperative writes (consecutive threads write consecutive bytes within a row, then move to the next row) | ~60% per `ncu`. Cheap, mechanical change |
| v5 | Async copies (`cp.async` + commit/wait groups) | Modest gains since DRAM throughput is still ~1%, but next ceiling once coalescing is fixed |
| v6 | Double-buffered shared memory for K/V tiles | Compounds with v5 |
| v7 | XOR swizzling (replace padding to get 0-way conflicts and recover ~7 KB smem) | ~30% per `ncu` on residual smem patterns |
| v8 | Reduce shared memory footprint by holding P in fragment registers across iterations (eliminates `smem_p`) | Drops smem usage enough to fit 3 blocks/SM (hdim64); addresses tail-wave penalty and register pressure at hdim128 |
| v9 | Causal masking with diagonal early-exit | Orthogonal feature; ~2× on causal workloads at long sequences |
| v10+ | Q-fragment hoisting (4 accumulators per warp), warp-shuffle softmax, fragment-layout-aware O rescale, BLOCK_M / BLOCK_N tuning | Diminishing returns; required for parity with production |

**Quick wins (implementable anytime):**
- Coalesce final O writeback to gmem — currently has 31% wasted sectors per `ncu`. ~3% speedup, ~5-line change.
- Add `-lineinfo` to `nvcc` flags so `ncu` source attribution works.

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

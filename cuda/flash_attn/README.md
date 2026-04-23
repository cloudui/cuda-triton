# WMMA FlashAttention-2

CUDA forward kernel for FlashAttention-2 using `nvcuda::wmma`. fp16 inputs, fp32 accumulators, online softmax, per-thread row decomposition.

Reference: PyTorch's `scaled_dot_product_attention` (Tri Dao's CUDA FlashAttention).

Hardware: NVIDIA A100 80GB. Configs: `batch=4, heads=8, fp16`.

## Performance progression

Latency at `batch=4, heads=8, head_dim=64, seq=2048`.

| Version | ms | vs Native | % of fp16 peak |
|---|---|---|---|
| v1 — single-warp WMMA, sync loads | 7.80 | 36.0× | 1.4% |
| v2 — multi-warp WMMA distribution | 6.23 | 29.1× | 1.8% |
| v3 — smem padding to break bank-conflict alignment | 1.14 | 5.3× | 10.0% |
| v4 — coalesced O writeback | 1.09 | 5.0× | 10.4% |

Profile metric trends:

| Version | L1/TEX % | Compute % | DRAM % | Bank conflicts (loads) | Top stall reason |
|---|---|---|---|---|---|
| v2 | 98.5% | 3.5% | 0.19% | 32-way, 92% wasted | `stall_short_scoreboard` 84.9% |
| v3 | 72.7% | 16.1% | 1.04% | 4-way, 21% wasted | `stall_short_scoreboard` 40.8% |
| v4 | 77.5% | 17.5% | 1.13% | 4-way, 21% wasted | `stall_short_scoreboard` 43.2% |

Native FA-2 on A100 reaches ~50% of fp16 peak (~156 TFLOPs). The realistic ceiling for a WMMA-based kernel is ~25-30% — beyond that requires raw MMA PTX (CUTLASS-style). This implementation stops at v4; further optimizations are continued in [`cuda/flash_attn_cutlass/`](../flash_attn_cutlass/) using CuTe.

---

## v1 — single-warp WMMA, synchronous loads

One warp performs both WMMAs while three idle. Cooperative loads via 128 threads with `__syncthreads()`. P round-trips through smem; O accumulator in registers (one thread per row). Scores buffer aliased with O temp (32 KB savings on hdim64).

Smem: 72 KB (hdim64) / 128 KB (hdim128), allocated via `cudaFuncSetAttribute`.

| Seq | hdim64 (ms) | hdim128 (ms) | hdim64 vs Native |
|---|---|---|---|
| 128 | 0.16 | 0.21 | 11.4× |
| 512 | 0.68 | 1.42 | 21.6× |
| 2048 | 7.80 | 13.53 | 36.0× |

## v2 — multi-warp WMMA distribution

Distribute WMMA tiles across all 4 warps; each owns a 32-row band of the output. Removes the single-warp gate.

| Seq | hdim64 (ms) | vs v1 | hdim128 (ms) | vs v1 |
|---|---|---|---|---|
| 128 | 0.12 | 1.34× | 0.15 | 1.40× |
| 512 | 0.65 | 1.05× | 0.95 | 1.49× |
| 2048 | 6.23 | 1.25× | 8.84 | 1.53× |

Speedup is well below the theoretical 4× ceiling because removing the warp idle reveals the next bottleneck — synchronous K/V loads — which the profile then localizes to bank conflicts:

> L1/TEX 98.5% saturated. 32-way average bank conflict on every WMMA load (92% of wavefronts wasted). DRAM throughput 0.19%.

The 32-way conflict is structural: every smem buffer uses stride 64 halves = 128 bytes = exactly one full row of 32 banks, so all 16 rows of a WMMA fragment hit the same bank columns. The kernel is bank-conflict-bound, not memory-bound.

## v3 — smem padding to break bank-conflict alignment

Pad each smem buffer's row stride so consecutive rows start in different banks. fp16 buffers padded by 8 halves (16 bytes); fp32 buffers padded by 4 floats (also 16 bytes). Both maintain the 16-byte alignment required by `ldmatrix` and `store_matrix_sync`.

Smem grew 72 → 79 KB (hdim64) / 128 → 136 KB (hdim128).

| Seq | hdim64 (ms) | vs v2 | hdim128 (ms) | vs v2 |
|---|---|---|---|---|
| 128 | 0.04 | 3.0× | 0.07 | 2.1× |
| 512 | 0.13 | 5.0× | 0.32 | 3.0× |
| 2048 | 1.14 | 5.5× | 2.51 | 3.5× |

Profile delta:

| Metric | v2 | v3 |
|---|---|---|
| L1/TEX Throughput | 98.5% | 72.7% |
| Compute Throughput | 3.5% | 16.1% |
| Bank conflicts (loads) | 32-way, 92% wasted | 4-way, 21% wasted |
| `stall_short_scoreboard` | 84.9% | 40.8% |

The 4-way residual is the consequence of padding (vs full XOR swizzling). Stride 72 halves keeps a periodic shift of 4 banks per row — better than 32-way but not zero.

## v4 — coalesced O writeback

Stage normalized O into smem (reusing `smem_q`, free after the KV loop), sync, then have all threads cooperatively write smem → gmem with consecutive threads writing consecutive bytes. Replaces the previous "each thread writes its full row" pattern, which scattered 32-byte cache sectors across rows.

| Seq | hdim64 (ms) | vs v3 | hdim128 (ms) | vs v3 |
|---|---|---|---|---|
| 128 | 0.036 | 1.14× | 0.048 | 1.47× |
| 512 | 0.105 | 1.23× | 0.263 | 1.20× |
| 2048 | 1.095 | 1.04× | 2.382 | 1.05× |

Modest gains. `ncu` predicted ~60% potential speedup from coalescing, but DRAM throughput was 1% in v3 (HBM had massive slack) — the wasted bandwidth wasn't on the critical path. `ncu`'s "Est. Speedup" is an upper bound assuming the affected subsystem is the bottleneck. Cross-check the throughput of that subsystem before acting on a recommendation.

hdim128 saw larger relative gains because its writeback transfers 2× more bytes, so the wasted fraction was a larger share of total time.

## Stopping point

After v4, the residual smem bank conflicts (4-way) and the WMMA instruction granularity itself become the primary limits. The remaining ~3-5× gap to native FA-2 requires:

- True XOR swizzling (0-way conflicts), which cannot be expressed with `load_matrix_sync` — needs raw `ldmatrix` PTX with per-thread address computation.
- Register-resident P across iterations, requiring hardcoded fragment-to-thread mapping for warp-shuffle softmax reductions and accumulator → matrix_a layout conversion.
- Software-pipelined `cp.async`, double-buffered K/V tiles.
- Replacing WMMA's `m16n16k16` HMMA with the finer-grained `m16n8k16` MMA PTX.

Each of these is an extensive manual implementation in raw WMMA, and the resulting code is essentially what CUTLASS abstracts via `Layout`, `TiledMma`, and `Copy_Atom` types. Continued in [`cuda/flash_attn_cutlass/`](../flash_attn_cutlass/) using CuTe (CUTLASS 3.x's layout algebra), the same idiom used in production FA-2.

---

## Build, test, benchmark

From the repo root:

```bash
make build-fac     # compile the extension
make test-fac      # parametrized correctness tests vs scaled_dot_product_attention
make bench-fac     # benchmark vs PyTorch SDPA
```

## Profiling

```bash
make prof-fac PROF_ARGS="--seq 2048"             # basic metrics
make prof-fac-full PROF_ARGS="--seq 2048"        # adds Warp State Stalls + Memory Workload Analysis
make prof-fac-rep PROF_ARGS="--seq 2048"         # .ncu-rep file for ncu-ui
make prof-nsys-fac PROF_ARGS="--seq 2048"        # Nsight Systems timeline (works when ncu is blocked)
```

On shared dev hosts where the GPU performance counters are held by a monitoring agent, `ncu` will report "driver resource unavailable". `nsys` works as a fallback for kernel timing.

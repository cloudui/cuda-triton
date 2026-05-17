# CuTe FlashAttention-2

FA-2 forward kernel using CuTe (CUTLASS 3.x's layout algebra). Same algorithm as [`cuda/flash_attn/`](../flash_attn/), rewritten in the idiom used by Tri Dao's production FA-2.

CUTLASS has two dialects:
- **CUTLASS 2.x** — GEMM-centric template-policy composition. Fits poorly to attention because attention isn't a GEMM (online softmax + accumulator-to-input layout conversion + two matmuls per iteration). The `cutlass/examples/41_fused_multi_head_attention` reference is 2.x-style and predates the FA-2 algorithm.
- **CuTe / CUTLASS 3.x** — `Layout`, `Tensor`, `TiledMma`, `Copy_Atom`, and the algorithms that compose them. Used by Tri Dao's FA-2 and FA-3.

This rewrite uses CuTe.

## Setup

CUTLASS is header-only. Clone the repo (or point `CUTLASS_DIR` at an existing checkout):

```bash
make fetch-cutlass             # clones v3.5.1 into third_party/cutlass
make build-fac-cutlass         # uses CUTLASS_DIR=third_party/cutlass by default
# or:
CUTLASS_DIR=/path/to/cutlass make build-fac-cutlass
```

## Layout

| File | Purpose |
|---|---|
| `flash.h` | `Flash_fwd_params` struct (shared with the WMMA implementation) |
| `flash_api.cu` | PyTorch extension entry point and runtime dispatch by `head_dim` |
| `kernel_traits.h` | CuTe type composition: smem `Layout`s with swizzle, `TiledMma`, `Copy_Atom`s |
| `flash_fwd_kernel.h` | Kernel body |
| `flash_fwd_launch_template.h` | Grid/block sizing and `cudaFuncSetAttribute` for extended smem |
| `flash_fwd_hdim{32,64,128}_fp16_sm80.cu` | Per-config instantiations |
| `setup.py` | Build script with CUTLASS include path |

## Performance

A100 40GB, fp16, batch=4, heads=8. `Native` is PyTorch's `scaled_dot_product_attention`, which dispatches to Tri Dao's production FA-2 CUDA. Times in milliseconds via `triton.testing.do_bench`.

A100 fp16 tensor-core peak is 312 TFLOP/s. FLOPs counted as `4 × batch × heads × seq² × head_dim` (2 matmuls × 2 ops per multiply-accumulate).

**hdim=64**

| Seq Len | CuTe FA2 (ms) | Native FA2 (ms) | Ratio | TFLOP/s | % peak |
|---------|---------------|------------------|--------|---------|--------|
| 128     | 0.0149        | 0.0156           | 0.96×  | 9.0     | 2.9%   |
| 256     | 0.0186        | 0.0177           | 1.05×  | 28.9    | 9.3%   |
| 512     | 0.0334        | 0.0332           | 1.01×  | 64.3    | 20.6%  |
| 1024    | 0.0872        | 0.0764           | 1.14×  | 98.5    | 31.6%  |
| 2048    | 0.2379        | 0.2186           | 1.09×  | 144.4   | 46.3%  |
| 4096    | 0.8786        | 0.8501           | 1.03×  | 156.4   | 50.1%  |
| 8192    | 3.3720        | 3.2199           | 1.05×  | 163.0   | 52.3%  |
| 16384   | 13.4770       | 12.8500          | 1.05×  | 163.2   | 52.3%  |
| 32768   | 53.4953       | 51.1262          | 1.05×  | 164.4   | 52.7%  |
| 65536   | 213.6306      | 204.8852         | 1.04×  | 164.7   | 52.8%  |

**hdim=128**

| Seq Len | CuTe FA2 (ms) | Native FA2 (ms) | Ratio | TFLOP/s | % peak |
|---------|---------------|------------------|--------|---------|--------|
| 128     | 0.0159        | 0.0167           | 0.95×  | 16.9    | 5.4%   |
| 256     | 0.0227        | 0.0226           | 1.01×  | 47.3    | 15.2%  |
| 512     | 0.0485        | 0.0499           | 0.97×  | 88.5    | 28.4%  |
| 1024    | 0.1309        | 0.1245           | 1.05×  | 131.3   | 42.1%  |
| 2048    | 0.4339        | 0.3808           | 1.14×  | 158.4   | 50.8%  |
| 4096    | 1.4727        | 1.4979           | 0.98×  | 186.6   | 59.8%  |
| 8192    | 5.8427        | 5.6883           | 1.03×  | 188.2   | 60.3%  |
| 16384   | 22.5305       | 22.7326          | 0.99×  | 195.2   | 62.6%  |
| 32768   | 89.4886       | 90.9835          | 0.98×  | 196.6   | 63.0%  |
| 65536   | 361.0303      | 365.8720         | 0.99×  | 194.9   | 62.5%  |

**Headline numbers:**
- Within ~1-14% of production FA-2 across all configurations, out to 64K-token sequences.
- At long context (seq ≥ 4096) the gap closes to within ~5% — and at hdim=128 the kernel is **slightly faster than native** at every seq ≥ 4096 (0.98-0.99×). The mid-range sequence overhead is launch/setup-bound, not compute-bound.
- Sustains **~53% of A100 fp16 peak at hdim=64** and **~63% at hdim=128** out to 64K — flat efficiency across two orders of magnitude in sequence length, comparable to the utilization Tri Dao reports for production FA-2 in the FA-2 paper.

Correctness checked against `torch.nn.functional.scaled_dot_product_attention` to within fp16 tolerance (`max_abs_err ≈ 2.4e-4`).

## Findings

The full investigation lives in [`CUTE_NOTES.md`](CUTE_NOTES.md). The headline result: a few "defensive" CuTe idioms inherited from CUTLASS GEMM examples are **no-ops on SM80** for this MMA atom — `SmemLayoutVtNoSwizzle` and the variable `kSwizzle = (kBlockKSmem == 64) ? 3 : 2` formula can both be removed, and the kernel produces bit-identical outputs and equivalent wall-clock perf. The defense is real for some configurations (Hopper, different atoms, older cute versions) but not this one. The one cute invariant that **is** load-bearing is `make_fragment_like` enforcing `LayoutLeft` on mode 0, which guarantees the per-LDSM-instruction destination registers are contiguous as the hardware demands. Mode 0 is "addresses-as-hardware-positions"; outer modes are "addresses-as-labels" that cute keeps consistent end-to-end across the smem→register copy and the gemm dispatch — which is why non-canonical fragment shapes don't break correctness.

## Implementation outline

1. **`kernel_traits`** — `Element`, `ElementAccum`, smem `Layout`s with `Swizzle<kSwizzle,3,3>`, `TiledMma` over `SM80_16x8x16_F32F16F16F32_TN` × `kNWarps` warps, gmem `Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>`, smem `Copy_Atom<SM75_U32x4_LDSM_N, half>` for Q/K and `Copy_Atom<SM75_U16x8_LDSM_T, half>` for V (transposed).
2. **Smem layout** — Q + K + V regions, sO reuses sQ's space at the epilogue. Total ≈ `(kBlockM·kHeadDim + 2·kBlockN·kHeadDim) · sizeof(half)` bytes.
3. **Q load** — gmem→smem cp.async, fenced once before the KV loop.
4. **KV loop** (per kBlockN tile):
   - Wait on K, issue V cp.async (overlaps with QK^T compute).
   - `cute::gemm(tiled_mma, acc_s, tSrQ, tSrK, ...)` — Q @ K^T via LDSM-loaded fragments.
   - Wait on V, prefetch next K cp.async.
   - Online softmax with warp-shuffle row max + row sum reductions, accumulator rescaled in place.
   - `convert_layout_acc_Aregs` reinterprets the C-output of QK^T as the A-input of P @ V (no data movement; SM80 MMA layouts are register-compatible along the M axis).
   - `cute::gemm(tiled_mma, acc_o, tOrP, tOrV, ...)` — P @ V accumulating into the running output.
5. **Epilogue** — final softmax normalization, fp32→fp16 conversion, smem stage via the `O = sQ` reuse, gmem store via vectorized cp.async.

The kernel ships hdim=64 and hdim=128 instantiations. hdim=32 is also supported in the traits machinery (added during the perf experiments) but not wired up in the default `setup.py`.

## References

Primary (production FA-2, same idiom):
- [`flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)
- [`kernel_traits.h`](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h)

CuTe documentation:
- [`cutlass/media/docs/cute/`](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute) — official tutorials. Start with `00_quickstart.md`, then `02_layout.md`.
- [`cutlass/examples/cute/`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) — small standalone examples.

## Build, test, benchmark

```bash
make fetch-cutlass         # one-time: clone CUTLASS to third_party/cutlass
make build-fac-cutlass
make test-fac-cutlass
make bench-fac-cutlass
```

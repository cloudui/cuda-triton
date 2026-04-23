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
| `flash_fwd_hdim{64,128}_fp16{,_causal}_sm80.cu` | Per-config instantiations |
| `setup.py` | Build script with CUTLASS include path |

The kernel body is currently a stub. `kernel_traits.h` and `flash_fwd_kernel.h` contain TODO comments sketching the type definitions and algorithm structure.

## Implementation outline

1. **`kernel_traits`** — define `Element`, `ElementAccum`, smem `Layout`s with `Swizzle<3,3,3>`, `TiledMma` (composing one or more `SM80_16x8x16_F32F16F16F32_TN` MMA atoms), and `Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL, ...>` for cp.async loads.
2. **`SharedStorage`** — typed wrapper over the extern smem region, sized via `cosize_v<Layout> * sizeof(Element)`.
3. **Q load** via `TiledCopy` (cp.async, then commit/wait).
4. **KV loop**:
   - `cute::gemm(tiled_mma, S_frag, Q_frag, K_frag)`
   - Online softmax in fragment registers, with warp-shuffle reductions for row max and row sum.
   - `convert_layout_acc_Aregs(S_frag) → P_frag` to rebind the accumulator layout into matrix_a layout for the next matmul.
   - `cute::gemm(tiled_mma, O_frag, P_frag, V_frag, O_frag)` — accumulator stays in registers across iterations.
   - Per-iteration rescale of `O_frag` by `correction`, applied through fragment-aware traversal.
5. **Epilogue** — normalize O by `l_state`, store coalesced via `TiledCopy`.
6. **Pipelining** — wrap the KV loop with `cute::cp_async_pipeline` for 3-stage overlap of loads and compute.

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

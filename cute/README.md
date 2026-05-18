# CuTe Python DSL — FlashAttention-2 learning track

> Section below is AI-generated, currently WIP almost done.

> **Status: in-progress.** This is the planned follow-up to the [CuTe FA2 blog post](https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/), porting the same kernel from C++ CUTLASS to NVIDIA's new Python CuTe DSL.

## Why bother with the Python DSL?

Same CuTe layout algebra, same MMA/Copy atoms, same per-thread/per-CTA
mental model. What changes:

| Concept | C++ CUTLASS | Python DSL |
|---|---|---|
| Compile-time int | `cute::Int<N>{}`, `_4`, `_8` | plain Python `int` (when in `@cute.jit`/`@cute.kernel` body, traced as constexpr) |
| Layout build | `Layout<Shape<_8,_64>, Stride<_64,_1>>{}` | `cute.make_layout((8, 64), stride=(64, 1))` |
| Tensor over ptr | `make_tensor(make_smem_ptr(p), layout)` | `cute.make_tensor(ptr, layout)` |
| Swizzle | `composition(Swizzle<3,3,3>{}, atom)` | `cute.make_composed_layout(cute.make_swizzle(3,3,3), 0, atom)` |
| MMA atom | `MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>` | `cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, (16,8,16))` |
| Tiled MMA | `TiledMMA<atom, Layout<Shape<W,_1,_1>>, Tile<...>>` | `cute.make_tiled_mma(atom, atom_layout_mnk=(W,1,1), permutation_mnk=...)` |
| cp.async atom | `Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>` | `cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=...), Float16, num_bits_per_copy=128)` |
| ldmatrix atom | `Copy_Atom<SM75_U32x4_LDSM_N, half_t>` | `cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), Float16)` |
| Tiled copy | `make_tiled_copy(atom, thr_layout, val_layout)` | `cute.make_tiled_copy_tv(atom, thr_layout, val_layout)` |
| Partition src/dst | `gmem_thr_copy.partition_S(gQ)` | `thr_copy.partition_S(gQ)` |
| GEMM call | `cute::gemm(tiled_mma, acc, A, B, acc)` | `cute.gemm(tiled_mma, acc, A, B, acc)` |
| Copy call | `cute::copy(tiled_copy, src, dst)` | `cute.copy(atom, src, dst)` |
| Kernel entry | `__global__ void kernel(...)` | `@cute.kernel def kernel(...)` |
| Launcher | host C++ `<<<grid, block>>>` | `@cute.jit` host fn that calls `kernel(...).launch(grid=..., block=..., smem=...)` |
| Compile | nvcc | `cute.compile(host_fn, *args)` (lazy / cached) |
| smem alloc | `extern __shared__ char smem[]` + manual ptr math | `cutlass.utils.SmemAllocator.allocate_tensor(dtype, layout, swizzle=...)` |

## Two big mental shifts

1. **No headers, no templates.** Constants like `kBlockM`, `kHeadDim`,
   `kNWarps` are just Python ints in `@cute.jit` host code. They become
   compile-time constants in the generated MLIR because the function is
   traced once per (constexpr) argument signature.

2. **Tracing model.** Inside `@cute.kernel`/`@cute.jit`, your Python is
   traced into MLIR. Plain `int`/`float`/tuples become MLIR constants.
   Tensors become MLIR SSA values. Standard Python `for`/`if` over
   constexpr ranges unrolls; over `cute.range_dynamic(...)` becomes a real
   loop. This is the cuTeDSL analog of "constexpr if" + template metaprog.

See `00_layouts.py` first — it runs on the CPU, no GPU needed, and shows
that the *layout algebra* is identical to C++.

## File ordering

| File | Maps to (C++) | What it teaches |
|---|---|---|
| `00_layouts.py` | `kernel_traits.cuh` (`SmemLayoutAtomQ`, `SmemLayoutQ`, swizzle) | Layouts + swizzle — host-only, pure algebra |
| `01_vector_add.py` | none (warmup) | The `@cute.kernel` + `@cute.jit` + `cute.compile` + launch pattern with torch tensors |
| `02_tiled_copy_g2s.py` | `gmem_tiled_copy_QKV` block in `flash_fwd_kernel.h` | gmem→smem cp.async, `make_tiled_copy_tv`, `partition_S/D` |
| `03_tiled_mma.py` | `TiledMma` block + GEMM #1 in `flash_fwd_kernel.h` | warp-level MMA atom, accumulator partitioning, `cute.gemm` |
| `flash_fwd.py` | `flash_fwd_kernel.h` | Skeleton FA2 fwd kernel, marked TODOs at each stage |
| `test_smoke.py` | n/a | pytest harness — runs each example as a smoke check |

## Running

```bash
pytest cute/test_smoke.py -v        # all examples
python cute/00_layouts.py           # standalone, prints layouts
python cute/01_vector_add.py        # standalone, runs and verifies
```

Requires `nvidia-cutlass-dsl` (already in `requirements.txt`). A100 / sm80
is what the examples target.

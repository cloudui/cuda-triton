# CuTe / CUTLASS Intricacies

A reference of the non-obvious things that bit me writing FA2 in CuTe.
Not a tutorial — assumes you know basic CUDA, smem, warp shuffles.

## Mental models

### Layouts are change of basis

CuTe layouts are basis vectors over a "vector" of memory cells. Same data,
different coordinates depending on the layout function applied.

- `retile_D(tCrA)` → same registers, different per-thread mode structure
- `convert_layout_acc_rowcol(acc_s)` → same registers, 2D row/col view
- `convert_layout_acc_Aregs(acc_s)` → same registers, A-frag layout
- `sVtNoSwizzle` vs `sVt` → same smem bytes, different addressing scheme

All of these are layout function rewrites at compile time. Zero runtime cost.
The "tensor" is just `(pointer, layout function)`. Multiple tensors over the
same pointer with different layouts are valid views.

### Registers don't have addresses

- Registers are per-thread, not shared. Thread 0's R5 ≠ thread 1's R5
  (different physical hardware in different banks).
- "Partition" for registers ≠ "partition" for smem/gmem. Smem partition divides
  shared territory; register partition just allocates each thread's private
  fragment with a layout.
- Layout for register tensors tells the compiler "the value at coord (i, m, k)
  is in register Rn." Compile-time SSA mapping.
- `partition_fragment_*` derives the right layout from the MMA atom's TV layout.
  You don't compute it manually.

### The "last partition" rule

Only the **final partition into gmem** has to land values at the correct logical
coords. Intermediate stages (regs → smem → regs → gmem) can use arbitrary
layouts as long as each stage round-trips data faithfully.

Within one `cute::copy`, source and destination must use the **same TiledCopy**
(so thread t's source slice and destination slice cover the same logical
coords). Across stages, different TiledCopies are fine — values pass through
the same physical smem cells, just accessed by different per-thread mappings.

### TiledCopy is policy, not data

`TiledCopy` describes **"how do my threads cooperate to access something"** —
it has no inherent knowledge of what it copies. It's a threading policy you
apply to whatever source/destination tensors you hand to `partition_S` /
`partition_D`. The actual SASS emitted (LDG, LDS, ST, STS, cp.async) is
derived from the source/dest memory spaces, not from the TiledCopy itself.

This separation is why the same smem layout can be accessed by multiple
TiledCopies (cp.async writer + ldmatrix reader use the same `SmemLayoutQ` with
different threadings). And why the same TiledCopy can be re-applied to
different data tensors.

### Design dependency arrow: smem layout → gmem TiledCopy

Smem layout is the **constraint source** because hardware is most picky there
(ldmatrix shape, swizzle bits, bank patterns). Once you've picked a smem
layout that satisfies all the hardware constraints, the gmem TiledCopy is the
**consequence** — its threading is chosen to feed the smem layout without
bank conflicts on the writes.

Order of design decisions for a new kernel:

1. Pick the MMA atom (forces register / operand layouts)
2. Pick the smem layout (must support ldmatrix patterns + bank avoidance)
3. Pick the gmem TiledCopy (must feed the smem layout without conflict)
4. Write the kernel that uses TiledCopies to move data through the layouts

Trying to start with "how should I distribute threads" first leads you to
TiledCopies that conflict with the smem layout you eventually need.

Concrete: smem atom shape `(8, kBlockKSmem)` is co-designed with the
threading. 8 rows matches ldmatrix's 8x8 tile structure; `kBlockKSmem` matches
the per-pass write width that avoids bank conflicts (8 threads × 8 halfs per
thread = 64 cols at head_dim=64). The gmem TiledCopy's `kThreadsPerRow` is
then derived as `kBlockKSmem / 8`. Both layouts agree on cell positions even
though only the gmem layout names threads.

## The MMA atom: SM80_16x8x16

The standard FA2 atom. Per-atom shapes:
- A operand: 16×16 halfs, 8 halfs per thread
- B operand:  8×16 halfs, 4 halfs per thread
- C accumulator: 16×8 fp32, 4 fp32 per thread

C's per-thread 4 elements are arranged as **(2 rows, 2 cols)** within one atom.
The 2 rows are 8 apart (lane t holds rows {t/4, t/4+8}), 2 cols are adjacent.

### Constants that fall out of this layout

| constant | value | reason |
|---|---|---|
| `MMA = 4` | 4 fp32 per thread per C atom | hardware TV layout |
| Substructure `(2, 2)` | 2 rows × 2 cols per C atom per thread | hardware |
| `kNRows = 2 * MMA_M` | per-thread row count | 2 rows per atom × MMA_M atoms |
| `Allreduce<4>` | 4 lanes share each row | 4 lanes × 2 cols/lane = 8 cols/row |
| `MMA_K` lives in A and B but not C | C accumulates; A, B feed K-tiles | contraction collapses K |

These constants are **specific to SM80_16x8x16**. Different atoms (fp64, Hopper
WGMMA) have different per-thread shapes, different `kNRows` formula, different
Allreduce widths.

### Per-atom shape mismatch between C and A

C is 16×8 per atom. A is 16×16. **One A atom = TWO C atoms paired in K direction.**

`convert_layout_acc_Aregs` pairs 2 consecutive C atoms (in N) into 1 A atom (in
K). This is why `MMA_N` must be even when going from GEMM #1 (acc_s as C) to
GEMM #2 (rP as A).

Per thread: 4 fp32 (C) → cast to 4 fp16 → pair 2 atoms → 8 fp16 (A). Element
count works out.

## The reshape from (MMA, MMA_M, MMA_N) to ((2, MMA_M), (2, MMA_N))

`convert_layout_acc_rowcol` exposes the per-atom (2, 2) sub-structure as
outer (rows, cols) dims, giving a clean 2D view.

```cpp
auto sl = logical_divide(acc_layout, Shape<_2>{});
// ((2, 2), MMA_M, MMA_N)

return make_layout(make_layout(get<0,1>(sl), get<1>(sl)),  // ROW dim: outer-2 + MMA_M
                   make_layout(get<0,0>(sl), get<2>(sl))); // COL dim: inner-2 + MMA_N
```

**Why `get<0,1>` for rows and `get<0,0>` for cols** (looks backwards):
- `logical_divide` puts inner (stride-1) sub-mode at index 0
- For SM80, consecutive registers (stride 1) hold same-row adjacent-cols → inner-2 = COLS
- Skip-pair registers (stride 2) hold different-row same-col → outer-2 = ROWS
- So row dim takes outer-2, col dim takes inner-2

Strides come out as `((s_MMA, s_MMA_M), (2*s_MMA, s_MMA_N))`. CuTe inherits
strides automatically when `make_layout` is given existing Layout pieces (not
bare shapes).

## Swizzling

`Swizzle<B, M, S>` XORs B bits at offset M+S into B bits at offset M.

For `Swizzle<3, 3, 3>`:
- Bottom 3 bits ([0:3)) — **untouched** = the 16-byte (8-half) contiguous vector
- Middle 3 bits ([3:6)) — XOR destination = row index when row stride = 8
- Top 3 bits ([6:9)) — XOR source = K-tile index when col stride = 1

### Key invariant

**Swizzles preserve contiguity within the bottom M bits.** The 8-half
(16-byte) vector that each thread loads is always intact. Only WHICH 16-byte
chunk lives at WHICH smem address gets scrambled.

Hardware that loads/stores N-byte vectors (cp.async 16B, ldmatrix per-row 16B,
ST.E.128) works fine because each transaction operates on one chunk at a time.

### Why the swizzle alone can't fix gmem→smem write conflicts at head_dim ≥ 128

Swizzle XORs **row bits with col bits** — only permutes across rows. If 16
threads write to the same row at different col chunks (cols 0-63 and 64-127),
they collide on banks regardless of swizzle. Swizzle can't disambiguate writes
within a row.

Solution: **kBlockKSmem = 64** splits head_dim into 64-wide pages, limiting
gmem load to 8 threads per row per page. Each page's 8 threads cover 8
disjoint bank sets (no conflict). Two passes per row instead of one.

Three bank-conflict-avoidance techniques (interchangeable for some patterns):
- **Padding**: shift in space (extra cols)
- **Swizzle**: shift via XOR
- **Staggering / page split**: shift in time (fewer simultaneous writes)

Page split = staggering, expressed at the layout level.

## TiledMma layout for FA2

```cpp
Layout<Shape<Int<kNWarps>, _1, _1>>  // all warps split M, none split N
```

**Hard constraint for any kernel with row-wise reductions in softmax**: warps
must NOT split N. Otherwise rows are spread across multiple warps, and rowmax
/ rowsum need cross-warp reduction (slow, requires smem + barriers).

3D layout because matmul has 3 axes (M, N, K). Split-K (warps_K > 1) is rare
and would require partial output combination at the end.

For 2x2 warps as in plain GEMM tutorials → breaks FA2's softmax. Don't copy
that layout.

### Tile<>

`Tile<Int<16 * kNWarps>, _16, _16>` extends per-pass extent. Each warp does
multiple atoms per logical pass. For N=16: each warp does 2 atoms in N (since
atom_N=8). Lets you cover more output per `cute::gemm` call.

## The two-pass softmax structure

GEMM #1 produces raw S in `acc_s` (fp32, MMA C-frag layout). Then per
iteration:

1. Compute `row_max = max(prev_row_max, rowmax(acc_s))`
2. Compute `correction = exp2((prev_max - new_max) * softmax_scale_log2)`
3. Multiply `acc_o *= correction` (rescale prior accumulator)
4. Multiply `row_sum *= correction` (rescale prior sum)
5. Compute P in place: `acc_s = exp2(acc_s * scale - new_max * scale)`
6. `row_sum += rowsum(acc_s)` (accumulate this block's sum)

`acc_s` is fresh each iteration (cleared, recomputed by GEMM). `acc_o` and
`row_sum` carry across iterations. The correction is **multiplied through**,
not exp'd again — no double-scaling.

### Asymmetric quad_allreduce in reduce_max vs reduce_sum

- `reduce_max` does the cross-lane shuffle EVERY iteration — needed because
  the 4 lanes that share a row must agree on `max` for the correction factor.
- `reduce_sum` SKIPS the cross-lane shuffle during the loop. The 4 lanes keep
  their own partial sums. Only when `normalize_softmax_lse` runs (once at end)
  does the final `quad_allreduce_(row_sum)` combine them.

This saves N shuffles over the n_block loop. Tri Dao's optimization.

### softmax_scale_log2

Combines the attention scale (`1/sqrt(d)`) and the log_2(e) conversion into
one constant: `softmax_scale_log2 = (1/sqrt(d)) * log_2(e)`. Computed on host,
passed as a kernel arg. Used inside `exp2(x * scale_log2 - max_scaled)` —
single FMA per element.

The hidden_dim scaling is **always** applied to `tensor` (via `scale_log2`).
The `Scale_max` flag controls whether it's also applied to `max` in
`scale_apply_exp2` — `Scale_max = false` is for the rare case where `max`
was pre-scaled upstream (e.g., Q-prescaled variant). Default `true` for
standard FA2.

## Pipelining (cp.async)

### How fences work

- `cp_async_fence()` commits all cp.asyncs since the last fence as a batch
- `cp_async_wait<N>()` waits until ≤ N batches still pending (per-thread)

You can have multiple batches alive and drain them selectively. FA2 uses
`<0>` at each wait point because, at each point, only one batch happens to be
in flight.

### FA2's two overlap phases per iteration

```
1. wait K[n]            ← K landed during prev iter's P·V compute
2. issue V[n] async
3. compute S = Q · K^T  ← V loads in parallel
4. wait V[n]            ← V mostly arrived
5. issue K[n+1] async
6. compute P · V        ← K[n+1] loads in parallel
```

Each gmem load is hidden behind a compute phase that doesn't depend on it.
Loading K and V together upfront would lose this — both loads serialize with
compute.

### Sync rules

- `cp_async_wait<N>` is **per-thread**. Need `__syncthreads()` after if reads
  will be cross-thread.
- `__syncthreads()` needed when: a thread reads smem cells that other threads
  recently wrote.
- NOT needed for: register-only ops (mma.sync, scalar arithmetic), ldmatrix
  reading already-stable smem, register copies via `cute::copy`.
- The epilogue's `regs → smem → regs → gmem` needs a sync between the smem
  write (MMA partition) and the smem read (gmem-copy partition) because they
  use different per-thread mappings.

### Cache: .ca vs .cg

- `.ca` (CACHEALWAYS): caches in L1 + L2
- `.cg` (CACHEGLOBAL): caches only in L2, bypasses L1

For FA2's streaming pattern (each tile read once per CTA, reuse caught by L2
across CTAs), `.cg` is slightly preferred. For Ampere with full smem
carveout, L1 is essentially unavailable so `.ca` and `.cg` behave the same.

Difference is ~1-3% in profiled kernels. Both work.

## Epilogue: regs → smem → regs → gmem

There's no smem→gmem direct instruction. Every gmem write goes through
registers as a staging buffer.

The smem detour serves a purpose: **reorganize per-thread data distribution
from MMA-scattered to gmem-coalesced**. Without it, direct register→gmem
writes are non-coalesced (each thread writes scattered cells per the MMA
partition).

```
Stage 1: regs (MMA layout) → smem (MMA partition writes scattered cells)
        [__syncthreads() — different partitions cross-read]
Stage 2: smem → regs (gmem-copy partition reads contiguous 16B per thread)
Stage 3: regs → gmem (vectorized 16B stores, coalesced)
```

Skip stage 1+2 → ~5-10% slower epilogue (non-coalesced writes). For v1, this
is acceptable.

### Naming convention in the epilogue

Tri Dao's prefix system:
- `tO` — gmem-copy partition
- `tacc` — MMA-derived partition
- `r`, `s`, `g`, `c` — register, smem, gmem, coordinate (identity)
- Trailing `O` — "for the output tensor"

Examples:
- `taccOrO` = thread + acc-partition + reg + O
- `taccOsO` = thread + acc-partition + smem + O
- `tOrO`, `tOsO`, `tOgO` = thread + gmem-copy partition + (reg/smem/gmem) + O
- `tOcO`, `taccOcO` = identity coord tensors for predication / row-owner
  detection

## Compile-time integers (cute::Int<N>)

CuTe distinguishes:
- `int x` — runtime integer
- `cute::Int<N>` — compile-time integer (type-encoded value)
- `_4`, `_8`, ..., `_64` — shorthand for common `Int<N>{}`

For `Shape<...>` you must use the type form (`Int<kNWarps>`, not `kNWarps`).

Compile-time integers enable:
- Loop unrolling (`#pragma unroll` works)
- Static layout algebra (`logical_divide`, etc.)
- Register allocation (per-thread fragment shapes known)
- Constant folding throughout the layout function

For TiledMma's warp counts, MMA atom dims, kBlockM/N/K — always use compile
time. For runtime values (seqlen, batch, num_heads), use plain `int`.

## Kernel specialization per head_dim

FA2 compiles a **separate kernel per head_dim value** (32, 64, 96, 128, 160,
192, 224, 256). Each kernel hardcodes `kHeadDim` as a template constant,
enabling full unrolling.

Files like `flash_fwd_hdim64_fp16_sm80.cu` instantiate the kernel for one
specific config. Dispatcher in `flash_api.cu` picks the right one based on
runtime head_dim.

Cost: ~8 head_dims × 2 dtypes × 2 (causal/non-causal) × N archs = dozens of
kernels, hundreds of MB of binary. For server-side ML, this is fine.

For your kernel: pick 1-2 head_dims to support. Each compiled kernel has all
inner loops unrollable.

## Things that do NOT need separate handling

For an FA2 forward-only kernel with aligned shapes:

- **Predication** (`Is_even_MN`, `Is_even_K`, `cQ`, `tOpO`, etc.) — skip if
  you enforce `seqlen % kBlockM == 0`, `head_dim == kHeadDim`, etc.
- **LSE write-back** — only needed for backward pass / split-KV combine. Just
  normalize `acc_o` by `row_sum` and write.
- **Dropout, softcap, ALiBi, sliding window** — research bells and whistles.
- **Split-KV** (`compute_attn_1rowblock_splitkv`) — for inference KV cache
  splitting. Skip.
- **Append KV / paged cache / rotary** — inference-specific.
- **`Share_Q_K_smem`** — smem optimization for head_dim ≥ 128. Skip for v1.

After dropping these, the core FA2 algorithm is ~150 lines.

## Subtle gotchas / footguns

### `softmax.template softmax_rescale_o<...>(...)` requires `template`

When calling a member template through an instance of a templated struct, the
`template` keyword is required to disambiguate `<` from less-than:

```cpp
softmax.template softmax_rescale_o<true, false>(acc_s, acc_o, scale);
```

Without `template`, compiler error: "expected primary-expression before '<'".

This is needed for member templates of dependent types or templated structs.
For free functions (`FLASH::reduce_max<true>(...)`), no `template` keyword
needed.

### `scale_apply_exp2` and `scale_apply_` are different functions

- `scale_apply_` is a per-row multiply (used to rescale `acc_o` by correction
  factor)
- `scale_apply_exp2` is the per-element S → P transformation (multiply by
  scale, subtract max-scaled, exp2)

Both have "scale_apply" in the name but do different things. The exp2
version is the per-element softmax computation; the plain one is just a
multiply.

### `_0` strides on size-1 modes

When a layout has a size-1 mode, its stride is normalized to 0 (since
incrementing it doesn't change the offset). Don't be confused by `_0` strides
in printed layouts — they just mean "this mode has size 1."

### "use namespace cute" in headers

Don't put it at file scope (pollutes every includer). Put it inside your
namespace:

```cpp
namespace FLASH {
using namespace cute;
// ...
}
```

This scopes the import to your namespace. Each header that uses cute names
can include this at the top — redundant but defensive (no include-order
dependency).

### `Tensor` arguments by `const&`

Be careful about read-only inputs vs in-place modification. Tensors that
won't be modified should be passed as `Tensor const&`. Otherwise CuTe might
let you accidentally mutate.

### Partition shapes must match between source and destination

```cpp
cute::copy(tiled_copy, src, dst);
```

`src` and `dst` must have the **same per-thread shape** (the same TiledCopy
should be used to partition both). Compile error or wrong data otherwise.

If you need to bridge between different partitions (e.g., MMA partition →
gmem-copy partition), you must go through smem with a sync between.

## Summary: what made this hard

1. **Implicit hardware constraints.** The MMA atom's TV layout, swizzle bit
   alignment, ldmatrix patterns — all implicit, learned from convention or
   docs. Numbers like `(2, 2)`, `Allreduce<4>`, `kNRows = 2 * MMA_M` look
   magical until you trace them back to one specific hardware spec.

2. **Layout reshapes look opaque.** `convert_layout_acc_rowcol`,
   `convert_layout_acc_Aregs`, `retile_D` — each is a one-liner using
   `logical_divide` + `make_layout`, but the algebra is non-trivial and the
   intent (e.g., "pair 2 C atoms to make 1 A atom") isn't obvious from
   syntax.

3. **Multiple co-existing layout views.** `sV`, `sVt`, `sVtNoSwizzle` all
   point to the same smem; `tCrA` and `retile_D(tCrA)` are the same
   registers. Hard to tell at first which view is for what.

4. **Implicit per-thread vs CTA-level reasoning.** Code looks CTA-shaped but
   executes per-thread. The partition functions hide this — easy to forget
   you're in SIMT-land.

5. **Pipelining requires careful sync placement.** `cp_async_wait` is
   per-thread, `__syncthreads` is CTA-wide; different patterns need
   different combinations. Misplaced syncs are silent perf killers.

6. **Many things look the same but aren't.** `make_tensor` vs
   `make_fragment_like`, `partition_S` vs `retile_S`, `Layout` vs `Shape`,
   `int` vs `Int<N>`. Each pair has a subtle difference.

The CUTLASS team has internalized all of this and writes it fluently. The
docs assume you have too. The learning curve is real — most people who claim
to understand CUTLASS are shipping kernels by copying patterns and only
deeply understanding pieces. That's fine.

## References

- CUTLASS examples: `cutlass/examples/cute/tutorial/`
- CuTe layout algebra: `cutlass/media/docs/cute/02_layout_algebra.md`
- FA2 source: `flash-attention/csrc/flash_attn/src/`
- MMA TV layouts: `cutlass/include/cute/atom/mma_traits_sm80.hpp`
- Lei Mao's CuTe blog series — generally good for explaining the basics

When stuck, copy from FA2's `utils.h` / `softmax.h` / `kernel_traits.h` and
move on. Most CuTe code in production is composed from a small set of
canonical patterns.

---

## Empirical experiments: what's actually load-bearing on SM80

This section records experiments that tested whether several "defensive" CuTe
idioms inherited from CUTLASS examples are actually required for correctness
or performance on SM80 with the `SM80_16x8x16_F32F16F16F32_TN` MMA atom and
`SM75_U16x8_LDSM_T` copy atom (i.e., this kernel's exact configuration).

**TL;DR**: `SmemLayoutVtNoSwizzle` and the variable `kSwizzle = (kBlockKSmem == 64) ? 3 : 2`
formula are both no-ops for this kernel. They affect neither correctness nor
performance. A simpler kernel with `Sw<3,3,3>` for all head_dims and `sVt`
passed directly to `partition_fragment_B` produces bit-identical outputs and
the same wall-clock timing as the FA2-style defensive version.

### Setup

- A100 (SM80), CUDA 13, fp16.
- Variants tested (2×2 grid):
  - `kSwizzle`: original `(kBlockKSmem == 64) ? 3 : 2` vs. forced `3`.
  - V fragment source: `sVtNoSwizzle` (FA2 default) vs. `sVt` directly.
- Head dims: 32 (added support for this experiment), 64, 128.
- Tests: existing `tests/test_kernels.py::TestCUTLASSFlashAttention` (11 cases).
- Benchmark: `benchmarks/bench_cutlass_flash_attention.py` + a custom hdim=32
  sweep (batch=4, heads=8, seq ∈ {128, 256, 512, 1024, 2048, 4096}).

### Correctness

All 11 tests pass for all 4 variants × all 3 head_dims. `max_abs_err` vs.
`torch.nn.functional.scaled_dot_product_attention` is identical to baseline
at every shape (typically `2.44e-4`, occasionally `4.88e-4` for short seqs)
— bit-identical, not just within-tolerance.

This is initially surprising. `partition_fragment_B(sVt)` at hdim=32 returns
a non-canonical layout (`((2,2),(2,2),4)`) vs. the canonical
`((2,2),4,4)` from `sVtNoSwizzle`. And `partition_fragment_A(sQ)` at
hdim=32 returns swapped outer strides vs. its NoSwizzle counterpart. Both
"should" break the gemm if cute hardcoded canonical fragment layouts.

It doesn't break because **cute is layout-aware end-to-end**: the same
fragment layout drives both the smem→register copy (writes registers per the
fragment's layout) and the `cute::gemm` dispatch (slices fragments per the
fragment's layout to generate PTX register operand lists). Physical
registers may be allocated differently between paths, but the data routing
is consistent — the right values land in the right MMA hardware operand
positions either way. The only invariant cute strictly requires is that
mode 0 (the per-MMA-instruction payload) be `LayoutLeft`, which
`make_fragment_like` enforces unconditionally regardless of input.

### Performance

CUTLASS time in milliseconds (`triton.testing.do_bench`), batch=4, heads=8.

**hdim=64:**

| seq  | Baseline | Sw333+NoSw | Base+sVt | Sw333+sVt |
|------|----------|------------|----------|-----------|
| 128  | 0.0139   | 0.0138     | 0.0148   | 0.0154    |
| 256  | 0.0183   | 0.0179     | 0.0183   | 0.0178    |
| 512  | 0.0333   | 0.0328     | 0.0328   | 0.0328    |
| 1024 | 0.0870   | 0.0870     | 0.0871   | 0.0870    |
| 2048 | 0.2379   | 0.2376     | 0.2376   | 0.2372    |
| 4096 | 0.8749   | 0.8792     | 0.8745   | 0.8800    |

**hdim=128:**

| seq  | Baseline | Sw333+NoSw | Base+sVt | Sw333+sVt |
|------|----------|------------|----------|-----------|
| 128  | 0.0159   | 0.0159     | 0.0153   | 0.0154    |
| 256  | 0.0230   | 0.0222     | 0.0222   | 0.0226    |
| 512  | 0.0491   | 0.0486     | 0.0487   | 0.0485    |
| 1024 | 0.1303   | 0.1302     | 0.1308   | 0.1309    |
| 2048 | 0.4322   | 0.4344     | 0.4345   | 0.4336    |

**hdim=32** (added for this experiment):

| seq  | Baseline (Sw<2,3,3>+NoSw) | Sw333+NoSw | Base+sVt | Sw333+sVt |
|------|---------------------------|------------|----------|-----------|
| 128  | 0.0115                    | 0.0129     | 0.0137   | 0.0133    |
| 256  | 0.0151                    | 0.0171     | 0.0177   | 0.0151    |
| 512  | 0.0262                    | 0.0263     | 0.0262   | 0.0267    |
| 1024 | 0.0521                    | 0.0522     | 0.0521   | 0.0520    |
| 2048 | 0.1535                    | 0.1541     | 0.1541   | 0.1535    |
| 4096 | 0.5538                    | 0.5517     | 0.5509   | 0.5591    |

For seq ≥ 512 (where launch overhead doesn't dominate), all variants are
within ~1% of each other across all head_dims — well inside `do_bench`
run-to-run noise. Small-seq numbers (128, 256) show 5-15% spread but those
are sub-microsecond regimes where the timer is unreliable.

### Bank-conflict math (sanity check)

For LDSM accessing 8 consecutive rows at hdim=32 with `Sw<3,3,3>`:

| row | unswizzled offset | swizzled phys | byte addr | bank |
|-----|-------------------|---------------|-----------|------|
| 0   | 0                 | 0             | 0         | 0    |
| 1   | 32                | 32            | 64        | 16   |
| 2   | 64                | 72            | 144       | 4    |
| 3   | 96                | 104           | 208       | 20   |
| 4   | 128               | 144           | 288       | 8    |
| 5   | 160               | 176           | 352       | 24   |
| 6   | 192               | 216           | 432       | 12   |
| 7   | 224               | 248           | 496       | 28   |

Banks `{0, 16, 4, 20, 8, 24, 12, 28}` — all distinct, conflict-free. Same
result for `Sw<2,3,3>` because for rows 0..7 within one atom, bit 8 is 0,
so the source bits `[6..8]` for B=3 reduce to `[6..7]` — same as B=2.

### Why FA2 has the defensive code anyway

Best guess: inherited from CUTLASS GEMM examples written when cute was less
mature, never re-audited because the kernel ships only hdim=64/128 (where
the differences happen to vanish). The asymmetric "NoSwizzle for V only"
pattern probably comes from a CUTLASS GEMM example where only the
B-with-transpose operand had the issue. None of it bites in production
because no one runs the kernel at hdim=32.

### What CAN break

The one cute invariant that is genuinely load-bearing:
`make_fragment_like` enforces `LayoutLeft` on mode 0. This guarantees the
per-LDSM-instruction destination registers are contiguous, which the
hardware instruction requires. If cute didn't pin mode 0, retile_D could
produce non-contiguous destinations that LDSM can't issue. Mode 0 is
"addresses-as-hardware-positions"; outer modes are "addresses-as-labels"
that cute keeps consistent between copy and gemm.

### Reproducing

```bash
# Patch the kernel for whichever variant, then:
make build-fac-cutlass
make test-fac-cutlass
make bench-fac-cutlass
```

The variant matrix is two lines to edit:
- `kernel_traits.cuh:26` — `static constexpr int kSwizzle = ...;`
- `flash_fwd_kernel.h:91` — `Tensor tOrV = thr_mma.partition_fragment_B(...);`

For hdim=32 support: add `Traits_hdim32` in `kernel_traits.cuh`,
`run_mha_fwd_hdim32` in `flash_fwd_launch_template.h`, dispatch in
`flash_api.cu`, and `flash_fwd_hdim32_fp16_sm80.cu` to `setup.py`.

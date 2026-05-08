"""
02 — Tiled gmem→smem copy with cp.async.

This is the first half of the FA2 prologue: load a (kBlockM, kHeadDim)
tile of Q from gmem to smem using cp.async, then copy it back out to a
"sink" gmem buffer so we can verify the round-trip.

What we exercise:
  - cute.nvgpu.cpasync.CopyG2SOp     -> cp.async atom
  - cute.make_copy_atom              -> bake the copy width / dtype
  - cute.make_tiled_copy_tv          -> wrap atom with thread + value layouts
  - tiled_copy.get_slice(tid)        -> per-thread partitioner
  - thr_copy.partition_S / D         -> per-thread src / dst views
  - cute.copy(atom, src, dst)        -> emit the copy
  - cute.arch.cp_async_commit_group  -> cp.async fence (== `cp_async_fence()`)
  - cute.arch.cp_async_wait_group(0) -> drain (== `cp_async_wait<0>()`)
  - SmemAllocator.allocate_tensor    -> the Python equivalent of
                                        `extern __shared__ char smem[]`
                                        + manual ptr math

C++ counterpart (kernel_traits.cuh + flash_fwd_kernel.h prologue):

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy = gmem_tiled_copy_QKV.get_thread_slice(tid);
    Tensor tQgQ = gmem_thr_copy.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy.partition_D(sQ);
    cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

Run: python cute/02_tiled_copy_g2s.py
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.utils import SmemAllocator


# ---- Tile shape (compile-time-ish; pinned in this file). ----
kBlockM = 64
kHeadDim = 64
kNThreads = 128

# cp.async carries 16B = 8 fp16 per thread.
ELTS_PER_LOAD = 8
THREADS_PER_ROW = kHeadDim // ELTS_PER_LOAD  # = 8


@cute.kernel
def copy_kernel(
    gQ: cute.Tensor,         # (M, K) global Q
    gOut: cute.Tensor,       # (M, K) global sink
    sQ_layout: cute.ComposedLayout,  # swizzled smem layout, constexpr
):
    tid, _, _ = cute.arch.thread_idx()

    # ---- 1. Allocate sQ in smem with the swizzled layout. ----
    smem = SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float16, sQ_layout, byte_alignment=16)

    # ---- 2. Build the cp.async atom + tiled copy. ----
    # The atom is "what one thread does in one issue": 16B of fp16.
    g2s_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    g2s_atom = cute.make_copy_atom(
        g2s_op, cutlass.Float16, num_bits_per_copy=128
    )
    # The TiledCopy is "how N threads cooperate". 16 rows × 8 cols of
    # threads, each thread handling 8 fp16 contiguous in the col dim.
    thr_layout = cute.make_layout(
        (kNThreads // THREADS_PER_ROW, THREADS_PER_ROW),  # (16, 8)
        stride=(THREADS_PER_ROW, 1),
    )
    val_layout = cute.make_layout((1, ELTS_PER_LOAD))
    tiled_copy = cute.make_tiled_copy_tv(g2s_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tid)

    # ---- 3. gmem -> smem with cp.async. ----
    tQgQ = thr_copy.partition_S(gQ)
    tQsQ = thr_copy.partition_D(sQ)
    cute.copy(tiled_copy, tQgQ, tQsQ)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # ---- 4. smem -> gmem with a plain (universal) copy. Just to verify. ----
    # We re-use the same TiledCopy pattern. The atom would emit cp.async if
    # the source were gmem, but with smem-source it falls back to LDS+ST.
    # For pedagogy that's fine; in real FA2 we'd use ldmatrix here.
    tQsQ_out = thr_copy.partition_S(sQ)
    tQgOut = thr_copy.partition_D(gOut)
    cute.autovec_copy(tQsQ_out, tQgOut)
    # No fence needed; smem reads + gmem writes finish in-order per thread.


@cute.jit
def copy_host(q: cute.Tensor, out: cute.Tensor):
    # Build SmemLayoutQ (the swizzled 64x64 atom) at trace time and pass
    # it as a constexpr argument to the kernel.
    sw = cute.make_swizzle(3, 3, 3)
    inner = cute.make_layout((8, kHeadDim), stride=(kHeadDim, 1))
    atom = cute.make_composed_layout(sw, 0, inner)
    sQ_layout = cute.tile_to_shape(atom, (kBlockM, kHeadDim), order=(0, 1))

    copy_kernel(q, out, sQ_layout).launch(
        grid=(1, 1, 1),
        block=(kNThreads, 1, 1),
    )


def run():
    torch.manual_seed(0)
    Q = torch.randn(kBlockM, kHeadDim, device="cuda", dtype=torch.float16)
    Out = torch.empty_like(Q)

    # 16B alignment is required for cp.async 128-bit loads.
    cQ = cute_rt.from_dlpack(Q, assumed_align=16)
    cOut = cute_rt.from_dlpack(Out, assumed_align=16)

    compiled = cute.compile(copy_host, cQ, cOut)
    compiled(cQ, cOut)
    torch.cuda.synchronize()

    torch.testing.assert_close(Out, Q)
    print(f"tiled cp.async OK   tile={kBlockM}x{kHeadDim}   max|err|={(Out-Q).abs().max().item():.3e}")


if __name__ == "__main__":
    run()

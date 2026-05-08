"""
flash_fwd.py — FA2 forward, Python DSL skeleton.

This is a *skeleton*, not a working kernel. The structure mirrors
cuda/flash_attn_cutlass/flash_fwd_kernel.h section by section so you can
diff the two as you fill in each block.

The ordering follows your CUTE_NOTES.md "design dependency arrow":

    1. Pick the MMA atom            (Stage A: traits)
    2. Pick the smem layouts        (Stage A: traits)
    3. Pick the gmem TiledCopy      (Stage A: traits)
    4. Allocate gmem / smem tensors (Stage B: prologue)
    5. Build per-thread partitions  (Stage B: prologue)
    6. Pipeline loop                (Stage C: main loop)
    7. Epilogue (regs->smem->gmem)  (Stage D: epilogue)

Each TODO marker is paired with a pointer to the matching block in the
C++ kernel.

Usage when complete:
    python cute/flash_fwd.py
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.nvgpu.warp as warp
import cutlass.cute.runtime as cute_rt
import torch
from cutlass.utils import SmemAllocator

# =============================================================================
# Stage A: traits (compile-time constants — Python equivalent of
#          template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
#          struct Flash_fwd_kernel_traits { ... }; )
# =============================================================================

# Tweak these per kernel specialization, same as the C++ traits struct.
kHeadDim = 64
kBlockM = 64
kBlockN = 64
kNWarps = 4
kNThreads = kNWarps * 32

# Bank-conflict avoidance constants (see CUTE_NOTES.md "page split").
kBlockKSmem = 64 if (kHeadDim % 64 == 0) else 32
kSwizzle = 3 if (kBlockKSmem == 64) else 2

# cp.async / vectorization
ELTS_PER_LOAD = 16 // 2  # 16B / sizeof(fp16) = 8
THREADS_PER_ROW = kBlockKSmem // ELTS_PER_LOAD  # = 8 for kBlockKSmem=64


def make_smem_layouts():
    """Build sQ, sK, sV, sVt layouts. Constexpr — call from inside @cute.jit."""
    sw = cute.make_swizzle(kSwizzle, 3, 3)
    inner = cute.make_layout((8, kBlockKSmem), stride=(kBlockKSmem, 1))
    atom = cute.make_composed_layout(sw, 0, inner)
    sQ_layout = cute.tile_to_shape(atom, (kBlockM, kHeadDim), order=(0, 1))
    sKV_layout = cute.tile_to_shape(atom, (kBlockN, kHeadDim), order=(0, 1))
    transpose = cute.make_layout((kHeadDim, kBlockN), stride=(kBlockN, 1))
    sVt_layout = cute.composition(sKV_layout, transpose)
    return sQ_layout, sKV_layout, sVt_layout


def make_gmem_tiled_copy():
    """The cp.async TiledCopy used for Q, K, V loads."""
    op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    atom = cute.make_copy_atom(op, cutlass.Float16, num_bits_per_copy=128)
    thr = cute.make_layout(
        (kNThreads // THREADS_PER_ROW, THREADS_PER_ROW),
        stride=(THREADS_PER_ROW, 1),
    )
    val = cute.make_layout((1, ELTS_PER_LOAD))
    return cute.make_tiled_copy_tv(atom, thr, val)


def make_tiled_mma():
    """SM80 16x8x16 fp16->fp32 MMA, kNWarps split M only (FA2 rule)."""
    op = warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
    return cute.make_tiled_mma(
        op,
        atom_layout_mnk=(kNWarps, 1, 1),
        permutation_mnk=(16 * kNWarps, 16, 16),
    )


# =============================================================================
# Stage B + C + D: the device kernel.
# =============================================================================


@cute.kernel
def flash_fwd_kernel(
    mQ: cute.Tensor,  # (seqlen_q, head_dim) fp16, single batch+head
    mK: cute.Tensor,  # (seqlen_k, head_dim) fp16
    mV: cute.Tensor,  # (seqlen_k, head_dim) fp16
    mO: cute.Tensor,  # (seqlen_q, head_dim) fp16
    softmax_scale_log2: cutlass.Float32,
    sQ_layout: cute.ComposedLayout,
    sKV_layout: cute.ComposedLayout,
    sVt_layout: cute.ComposedLayout,
):
    tid, _, _ = cute.arch.thread_idx()
    m_block, _, _ = cute.arch.block_idx()

    # ---- B.1 gmem tile views (mirrors `local_tile(mQ, ...)`) ----
    # gQ: (kBlockM, kHeadDim)
    # gK, gV: (kBlockN, kHeadDim, n_block_max)  — last mode iterated in main loop
    gQ = cute.local_tile(mQ, (kBlockM, kHeadDim), (m_block, 0))
    gK = cute.local_tile(mK, (kBlockN, kHeadDim), (None, 0))
    gV = cute.local_tile(mV, (kBlockN, kHeadDim), (None, 0))
    gO = cute.local_tile(mO, (kBlockM, kHeadDim), (m_block, 0))

    # ---- B.2 smem allocation ----
    smem = SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float16, sQ_layout, byte_alignment=16)
    sK = smem.allocate_tensor(cutlass.Float16, sKV_layout, byte_alignment=16)
    sV = smem.allocate_tensor(cutlass.Float16, sKV_layout, byte_alignment=16)
    # sVt aliases sV's bytes — same pointer, different (transposed) layout.
    # TODO(eric): build sVt as a view over sV.iterator with sVt_layout.
    # In C++:  Tensor sVt = make_tensor(sV.data(), SmemLayoutVt{});

    # ---- B.3 gmem TiledCopy partitioning ----
    gmem_tiled_copy = make_gmem_tiled_copy()
    gmem_thr_copy = gmem_tiled_copy.get_slice(tid)
    tQgQ = gmem_thr_copy.partition_S(gQ)
    tQsQ = gmem_thr_copy.partition_D(sQ)
    # tKgK, tKsK, tVgV, tVsV — same pattern. Note gK/gV are 3D (last mode is
    # the n_block iteration), so partition_S returns a tensor with that mode
    # preserved; index it as `tKgK[..., n_block]` per loop iteration.
    # TODO(eric): build tKgK, tKsK, tVgV, tVsV.

    # ---- B.4 tiled MMA + register fragments ----
    tiled_mma = make_tiled_mma()
    thr_mma = tiled_mma.get_slice(tid)
    # acc_o is the running output accumulator (M, N=kHeadDim, fp32).
    # In C++: partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{})
    # TODO(eric): allocate acc_o as a register fragment with shape
    #             (MMA, MMA_M, MMA_N_HEADDIM) using tiled_mma.make_fragment_C.

    # ---- B.5 ldmatrix tiled copies for sQ, sK, sV ----
    # SM75 ldmatrix.x4 for A; ldmatrix.x4 for B; ldmatrix.x4.trans for V^T.
    # ld_op_A = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
    # ld_atom_A = cute.make_copy_atom(ld_op_A, cutlass.Float16)
    # smem_tiled_copy_Q = cute.make_tiled_copy_A(ld_atom_A, tiled_mma)
    # ... etc for K and V.
    # TODO(eric): wire the three ldmatrix TiledCopies.

    # ---- C.1 prologue: issue Q + first K loads ----
    # cute.copy(gmem_tiled_copy, tQgQ, tQsQ)
    # cute.copy(gmem_tiled_copy, tKgK[..., 0], tKsK)
    # cute.arch.cp_async_commit_group()
    # TODO(eric): emit prologue cp.asyncs.

    # ---- C.2 main loop ----
    # n_block_max = cute.size(gK, mode=[2])  # last mode of the 3D tile
    # For-loop pattern (use cute.range_dynamic for runtime range):
    #
    #   for n in cute.range_dynamic(0, n_block_max, 1, unroll=1):
    #       cute.arch.cp_async_wait_group(0)
    #       cute.arch.sync_threads()
    #
    #       # issue V[n] load while we GEMM #1
    #       cute.copy(gmem_tiled_copy, tVgV[..., n], tVsV)
    #       cute.arch.cp_async_commit_group()
    #
    #       # GEMM #1: S = Q @ K^T  (acc_s fresh each iter)
    #       acc_s = ...  # make_fragment_C, shape (M, N=kBlockN), fp32
    #       cute.gemm(tiled_mma, acc_s, tCrQ, tCrK, acc_s)
    #
    #       # online softmax: rowmax, rescale acc_o + row_sum, exp2 acc_s
    #       # (mirrors softmax.cuh)
    #
    #       cute.arch.cp_async_wait_group(0)
    #       cute.arch.sync_threads()
    #
    #       # issue K[n+1] (predicated for last iter) while we GEMM #2
    #       if n + 1 < n_block_max:
    #           cute.copy(gmem_tiled_copy, tKgK[..., n + 1], tKsK)
    #           cute.arch.cp_async_commit_group()
    #
    #       # convert acc_s -> P (fp16) via convert_layout_acc_Aregs analog
    #       rP = ...
    #       # GEMM #2: acc_o += P @ V
    #       cute.gemm(tiled_mma, acc_o, rP, tCrV, acc_o)
    #
    # TODO(eric): implement the loop body.

    # ---- D epilogue: normalize, regs->smem->regs->gmem ----
    # 1. acc_o /= row_sum   (final normalization; row_sum was tracked above)
    # 2. acc_o (fp32) -> sO (fp16) using a smem-staging TiledCopy so the gmem
    #    write is coalesced. See CUTE_NOTES.md "Epilogue: regs -> smem -> ...".
    # 3. cute.copy of sO -> gO with the gmem TiledCopy.
    # TODO(eric): implement the epilogue.
    pass


@cute.jit
def flash_fwd_host(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    softmax_scale_log2: cutlass.Float32,
):
    sQ_layout, sKV_layout, sVt_layout = make_smem_layouts()
    seqlen_q = cute.size(Q, mode=[0])
    grid_m = cute.ceil_div(seqlen_q, kBlockM)
    flash_fwd_kernel(
        Q,
        K,
        V,
        O,
        softmax_scale_log2,
        sQ_layout,
        sKV_layout,
        sVt_layout,
    ).launch(
        grid=(grid_m, 1, 1),
        block=(kNThreads, 1, 1),
    )


def run():
    """Smoke-launch the (currently empty) kernel and check that compile +
    launch don't blow up. Once the kernel body is filled in, replace this
    with a numerical check against torch.nn.functional.scaled_dot_product_attention.
    """
    import math

    torch.manual_seed(0)
    seqlen = 128
    Q = torch.randn(seqlen, kHeadDim, device="cuda", dtype=torch.float16)
    K = torch.randn(seqlen, kHeadDim, device="cuda", dtype=torch.float16)
    V = torch.randn(seqlen, kHeadDim, device="cuda", dtype=torch.float16)
    O = torch.zeros_like(Q)

    cQ = cute_rt.from_dlpack(Q, assumed_align=16)
    cK = cute_rt.from_dlpack(K, assumed_align=16)
    cV = cute_rt.from_dlpack(V, assumed_align=16)
    cO = cute_rt.from_dlpack(O, assumed_align=16)

    scale_log2 = (1.0 / math.sqrt(kHeadDim)) * math.log2(math.e)
    compiled = cute.compile(flash_fwd_host, cQ, cK, cV, cO, scale_log2)
    compiled(cQ, cK, cV, cO, scale_log2)
    torch.cuda.synchronize()
    print("flash_fwd kernel compiled and launched (skeleton — no compute yet).")


if __name__ == "__main__":
    run()

"""
03 — A single-tile fp16 GEMM with the SM80 16x8x16 MMA atom.

We compute C = A @ B^T for a single (M, N, K) = (kBlockM, kBlockN, kHeadDim)
tile in one CTA. Same MMA shape FA2 uses for the S = Q @ K^T half.

What we exercise:
  - cute.nvgpu.warp.MmaF16BF16Op(Float16, Float32, (16,8,16))   -> warp atom
  - cute.make_tiled_mma(op, atom_layout_mnk=(W,1,1), permutation_mnk=...)
        equivalent to:
            TiledMMA<MMA_Atom<...>, Layout<Shape<W,_1,_1>>, Tile<16*W,_16,_16>>
  - tiled_mma.get_slice(tid)
  - thr_mma.partition_A / B / C
  - tiled_mma.make_fragment_C(...)              -> register accumulator
  - cute.gemm(tiled_mma, acc, A_frag, B_frag, acc)

Notes / tradeoffs vs the C++ version:
  - We *don't* yet use ldmatrix here. We use cute.autovec_copy from smem,
    which the compiler will lower to LDS.* loads. ldmatrix becomes
    important for performance + the right register layout, and we'll wire
    it in for the FA2 skeleton (`flash_fwd.py`).
  - We *don't* use cp.async to load smem here — just plain copies — so
    this file is short and focused on the MMA piece. Combine with
    02_tiled_copy_g2s.py to see the full pipeline.

Run: python cute/03_tiled_mma.py
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import cutlass.cute.nvgpu.warp as warp
from cutlass.utils import SmemAllocator


kBlockM = 64    # rows per CTA's C tile
kBlockN = 64    # cols per CTA's C tile
kHeadDim = 64   # K (single pass for this example)
kNWarps = 4
kNThreads = kNWarps * 32


@cute.kernel
def gemm_kernel(
    gA: cute.Tensor,            # (M, K) fp16
    gB: cute.Tensor,            # (N, K) fp16  (note: B^T contraction)
    gC: cute.Tensor,            # (M, N) fp32
    sA_layout: cute.Layout,     # smem layouts (constexpr)
    sB_layout: cute.Layout,
):
    tid, _, _ = cute.arch.thread_idx()

    # ---- 1. Stage A and B into smem so threads from a warp share rows. ----
    smem = SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float16, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.Float16, sB_layout, byte_alignment=16)
    cute.autovec_copy(gA, sA)
    cute.autovec_copy(gB, sB)
    cute.arch.sync_threads()

    # ---- 2. Build the tiled MMA. ----
    # Per-atom shape = (16, 8, 16) M,N,K with fp16 inputs and fp32 acc.
    # atom_layout_mnk = (kNWarps, 1, 1)  -> warps split M, NOT N (FA2 rule).
    # permutation_mnk = (16*kNWarps, 16, 16)  -> per-pass extent. Each warp
    # does 1 atom in M (16 rows), 2 atoms in N (since atom_N=8, perm_N=16),
    # 1 atom in K (16). This is the same Tile<16*W, _16, _16> in C++.
    mma_op = warp.MmaF16BF16Op(
        cutlass.Float16, cutlass.Float32, (16, 8, 16)
    )
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        atom_layout_mnk=(kNWarps, 1, 1),
        permutation_mnk=(16 * kNWarps, 16, 16),
    )
    thr_mma = tiled_mma.get_slice(tid)

    # ---- 3. Partition A, B, C. ----
    tCsA = thr_mma.partition_A(sA)         # ((MMA_K, MMA_M), MMA_K_iters?)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC)

    # Register-resident A, B, C fragments (per-thread).
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC = tiled_mma.make_fragment_C(tCgC)

    # ---- 4. Load smem -> regs and run the MMA. ----
    # autovec_copy here will lower to LDS.* per the smem tensor layouts.
    cute.autovec_copy(tCsA, tCrA)
    cute.autovec_copy(tCsB, tCrB)
    tCrC.fill(0.0)

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # ---- 5. Write fp32 acc back to gmem. ----
    cute.autovec_copy(tCrC, tCgC)


@cute.jit
def gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    sA_layout = cute.make_layout((kBlockM, kHeadDim), stride=(kHeadDim, 1))
    sB_layout = cute.make_layout((kBlockN, kHeadDim), stride=(kHeadDim, 1))
    gemm_kernel(a, b, c, sA_layout, sB_layout).launch(
        grid=(1, 1, 1),
        block=(kNThreads, 1, 1),
    )


def run():
    torch.manual_seed(0)
    A = torch.randn(kBlockM, kHeadDim, device="cuda", dtype=torch.float16)
    B = torch.randn(kBlockN, kHeadDim, device="cuda", dtype=torch.float16)
    C = torch.zeros(kBlockM, kBlockN, device="cuda", dtype=torch.float32)

    cA = cute_rt.from_dlpack(A, assumed_align=16)
    cB = cute_rt.from_dlpack(B, assumed_align=16)
    cC = cute_rt.from_dlpack(C, assumed_align=16)

    compiled = cute.compile(gemm_host, cA, cB, cC)
    compiled(cA, cB, cC)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float().T)
    err = (C - ref).abs().max().item()
    print(f"single-tile GEMM   {kBlockM}x{kBlockN}x{kHeadDim}   max|err|={err:.3e}")
    torch.testing.assert_close(C, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run()

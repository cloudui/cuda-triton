# wgmma_gemm_minimal.py
#
# Checkpoint 1: bare wgmma GEMM on Hopper (sm_90), CuTeDSL.
#
# Deliberately NOT included (add these back in later checkpoints):
#   - TMA loads (using a simple cooperative copy into SMEM instead)
#   - Multi-stage / async pipelining (single blocking load -> single wgmma)
#   - Cluster shapes / multicast
#   - TMA-store epilogue (plain store instead)
#   - Persistence, stream-K, L2 swizzle
#
# Goal: get the wgmma fence -> issue -> commit_group -> wait_group sequence
# correct against a torch reference, and confirm via NCU that tensor cores
# are actually being hit. Everything else comes later.
#
# Reference for API shapes: NVIDIA/cutlass
#   examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py
# That file is the "answer key" for the TMA/pipelining/epilogue pieces
# stripped out here -- keep it open in a tab. In particular, if the plain
# copy below produces wrong results, the likely cause is that A/B in SMEM
# don't satisfy wgmma's required swizzle/layout -- go look at how
# `_make_smem_layouts` / `sm90_utils` build the swizzled layout in that
# file and mirror it here, even though you're not using TMA to load into it.

import argparse
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack


class MinimalWgmmaGemm:
    """
    C = A @ B on a single CTA-tile, single warpgroup, single wgmma "stage".
    No batching, no persistence -- one CTA computes one output tile.

    Fixed for simplicity:
      - A, B: Float16, K-major
      - C, acc: Float16 in / Float32 accumulate
      - CTA tile: (128, 128, 64)  (M, N, K)
      - 1 warpgroup (128 threads) -- no cooperative 2-warpgroup split yet
    """

    def __init__(self):
        self.a_dtype = cutlass.Float16
        self.b_dtype = cutlass.Float16
        self.c_dtype = cutlass.Float16
        self.acc_dtype = cutlass.Float32

        self.tile_shape_mnk = (128, 128, 64)
        self.atom_layout_mnk = (1, 1, 1)  # single warpgroup
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 128

        self.tiled_mma = None
        self.a_smem_layout = None
        self.b_smem_layout = None
        self.shared_storage = None

    # ------------------------------------------------------------------
    # Host-side setup: build the tiled_mma + smem layouts once shapes/
    # dtypes are known. This is the part to cross-check against
    # `_setup_attributes` / `_make_smem_layouts` in dense_gemm.py if
    # wgmma silently produces garbage.
    # ------------------------------------------------------------------
    def _setup_attributes(self, a_layout: "utils.LayoutEnum", b_layout: "utils.LayoutEnum"):
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        # NOTE: this is the piece most worth diffing against dense_gemm.py.
        # You need an SMEM layout for A/B that (a) fits your tile shape and
        # (b) has a swizzle mode wgmma can actually read via its SMEM
        # descriptor. Pull the swizzle-selection logic out of
        # `_make_smem_layouts` in the reference file rather than guessing.
        self.a_smem_layout = sm90_utils.make_smem_layout_a(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.a_dtype,
            a_layout,
        )
        self.b_smem_layout = sm90_utils.make_smem_layout_b(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.b_dtype,
            b_layout,
        )

    # ------------------------------------------------------------------
    # Host entry point
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream):
        a_layout = utils.LayoutEnum.from_tensor(a)
        b_layout = utils.LayoutEnum.from_tensor(b)

        self._setup_attributes(a_layout, b_layout)

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout)],
                1024,
            ]

        self.shared_storage = SharedStorage

        m, n, k, _ = a.shape[0], b.shape[0], a.shape[1], 1
        grid = (
            (m + self.tile_shape_mnk[0] - 1) // self.tile_shape_mnk[0],
            (n + self.tile_shape_mnk[1] - 1) // self.tile_shape_mnk[1],
            1,
        )

        self.kernel(a, b, c, self.tiled_mma, self.a_smem_layout, self.b_smem_layout).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        gA = cute.local_tile(mA, (BLOCK_M, BLOCK_K), (pid_m, None))
        gB = cute.local_tile(mA, (BLOCK_M, BLOCK_K), (pid_n, None))
        gC = cute.local_tile(mC, (BLOCK_M, BLOCK_N), (pid_m, pid_n))

        


def run(m=512, n=512, k=512):
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")  # K-major B
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    a_cute = from_dlpack(a)
    b_cute = from_dlpack(b)
    c_cute = from_dlpack(c)

    gemm = MinimalWgmmaGemm()
    stream = cutlass.cuda.default_stream()
    gemm(a_cute, b_cute, c_cute, stream)
    torch.cuda.synchronize()

    ref = a.float() @ b.float().T
    err = (c.float() - ref).abs().max().item()
    print(f"max abs error: {err:.4f}")
    ok = torch.allclose(c.float(), ref, atol=1e-1, rtol=1e-2)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=str, default="512,512,512")
    args = parser.parse_args()
    m, n, k = (int(x) for x in args.mnk.split(","))
    run(m, n, k)
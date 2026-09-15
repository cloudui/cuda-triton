
from torch import storage
import argparse
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync, warp


@cute.jit
def launch_wgmma_matmul(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tile_shape_mnk = (128, 128, 64)
    BLOCK_M, BLOCK_N, BLOCK_K = tile_shape_mnk

    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        cutlass.utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.utils.LayoutEnum.COL_MAJOR.sm90_mma_major_mode(),
        cutlass.Float32,
        (1, 1, 1),
        tiler_mn=(64, 128),
    )

    a_smem_layout_staged = sm90_utils.make_smem_layout_a(
        tiled_mma, tile_shape_mnk, cutlass.Float16, ab_stage=1,
    )
    b_smem_layout_staged = sm90_utils.make_smem_layout_b(
        tiled_mma, tile_shape_mnk, cutlass.Float16, ab_stage=1,
    )

    @cute.struct
    class SharedStorage:
        sA: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float16, cute.cosize(a_smem_layout_staged)], 1024,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float16, cute.cosize(b_smem_layout_staged)], 1024,
        ]

    smem_size = SharedStorage.size_in_bytes()   # <-- host needs this for launch

    grid = (cute.ceil_div(mC.shape[0], BLOCK_M), cute.ceil_div(mC.shape[1], BLOCK_N), 1)

    wgmma_matmul_kernel(
        mA, mB, mC, tiled_mma,
        a_smem_layout_staged, b_smem_layout_staged, SharedStorage,
    ).launch(grid=grid, block=[128, 1, 1], smem=smem_size)


@cute.kernel
def wgmma_matmul_kernel(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    a_smem_layout_staged, b_smem_layout_staged,
    SharedStorage,
):
    tid, _, _ = cute.arch.thread_idx()
    pid_m, pid_n, _ = cute.arch.block_idx()

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64

    gA = cute.local_tile(mA, (BLOCK_M, BLOCK_K), (pid_m, None))
    gB = cute.local_tile(mB, (BLOCK_N, BLOCK_K), (pid_n, None))
    gC = cute.local_tile(mC, (BLOCK_M, BLOCK_N), (pid_m, pid_n))

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

    atom_async_copy = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float16,
        num_bits_per_copy=128,
    )
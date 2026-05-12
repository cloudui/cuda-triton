import cutlass
import cutlass.cute as cute


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr[bool] = False,
    B_in_regs: cutlass.Constexpr[bool] = False,
):
    tXrA = smem_thr_copy_A.retile(tCrA)
    tXrB = smem_thr_copy_A.retile(tCrB)

    # initial fragment fetch
    if cutlass.const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tXrA[None, None, 0])
    if cutlass.const_expr(not B_in_regs):
        cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tXrB[None, None, 0])

    for i in cutlass.range_constexpr(tCrA.shape[2]):
        if i < tCrA.shape[2] - 1:
            # next frag fetch until k-1
            if cutlass.const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A, tCsA[None, None, i + 1], tXrA[None, None, i + 1]
                )
            if cutlass.const_expr(not B_in_regs):
                cute.copy(
                    smem_thr_copy_B, tCsB[None, None, i + 1], tXrB[None, None, i + 1]
                )
        cute.gemm(tiled_mma, acc, tCrA[None, None, i], tCrB[None, None, i], acc)


@cute.jit
def gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
):
    tXrB = smem_thr_copy_A.retile(tCrB)

    # initial fragment fetch
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tXrB[None, None, 0])
    for i in cutlass.range_constexpr(tCrA.shape[2]):
        if i < tCrA.shape[2] - 1:
            cute.copy(smem_thr_copy_B, tCsB[None, None, i + 1], tXrB[None, None, i + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, i], tCrB[None, None, i], acc)

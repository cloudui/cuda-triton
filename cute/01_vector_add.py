"""
01 — Vector add (the "hello world" of CuTe DSL).

End-to-end pattern you'll reuse in every later example:

    @cute.kernel  device function (runs on GPU)
    @cute.jit     host function (builds + launches kernels)
    cute.compile  trace + lower the host fn to MLIR -> CUBIN once
    callable(...) actual launch with concrete tensors

Torch tensors are passed in via DLPack. `cute.runtime.from_dlpack(t)` wraps
a torch.Tensor as a `cute.Tensor` with a layout derived from torch strides.

We compute `c[i] = a[i] + b[i]` for fp16 vectors of length N, with one
element per thread. (Simplest possible — no tiling, no smem.)

Run: python cute/01_vector_add.py
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt


@cute.kernel
def vec_add_kernel(
    gA: cute.Tensor,   # (N,)
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    # CTA + thread coordinates. `cute.arch.thread_idx()` returns a 3-tuple,
    # same as threadIdx in CUDA.
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    i = bx * bdim + tx
    # Plain Python `if` over a runtime value compiles to scf.if in MLIR.
    if i < cute.size(gA):
        gC[i] = gA[i] + gB[i]


@cute.jit
def vec_add_host(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    # All three are 1D fp16 cute.Tensors. Pick a block size and derive
    # grid from the runtime size.
    BLOCK = 256
    N = cute.size(a)
    grid = (cute.ceil_div(N, BLOCK), 1, 1)
    vec_add_kernel(a, b, c).launch(grid=grid, block=(BLOCK, 1, 1))


def run():
    torch.manual_seed(0)
    N = 4096
    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    c = torch.empty(N, device="cuda", dtype=torch.float16)

    # Wrap as CuTe tensors — view, not copy.
    ca = cute_rt.from_dlpack(a)
    cb = cute_rt.from_dlpack(b)
    cc = cute_rt.from_dlpack(c)

    # Compile once (cached on the function + arg signatures).
    compiled = cute.compile(vec_add_host, ca, cb, cc)
    compiled(ca, cb, cc)
    torch.cuda.synchronize()

    ref = a + b
    torch.testing.assert_close(c, ref, atol=1e-3, rtol=1e-3)
    print(f"vec_add OK   N={N}   max|err|={(c - ref).abs().max().item():.3e}")


if __name__ == "__main__":
    run()

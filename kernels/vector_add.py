"""
Vector Add — PyTorch reference + Triton kernel

out = X + Y, elementwise. The simplest possible kernel: one load per input,
one store, no reduction. Good baseline for comparing launch/memory overhead
against the fancier kernels in this repo.

Autotuned per the standard Triton pattern (see the official matmul tutorial):
a `triton.autotune` decorator wraps the `@triton.jit` kernel with a list of
candidate `triton.Config`s (BLOCK_SIZE x num_warps). On the first call for a
given `key` (here, `n_elements`), Triton benchmarks every config against the
real inputs and caches the fastest one; subsequent calls with the same key
reuse that cached config at zero extra cost. Because BLOCK_SIZE is now chosen
at autotune time rather than fixed up front, the launch grid must be computed
from a `meta` dict (`grid = lambda meta: (...)`) instead of a plain tuple, so
it stays correct for whichever config gets picked.
"""

import torch
import triton
import triton.language as tl


def vector_add_pytorch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return X + Y


def _autotune_configs():
    configs = []
    for block_size in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        for num_warps in [1, 2, 4, 8, 16]:
            configs.append(triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=_autotune_configs(),
    key=["n_elements"],
)
@triton.jit
def vector_add_kernel(
    X,
    Y,
    Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    out = x + y

    tl.store(Out + offsets, out, mask=mask)


def vector_add(X: torch.Tensor, Y: torch.Tensor):
    out = torch.empty_like(X)

    n_elements = X.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](X, Y, out, n_elements)

    return out

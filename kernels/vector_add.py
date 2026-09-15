"""
Vector Add — PyTorch reference + Triton kernel

out = X + Y, elementwise. The simplest possible kernel: one load per input,
one store, no reduction. Good baseline for comparing launch/memory overhead
against the fancier kernels in this repo.
"""

import torch
import triton
import triton.language as tl


def vector_add_pytorch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return X + Y


@triton.jit
def vector_add_kernel(
    X, 
    Y, 
    Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr
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

    BLOCK_SIZE = 1024

    n_elements = X.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    vector_add_kernel[grid](X, Y, out, n_elements, BLOCK_SIZE)

    return out

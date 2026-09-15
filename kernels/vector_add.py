
import torch
import triton
import triton.language as tl

@triton.jit
@triton.jit
def vector_add_kernel(
    A, 
    B, 
    Out,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)

    out = a + b

    tl.store(Out + offsets, out, mask=mask)


def vector_add(A: torch.Tensor, B: torch.Tensor):
    out = torch.empty_like(A)

    BLOCK_SIZE = 1024

    n_elements = A.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), )
    vector_add_kernel[grid](A, B, out, n_elements, BLOCK_SIZE)

    return out
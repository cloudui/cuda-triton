# Scratch

You can't write CuTe without testing. These are the experiments I built up while writing the blog post -- each file is keyed to one or more sections of my blog post (to be linked soon) so you can run the demo for the specific concept you're stuck on.

https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/

## Usage

```sh
# from repo root
make run-cuda fn=scratch/${FILE}
```

## CuTe Basics

| File | Blog section |
|---|---|
| [`01_layouts.cu`](01_layouts.cu) | Layouts, Shapes, and Strides |
| [`02_tensor.cu`](02_tensor.cu) | Tensors |
| [`tensor.py`](tensor.py) | Layout Hell (Python toy, WIP) |

## MMA + Tiled Copy

| File | Blog section |
|---|---|
| [`03_mma.cu`](03_mma.cu) | Tiled MMA -- single tensor-core instruction, no SMEM |
| [`04_mma.cu`](04_mma.cu) | Tiled MMA + Tiled Copy A,B,C -- full gmem->smem->regs->MMA pipeline, verified |
| [`05_gemm.cu`](05_gemm.cu) | MMA Loop: QK^T GEMM -- standalone tiled GEMM kernel |
| [`retile_viz.cu`](retile_viz.cu) | Partition vs. Retile -- visualize how `retile_D`/`retile_S` rebind register tensor layouts |

## Swizzling

| File | Blog section |
|---|---|
| [`swizzle_sim.py`](swizzle_sim.py) | Swizzling -- pure-Python `Swizzle<B,M,S>` simulator, toy with the bit math |
| [`bench_swizzle_writes.cu`](bench_swizzle_writes.cu) | Swizzling FA2 -- benchmark cp.async / STS.128 with vs. without `Swizzle<3,3,3>` |
| [`swizzle_layouts.cu`](swizzle_layouts.cu) | Swizzling FA2 + sVtNoSwizzle -- print the FA2 SMEM layouts (Q/K, Vt, VtNoSwizzle) for hdim=32 and hdim=64 |
| [`v_fragment_test.cu`](v_fragment_test.cu) | sVtNoSwizzle: The No-Op Nobody Caught -- minimal repro that swapping `sVt` for `sVtNoSwizzle` doesn't break anything |

## Softmax

| File | Blog section |
|---|---|
| [`fragment_reshape.cu`](fragment_reshape.cu) | Fragment Reshape -- `convert_layout_rowcol` demo |
| [`convert_acc_to_a.cu`](convert_acc_to_a.cu) | (extra) `convert_layout_acc_Aregs` -- rebind C-layout to A-layout for the SV GEMM |
| [`test_allreduce.cu`](test_allreduce.cu) | Warp Reduce -- standalone test for `FLASH::Allreduce<N>::run` |
| [`test_softmax.cu`](test_softmax.cu) | Online Softmax -- end-to-end test for `Softmax::softmax_rescale_o` against a CPU reference |

## End-to-end

| File | Blog section |
|---|---|
| [`fa_test.cu`](fa_test.cu) | Wrapping Up -- runnable end-to-end test for the full `flash_fwd_kernel` against a CPU reference |
| [`bench.py`](bench.py), [`bench_matmul.py`](bench_matmul.py) | Wrapping Up -- benchmarks |

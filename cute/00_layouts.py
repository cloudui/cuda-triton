"""
00 — Layouts and swizzle.

Everything here happens at *trace time* (when `cute.compile` walks the
@cute.jit body) — no kernel is launched. We just build layouts and `print`
them so you can see what CuTe constructs, the same way you'd `cute::print`
in a small C++ test program.

Why everything is wrapped in `@cute.jit`: layout creation needs an MLIR
context. The DSL provides that context only inside a jit-decorated
function. `print(...)` runs at trace time, so the layouts show up exactly
once when `cute.compile(...)` is called.

This file mirrors `kernel_traits.cuh` from cuda/flash_attn_cutlass:

    SmemLayoutAtomQ = composition(Swizzle<3,3,3>{},
                                   Layout<Shape<_8, kBlockKSmem>,
                                          Stride<kBlockKSmem, _1>>{});
    SmemLayoutQ     = tile_to_shape(SmemLayoutAtomQ, Shape<kBlockM, kHeadDim>{});
    SmemLayoutKV    = tile_to_shape(SmemLayoutAtomQ, Shape<kBlockN, kHeadDim>{});
    SmemLayoutVt    = composition(SmemLayoutKV, Layout<Shape<kHeadDim, kBlockN>, GenRowMajor>);

Run: python cute/00_layouts.py
"""

import cutlass.cute as cute


@cute.jit
def show_layouts():
    # ---- 1. plain (8, 64) row-major layout ----
    L = cute.make_layout((8, 64), stride=(64, 1))
    print("\n[1] plain layout")
    print("    L =", L)
    print("    size  =", cute.size(L))     # 512
    print("    cosize=", cute.cosize(L))   # 512

    # ---- 2. swizzled atom: Swizzle<3,3,3> over (8, 64) row-major ----
    # In C++:
    #   SmemLayoutAtomQ = composition(Swizzle<3,3,3>{},
    #                                 Layout<Shape<_8,_64>, Stride<_64,_1>>{});
    kBlockKSmem = 64
    sw = cute.make_swizzle(3, 3, 3)              # Swizzle<B=3, M=3, S=3>
    inner = cute.make_layout((8, kBlockKSmem), stride=(kBlockKSmem, 1))
    atom = cute.make_composed_layout(sw, 0, inner)
    print("\n[2] SmemLayoutAtomQ (Swizzle<3,3,3> ∘ (8,64):(64,1))")
    print("    atom =", atom)
    # Read the printed form: `S<3,3,3> o 0 o (8,64):(64,1)`. The "o 0" is
    # the offset (0 means: no extra base offset). The XOR happens via the
    # Swizzle prefix during address computation.

    # ---- 3. tile_to_shape: tile the atom to cover (kBlockM, kHeadDim) ----
    # In C++:
    #   SmemLayoutQ = tile_to_shape(SmemLayoutAtomQ{},
    #                               Shape<Int<kBlockM>, Int<kHeadDim>>{});
    kBlockM, kHeadDim = 64, 64
    SmemLayoutQ = cute.tile_to_shape(atom, (kBlockM, kHeadDim), order=(0, 1))
    print("\n[3] SmemLayoutQ (tile (8,64) atom up to (64, 64))")
    print("    Q =", SmemLayoutQ)

    # ---- 4. SmemLayoutKV — same atom, different M extent ----
    kBlockN = 64
    SmemLayoutKV = cute.tile_to_shape(atom, (kBlockN, kHeadDim), order=(0, 1))
    print("\n[4] SmemLayoutKV")
    print("    KV =", SmemLayoutKV)

    # ---- 5. SmemLayoutVt — transposed view of KV (same memory) ----
    # In C++:
    #   SmemLayoutVt = composition(SmemLayoutKV{},
    #                              make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
    #                                          GenRowMajor{}));
    transpose = cute.make_layout((kHeadDim, kBlockN), stride=(kBlockN, 1))
    SmemLayoutVt = cute.composition(SmemLayoutKV, transpose)
    SmemLayoutVtNoSwizzle = cute.get_nonswizzle_portion(SmemLayoutVt)
    print("\n[5] SmemLayoutVt (transposed K/V view)")
    print("    Vt           =", SmemLayoutVt)
    print("    Vt no-swizzle=", SmemLayoutVtNoSwizzle)

    # ---- 6. The gmem TiledCopy thread layout ----
    # In C++:
    #   GmemLayout = Layout<Shape<kNThreads/kThreadsPerRow, kThreadsPerRow>,
    #                       Stride<kThreadsPerRow, _1>>;
    #   GmemTiledCopyQKV = make_tiled_copy(
    #       GmemCopyAtom{}, GmemLayout{}, Layout<Shape<_1,_8>>{});
    kNThreads = 128
    elements_per_load = 16 // 2          # cp.async vector = 16B = 8 fp16
    kThreadsPerRow = kBlockKSmem // elements_per_load  # = 8
    thr = cute.make_layout(
        (kNThreads // kThreadsPerRow, kThreadsPerRow),
        stride=(kThreadsPerRow, 1),
    )
    val = cute.make_layout((1, elements_per_load))
    print("\n[6] gmem TiledCopy decomposition")
    print("    threads =", thr, "    (16 rows × 8 cols)")
    print("    values  =", val, "    (1 × 8 per thread)")
    # In a kernel we'd then build the TiledCopy as:
    #   atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(...),
    #                              cutlass.Float16, num_bits_per_copy=128)
    #   tiled_copy = cute.make_tiled_copy_tv(atom, thr, val)


if __name__ == "__main__":
    cute.compile(show_layouts)

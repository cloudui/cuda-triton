// scratch/retile_viz.cu
//
// Visualizes how retile_D and retile_S transform register tensor layouts
// during SMEM <-> register copies in CuTe (mirroring flash_attn_cutlass).
//
//   retile_D: registers are the DESTINATION  (SMEM → regs, e.g. loading Q/K/V)
//   retile_S: registers are the SOURCE       (regs → SMEM, e.g. writing output O)
//
// Both are zero-cost layout-only operations: same physical registers, different
// CuTe shape/stride view. The retiling makes cute::copy() emit the right number
// and grouping of ldmatrix / lds / sts instructions.
//
// Compile (from repo root):
//   nvcc -I third_party/cutlass/include -std=c++17 -arch=sm_80 \
//        --expt-relaxed-constexpr --expt-extended-lambda        \
//        scratch/retile_viz.cu -o scratch/retile_viz && ./scratch/retile_viz

#include <cstdio>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// ─── types (mirror flash_attn_cutlass/kernel_traits.cuh) ──────────────────

// 4-warp TiledMMA matching FA's kNWarps=4, kBlockM=64, kBlockN=8.
// Using kHeadDim=16 (small) so 2 K-tiles fit (32 cols / 16 per tile = 2).
using TiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    Layout<Shape<_4, _1, _1>>,  // 4 warps along M
    Tile<_64, _8, _16>>;        // MMA tile: M=64 N=8 K=16

// SMEM A: (kBlockM=64) × (kHeadDim=32), row-major — 2 K-tiles
// Real FA uses Swizzle<3,3,3> to eliminate bank conflicts; the retile logic
// is identical either way, so we omit the swizzle here for readability.
using SmemLayoutA = Layout<Shape<_64, _32>, Stride<_32, _1>>;

// SMEM C: (kBlockM=64) × (kBlockN=8), row-major — output O tile
using SmemLayoutC = Layout<Shape<_64, _8>, Stride<_8, _1>>;

// Copy atoms (same as FA kernel)
// SMEM → regs: LDSM.SYNC.M88.4 (non-transposed) — 4×u32 = 8 fp16 per thread
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
// regs → SMEM: 128-bit vectorized store — AutoVectorizing picks the right width
using SmemCopyAtomC = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half_t>;

// ─── helpers ──────────────────────────────────────────────────────────────

// Print flat register values for threads [0, nthr).
// All threads must call this (syncthreads inside).
template <class Tensor>
__device__ void print_regs(const char* label, Tensor const& frag, int tid, int nthr = 8) {
    for (int thr = 0; thr < nthr; thr++) {
        if (tid == thr) {
            printf("  thr%3d | %-22s | %2d vals: [", tid, label, (int)size(frag));
            for (int i = 0; i < size(frag); i++)
                printf("%6.0f%s", float(frag(i)), i + 1 < size(frag) ? "," : "");
            printf("]\n");
        }
        __syncthreads();
    }
}

// ─── kernel 1: retile_D  (SMEM → registers) ───────────────────────────────
//
// Setup:  SMEM A = 64×32 half_t, filled with linear offsets.
//         TiledMMA partitions give tCrA with shape (MMA_VAL, MMA_M, MMA_K).
//         make_tiled_copy_A + partition_S gives tCsA  (copy's view of SMEM).
//         retile_D reshapes tCrA → tXrA so cute::copy() knows how many
//         LDSM atoms to issue and which registers to fill.
//         tCrA and tXrA alias the same registers — cute::gemm() later uses tCrA.

__global__ void viz_retile_D() {
    int tid = threadIdx.x;

    // ── init SMEM ─────────────────────────────────────────────────────────
    __shared__ half_t smem_A[64 * 32];
    for (int i = tid; i < 64 * 32; i += blockDim.x)
        smem_A[i] = half_t(i);  // linear SMEM offset as value
    __syncthreads();

    Tensor sA = make_tensor(make_smem_ptr(smem_A), SmemLayoutA{});

    // ── MMA partitioning ──────────────────────────────────────────────────
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // tCrA: register fragment shaped by TiledMMA.  This is what cute::gemm() sees.
    Tensor tCrA = thr_mma.partition_fragment_A(sA);  // (MMA_VAL, MMA_M, MMA_K)

    // ── copy setup ────────────────────────────────────────────────────────
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(tid);

    // tCsA: per-thread view of sA for the copy (source side)
    Tensor tCsA = smem_thr_copy_A.partition_S(sA);   // (CPY_VAL, CPY_M, CPY_K)

    // retile_D: same physical registers as tCrA, regrouped so dim-0 matches
    // the copy atom's vector width (8 fp16 for LDSM.M88.4).
    // cute::copy() iterates over (CPY_M, CPY_K) and issues one LDSM per step.
    Tensor tXrA = smem_thr_copy_A.retile_D(tCrA);    // (CPY_VAL, CPY_M, CPY_K)

    // ── layout printout (thread 0 only) ───────────────────────────────────
    if (tid == 0) {
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║           retile_D: SMEM → Registers  (loading A/Q/K)   ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        printf("sA layout (64×32 half_t, row-major):\n  ");
        print(sA.layout());
        printf("\n\n");

        printf("tCrA  — MMA fragment A (what cute::gemm() consumes):\n  ");
        print(tCrA.layout());
        printf("\n");
        printf("  rank-0 MMA_VAL = %d  (fp16 values per thread per MMA tile)\n",
               (int)size<0>(tCrA));
        printf("  rank-1 MMA_M   = %d  (M-tiles; 64M / 64-per-TiledMMA = 1)\n",
               (int)size<1>(tCrA));
        printf("  rank-2 MMA_K   = %d  (K-tiles; 32K / 16-per-MMA = 2)\n\n",
               (int)size<2>(tCrA));

        printf("tCsA  — SMEM partition_S (copy's source view):\n  ");
        print(tCsA.layout());
        printf("\n\n");

        printf("tXrA = retile_D(tCrA)  — copy's destination view:\n  ");
        print(tXrA.layout());
        printf("\n");
        printf("  rank-0 CPY_VAL = %d  (fp16 per LDSM atom = 4×u32 = 8 fp16)\n",
               (int)size<0>(tXrA));
        printf("  rank-1 CPY_M   = %d\n", (int)size<1>(tXrA));
        printf("  rank-2 CPY_K   = %d\n\n", (int)size<2>(tXrA));

        printf("size(tCrA) = %d   size(tXrA) = %d   (must be equal — same registers)\n",
               (int)size(tCrA), (int)size(tXrA));
        printf("ptr(tCrA)  = %p\n", (void*)tCrA.data());
        printf("ptr(tXrA)  = %p   (same address → zero-cost layout alias)\n\n",
               (void*)tXrA.data());

        printf("─────────────────────────────────────────────────────────────\n");
        printf("copy() loop: iterates over CPY_M=%d × CPY_K=%d = %d LDSM atoms\n",
               (int)size<1>(tXrA), (int)size<2>(tXrA),
               (int)size<1>(tXrA) * (int)size<2>(tXrA));
        printf("─────────────────────────────────────────────────────────────\n\n");
        printf("Per-thread registers after copy (first 8 threads):\n");
    }
    __syncthreads();

    // Perform SMEM → register copy.  All 128 threads must call — LDSM is warp-sync.
    cute::copy(smem_tiled_copy_A, tCsA, tXrA);

    // tCrA and tXrA alias the same storage — reading tCrA shows what LDSM loaded.
    print_regs("tCrA[:,K=0]", tCrA(_, _, _0{}), tid, 8);
    if (tid == 0) printf("  (K-tile 0)\n");
    __syncthreads();
    print_regs("tCrA[:,K=1]", tCrA(_, _, _1{}), tid, 8);
    if (tid == 0) printf("  (K-tile 1)\n\n");
}

// ─── kernel 2: retile_S  (registers → SMEM) ───────────────────────────────
//
// Setup:  Simulate the O epilogue in FA: fp32 accumulator → fp16 → SMEM.
//         acc_o is partitioned by TiledMMA's C layout: (MMA_VAL, MMA_M, MMA_N).
//         o_fp16 has the same layout after fp32→fp16 conversion.
//         retile_S reshapes o_fp16 → trC so cute::copy() groups values to
//         match the copy atom's store width (AutoVectorizing picks 64 or 128 bit).
//         o_fp16 and trC alias the same registers — the STS instruction sees trC.

__global__ void viz_retile_S() {
    int tid = threadIdx.x;

    // ── init SMEM output buffer ────────────────────────────────────────────
    __shared__ half_t smem_C[64 * 8];
    for (int i = tid; i < 64 * 8; i += blockDim.x)
        smem_C[i] = half_t(0);
    __syncthreads();

    // ── simulate MMA accumulator ──────────────────────────────────────────
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // partition_fragment_C needs a tensor whose shape defines the tile extent
    Tensor sC_shape = make_tensor(make_smem_ptr(smem_C), SmemLayoutC{});
    Tensor acc_o    = thr_mma.partition_fragment_C(sC_shape); // (MMA_VAL, MMA_M, MMA_N) fp32

    // Fill accumulator: thread t, slot i → t*100 + i  (readable in SMEM dump)
    for (int i = 0; i < size(acc_o); i++)
        acc_o(i) = float(tid * 100 + i);

    // fp32 → fp16 conversion (FA does this via convert_type before the STS)
    Tensor o_fp16 = make_tensor<half_t>(acc_o.layout());
    for (int i = 0; i < size(acc_o); i++)
        o_fp16(i) = half_t(acc_o(i));

    // ── copy setup ────────────────────────────────────────────────────────
    auto smem_tiled_copy_C = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
    auto smem_thr_copy_C   = smem_tiled_copy_C.get_thread_slice(tid);

    Tensor sC  = make_tensor(make_smem_ptr(smem_C), SmemLayoutC{});
    Tensor tsC = smem_thr_copy_C.partition_D(sC);        // SMEM destination

    // retile_S: same registers as o_fp16, regrouped so dim-0 matches the
    // copy atom's store width.  cute::copy() iterates over (CPY_M, CPY_N)
    // and issues one STS per step.
    Tensor trC = smem_thr_copy_C.retile_S(o_fp16);       // (CPY_VAL, CPY_M, CPY_N)

    // ── layout printout (thread 0 only) ───────────────────────────────────
    if (tid == 0) {
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║           retile_S: Registers → SMEM  (writing O)       ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        printf("sC layout (64×8 half_t, row-major):\n  ");
        print(sC.layout());
        printf("\n\n");

        printf("acc_o — MMA C fragment (fp32 accumulator, what gemm() writes):\n  ");
        print(acc_o.layout());
        printf("\n");
        printf("  rank-0 MMA_VAL = %d  (fp32 values per thread per MMA tile)\n",
               (int)size<0>(acc_o));
        printf("  rank-1 MMA_M   = %d  (M-tiles)\n", (int)size<1>(acc_o));
        printf("  rank-2 MMA_N   = %d  (N-tiles)\n\n", (int)size<2>(acc_o));

        printf("o_fp16 — same layout, half_t:\n  ");
        print(o_fp16.layout());
        printf("\n\n");

        printf("tsC — SMEM partition_D (copy's destination view):\n  ");
        print(tsC.layout());
        printf("\n\n");

        printf("trC = retile_S(o_fp16) — copy's source view:\n  ");
        print(trC.layout());
        printf("\n");
        printf("  rank-0 CPY_VAL = %d  (fp16 per STS atom)\n", (int)size<0>(trC));
        printf("  rank-1 CPY_M   = %d\n", (int)size<1>(trC));
        printf("  rank-2 CPY_N   = %d\n\n", (int)size<2>(trC));

        printf("size(o_fp16) = %d   size(trC) = %d   (same registers)\n",
               (int)size(o_fp16), (int)size(trC));
        printf("ptr(o_fp16)  = %p\n", (void*)o_fp16.data());
        printf("ptr(trC)     = %p   (same address → zero-cost layout alias)\n\n",
               (void*)trC.data());

        printf("─────────────────────────────────────────────────────────────\n");
        printf("Register values before copy (first 8 threads):\n");
    }
    __syncthreads();

    print_regs("o_fp16", o_fp16, tid, 8);
    __syncthreads();

    // Perform register → SMEM copy
    cute::copy(smem_tiled_copy_C, trC, tsC);
    __syncthreads();

    // Print SMEM: value encodes (orig_thread * 100 + orig_slot) so you can
    // trace exactly which register from which thread landed at each SMEM cell.
    if (tid == 0) {
        printf("\nSMEM after retile_S + copy  (value = thread*100 + reg_slot):\n");
        for (int r = 0; r < 64; r++) {
            printf("  row%2d: ", r);
            for (int c = 0; c < 8; c++)
                printf("%6.0f", float(sC(r, c)));
            printf("\n");
        }
    }
}

// ─── main ─────────────────────────────────────────────────────────────────

int main() {
    printf("\n");
    viz_retile_D<<<1, 128>>>();
    cudaDeviceSynchronize();

    printf("\n");
    viz_retile_S<<<1, 128>>>();
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("\nCUDA error: %s\n", cudaGetErrorString(err));

    return 0;
}

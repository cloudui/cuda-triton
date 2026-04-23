/*
 * flash.h — Core data structures for FlashAttention-2 CUDA implementation.
 *
 * This is the central header that defines Flash_fwd_params — the parameter
 * struct passed from the PyTorch C++ API to the CUDA kernel. Contains all
 * pointers, dimensions, and strides needed to compute attention.
 *
 * Mirrors Tri Dao's flash.h but scoped to forward pass only.
 *
 * What you need to implement:
 *   - Flash_fwd_params struct with all the fields the kernel needs
 *   - Helper to compute the softmax scale (1/sqrt(d) or 1/sqrt(d) * log2(e))
 *   - Any runtime config (e.g., whether to use the split-KV codepath)
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

struct Flash_fwd_params {
    // Input pointers (device memory)
    // Q, K, V are (batch, seqlen, num_heads, head_dim) — NOTE: Tri Dao uses
    // this layout (not batch, heads, seq, dim) because it's more natural for
    // the strided access pattern. Each head is a contiguous (seqlen, head_dim)
    // block offset by head strides.
    const void *__restrict__ q_ptr;
    const void *__restrict__ k_ptr;
    const void *__restrict__ v_ptr;

    // Output pointer
    void *__restrict__ o_ptr;

    // Softmax log-sum-exp (LSE) — stored per (batch, head, seqlen_q) for
    // backward pass. Even if you only do forward, storing LSE is cheap and
    // lets you verify numerical stability.
    // Shape: (batch, num_heads, seqlen_q)
    float *__restrict__ softmax_lse_ptr;

    // Dimensions
    int batch_size;
    int seqlen_q;
    int seqlen_k;
    int num_heads;
    int num_heads_k;   // for MQA/GQA: num_heads_k <= num_heads
    int head_dim;

    // Strides for Q (in elements, not bytes)
    // Layout: (batch, seqlen, num_heads, head_dim)
    int q_batch_stride;
    int q_row_stride;     // stride between sequence positions
    int q_head_stride;
    // head_dim stride is 1 (contiguous innermost dim)

    // Strides for K
    int k_batch_stride;
    int k_row_stride;
    int k_head_stride;

    // Strides for V
    int v_batch_stride;
    int v_row_stride;
    int v_head_stride;

    // Strides for O (same layout as Q)
    int o_batch_stride;
    int o_row_stride;
    int o_head_stride;

    // Softmax scale = 1/sqrt(head_dim), pre-multiplied by log2(e) so we can
    // use exp2() instead of exp() in the kernel (exp2 is a single PTX
    // instruction on NVIDIA GPUs, exp is not).
    float scale_softmax;
    float scale_softmax_log2;

    // Causal masking
    bool is_causal;

    // TODO: Add these if you extend beyond the basics:
    // - Dropout probability and RNG state
    // - Alibi slopes for position-dependent attention bias
    // - Paged KV cache pointers (for inference)
    // - Window size for sliding window attention
    // - Split-KV parameters (num_splits, workspace pointer)
};

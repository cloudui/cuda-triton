"""
Correctness tests — run these before benchmarking.

  python -m pytest tests/test_kernels.py -v
"""

import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "cuda/flash_attn")
sys.path.insert(0, "cuda/flash_attn_cutlass")

try:
    import cuda_kernels
except ImportError as e:
    print(e)
    cuda_kernels = None

try:
    import flash_attn_cuda
except ImportError as e:
    print(e)
    flash_attn_cuda = None

try:
    import flash_attn_cutlass
except ImportError as e:
    print(e)
    flash_attn_cutlass = None
from kernels.attention import attention_native, attention_pytorch, attention_triton
from kernels.flash_attention import (
    flash_attention_naive,
    flash_attention_pytorch,
    flash_attention_triton,
)
from kernels.flash_attention_full import (
    flash_attention_full_naive,
    flash_attention_full_native,
    flash_attention_full_triton,
)
from kernels.quantized_matmul import (
    dequantize_int4,
    dequantize_int8,
    matmul_fp16,
    matmul_fp16_triton,
    matmul_int4_pytorch,
    matmul_int4_triton,
    matmul_int8_pytorch,
    matmul_int8_triton,
    quantize_int4,
    quantize_int8,
)
from kernels.rmsnorm import rmsnorm_native, rmsnorm_pytorch, rmsnorm_triton
from kernels.softmax import softmax_native, softmax_pytorch, softmax_triton
from kernels.swiglu import swiglu_native, swiglu_pytorch, swiglu_triton


class TestRMSNorm:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        self.weight = torch.randn(2048, device="cuda", dtype=torch.float16)

    def test_pytorch_rmsnorm_shape(self):
        out = rmsnorm_pytorch(self.x, self.weight)
        assert out.shape == self.x.shape

    def test_pytorch_rmsnorm_not_identity(self):
        out = rmsnorm_pytorch(self.x, self.weight)
        assert not torch.allclose(out, self.x)

    def test_pytorch_matches_native(self):
        ref = rmsnorm_native(self.x, self.weight)
        out = rmsnorm_pytorch(self.x, self.weight)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = rmsnorm_pytorch(self.x, self.weight)
        out = rmsnorm_triton(self.x, self.weight)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestSwiGLU:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        self.gate = torch.randn(32, 2048, device="cuda", dtype=torch.float16)

    def test_pytorch_swiglu_shape(self):
        out = swiglu_pytorch(self.x, self.gate)
        assert out.shape == self.x.shape

    def test_pytorch_matches_native(self):
        ref = swiglu_native(self.x, self.gate)
        out = swiglu_pytorch(self.x, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = swiglu_pytorch(self.x, self.gate)
        out = swiglu_triton(self.x, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestFused:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        self.gate = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        self.weight = torch.randn(2048, device="cuda", dtype=torch.float16)

    def test_fused_matches_separate(self):
        from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton

        normed = rmsnorm_pytorch(self.x, self.weight)
        ref = swiglu_pytorch(normed, self.gate)
        out = fused_rmsnorm_swiglu_triton(self.x, self.gate, self.weight)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestSoftmax:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 2048, device="cuda", dtype=torch.float16)

    def test_pytorch_softmax_shape(self):
        out = softmax_pytorch(self.x)
        assert out.shape == self.x.shape

    def test_pytorch_softmax_sums_to_one(self):
        out = softmax_pytorch(self.x)
        row_sums = out.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones(32, device="cuda", dtype=torch.float16),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_pytorch_matches_native(self):
        ref = softmax_native(self.x)
        out = softmax_pytorch(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = softmax_pytorch(self.x)
        out = softmax_triton(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestAttention:
    def setup_method(self):
        torch.manual_seed(42)
        self.seq_len = 256
        self.d_k = 64
        self.Q = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)
        self.K = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)
        self.V = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)

    def test_pytorch_attention_shape(self):
        out = attention_pytorch(self.Q, self.K, self.V)
        assert out.shape == (self.seq_len, self.d_k)

    def test_pytorch_matches_native(self):
        ref = attention_native(self.Q, self.K, self.V)
        out = attention_pytorch(self.Q, self.K, self.V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_matches_pytorch(self):
        ref = attention_pytorch(self.Q, self.K, self.V)
        out = attention_triton(self.Q, self.K, self.V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestFlashAttention:
    def setup_method(self):
        torch.manual_seed(42)
        self.seq_len = 256
        self.d_k = 64
        self.Q = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)
        self.K = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)
        self.V = torch.randn(self.seq_len, self.d_k, device="cuda", dtype=torch.float16)

    def test_pytorch_flash_matches_naive(self):
        ref = flash_attention_naive(self.Q, self.K, self.V)
        out = flash_attention_pytorch(self.Q, self.K, self.V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_triton_flash_matches_naive(self):
        ref = flash_attention_naive(self.Q, self.K, self.V)
        out = flash_attention_triton(self.Q, self.K, self.V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_flash_attention_shape(self):
        out = flash_attention_triton(self.Q, self.K, self.V)
        assert out.shape == (self.seq_len, self.d_k)

    def test_flash_long_sequence(self):
        """Test with sequence longer than BLOCK_SEQ to verify tiling works."""
        seq_len = 1024
        Q = torch.randn(seq_len, self.d_k, device="cuda", dtype=torch.float16)
        K = torch.randn(seq_len, self.d_k, device="cuda", dtype=torch.float16)
        V = torch.randn(seq_len, self.d_k, device="cuda", dtype=torch.float16)
        ref = flash_attention_naive(Q, K, V)
        out = flash_attention_triton(Q, K, V)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestFlashAttentionFull:
    def setup_method(self):
        torch.manual_seed(42)
        self.batch = 2
        self.n_heads = 4
        self.seq_len = 256
        self.d_k = 64
        self.Q = torch.randn(
            self.batch,
            self.n_heads,
            self.seq_len,
            self.d_k,
            device="cuda",
            dtype=torch.float16,
        )
        self.K = torch.randn(
            self.batch,
            self.n_heads,
            self.seq_len,
            self.d_k,
            device="cuda",
            dtype=torch.float16,
        )
        self.V = torch.randn(
            self.batch,
            self.n_heads,
            self.seq_len,
            self.d_k,
            device="cuda",
            dtype=torch.float16,
        )

    def test_non_causal_matches_naive(self):
        ref = flash_attention_full_naive(self.Q, self.K, self.V, causal=False)
        out = flash_attention_full_triton(self.Q, self.K, self.V, causal=False)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_causal_matches_naive(self):
        ref = flash_attention_full_naive(self.Q, self.K, self.V, causal=True)
        out = flash_attention_full_triton(self.Q, self.K, self.V, causal=True)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_matches_native_non_causal(self):
        ref = flash_attention_full_native(self.Q, self.K, self.V, causal=False)
        out = flash_attention_full_triton(self.Q, self.K, self.V, causal=False)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_matches_native_causal(self):
        ref = flash_attention_full_native(self.Q, self.K, self.V, causal=True)
        out = flash_attention_full_triton(self.Q, self.K, self.V, causal=True)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_output_shape(self):
        out = flash_attention_full_triton(self.Q, self.K, self.V, causal=False)
        assert out.shape == (self.batch, self.n_heads, self.seq_len, self.d_k)

    def test_long_sequence_causal(self):
        seq_len = 1024
        Q = torch.randn(1, 2, seq_len, self.d_k, device="cuda", dtype=torch.float16)
        K = torch.randn(1, 2, seq_len, self.d_k, device="cuda", dtype=torch.float16)
        V = torch.randn(1, 2, seq_len, self.d_k, device="cuda", dtype=torch.float16)
        ref = flash_attention_full_naive(Q, K, V, causal=True)
        out = flash_attention_full_triton(Q, K, V, causal=True)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


import pytest

# Sizes must be divisible by block sizes: BLOCK_M=64, BLOCK_K=32, BLOCK_N=64
INT8_SIZES = [(64, 32, 64), (128, 256, 512), (256, 128, 128)]
# Int4 additionally needs K % group_size == 0
INT4_SIZES = [(64, 128, 64), (128, 256, 512), (128, 256, 128)]


def _make_int8_data(M, K, N):
    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    W = torch.randn(K, N, device="cuda", dtype=torch.float16)
    W_int8, scale, zero_point = quantize_int8(W)
    return x, W, W_int8, scale, zero_point


def _make_int4_data(M, K, N, group_size=128):
    torch.manual_seed(42)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    W = torch.randn(K, N, device="cuda", dtype=torch.float16)
    W_packed, scales, zeros = quantize_int4(W, group_size)
    return x, W, W_packed, scales, zeros


# ── fp16 Triton matmul ─────────────────────────────────────────

FP16_SIZES = [(64, 64, 64), (128, 256, 512), (256, 128, 128)]


class TestFp16Matmul:
    """Triton tiled fp16 matmul vs cuBLAS (x @ W)."""

    @pytest.mark.parametrize("M,K,N", FP16_SIZES)
    def test_triton_vs_cublas(self, M, K, N):
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        W = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = matmul_fp16(x, W)
        out = matmul_fp16_triton(x, W)
        torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("M,K,N", FP16_SIZES)
    def test_output_shape(self, M, K, N):
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        W = torch.randn(K, N, device="cuda", dtype=torch.float16)
        out = matmul_fp16_triton(x, W)
        assert out.shape == (M, N)

    @pytest.mark.parametrize("M,K,N", FP16_SIZES)
    def test_output_dtype(self, M, K, N):
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        W = torch.randn(K, N, device="cuda", dtype=torch.float16)
        out = matmul_fp16_triton(x, W)
        assert out.dtype == torch.float16


# ── int8 quantization ──────────────────────────────────────────


class TestInt8Quantization:
    """Tests for quantize/dequantize utilities — no Triton needed."""

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_shapes(self, M, K, N):
        _, W, W_int8, scale, zero_point = _make_int8_data(M, K, N)
        assert W_int8.shape == (K, N)
        assert scale.shape == (N,)
        assert zero_point.shape == (N,)

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_dtypes(self, M, K, N):
        _, W, W_int8, scale, zero_point = _make_int8_data(M, K, N)
        assert W_int8.dtype in (torch.int8, torch.uint8)
        assert scale.dtype == torch.float16

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_values_in_range(self, M, K, N):
        _, _, W_int8, _, _ = _make_int8_data(M, K, N)
        if W_int8.dtype == torch.uint8:
            assert W_int8.min() >= 0 and W_int8.max() <= 255
        else:
            assert W_int8.min() >= -128 and W_int8.max() <= 127

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_scale_positive(self, M, K, N):
        _, _, _, scale, _ = _make_int8_data(M, K, N)
        assert (scale > 0).all(), "scale must be positive for every column"

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_roundtrip_error(self, M, K, N):
        """Quantize → dequantize should recover weights within ~1 quantization step."""
        _, W, W_int8, scale, zero_point = _make_int8_data(M, K, N)
        W_deq = dequantize_int8(W_int8, scale, zero_point).half()
        mean_err = (W_deq - W).abs().mean().item()
        max_err = (W_deq - W).abs().max().item()
        # Mean error should be well under one quantization step
        assert mean_err < 0.02, f"mean roundtrip error {mean_err:.4f} exceeds 0.02"
        # Max error should be at most ~1 step (scale ≈ range/255 ≈ 0.024 for randn)
        assert max_err < 0.1, f"max roundtrip error {max_err:.4f} exceeds 0.1"

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_memory_savings(self, M, K, N):
        """int8 weights should use exactly half the memory of fp16."""
        _, W, W_int8, _, _ = _make_int8_data(M, K, N)
        fp16_bytes = W.nelement() * W.element_size()
        int8_bytes = W_int8.nelement() * W_int8.element_size()
        assert int8_bytes == fp16_bytes // 2


# ── int8 matmul ─────────────────────────────────────────────────


class TestInt8Matmul:
    """Matmul correctness — PyTorch reference and Triton kernel."""

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_pytorch_vs_fp16(self, M, K, N):
        """PyTorch dequantized matmul should approximate full-precision matmul."""
        x, W, W_int8, scale, zero_point = _make_int8_data(M, K, N)
        ref = matmul_fp16(x, W).float()
        out = matmul_int8_pytorch(x, W_int8, scale, zero_point)
        # Error accumulates over K: expect ~sqrt(K) * quant_step per element
        torch.testing.assert_close(out, ref, atol=1.0, rtol=0.05)

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_triton_vs_pytorch(self, M, K, N):
        """Triton should match PyTorch exactly (same math, different execution)."""
        x, _, W_int8, scale, zero_point = _make_int8_data(M, K, N)
        ref = matmul_int8_pytorch(x, W_int8, scale, zero_point)
        out = matmul_int8_triton(x, W_int8, scale, zero_point)
        torch.testing.assert_close(out.float(), ref, atol=0.1, rtol=1e-2)

    @pytest.mark.parametrize("M,K,N", INT8_SIZES)
    def test_output_shape(self, M, K, N):
        x, _, W_int8, scale, zero_point = _make_int8_data(M, K, N)
        out = matmul_int8_triton(x, W_int8, scale, zero_point)
        assert out.shape == (M, N)


# ── int4 quantization ──────────────────────────────────────────


class TestInt4Quantization:
    """Tests for int4 quantize/dequantize utilities — no Triton needed."""

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_shapes(self, M, K, N):
        group_size = 128
        _, W, W_packed, scales, zeros = _make_int4_data(M, K, N, group_size)
        assert W_packed.shape == (K, N // 2), "two int4 values packed per byte along N"
        num_groups = K // group_size
        assert scales.shape == (num_groups, N)
        assert zeros.shape == (num_groups, N)

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_dtypes(self, M, K, N):
        _, _, W_packed, scales, zeros = _make_int4_data(M, K, N)
        assert W_packed.dtype == torch.uint8, "packed int4 pairs stored as uint8"
        assert scales.dtype == torch.float16
        assert zeros.dtype == torch.float16

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_packed_nibbles_in_range(self, M, K, N):
        """Each nibble (4-bit value) should be in [0, 15]."""
        _, _, W_packed, _, _ = _make_int4_data(M, K, N)
        lo = W_packed & 0xF
        hi = (W_packed >> 4) & 0xF
        assert lo.max() <= 15 and lo.min() >= 0
        assert hi.max() <= 15 and hi.min() >= 0

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_roundtrip_error(self, M, K, N):
        """int4 has only 16 bins — expect larger error than int8."""
        group_size = 128
        _, W, W_packed, scales, zeros = _make_int4_data(M, K, N, group_size)
        W_deq = dequantize_int4(W_packed, scales, zeros, group_size).half()
        mean_err = (W_deq - W).abs().mean().item()
        max_err = (W_deq - W).abs().max().item()
        # 16 bins over randn range → step ≈ range/15 ≈ 0.4, mean err ≈ step/4
        assert mean_err < 0.15, f"mean roundtrip error {mean_err:.4f} exceeds 0.15"
        assert max_err < 0.5, f"max roundtrip error {max_err:.4f} exceeds 0.5"

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_memory_savings(self, M, K, N):
        """int4 packed weights should use 1/4 the memory of fp16."""
        _, W, W_packed, _, _ = _make_int4_data(M, K, N)
        fp16_bytes = W.nelement() * W.element_size()
        int4_bytes = W_packed.nelement() * W_packed.element_size()
        assert int4_bytes == fp16_bytes // 4


# ── int4 matmul ─────────────────────────────────────────────────


class TestInt4Matmul:
    """Matmul correctness — PyTorch reference and Triton kernel."""

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_pytorch_vs_fp16(self, M, K, N):
        """PyTorch dequantized matmul should approximate full-precision matmul."""
        group_size = 128
        x, W, W_packed, scales, zeros = _make_int4_data(M, K, N, group_size)
        ref = matmul_fp16(x, W).float()
        out = matmul_int4_pytorch(x, W_packed, scales, zeros, group_size)
        # int4 has 16x fewer bins than int8, so larger matmul error
        mean_err = (out - ref).abs().mean().item()
        assert mean_err < 1.5, f"mean matmul error {mean_err:.4f} exceeds 1.5"

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_triton_vs_pytorch(self, M, K, N):
        """Triton should match PyTorch exactly (same math, different execution)."""
        group_size = 128
        x, _, W_packed, scales, zeros = _make_int4_data(M, K, N, group_size)
        ref = matmul_int4_pytorch(x, W_packed, scales, zeros, group_size)
        out = matmul_int4_triton(x, W_packed, scales, zeros, group_size)
        torch.testing.assert_close(out.float(), ref, atol=1.0, rtol=0.1)

    @pytest.mark.parametrize("M,K,N", INT4_SIZES)
    def test_output_shape(self, M, K, N):
        group_size = 128
        x, _, W_packed, scales, zeros = _make_int4_data(M, K, N, group_size)
        out = matmul_int4_triton(x, W_packed, scales, zeros, group_size)
        assert out.shape == (M, N)


# ── CUDA kernels ─────────────────────────────────────────────────


@pytest.mark.skipif(
    cuda_kernels is None, reason="CUDA extension not built — run `make build-cuda`"
)
class TestWMMAMatmul:
    """WMMA 16×16 tiled matmul — fp16 in, fp32 out."""

    SIZES = [(16, 16, 16), (16, 32, 16), (32, 64, 32), (64, 128, 64)]

    @pytest.mark.parametrize("M,K,N", SIZES)
    def test_matches_pytorch(self, M, K, N):
        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = A.float() @ B.float()
        out = cuda_kernels.wmma_matmul(A, B)
        torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)

    @pytest.mark.parametrize("M,K,N", SIZES)
    def test_output_shape(self, M, K, N):
        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        out = cuda_kernels.wmma_matmul(A, B)
        assert out.shape == (M, N)

    @pytest.mark.parametrize("M,K,N", SIZES)
    def test_output_dtype(self, M, K, N):
        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        out = cuda_kernels.wmma_matmul(A, B)
        assert out.dtype == torch.float32


@pytest.mark.skipif(
    cuda_kernels is None, reason="CUDA extension not built — run `make build-cuda`"
)
class TestCUDASoftmax:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device="cuda", dtype=torch.float16)

    def test_cuda_softmax_shape(self):
        out = cuda_kernels.softmax(self.x)
        assert out.shape == self.x.shape

    def test_cuda_softmax_sums_to_one(self):
        out = cuda_kernels.softmax(self.x)
        row_sums = out.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones(32, device="cuda", dtype=torch.float16),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_cuda_matches_pytorch(self):
        ref = torch.softmax(self.x.float(), dim=-1).half()
        out = cuda_kernels.softmax(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_cuda_matches_triton(self):
        ref = softmax_triton(self.x)
        out = cuda_kernels.softmax(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    cuda_kernels is None, reason="CUDA extension not built — run `make build-cuda`"
)
class TestCUDASoftmaxTriton:
    """Vectorized CUDA softmax (Triton-style single-pass caching)."""

    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 128, device="cuda", dtype=torch.float16)

    def test_shape(self):
        out = cuda_kernels.softmax_triton(self.x)
        assert out.shape == self.x.shape

    def test_sums_to_one(self):
        out = cuda_kernels.softmax_triton(self.x)
        row_sums = out.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones(32, device="cuda", dtype=torch.float16),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_matches_pytorch(self):
        ref = torch.softmax(self.x.float(), dim=-1).half()
        out = cuda_kernels.softmax_triton(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_matches_cuda_softmax(self):
        ref = cuda_kernels.softmax(self.x)
        out = cuda_kernels.softmax_triton(self.x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_large_input(self):
        x = torch.randn(32, 8192, device="cuda", dtype=torch.float16)
        ref = torch.softmax(x.float(), dim=-1).half()
        out = cuda_kernels.softmax_triton(x)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    cuda_kernels is None, reason="CUDA extension not built — run `make build-cuda`"
)
class TestCUDAFusedRMSNormSwiGLU:
    def setup_method(self):
        torch.manual_seed(42)
        self.x = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        self.gate = torch.randn(32, 2048, device="cuda", dtype=torch.float16)
        self.weight = torch.randn(2048, device="cuda", dtype=torch.float16)

    def test_shape(self):
        out = cuda_kernels.fused_rmsnorm_swiglu(self.x, self.weight, self.gate)
        assert out.shape == self.x.shape

    def test_matches_pytorch(self):
        normed = rmsnorm_pytorch(self.x, self.weight)
        ref = swiglu_pytorch(normed, self.gate)
        out = cuda_kernels.fused_rmsnorm_swiglu(self.x, self.weight, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_matches_triton(self):
        from kernels.fused_rmsnorm_swiglu import fused_rmsnorm_swiglu_triton

        ref = fused_rmsnorm_swiglu_triton(self.x, self.gate, self.weight)
        out = cuda_kernels.fused_rmsnorm_swiglu(self.x, self.weight, self.gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_large_input(self):
        x = torch.randn(32, 8192, device="cuda", dtype=torch.float16)
        gate = torch.randn(32, 8192, device="cuda", dtype=torch.float16)
        weight = torch.randn(8192, device="cuda", dtype=torch.float16)
        normed = rmsnorm_pytorch(x, weight)
        ref = swiglu_pytorch(normed, gate)
        out = cuda_kernels.fused_rmsnorm_swiglu(x, weight, gate)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# ── CUDA FlashAttention (WMMA) ────────────────────────────────────


@pytest.mark.skipif(
    flash_attn_cuda is None, reason="flash_attn_cuda extension not built — run `make build-fa`"
)
class TestCUDAFlashAttention:
    """WMMA-based FlashAttention-2 forward pass.

    Layout convention: (batch, seqlen, num_heads, head_dim).
    Compares against PyTorch's scaled_dot_product_attention (Tri Dao's CUDA FA).
    """

    @pytest.mark.parametrize("seqlen", [128, 256, 512])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_single_head(self, seqlen, head_dim):
        torch.manual_seed(0)
        q = torch.randn(1, seqlen, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(1, seqlen, 1, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(1, seqlen, 1, head_dim, device="cuda", dtype=torch.float16)

        out, _lse = flash_attn_cuda.mha_fwd(q, k, v, False)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("batch,heads,seqlen,head_dim", [
        (1, 4, 128, 64),
        (2, 4, 256, 64),
        (1, 1, 128, 128),
        (2, 4, 256, 128),
    ])
    def test_multi_head(self, batch, heads, seqlen, head_dim):
        torch.manual_seed(0)
        q = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.float16)

        out, _lse = flash_attn_cuda.mha_fwd(q, k, v, False)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_output_shape(self):
        torch.manual_seed(0)
        q = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
        out, lse = flash_attn_cuda.mha_fwd(q, k, v, False)
        assert out.shape == (2, 256, 4, 64)
        assert lse.shape == (2, 4, 256)


@pytest.mark.skipif(flash_attn_cutlass is None, reason="flash_attn_cutlass not built")
class TestCUTLASSFlashAttention:
    """CUTLASS/CuTe-based FlashAttention-2 forward pass.

    Layout convention: (batch, seqlen, num_heads, head_dim).
    Compares against PyTorch's scaled_dot_product_attention.
    """

    @pytest.mark.parametrize("seqlen", [128, 256, 512])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_single_head(self, seqlen, head_dim):
        torch.manual_seed(0)
        q = torch.randn(1, seqlen, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(1, seqlen, 1, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(1, seqlen, 1, head_dim, device="cuda", dtype=torch.float16)

        out, _lse = flash_attn_cutlass.mha_fwd(q, k, v, False)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("batch,heads,seqlen,head_dim", [
        (1, 4, 128, 64),
        (2, 4, 256, 64),
        (1, 1, 128, 128),
        (2, 4, 256, 128),
    ])
    def test_multi_head(self, batch, heads, seqlen, head_dim):
        torch.manual_seed(0)
        q = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.float16)

        out, _lse = flash_attn_cutlass.mha_fwd(q, k, v, False)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_output_shape(self):
        torch.manual_seed(0)
        q = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 256, 4, 64, device="cuda", dtype=torch.float16)
        out, lse = flash_attn_cutlass.mha_fwd(q, k, v, False)
        assert out.shape == (2, 256, 4, 64)
        assert lse.shape == (2, 4, 256)

.PHONY: setup-cuda build-cuda test test-triton test-cuda clean-cuda

# Install CUDA dev headers (only need to run once)
setup-cuda:
	conda install -c nvidia/label/cuda-13.0.0 cuda-cudart-dev cuda-nvcc cuda-cccl --no-deps -y
	@echo "Done. Make sure CUDA_HOME is set:"
	@echo "  export CUDA_HOME=$$CONDA_PREFIX"

# Build CUDA extension
build-cuda:
	cd cuda && TORCH_CUDA_ARCH_LIST="8.0" python setup.py build_ext --inplace
	cp cuda/cuda_kernels*.so . 2>/dev/null || cp cuda/build/lib*/cuda_kernels*.so . 2>/dev/null
	@echo "Built cuda_kernels extension"

# Run all tests
test:
	python -m pytest tests/test_kernels.py -v

# Run only Triton tests
test-triton:
	python -m pytest tests/test_kernels.py -v -k "not CUDA"

# Run only CUDA tests
test-cuda:
	python -m pytest tests/test_kernels.py -v -k "CUDA"

# Benchmark softmax (Triton only — no CUDA build needed)
bench-softmax:
	python benchmarks/bench_softmax.py

# Benchmark softmax with CUDA (requires make build-cuda)
bench-cuda-softmax:
	python benchmarks/bench_cuda_softmax.py

# Clean CUDA build artifacts
clean-cuda:
	rm -rf cuda/build cuda/dist cuda/*.egg-info cuda/*.so cuda_kernels*.so

# Build the WMMA FlashAttention CUDA extension
build-fac:
	cd cuda/flash_attn && python setup.py build_ext --inplace
	@echo "Built flash_attn_cuda extension"

# Run FlashAttention CUDA correctness tests
test-fac:
	LD_PRELOAD=$$CONDA_PREFIX/lib/libstdc++.so.6 python -m pytest tests/test_kernels.py -v -k "CUDAFlashAttention"

# Benchmark FlashAttention CUDA vs PyTorch SDPA
bench-fac:
	LD_PRELOAD=$$CONDA_PREFIX/lib/libstdc++.so.6 python benchmarks/bench_cuda_flash_attention.py

# Clean FlashAttention build artifacts
clean-fac:
	rm -rf cuda/flash_attn/build cuda/flash_attn/*.so cuda/flash_attn/*.egg-info

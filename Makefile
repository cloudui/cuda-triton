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

# Profile FlashAttention CUDA with Nsight Compute.
# Override the workload via PROF_ARGS, e.g. `make prof-fac PROF_ARGS="--seq 4096 --causal"`.
# Output is teed to profiles/prof_<set>_<seq>.txt for diffing across versions.
# Note: LD_PRELOAD is needed because conda's libstdc++ is newer than the system's,
# and the extension's .so requires the conda version's CXXABI.
seq ?= 2048
PROF_ARGS ?= --seq $(seq)
PROF_SEQ  := $(shell echo "$(PROF_ARGS)" | grep -oE -- "--seq [0-9]+" | awk '{print $$2}')
PROF_PRELOAD := LD_PRELOAD=$$CONDA_PREFIX/lib/libstdc++.so.6

# Quick metrics — Speed of Light, Launch Stats, Occupancy. Fast.
prof-fac:
	@mkdir -p profiles
	$(PROF_PRELOAD) ncu --set basic --target-processes all \
		--kernel-name flash_fwd_kernel \
		--launch-skip 5 --launch-count 1 \
		python cuda/flash_attn/profile_runner.py $(PROF_ARGS) \
		2>&1 | tee profiles/prof_basic_seq$(PROF_SEQ).txt

# Full metrics — adds Warp State Stall Reasons, Memory Workload Analysis, Source/SASS counters.
# Slower (~10x) but tells you WHY warps are stalling and whether you have bank conflicts.
prof-fac-full:
	@mkdir -p profiles
	$(PROF_PRELOAD) ncu --set full --target-processes all \
		--kernel-name flash_fwd_kernel \
		--launch-skip 5 --launch-count 1 \
		python cuda/flash_attn/profile_runner.py $(PROF_ARGS) \
		2>&1 | tee profiles/prof_full_seq$(PROF_SEQ).txt

# GUI-loadable .ncu-rep file for inspection in nsight-compute UI (`ncu-ui profiles/prof_<seq>.ncu-rep`).
prof-fac-rep:
	@mkdir -p profiles
	$(PROF_PRELOAD) ncu --set full -o profiles/prof_seq$(PROF_SEQ) --force-overwrite \
		--target-processes all \
		--kernel-name flash_fwd_kernel \
		--launch-skip 5 --launch-count 1 \
		python cuda/flash_attn/profile_runner.py $(PROF_ARGS)

# Nsight Systems timeline (works when ncu is blocked by host monitoring).
prof-nsys-fac:
	@mkdir -p profiles
	$(PROF_PRELOAD) nsys profile --stats=true -o profiles/nsys_seq$(PROF_SEQ) --force-overwrite=true \
		python cuda/flash_attn/profile_runner.py $(PROF_ARGS)

# ---- CUTLASS FlashAttention ---------------------------------------------------
# Set CUTLASS_DIR to point at your CUTLASS checkout (defaults to third_party/cutlass).
# Clone first: git clone https://github.com/NVIDIA/cutlass.git third_party/cutlass

CUTLASS_DIR ?= third_party/cutlass

# Clone CUTLASS as a submodule under third_party/
fetch-cutlass:
	@if [ -d "$(CUTLASS_DIR)" ]; then \
		echo "CUTLASS already present at $(CUTLASS_DIR)"; \
	else \
		mkdir -p $$(dirname $(CUTLASS_DIR)); \
		git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR); \
	fi

build-fac-cutlass:
	cd cuda/flash_attn_cutlass && CUTLASS_DIR=$(abspath $(CUTLASS_DIR)) python setup.py build_ext --inplace
	@echo "Built flash_attn_cutlass extension"

test-fac-cutlass:
	$(PROF_PRELOAD) python -m pytest tests/test_kernels.py -v -k "CUTLASSFlashAttention"

bench-fac-cutlass:
	$(PROF_PRELOAD) python benchmarks/bench_cutlass_flash_attention.py

clean-fac-cutlass:
	rm -rf cuda/flash_attn_cutlass/build cuda/flash_attn_cutlass/*.so cuda/flash_attn_cutlass/*.egg-info

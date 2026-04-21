import torch
from torch.utils.cpp_extension import CUDA_HOME

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch built with CUDA: {torch.version.cuda}")
print(f"CUDA_HOME (for WMMA/cuBLAS headers): {CUDA_HOME}")

# Check if Tensor Cores (WMMA) are accessible
major, minor = torch.cuda.get_device_capability()
if major >= 7:
    print(f"GPU Capability {major}.{minor}: Tensor Cores (WMMA) supported.")

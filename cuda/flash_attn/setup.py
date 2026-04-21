from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_cuda",
    ext_modules=[
        CUDAExtension(
            "flash_attn_cuda",
            [
                "flash_api.cu",
                "flash_fwd_hdim64_fp16_sm80.cu",
                "flash_fwd_hdim64_fp16_causal_sm80.cu",
                "flash_fwd_hdim128_fp16_sm80.cu",
                "flash_fwd_hdim128_fp16_causal_sm80.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                    "-std=c++17",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

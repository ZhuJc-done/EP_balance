"""
Build script for nccl_gin extension.

Usage:
    # JIT build (preferred, no install needed):
    import nccl_gin  # auto-compiles on first import

    # Or pre-build with:
    cd nccl_gin && python setup.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NCCL_HOME = os.environ.get("NCCL_HOME", "/opt/tiger/nccl")

setup(
    name="nccl_gin",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="nccl_gin._nccl_gin_C",
            sources=["csrc/nccl_gin.cu"],
            include_dirs=[
                os.path.join(NCCL_HOME, "include"),
            ],
            library_dirs=[
                os.path.join(NCCL_HOME, "lib"),
            ],
            libraries=["nccl"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": [
                    "-std=c++17",
                    "-O3",
                    "-gencode", "arch=compute_90,code=sm_90",
                    "-gencode", "arch=compute_100,code=sm_100",
                    "-DNCCL_GIN_PROXY_ENABLE=1",
                ],
            },
            extra_link_args=[
                f"-Wl,-rpath,{os.path.join(NCCL_HOME, 'lib')}",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

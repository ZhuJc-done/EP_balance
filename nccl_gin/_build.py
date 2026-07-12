"""JIT build helper for the _nccl_gin_C extension."""

import os
import torch
from torch.utils.cpp_extension import load

_backend = None
_NCCL_HOME = os.environ.get("NCCL_HOME", "/opt/tiger/nccl")


def _get_backend():
    global _backend
    if _backend is not None:
        return _backend

    src_dir = os.path.join(os.path.dirname(__file__), "csrc")
    sources = [os.path.join(src_dir, "nccl_gin.cu")]

    extra_include_paths = [
        os.path.join(_NCCL_HOME, "include"),
    ]

    extra_ldflags = [
        f"-L{os.path.join(_NCCL_HOME, 'lib')}",
        "-lnccl",
        f"-Wl,-rpath,{os.path.join(_NCCL_HOME, 'lib')}",
    ]

    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        # "-gencode", "arch=compute_90,code=sm_90", # speedup build
        "-gencode", "arch=compute_100,code=sm_100",
        "-DNCCL_GIN_PROXY_ENABLE=1",
    ]

    _backend = load(
        name="_nccl_gin_C",
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_ldflags=extra_ldflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    return _backend

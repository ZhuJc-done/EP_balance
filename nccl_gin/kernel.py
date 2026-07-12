"""
Drop-in replacement for the Triton getmem_kernel and putmem_kernel using NCCL GIN.

Uses the nccl_gin C++ extension to execute P2P gets and puts over NVLink/RDMA.
The extension is JIT-compiled on first use.
"""

import torch
from ._build import _get_backend

class _NcclGinGetLauncher:
    """Returned by ``getmem_kernel[(grid,)]``; calling it launches the GIN get kernel."""

    __slots__ = ("_grid",)

    def __init__(self, grid_size: int):
        self._grid = grid_size

    def __call__(
        self,
        symm_inputs_buffer,
        symm_outputs_buffer,
        symm_inputs_buffer_start_idx,
        symm_outputs_buffer_start_idx,
        size,
        element_size,
        peer,
        num_warps=32,  # ignored by nccl_gin wrapper as it uses predefined block sizes
        stream_handle=None,
    ):
        _C = _get_backend()
        
        src_offset = symm_inputs_buffer_start_idx
        dst_offset = symm_outputs_buffer_start_idx
        num_bytes = size * element_size
        
        stream_opt = None
        if stream_handle is not None:
            stream_opt = torch.cuda.ExternalStream(stream_handle)
        
        _C.get(
            symm_inputs_buffer,
            symm_outputs_buffer,
            src_offset,
            dst_offset,
            num_bytes,
            peer,
            self._grid,
            stream_opt
        )


class _NcclGinGetKernel:
    """Mimics Triton's ``@jit`` kernel object: ``kernel[(grid,)](args…)``."""

    supports_explicit_stream_handle = True

    def __getitem__(self, grid):
        if isinstance(grid, tuple):
            grid = grid[0]
        return _NcclGinGetLauncher(grid)


getmem_kernel = _NcclGinGetKernel()


class _NcclGinPutLauncher:
    """Returned by ``putmem_kernel[(grid,)]``; calling it launches the GIN put kernel."""

    __slots__ = ("_grid",)

    def __init__(self, grid_size: int):
        self._grid = grid_size

    def __call__(
        self,
        symm_inputs_buffer,
        symm_outputs_buffer,
        symm_inputs_buffer_start_idx,
        symm_outputs_buffer_start_idx,
        size,
        element_size,
        peer,
        num_warps=32,  # ignored by nccl_gin wrapper as it uses predefined block sizes
        stream_handle=None,
    ):
        _C = _get_backend()
        
        src_offset = symm_inputs_buffer_start_idx
        dst_offset = symm_outputs_buffer_start_idx
        num_bytes = size * element_size
        
        stream_opt = None
        if stream_handle is not None:
            stream_opt = torch.cuda.ExternalStream(stream_handle)
        
        _C.put(
            symm_inputs_buffer,
            symm_outputs_buffer,
            src_offset,
            dst_offset,
            num_bytes,
            peer,
            self._grid,
            stream_opt
        )


class _NcclGinPutKernel:
    """Mimics Triton's ``@jit`` kernel object: ``kernel[(grid,)](args…)``."""

    supports_explicit_stream_handle = True

    def __getitem__(self, grid):
        if isinstance(grid, tuple):
            grid = grid[0]
        return _NcclGinPutLauncher(grid)


putmem_kernel = _NcclGinPutKernel()

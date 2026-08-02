"""
NCCL Symmetric Memory P2P Put/Get Library

Drop-in replacement for nvshmem-based P2P communication using NCCL symmetric
memory (ncclMemAlloc + ncclCommWindowRegister) and peer device pointers.
Works on NVLink P2P without requiring NCCL_P2P_DISABLE or NCCL_SHM_DISABLE.

The batched entry points pick their transport per descriptor: peers inside the
local LSA team (NVLink / P2P mapped) are moved with load/store, everyone else
over GIN's network RDMA. One window registration covers both. Set
``EPLB_GIN_LSA=0`` to force the network path for every peer, which is only
useful for A/B measurement -- it sends intra-node traffic out to the NIC.

Usage:
    import nccl_gin
    nccl_gin.init(process_group)
    buf = nccl_gin.create_tensor(size, torch.uint8)
    nccl_gin.put(buf, buf, src_off, dst_off, nbytes, peer, grid_size=4)
    nccl_gin.get(buf, buf, remote_off, local_off, nbytes, peer, grid_size=4)
    nccl_gin.destroy()
"""

import hashlib
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

from ._build import _get_backend

_C = None


def _log_init(message: str) -> None:
    if os.environ.get("MARIANA_GIN_PP_LOG_INIT", "1").lower() not in (
            "1", "true", "yes", "on"):
        return
    try:
        rank = dist.get_rank() if dist.is_initialized() else -1
    except Exception:
        rank = -1
    print(f"[nccl_gin_py] rank={rank} {message}",
          file=sys.stderr, flush=True)


def _ensure_loaded():
    global _C
    if _C is None:
        _log_init(f"backend load start file={__file__}")
        _C = _get_backend()
        _log_init("backend load done")


def _lsa_default() -> bool:
    """Whether the batched paths may serve LSA-team peers over NVLink (``EPLB_GIN_LSA``)."""
    return os.environ.get("EPLB_GIN_LSA", "1").strip().lower() not in ("0", "false", "no", "off")


def _run_with_stream(stream: Optional[torch.cuda.Stream], fn):
    """Run *fn* on the requested CUDA stream while passing None to pybind.

    The C++ binding accepts ``c10::cuda::CUDAStream | None``, but pybind does
    not reliably convert Python ``torch.cuda.Stream`` objects. Switch the
    current stream in Python and let the extension fetch it via
    ``getCurrentCUDAStream()`` when ``None`` is passed.
    """
    if stream is None:
        return fn(None)
    with torch.cuda.stream(stream):
        return fn(None)


def _group_ranks(process_group: dist.ProcessGroup) -> list[int]:
    try:
        return list(dist.get_process_group_ranks(process_group))
    except Exception:
        if process_group is dist.group.WORLD:
            return list(range(dist.get_world_size()))
        return []


def _broadcast_uid_via_store(process_group: dist.ProcessGroup,
                             uid_bytes: bytes,
                             root_global_rank: int) -> bytes:
    """Distribute NCCL unique id without enqueueing a NCCL collective.

    PP setup may have just used unbatched NCCL P2P on the same process group.
    Using the default rendezvous store here avoids mixing that lazy P2P
    communicator creation with a CUDA broadcast collective.
    """
    store = dist.distributed_c10d._get_default_store()
    ranks = _group_ranks(process_group)
    if ranks:
        key_src = ",".join(str(r) for r in ranks)
    else:
        key_src = f"root={root_global_rank};world={dist.get_world_size(group=process_group)}"
    key_hash = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    key = f"nccl_gin/uid/{key_hash}"

    if dist.get_rank() == root_global_rank:
        _log_init(f"store set start key={key} ranks={key_src}")
        store.set(key, uid_bytes)
        _log_init(f"store set done key={key} ranks={key_src}")

    _log_init(f"store get start key={key} ranks={key_src}")
    uid = store.get(key)
    _log_init(f"store get done key={key} bytes={len(uid)}")
    return bytes(uid)


def init(process_group: Optional[dist.ProcessGroup] = None):
    """Initialize from a PyTorch distributed process group.

    Creates an independent ncclComm with the same topology as *process_group*
    (defaults to the global group).  All env configuration (LD_PRELOAD,
    NCCL_NET, IB vars, etc.) must be set by the launch script.
    """
    uid_transport = os.environ.get("MARIANA_GIN_INIT_UID_TRANSPORT", "store").lower()
    _log_init(f"init enter uid_transport={uid_transport}")
    _ensure_loaded()
    if process_group is None:
        process_group = dist.group.WORLD
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    try:
        root_global_rank = dist.get_global_rank(process_group, 0)
    except Exception:
        root_global_rank = 0

    _log_init(
        f"init ranks group_rank={rank} world_size={world_size} "
        f"root_global_rank={root_global_rank}")

    if rank == 0:
        _log_init("get_unique_id start")
        uid_bytes = _C.get_unique_id()
        _log_init(f"get_unique_id done bytes={len(uid_bytes)}")
    else:
        uid_bytes = b""

    if uid_transport == "broadcast":
        _log_init("uid broadcast start")
        uid_len = len(uid_bytes) if rank == 0 else 0
        # /tmp is not shared across nodes, so distribute the NCCL unique id through
        # the requested process group. Use CUDA tensors so NCCL-backed groups work.
        device = torch.device(
            "cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        uid_len_tensor = torch.tensor([uid_len], dtype=torch.int64, device=device)
        dist.broadcast(uid_len_tensor, src=root_global_rank, group=process_group)
        uid_len = int(uid_len_tensor.item())

        uid_tensor = torch.empty(uid_len, dtype=torch.uint8, device=device)
        if rank == 0:
            uid_tensor.copy_(torch.tensor(
                list(uid_bytes), dtype=torch.uint8, device=device))
        dist.broadcast(uid_tensor, src=root_global_rank, group=process_group)
        uid_bytes = bytes(uid_tensor.cpu().tolist())
        _log_init(f"uid broadcast done bytes={len(uid_bytes)}")
    else:
        uid_bytes = _broadcast_uid_via_store(
            process_group, uid_bytes, root_global_rank)

    _log_init(f"C.init start group_rank={rank} world_size={world_size}")
    _C.init(uid_bytes, rank, world_size)
    _log_init("C.init done")


def init_from_comm(comm_handle: int):
    """Initialize from an existing ncclComm_t pointer.

    *comm_handle* is the integer value of the ``ncclComm_t`` pointer,
    e.g. obtained via ``ProcessGroupNCCL._comm_ptr()``.  The communicator
    is borrowed — it will NOT be destroyed on :func:`destroy`.
    """
    _ensure_loaded()
    _C.init_from_comm(comm_handle)


def create_tensor(numel: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Allocate NCCL symmetric memory and register as window.

    Compatible with ``triton_dist.utils.nvshmem_create_tensor``.
    """
    _ensure_loaded()
    return _C.create_tensor(numel, dtype)


def is_symmetric_tensor(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* aliases a registered NCCL GIN symmetric window."""
    _ensure_loaded()
    if not torch.is_tensor(tensor):
        return False
    return bool(_C.is_symmetric_tensor(tensor))


def put(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_offset: int,
    dst_offset: int,
    num_bytes: int,
    peer: int,
    grid_size: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
):
    """P2P put using a vectorized copy kernel (fire-and-forget).

    Compatible with ``putmem_kernel`` usage in ``dist_triton_kernel.py``.
    Transfers *num_bytes* from ``src_buffer[src_offset:]`` on the local rank
    to ``dst_buffer[dst_offset:]`` on *peer*.  Both buffers must have been
    created via :func:`create_tensor`.

    The operation is enqueued on *stream* (or the current CUDA stream) and
    returns immediately.
    """
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.put(
            src_buffer, dst_buffer, src_offset, dst_offset, num_bytes, peer,
            grid_size, stream_opt))


def get(
    remote_buffer: torch.Tensor,
    local_buffer: torch.Tensor,
    remote_offset: int,
    local_offset: int,
    num_bytes: int,
    peer: int,
    grid_size: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
):
    """P2P get using a vectorized copy kernel (fire-and-forget).

    Compatible with ``getmem_kernel`` usage in ``dist_triton_kernel.py``.
    Reads *num_bytes* from ``remote_buffer[remote_offset:]`` on *peer*
    into ``local_buffer[local_offset:]`` on the local rank.  Both buffers
    must have been created via :func:`create_tensor`.

    The operation is enqueued on *stream* (or the current CUDA stream) and
    returns immediately.
    """
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.get(
            remote_buffer, local_buffer, remote_offset, local_offset,
            num_bytes, peer, grid_size, stream_opt))


def get_batched(
    remote_buffer: torch.Tensor,
    local_buffer: torch.Tensor,
    remote_off: torch.Tensor,
    local_off: torch.Tensor,
    nbytes: torch.Tensor,
    peers: torch.Tensor,
    k: Optional[int] = None,
    blocks_per_desc: int = 1,
    use_lsa: Optional[bool] = None,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Batched P2P get from a device-resident schedule (no host peer loop, no D2H).

    Services ``k`` transfers in a single launch. ``remote_off`` / ``local_off`` / ``nbytes``
    are ``int64`` CUDA tensors of window-relative byte offsets and sizes; ``peers`` is an
    ``int32`` CUDA tensor of source ranks. Descriptors with ``peers[i] < 0`` are skipped on
    device, so empty / local entries need no host branch. Both buffers must be symmetric
    (:func:`create_tensor`). The transfer completes on ``stream`` (GIN kernels flush).

    Descriptors whose peer is in the local LSA team are read over NVLink with load/store
    rather than through the NIC; pass ``use_lsa=False`` (or ``EPLB_GIN_LSA=0``) to disable.
    """
    _ensure_loaded()
    if k is None:
        k = int(peers.numel())
    if use_lsa is None:
        use_lsa = _lsa_default()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.get_batched(
            remote_buffer, local_buffer, remote_off, local_off, nbytes, peers,
            int(k), int(blocks_per_desc), bool(use_lsa), stream_opt))


def put_batched(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_off: torch.Tensor,
    dst_off: torch.Tensor,
    nbytes: torch.Tensor,
    peers: torch.Tensor,
    k: Optional[int] = None,
    blocks_per_desc: int = 1,
    use_lsa: Optional[bool] = None,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Batched P2P put from a device-resident schedule (no host peer loop, no D2H).

    Mirror of :func:`get_batched`; ``peers`` holds destination ranks and each transfer
    pushes ``src_buffer[src_off[i]:]`` into ``dst_buffer[dst_off[i]:]`` on ``peers[i]``.
    LSA-team peers are written over NVLink with load/store; the kernel issues a system-scope
    fence before it exits so the caller's ordering fence covers those stores too.
    """
    _ensure_loaded()
    if k is None:
        k = int(peers.numel())
    if use_lsa is None:
        use_lsa = _lsa_default()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.put_batched(
            src_buffer, dst_buffer, src_off, dst_off, nbytes, peers,
            int(k), int(blocks_per_desc), bool(use_lsa), stream_opt))


def put_signal(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_offset: int,
    dst_offset: int,
    num_bytes: int,
    peer: int,
    sig_idx: int = 0,
    ctx: int = 0,
    stream: Optional[torch.cuda.Stream] = None,
):
    """P2P put via host-side ``ncclPutSignal`` (transfer + completion signal)."""
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.put_signal(
            src_buffer, dst_buffer, src_offset, dst_offset, num_bytes,
            peer, sig_idx, ctx, stream_opt))


def put_signal_device(
    src_buffer: torch.Tensor,
    dst_buffer: torch.Tensor,
    src_offset: int,
    dst_offset: int,
    num_bytes: int,
    peer: int,
    sig_idx: int,
    stream: Optional[torch.cuda.Stream] = None,
):
    """P2P put from a device kernel with official ``ncclGin_SignalInc``.

    Unlike :func:`put_signal`, this does not use host-side ``ncclPutSignal``.
    The kernel stripes the payload across active GIN contexts and increments
    *sig_idx* once per active context after that context's payload is settled.
    Pair it with :func:`wait_signal_device`.
    """
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.put_signal_device(
            src_buffer, dst_buffer, src_offset, dst_offset, num_bytes,
            peer, sig_idx, stream_opt))


def signal(
    peer: int,
    sig_idx: int = 0,
    ctx: int = 0,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Send signal *sig_idx* to *peer* without transferring a payload.

    Only defined for peers this rank reaches over the network: signals ride a GIN connection, and
    there is none to a peer inside the node. Use :func:`world_fence` to order the whole group.
    """
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.signal(peer, sig_idx, ctx, stream_opt))


def world_fence(index: int = 0, stream: Optional[torch.cuda.Stream] = None):
    """Stream-ordered barrier across every rank, over whichever transport reaches each one.

    Composes an LSA barrier inside the node with a GIN rail barrier across nodes, so unlike a
    ``signal``/``wait_signal`` mesh it works when some or all peers are intra-node. Being a kernel
    on ``stream`` rather than a host call, it is capture-safe and orders against side streams.
    ``index`` selects one of the provisioned barrier slots; the same index must be used by every
    rank for a given fence, and successive fences on one index are matched by epoch.
    """
    _ensure_loaded()
    return _run_with_stream(stream, lambda stream_opt: _C.world_fence(int(index), stream_opt))


def wait_signal(
    peer: int,
    sig_idx: int = 0,
    op_cnt: int = 1,
    ctx: int = 0,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Wait on *stream* until signal *sig_idx* from *peer* reaches *op_cnt*."""
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.wait_signal(peer, sig_idx, op_cnt, ctx, stream_opt))


def wait_signal_device(
    sig_idx: int,
    num_bytes: int,
    stream: Optional[torch.cuda.Stream] = None,
):
    """Wait for official device-side GIN signals on *sig_idx*.

    The device kernel uses NCCL's signal shadow to consume exactly one
    increment per active context. This avoids over-consuming when multiple
    signals arrive before the wait kernel runs.
    """
    _ensure_loaded()
    return _run_with_stream(
        stream,
        lambda stream_opt: _C.wait_signal_device(sig_idx, num_bytes, stream_opt))


def test_signal_device(
    sig_idx: int,
    num_bytes: int,
    ready: Optional[torch.Tensor] = None,
    consume: bool = True,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Nonblocking device-side signal test.

    Returns a CUDA int32 tensor whose first element is set to 1 when one signal
    increment is ready for every active GIN context, otherwise 0. When
    ``consume`` is true, the signal shadow is advanced by exactly one increment
    only if all active contexts are ready.
    """
    _ensure_loaded()
    if ready is None:
        ready = torch.empty((1,), dtype=torch.int32, device="cuda")
    _run_with_stream(
        stream,
        lambda stream_opt: _C.test_signal_device(
            sig_idx, num_bytes, ready, consume, stream_opt))
    return ready


def query_signal_device(
    sig_idx: int,
    num_bytes: int,
    ready: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Non-consuming alias for :func:`test_signal_device`."""
    return test_signal_device(sig_idx, num_bytes, ready, False, stream)


def get_rank() -> int:
    _ensure_loaded()
    return _C.get_rank()


def get_world_size() -> int:
    _ensure_loaded()
    return _C.get_world_size()


def get_comm_properties() -> dict:
    """Return NCCL communicator properties for the initialized GIN comm."""
    _ensure_loaded()
    return dict(_C.get_comm_properties())


def destroy():
    """Release all NCCL resources."""
    _ensure_loaded()
    _C.destroy()

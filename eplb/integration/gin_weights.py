"""Device-initiated EPLB weight replication over NCCL GIN (symmetric memory).

Sync-free (Level 2): the per-slot replication schedule is built and consumed entirely on device.
``materialize`` derives ``slot_to_e`` / ``main_of_slot`` with the same cumsum/scatter tables as the
token path (no ``nonzero``), turns them into device-resident descriptor arrays
``(remote_off, local_off, nbytes, peers)`` and drives one batched GIN get (forward) / put (backward)
per weight tensor -- the kernel skips ``peers < 0`` on device, so empty and local slots need no host
branch and there is **no D2H** to read the schedule. Local (``main(e) == my_rank``) slots are filled
with a vectorised on-device gather instead of a self-routed GIN transfer.

Two ordering fences per direction remain (homes-visible-before-get, reads-done-before-recycle, and the
backward puts-landed-before-sum). They default to ``dist.barrier`` (host-launched); set
``EPLB_GIN_FENCE=signal`` for the device-stream ``ncclSignal``/``ncclWaitSignal`` fence that is CUDA-graph
capturable (cluster-validate the signal-counter epochs before relying on it in capture).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist


def gin_enabled() -> bool:
    """Whether the GIN weight-replication backend is selected (``EPLB_WEIGHT_COMM=gin``)."""
    return os.environ.get("EPLB_WEIGHT_COMM", "").strip().lower() in ("gin", "nccl_gin", "devnccl")


def _nbytes(shape: torch.Size, elem_size: int) -> int:
    return int(math.prod(shape)) * int(elem_size)


def _replica_schedule(
    slot_to_e: torch.Tensor,      # int64 [S], logical expert at each local slot (-1 if empty)
    main_of_slot: torch.Tensor,   # int64 [S], main(e) of that slot's expert (-1 if empty)
    local_slot_of_e: torch.Tensor,  # int64 [E], index of e inside main(e)'s home buffer
    my_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device-resident replication schedule (no host loop, no D2H).

    Returns three device tensors over the ``S`` local slots:
      * ``peers`` int32 ``[S]``: source/destination rank for each slot, or ``-1`` for slots the
        batched GIN kernel must skip (empty slots and slots whose main is this rank).
      * ``ls``    int64 ``[S]``: ``local_slot(e)`` -- the row of ``e`` in ``main(e)``'s home buffer.
      * ``is_local`` bool ``[S]``: slots hosted by this rank's own main copy (filled by a local gather).
    """
    valid = slot_to_e >= 0
    ls = local_slot_of_e[slot_to_e.clamp(min=0)]
    is_remote = valid & (main_of_slot != my_rank)
    is_local = valid & (main_of_slot == my_rank)
    peers = torch.where(is_remote, main_of_slot, torch.full_like(main_of_slot, -1)).to(torch.int32)
    return peers, ls, is_local


class GinWeightReplicator:
    """Owns the NCCL GIN comm + symmetric buffers and the static ``main(e)`` layout.

    One instance is reused across layers/micro-batches for a given (EP group, weight
    shapes, dtype, ``n_slot``, ``E``); buffers are allocated once and recycled.
    """

    def __init__(
        self,
        *,
        group,
        num_experts: int,
        n_slot: int,
        main_rank: torch.Tensor,   # int64 [E], group-local main(e) (static, C7)
        weight_shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        device,
    ) -> None:
        import nccl_gin  # local import: only needed on the cluster

        self._gin = nccl_gin
        self.group = group
        self.E = int(num_experts)
        self.n_slot = int(n_slot)
        self.dtype = dtype
        self.device = device
        self.weight_shapes = [torch.Size(s) for s in weight_shapes]
        self.J = len(self.weight_shapes)
        self.grid = int(os.environ.get("EPLB_GIN_GRID", "8"))

        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        self.my_rank = dist.get_rank(group) if dist.is_initialized() else 0

        elem = torch.empty((), dtype=dtype).element_size()
        self.wb = [_nbytes(s, elem) for s in self.weight_shapes]  # bytes per weight tensor j

        # static per-expert layout: local_slot(e) = index of e within main(e)'s home buffer,
        # in ascending expert-id order. Computed once on host from the immutable main_rank.
        main_host = main_rank.to(torch.int64).tolist()
        self.main_host: List[int] = [int(m) for m in main_host]
        counts = [0] * self.world
        local_slot_of_e: List[int] = [0] * self.E
        for e in range(self.E):
            m = self.main_host[e]
            local_slot_of_e[e] = counts[m]
            counts[m] += 1
        self.home_cap = max(counts) if counts else 0  # max experts any rank is main of

        # device-resident copies of the static tables, so the per-step schedule never touches host
        self.main_dev = torch.as_tensor(self.main_host, dtype=torch.int64, device=device)
        self.local_slot_dev = torch.as_tensor(local_slot_of_e, dtype=torch.int64, device=device)
        self.local_slot_of_e = local_slot_of_e  # host list (used only for the static home-publish loop)
        self.slot_arange = torch.arange(self.n_slot, dtype=torch.int64, device=device)

        # fence: default host barrier (tested); 'signal' = device-stream GIN signal (capturable)
        self._fence_mode = os.environ.get("EPLB_GIN_FENCE", "barrier").strip().lower()
        self._epoch: Dict[int, int] = {}

        # lazily initialise the GIN comm (idempotent guard on the module singleton)
        try:
            self._gin.get_rank()
        except Exception:
            self._gin.init(group)

        # symmetric byte buffers (allocated once, recycled every step)
        self.home = [self._gin.create_tensor(self.home_cap * self.wb[j], torch.uint8)
                     for j in range(self.J)]
        self.slot = [self._gin.create_tensor(self.n_slot * self.wb[j], torch.uint8)
                     for j in range(self.J)]
        self.scratch = [self._gin.create_tensor(self.home_cap * self.world * self.wb[j], torch.uint8)
                        for j in range(self.J)]

    # -- schedule ---------------------------------------------------------------
    def slot_tables(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Device tables ``(slot_to_e[S], main_of_slot[S])`` for ``my_rank`` (no D2H, no ``nonzero``).

        ``slot_to_e[s]`` is the logical expert in local slot ``s`` (-1 if empty), placed in ascending
        expert-id order; ``main_of_slot[s] = main(slot_to_e[s])`` (-1 if empty).
        """
        from .eplb_manager import _slot_tables  # reuse the sync-free cumsum/scatter tables

        slot_to_e, _, _ = _slot_tables(x, self.my_rank, self.n_slot)  # device, -1 where empty
        main_of_slot = torch.where(
            slot_to_e >= 0,
            self.main_dev[slot_to_e.clamp(min=0)],
            torch.full((self.n_slot,), -1, dtype=torch.int64, device=x.device),
        )
        return slot_to_e, main_of_slot

    # -- ordering fence ---------------------------------------------------------
    def fence(self, sig_idx: int) -> None:
        """Group-wide ordering fence. ``dist.barrier`` (default) or a device-stream GIN signal fence.

        The signal fence is a symmetric all-to-all ``ncclSignal`` + ``ncclWaitSignal`` on a per-``sig_idx``
        monotonically increasing epoch (all ranks run in lockstep, so the expected op-count matches). It
        is enqueued on the current CUDA stream, so unlike ``dist.barrier`` it can be captured in a graph.
        """
        if not dist.is_initialized() or self.world <= 1:
            return
        if self._fence_mode != "signal":
            dist.barrier(self.group)
            return
        cnt = self._epoch.get(sig_idx, 0) + 1
        self._epoch[sig_idx] = cnt
        for p in range(self.world):
            if p != self.my_rank:
                self._gin.signal(p, sig_idx=sig_idx)
        for p in range(self.world):
            if p != self.my_rank:
                self._gin.wait_signal(p, sig_idx=sig_idx, op_cnt=cnt)

    # -- byte helpers -----------------------------------------------------------
    def _copy_into(self, buf: torch.Tensor, off_bytes: int, src: torch.Tensor) -> None:
        b = src.detach().contiguous().view(torch.uint8).reshape(-1)
        buf.narrow(0, off_bytes, b.numel()).copy_(b)

    # -- forward / backward transport ------------------------------------------
    def materialize(
        self,
        plan_x: torch.Tensor,
        weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, ...]:
        """Return the per-slot stacked weights ``(W_j[n_slot, *shape_j], ...)`` (autograd-aware, no D2H)."""
        slot_to_e, main_of_slot = self.slot_tables(plan_x)
        main_expert_ids = tuple(sorted(weights_local.keys()))
        flat_params: List[torch.Tensor] = [weights_local[e][j] for e in main_expert_ids for j in range(self.J)]
        return _GinReplicate.apply(self, slot_to_e, main_of_slot, main_expert_ids, *flat_params)


class _GinReplicate(torch.autograd.Function):
    """Device-initiated replica materialisation; backward reduces grads to ``main(e)``. Zero D2H on schedule."""

    @staticmethod
    def forward(ctx, repl: GinWeightReplicator, slot_to_e, main_of_slot, main_expert_ids, *flat_params):
        J, wb, nb = repl.J, repl.wb, repl.n_slot

        # 1) publish this rank's main weights into the symmetric home buffers.
        #    (static host loop over the experts this rank owns -- not data-dependent, no D2H)
        for idx, e in enumerate(main_expert_ids):
            base = repl.local_slot_of_e[e]
            for j in range(J):
                repl._copy_into(repl.home[j], base * wb[j], flat_params[idx * J + j])
        repl.fence(0)  # all homes globally visible before any get

        # 2) device-resident schedule; batched GIN get for replica slots, on-device gather for local slots
        peers, ls, is_local = _replica_schedule(slot_to_e, main_of_slot, repl.local_slot_dev, repl.my_rank)
        for j in range(J):
            repl.slot[j].zero_()
            remote_off = ls * wb[j]
            local_off = repl.slot_arange * wb[j]
            nbytes = torch.full((nb,), wb[j], dtype=torch.int64, device=slot_to_e.device)
            repl._gin.get_batched(
                repl.home[j], repl.slot[j], remote_off, local_off, nbytes, peers,
                blocks_per_desc=repl.grid,
            )
            # local (main == my_rank) slots: vectorised on-device copy home -> slot (no GIN self-read)
            slot_view = repl.slot[j].view(nb, wb[j])
            home_view = repl.home[j].view(repl.home_cap, wb[j]) if repl.home_cap > 0 else repl.home[j].view(0, wb[j])
            slot_view[is_local] = home_view[ls[is_local]]
        repl.fence(1)  # all peers finished reading my home before it is recycled next step

        out = tuple(
            repl.slot[j].view(repl.dtype).reshape(nb, *repl.weight_shapes[j]).clone()
            for j in range(J)
        )
        ctx.repl = repl
        ctx.slot_to_e = slot_to_e
        ctx.main_of_slot = main_of_slot
        ctx.main_expert_ids = main_expert_ids
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        repl: GinWeightReplicator = ctx.repl
        slot_to_e, main_of_slot, main_expert_ids = ctx.slot_to_e, ctx.main_of_slot, ctx.main_expert_ids
        J, wb, nb, world = repl.J, repl.wb, repl.n_slot, repl.world

        peers, ls, is_local = _replica_schedule(slot_to_e, main_of_slot, repl.local_slot_dev, repl.my_rank)
        # destination column in main(e)'s scratch: (local_slot(e), source_rank == my_rank)
        col = ls * world + repl.my_rank                                   # [S] in units of one weight

        for j in range(J):
            repl.scratch[j].zero_()
            # stage every slot's grad into the recyclable symmetric slot buffer (bytes)
            g_bytes = grad_out[j].detach().contiguous().view(torch.uint8).reshape(nb, wb[j])
            slot_view = repl.slot[j].view(nb, wb[j])
            slot_view.copy_(g_bytes)
            src_off = repl.slot_arange * wb[j]
            dst_off = col * wb[j]
            nbytes = torch.full((nb,), wb[j], dtype=torch.int64, device=slot_to_e.device)
            # replica slots: batched GIN put grad -> main(e)'s scratch column (peers<0 skipped on device)
            repl._gin.put_batched(
                repl.slot[j], repl.scratch[j], src_off, dst_off, nbytes, peers,
                blocks_per_desc=repl.grid,
            )
            # local slots: vectorised on-device copy grad -> my own scratch column
            scratch_view = repl.scratch[j].view(repl.home_cap * world, wb[j]) if repl.home_cap > 0 \
                else repl.scratch[j].view(0, wb[j])
            scratch_view[col[is_local]] = g_bytes[is_local]
        repl.fence(2)  # all remote grad puts landed before local sum

        # main(e) sums its source columns -> gradient of its own parameter (static host loop over owned e)
        grads: List[torch.Tensor] = []
        for e in main_expert_ids:
            base = repl.local_slot_of_e[e]
            for j in range(J):
                seg = repl.scratch[j].narrow(0, base * world * wb[j], world * wb[j])
                stacked = seg.view(repl.dtype).reshape(world, *repl.weight_shapes[j])
                grads.append(stacked.sum(dim=0))
        return (None, None, None, None, *grads)

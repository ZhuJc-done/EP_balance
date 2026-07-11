"""Device-initiated EPLB weight replication over NCCL GIN (symmetric memory)."""

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
        self.local_slot_of_e: List[int] = [0] * self.E
        for e in range(self.E):
            m = self.main_host[e]
            self.local_slot_of_e[e] = counts[m]
            counts[m] += 1
        self.home_cap = max(counts) if counts else 0  # max experts any rank is main of

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
    def slot_schedule(self, x: torch.Tensor) -> Tuple[List[int], List[int]]:
        """One consolidated D2H: ``(slot_e[n_slot], slot_main[n_slot])`` for ``my_rank``.

        ``slot_e[s]`` is the logical expert in local slot ``s`` (-1 if empty), placed in
        ascending expert-id order; ``slot_main[s]`` is ``main(slot_e[s])`` (-1 if empty).
        """
        from .eplb_manager import _slot_tables  # reuse the sync-free cumsum/scatter tables

        slot_to_e, _, _ = _slot_tables(x, self.my_rank, self.n_slot)  # device, -1 where empty
        main_tbl = torch.as_tensor(self.main_host, dtype=torch.int64, device=x.device)
        main_of_slot = torch.where(
            slot_to_e >= 0,
            main_tbl[slot_to_e.clamp(min=0)],
            torch.full((self.n_slot,), -1, dtype=torch.int64, device=x.device),
        )
        pair = torch.stack([slot_to_e, main_of_slot]).tolist()  # single consolidated D2H
        return pair[0], pair[1]

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
        """Return the per-slot stacked weights ``(W_j[n_slot, *shape_j], ...)`` (autograd-aware)."""
        slot_e, slot_main = self.slot_schedule(plan_x)
        main_expert_ids = sorted(weights_local.keys())
        flat_params: List[torch.Tensor] = [weights_local[e][j] for e in main_expert_ids for j in range(self.J)]
        return _GinReplicate.apply(self, tuple(slot_e), tuple(slot_main), tuple(main_expert_ids), *flat_params)


class _GinReplicate(torch.autograd.Function):
    """Device-initiated replica materialisation; backward reduces grads to ``main(e)``."""

    @staticmethod
    def forward(ctx, repl: GinWeightReplicator, slot_e, slot_main, main_expert_ids, *flat_params):
        J, wb, nb = repl.J, repl.wb, repl.n_slot

        # 1) publish this rank's main weights into the symmetric home buffers
        for idx, e in enumerate(main_expert_ids):
            base = repl.local_slot_of_e[e]
            for j in range(J):
                repl._copy_into(repl.home[j], base * wb[j], flat_params[idx * J + j])
        if dist.is_initialized():
            dist.barrier(repl.group)  # all homes globally visible before any get

        # 2) fill each local slot: local copy for a main slot, GIN get for a replica slot
        for j in range(J):
            repl.slot[j].zero_()
        for s in range(nb):
            e = slot_e[s]
            if e < 0:
                continue
            m = slot_main[s]
            for j in range(J):
                loff = s * wb[j]
                roff = repl.local_slot_of_e[e] * wb[j]
                if m == repl.my_rank:
                    repl.slot[j].narrow(0, loff, wb[j]).copy_(repl.home[j].narrow(0, roff, wb[j]))
                else:
                    repl._gin.get(repl.home[j], repl.slot[j], roff, loff, wb[j], m, grid_size=repl.grid)
        if dist.is_initialized():
            dist.barrier(repl.group)  # all peers finished reading my home before it is recycled

        # 3) hand back normal-memory clones so the symmetric slot buffer can be recycled next step
        out = tuple(
            repl.slot[j].view(repl.dtype).reshape(nb, *repl.weight_shapes[j]).clone()
            for j in range(J)
        )
        ctx.repl = repl
        ctx.slot_e = slot_e
        ctx.slot_main = slot_main
        ctx.main_expert_ids = main_expert_ids
        return out

    @staticmethod
    def backward(ctx, *grad_out):
        repl: GinWeightReplicator = ctx.repl
        slot_e, slot_main, main_expert_ids = ctx.slot_e, ctx.slot_main, ctx.main_expert_ids
        J, wb, nb, world = repl.J, repl.wb, repl.n_slot, repl.world

        for j in range(J):
            repl.scratch[j].zero_()

        # each slot's grad -> column (local_slot(e), source_rank) of main(e)'s scratch
        for s in range(nb):
            e = slot_e[s]
            if e < 0:
                continue
            m = slot_main[s]
            for j in range(J):
                col = (repl.local_slot_of_e[e] * world + repl.my_rank) * wb[j]
                gbytes = grad_out[j][s].detach().contiguous().view(torch.uint8).reshape(-1)
                if m == repl.my_rank:
                    repl.scratch[j].narrow(0, col, wb[j]).copy_(gbytes)
                else:
                    # stage into the (recyclable) symmetric slot buffer, then GIN put to main's scratch
                    repl.slot[j].narrow(0, s * wb[j], wb[j]).copy_(gbytes)
                    repl._gin.put(repl.slot[j], repl.scratch[j], s * wb[j], col, wb[j], m,
                                  grid_size=repl.grid)
        if dist.is_initialized():
            dist.barrier(repl.group)  # all remote grad puts landed before local sum

        # main(e) sums the source columns -> gradient of its own parameter
        grads: List[torch.Tensor] = []
        for e in main_expert_ids:
            base = repl.local_slot_of_e[e]
            for j in range(J):
                seg = repl.scratch[j].narrow(0, base * world * wb[j], world * wb[j])
                stacked = seg.view(repl.dtype).reshape(world, *repl.weight_shapes[j])
                grads.append(stacked.sum(dim=0))
        return (None, None, None, None, *grads)

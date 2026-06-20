"""Collect the global load matrix ``Lambda`` with a single all-gather (the solver's only communication)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from .loads import Loads


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_gather_lambda(
    local_row: torch.Tensor,
    group: Optional["dist.ProcessGroup"] = None,
) -> Loads:
    """All-gather each rank's expert-count row into the full ``Lambda`` matrix.

    Args:
        local_row: int64 tensor ``[E]`` -- this rank's token counts per expert.
        group: Optional process group (defaults to the world group).

    Returns:
        :class:`~eplb.loads.Loads` wrapping the ``[R, E]`` matrix, with row order
        equal to global rank order (so every rank sees an identical matrix).
    """
    if local_row.dtype != torch.int64:
        local_row = local_row.to(torch.int64)
    local_row = local_row.contiguous()

    if not is_distributed():
        return Loads(local_row.unsqueeze(0))

    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(local_row) for _ in range(world_size)]
    dist.all_gather(gathered, local_row, group=group)
    lam = torch.stack(gathered, dim=0)  # [R, E], ordered by global rank
    return Loads(lam)


def local_counts_from_routing(
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Build this rank's ``Lambda`` row from a flat list of routed expert ids.

    Args:
        expert_ids: int tensor of the expert id chosen for each (token, top-k)
            slot handled by this rank.
        num_experts: ``E``.

    Returns:
        int64 ``[E]`` token counts per expert.
    """
    return torch.bincount(
        expert_ids.to(torch.int64).flatten(), minlength=num_experts
    ).to(torch.int64)

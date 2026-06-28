"""Sync-free bridge mapping each routing unit to a physical-instance id per ``plan.q`` (LPLB-style redirect for DeepEP)."""

from __future__ import annotations

from typing import Tuple

import torch

from ..plan import Plan
from ..problem import ProblemSpec


def build_phys_slot_table(x: torch.Tensor, n_slot: int) -> torch.Tensor:
    """Map each hosted ``(expert e, rank r)`` to a physical id ``r * n_slot + slot(e, r)``.

    Args:
        x: int8/int64 ``[E, R]`` placement table (1 = instance present).
        n_slot: Per-rank slot budget ``N_slot`` (host-known constant).

    Returns:
        int64 ``[E, R]`` physical-id table.
    """
    E, R = x.shape
    xi = x.to(torch.int64)
    slot = xi.cumsum(dim=0) - 1  # slot index of e on r (valid where hosted)
    rank_ids = torch.arange(R, device=x.device, dtype=torch.int64).view(1, R)
    return rank_ids * int(n_slot) + slot


def assign_physical(
    unit_expert: torch.Tensor,
    plan: Plan,
    spec: ProblemSpec,
    src_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each routing unit on ``src_rank`` to a physical instance, sync-free (matches ``assign_unit_dst``).

    Args:
        unit_expert: int64 ``[U]`` logical expert id of each routing unit (``U = N*topk``).
        plan: The solved plan (global ``x`` / ``q``); identical on every rank.
        spec: Problem spec (provides ``num_experts`` and ``n_slot``).
        src_rank: This rank's group-local id (selects the ``plan.q[src_rank]`` row).

    Returns:
        ``(phys_id, dst_rank)`` int64 ``[U]`` device tensors: the physical instance id and
        the destination (group-local) rank of each unit.
    """
    device = unit_expert.device
    E = int(spec.num_experts)
    R = int(plan.num_ranks)
    n_slot = int(spec.n_slot)
    U = int(unit_expert.shape[0])  # static (= N * topk)
    r = int(src_rank)

    x = plan.x
    q_r = plan.q[r].to(torch.int64)  # per-(e, dst) counts from this rank
    phys_table = build_phys_slot_table(x, n_slot)

    # build the (e asc, host asc) assignment of length U via searchsorted on prefix sums (no repeat_interleave sync)
    counts_flat = q_r.reshape(-1)
    phys_flat = phys_table.reshape(-1)
    rank_flat = (
        torch.arange(R, device=device, dtype=torch.int64).view(1, R).expand(E, R).reshape(-1)
    )
    prefix = torch.cumsum(counts_flat, dim=0)
    pos = torch.arange(U, device=device, dtype=torch.int64)
    pair_idx = torch.searchsorted(prefix, pos, right=True)  # which (e, host) pair each slot belongs to
    assign_phys_sorted = phys_flat[pair_idx]
    assign_rank_sorted = rank_flat[pair_idx]

    # units grouped by expert (stable sort) align with the concatenation; scatter back to original order
    order = torch.argsort(unit_expert, stable=True)
    phys_id = torch.empty(U, dtype=torch.int64, device=device)
    dst_rank = torch.empty(U, dtype=torch.int64, device=device)
    phys_id[order] = assign_phys_sorted
    dst_rank[order] = assign_rank_sorted
    return phys_id, dst_rank

"""Vendored FlexMoE dynamic device-placement load balancer (SIGMOD'23, Article 110).

Adapted from *FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via
Dynamic Device Placement* (Nie et al.).  Only the *load-balancing* decision is
reproduced -- FlexMoE never modifies token routing (so model quality is
untouched); instead it changes the expert->device mapping ``P`` by giving hot
experts more GPU replicas and reclaiming replicas from cold experts.

The unit of management is a **vExpert**: every rank exposes ``n_slot`` vExpert
slots (``|G| * n_slot`` in total), each expert ``e`` is allocated ``n_e`` vExperts,
and its load is split evenly so each vExpert carries ``cap_e = I_e / n_e`` tokens.

* ``flexmoe_schedule`` implements the Scheduler + Policy Maker greedy (paper
  Algorithms 1-2): while the balance ratio (Eq. 6, ``max_g load_g / mean_g``)
  exceeds ``threshold``, repeatedly Expand the expert with the largest ``cap_e``
  ("rob the rich to help the poor"), as long as the cost model predicts a faster
  step (Eq. 5, dominated by the compute term ``T_C = I / TPS``).
* ``_balanced_pack`` places the resulting vExperts onto ranks (an LPT bin-pack,
  capturing the effect of the Migrate primitive that clusters replicas), yielding
  the realized per-rank load used for the balance ratio.

The profiled constants of the paper (``TPS``, per-link bandwidth, all-reduce
``BPS``) are replaced by the harness's own quantities; only the *relative* cost
comparison drives the decision, so the placement is preserved.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch


@dataclass
class FlexMoECostModel:
    """FlexMoE cost-model + trigger constants (paper Eqs. 5-9, Algorithms 1-2)."""

    tps: float = 1.0e6  # Eq. 7: tokens-per-second of one vExpert (compute rate)
    sync_bps: float = 50e9 / 8  # Eq. 9: all-reduce bytes-per-second across replicas
    sync_weight: float = 0.0  # weight of the T_Sync replication penalty in the decision
    threshold: float = 1.2  # Eq. 6 balance-ratio trigger for the Scheduler (Algorithm 1)


def _balanced_pack(
    cap_e: torch.Tensor,
    counts: list[int],
    num_ranks: int,
    n_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LPT-pack the vExperts onto ranks, each rank holding at most ``n_slot`` of them.

    Expert ``e`` contributes ``counts[e]`` vExperts, each weighing ``cap_e[e]``.
    Every vExpert is placed on the least-loaded rank that still has a free slot.
    """
    num_experts = len(counts)
    items: list[tuple[float, int]] = []
    for e in range(num_experts):
        w = float(cap_e[e].item())
        items.extend((w, e) for _ in range(counts[e]))
    items.sort(key=lambda t: t[0], reverse=True)

    rank_load = torch.zeros(num_ranks, dtype=torch.float64)
    placement = torch.zeros((num_experts, num_ranks), dtype=torch.int64)
    # min-heap keyed by (load, rank); only ranks with a free slot are present.
    heap = [(0.0, g) for g in range(num_ranks)]
    heapq.heapify(heap)
    free = [n_slot] * num_ranks

    for w, e in items:
        load, g = heapq.heappop(heap)
        rank_load[g] = load + w
        placement[e, g] = 1
        free[g] -= 1
        if free[g] > 0:
            heapq.heappush(heap, (load + w, g))
    return rank_load, placement


def flexmoe_schedule(
    omega: torch.Tensor,
    weight_bytes: torch.Tensor,
    num_ranks: int,
    n_slot: int,
    cost: FlexMoECostModel | None = None,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Run the FlexMoE Scheduler/Policy-Maker greedy on the current load.

    Args:
        omega: int ``[R, E]`` load matrix, ``Ω[r, e]`` tokens from rank ``r``
            to expert ``e`` (this repository's :class:`~eplb.loads.Loads.omega`).
        weight_bytes: numeric ``[E]`` byte size of each expert's parameters
            (drives the ``T_Sync`` replication penalty, paper Eq. 9).
        num_ranks: number of ranks/GPUs ``|G|``.
        n_slot: vExpert slots per rank; the total slot budget is ``num_ranks * n_slot``.
        cost: cost-model + trigger constants; defaults to :class:`FlexMoECostModel`.

    Returns:
        ``(n_e, rank_load, placement)``: per-expert vExpert counts, the realized
        float ``[R]`` per-rank load, and a binary ``[E, R]`` presence matrix.
    """
    cost = cost or FlexMoECostModel()

    expert_load = (
        omega.detach().to(device="cpu", dtype=torch.float64).sum(dim=0)
    )  # I_e
    w_bytes = weight_bytes.detach().to(device="cpu", dtype=torch.float64)
    num_experts = expert_load.numel()

    total_slots = num_ranks * n_slot
    if total_slots < num_experts:
        raise ValueError("num_ranks * n_slot must be >= number of experts")

    total = float(expert_load.sum().item())
    mean = total / num_ranks if num_ranks else 0.0

    n = [1] * num_experts  # every expert starts with one vExpert (uses E slots)
    cap = expert_load.clone()  # cap_e = I_e / n_e, initially I_e
    slots_left = total_slots - num_experts
    sync_bytes = 0.0

    def objective(max_cap: float, sync_total: float) -> float:
        # Eq. 5 dominated by compute; the slowest GPU is >= max(max_cap, mean).
        compute = max(max_cap, mean) / cost.tps
        return compute + cost.sync_weight * sync_total / cost.sync_bps

    # Scheduler (Alg. 1) + Policy Maker (Alg. 2): expand the hottest vExpert while
    # the balance ratio is above threshold and the cost model predicts a win.
    while slots_left > 0 and mean > 0.0:
        max_cap = float(cap.max().item())
        if max_cap / mean <= cost.threshold:
            break
        e0 = int(torch.argmax(cap).item())

        t0 = objective(max_cap, sync_bytes)
        prev = float(cap[e0].item())
        cap[e0] = expert_load[e0] / (n[e0] + 1)
        t1 = objective(float(cap.max().item()), sync_bytes + float(w_bytes[e0].item()))

        if t1 < t0:
            n[e0] += 1
            sync_bytes += float(w_bytes[e0].item())
            slots_left -= 1
        else:  # diminishing returns: Policy Maker returns an empty plan -> stop.
            cap[e0] = prev
            break

    rank_load, placement = _balanced_pack(cap, n, num_ranks, n_slot)
    return n, rank_load, placement


__all__ = ["FlexMoECostModel", "flexmoe_schedule"]

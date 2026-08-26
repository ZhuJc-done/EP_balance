"""Logical token-flow statistics before and after an EPLB routing plan.

``omega[src, expert]`` records the router's logical expert assignments.  A
runtime :class:`~eplb.plan.Plan` turns each assignment into a physical
destination through ``q[src, expert, dst]``.  Comparing those tensors separates
load balancing from the communication it induces: local execution, remote
traffic inside one NVLink domain, and traffic crossing domains.
"""

from __future__ import annotations

from typing import Any

import torch


def _max_mean(values: torch.Tensor) -> float:
    values = values.to(torch.float64)
    mean = values.mean()
    return float(values.max() / mean) if mean > 0 else 0.0


def _domain_flow(rank_flow: torch.Tensor, domain_of_rank: torch.Tensor) -> torch.Tensor:
    num_domains = int(domain_of_rank.max().item()) + 1 if domain_of_rank.numel() else 0
    out = torch.zeros((num_domains, num_domains), dtype=torch.int64)
    for source in range(rank_flow.shape[0]):
        for destination in range(rank_flow.shape[1]):
            out[domain_of_rank[source], domain_of_rank[destination]] += rank_flow[
                source, destination
            ]
    return out


def compute_routing_stats(
    omega: torch.Tensor,
    q: torch.Tensor,
    main_rank: torch.Tensor,
    domain_of_rank: torch.Tensor,
    *,
    s_tok: int,
    cost: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return exact assignment, load, topology and logical-byte statistics.

    Byte counts use one hidden-state row of ``s_tok`` bytes and therefore omit
    transport metadata, padding and protocol headers.  ``training_*_bytes``
    multiplies one-way dispatch volume by four for forward dispatch/combine and
    their two backward transposes.
    """

    omega = torch.as_tensor(omega, dtype=torch.int64, device="cpu")
    q = torch.as_tensor(q, dtype=torch.int64, device="cpu")
    main_rank = torch.as_tensor(main_rank, dtype=torch.int64, device="cpu")
    domain_of_rank = torch.as_tensor(
        domain_of_rank, dtype=torch.int64, device="cpu"
    )
    num_ranks, num_experts = omega.shape
    if q.shape != (num_ranks, num_experts, num_ranks):
        raise ValueError(
            f"q shape {tuple(q.shape)} != "
            f"({num_ranks}, {num_experts}, {num_ranks})"
        )
    if main_rank.shape != (num_experts,):
        raise ValueError(
            f"main_rank shape {tuple(main_rank.shape)} != ({num_experts},)"
        )
    if domain_of_rank.shape != (num_ranks,):
        raise ValueError(
            f"domain_of_rank shape {tuple(domain_of_rank.shape)} != ({num_ranks},)"
        )
    if int(s_tok) <= 0:
        raise ValueError("s_tok must be positive")
    if torch.any(omega < 0) or torch.any(q < 0):
        raise ValueError("routing counts must be non-negative")
    if not torch.equal(q.sum(dim=2), omega):
        raise ValueError("q does not conserve omega over destination ranks")
    if torch.any(main_rank < 0) or torch.any(main_rank >= num_ranks):
        raise ValueError("main_rank contains an out-of-range rank")
    if torch.any(domain_of_rank < 0):
        raise ValueError("domain ids must be non-negative")

    source = torch.arange(num_ranks, dtype=torch.int64).view(num_ranks, 1, 1)
    destination = torch.arange(num_ranks, dtype=torch.int64).view(1, 1, num_ranks)
    source_domain = domain_of_rank.view(num_ranks, 1, 1)
    destination_domain = domain_of_rank.view(1, 1, num_ranks)
    same_rank = source == destination
    same_domain = source_domain == destination_domain

    planned_local = int(q.masked_select(same_rank).sum())
    planned_intra = int(q.masked_select(same_domain & ~same_rank).sum())
    planned_inter = int(q.masked_select(~same_domain).sum())

    source_2d = torch.arange(num_ranks, dtype=torch.int64).view(num_ranks, 1)
    main_2d = main_rank.view(1, num_experts)
    baseline_same_rank = source_2d == main_2d
    baseline_same_domain = (
        domain_of_rank.view(num_ranks, 1)
        == domain_of_rank[main_rank].view(1, num_experts)
    )
    baseline_local = int(omega.masked_select(baseline_same_rank).sum())
    baseline_intra = int(
        omega.masked_select(baseline_same_domain & ~baseline_same_rank).sum()
    )
    baseline_inter = int(omega.masked_select(~baseline_same_domain).sum())

    rerouted_mask = destination != main_rank.view(1, num_experts, 1)
    rerouted = int(q.masked_select(rerouted_mask).sum())
    rerouted_local = int(q.masked_select(rerouted_mask & same_rank).sum())
    rerouted_intra = int(
        q.masked_select(rerouted_mask & same_domain & ~same_rank).sum()
    )
    rerouted_inter = int(q.masked_select(rerouted_mask & ~same_domain).sum())

    baseline_inter_3d = (~baseline_same_domain).unsqueeze(2)
    baseline_same_3d = baseline_same_domain.unsqueeze(2)
    avoided_inter = int(q.masked_select(baseline_inter_3d & same_domain).sum())
    remaining_inter = int(q.masked_select(baseline_inter_3d & ~same_domain).sum())
    introduced_inter = int(q.masked_select(baseline_same_3d & ~same_domain).sum())

    rank_flow = q.sum(dim=1)
    baseline_rank_flow = torch.zeros(
        (num_ranks, num_ranks), dtype=torch.int64
    )
    baseline_rank_flow.scatter_add_(
        1, main_rank.view(1, num_experts).expand(num_ranks, -1), omega
    )
    planned_rank_load = rank_flow.sum(dim=0)
    baseline_rank_load = baseline_rank_flow.sum(dim=0)
    total = int(omega.sum())
    excess = _max_mean(baseline_rank_load) - 1.0
    planned_imbalance = _max_mean(planned_rank_load)
    absorbed = (
        (_max_mean(baseline_rank_load) - planned_imbalance) / excess
        if excess > 1e-12
        else 0.0
    )

    stats: dict[str, Any] = {
        "total_assignments": total,
        "baseline_local_assignments": baseline_local,
        "baseline_intra_domain_assignments": baseline_intra,
        "baseline_inter_domain_assignments": baseline_inter,
        "planned_local_assignments": planned_local,
        "planned_intra_domain_assignments": planned_intra,
        "planned_inter_domain_assignments": planned_inter,
        "rerouted_assignments": rerouted,
        "rerouted_local_assignments": rerouted_local,
        "rerouted_intra_domain_assignments": rerouted_intra,
        "rerouted_inter_domain_assignments": rerouted_inter,
        "avoided_inter_domain_assignments": avoided_inter,
        "remaining_inter_domain_assignments": remaining_inter,
        "introduced_inter_domain_assignments": introduced_inter,
        "baseline_rank_max_mean": _max_mean(baseline_rank_load),
        "planned_rank_max_mean": planned_imbalance,
        "absorbed_excess_load": absorbed,
        "rerouted_fraction": rerouted / total if total else 0.0,
        "baseline_inter_domain_fraction": baseline_inter / total if total else 0.0,
        "planned_inter_domain_fraction": planned_inter / total if total else 0.0,
        "inter_domain_reduction_fraction": (
            (baseline_inter - planned_inter) / baseline_inter
            if baseline_inter
            else 0.0
        ),
        "baseline_one_way_remote_bytes": (baseline_intra + baseline_inter)
        * int(s_tok),
        "baseline_one_way_inter_domain_bytes": baseline_inter * int(s_tok),
        "planned_one_way_remote_bytes": (planned_intra + planned_inter)
        * int(s_tok),
        "planned_one_way_inter_domain_bytes": planned_inter * int(s_tok),
        "baseline_training_remote_bytes": 4
        * (baseline_intra + baseline_inter)
        * int(s_tok),
        "baseline_training_inter_domain_bytes": 4 * baseline_inter * int(s_tok),
        "planned_training_remote_bytes": 4
        * (planned_intra + planned_inter)
        * int(s_tok),
        "planned_training_inter_domain_bytes": 4 * planned_inter * int(s_tok),
        "baseline_rank_load": baseline_rank_load,
        "planned_rank_load": planned_rank_load,
        "baseline_rank_flow": baseline_rank_flow,
        "planned_rank_flow": rank_flow,
        "baseline_domain_flow": _domain_flow(
            baseline_rank_flow, domain_of_rank
        ),
        "planned_domain_flow": _domain_flow(rank_flow, domain_of_rank),
    }
    if cost is not None:
        cost = torch.as_tensor(cost, dtype=torch.int64, device="cpu")
        if cost.shape != (num_ranks, num_ranks):
            raise ValueError(
                f"cost shape {tuple(cost.shape)} != ({num_ranks}, {num_ranks})"
            )
        baseline_cost = (
            omega * cost.gather(1, main_2d.expand(num_ranks, -1))
        ).sum()
        planned_cost = (q * cost.view(num_ranks, 1, num_ranks)).sum()
        stats["baseline_topology_cost"] = int(baseline_cost)
        stats["planned_topology_cost"] = int(planned_cost)
        stats["topology_cost_reduction_fraction"] = (
            float((baseline_cost - planned_cost).to(torch.float64) / baseline_cost)
            if baseline_cost
            else 0.0
        )
    return stats


__all__ = ["compute_routing_stats"]

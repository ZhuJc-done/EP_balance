"""Build a runtime :class:`Plan` from an externally selected expert placement."""

from __future__ import annotations

import torch

from .config import EPLBConfig
from .loads import Loads
from .plan import Plan
from .problem import ProblemSpec
from .topology import Topology


def _validate_placement(
    placement: torch.Tensor,
    loads: Loads,
    topology: Topology,
    spec: ProblemSpec,
) -> None:
    """Validate the placement invariants required by the shared data plane."""
    num_ranks = topology.num_ranks
    num_experts = spec.num_experts
    if placement.shape != (num_experts, num_ranks):
        raise ValueError(
            "placement must have shape "
            f"[{num_experts}, {num_ranks}], got {tuple(placement.shape)}"
        )
    if placement.device != loads.device:
        raise ValueError(
            f"placement and loads must share a device, got {placement.device} "
            f"and {loads.device}"
        )
    if not torch.all((placement == 0) | (placement == 1)):
        raise ValueError("placement must be binary")
    if not torch.all(placement.sum(dim=1) >= 1):
        raise ValueError("every logical expert must have at least one instance")
    if not torch.all(placement.sum(dim=0) <= int(spec.n_slot)):
        raise ValueError(
            f"placement exceeds the per-rank N_slot={int(spec.n_slot)} budget"
        )
    expert = torch.arange(num_experts, device=loads.device, dtype=torch.int64)
    main_rank = spec.main_rank.to(device=loads.device, dtype=torch.int64)
    if not torch.all(placement[expert, main_rank] == 1):
        raise ValueError("placement must preserve every main expert (C7)")


def plan_from_placement(
    loads: Loads,
    topology: Topology,
    spec: ProblemSpec,
    placement: torch.Tensor,
    cfg: EPLBConfig | None = None,
    *,
    validate: bool = True,
) -> Plan:
    """Construct deterministic quotas for a fixed binary placement.

    A source rank uses only instances in its own topology domain when at least
    one exists, matching Scale-EPLB's strict domain-local serving policy.  Its
    ``Omega[src, expert]`` tokens are then split evenly, by ascending destination
    rank, over as many legal instances as the quota floor permits.

    CUDA inputs use the same compiled Update Routing kernel as Scale-EPLB;
    CPU inputs use a tensor reference. In particular, ``validate=False``
    performs no scalar extraction or implicit CUDA host sync.
    """
    cfg = cfg or EPLBConfig()
    num_ranks = topology.num_ranks
    num_experts = spec.num_experts

    if validate:
        topology.validate()
        spec.validate(num_ranks)
        loads.validate(num_ranks, num_experts)

    x = placement.to(device=loads.device, dtype=torch.int8).contiguous()
    if validate:
        _validate_placement(x, loads, topology, spec)
    if loads.omega.is_cuda:
        from .cuda_solve import plan_fixed_cuda

        return plan_fixed_cuda(loads, topology, x, cfg)

    omega = loads.omega.to(dtype=torch.int64)
    domain = topology.domain_of_rank.to(device=loads.device, dtype=torch.int64)
    hosts = x.to(torch.bool)
    quota = torch.zeros(
        (num_ranks, num_experts, num_ranks),
        dtype=torch.int64,
        device=loads.device,
    )
    quota_floor = int(cfg.u_min)
    if quota_floor <= 0:
        raise ValueError("cfg.u_min must be positive")

    # Looping over sources keeps temporary tensors at O(E*R), instead of
    # materialising an additional O(R*E*R) host/domain mask beside Q.
    destination_domain = domain.view(1, num_ranks)
    for source in range(num_ranks):
        local = hosts & (destination_domain == domain[source])
        active = torch.where(local.any(dim=1, keepdim=True), local, hosts)
        position = active.to(torch.int64).cumsum(dim=1)
        active_count = position[:, -1:].clamp_min(1)
        need = omega[source].view(num_experts, 1)

        if quota_floor > 1:
            floor_count = torch.div(
                need, quota_floor, rounding_mode="floor"
            ).clamp_min(1)
            destination_count = torch.where(
                need >= quota_floor,
                torch.minimum(active_count, floor_count),
                torch.ones_like(active_count),
            )
        else:
            destination_count = active_count

        selected = active & (position <= destination_count)
        base = torch.div(need, destination_count, rounding_mode="floor")
        remainder = torch.remainder(need, destination_count)
        quota[source] = torch.where(
            selected,
            base + (position <= remainder).to(torch.int64),
            torch.zeros((), dtype=torch.int64, device=loads.device),
        )

    rank_load = quota.sum(dim=(0, 1))
    theta = rank_load.amax() if num_ranks else torch.zeros(
        (), dtype=torch.int64, device=loads.device
    )
    return Plan(x=x, q=quota, theta=theta)


__all__ = ["plan_from_placement"]

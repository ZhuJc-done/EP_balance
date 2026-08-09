"""Common adapters and quality metrics for load-balancer comparisons."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.loads import Loads

from .deepseek_eplb import rebalance_experts
from .fastermoe import ShadowCostModel, select_shadow_experts
from .flexmoe import FlexMoECostModel, flexmoe_schedule


@dataclass
class BaselineResult:
    """A strategy result reduced to comparable placement/load metrics."""

    name: str
    rank_load: torch.Tensor
    placement: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def theta(self) -> float:
        return float(self.rank_load.max().item()) if self.rank_load.numel() else 0.0

    @property
    def mean_load(self) -> float:
        return float(self.rank_load.double().mean().item()) if self.rank_load.numel() else 0.0

    @property
    def imbalance(self) -> float:
        mean = self.mean_load
        return self.theta / mean if mean > 0 else 1.0


class LPLBUnavailableError(RuntimeError):
    """The optional compiled LPLB package is not importable."""


def _slot_budget(
    num_experts: int, num_ranks: int, n_slot: int
) -> tuple[int, int, int]:
    """Return ``(main, replica, physical)`` slots per rank.

    The baseline-facing ``n_slot`` has one uniform meaning: the number of
    *additional expert replica* slots available on every rank. Logical experts
    are placed evenly, so each rank also owns ``num_experts / num_ranks`` mains.
    """
    if num_ranks <= 0 or num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_ranks")
    if int(n_slot) < 0:
        raise ValueError("n_slot must be non-negative")
    main_slots = num_experts // num_ranks
    replica_slots = int(n_slot)
    return main_slots, replica_slots, main_slots + replica_slots


def _budget_metadata(
    num_experts: int,
    num_ranks: int,
    n_slot: int,
    physical_instances: int,
) -> dict[str, int]:
    main_slots, replica_slots, physical_slots = _slot_budget(
        num_experts, num_ranks, n_slot
    )
    return {
        "main_slots_per_rank": main_slots,
        "replica_slots_per_rank": replica_slots,
        "physical_slots_per_rank": physical_slots,
        "replica_budget": num_ranks * replica_slots,
        "physical_instances": int(physical_instances),
        "replicas": max(0, int(physical_instances) - num_experts),
    }


def run_scale_eplb(
    loads: Loads,
    topology: Topology,
    spec: ProblemSpec,
    cfg: EPLBConfig | None = None,
) -> BaselineResult:
    """Run this repository's exact placement-and-quota solver."""
    plan = solve(loads, topology, spec, cfg or EPLBConfig(), validate=False)
    main_slots, _, _ = _slot_budget(
        spec.num_experts, topology.num_ranks, 0
    )
    replica_slots = int(spec.n_slot) - main_slots
    physical_instances = int(plan.x.sum().item())
    return BaselineResult(
        name="scale-eplb",
        rank_load=plan.rank_load(),
        placement=plan.x,
        metadata={
            "load_kind": "realized quota load",
            "placement_kind": "binary expert presence",
            **_budget_metadata(
                spec.num_experts,
                topology.num_ranks,
                replica_slots,
                physical_instances,
            ),
        },
    )


def run_deepseek_eplb(
    loads: Loads,
    *,
    num_nodes: int,
    num_gpus: int,
    n_slot: int,
    num_groups: int,
    placement_loads: Loads | None = None,
) -> BaselineResult:
    """Run DeepSeek EPLB and report its own equal-split predicted load.

    DeepSeek EPLB does not produce source-to-destination quotas.  Its quality
    metric therefore follows the algorithm's assumption that an expert's load
    is divided equally among all of its replicas. ``n_slot`` is the common
    per-rank *additional replica* budget, excluding main experts.
    """
    num_experts = loads.num_experts
    _, _, physical_slots = _slot_budget(num_experts, num_gpus, n_slot)
    num_physical = num_experts + num_gpus * int(n_slot)

    current_load = loads.expert_load().cpu()
    placement_load = (
        placement_loads.expert_load().cpu()
        if placement_loads is not None
        else current_load
    )
    phy2log, _, logcnt = rebalance_experts(
        placement_load.unsqueeze(0),
        num_physical,
        num_groups,
        num_nodes,
        num_gpus,
    )
    phy2log = phy2log[0]
    logcnt = logcnt[0]
    per_physical = current_load.double() / logcnt.double()
    rank_load = per_physical[phy2log].view(num_gpus, physical_slots).sum(dim=1)
    placement = torch.zeros(
        (num_experts, num_gpus), dtype=torch.int64, device=phy2log.device
    )
    ranks = torch.arange(num_gpus, dtype=torch.int64).repeat_interleave(
        physical_slots
    )
    placement.index_put_(
        (phy2log, ranks), torch.ones_like(phy2log), accumulate=True
    )
    return BaselineResult(
        name="deepseek-eplb",
        rank_load=rank_load,
        placement=placement,
        metadata={
            "load_kind": "ideal equal-split predicted load",
            "placement_kind": "physical replica count",
            "num_groups": num_groups,
            **_budget_metadata(
                num_experts, num_gpus, n_slot, num_physical
            ),
        },
    )


def run_fastermoe(
    loads: Loads,
    spec: ProblemSpec,
    *,
    num_ranks: int,
    n_slot: int,
    cost: ShadowCostModel | None = None,
) -> BaselineResult:
    """Run FasterMoE dynamic shadowing and report its realized per-rank load.

    FasterMoE re-selects the shadowed (globally replicated) experts every
    micro-batch from the current load, so the reported load is the token load
    after the chosen hot experts have been spread back onto their source ranks.
    """
    shadow_mask, rank_load = select_shadow_experts(
        loads.omega,
        spec.main_rank,
        spec.weight_bytes,
        spec.s_tok,
        num_ranks,
        cost,
        n_slot,
    )
    placement = torch.zeros(
        (spec.num_experts, num_ranks), dtype=torch.int64
    )
    experts = torch.arange(spec.num_experts, dtype=torch.int64)
    placement[experts, spec.main_rank.cpu()] = 1
    # A shadowed expert is transiently replicated onto every rank.
    placement[shadow_mask] = 1
    physical_instances = int(placement.sum().item())
    return BaselineResult(
        name="fastermoe",
        rank_load=rank_load,
        placement=placement,
        metadata={
            "load_kind": "shadow-redistributed token load",
            "placement_kind": "binary expert presence (shadow=all ranks)",
            "shadowed_experts": int(shadow_mask.sum().item()),
            **_budget_metadata(
                spec.num_experts, num_ranks, n_slot, physical_instances
            ),
        },
    )


def run_flexmoe(
    loads: Loads,
    spec: ProblemSpec,
    *,
    num_ranks: int,
    n_slot: int,
    cost: FlexMoECostModel | None = None,
) -> BaselineResult:
    """Run FlexMoE dynamic device placement and report its realized per-rank load.

    FlexMoE replicates hot experts across more vExpert slots (never changing token
    routing) until the balance ratio falls under its trigger threshold, then packs
    the vExperts onto ranks. The reported load is the even-split load per Eq. 6.
    """
    counts, rank_load, placement = flexmoe_schedule(
        loads.omega,
        spec.weight_bytes,
        num_ranks,
        n_slot,
        cost,
    )
    mean = rank_load.mean().item() if rank_load.numel() else 0.0
    physical_instances = int(sum(counts))
    return BaselineResult(
        name="flexmoe",
        rank_load=rank_load,
        placement=placement,
        metadata={
            "load_kind": "even-split vExpert load (balance ratio)",
            "placement_kind": "binary expert presence (vExpert replicas)",
            "balance_ratio": (rank_load.max().item() / mean) if mean > 0 else 1.0,
            **_budget_metadata(
                spec.num_experts, num_ranks, n_slot, physical_instances
            ),
        },
    )


def cube8_topology() -> torch.Tensor:
    """Official LPLB 8-rank, two-redundant cube topology."""
    return torch.tensor(
        [
            [3, 6],
            [0, 7],
            [1, 4],
            [2, 5],
            [7, 0],
            [4, 1],
            [5, 2],
            [6, 3],
        ],
        dtype=torch.int32,
    )


def ring8_topology() -> torch.Tensor:
    """Official LPLB 8-rank, one-redundant directed ring topology."""
    return torch.tensor(
        [[1], [2], [3], [4], [5], [6], [7], [0]],
        dtype=torch.int32,
    )


def select_lplb_topology(name: str, n_slot: int) -> tuple[str, torch.Tensor]:
    """Select an official LPLB topology compatible with the replica budget."""
    normalized = name.strip().lower()
    if normalized == "auto":
        normalized = "cube" if int(n_slot) % 2 == 0 else "ring"
    if normalized == "cube":
        return normalized, cube8_topology()
    if normalized == "ring":
        return normalized, ring8_topology()
    raise ValueError("LPLB topology must be one of: auto, cube, ring")


def load_lplb(lplb_root: str | Path | None = None):
    """Import the optional compiled LPLB package with an actionable error."""
    if lplb_root is not None:
        root = str(Path(lplb_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        return importlib.import_module("lplb")
    except (ImportError, OSError) as exc:
        raise LPLBUnavailableError(
            "LPLB is unavailable. Build it first with CUDA >= 12.6.3 and "
            "`pip install --no-build-isolation -e /home/tiger/LPLB`."
        ) from exc


class LPLBBaseline:
    """Thin adapter around the official compiled LPLB Planner."""

    def __init__(
        self,
        *,
        num_experts: int,
        ep_size: int,
        n_slot: int,
        lplb_root: str | Path | None = None,
        topology: torch.Tensor | None = None,
        topology_name: str = "auto",
    ) -> None:
        if not torch.cuda.is_available():
            raise LPLBUnavailableError("LPLB requires CUDA")
        _, redundant_per_rank, physical_per_rank = _slot_budget(
            num_experts, ep_size, n_slot
        )
        if redundant_per_rank <= 0:
            raise ValueError("LPLB requires at least one redundant slot per rank")

        if topology is None:
            self.topology_name, topology = select_lplb_topology(
                topology_name, redundant_per_rank
            )
        else:
            self.topology_name = "custom"
        self.r2o = topology.int().cuda()
        if ep_size % self.r2o.shape[0] != 0:
            raise ValueError("ep_size must be divisible by the LPLB topology size")
        if redundant_per_rank % self.r2o.shape[1] != 0:
            raise ValueError(
                "redundant slots per rank must be divisible by topology edge count"
            )

        lplb = load_lplb(lplb_root)
        self.planner = lplb.Planner(
            self.r2o,
            ep_size * physical_per_rank,
            num_experts,
            ep_size=ep_size,
        )
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.n_slot = int(n_slot)
        self.physical_slots_per_rank = physical_per_rank

    def update_mapping(self, workload_history: torch.Tensor) -> None:
        self.planner.update_redundancy_mapping(workload_history)

    def solve(self, workload: torch.Tensor) -> BaselineResult:
        """Run the official LP solver and derive its predicted per-rank loads."""
        workload = workload.to(device="cuda", dtype=torch.int32)
        available = torch.zeros((), dtype=torch.int32, device="cuda")
        probabilities, _ = self.planner.solve_probs(workload, available)

        physical_load = workload[self.planner.phy2log.long()].reshape(
            self.planner.n_group, self.planner.group_size, -1
        )
        width = self.planner.combined_redundant_experts * self.planner.num_redundants
        duplicated = physical_load[:, :, :width].reshape(
            self.planner.n_group,
            self.planner.group_size,
            self.planner.combined_redundant_experts,
            self.planner.num_redundants,
        ).sum(2)
        duplicated = duplicated * probabilities + (
            duplicated * (1 - probabilities)
        ).gather(1, self.r2o.expand_as(probabilities).long())
        duplicated = duplicated.sum(2)
        fixed = physical_load[:, :, width:-width].sum(2)
        rank_load = (duplicated + fixed).flatten()
        phy2log = self.planner.phy2log.long()
        ranks = torch.arange(
            self.ep_size, dtype=torch.int64, device=phy2log.device
        ).repeat_interleave(self.physical_slots_per_rank)
        placement = torch.zeros(
            (self.num_experts, self.ep_size),
            dtype=torch.int64,
            device=phy2log.device,
        )
        placement.index_put_(
            (phy2log, ranks), torch.ones_like(phy2log), accumulate=True
        )
        return BaselineResult(
            name="lplb",
            rank_load=rank_load,
            placement=placement,
            metadata={
                "load_kind": "LP predicted load",
                "placement_kind": "fixed LPLB redundant topology",
                "lplb_topology": self.topology_name,
                "topology_edges_per_rank": int(self.r2o.shape[1]),
                "available_groups": int(available.item()),
                **_budget_metadata(
                    self.num_experts,
                    self.ep_size,
                    self.n_slot,
                    self.ep_size * self.physical_slots_per_rank,
                ),
            },
        )


__all__ = [
    "BaselineResult",
    "FlexMoECostModel",
    "LPLBBaseline",
    "LPLBUnavailableError",
    "ShadowCostModel",
    "cube8_topology",
    "ring8_topology",
    "select_lplb_topology",
    "run_deepseek_eplb",
    "run_fastermoe",
    "run_flexmoe",
    "run_scale_eplb",
]

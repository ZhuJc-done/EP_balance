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
    def tau(self) -> float:
        return float(self.rank_load.max().item()) if self.rank_load.numel() else 0.0

    @property
    def mean_load(self) -> float:
        return float(self.rank_load.double().mean().item()) if self.rank_load.numel() else 0.0

    @property
    def imbalance(self) -> float:
        mean = self.mean_load
        return self.tau / mean if mean > 0 else 1.0


class LPLBUnavailableError(RuntimeError):
    """The optional compiled LPLB package is not importable."""


def run_scale_eplb(
    loads: Loads,
    topology: Topology,
    spec: ProblemSpec,
    cfg: EPLBConfig | None = None,
) -> BaselineResult:
    """Run this repository's exact placement-and-quota solver."""
    plan = solve(loads, topology, spec, cfg or EPLBConfig(), validate=False)
    return BaselineResult(
        name="scale-eplb",
        rank_load=plan.rank_load(),
        placement=plan.x,
        metadata={
            "load_kind": "realized quota load",
            "placement_kind": "binary expert presence",
            "replicas": int(plan.x.sum().item()),
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
    is divided equally among all of its replicas.
    """
    num_experts = loads.num_experts
    num_physical = num_gpus * n_slot
    if num_physical < num_experts:
        raise ValueError("num_gpus * n_slot must cover all logical experts")

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
    rank_load = per_physical[phy2log].view(num_gpus, n_slot).sum(dim=1)
    placement = torch.zeros(
        (num_experts, num_gpus), dtype=torch.int64, device=phy2log.device
    )
    ranks = torch.arange(num_gpus, dtype=torch.int64).repeat_interleave(n_slot)
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
            "replicas": num_physical,
            "num_groups": num_groups,
        },
    )


def run_fastermoe(
    loads: Loads,
    spec: ProblemSpec,
    *,
    num_ranks: int,
    cost: ShadowCostModel | None = None,
) -> BaselineResult:
    """Run FasterMoE dynamic shadowing and report its realized per-rank load.

    FasterMoE re-selects the shadowed (globally replicated) experts every
    micro-batch from the current load, so the reported load is the token load
    after the chosen hot experts have been spread back onto their source ranks.
    """
    shadow_mask, rank_load = select_shadow_experts(
        loads.lam,
        spec.main_rank,
        spec.weight_bytes,
        spec.s_tok,
        num_ranks,
        cost,
    )
    placement = torch.zeros(
        (spec.num_experts, num_ranks), dtype=torch.int64
    )
    experts = torch.arange(spec.num_experts, dtype=torch.int64)
    placement[experts, spec.main_rank.cpu()] = 1
    # A shadowed expert is transiently replicated onto every rank.
    placement[shadow_mask] = 1
    return BaselineResult(
        name="fastermoe",
        rank_load=rank_load,
        placement=placement,
        metadata={
            "load_kind": "shadow-redistributed token load",
            "placement_kind": "binary expert presence (shadow=all ranks)",
            "replicas": int(placement.sum().item()),
            "shadowed_experts": int(shadow_mask.sum().item()),
        },
    )


def run_flexmoe(
    loads: Loads,
    spec: ProblemSpec,
    *,
    num_ranks: int,
    cost: FlexMoECostModel | None = None,
) -> BaselineResult:
    """Run FlexMoE dynamic device placement and report its realized per-rank load.

    FlexMoE replicates hot experts across more vExpert slots (never changing token
    routing) until the balance ratio falls under its trigger threshold, then packs
    the vExperts onto ranks. The reported load is the even-split load per Eq. 6.
    """
    counts, rank_load, placement = flexmoe_schedule(
        loads.lam,
        spec.weight_bytes,
        num_ranks,
        spec.n_slot,
        cost,
    )
    mean = rank_load.mean().item() if rank_load.numel() else 0.0
    return BaselineResult(
        name="flexmoe",
        rank_load=rank_load,
        placement=placement,
        metadata={
            "load_kind": "even-split vExpert load (balance ratio)",
            "placement_kind": "binary expert presence (vExpert replicas)",
            "replicas": int(sum(counts)),
            "balance_ratio": (rank_load.max().item() / mean) if mean > 0 else 1.0,
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
    ) -> None:
        if not torch.cuda.is_available():
            raise LPLBUnavailableError("LPLB requires CUDA")
        if num_experts % ep_size != 0:
            raise ValueError("LPLB requires num_experts divisible by ep_size")
        base_per_rank = num_experts // ep_size
        redundant_per_rank = n_slot - base_per_rank
        if redundant_per_rank <= 0:
            raise ValueError("LPLB requires at least one redundant slot per rank")

        self.r2o = (topology if topology is not None else cube8_topology()).int().cuda()
        if ep_size % self.r2o.shape[0] != 0:
            raise ValueError("ep_size must be divisible by the LPLB topology size")
        if redundant_per_rank % self.r2o.shape[1] != 0:
            raise ValueError(
                "redundant slots per rank must be divisible by topology edge count"
            )

        lplb = load_lplb(lplb_root)
        self.planner = lplb.Planner(
            self.r2o,
            ep_size * n_slot,
            num_experts,
            ep_size=ep_size,
        )
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.n_slot = n_slot

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
        return BaselineResult(
            name="lplb",
            rank_load=rank_load,
            metadata={
                "load_kind": "LP predicted load",
                "placement_kind": "fixed LPLB redundant topology",
                "replicas": self.ep_size * self.n_slot,
                "available_groups": int(available.item()),
            },
        )


__all__ = [
    "BaselineResult",
    "FlexMoECostModel",
    "LPLBBaseline",
    "LPLBUnavailableError",
    "ShadowCostModel",
    "cube8_topology",
    "run_deepseek_eplb",
    "run_fastermoe",
    "run_flexmoe",
    "run_scale_eplb",
]

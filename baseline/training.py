"""Adapters that turn comparison solvers into Scale-EPLB runtime ``Plan`` objects.

The solvers in :mod:`baseline` were originally offline quality comparators.
This module keeps their placement decisions while normalising them to the
invariants required by the shared training data plane:

* every optimizer-owned main expert remains on ``main_rank``;
* placement is binary and respects the physical slot budget;
* routing quotas conserve every ``Omega[source, expert]`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch

from eplb.algorithm import solve
from eplb.config import EPLBConfig
from eplb.loads import Loads
from eplb.placement import plan_from_placement
from eplb.plan import Plan
from eplb.problem import ProblemSpec
from eplb.topology import Topology

from .adapters import LPLBBaseline
from .deepseek_eplb import rebalance_experts
from .fastermoe import ShadowCostModel, select_shadow_experts
from .flexmoe import FlexMoECostModel, flexmoe_schedule


PLAN_SOLVERS = ("scale", "fastermoe", "deepseek", "flexmoe", "lplb")


def _additional_slots(spec: ProblemSpec, num_ranks: int) -> int:
    """Translate Scale-EPLB's total ``N_slot`` into baseline replica slots."""
    if num_ranks <= 0 or spec.num_experts % num_ranks != 0:
        raise ValueError(
            "baseline training solvers require num_experts divisible by EP size"
        )
    mains_per_rank = spec.num_experts // num_ranks
    additional = int(spec.n_slot) - mains_per_rank
    if additional < 0:
        raise ValueError(
            f"N_slot={int(spec.n_slot)} cannot hold {mains_per_rank} main "
            "experts per rank"
        )
    return additional


def _safe_deepseek_groups(
    num_experts: int, num_nodes: int, requested: int
) -> int:
    """Choose a DeepSeek group count satisfying its hierarchy constraints."""
    if (
        requested > 0
        and num_experts % requested == 0
        and requested % num_nodes == 0
    ):
        return requested
    if num_nodes > 0 and num_experts % num_nodes == 0:
        return num_nodes
    return 1


def _main_fixed_placement(
    replica_counts: Sequence[int] | torch.Tensor,
    expert_load: torch.Tensor,
    main_rank: torch.Tensor,
    *,
    num_ranks: int,
    slots_per_rank: int,
    preferred: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack requested expert copies while pinning all main instances.

    DeepSeek EPLB and FlexMoE may relocate the sole physical copy of an expert,
    and their packers may place two same-expert vExperts on one rank.  Neither is
    representable by Scale-EPLB's binary placement or optimizer ownership model.
    This deterministic b-matching projection preserves each requested replica
    count whenever possible, uses the baseline placement as a preference, and
    never moves a main expert.
    """
    load_tensor = expert_load.detach().to(device="cpu", dtype=torch.float64)
    home_tensor = main_rank.detach().to(device="cpu", dtype=torch.int64)
    count_tensor = torch.as_tensor(
        replica_counts, dtype=torch.int64, device="cpu"
    )
    num_experts = int(load_tensor.numel())

    if count_tensor.shape != (num_experts,):
        raise ValueError(
            f"replica_counts must have shape [{num_experts}], got "
            f"{tuple(count_tensor.shape)}"
        )
    if home_tensor.shape != (num_experts,):
        raise ValueError("main_rank shape does not match expert count")
    load = [float(value) for value in load_tensor.tolist()]
    home = [int(value) for value in home_tensor.tolist()]
    counts = [int(value) for value in count_tensor.tolist()]
    if any(rank < 0 or rank >= num_ranks for rank in home):
        raise ValueError("main_rank contains an out-of-range rank")

    main_counts = [0] * num_ranks
    for rank in home:
        main_counts[rank] += 1
    if any(count > slots_per_rank for count in main_counts):
        raise ValueError("main placement already exceeds the physical slot budget")

    total_capacity = num_ranks * int(slots_per_rank)
    raw_desired = [max(1, count) for count in counts]
    target_instances = min(
        sum(raw_desired),
        total_capacity,
        num_experts * num_ranks,
    )
    # Binary placement cannot express duplicate same-expert slots on one rank.
    # Reassign such unusable physical copies by marginal load-reduction benefit,
    # so every representable baseline slot remains available to the comparison.
    desired = [min(count, num_ranks) for count in raw_desired]
    while sum(desired) > target_instances:
        removable = [
            expert for expert, count in enumerate(desired) if count > 1
        ]
        if not removable:
            raise ValueError("requested replica counts exceed total slot capacity")
        # Remove the least useful marginal copy first.
        expert = min(
            removable,
            key=lambda item: (
                load[item] / desired[item],
                -item,
            ),
        )
        desired[expert] -= 1
    while sum(desired) < target_instances:
        expandable = [
            expert
            for expert, count in enumerate(desired)
            if count < num_ranks
        ]
        if not expandable:
            break
        expert = max(
            expandable,
            key=lambda item: (
                load[item] / (desired[item] * (desired[item] + 1.0)),
                -item,
            ),
        )
        desired[expert] += 1

    if preferred is None:
        preference = [[0] * num_ranks for _ in range(num_experts)]
    else:
        preference_tensor = preferred.detach().to(
            device="cpu", dtype=torch.int64
        )
        if preference_tensor.shape != (num_experts, num_ranks):
            raise ValueError(
                "preferred placement must have shape "
                f"[{num_experts}, {num_ranks}]"
            )
        preference = preference_tensor.tolist()

    placement = [[0] * num_ranks for _ in range(num_experts)]
    for expert, rank in enumerate(home):
        placement[expert][rank] = 1
    slots_used = list(main_counts)

    # Fast path: requested replica counts normally form a feasible bipartite
    # packing. Place only O(replica_budget * R) candidates and fall back to the
    # general marginal projection below if fixed mains make those exact counts
    # infeasible.
    requested = [
        (load[expert] / desired[expert], expert, copy_index)
        for expert in range(num_experts)
        for copy_index in range(desired[expert] - 1)
    ]
    requested.sort(key=lambda item: (-item[0], item[1], item[2]))
    fast_placement = [row.copy() for row in placement]
    fast_slots = list(slots_used)
    fast_rank_load = [0.0] * num_ranks
    for expert, rank in enumerate(home):
        fast_rank_load[rank] += load[expert] / desired[expert]
    fast_ok = True
    for per_copy, expert, _ in requested:
        candidates = [
            rank
            for rank in range(num_ranks)
            if fast_placement[expert][rank] == 0
            and fast_slots[rank] < slots_per_rank
        ]
        if not candidates:
            fast_ok = False
            break
        rank = min(
            candidates,
            key=lambda item: (
                -preference[expert][item],
                fast_rank_load[item],
                fast_slots[item],
                item,
            ),
        )
        fast_placement[expert][rank] = 1
        fast_slots[rank] += 1
        fast_rank_load[rank] += per_copy
    if fast_ok:
        return torch.tensor(fast_placement, dtype=torch.int8)

    current_count = [1] * num_experts
    physical_instances = num_experts
    while physical_instances < target_instances:
        per_copy = [
            load[expert] / current_count[expert]
            for expert in range(num_experts)
        ]
        predicted_load = [0.0] * num_ranks
        for expert in range(num_experts):
            for rank in range(num_ranks):
                if placement[expert][rank]:
                    predicted_load[rank] += per_copy[expert]
        best: tuple[tuple[float, ...], int, int] | None = None
        for expert in range(num_experts):
            count = current_count[expert]
            marginal = load[expert] / (count * (count + 1))
            unmet = count < desired[expert]
            for rank in range(num_ranks):
                if (
                    placement[expert][rank] != 0
                    or slots_used[rank] >= slots_per_rank
                ):
                    continue
                key = (
                    float(unmet),
                    marginal,
                    float(preference[expert][rank]),
                    -predicted_load[rank],
                    -float(slots_used[rank]),
                    -float(expert),
                    -float(rank),
                )
                if best is None or key > best[0]:
                    best = (key, expert, rank)
        if best is None:
            raise ValueError(
                "cannot fill the requested physical budget with a binary "
                "main-fixed placement"
            )
        _, expert, rank = best
        placement[expert][rank] = 1
        slots_used[rank] += 1
        current_count[expert] += 1
        physical_instances += 1

    return torch.tensor(placement, dtype=torch.int8)


@dataclass(frozen=True)
class ScalePlanSolver:
    """The native Scale-EPLB solver behind the common plugin interface."""

    name: str = "scale"

    def __call__(
        self,
        loads: Loads,
        topology: Topology,
        spec: ProblemSpec,
        cfg: EPLBConfig,
    ) -> Plan:
        return solve(loads, topology, spec, cfg, validate=False)


@dataclass(frozen=True)
class FasterMoEPlanSolver:
    """FasterMoE global-shadow selection with source-local shadow quotas."""

    cost: ShadowCostModel = field(default_factory=ShadowCostModel)
    name: str = "fastermoe"

    def __call__(
        self,
        loads: Loads,
        topology: Topology,
        spec: ProblemSpec,
        cfg: EPLBConfig,
    ) -> Plan:
        num_ranks = topology.num_ranks
        additional_slots = _additional_slots(spec, num_ranks)
        shadow_cpu, _ = select_shadow_experts(
            loads.omega,
            spec.main_rank,
            spec.weight_bytes,
            int(spec.s_tok),
            num_ranks,
            self.cost,
            additional_slots,
        )
        shadow = shadow_cpu.to(device=loads.device)
        main_rank = spec.main_rank.to(device=loads.device, dtype=torch.int64)
        expert = torch.arange(
            spec.num_experts, device=loads.device, dtype=torch.int64
        )
        source = torch.arange(
            num_ranks, device=loads.device, dtype=torch.int64
        ).view(num_ranks, 1)

        placement = torch.zeros(
            (spec.num_experts, num_ranks),
            dtype=torch.int8,
            device=loads.device,
        )
        placement[expert, main_rank] = 1
        placement[shadow] = 1

        # Original FasterMoE semantics: a shadowed expert is computed on every
        # token's source worker; an unshadowed expert stays on its main worker.
        destination = torch.where(
            shadow.view(1, spec.num_experts),
            source,
            main_rank.view(1, spec.num_experts),
        )
        quota = torch.zeros(
            (num_ranks, spec.num_experts, num_ranks),
            dtype=torch.int64,
            device=loads.device,
        )
        quota.scatter_(
            2,
            destination.unsqueeze(-1),
            loads.omega.to(torch.int64).unsqueeze(-1),
        )
        theta = quota.sum(dim=(0, 1)).amax()
        return Plan(x=placement, q=quota, theta=theta)


@dataclass(frozen=True)
class DeepSeekPlanSolver:
    """DeepSeek replica-count/hierarchy solver projected to fixed mains."""

    num_groups: int = 8
    name: str = "deepseek"

    def __call__(
        self,
        loads: Loads,
        topology: Topology,
        spec: ProblemSpec,
        cfg: EPLBConfig,
    ) -> Plan:
        num_ranks = topology.num_ranks
        additional_slots = _additional_slots(spec, num_ranks)
        groups = _safe_deepseek_groups(
            spec.num_experts, topology.num_domains, int(self.num_groups)
        )
        physical = spec.num_experts + num_ranks * additional_slots
        physical_per_rank = physical // num_ranks
        expert_load = loads.expert_load().detach().to(device="cpu")
        phy2log, _, log_count = rebalance_experts(
            expert_load.view(1, -1),
            physical,
            groups,
            topology.num_domains,
            num_ranks,
        )
        phy2log = phy2log[0]
        preferred = torch.zeros(
            (spec.num_experts, num_ranks), dtype=torch.int64
        )
        destination = torch.arange(num_ranks).repeat_interleave(
            physical_per_rank
        )
        preferred.index_put_(
            (phy2log, destination),
            torch.ones_like(phy2log),
            accumulate=True,
        )
        placement_cpu = _main_fixed_placement(
            log_count[0],
            expert_load,
            spec.main_rank,
            num_ranks=num_ranks,
            slots_per_rank=int(spec.n_slot),
            preferred=preferred,
        )
        return plan_from_placement(
            loads,
            topology,
            spec,
            placement_cpu.to(device=loads.device),
            cfg,
            validate=False,
        )


@dataclass(frozen=True)
class FlexMoEPlanSolver:
    """FlexMoE vExpert-count solver projected to fixed optimizer ownership."""

    cost: FlexMoECostModel = field(default_factory=FlexMoECostModel)
    name: str = "flexmoe"

    def __call__(
        self,
        loads: Loads,
        topology: Topology,
        spec: ProblemSpec,
        cfg: EPLBConfig,
    ) -> Plan:
        num_ranks = topology.num_ranks
        additional_slots = _additional_slots(spec, num_ranks)
        counts, _, preferred = flexmoe_schedule(
            loads.omega,
            spec.weight_bytes,
            num_ranks,
            additional_slots,
            self.cost,
        )
        placement_cpu = _main_fixed_placement(
            counts,
            loads.expert_load(),
            spec.main_rank,
            num_ranks=num_ranks,
            slots_per_rank=int(spec.n_slot),
            preferred=preferred,
        )
        return plan_from_placement(
            loads,
            topology,
            spec,
            placement_cpu.to(device=loads.device),
            cfg,
            validate=False,
        )


class LPLBPlanSolver:
    """Official LPLB ratio solver with a main-fixed redundancy mapping."""

    name = "lplb"

    def __init__(
        self,
        *,
        lplb_root: str | Path | None = None,
        topology_name: str = "auto",
    ) -> None:
        self.lplb_root = lplb_root
        self.topology_name = topology_name
        self._planner: LPLBBaseline | None = None
        self._shape: tuple[int, int, int] | None = None

    def __call__(
        self,
        loads: Loads,
        topology: Topology,
        spec: ProblemSpec,
        cfg: EPLBConfig,
    ) -> Plan:
        num_ranks = topology.num_ranks
        additional_slots = _additional_slots(spec, num_ranks)
        shape = (spec.num_experts, num_ranks, additional_slots)
        if self._planner is None:
            self._planner = LPLBBaseline(
                num_experts=spec.num_experts,
                ep_size=num_ranks,
                n_slot=additional_slots,
                lplb_root=self.lplb_root,
                topology_name=self.topology_name,
            )
            self._shape = shape
        elif self._shape != shape:
            raise ValueError(
                "one LPLBPlanSolver instance cannot serve incompatible layer shapes"
            )

        workload = loads.expert_load().to(
            device=loads.device, dtype=torch.int32
        )
        planner = self._planner.planner
        group_size = int(planner.group_size)
        num_groups = int(planner.n_group)
        num_redundants = int(planner.num_redundants)
        combined = int(planner.combined_redundant_experts)
        mains_per_rank = spec.num_experts // num_ranks
        selected_per_rank = combined * num_redundants
        if selected_per_rank != additional_slots:
            raise RuntimeError(
                "LPLB planner shape does not match the replica slot budget"
            )
        if selected_per_rank > mains_per_rank:
            raise ValueError(
                "main-fixed LPLB requires additional replica slots per rank "
                "to be no greater than the local main-expert count"
            )

        # Keep every original expert on its optimizer-owned rank. Within each
        # rank, reorder only physical slots so the hottest local experts occupy
        # LPLB's redundant prefix.
        expert = torch.arange(
            spec.num_experts, device=loads.device, dtype=torch.int64
        )
        home = spec.main_rank.to(device=loads.device, dtype=torch.int64)
        local_order = torch.argsort(
            home * spec.num_experts + expert, stable=True
        ).view(num_ranks, mains_per_rank)
        local_order = local_order.gather(
            1,
            torch.argsort(
                workload[local_order], dim=1, descending=True, stable=True
            ),
        )
        original = local_order.view(
            num_groups, group_size, mains_per_rank
        )
        selected = original[:, :, :selected_per_rank].view(
            num_groups,
            group_size,
            combined,
            num_redundants,
        )
        r2o = self._planner.r2o.to(device=loads.device, dtype=torch.int64)
        duplicate = selected.gather(
            1,
            r2o.view(1, group_size, 1, num_redundants).expand_as(selected),
        )
        planner.phy2log = torch.cat(
            (original, duplicate.flatten(2)), dim=2
        ).flatten().to(torch.int32)

        available = torch.zeros((), dtype=torch.int32, device=loads.device)
        solution = planner.solve_probs(workload, available)
        probabilities = solution[0] if isinstance(solution, tuple) else solution
        probabilities = probabilities.reshape(
            num_groups, group_size, num_redundants
        ).to(dtype=torch.float64).clamp_(0.0, 1.0)

        # Column j of r2o is a permutation from duplicate rank -> original
        # rank. Its argsort therefore maps each selected original to the rank
        # hosting that redundant copy.
        original_to_duplicate = torch.argsort(r2o, dim=0)
        group = torch.arange(
            num_groups, device=loads.device, dtype=torch.int64
        ).view(num_groups, 1, 1, 1)
        source_local = torch.arange(
            group_size, device=loads.device, dtype=torch.int64
        ).view(1, group_size, 1, 1)
        edge = torch.arange(
            num_redundants, device=loads.device, dtype=torch.int64
        ).view(1, 1, 1, num_redundants)
        duplicate_local = original_to_duplicate[
            source_local, edge
        ].expand(num_groups, group_size, combined, num_redundants)
        selected_expert = selected.flatten()
        selected_main = (
            group * group_size + source_local
        ).expand_as(selected).flatten()
        selected_duplicate = (
            group * group_size + duplicate_local
        ).flatten()
        retention = probabilities[
            group.expand(
                num_groups, group_size, combined, num_redundants
            ),
            source_local.expand(
                num_groups, group_size, combined, num_redundants
            ),
            edge.expand(
                num_groups, group_size, combined, num_redundants
            ),
        ].flatten()

        placement = torch.zeros(
            (spec.num_experts, num_ranks),
            dtype=torch.int8,
            device=loads.device,
        )
        placement[expert, home] = 1
        placement[selected_expert, selected_duplicate] = 1

        quota = torch.zeros(
            (num_ranks, spec.num_experts, num_ranks),
            dtype=torch.int64,
            device=loads.device,
        )
        quota.scatter_(
            2,
            home.view(1, spec.num_experts, 1).expand(
                num_ranks, spec.num_experts, 1
            ),
            loads.omega.to(torch.int64).unsqueeze(-1),
        )
        source = torch.arange(
            num_ranks, device=loads.device, dtype=torch.int64
        ).view(num_ranks, 1).expand(num_ranks, selected_expert.numel())
        selected_expert_2d = selected_expert.view(1, -1).expand_as(source)
        need = loads.omega[:, selected_expert].to(torch.int64)
        cumulative_main = torch.round(
            need.cumsum(dim=0).to(torch.float64) * retention.view(1, -1)
        ).to(torch.int64)
        main_quota = torch.diff(
            cumulative_main,
            dim=0,
            prepend=torch.zeros_like(cumulative_main[:1]),
        )
        duplicate_quota = need - main_quota

        if int(cfg.u_min) > 1:
            floor = int(cfg.u_min)
            small_main = (main_quota > 0) & (main_quota < floor)
            duplicate_quota = torch.where(
                small_main, need, duplicate_quota
            )
            main_quota = torch.where(
                small_main, torch.zeros_like(main_quota), main_quota
            )
            small_duplicate = (
                (duplicate_quota > 0) & (duplicate_quota < floor)
            )
            main_quota = torch.where(
                small_duplicate, need, main_quota
            )
            duplicate_quota = torch.where(
                small_duplicate,
                torch.zeros_like(duplicate_quota),
                duplicate_quota,
            )

        quota[
            source,
            selected_expert_2d,
            selected_main.view(1, -1).expand_as(source),
        ] = main_quota
        quota[
            source,
            selected_expert_2d,
            selected_duplicate.view(1, -1).expand_as(source),
        ] = duplicate_quota
        theta = quota.sum(dim=(0, 1)).amax()
        return Plan(
            x=placement.contiguous(),
            q=quota.contiguous(),
            theta=theta,
        )


def make_training_plan_solver(
    name: str,
    *,
    fastermoe_bw_net: float = 50e9 / 8,
    fastermoe_bw_mm: float = 11.5e12,
    flexmoe_threshold: float = 1.2,
    deepseek_num_groups: int = 8,
    lplb_root: str | Path | None = None,
    lplb_topology: str = "auto",
):
    """Create a per-layer plan solver selected by the training launcher."""
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "scale": "scale",
        "scale-eplb": "scale",
        "fastermoe": "fastermoe",
        "faster-moe": "fastermoe",
        "deepseek": "deepseek",
        "deepseek-eplb": "deepseek",
        "flexmoe": "flexmoe",
        "flex-moe": "flexmoe",
        "lplb": "lplb",
    }
    try:
        selected = aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(PLAN_SOLVERS)
        raise ValueError(
            f"unknown EPLB plan solver {name!r}; expected one of: {choices}"
        ) from exc

    if selected == "scale":
        return ScalePlanSolver()
    if selected == "fastermoe":
        return FasterMoEPlanSolver(
            ShadowCostModel(
                bw_net=float(fastermoe_bw_net),
                bw_mm=float(fastermoe_bw_mm),
            )
        )
    if selected == "deepseek":
        return DeepSeekPlanSolver(num_groups=int(deepseek_num_groups))
    if selected == "flexmoe":
        return FlexMoEPlanSolver(
            FlexMoECostModel(threshold=float(flexmoe_threshold))
        )
    return LPLBPlanSolver(
        lplb_root=lplb_root,
        topology_name=lplb_topology,
    )


__all__ = [
    "PLAN_SOLVERS",
    "DeepSeekPlanSolver",
    "FasterMoEPlanSolver",
    "FlexMoEPlanSolver",
    "LPLBPlanSolver",
    "ScalePlanSolver",
    "make_training_plan_solver",
]

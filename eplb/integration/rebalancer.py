"""Per-micro-batch rebalancing orchestrator (collect Ω -> solve -> apply)."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from ..algorithm import solve
from ..config import EPLBConfig
from ..distributed import all_gather_omega
from ..loads import Loads
from ..plan import Plan
from ..problem import ProblemSpec
from ..topology import Topology
from . import profiling
from .hooks import NullWeightMaterializer, RebalanceResult, WeightMaterializer


PlanSolver = Callable[[Loads, Topology, ProblemSpec, EPLBConfig], Plan]


class EPLBRebalancer:
    """Owns the topology/spec/config and runs the collect->solve->apply loop.

    Args:
        topo: Cluster topology.
        spec: Static problem spec.
        cfg: Solver config (defaults to :class:`EPLBConfig`).
        materializer: Backend weight materializer (defaults to no-op placeholder).
        plan_solver: Optional placement/plan plugin. The default is Scale-EPLB's
            native solver; plugins receive the same ``(loads, topo, spec, cfg)``.
        cache_plans: If True, cache solved plans for backward; else recompute from
            cached ``Ω`` (less memory, relies on determinism; default for K=1).
        ring_size: Max in-flight (layer, mb) entries to retain (FIFO eviction). ``0`` retains
            nothing, which is what the apply path wants: its backward is pure autograd (the
            replica broadcast/GIN transfer carries its own grad_fn) and never calls
            :meth:`backward`, so every retained ``Ω`` would be device memory nothing reads.
    """

    def __init__(
        self,
        topo: Topology,
        spec: ProblemSpec,
        cfg: Optional[EPLBConfig] = None,
        materializer: Optional[WeightMaterializer] = None,
        plan_solver: Optional[PlanSolver] = None,
        *,
        cache_plans: bool = False,
        ring_size: int = 64,
    ) -> None:
        topo.validate()
        spec.validate(topo.num_ranks)
        self.topo = topo
        self.spec = spec
        self.cfg = cfg or EPLBConfig()
        self.materializer = materializer or NullWeightMaterializer()
        self.plan_solver = plan_solver
        self.cache_plans = cache_plans
        self.ring_size = int(ring_size)

        # ring buffers keyed by (layer_id, micro_batch_id)
        self._omega_ring: Dict[Tuple[int, int], torch.Tensor] = {}
        self._plan_ring: Dict[Tuple[int, int], Plan] = {}
        self._order: list = []

    # -- forward ----------------------------------------------------------------
    def plan_from_omega(self, loads: Loads) -> Plan:
        """Solve directly from an already-gathered ``Ω`` (no communication)."""
        with profiling.record("solve", time_it=True, device=loads.device):
            if self.plan_solver is None:
                return solve(loads, self.topo, self.spec, self.cfg, validate=False)
            return self.plan_solver(loads, self.topo, self.spec, self.cfg)

    def rebalance_from_omega(
        self, loads: Loads, layer_id: int, micro_batch_id: int
    ) -> RebalanceResult:
        """Rebalance an already-gathered ``Ω`` in one process."""
        plan = self.plan_from_omega(loads)
        self._remember(layer_id, micro_batch_id, loads.omega, plan)
        handle = self.materializer.materialize(plan, layer_id, micro_batch_id)
        return RebalanceResult(plan=plan, weight_handle=handle)

    def rebalance(
        self,
        local_row: torch.Tensor,
        layer_id: int,
        micro_batch_id: int,
        *,
        group=None,
    ) -> RebalanceResult:
        """Collect ``Ω`` (all-gather), solve, and materialize replica weights.

        Args:
            local_row: int64 ``[E]`` this rank's per-expert token counts.
            layer_id: MoE layer id.
            micro_batch_id: Micro-batch id (the "virtual layer" key for backward).
            group: Optional process group for the all-gather.

        Returns:
            :class:`RebalanceResult` with the plan and a weight handle.
        """
        with profiling.record("all_gather_omega", time_it=True, device=local_row.device):
            loads = all_gather_omega(local_row, group=group)
        plan = self.plan_from_omega(loads)
        self._remember(layer_id, micro_batch_id, loads.omega, plan)
        handle = self.materializer.materialize(plan, layer_id, micro_batch_id)
        return RebalanceResult(plan=plan, weight_handle=handle)

    # -- backward ---------------------------------------------------------------
    def backward(self, layer_id: int, micro_batch_id: int) -> Plan:
        """Re-derive the forward plan for ``(layer, mb)`` and aggregate gradients."""
        key = (int(layer_id), int(micro_batch_id))
        if self.cache_plans and key in self._plan_ring:
            plan = self._plan_ring[key]
        else:
            if key not in self._omega_ring:
                raise KeyError(
                    f"no cached Ω for (layer={layer_id}, mb={micro_batch_id}); "
                    f"increase ring_size (current={self.ring_size})"
                )
            plan = self.plan_from_omega(Loads(self._omega_ring[key]))
        self.materializer.aggregate_gradients(plan, layer_id, micro_batch_id)
        return plan

    # -- ring buffer ------------------------------------------------------------
    def _remember(
        self, layer_id: int, micro_batch_id: int, omega: torch.Tensor, plan: Plan
    ) -> None:
        if self.ring_size <= 0:
            return
        key = (int(layer_id), int(micro_batch_id))
        if key not in self._omega_ring:
            self._order.append(key)
        self._omega_ring[key] = omega
        if self.cache_plans:
            self._plan_ring[key] = plan
        while len(self._order) > self.ring_size:
            old = self._order.pop(0)
            self._omega_ring.pop(old, None)
            self._plan_ring.pop(old, None)

    def clear(self) -> None:
        self._omega_ring.clear()
        self._plan_ring.clear()
        self._order.clear()

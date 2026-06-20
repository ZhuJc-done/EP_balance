"""Per-micro-batch rebalancing orchestrator (K=1 by default).

Usage inside a Megatron MoE layer (pseudo-code)::

    rebalancer = EPLBRebalancer(topo, spec, cfg)          # once, per EP layer
    ...
    # forward, every micro-batch:
    local_row = local_counts_from_routing(routed_expert_ids, E)
    result = rebalancer.rebalance(local_row, layer_id, mb_id)
    # -> dispatch with result.plan.q via your DeepEP Dispatcher
    ...
    # backward:
    rebalancer.backward(layer_id, mb_id)   # re-derives plan, aggregates grads

With ``K=1`` the solver runs for every (layer, micro-batch). Because the plan is
a deterministic function of ``Lambda``, the backward pass can *recompute* it from
the stored ``Lambda`` instead of caching the whole plan -- controlled by
``cache_plans``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ..algorithm import solve
from ..config import EPLBConfig
from ..distributed import all_gather_lambda
from ..loads import Loads
from ..plan import Plan
from ..problem import ProblemSpec
from ..topology import Topology
from .hooks import NullWeightMaterializer, RebalanceResult, WeightMaterializer


class EPLBRebalancer:
    """Owns the topology/spec/config and runs the collect->solve->apply loop.

    Args:
        topo: Cluster topology.
        spec: Static problem spec.
        cfg: Solver config (defaults to :class:`EPLBConfig`).
        materializer: Backend weight materializer (defaults to no-op placeholder).
        cache_plans: If True, keep solved plans keyed by ``(layer, mb)`` for the
            backward pass (uses more memory). If False, the backward pass
            recomputes the plan from the cached ``Lambda`` (less memory, relies on
            determinism -- the recommended default for K=1).
        ring_size: Max number of in-flight (layer, mb) entries to retain
            (>= max in-flight micro-batches for PP/VPP). Older entries are
            evicted FIFO.
    """

    def __init__(
        self,
        topo: Topology,
        spec: ProblemSpec,
        cfg: Optional[EPLBConfig] = None,
        materializer: Optional[WeightMaterializer] = None,
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
        self.cache_plans = cache_plans
        self.ring_size = int(ring_size)

        # ring buffers keyed by (layer_id, micro_batch_id)
        self._lambda_ring: Dict[Tuple[int, int], torch.Tensor] = {}
        self._plan_ring: Dict[Tuple[int, int], Plan] = {}
        self._order: list = []

    # -- forward ----------------------------------------------------------------
    def plan_from_lambda(self, loads: Loads) -> Plan:
        """Solve directly from an already-gathered ``Lambda`` (no communication)."""
        return solve(loads, self.topo, self.spec, self.cfg, validate=False)

    def rebalance_from_lambda(
        self, loads: Loads, layer_id: int, micro_batch_id: int
    ) -> RebalanceResult:
        """Like :meth:`rebalance` but for an already-gathered ``Lambda``.

        Use this in single-process simulation/tests where the full ``[R, E]``
        matrix is available without an all-gather.
        """
        plan = self.plan_from_lambda(loads)
        self._remember(layer_id, micro_batch_id, loads.lam, plan)
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
        """Collect ``Lambda`` (all-gather), solve, and materialize replica weights.

        Args:
            local_row: int64 ``[E]`` this rank's per-expert token counts.
            layer_id: MoE layer id.
            micro_batch_id: Micro-batch id (the "virtual layer" key for backward).
            group: Optional process group for the all-gather.

        Returns:
            :class:`RebalanceResult` with the plan and a weight handle.
        """
        loads = all_gather_lambda(local_row, group=group)
        plan = self.plan_from_lambda(loads)
        self._remember(layer_id, micro_batch_id, loads.lam, plan)
        handle = self.materializer.materialize(plan, layer_id, micro_batch_id)
        return RebalanceResult(plan=plan, weight_handle=handle)

    # -- backward ---------------------------------------------------------------
    def backward(self, layer_id: int, micro_batch_id: int) -> Plan:
        """Re-derive the forward plan for ``(layer, mb)`` and aggregate gradients.

        Returns the plan used (recomputed from cached ``Lambda`` unless plan
        caching is on), after invoking the materializer's gradient aggregation.
        """
        key = (int(layer_id), int(micro_batch_id))
        if self.cache_plans and key in self._plan_ring:
            plan = self._plan_ring[key]
        else:
            if key not in self._lambda_ring:
                raise KeyError(
                    f"no cached Lambda for (layer={layer_id}, mb={micro_batch_id}); "
                    f"increase ring_size (current={self.ring_size})"
                )
            plan = self.plan_from_lambda(Loads(self._lambda_ring[key]))
        self.materializer.aggregate_gradients(plan, layer_id, micro_batch_id)
        return plan

    # -- ring buffer ------------------------------------------------------------
    def _remember(
        self, layer_id: int, micro_batch_id: int, lam: torch.Tensor, plan: Plan
    ) -> None:
        key = (int(layer_id), int(micro_batch_id))
        if key not in self._lambda_ring:
            self._order.append(key)
        self._lambda_ring[key] = lam
        if self.cache_plans:
            self._plan_ring[key] = plan
        while len(self._order) > self.ring_size:
            old = self._order.pop(0)
            self._lambda_ring.pop(old, None)
            self._plan_ring.pop(old, None)

    def clear(self) -> None:
        self._lambda_ring.clear()
        self._plan_ring.clear()
        self._order.clear()

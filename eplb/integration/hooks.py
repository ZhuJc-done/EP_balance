"""Framework-facing interfaces (weight materialization, dispatch) between the solver and a backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ..plan import Plan


@dataclass
class RebalanceResult:
    """What :meth:`EPLBRebalancer.rebalance` returns for one (layer, mb)."""

    plan: Plan
    weight_handle: object = None  # opaque handle from the WeightMaterializer (None for placeholder)


@runtime_checkable
class WeightMaterializer(Protocol):
    """Materialises and releases replica expert weights for a plan (backend placeholder)."""

    def materialize(self, plan: Plan, layer_id: int, micro_batch_id: int) -> object:
        """Bring replica weights on-device and return a handle for :meth:`release`."""
        ...

    def release(self, handle: object) -> None:
        """Release weights materialised by a prior :meth:`materialize` call."""
        ...

    def aggregate_gradients(
        self, plan: Plan, layer_id: int, micro_batch_id: int
    ) -> None:
        """Aggregate replica gradients back to each expert's main optimizer owner."""
        ...


class NullWeightMaterializer:
    """No-op materializer for simulation/tests (no data movement)."""

    def materialize(self, plan: Plan, layer_id: int, micro_batch_id: int) -> object:
        # a real backend would copy replica weights here
        return {"layer_id": layer_id, "micro_batch_id": micro_batch_id}

    def release(self, handle: object) -> None:
        return None

    def aggregate_gradients(
        self, plan: Plan, layer_id: int, micro_batch_id: int
    ) -> None:
        # a real backend would reduce replica grads to main(e)
        return None


@runtime_checkable
class Dispatcher(Protocol):
    """Routes tokens to expert instances per the plan's quota table (backend placeholder)."""

    def dispatch(self, hidden_states: torch.Tensor, plan: Plan, src_rank: int):
        ...

    def combine(self, expert_outputs: torch.Tensor, plan: Plan, src_rank: int):
        ...

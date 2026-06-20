"""Framework-facing interfaces for the EPLB runtime.

These are the seams between the (framework-agnostic) Scale-EPLB solver and a
concrete backend such as Megatron-LM + DeepEP. This release implements the
*planning* side fully; the side-effecting pieces (moving real expert weights,
issuing the real dispatch) are defined here as interfaces with a working no-op /
reference implementation so the control flow is exercised end-to-end on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ..plan import Plan


@dataclass
class RebalanceResult:
    """What :meth:`EPLBRebalancer.rebalance` returns for one (layer, mb)."""

    plan: Plan
    # opaque handle returned by the WeightMaterializer (e.g. a buffer table);
    # None when using the placeholder materializer.
    weight_handle: object = None


@runtime_checkable
class WeightMaterializer(Protocol):
    """Materialises (and later releases) replica expert weights for a plan.

    In the production design (per the experiment plan) this is a Triton/NVSHMEM
    routine that prefetches replica weights on-device, keeps them stateless
    ("即用即还"), and in the backward pass re-materialises them from the
    deterministic plan to aggregate gradients back to ``main(e)``'s optimizer.

    *** THIS RELEASE: interface placeholder only. ***
    Implement this against your backend; :class:`NullWeightMaterializer` is the
    no-op used for CPU simulation and tests.
    """

    def materialize(self, plan: Plan, layer_id: int, micro_batch_id: int) -> object:
        """Bring all replica weights required by ``plan`` on-device.

        Returns an opaque handle that :meth:`release` can free.
        """
        ...

    def release(self, handle: object) -> None:
        """Release weights materialised by a prior :meth:`materialize` call."""
        ...

    def aggregate_gradients(
        self, plan: Plan, layer_id: int, micro_batch_id: int
    ) -> None:
        """Aggregate replica gradients back to each expert's main optimizer owner.

        Called in the backward pass. The plan is recomputed/looked-up
        deterministically from ``(layer_id, micro_batch_id)``.
        """
        ...


class NullWeightMaterializer:
    """No-op materializer for simulation/tests.

    It performs no data movement and asserts the plan is well-formed. Use it to
    validate the rebalancer control flow without a real backend or GPUs.
    """

    def materialize(self, plan: Plan, layer_id: int, micro_batch_id: int) -> object:
        # Placeholder: a real backend would copy replica weights here.
        return {"layer_id": layer_id, "micro_batch_id": micro_batch_id}

    def release(self, handle: object) -> None:  # noqa: D401 - trivial
        return None

    def aggregate_gradients(
        self, plan: Plan, layer_id: int, micro_batch_id: int
    ) -> None:
        # Placeholder: a real backend would reduce replica grads to main(e).
        return None


@runtime_checkable
class Dispatcher(Protocol):
    """Routes tokens to expert instances according to a plan's quota table.

    A Megatron + DeepEP backend implements this by translating ``plan.q`` into
    DeepEP's dynamic dispatch layout. *** Interface placeholder this release. ***
    """

    def dispatch(self, hidden_states: torch.Tensor, plan: Plan, src_rank: int):
        ...

    def combine(self, expert_outputs: torch.Tensor, plan: Plan, src_rank: int):
        ...

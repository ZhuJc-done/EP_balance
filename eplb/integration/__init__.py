"""Integration glue for plugging Scale-EPLB into a real EP training framework.

The :class:`~eplb.integration.rebalancer.EPLBRebalancer` orchestrates the
per-micro-batch loop (collect Lambda -> solve -> hand back a Plan). The
:mod:`~eplb.integration.hooks` module defines the framework-facing interfaces a
backend (Megatron-LM + DeepEP) must implement -- in particular the *weight
materialization* interface, which is a documented placeholder in this release.
"""

from .rebalancer import EPLBRebalancer
from .hooks import (
    WeightMaterializer,
    NullWeightMaterializer,
    Dispatcher,
    RebalanceResult,
)

__all__ = [
    "EPLBRebalancer",
    "WeightMaterializer",
    "NullWeightMaterializer",
    "Dispatcher",
    "RebalanceResult",
]

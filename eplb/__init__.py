"""Scale-EPLB: deterministic Expert-Parallelism Load Balancer.

Key ideas (from the Scale-EPLB problem definition):
  * Expert *replication*, not rearrangement. The main instance ``main(e)`` of an
    expert is immutable; balancing only *adds* replicas. This keeps the
    logical->physical mapping stable and the gradient back-aggregation simple.
  * Heterogeneous topology aware: NVLink domains (cheap, intra-node) + RDMA
    (expensive, inter-node). Cross-domain replicas are only created when the
    one-time weight movement is cheaper than repeatedly shipping tokens.
  * No CPU sync: every rank all-gathers the integer load matrix ``Lambda`` once,
    then independently computes a *bit-identical* plan on-device.

Public API::

    from eplb import (
        EPLBConfig, Topology, ProblemSpec, Loads, Plan,
        solve, compute_metrics, check_constraints,
        EPLBRebalancer,
    )
"""

from .config import EPLBConfig
from .topology import Topology
from .problem import ProblemSpec
from .loads import Loads
from .plan import Plan
from .algorithm import solve
from .metrics import compute_metrics, check_constraints, Metrics, ConstraintReport
from .integration.rebalancer import EPLBRebalancer

__all__ = [
    "EPLBConfig",
    "Topology",
    "ProblemSpec",
    "Loads",
    "Plan",
    "solve",
    "compute_metrics",
    "check_constraints",
    "Metrics",
    "ConstraintReport",
    "EPLBRebalancer",
]

__version__ = "0.1.0"

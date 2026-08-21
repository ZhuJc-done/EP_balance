"""Scale-EPLB: deterministic, CPU-sync-free Expert-Parallelism Load Balancer via expert replication."""

from .config import EPLBConfig
from .topology import Topology
from .problem import ProblemSpec
from .loads import Loads
from .plan import Plan
from .algorithm import solve
from .placement import plan_from_placement
from .metrics import compute_metrics, check_constraints, Metrics, ConstraintReport
from .integration.rebalancer import EPLBRebalancer

__all__ = [
    "EPLBConfig",
    "Topology",
    "ProblemSpec",
    "Loads",
    "Plan",
    "solve",
    "plan_from_placement",
    "compute_metrics",
    "check_constraints",
    "Metrics",
    "ConstraintReport",
    "EPLBRebalancer",
]

__version__ = "0.1.0"

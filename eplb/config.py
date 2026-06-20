"""Configuration for the Scale-EPLB solver.

All knobs that influence a *decision* are integers (fixed-point). Floating point
is intentionally avoided on the decision path so that every rank produces a
bit-identical plan regardless of its rank id or GPU mapping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EPLBConfig:
    """Solver configuration.

    Objective (from the problem definition)::

        min  alpha * tau
           + beta  * sum_{r,e,r'} c[r,r'] * q[r,e,r']        (token comm, Phi_token)
           + gamma * sum_{e, r!=main(e)} 2 c[main,r] |W_e| x  (weight move, Phi_weight)

    The weights are kept as integers; the reference solver treats makespan
    (``alpha``) as the dominant term and uses communication cost only as a
    deterministic tie-breaker. The full alpha/beta/gamma LP trade-off is left to
    the offline MILP oracle (see README).

    Attributes:
        alpha: Weight on the per-rank makespan ``tau``.
        beta:  Weight on token communication cost ``Phi_token``.
        gamma: Weight on expert-weight movement cost ``Phi_weight``.
        eta_milli: Fixed-point efficiency factor ``eta`` (x1000) used in the
            cross-domain break-even threshold ``T*[e] = ceil(2|W_e| / (eta s_tok))``.
            ``eta_milli = 1000`` means ``eta = 1.0``.
        u_min: Minimum routing quota granularity (C5). The reference assumes 1.
        allow_cross_domain: If False, Stage 1 is skipped and replicas never cross
            an NVLink domain boundary (useful for the single-domain experiments).
        max_stage2_iters: Safety cap on the number of replicas Stage 2 may add
            (also implicitly bounded by the free slot budget).
        tau_bisect_iters: Reserved for the explicit tau-bisection variant; the
            reference solver lowers tau by iterative replica insertion instead.
    """

    alpha: int = 1
    beta: int = 1
    gamma: int = 1
    eta_milli: int = 1000
    u_min: int = 1
    allow_cross_domain: bool = True
    max_stage2_iters: int = 4096
    tau_bisect_iters: int = 24

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "eta_milli", "u_min", "max_stage2_iters"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"EPLBConfig.{name} must be a positive integer")

    @property
    def eta(self) -> float:
        """Floating-point view of ``eta`` (for reporting only, never for decisions)."""
        return self.eta_milli / 1000.0

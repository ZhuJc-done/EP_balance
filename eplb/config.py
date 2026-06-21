"""Configuration for the Scale-EPLB solver (all decision knobs are integers for determinism)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EPLBConfig:
    """Solver weights and knobs for ``min alpha*tau + beta*Phi_token + gamma*Phi_weight``."""

    alpha: int = 1  # weight on makespan tau
    beta: int = 1  # weight on token communication cost
    gamma: int = 1  # weight on expert-weight movement cost
    eta_milli: int = 1000  # fixed-point eta (x1000) for the C6 break-even threshold
    u_min: int = 1  # minimum routing quota granularity (C5)
    allow_cross_domain: bool = True  # if False, Stage 1 is skipped (single-domain runs)
    max_stage2_iters: int = 4096  # safety cap on replicas Stage 2 may add
    tau_bisect_iters: int = 24  # reserved for an explicit tau-bisection variant

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "eta_milli", "u_min", "max_stage2_iters"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"EPLBConfig.{name} must be a positive integer")

    @property
    def eta(self) -> float:
        """Floating-point view of ``eta`` (for reporting only, never for decisions)."""
        return self.eta_milli / 1000.0

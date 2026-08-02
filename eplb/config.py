"""Configuration for the Scale-EPLB solver (all decision knobs are integers for determinism)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EPLBConfig:
    """Solver knobs for ``min αθ + βΦ_token + γΦ_weight``."""

    alpha: int = 1  # weight on maximum rank load θ
    beta: int = 1  # weight on token communication cost
    gamma: int = 1  # weight on expert-weight movement cost
    eta_milli: int = 1000  # fixed-point eta (x1000) for the C6 break-even threshold
    u_min: int = 1  # minimum routing quota granularity (C5)
    allow_cross_domain: bool = True  # if False, Stage 1 is skipped (single-domain runs)
    max_stage2_iters: int = 4096  # safety cap on replicas Stage 2 may add
    max_fast_stage2_iters: int = 64  # bounded repair budget for the latency backend
    theta_bisect_iters: int = 24  # reserved for an explicit θ-bisection variant

    def __post_init__(self) -> None:
        for name in (
            "alpha",
            "beta",
            "gamma",
            "eta_milli",
            "u_min",
            "max_stage2_iters",
            "max_fast_stage2_iters",
            "theta_bisect_iters",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"EPLBConfig.{name} must be a positive integer")

    @property
    def eta(self) -> float:
        """Floating-point view of ``eta`` (for reporting only, never for decisions)."""
        return self.eta_milli / 1000.0

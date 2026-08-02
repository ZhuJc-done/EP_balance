"""Dynamic per-micro-batch load matrix ``Ω[r,e]`` and its aggregates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Loads:
    """The dynamic routing load for one (layer, micro-batch)."""

    omega: torch.Tensor  # int64 [R, E] token counts, Ω[r, e]

    def __post_init__(self) -> None:
        self.omega = self.omega.to(torch.int64)

    @property
    def num_ranks(self) -> int:
        return int(self.omega.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.omega.shape[1])

    @property
    def device(self) -> torch.device:
        return self.omega.device

    def expert_load(self) -> torch.Tensor:
        """``ω_e = sum_r Ω[r, e]`` -> int64 ``[E]``."""
        return self.omega.sum(dim=0)

    def domain_demand(self, domain_of_rank: torch.Tensor, num_domains: int) -> torch.Tensor:
        """``T[d, e] = sum_{r in d} Ω[r, e]`` -> int64 ``[M, E]``."""
        E = self.num_experts
        out = torch.zeros((num_domains, E), dtype=torch.int64, device=self.omega.device)
        out.index_add_(0, domain_of_rank.to(torch.int64), self.omega)
        return out

    def validate(self, num_ranks: int, num_experts: int) -> None:
        if self.omega.shape != (num_ranks, num_experts):
            raise ValueError(
                f"omega must be [R, E]=[{num_ranks},{num_experts}], got {tuple(self.omega.shape)}"
            )
        if torch.any(self.omega < 0):
            raise ValueError("token counts must be non-negative")

"""Dynamic per-micro-batch load matrix ``Lambda`` and its derived aggregates.

``Lambda[r, e]`` = number of tokens that *originate* on rank ``r`` and are routed
by the gate to expert ``e`` for this micro-batch / layer. Counts are integers so
the plan is bit-identical across ranks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Loads:
    """The dynamic routing load for one (layer, micro-batch).

    Attributes:
        lam: int64 tensor ``[R, E]`` token counts, ``Lambda[r, e]``.
    """

    lam: torch.Tensor

    def __post_init__(self) -> None:
        self.lam = self.lam.to(torch.int64)

    @property
    def num_ranks(self) -> int:
        return int(self.lam.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.lam.shape[1])

    @property
    def device(self) -> torch.device:
        return self.lam.device

    def expert_load(self) -> torch.Tensor:
        """``lambda_e = sum_r Lambda[r, e]`` -> int64 ``[E]``."""
        return self.lam.sum(dim=0)

    def domain_demand(self, domain_of_rank: torch.Tensor, num_domains: int) -> torch.Tensor:
        """``T[d, e] = sum_{r in d} Lambda[r, e]`` -> int64 ``[M, E]``."""
        E = self.num_experts
        out = torch.zeros((num_domains, E), dtype=torch.int64, device=self.lam.device)
        out.index_add_(0, domain_of_rank.to(torch.int64), self.lam)
        return out

    def validate(self, num_ranks: int, num_experts: int) -> None:
        if self.lam.shape != (num_ranks, num_experts):
            raise ValueError(
                f"lam must be [R, E]=[{num_ranks},{num_experts}], got {tuple(self.lam.shape)}"
            )
        if torch.any(self.lam < 0):
            raise ValueError("token counts must be non-negative")

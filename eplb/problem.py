"""Static problem specification: the per-layer constants that do not change every micro-batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ProblemSpec:
    """Static per-layer specification (main placement, expert weights, slot budget)."""

    num_experts: int  # number of logical routed experts E
    main_rank: torch.Tensor  # int64 [E], immutable main(e) rank (C7)
    weight_bytes: torch.Tensor  # int64 [E], |W_e| used by the C6 gate
    s_tok: int  # bytes of one token's activation (hidden_dim * dtype_size)
    n_slot: int  # per-rank instance slot budget N_slot (C4)

    def __post_init__(self) -> None:
        self.main_rank = self.main_rank.to(torch.int64)
        self.weight_bytes = self.weight_bytes.to(torch.int64)

    @property
    def device(self) -> torch.device:
        return self.main_rank.device

    def validate(self, num_ranks: int) -> None:
        E = self.num_experts
        if self.main_rank.shape != (E,):
            raise ValueError(f"main_rank must be [E]=[{E}], got {tuple(self.main_rank.shape)}")
        if self.weight_bytes.shape != (E,):
            raise ValueError(
                f"weight_bytes must be [E]=[{E}], got {tuple(self.weight_bytes.shape)}"
            )
        if torch.any(self.main_rank < 0) or torch.any(self.main_rank >= num_ranks):
            raise ValueError("main_rank entries must be in [0, num_ranks)")
        if self.s_tok <= 0:
            raise ValueError("s_tok must be a positive integer")
        if self.n_slot <= 0:
            raise ValueError("n_slot must be a positive integer")
        # each rank must be able to hold the mains assigned to it (C4 feasibility)
        mains_per_rank = torch.bincount(self.main_rank, minlength=num_ranks)
        if torch.any(mains_per_rank > self.n_slot):
            raise ValueError(
                "n_slot too small: some rank's primary instances already exceed N_slot"
            )

    @staticmethod
    def uniform_main_placement(
        num_experts: int,
        num_ranks: int,
        weight_bytes_each: int,
        s_tok: int,
        n_slot: int,
        device: torch.device | str = "cpu",
    ) -> "ProblemSpec":
        """Round-robin ``main(e) = e % num_ranks`` placement with uniform weights."""
        main_rank = torch.arange(num_experts, device=device, dtype=torch.int64) % num_ranks
        weight_bytes = torch.full(
            (num_experts,), int(weight_bytes_each), dtype=torch.int64, device=device
        )
        spec = ProblemSpec(num_experts, main_rank, weight_bytes, s_tok, n_slot)
        spec.validate(num_ranks)
        return spec

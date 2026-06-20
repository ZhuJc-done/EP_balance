"""The solver output: placement table ``x`` and routing quota table ``q`` for dispatch/combine."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Plan:
    """Solver output (placement, routing quota, makespan)."""

    x: torch.Tensor  # int8 [E, R] placement table (1 = instance present)
    q: torch.Tensor  # int64 [R, E, R] routing quota q[src, e, dst]
    tau: int  # resulting per-rank makespan (max destination token load)

    @property
    def num_experts(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_ranks(self) -> int:
        return int(self.x.shape[1])

    def num_replicas(self) -> torch.Tensor:
        """int64 ``[E]``: number of physical instances per expert (>=1)."""
        return self.x.sum(dim=1).to(torch.int64)

    def replicas_of(self, e: int) -> torch.Tensor:
        """Sorted int64 tensor of rank ids hosting expert ``e``."""
        return torch.nonzero(self.x[e] == 1, as_tuple=False).flatten()

    def rank_load(self) -> torch.Tensor:
        """int64 ``[R]``: ``L[r'] = sum_{r,e} q[r, e, r']`` (tokens computed per rank)."""
        return self.q.sum(dim=(0, 1)).to(torch.int64)

    def slots_used(self) -> torch.Tensor:
        """int64 ``[R]``: number of instances hosted per rank."""
        return self.x.sum(dim=0).to(torch.int64)

    def dispatch_indices(self, src_rank: int):
        """Per-expert destination ranks and token counts for ``src_rank``.

        Returns:
            dict ``{e: (dst_ranks: LongTensor, counts: LongTensor)}`` for experts
            with nonzero quota originating at ``src_rank``.
        """
        out = {}
        qe = self.q[src_rank]
        for e in range(self.num_experts):
            row = qe[e]
            dsts = torch.nonzero(row > 0, as_tuple=False).flatten()
            if dsts.numel() > 0:
                out[e] = (dsts, row[dsts].clone())
        return out

    def equals(self, other: "Plan") -> bool:
        """Bit-identical comparison (used by determinism tests)."""
        return (
            self.tau == other.tau
            and torch.equal(self.x, other.x)
            and torch.equal(self.q, other.q)
        )

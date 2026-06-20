"""Cluster topology: NVLink domains, ranks, and the per-token communication cost matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Topology:
    """Static cluster topology (domain id per rank + per-token cost matrix)."""

    domain_of_rank: torch.Tensor  # int64 [R], contiguous domain id 0..M-1
    cost: torch.Tensor  # int64 [R, R] per-token comm cost, c[r,r]=0

    def __post_init__(self) -> None:
        self.domain_of_rank = self.domain_of_rank.to(torch.int64)
        self.cost = self.cost.to(torch.int64)

    @property
    def num_ranks(self) -> int:
        return int(self.domain_of_rank.numel())

    @property
    def num_domains(self) -> int:
        if self.num_ranks == 0:
            return 0
        return int(self.domain_of_rank.max().item()) + 1

    @property
    def device(self) -> torch.device:
        return self.domain_of_rank.device

    def ranks_in_domain(self, d: int) -> torch.Tensor:
        """Sorted int64 tensor of rank ids belonging to domain ``d``."""
        return torch.nonzero(self.domain_of_rank == d, as_tuple=False).flatten()

    def validate(self) -> None:
        R = self.num_ranks
        if self.cost.shape != (R, R):
            raise ValueError(f"cost must be [R, R]=[{R},{R}], got {tuple(self.cost.shape)}")
        if torch.any(torch.diagonal(self.cost) != 0):
            raise ValueError("cost matrix diagonal must be zero (c[r,r]=0)")
        if torch.any(self.cost < 0):
            raise ValueError("cost matrix must be non-negative")
        doms = torch.unique(self.domain_of_rank)
        expected = torch.arange(doms.numel(), device=self.device, dtype=torch.int64)
        if not torch.equal(doms, expected):
            raise ValueError("domain ids must be contiguous starting at 0")

    @staticmethod
    def from_nvlink_rdma(
        num_nodes: int,
        gpus_per_node: int,
        intra_cost: int = 1,
        inter_cost: int = 8,
        device: torch.device | str = "cpu",
    ) -> "Topology":
        """Build a topology with one NVLink domain per node.

        Args:
            num_nodes: Number of physical nodes (= number of NVLink domains).
            gpus_per_node: GPUs per node (ranks per domain).
            intra_cost: Per-token cost within a domain (NVLink).
            inter_cost: Per-token cost across domains (RDMA). Calibrate from
                measured NVLink:RDMA bandwidth ratio.
            device: Tensor device.
        """
        if intra_cost <= 0 or inter_cost <= 0:
            raise ValueError("costs must be positive integers")
        R = num_nodes * gpus_per_node
        dom = torch.arange(R, device=device, dtype=torch.int64) // gpus_per_node
        same = dom.unsqueeze(0) == dom.unsqueeze(1)
        cost = torch.where(
            same,
            torch.full((R, R), int(intra_cost), dtype=torch.int64, device=device),
            torch.full((R, R), int(inter_cost), dtype=torch.int64, device=device),
        )
        cost.fill_diagonal_(0)
        topo = Topology(dom, cost)
        topo.validate()
        return topo

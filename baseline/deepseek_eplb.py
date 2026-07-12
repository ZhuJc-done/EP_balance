"""Vendored DeepSeek EPLB placement heuristic.

Adapted from https://github.com/deepseek-ai/EPLB (MIT license).  The public
entry point and algorithm are intentionally kept equivalent so benchmarks do
not depend on the conflicting top-level ``eplb`` package name.
"""

from __future__ import annotations

from typing import Tuple

import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """LPT-pack equally many weighted objects into each pack."""
    num_layers, num_groups = weight.shape
    if num_groups % num_packs != 0:
        raise ValueError("the number of objects must be divisible by num_packs")
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        return pack_index, torch.zeros_like(weight, dtype=torch.int64)

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, -1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, -1)
    for layer in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[layer]:
            pack = min(
                (p for p in range(num_packs) if pack_items[p] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            pack_index[layer, group] = pack
            rank_in_pack[layer, group] = pack_items[pack]
            pack_weights[pack] += weight[layer, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_physical: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily duplicate the expert with largest load-per-replica."""
    num_layers, num_logical = weight.shape
    if num_physical < num_logical:
        raise ValueError("num_physical must be at least num_logical")
    device = weight.device
    phy2log = torch.arange(num_physical, dtype=torch.int64, device=device).repeat(
        num_layers, 1
    )
    rank = torch.zeros(num_layers, num_physical, dtype=torch.int64, device=device)
    logcnt = torch.ones(num_layers, num_logical, dtype=torch.int64, device=device)
    layers = torch.arange(num_layers, dtype=torch.int64, device=device)
    for physical in range(num_logical, num_physical):
        expert = (weight / logcnt).max(dim=-1).indices
        phy2log[:, physical] = expert
        rank[:, physical] = logcnt[layers, expert]
        logcnt[layers, expert] += 1
    return phy2log, rank, logcnt


def _inverse(perm: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(perm)
    inverse.scatter_(
        1,
        perm,
        torch.arange(
            perm.size(1), dtype=torch.int64, device=perm.device
        ).expand(perm.shape),
    )
    return inverse


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run DeepSeek EPLB's group-to-node, replicate, then GPU-pack policy."""
    num_layers, num_logical = weight.shape
    if num_logical % num_groups != 0:
        raise ValueError("num_logical must be divisible by num_groups")
    if num_groups % num_nodes != 0:
        raise ValueError("num_groups must be divisible by num_nodes")
    if num_gpus % num_nodes != 0:
        raise ValueError("num_gpus must be divisible by num_nodes")
    if num_physical % num_gpus != 0:
        raise ValueError("num_physical must be divisible by num_gpus")

    group_size = num_logical // num_groups
    groups_per_node = num_groups // num_nodes
    physical_per_gpu = num_physical // num_gpus

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack, group_rank = balanced_packing(tokens_per_group, num_nodes)
    log2middle = (
        ((group_pack * groups_per_node + group_rank) * group_size).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack.device)
    ).flatten(-2)
    middle2log = _inverse(log2middle)

    tokens_per_middle = weight.gather(-1, middle2log).view(
        -1, num_logical // num_nodes
    )
    phy2middle, replica_rank, middle_count = replicate_experts(
        tokens_per_middle, num_physical // num_nodes
    )

    tokens_per_physical = (tokens_per_middle / middle_count).gather(
        -1, phy2middle
    )
    gpu_pack, rank_in_gpu = balanced_packing(
        tokens_per_physical, num_gpus // num_nodes
    )
    phy2packed = gpu_pack * physical_per_gpu + rank_in_gpu
    packed2phy = _inverse(phy2packed)

    packed2middle = phy2middle.gather(-1, packed2phy)
    packed2middle = (
        packed2middle.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical,
            num_logical // num_nodes,
            device=group_pack.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    packed2log = middle2log.gather(-1, packed2middle)
    packed_rank = replica_rank.gather(-1, packed2phy).view(num_layers, -1)
    logcnt = middle_count.view(num_layers, -1).gather(-1, log2middle)
    return packed2log, packed_rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DeepSeek EPLB physical/logical expert mappings."""
    num_layers, num_logical = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, replica_rank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, replica_rank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )

    max_count = int(logcnt.max().item())
    log2phy = torch.full(
        (num_layers, num_logical, max_count), -1, dtype=torch.int64
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * max_count + replica_rank,
        torch.arange(num_replicas, dtype=torch.int64).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


__all__ = ["balanced_packing", "replicate_experts", "rebalance_experts"]

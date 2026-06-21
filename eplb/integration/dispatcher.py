"""Phase C reference dispatcher: compute-invariant replication-aware MoE dispatch/combine over torch.distributed."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from ..plan import Plan
from ..problem import ProblemSpec
from .comm import all_to_all_single, broadcast_from_main


def assign_unit_dst(unit_expert: torch.Tensor, plan: Plan, src_local_rank: int) -> torch.Tensor:
    """Map each routing unit to a destination rank, consistent with ``plan.q[src, e, :]``.

    Args:
        unit_expert: int64 ``[U]`` expert id of each routing unit on this rank.
        plan: The solved plan (global ``x`` / ``q``).
        src_local_rank: This rank's group-local id.

    Returns:
        int64 ``[U]`` destination (group-local) rank per unit.
    """
    U = int(unit_expert.numel())
    device = unit_expert.device
    dst = torch.full((U,), -1, dtype=torch.int64, device=device)
    r = int(src_local_rank)
    for e in torch.unique(unit_expert).tolist():
        idxs = torch.nonzero(unit_expert == e, as_tuple=False).flatten()
        hosts = torch.nonzero(plan.x[e] == 1, as_tuple=False).flatten()
        counts = plan.q[r, e, hosts].to(torch.int64)
        if int(counts.sum().item()) != int(idxs.numel()):
            raise ValueError(
                f"quota/unit mismatch for expert {e} on rank {r}: "
                f"q sums to {int(counts.sum().item())} but {int(idxs.numel())} units present"
            )
        dst[idxs] = torch.repeat_interleave(hosts, counts)
    return dst


def _all_to_all_ids(send_ids: torch.Tensor, out_splits: List[int], in_splits: List[int], group) -> torch.Tensor:
    """Plain (non-autograd) all-to-all for integer metadata."""
    recv = send_ids.new_empty([int(sum(out_splits))])
    dist.all_to_all_single(recv, send_ids.contiguous(), out_splits, in_splits, group=group)
    return recv


def replicated_moe_forward(
    tokens: torch.Tensor,
    unit_token_idx: torch.Tensor,
    unit_expert: torch.Tensor,
    unit_prob: torch.Tensor,
    plan: Plan,
    spec: ProblemSpec,
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    weight_shapes: Sequence[torch.Size],
    mlp_fn: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor],
    group=None,
) -> torch.Tensor:
    """One replication-aware MoE layer forward for this rank's tokens.

    Args:
        tokens: float ``[T, H]`` hidden states for this rank's tokens.
        unit_token_idx: int64 ``[U]`` token index (into ``tokens``) of each routing unit.
        unit_expert: int64 ``[U]`` expert id of each routing unit.
        unit_prob: float ``[U]`` gate weight of each routing unit.
        plan: Solved plan (global ``x`` / ``q``).
        spec: Problem spec (provides ``main(e)`` as group-local rank ids).
        weights_local: ``{e: weight_tuple}`` for experts whose ``main(e)`` is this rank.
        weight_shapes: Shape of each weight tensor in an expert's tuple (same for all experts).
        mlp_fn: ``(x, weight_tuple) -> y`` expert forward.
        group: EP process group (defaults to the world group).

    Returns:
        float ``[T, H]`` combined MoE output for this rank's tokens.
    """
    device = tokens.device
    dtype = tokens.dtype
    H = tokens.shape[1]
    R = plan.num_ranks
    my_rank = dist.get_rank(group) if (group is not None and dist.is_initialized()) else (
        dist.get_rank() if dist.is_initialized() else 0
    )

    # materialise replicated experts' weights from main(e) (collective; every rank participates)
    num_replicas = plan.num_replicas()
    replicated = [e for e in range(plan.num_experts) if int(num_replicas[e].item()) > 1]
    We: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for e in replicated:
        main_local = int(spec.main_rank[e].item())
        local_w = weights_local.get(e)
        We[e] = tuple(
            broadcast_from_main(
                local_w[j] if local_w is not None else None,
                weight_shapes[j], dtype, device, main_local, group,
            )
            for j in range(len(weight_shapes))
        )

    # assign units -> dst, group contiguously by dst for all-to-all
    dst_of_unit = assign_unit_dst(unit_expert, plan, my_rank)
    perm = torch.argsort(dst_of_unit, stable=True)
    in_splits = torch.bincount(dst_of_unit, minlength=R).to(torch.int64).tolist()
    out_splits = plan.q[:, :, my_rank].sum(dim=1).to(torch.int64).tolist()

    send_tokens = tokens[unit_token_idx][perm]
    send_eid = unit_expert[perm]
    recv_tokens = all_to_all_single(send_tokens, out_splits, in_splits, group)
    recv_eid = _all_to_all_ids(send_eid, out_splits, in_splits, group)

    # compute each hosted expert on its received units
    out_units = recv_tokens.new_zeros((recv_tokens.shape[0], H))
    for e in torch.nonzero(plan.x[:, my_rank] == 1, as_tuple=False).flatten().tolist():
        midx = torch.nonzero(recv_eid == e, as_tuple=False).flatten()
        if midx.numel() == 0:
            continue
        w = We[e] if e in We else weights_local[e]
        out_units = out_units.index_copy(0, midx, mlp_fn(recv_tokens[midx], w))

    # send outputs back (transposed splits) and invert the dst-grouping permutation
    combined_back = all_to_all_single(out_units, in_splits, out_splits, group)
    out_per_unit = combined_back[torch.argsort(perm)]

    result = torch.zeros((tokens.shape[0], H), dtype=out_per_unit.dtype, device=device)
    result = result.index_add(0, unit_token_idx, unit_prob.unsqueeze(1) * out_per_unit)

    # keepalive (contributes 0): force every rank through the same collective backward branches so they can't deadlock
    keep = recv_tokens.sum() * 0.0
    for e in replicated:
        for w in We[e]:
            keep = keep + w.sum() * 0.0
    return result + keep

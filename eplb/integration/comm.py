"""Autograd-aware distributed primitives used by the Phase C replication dispatcher."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist


def global_rank(group, local_rank: int) -> int:
    """Translate a group-local rank to its global rank (identity for the world group)."""
    if group is None:
        return int(local_rank)
    return int(dist.get_global_rank(group, int(local_rank)))


class _AllToAllSingle(torch.autograd.Function):
    """``all_to_all_single`` with autograd (backward is the transposed all-to-all)."""

    @staticmethod
    def forward(ctx, inp, out_splits: List[int], in_splits: List[int], group):
        ctx.out_splits = out_splits
        ctx.in_splits = in_splits
        ctx.group = group
        out = inp.new_empty([int(sum(out_splits)), *inp.shape[1:]])
        dist.all_to_all_single(out, inp.contiguous(), out_splits, in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = grad_out.new_empty([int(sum(ctx.in_splits)), *grad_out.shape[1:]])
        dist.all_to_all_single(
            grad_in, grad_out.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return grad_in, None, None, None


def all_to_all_single(
    inp: torch.Tensor,
    out_splits: List[int],
    in_splits: List[int],
    group=None,
) -> torch.Tensor:
    """Differentiable all-to-all with per-rank split sizes (forward and transposed backward)."""
    return _AllToAllSingle.apply(inp, out_splits, in_splits, group)


class _BroadcastFromRoot(torch.autograd.Function):
    """Broadcast a tensor from ``root``; backward sum-reduces grads from all ranks back to ``root``."""

    @staticmethod
    def forward(ctx, tensor, root_global: int, group):
        ctx.root_global = root_global
        ctx.group = group
        out = tensor.clone()
        dist.broadcast(out, src=root_global, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        g = grad_out.clone().contiguous()
        dist.reduce(g, dst=ctx.root_global, op=dist.ReduceOp.SUM, group=ctx.group)
        # only the root's input is a real parameter; non-root inputs are ignored placeholders
        return g, None, None


def broadcast_from_main(
    weight: Optional[torch.Tensor],
    shape,
    dtype: torch.dtype,
    device,
    main_local_rank: int,
    group=None,
) -> torch.Tensor:
    """Materialise an expert's weight on every rank from its main owner (grads reduce to main).

    Args:
        weight: The real parameter on ``main(e)``; ``None`` on every other rank.
        shape: Weight shape (needed to build the placeholder on non-main ranks).
        dtype: Weight dtype.
        device: Weight device.
        main_local_rank: ``main(e)`` as a group-local rank id.
        group: Process group (defaults to the world group).

    Returns:
        The expert weight value on this rank; usable in a differentiable forward so that
        backward accumulates the summed gradient into the main owner's ``.grad``.
    """
    root = global_rank(group, main_local_rank)
    my_rank = dist.get_rank() if group is None else dist.get_rank(group)
    is_main = global_rank(group, my_rank) == root
    if is_main:
        src = weight
    else:
        src = torch.zeros(tuple(shape), dtype=dtype, device=device, requires_grad=True)
    return _BroadcastFromRoot.apply(src, root, group)

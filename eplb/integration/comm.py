"""Autograd-aware distributed primitives used by the sync-free Phase C dispatcher."""

from __future__ import annotations

import contextlib
from typing import List, Optional

import torch
import torch.distributed as dist

from . import profiling


def global_rank(group, local_rank: int) -> int:
    """Translate a group-local rank to its global rank (identity for the world group)."""
    if group is None:
        return int(local_rank)
    return int(dist.get_global_rank(group, int(local_rank)))


class _AllToAllSingle(torch.autograd.Function):
    """``all_to_all_single`` with autograd (backward is the transposed all-to-all)."""

    @staticmethod
    def forward(
        ctx,
        inp,
        out_splits: List[int],
        in_splits: List[int],
        group,
        forward_phase,
        backward_phase,
    ):
        ctx.out_splits = out_splits
        ctx.in_splits = in_splits
        ctx.group = group
        ctx.backward_phase = backward_phase
        out = inp.new_empty([int(sum(out_splits)), *inp.shape[1:]])
        contiguous = inp.contiguous()
        time_wire = forward_phase is not None and profiling.enabled()
        payload_bytes = (
            _remote_payload_bytes(contiguous, in_splits, group)
            if time_wire
            else 0
        )
        timer = (
            profiling.record(
                forward_phase,
                time_it=True,
                device=inp.device,
                payload_bytes=payload_bytes,
            )
            if time_wire
            else contextlib.nullcontext()
        )
        with timer:
            dist.all_to_all_single(
                out,
                contiguous,
                out_splits,
                in_splits,
                group=group,
            )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in = grad_out.new_empty([int(sum(ctx.in_splits)), *grad_out.shape[1:]])
        contiguous = grad_out.contiguous()
        time_wire = (
            ctx.backward_phase is not None and profiling.enabled()
        )
        payload_bytes = (
            _remote_payload_bytes(
                contiguous,
                ctx.out_splits,
                ctx.group,
            )
            if time_wire
            else 0
        )
        timer = (
            profiling.record(
                ctx.backward_phase,
                time_it=True,
                device=grad_out.device,
                payload_bytes=payload_bytes,
            )
            if time_wire
            else contextlib.nullcontext()
        )
        with timer:
            dist.all_to_all_single(
                grad_in,
                contiguous,
                ctx.in_splits,
                ctx.out_splits,
                group=ctx.group,
            )
        return grad_in, None, None, None, None, None


def _remote_payload_bytes(
    inp: torch.Tensor,
    in_splits: List[int],
    group,
) -> int:
    if not dist.is_initialized():
        return 0
    rank = dist.get_rank(group)
    local_rows = int(in_splits[rank])
    row_elements = inp[0].numel() if inp.shape[0] else 0
    return (
        max(int(inp.shape[0]) - local_rows, 0)
        * row_elements
        * inp.element_size()
    )


def all_to_all_single(
    inp: torch.Tensor,
    out_splits: List[int],
    in_splits: List[int],
    group=None,
    *,
    forward_phase: Optional[str] = None,
    backward_phase: Optional[str] = None,
) -> torch.Tensor:
    """Differentiable all-to-all with per-rank split sizes (forward and transposed backward)."""
    return _AllToAllSingle.apply(
        inp,
        out_splits,
        in_splits,
        group,
        forward_phase,
        backward_phase,
    )


def a2a_raw(
    inp: torch.Tensor,
    out_splits: List[int],
    in_splits: List[int],
    group=None,
    *,
    phase: Optional[str] = None,
) -> torch.Tensor:
    """Non-differentiable all-to-all, for callers that schedule the transpose leg themselves.

    Same collective as :func:`all_to_all_single` without the autograd node, so a hand-written backward
    can place it on a stream of its choosing instead of inheriting the one autograd replays on.
    """
    out = inp.new_empty([int(sum(out_splits)), *inp.shape[1:]])
    contiguous = inp.contiguous()
    time_wire = phase is not None and profiling.enabled()
    payload_bytes = (
        _remote_payload_bytes(contiguous, in_splits, group)
        if time_wire
        else 0
    )
    timer = (
        profiling.record(
            phase,
            time_it=True,
            device=inp.device,
            payload_bytes=payload_bytes,
        )
        if time_wire
        else contextlib.nullcontext()
    )
    with timer:
        dist.all_to_all_single(
            out,
            contiguous,
            out_splits,
            in_splits,
            group=group,
        )
    return out


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

"""Sync-free grouped expert MLP: run every hosted physical-slot expert in one batched pass over host-static ``[S, cap, H]`` buckets."""

from __future__ import annotations

from typing import Callable, Tuple

import torch

from . import profiling


def make_batched_gated_mlp(gated: bool, act: Callable) -> Callable:
    """Batched expert MLP matching Megatron's ``Linear`` ``[out, in]`` weights (compute ``x @ W.t()``).

    Args:
        gated: Whether the first projection is gated (SwiGLU-style).
        act: Activation function.

    Returns:
        ``fn(x[S, N, H], (W1[S, *, *], W2[S, *, *])) -> y[S, N, H]`` using batched matmuls.
    """

    def fn(x: torch.Tensor, w: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        h = torch.bmm(x, w[0].transpose(1, 2))  # x @ W1.t()
        if gated:
            gate, up = torch.chunk(h, 2, dim=-1)
            h = act(gate) * up
        else:
            h = act(h)
        return torch.bmm(h, w[1].transpose(1, 2))  # h @ W2.t()

    return fn


def grouped_expert_mlp(
    recv_tokens: torch.Tensor,
    recv_slot: torch.Tensor,
    group_sizes: torch.Tensor,
    weights: Tuple[torch.Tensor, ...],
    batched_mlp_fn: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor],
    cap: int,
    *,
    valid_mask: torch.Tensor | None = None,
    check_overflow: bool = False,
) -> torch.Tensor:
    """Compute per-slot expert outputs for received tokens, fully sync-free.

    Args:
        recv_tokens: float ``[T, H]`` tokens received by this rank (any order).
        recv_slot: int64 ``[T]`` local physical-slot id in ``[0, S)`` of each token.
        group_sizes: int64 ``[S]`` token count per slot (on-device; ``sum == T``).
        weights: stacked expert weights, each tensor shaped ``[S, ...]``.
        batched_mlp_fn: ``(x[S, cap, H], weights) -> y[S, cap, H']`` batched expert forward.
        cap: Per-slot capacity (host-static upper bound; must satisfy ``max(group_sizes) <= cap``).
        valid_mask: Optional device bool ``[T]``; false rows are ElasticBuffer worst-case padding.
        check_overflow: If True, assert no slot exceeds ``cap`` (forces one host sync; debug only).

    Returns:
        float ``[T, H']`` expert outputs in the original ``recv_tokens`` order.
    """
    T, H = recv_tokens.shape
    S = int(group_sizes.shape[0])
    device = recv_tokens.device

    if valid_mask is None:
        valid_mask = torch.ones(T, dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(torch.bool)

    # Invalid ElasticBuffer rows sort into an overflow slot and never enter the GEMM buckets.
    sort_slot = torch.where(valid_mask, recv_slot, torch.full_like(recv_slot, S))
    order = torch.argsort(sort_slot, stable=True)
    slot_sorted = sort_slot[order]
    valid_sorted = valid_mask[order]
    safe_slot = slot_sorted.clamp(max=S - 1)

    # each token's position within its slot via exclusive-cumsum offsets
    seg_start = torch.zeros(S, dtype=torch.int64, device=device)
    if S > 1:
        seg_start[1:] = torch.cumsum(group_sizes, dim=0)[:-1]
    pos_in_slot = torch.arange(T, device=device, dtype=torch.int64) - seg_start[safe_slot]

    if check_overflow:  # debug-only host sync
        assert bool((~valid_sorted | (pos_in_slot < cap)).all()), "grouped_expert_mlp: a slot exceeded cap"

    overflow = S * cap
    flat_idx = torch.where(
        valid_sorted,
        safe_slot * cap + pos_in_slot.clamp(max=cap - 1),
        torch.full_like(pos_in_slot, overflow),
    )

    # The extra row absorbs all invalid writes and is sliced away before expert compute.
    x_ext = recv_tokens.new_zeros((S * cap + 1, H))
    x_ext = x_ext.index_copy(0, flat_idx, recv_tokens[order])
    with profiling.record("apply/expert_gemm", time_it=True, device=device):
        y_pad = batched_mlp_fn(x_ext[:overflow].view(S, cap, H), weights)  # [S, cap, H']
    Hout = y_pad.shape[-1]
    gather_idx = flat_idx.clamp(max=overflow - 1)
    out_sorted = y_pad.reshape(S * cap, Hout)[gather_idx]
    out_sorted = out_sorted * valid_sorted.unsqueeze(1).to(out_sorted.dtype)

    out = out_sorted.new_empty((T, Hout))
    out = out.index_copy(0, order, out_sorted)
    return out

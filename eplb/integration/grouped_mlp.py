"""Sync-free grouped expert MLP: run every hosted physical-slot expert in one pass.

Two layouts are supported for that pass:

* **Ragged** (preferred): tokens stay packed slot-major and a variable-length grouped GEMM
  reads the slot boundaries from a device ``offs`` tensor. Work is exactly the received
  token count, and no host value is needed to launch it.
* **Padded** (fallback): every slot is padded to a host-static ``cap`` so the pass can be a
  plain ``bmm`` over ``[S, cap, H]``. Costs ``S * cap`` rows of GEMM regardless of occupancy
  -- empty slots included -- and ``cap`` has to be read to host before the layer can run.

The ragged path needs ``torch._grouped_mm`` (CUDA SM90+, 16-bit float), so the padded path
remains the fallback for CPU, fp32 and older GPUs.

Either way the rows arrive in a transport buffer sized for the worst case -- every peer sending
this rank everything -- which a balanced rank overshoots by about ``ep_size``. ``max_recv_rows``
optionally caps the ragged path's tensors at a host-static budget instead; see
:func:`compact_rows`.
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import torch

from . import profiling


def grouped_mm_usable(x: torch.Tensor) -> bool:
    """Whether ``torch._grouped_mm`` can run on tensors like ``x``."""
    return _grouped_mm_supported(x.device, x.dtype)


def _grouped_mm_supported(device: torch.device, dtype: torch.dtype) -> bool:
    if not hasattr(torch, "_grouped_mm") or device.type != "cuda":
        return False
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    return torch.cuda.get_device_capability(device)[0] >= 9


def _ragged_opt_out() -> bool:
    """``EPLB_GROUPED_GEMM=0`` forces the padded fallback even where the ragged path would run."""
    return os.environ.get("EPLB_GROUPED_GEMM", "1") == "0"


def ragged_enabled(x: torch.Tensor) -> bool:
    """Whether the ragged expert path should be taken for tensors like ``x``."""
    return not _ragged_opt_out() and grouped_mm_usable(x)


def ragged_available(dtype: torch.dtype = torch.bfloat16) -> bool:
    """Config-time counterpart of :func:`ragged_enabled`, before any tensor exists.

    Lets callers decide whether a host-static ``cap`` still has to be supplied.
    """
    if _ragged_opt_out() or not torch.cuda.is_available():
        return False
    return _grouped_mm_supported(torch.device("cuda", torch.cuda.current_device()), dtype)


def widths_aligned(element_size: int, *widths: int) -> bool:
    """Whether these matrix widths satisfy ``_grouped_mm``'s 16-byte stride rule.

    Every operand and result of a grouped GEMM is row-major, so each of their widths has to be a
    whole number of 16-byte chunks. Callers list the widths their own weight layout produces.
    """
    step = 16 // element_size
    return all(int(w) % step == 0 for w in widths)


def ragged_expert_ok(
    hidden: int,
    element_size: int,
    w1_eff: torch.Tensor,
    w2_eff: torch.Tensor,
    gated: bool,
) -> bool:
    """Stride-rule check for an expert 2-GEMM whose weights are already ``[S, in, out]``."""
    widths = [hidden, *w1_eff.shape[1:], *w2_eff.shape[1:]]
    if gated:
        widths.append(w1_eff.shape[-1] // 2)   # the gate/up split halves GEMM-1's output width
    return widths_aligned(element_size, *widths)


def make_grouped_gated_mlp(gated: bool, act: Callable) -> Callable:
    """Ragged counterpart of :func:`make_batched_gated_mlp` (Megatron ``[out, in]`` weights).

    Returns:
        ``fn(x[T, H], (W1[S, *, *], W2[S, *, *]), offs[S]) -> y[T, H]`` where ``offs`` holds the
        int32 exclusive-end row offset of each slot. Rows at or past ``offs[-1]`` belong to no
        slot and are left untouched by the GEMM.
    """

    def fn(x: torch.Tensor, w: Tuple[torch.Tensor, ...], offs: torch.Tensor) -> torch.Tensor:
        # _grouped_mm wants [S, in, out]; the transposed view is accepted as-is, and feeding it
        # directly avoids a contiguous copy of the whole weight stack (costlier than the GEMM).
        h = torch._grouped_mm(x, w[0].transpose(1, 2), offs=offs)
        if gated:
            gate, up = torch.chunk(h, 2, dim=-1)
            h = act(gate) * up
        else:
            h = act(h)
        return torch._grouped_mm(h, w[1].transpose(1, 2), offs=offs)

    def supports(x: torch.Tensor, w: Tuple[torch.Tensor, ...]) -> bool:
        return ragged_expert_ok(
            x.shape[-1], x.element_size(), w[0].transpose(1, 2), w[1].transpose(1, 2), gated
        )

    fn.supports = supports
    return fn


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


def slot_offsets(group_sizes: torch.Tensor, max_rows: int | None = None) -> torch.Tensor:
    """int32 ``[S]`` exclusive-end row offsets of each slot, for a grouped GEMM's ``offs``.

    ``max_rows`` clamps them to a trimmed row budget. Without it a budget set below the real
    receipt count would point the GEMM past the end of the tensor it was handed -- an opaque
    device-side fault, where the clamp instead leaves :func:`compact_rows`' guard to report it.
    """
    offs = torch.cumsum(group_sizes, dim=0)
    if max_rows is not None:
        offs = offs.clamp(max=max_rows)
    return offs.to(torch.int32)


def slot_sort(
    recv_slot: torch.Tensor,
    n_slot: int,
    valid_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order received rows slot-major, with ElasticBuffer's invalid rows trailing.

    Args:
        recv_slot: int64 ``[T]`` local physical-slot id of each row.
        n_slot: number of local physical slots.
        valid_mask: Optional device bool ``[T]``; false rows are worst-case padding.

    Returns:
        ``(order, valid_sorted, slot_sorted)``. Applying ``order`` yields the exact layout a
        ragged grouped GEMM consumes, so the sort doubles as the grouping. Invalid rows are
        given slot ``n_slot`` so they sort past every real slot.
    """
    T, device = recv_slot.shape[0], recv_slot.device
    if valid_mask is None:
        valid_mask = torch.ones(T, dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(torch.bool)
    sort_slot = torch.where(valid_mask, recv_slot, torch.full_like(recv_slot, n_slot))
    order = torch.argsort(sort_slot, stable=True)
    return order, valid_mask[order], sort_slot[order]


def compact_rows(
    order: torch.Tensor,
    valid_sorted: torch.Tensor,
    group_sizes: torch.Tensor,
    max_recv_rows: int | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim a slot-major row order to a host-static row budget.

    :func:`slot_sort` leaves every valid row in the leading ``sum(group_sizes)`` positions, so
    keeping the first ``max_recv_rows`` entries of ``order`` drops nothing while shrinking every
    downstream tensor from the transport's worst-case ``T = ep_size * max_tokens_per_rank`` rows
    to the budget. ``T`` is provisioned for "every peer sends me everything", which exceeds a
    balanced rank's actual receipts by roughly ``ep_size``.

    Unlike the padded path's per-slot ``cap``, this bounds only the rank's total, which is what
    stays stable under a skewed plan. Exceeding it would silently drop rows, so it is checked on
    device -- asynchronously, keeping the path free of host reads.
    """
    if max_recv_rows is None or max_recv_rows >= order.shape[0]:
        return order, valid_sorted
    if hasattr(torch, "_assert_async"):
        torch._assert_async(
            group_sizes.sum() <= max_recv_rows,
            "EPLB_MAX_RECV_ROWS is below this rank's received-token count",
        )
    return order[:max_recv_rows], valid_sorted[:max_recv_rows]


def scatter_rows(y: torch.Tensor, order: torch.Tensor, T: int) -> torch.Tensor:
    """Return ``y``'s rows to their pre-sort positions in a ``T``-row tensor.

    A trimmed ``order`` leaves the rows it does not name unwritten, so those have to start zeroed;
    a full one covers every row and can skip the fill.
    """
    shape = (T, y.shape[-1])
    out = y.new_zeros(shape) if order.shape[0] < T else y.new_empty(shape)
    return out.index_copy(0, order, y)


def ragged_mm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor | None) -> torch.Tensor:
    """``a @ b`` over the slot dimension: batched when ``offs`` is None, ragged otherwise.

    Shapes are ``[S, cap, K] x [S, K, N] -> [S, cap, N]`` batched, ``[T, K] x [S, K, N] -> [T, N]``
    ragged.
    """
    return torch.bmm(a, b) if offs is None else torch._grouped_mm(a, b, offs=offs)


def ragged_mm_tn(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor | None) -> torch.Tensor:
    """``aᵀ @ b`` contracted over the token dimension, yielding a per-slot stack ``[S, M, N]``.

    This is the weight-gradient product. Ragged mode contracts over the grouped dimension, so
    rows past the last offset contribute to no slot and are excluded.
    """
    if offs is None:
        return torch.bmm(a.transpose(1, 2), b)
    return torch._grouped_mm(a.transpose(0, 1), b, offs=offs)


def mask_rows(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Zero the rows of ``x`` where ``keep`` is false, without reading the dropped values.

    ``torch.where`` rather than a multiply: rows past the grouped GEMM's last offset are never
    written by it, so they hold uninitialised memory that a multiply would turn into NaN.
    """
    return torch.where(keep.unsqueeze(1), x, x.new_zeros(()))


def _ragged_expert_mlp(
    recv_tokens: torch.Tensor,
    weights: Tuple[torch.Tensor, ...],
    grouped_mlp_fn: Callable[..., torch.Tensor],
    group_sizes: torch.Tensor,
    order: torch.Tensor,
    valid_sorted: torch.Tensor,
    T: int,
) -> torch.Tensor:
    """Variable-length grouped GEMM over slot-major rows; see :func:`grouped_expert_mlp`."""
    device = recv_tokens.device
    # Sorting by slot already produced exactly the layout a grouped GEMM consumes: slot-major
    # rows with the invalid ones trailing past offs[-1], where no group covers them.
    x_sorted = recv_tokens.index_select(0, order)
    # Masking the input is what keeps the trailing rows out of Dgrad: the GEMM leaves their
    # grad slots unwritten, so without this they would reach the caller as uninitialised memory.
    x_sorted = mask_rows(x_sorted, valid_sorted)
    offs = slot_offsets(group_sizes, order.shape[0])
    with profiling.record("apply/expert_gemm", time_it=True, device=device):
        y_sorted = grouped_mlp_fn(x_sorted, weights, offs)
    y_sorted = mask_rows(y_sorted, valid_sorted)
    return scatter_rows(y_sorted, order, T)


def grouped_expert_mlp(
    recv_tokens: torch.Tensor,
    recv_slot: torch.Tensor,
    group_sizes: torch.Tensor,
    weights: Tuple[torch.Tensor, ...],
    batched_mlp_fn: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor],
    cap: int | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
    check_overflow: bool = False,
    grouped_mlp_fn: Callable[..., torch.Tensor] | None = None,
    max_recv_rows: int | None = None,
) -> torch.Tensor:
    """Compute per-slot expert outputs for received tokens, fully sync-free.

    Args:
        recv_tokens: float ``[T, H]`` tokens received by this rank (any order).
        recv_slot: int64 ``[T]`` local physical-slot id in ``[0, S)`` of each token.
        group_sizes: int64 ``[S]`` token count per slot (on-device; sums to the valid row count).
        weights: stacked expert weights, each tensor shaped ``[S, ...]``.
        batched_mlp_fn: ``(x[S, cap, H], weights) -> y[S, cap, H']`` padded expert forward.
        cap: Per-slot capacity for the padded path (must satisfy ``max(group_sizes) <= cap``).
            Unused, and may be None, when the ragged path is taken.
        valid_mask: Optional device bool ``[T]``; false rows are ElasticBuffer worst-case padding.
        check_overflow: If True, assert no slot exceeds ``cap`` (forces one host sync; debug only).
        grouped_mlp_fn: ``(x[T, H], weights, offs[S]) -> y[T, H']`` ragged expert forward. Taken
            whenever it is supplied and ``torch._grouped_mm`` supports these tensors.
        max_recv_rows: Optional host-static bound on this rank's total received rows, applied on
            the ragged path only. See :func:`compact_rows`; ``None`` keeps all ``T`` rows.

    Returns:
        float ``[T, H']`` expert outputs in the original ``recv_tokens`` order.
    """
    T, H = recv_tokens.shape
    S = int(group_sizes.shape[0])
    device = recv_tokens.device

    # Invalid ElasticBuffer rows sort into an overflow slot and never enter the GEMM buckets.
    order, valid_sorted, slot_sorted = slot_sort(recv_slot, S, valid_mask)
    safe_slot = slot_sorted.clamp(max=S - 1)

    if (
        grouped_mlp_fn is not None
        and T > 0
        and ragged_enabled(recv_tokens)
        and getattr(grouped_mlp_fn, "supports", lambda *_: True)(recv_tokens, weights)
    ):
        order_c, valid_c = compact_rows(order, valid_sorted, group_sizes, max_recv_rows)
        return _ragged_expert_mlp(
            recv_tokens, weights, grouped_mlp_fn, group_sizes, order_c, valid_c, T
        )

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
    return scatter_rows(out_sorted, order, T)

"""Level B apply-mode backward: re-materialise replica expert weights via async broadcast on a side
stream, overlapping the re-materialisation with weight-gradient (Wgrad) compute; grads reduce to main(e).

The key fact that makes the overlap valid: ``Wgrad = x^T . grad_y`` needs only saved activations (no
weight), while ``Dgrad = grad_y . W`` needs the weight. So at backward start we launch the replica
re-broadcast asynchronously and compute Wgrad while it is in flight, then consume the weight for Dgrad.
Only the standard gated/plain 2-GEMM expert MLP is supported (the structure ``grouped_mlp`` produces).
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from .comm import global_rank


class WeightPool:
    """Preallocated, shape-keyed scratch buffers reused across layers (borrow then give back)."""

    def __init__(self) -> None:
        self._free: Dict[tuple, List[torch.Tensor]] = {}

    def borrow(self, shape, dtype, device) -> torch.Tensor:
        key = (tuple(shape), dtype, str(device))
        free = self._free.get(key)
        if free:
            return free.pop()
        return torch.empty(tuple(shape), dtype=dtype, device=device)

    def give_back(self, t: torch.Tensor) -> None:
        key = (tuple(t.shape), t.dtype, str(t.device))
        self._free.setdefault(key, []).append(t)


_POOL = WeightPool()
_COMM_STREAMS: Dict[int, "torch.cuda.Stream"] = {}


def _comm_stream(device: torch.device):
    """A per-device side stream for async re-materialisation (None on CPU / no CUDA)."""
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _COMM_STREAMS.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=idx)
        _COMM_STREAMS[idx] = s
    return s


def _broadcast_replicas(meta, main_of, w1_eff_stack, w2_eff_stack, dtype, device, pool, cs) -> None:
    """Broadcast each replicated expert's W from main(e) and write it into the host's slot.

    Args:
        meta: Static layer metadata (slot map, main ranks, replicated experts, group, shapes).
        main_of: ``{e: (W1, W2)}`` resident params for experts this rank is main of.
        w1_eff_stack: ``[S, in, out1]`` effective GEMM-1 weight per slot (filled in place for replicas).
        w2_eff_stack: ``[S, mid, H]`` effective GEMM-2 weight per slot (filled in place for replicas).
        dtype: Weight dtype.
        device: Weight device.
        pool: :class:`WeightPool` for scratch broadcast buffers.
        cs: Side CUDA stream to enqueue the collectives on (None -> current/default stream).
    """
    slot_of: Dict[int, int] = {}
    for s, e in enumerate(meta["slot_to_e"]):
        if e >= 0:
            slot_of.setdefault(int(e), s)
    transpose_w = meta["transpose_w"]
    stream_ctx = torch.cuda.stream(cs) if cs is not None else contextlib.nullcontext()
    with stream_ctx:
        for e in meta["replicated"]:
            root = meta["root_global"][e]
            is_main = meta["main_rank"][e] == meta["my_rank"]
            buf1 = pool.borrow(meta["w1_shape"], dtype, device)
            buf2 = pool.borrow(meta["w2_shape"], dtype, device)
            if is_main:
                buf1.copy_(main_of[e][0])
                buf2.copy_(main_of[e][1])
            dist.broadcast(buf1, src=root, group=meta["group"])
            dist.broadcast(buf2, src=root, group=meta["group"])
            if (not is_main) and (e in slot_of):
                s = slot_of[e]
                w1_eff_stack[s].copy_(buf1.transpose(0, 1) if transpose_w else buf1)
                w2_eff_stack[s].copy_(buf2.transpose(0, 1) if transpose_w else buf2)
            pool.give_back(buf1)
            pool.give_back(buf2)


def _fill_main_slots(meta, main_of, w1_eff_stack, w2_eff_stack) -> None:
    """Write this rank's resident (main-owned) expert weights into their slots (effective layout)."""
    transpose_w = meta["transpose_w"]
    for s, e in enumerate(meta["slot_to_e"]):
        if e >= 0 and meta["main_rank"][e] == meta["my_rank"]:
            w1, w2 = main_of[e]
            w1_eff_stack[s].copy_(w1.transpose(0, 1) if transpose_w else w1)
            w2_eff_stack[s].copy_(w2.transpose(0, 1) if transpose_w else w2)


def _activation(meta, h_pre: torch.Tensor) -> torch.Tensor:
    """Apply the (gated or plain) activation to the GEMM-1 output ``h_pre``."""
    if meta["gated"]:
        gate, up = torch.chunk(h_pre, 2, dim=-1)
        return meta["act"](gate) * up
    return meta["act"](h_pre)


class _OverlappedExperts(torch.autograd.Function):
    """Batched expert 2-GEMM whose backward re-materialises replica weights (async) and reduces grads to main."""

    @staticmethod
    def forward(ctx, x_pad, meta, *main_w):  # x_pad: [S, cap, H]
        device, dtype = x_pad.device, x_pad.dtype
        S = int(meta["n_slot"])
        main_of = {e: (main_w[2 * i], main_w[2 * i + 1]) for i, e in enumerate(meta["main_experts"])}
        w1_eff = torch.zeros((S, *meta["w1_eff_shape"]), dtype=dtype, device=device)
        w2_eff = torch.zeros((S, *meta["w2_eff_shape"]), dtype=dtype, device=device)
        _fill_main_slots(meta, main_of, w1_eff, w2_eff)
        _broadcast_replicas(meta, main_of, w1_eff, w2_eff, dtype, device, _POOL, cs=None)

        h_pre = torch.bmm(x_pad, w1_eff)
        a = _activation(meta, h_pre)
        y = torch.bmm(a, w2_eff)

        ctx.meta = meta
        ctx.save_for_backward(x_pad, h_pre, *main_w)
        return y

    @staticmethod
    def backward(ctx, grad_y):  # grad_y: [S, cap, H]
        meta = ctx.meta
        saved = ctx.saved_tensors
        x_pad, h_pre = saved[0], saved[1]
        main_w = saved[2:]
        device, dtype = x_pad.device, x_pad.dtype
        S = int(meta["n_slot"])
        transpose_w = meta["transpose_w"]
        main_of = {e: (main_w[2 * i], main_w[2 * i + 1]) for i, e in enumerate(meta["main_experts"])}
        cs = _comm_stream(device)

        # resident (main-owned) weights are available immediately; replicas come via async broadcast
        w1_eff = torch.zeros((S, *meta["w1_eff_shape"]), dtype=dtype, device=device)
        w2_eff = torch.zeros((S, *meta["w2_eff_shape"]), dtype=dtype, device=device)
        _fill_main_slots(meta, main_of, w1_eff, w2_eff)
        if cs is not None:
            cs.wait_stream(torch.cuda.current_stream())
        _broadcast_replicas(meta, main_of, w1_eff, w2_eff, dtype, device, _POOL, cs)

        # --- Wgrad of GEMM-2 needs no weight -> overlaps the in-flight re-materialisation ---
        a = _activation(meta, h_pre)                                   # [S, cap, F]
        grad_w2_eff = torch.bmm(a.transpose(1, 2), grad_y)             # [S, F, H]

        if cs is not None:                                             # replica weights are now needed
            torch.cuda.current_stream().wait_stream(cs)

        # --- Dgrad chain (needs weights) ---
        grad_a = torch.bmm(grad_y, w2_eff.transpose(1, 2))            # [S, cap, F]
        with torch.enable_grad():
            hp = h_pre.detach().requires_grad_(True)
            a_g = _activation(meta, hp)
            (grad_h_pre,) = torch.autograd.grad(a_g, hp, grad_a)      # [S, cap, Fout]
        grad_w1_eff = torch.bmm(x_pad.transpose(1, 2), grad_h_pre)    # [S, H, Fout]
        grad_x = torch.bmm(grad_h_pre, w1_eff.transpose(1, 2))        # [S, cap, H]

        # back to per-expert parameter layout
        grad_w1_slot = grad_w1_eff.transpose(1, 2) if transpose_w else grad_w1_eff  # [S, *w1_shape]
        grad_w2_slot = grad_w2_eff.transpose(1, 2) if transpose_w else grad_w2_eff  # [S, *w2_shape]

        # --- reduce each replicated expert's Wgrad to its main owner (full-group collective) ---
        slot_of: Dict[int, int] = {}
        for s, e in enumerate(meta["slot_to_e"]):
            if e >= 0:
                slot_of.setdefault(int(e), s)
        reduced: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for e in meta["replicated"]:
            root = meta["root_global"][e]
            if e in slot_of:
                c1 = grad_w1_slot[slot_of[e]].contiguous()
                c2 = grad_w2_slot[slot_of[e]].contiguous()
            else:
                c1 = torch.zeros(meta["w1_shape"], dtype=dtype, device=device)
                c2 = torch.zeros(meta["w2_shape"], dtype=dtype, device=device)
            dist.reduce(c1, dst=root, group=meta["group"])
            dist.reduce(c2, dst=root, group=meta["group"])
            if meta["main_rank"][e] == meta["my_rank"]:
                reduced[e] = (c1, c2)

        # assemble grads for this rank's main weight inputs (reduced for replicated, local otherwise)
        grads: List[torch.Tensor] = []
        for e in meta["main_experts"]:
            if e in reduced:
                g1, g2 = reduced[e]
            else:
                s = slot_of[e]
                g1, g2 = grad_w1_slot[s].contiguous(), grad_w2_slot[s].contiguous()
            grads.extend([g1, g2])

        return (grad_x, None, *grads)


def overlapped_grouped_expert_mlp(
    recv_tokens: torch.Tensor,
    recv_slot: torch.Tensor,
    group_sizes: torch.Tensor,
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    slot_to_e: torch.Tensor,
    main_rank: torch.Tensor,
    replicated: Sequence[int],
    weight_shapes: Sequence[torch.Size],
    cap: int,
    *,
    gated: bool,
    act: Callable,
    transpose_w: bool,
    my_rank: int,
    n_slot: int,
    group=None,
    pool: WeightPool = _POOL,
) -> torch.Tensor:
    """Grouped expert MLP whose backward re-materialises replica weights with comm/Wgrad overlap.

    Args:
        recv_tokens: float ``[T, H]`` tokens received by this rank (any order).
        recv_slot: int64 ``[T]`` local physical-slot id of each token.
        group_sizes: int64 ``[n_slot]`` token count per slot (sum == T).
        weights_local: ``{e: (W1, W2)}`` resident params for experts this rank is main of.
        slot_to_e: int64 ``[n_slot]`` logical expert hosted at each local slot (-1 if empty).
        main_rank: int64 ``[E]`` group-local main rank of each expert.
        replicated: experts with more than one replica (need broadcast + reduce).
        weight_shapes: per-expert ``[(W1_shape), (W2_shape)]`` in parameter layout.
        cap: per-slot capacity (host-static).
        gated: whether GEMM-1 is gated (SwiGLU-style).
        act: activation function.
        transpose_w: True if weights are stored ``[out, in]`` (Megatron) and used as ``x @ W.t()``.
        my_rank: this rank's group-local id.
        n_slot: number of local physical slots.
        group: EP process group.
        pool: scratch buffer pool reused across layers.

    Returns:
        float ``[T, H]`` expert outputs in the original ``recv_tokens`` order.
    """
    T, H = recv_tokens.shape
    device = recv_tokens.device

    order = torch.argsort(recv_slot, stable=True)
    slot_sorted = recv_slot[order]
    seg_start = torch.zeros(n_slot, dtype=torch.int64, device=device)
    if n_slot > 1:
        seg_start[1:] = torch.cumsum(group_sizes, dim=0)[:-1]
    pos_in_slot = torch.arange(T, device=device, dtype=torch.int64) - seg_start[slot_sorted]
    flat_idx = slot_sorted * cap + pos_in_slot.clamp(max=cap - 1)

    x_pad = recv_tokens.new_zeros((n_slot * cap, H))
    x_pad = x_pad.index_copy(0, flat_idx, recv_tokens[order]).view(n_slot, cap, H)

    w1_shape = tuple(weight_shapes[0])
    w2_shape = tuple(weight_shapes[1])
    meta = {
        "slot_to_e": [int(v) for v in slot_to_e.tolist()],
        "main_rank": [int(v) for v in main_rank.tolist()],
        "replicated": [int(v) for v in replicated],
        "root_global": {int(e): global_rank(group, int(main_rank[int(e)].item())) for e in replicated},
        "main_experts": sorted(int(e) for e in weights_local.keys()),
        "w1_shape": w1_shape,
        "w2_shape": w2_shape,
        "w1_eff_shape": (w1_shape[1], w1_shape[0]) if transpose_w else w1_shape,
        "w2_eff_shape": (w2_shape[1], w2_shape[0]) if transpose_w else w2_shape,
        "gated": gated,
        "act": act,
        "transpose_w": transpose_w,
        "my_rank": int(my_rank),
        "n_slot": int(n_slot),
        "group": group,
        "pool": pool,
    }
    main_w: List[torch.Tensor] = []
    for e in meta["main_experts"]:
        main_w.extend([weights_local[e][0], weights_local[e][1]])

    y_pad = _OverlappedExperts.apply(x_pad, meta, *main_w)            # [n_slot, cap, H]
    out_sorted = y_pad.reshape(n_slot * cap, H)[flat_idx]
    out = out_sorted.new_empty((T, H))
    out = out.index_copy(0, order, out_sorted)
    return out

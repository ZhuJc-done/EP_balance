"""Level B apply-mode backward: re-materialise replica expert weights on a side stream, overlapping the
re-materialisation with weight-gradient (Wgrad) compute; grads reduce to main(e).

The key fact that makes the overlap valid: ``Wgrad = x^T . grad_y`` needs only saved activations (no
weight), while ``Dgrad = grad_y . W`` needs the weight. So the pull is launched asynchronously and
everything weight-free runs while it is in flight, with the weight consumed only at Dgrad. Only the
standard gated/plain 2-GEMM expert MLP is supported (``grouped_mlp``'s structure).

The pull is started from a pre-hook on the MoE block's output
(:meth:`OverlappedExperts.prefetch_on_backward_of`), not from the expert backward, so the window is
everything between the block output and Dgrad -- the scatter backward, the reverse combine
all-to-all, and then the Wgrad of GEMM-2 -- rather than that last term alone. It runs on a dedicated
weight stream (:func:`_weight_stream`), kept apart from the token stream that autograd replays the
dispatch/combine backwards on.

The transport that actually moves replica weights (materialise) and their grads (reduce-to-main) is
pluggable via a :class:`ReplicaTransport`: the default :class:`BroadcastReplicaTransport` uses
``dist.broadcast``/``dist.reduce``; the GIN backend injects a device-initiated get/put transport
(:class:`~eplb.integration.gin_weights.GinReplicaTransport`). The Wgrad/Dgrad overlap skeleton is
transport-agnostic, so it is validated once by the broadcast tests and reused by GIN.

:class:`OverlappedExperts` spreads this over any number of token chunks: the stacks are acquired once
per layer per direction and shared, so the token-side chunk pipeline (``EPLB_CHUNKS``) composes with
this backward without multiplying replica traffic.

This module leaves the ordering of the resulting nodes to autograd, which forces the reduce-to-main to
run last in the layer with nothing left to overlap.
:mod:`eplb.integration.manual_block` folds the same pipeline into a single node and schedules both
directions by hand to fix that; it is the default for the chunked path (``EPLB_MANUAL_BWD=0`` selects
this one instead). The pieces below -- :func:`build_meta`, :func:`_acquire_stacks`,
:class:`_ReplicaLease`, :class:`_PrefetchOnBackward` and the transports -- are shared by both.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist

from . import profiling
from .comm import global_rank
from .grouped_mlp import (
    compact_rows,
    mask_rows,
    ragged_enabled,
    ragged_expert_ok,
    ragged_mm,
    ragged_mm_tn,
    scatter_rows,
    slot_offsets,
    slot_sort,
)


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
_WEIGHT_STREAMS: Dict[int, "torch.cuda.Stream"] = {}


def _side_stream(registry: Dict[int, "torch.cuda.Stream"], device: torch.device):
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = registry.get(idx)
    if s is None:
        s = torch.cuda.Stream(device=idx)
        registry[idx] = s
    return s


def _comm_stream(device: torch.device):
    """Per-device side stream for the *token* channel: dispatch/combine (None on CPU / no CUDA)."""
    return _side_stream(_COMM_STREAMS, device)


def _weight_stream(device: torch.device):
    """Per-device side stream for the *weight* channel: replica get / grad put.

    Deliberately not the token stream. Autograd replays each backward node on the stream its forward
    op used, so the token all-to-alls come back on ``_comm_stream``; sharing it would make the
    ``wait_stream`` pair around a weight pull order the expert GEMMs against token transfers they have
    no dependency on.

    Only the pull runs here -- the grad put in ``reduce_grads`` stays on the compute stream, because
    autograd consumes its return value there. That still leaves the two uses of the single symmetric
    ``slot`` buffer ordered without a second buffer: each pull opens with
    ``wait_stream(compute)``, which covers the previous layer's put, and the consuming
    ``wait_stream(weight)`` in :meth:`_ReplicaLease.wait` covers the reverse.
    """
    return _side_stream(_WEIGHT_STREAMS, device)


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


class ReplicaTransport(Protocol):
    """Pluggable replica-weight transport for the overlapped expert backward.

    Three ops, all keyed off the static per-layer ``meta``:
      * :meth:`fill_main_slots` writes this rank's resident (main-owned) weights into their slots.
      * :meth:`materialize_replicas` fills the replica slots of the effective-layout weight stacks; it
        may enqueue its collectives on the side stream ``cs`` so the transfer overlaps the weight-free
        Wgrad compute.
      * :meth:`reduce_grads` reduces each replicated expert's per-slot Wgrad (parameter layout) to its
        ``main(e)`` owner and returns this rank's main-expert grads as ``[g1_e0, g2_e0, g1_e1, ...]``.
    """

    def fill_main_slots(self, meta, main_of, w1_eff, w2_eff, dtype, device) -> None:
        ...

    def materialize_replicas(self, meta, main_of, w1_eff, w2_eff, dtype, device, cs) -> None:
        ...

    def reduce_grads(self, meta, grad_w1_slot, grad_w2_slot, dtype, device) -> List[torch.Tensor]:
        ...


class BroadcastReplicaTransport:
    """``dist.broadcast`` / ``dist.reduce`` replica transport (the tested default)."""

    def __init__(self, pool: "WeightPool" = _POOL) -> None:
        self.pool = pool

    def fill_main_slots(self, meta, main_of, w1_eff, w2_eff, dtype, device) -> None:
        _fill_main_slots(meta, main_of, w1_eff, w2_eff)

    def materialize_replicas(self, meta, main_of, w1_eff, w2_eff, dtype, device, cs) -> None:
        _broadcast_replicas(meta, main_of, w1_eff, w2_eff, dtype, device, self.pool, cs)

    def reduce_grads(self, meta, grad_w1_slot, grad_w2_slot, dtype, device) -> List[torch.Tensor]:
        # reduce each replicated expert's Wgrad to its main owner (full-group collective)
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
        return grads


def _remote_transfer_bytes(meta):
    """Remote replica payload bytes for one pull/reduce (zero for unaccounted backends)."""
    return meta.get("remote_payload_bytes", 0)


def _acquire_stacks(meta, main_w, dtype, device, cs):
    """Build the per-slot effective weight stacks: main slots locally, replica slots over the transport.

    With ``cs`` non-None the replica transfer is enqueued on that side stream and the caller must
    ``wait_stream`` before reading the replica slots (that gap is where the weight-free Wgrad goes).
    """
    S = int(meta["n_slot"])
    main_of = {e: (main_w[2 * i], main_w[2 * i + 1]) for i, e in enumerate(meta["main_experts"])}
    w1_eff = torch.zeros((S, *meta["w1_eff_shape"]), dtype=dtype, device=device)
    w2_eff = torch.zeros((S, *meta["w2_eff_shape"]), dtype=dtype, device=device)
    meta["transport"].fill_main_slots(meta, main_of, w1_eff, w2_eff, dtype, device)
    if cs is not None:
        cs.wait_stream(torch.cuda.current_stream())
    meta["transport"].materialize_replicas(meta, main_of, w1_eff, w2_eff, dtype, device, cs)
    return w1_eff, w2_eff


class _ReplicaLease:
    """One layer's backward-side weight stacks, acquired once and shared by every token chunk.

    Each chunk's backward calls ``start`` / ``wait`` / ``release``; only the first ``start`` actually
    re-acquires, and the stacks drop when the last chunk releases. Two things depend on this being
    shared rather than per-chunk: the replica weights cross the wire once per layer per direction
    (what makes chunking free on the weight channel), and the transport's collectives are issued once
    per layer regardless of how many chunks a rank happens to have -- a per-chunk acquire would
    deadlock the moment two ranks disagreed on that count.
    """

    def __init__(self, meta, main_w: Sequence[torch.Tensor], dtype, device) -> None:
        self.meta = meta
        self.main_w = list(main_w)
        self.dtype, self.device = dtype, device
        self.consumers = 0            # incremented per forward chunk; each one backwards exactly once
        self.prefetched = False       # True if a backward pre-hook started the pull, not a chunk
        self._stacks: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cs = None
        self._waited = False

    def expect_consumer(self) -> None:
        self.consumers += 1

    def start(self) -> None:
        if self._stacks is not None:
            return
        self._cs = _weight_stream(self.device)
        with profiling.record(
            "apply/weight_repull",
            time_it=True,
            device=self.device,
            stream=self._cs,
            payload_bytes=_remote_transfer_bytes(self.meta),
        ):
            self._stacks = _acquire_stacks(
                self.meta, self.main_w, self.dtype, self.device, self._cs
            )
        self._waited = False

    def wait(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cs is not None and not self._waited:
            torch.cuda.current_stream().wait_stream(self._cs)
            self._waited = True
        assert self._stacks is not None, "_ReplicaLease.wait() before start()"
        return self._stacks

    def release(self) -> None:
        self.consumers -= 1
        if self.consumers <= 0:
            self._stacks = None


class _PrefetchOnBackward(torch.autograd.Function):
    """Identity forward; its backward starts the layer's replica-weight pull.

    Attached to the MoE block's output, so it is the first node of the block's backward and the pull
    is in flight across the scatter backward and the reverse combine all-to-all -- rather than only
    across the one weight-free Wgrad inside the expert backward, which is all the window would be if
    the chunk backward started it itself.
    """

    @staticmethod
    def forward(ctx, x, lease):
        ctx.lease = lease
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        ctx.lease.prefetched = True
        ctx.lease.start()
        return grad_out, None


class _ReplicaWeights(torch.autograd.Function):
    """Acquire the layer's per-slot weight stacks (forward) and reduce their grads to ``main(e)`` (backward).

    Sits upstream of every chunk's GEMM, so autograd accumulates all chunks' Wgrads into one grad
    before this node runs and the reduce-to-main happens exactly once per layer.
    """

    @staticmethod
    def forward(ctx, meta, dtype, device, *main_w):
        ctx.meta = meta
        with profiling.record(
            "apply/weight_move",
            time_it=True,
            device=device,
            payload_bytes=_remote_transfer_bytes(meta),
        ):
            return _acquire_stacks(meta, main_w, dtype, device, cs=None)

    @staticmethod
    def backward(ctx, grad_w1_eff, grad_w2_eff):
        meta = ctx.meta
        transpose_w = meta["transpose_w"]
        # back to per-expert parameter layout
        grad_w1_slot = grad_w1_eff.transpose(1, 2) if transpose_w else grad_w1_eff  # [S, *w1_shape]
        grad_w2_slot = grad_w2_eff.transpose(1, 2) if transpose_w else grad_w2_eff  # [S, *w2_shape]
        with profiling.record(
            "apply/grad_move",
            time_it=True,
            device=grad_w1_eff.device,
            payload_bytes=_remote_transfer_bytes(meta),
        ):
            grads = meta["transport"].reduce_grads(
                meta, grad_w1_slot, grad_w2_slot, grad_w1_eff.dtype, grad_w1_eff.device
            )
        return (None, None, None, *grads)


class _ChunkExperts(torch.autograd.Function):
    """One token chunk's batched expert 2-GEMM. Saves activations only; backward re-acquires the weights.

    The weight stacks are inputs but are deliberately **not** saved, so nothing keeps them resident
    past the forward: the backward gets them from the shared lease instead, overlapping that
    re-acquisition with the Wgrad of GEMM-2 (which needs only saved activations).
    """

    @staticmethod
    def forward(ctx, x, w1_eff, w2_eff, meta, lease, offs):  # x: [S, cap, H] padded / [T, H] ragged
        with profiling.record("apply/expert_gemm", time_it=True, device=x.device):
            h_pre = ragged_mm(x, w1_eff, offs)
            a = _activation(meta, h_pre)
            y = ragged_mm(a, w2_eff, offs)
        ctx.meta, ctx.lease, ctx.offs = meta, lease, offs
        ctx.save_for_backward(x, h_pre)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        meta, lease, offs = ctx.meta, ctx.lease, ctx.offs
        x, h_pre = ctx.saved_tensors
        lease.start()      # no-op if the block's backward pre-hook already kicked the pull off

        # --- Wgrad of GEMM-2 needs no weight -> overlaps the in-flight re-acquisition ---
        a = _activation(meta, h_pre)
        grad_w2_eff = ragged_mm_tn(a, grad_y, offs)                    # [S, F, H]

        w1_eff, w2_eff = lease.wait()                                  # replica weights are now needed

        # --- Dgrad chain (needs weights) ---
        grad_a = ragged_mm(grad_y, w2_eff.transpose(1, 2), offs)
        with torch.enable_grad():
            hp = h_pre.detach().requires_grad_(True)
            a_g = _activation(meta, hp)
            (grad_h_pre,) = torch.autograd.grad(a_g, hp, grad_a)
        grad_w1_eff = ragged_mm_tn(x, grad_h_pre, offs)                # [S, H, Fout]
        grad_x = ragged_mm(grad_h_pre, w1_eff.transpose(1, 2), offs)

        lease.release()
        return (grad_x, grad_w1_eff, grad_w2_eff, None, None, None)


def build_meta(
    *,
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    slot_to_e: torch.Tensor,
    main_rank: torch.Tensor,
    replicated: Sequence[int],
    weight_shapes: Sequence[torch.Size],
    gated: bool,
    act: Callable,
    transpose_w: bool,
    my_rank: int,
    n_slot: int,
    group=None,
    pool: WeightPool = _POOL,
    transport: "Optional[ReplicaTransport]" = None,
) -> Tuple[dict, List[torch.Tensor]]:
    """Build the static per-layer metadata + this rank's main weight list shared by both backends.

    Returns ``(meta, main_w)``, where ``main_w`` is ``[W1_e0, W2_e0, W1_e1, ...]`` over this rank's main
    experts in ascending id -- the order every consumer uses to line grads back up with parameters.
    """
    transport = transport or BroadcastReplicaTransport(pool)
    remote_bytes_fn = getattr(transport, "remote_payload_bytes", None)
    remote_payload_bytes = (
        remote_bytes_fn() if profiling.enabled() and callable(remote_bytes_fn) else 0
    )
    w1_shape = tuple(weight_shapes[0])
    w2_shape = tuple(weight_shapes[1])
    # slot_to_e/main_rank host lists + root_global drive the broadcast transport's host-side main-slot
    # fill and dist.broadcast/reduce (a per-slot/per-expert D2H). The GIN transport fills main slots and
    # schedules entirely on device, so these host copies are skipped there -> the GIN path is 0 D2H.
    is_broadcast = isinstance(transport, BroadcastReplicaTransport)
    root_global = (
        {int(e): global_rank(group, int(main_rank[int(e)].item())) for e in replicated}
        if is_broadcast else {}
    )
    meta = {
        "slot_to_e": [int(v) for v in slot_to_e.tolist()] if is_broadcast else None,
        "main_rank": [int(v) for v in main_rank.tolist()] if is_broadcast else None,
        "replicated": [int(e) for e in replicated],
        "root_global": root_global,
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
        "transport": transport,
        "remote_payload_bytes": remote_payload_bytes,
    }
    main_w: List[torch.Tensor] = []
    for e in meta["main_experts"]:
        main_w.extend([weights_local[e][0], weights_local[e][1]])
    return meta, main_w


class OverlappedExperts:
    """A layer's expert compute, over one or more token chunks, owning the replica-weight lifecycle.

    The per-slot weight stacks are acquired once in ``__init__`` and every :meth:`chunk` reads that one
    copy, so splitting tokens into chunks costs nothing on the weight channel. They are not saved for
    backward: each chunk's backward re-acquires them through a shared :class:`_ReplicaLease`, which
    also acquires once, and autograd accumulates the chunks' Wgrads before the single reduce-to-main.

    Chunking is a token-side pipeline (chunk k's dispatch/combine overlaps chunk k+1's expert GEMM)
    and lives in the caller; this class only has to make the weights chunk-count-agnostic.
    """

    def __init__(
        self,
        *,
        weights_local: Dict[int, Tuple[torch.Tensor, ...]],
        slot_to_e: torch.Tensor,
        main_rank: torch.Tensor,
        replicated: Sequence[int],
        weight_shapes: Sequence[torch.Size],
        gated: bool,
        act: Callable,
        transpose_w: bool,
        my_rank: int,
        n_slot: int,
        dtype: torch.dtype,
        device,
        group=None,
        pool: WeightPool = _POOL,
        transport: "Optional[ReplicaTransport]" = None,
    ) -> None:
        self.meta, main_w = build_meta(
            weights_local=weights_local, slot_to_e=slot_to_e, main_rank=main_rank,
            replicated=replicated, weight_shapes=weight_shapes, gated=gated, act=act,
            transpose_w=transpose_w, my_rank=my_rank, n_slot=n_slot, group=group, pool=pool,
            transport=transport,
        )
        self.n_slot = int(n_slot)
        self.lease = _ReplicaLease(self.meta, main_w, dtype, device)
        self.w1_eff, self.w2_eff = _ReplicaWeights.apply(self.meta, dtype, device, *main_w)

    def prefetch_on_backward_of(self, out: torch.Tensor) -> torch.Tensor:
        """Tag the block output so its backward starts this layer's weight pull as early as possible.

        Returns ``out`` (an identity view). Call it on the last tensor the MoE block produces; every
        node between there and the expert backward then runs with the pull in flight.
        """
        return _PrefetchOnBackward.apply(out, self.lease)

    def chunk(
        self,
        recv_tokens: torch.Tensor,
        recv_slot: torch.Tensor,
        group_sizes: torch.Tensor,
        cap: Optional[int] = None,
        valid_mask: Optional[torch.Tensor] = None,
        max_recv_rows: Optional[int] = None,
    ) -> torch.Tensor:
        """Expert compute for one chunk of received tokens; returns them in ``recv_tokens`` order."""
        T, H = recv_tokens.shape
        n_slot, device = self.n_slot, recv_tokens.device
        order, valid_sorted, slot_sorted = slot_sort(recv_slot, n_slot, valid_mask)

        if ragged_enabled(recv_tokens) and ragged_expert_ok(
            H, recv_tokens.element_size(), self.w1_eff, self.w2_eff, self.meta["gated"]
        ):
            order, valid_sorted = compact_rows(order, valid_sorted, group_sizes, max_recv_rows)
            # Sorting already grouped the rows; masking keeps the trailing worst-case rows out of
            # Dgrad, whose slots the grouped GEMM leaves unwritten.
            x = mask_rows(recv_tokens.index_select(0, order), valid_sorted)
            self.lease.expect_consumer()
            y_sorted = _ChunkExperts.apply(
                x, self.w1_eff, self.w2_eff, self.meta, self.lease,
                slot_offsets(group_sizes, order.shape[0]),
            )
            y_sorted = mask_rows(y_sorted, valid_sorted)
            return scatter_rows(y_sorted, order, T)

        if cap is None:
            raise ValueError("the padded expert path needs a host-static cap")
        safe_slot = slot_sorted.clamp(max=n_slot - 1)
        seg_start = torch.zeros(n_slot, dtype=torch.int64, device=device)
        if n_slot > 1:
            seg_start[1:] = torch.cumsum(group_sizes, dim=0)[:-1]
        pos_in_slot = torch.arange(T, device=device, dtype=torch.int64) - seg_start[safe_slot]
        overflow = n_slot * cap
        flat_idx = torch.where(
            valid_sorted,
            safe_slot * cap + pos_in_slot.clamp(max=cap - 1),
            torch.full_like(pos_in_slot, overflow),
        )

        x_ext = recv_tokens.new_zeros((overflow + 1, H))
        x_ext = x_ext.index_copy(0, flat_idx, recv_tokens[order])
        x_pad = x_ext[:overflow].view(n_slot, cap, H)

        self.lease.expect_consumer()
        y_pad = _ChunkExperts.apply(x_pad, self.w1_eff, self.w2_eff, self.meta, self.lease, None)
        out_sorted = y_pad.reshape(overflow, H)[flat_idx.clamp(max=overflow - 1)]
        out_sorted = out_sorted * valid_sorted.unsqueeze(1).to(out_sorted.dtype)
        return scatter_rows(out_sorted, order, T)


def overlapped_grouped_expert_mlp(
    recv_tokens: torch.Tensor,
    recv_slot: torch.Tensor,
    group_sizes: torch.Tensor,
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    slot_to_e: torch.Tensor,
    main_rank: torch.Tensor,
    replicated: Sequence[int],
    weight_shapes: Sequence[torch.Size],
    cap: Optional[int] = None,
    *,
    gated: bool,
    act: Callable,
    transpose_w: bool,
    my_rank: int,
    n_slot: int,
    group=None,
    pool: WeightPool = _POOL,
    transport: "Optional[ReplicaTransport]" = None,
    valid_mask: Optional[torch.Tensor] = None,
    max_recv_rows: Optional[int] = None,
) -> torch.Tensor:
    """Single-chunk :class:`OverlappedExperts`: grouped expert MLP whose backward re-acquires the weights.

    Args:
        recv_tokens: float ``[T, H]`` tokens received by this rank (any order).
        recv_slot: int64 ``[T]`` local physical-slot id of each token.
        group_sizes: int64 ``[n_slot]`` token count per slot (sum == T).
        weights_local: ``{e: (W1, W2)}`` resident params for experts this rank is main of.
        slot_to_e: int64 ``[n_slot]`` logical expert hosted at each local slot (-1 if empty).
        main_rank: int64 ``[E]`` group-local main rank of each expert.
        replicated: experts with more than one replica (need broadcast + reduce).
        weight_shapes: per-expert ``[(W1_shape), (W2_shape)]`` in parameter layout.
        cap: per-slot capacity (host-static); only needed on the padded fallback path.
        gated: whether GEMM-1 is gated (SwiGLU-style).
        act: activation function.
        transpose_w: True if weights are stored ``[out, in]`` (Megatron) and used as ``x @ W.t()``.
        my_rank: this rank's group-local id.
        n_slot: number of local physical slots.
        group: EP process group.
        pool: scratch buffer pool reused across layers.
        transport: replica-weight transport (defaults to :class:`BroadcastReplicaTransport`).
        max_recv_rows: host-static bound on total received rows; shrinks the ragged path's
            tensors below the transport's worst case. See :func:`grouped_mlp.compact_rows`.

    Returns:
        float ``[T, H]`` expert outputs in the original ``recv_tokens`` order.
    """
    layer = OverlappedExperts(
        weights_local=weights_local, slot_to_e=slot_to_e, main_rank=main_rank, replicated=replicated,
        weight_shapes=weight_shapes, gated=gated, act=act, transpose_w=transpose_w, my_rank=my_rank,
        n_slot=n_slot, dtype=recv_tokens.dtype, device=recv_tokens.device, group=group, pool=pool,
        transport=transport,
    )
    return layer.chunk(recv_tokens, recv_slot, group_sizes, cap, valid_mask, max_recv_rows)

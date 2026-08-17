"""Hand-scheduled MoE expert pipeline: one autograd node for dispatch -> expert GEMM -> combine.

Autograd's engine decides node order from sequence numbers and replays each backward on the stream its
forward ran on. That is enough to get the token chunks to pipeline, but it fixes two things we would
rather choose: the reduce-to-main of the replica gradients is forced to run *last* in the layer (it is
the node closest to the parameters, so every chunk's Wgrad must reach it first), and the replica-weight
pull can only start at a node boundary. Taking the whole pipeline into a single
:class:`_ManualExpertPipeline` node lets the backward be written out in the order we want:

* the replica-weight pull is issued first, on the weight stream, so it covers the reverse combine;
* ``combine^-1`` for every chunk is enqueued up front, so ``combine^-1(c2)`` overlaps ``expert^-1(c1)``;
* inside a chunk, Dgrad runs before Wgrad and ``dispatch^-1(k)`` is issued between them, so the token
  channel is unblocked a Wgrad earlier and the Wgrads then run under that transfer;
* the reduce-to-main trails on the weight stream, behind the last ``dispatch^-1``: it is the one
  transfer nothing in this layer's backward waits on, so it is scheduled last on purpose.

The forward gets the mirror treatment: the weight pull is issued on the weight stream before the first
dispatch, so it hides behind ``dispatch(c1)`` rather than sitting exposed ahead of it.

Scope: only the four things worth hand-scheduling are inside the node -- the two all-to-alls, the expert
GEMMs, and the replica weight/grad movement. Token gather/scatter, the gate-weight multiply and the
payload concat stay outside in ordinary autograd, where they are cheap, well-tested, and carry no
collective.

Overlap caveat for the ``dist.broadcast``/``dist.reduce`` transport: its reduce runs on the *same* NCCL
communicator as the token all-to-alls, and NCCL serialises same-communicator work regardless of the CUDA
stream it was enqueued on, so there the reduce/``dispatch^-1`` overlap is host-side only. The GIN
transport moves weights on its own device-initiated channel, so the overlap is real there.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import profiling
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
from .overlap import (
    _PrefetchOnBackward,
    _ReplicaLease,
    _acquire_stacks,
    _activation,
    _comm_stream,
    _remote_transfer_bytes,
    _weight_stream,
    build_meta,
)


def _sctx(stream):
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def _rec(t: torch.Tensor, stream) -> None:
    if stream is not None and t.is_cuda:
        t.record_stream(stream)


def _event(on_cuda: bool, stream):
    if not on_cuda:
        return None
    e = torch.cuda.Event()
    e.record(stream)
    return e


def _slot_layout(
    recv_slot: torch.Tensor,
    group_sizes: torch.Tensor,
    n_slot: int,
    cap: Optional[int],
    valid_mask: Optional[torch.Tensor] = None,
    ragged: bool = False,
    max_recv_rows: Optional[int] = None,
):
    """Group received tokens by physical slot, ragged or padded.

    Returns:
        ``(order, valid_sorted, flat, offs)``. Ragged mode leaves ``flat`` None and returns the
        grouped GEMM's row offsets; padded mode leaves ``offs`` None and returns the destination
        index into the ``[n_slot * cap + 1]`` bucket layout. ``max_recv_rows`` trims the ragged
        order to a host-static budget; see :func:`grouped_mlp.compact_rows`.
    """
    T, device = recv_slot.shape[0], recv_slot.device
    order, valid_sorted, slot_sorted = slot_sort(recv_slot, n_slot, valid_mask)
    if ragged:
        order, valid_sorted = compact_rows(order, valid_sorted, group_sizes, max_recv_rows)
        return order, valid_sorted, None, slot_offsets(group_sizes, order.shape[0])

    if cap is None:
        raise ValueError("the padded expert path needs a host-static cap")
    safe_slot = slot_sorted.clamp(max=n_slot - 1)
    seg_start = torch.zeros(n_slot, dtype=torch.int64, device=device)
    if n_slot > 1:
        seg_start[1:] = torch.cumsum(group_sizes, dim=0)[:-1]
    pos_in_slot = torch.arange(T, device=device, dtype=torch.int64) - seg_start[safe_slot]
    overflow = n_slot * cap
    flat = torch.where(
        valid_sorted,
        safe_slot * cap + pos_in_slot.clamp(max=cap - 1),
        torch.full_like(pos_in_slot, overflow),
    )
    return order, valid_sorted, flat, None


class _ManualExpertPipeline(torch.autograd.Function):
    """dispatch -> expert 2-GEMM -> combine for every chunk, with both directions hand-scheduled.

    Differentiable inputs are the per-chunk payloads and this rank's main expert weights; outputs are
    the per-chunk combined results. The replica weight stacks are intentionally not saved: backward
    re-acquires them through the shared lease, which is what keeps the resident cost at one layer's
    slots instead of the whole stack of layers.
    """

    @staticmethod
    def forward(ctx, cfg, *tensors):
        nc = int(cfg["nc"])
        payloads, main_w = list(tensors[:nc]), list(tensors[nc:])
        meta, adapter, group = cfg["meta"], cfg["adapter"], cfg["group"]
        cap = None if cfg["cap"] is None else int(cfg["cap"])
        max_recv_rows = cfg.get("max_recv_rows")
        n_slot, H = int(cfg["n_slot"]), int(cfg["H"])
        my_rank, sent, recv, routes = (
            int(cfg["my_rank"]), cfg["sent"], cfg["recv"], cfg["routes"]
        )
        dispatch_bytes, combine_bytes = cfg["dispatch_bytes"], cfg["combine_bytes"]
        device, dtype = payloads[0].device, payloads[0].dtype

        cs, ws = _comm_stream(device), _weight_stream(device)
        on_cuda = cs is not None
        ms = torch.cuda.current_stream(device) if on_cuda else None

        # Weight pull first, on the weight stream: it now hides behind chunk 1's dispatch below rather
        # than being an exposed prologue to the layer.
        with profiling.record(
            "apply/weight_move",
            time_it=True,
            device=device,
            stream=ws,
            payload_bytes=_remote_transfer_bytes(meta),
        ):
            w1_eff, w2_eff = _acquire_stacks(meta, main_w, dtype, device, ws)

        ragged = ragged_enabled(payloads[0]) and ragged_expert_ok(
            H, payloads[0].element_size(), w1_eff, w2_eff, meta["gated"]
        )

        if on_cuda:
            cs.wait_stream(ms)                      # payloads were written on the compute stream
        recv_pl: List[Optional[torch.Tensor]] = [None] * nc
        disp_evt: List[Optional[torch.cuda.Event]] = [None] * nc
        with _sctx(cs):
            for k in range(nc):
                with profiling.record(
                    "apply/dispatch",
                    time_it=True,
                    device=device,
                    stream=cs,
                    payload_bytes=dispatch_bytes[k],
                ):
                    recv_pl[k] = adapter.dispatch_chunk(
                        payloads[k], sent[k], recv[k], group, tag=k,
                        route_idx=routes[k], n_slot=n_slot, cap=cap,
                    )
                if on_cuda:
                    _rec(recv_pl[k], ms)
                    disp_evt[k] = _event(on_cuda, cs)

        saved: List[torch.Tensor] = []
        idx: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        comb: List[Optional[torch.Tensor]] = [None] * nc
        for k in range(nc):
            if on_cuda:
                ms.wait_event(disp_evt[k])
                if k == 0:
                    ms.wait_stream(ws)              # replica slots are read from here on
            rp = recv_pl[k]
            recv_tokens = rp[:, :H].contiguous()
            valid_mask = None
            if adapter.uses_padded_layout():
                recv_slot, valid_mask, group_sizes = adapter.recv_layout(k)
            else:
                recv_slot = rp[:, H].round().to(torch.int64) - my_rank * n_slot
                group_sizes = torch.bincount(
                    recv_slot.clamp(min=0, max=n_slot - 1), minlength=n_slot
                ).to(torch.int64)
            order, valid_sorted, flat, offs = _slot_layout(
                recv_slot, group_sizes, n_slot, cap, valid_mask, ragged, max_recv_rows
            )

            T = recv_tokens.shape[0]
            if ragged:
                # The sort already packed the rows slot-major, so it doubles as the grouping and
                # the [n_slot * cap, H] staging buffer disappears entirely.
                x = recv_tokens.index_select(0, order)
            else:
                overflow = n_slot * cap
                x_ext = recv_tokens.new_zeros((overflow + 1, H))
                x_ext.index_copy_(0, flat, recv_tokens[order])
                x = x_ext[:overflow].view(n_slot, cap, H)
            with profiling.record("apply/expert_gemm", time_it=True, device=device):
                h_pre = ragged_mm(x, w1_eff, offs)
                y_out = ragged_mm(_activation(meta, h_pre), w2_eff, offs)
            if ragged:
                y_sorted = mask_rows(y_out, valid_sorted)
            else:
                y_sorted = y_out.reshape(n_slot * cap, H)[flat.clamp(max=n_slot * cap - 1)]
                y_sorted = y_sorted * valid_sorted.unsqueeze(1).to(y_sorted.dtype)
            y = scatter_rows(y_sorted, order, T)

            saved.extend([x, h_pre])
            idx.append((order, flat, valid_sorted, offs))
            comp_evt = _event(on_cuda, ms)
            if on_cuda:
                _rec(y, cs)
            with _sctx(cs):
                if on_cuda:
                    cs.wait_event(comp_evt)
                with profiling.record(
                    "apply/combine",
                    time_it=True,
                    device=device,
                    stream=cs,
                    payload_bytes=combine_bytes[k],
                ):
                    comb[k] = adapter.combine_chunk(y, sent[k], recv[k], group, tag=k)
                if on_cuda:
                    _rec(comb[k], ms)
            cfg["state"].append(adapter.chunk_state(k))   # replay info for this chunk's transpose legs

        if on_cuda:
            ms.wait_stream(cs)
        ctx.cfg = cfg
        ctx.idx = idx
        ctx.save_for_backward(*saved)
        return tuple(comb)

    @staticmethod
    def backward(ctx, *grad_comb):
        cfg, meta = ctx.cfg, ctx.cfg["meta"]
        nc, adapter, group = int(cfg["nc"]), cfg["adapter"], cfg["group"]
        cap = None if cfg["cap"] is None else int(cfg["cap"])
        n_slot, H = int(cfg["n_slot"]), int(cfg["H"])
        pad_cols, lease, state = int(cfg["pad_cols"]), cfg["lease"], cfg["state"]
        dispatch_bytes, combine_bytes = cfg["dispatch_bytes"], cfg["combine_bytes"]
        saved = ctx.saved_tensors
        device, dtype = grad_comb[0].device, grad_comb[0].dtype

        cs, ws = _comm_stream(device), _weight_stream(device)
        on_cuda = cs is not None
        ms = torch.cuda.current_stream(device) if on_cuda else None

        # 1. weight pull (a no-op if the block's backward pre-hook already started it)
        lease.start()

        # 2. combine^-1 for every chunk up front, so combine^-1(c2) overlaps expert^-1(c1)
        if on_cuda:
            cs.wait_stream(ms)
        gy: List[Optional[torch.Tensor]] = [None] * nc
        comb_evt: List[Optional[torch.cuda.Event]] = [None] * nc
        with _sctx(cs):
            for k in range(nc):
                with profiling.record(
                    "apply/combine_bwd",
                    time_it=True,
                    device=device,
                    stream=cs,
                    payload_bytes=combine_bytes[k],
                ):
                    gy[k] = adapter.combine_chunk_bwd(
                        grad_comb[k].contiguous(), state[k], group
                    )
                if on_cuda:
                    _rec(gy[k], ms)
                    comb_evt[k] = _event(on_cuda, cs)

        # 3. per-chunk expert backward, Dgrad before Wgrad. Dgrad is the only half anything waits on --
        #    it produces grad_x, which the token channel needs -- so it runs first and dispatch^-1(k)
        #    goes out the moment it lands. The Wgrads feed only the parameter reduction, so they run
        #    afterwards, under that dispatch^-1. Ordering them the other way (Wgrad first, which does
        #    not need the weight and so cushions a late pull) buys nothing once the pull is prefetched
        #    at the block output, and it delays every dispatch^-1 by a Wgrad.
        #
        # The Wgrads are produced straight into *parameter* layout. Under Megatron's [out, in] weights
        # the natural product comes out transposed, and the transport has to stage a contiguous copy
        # before it can reinterpret the bytes -- a full [n_slot x |W_e|] copy per direction per layer.
        # Swapping the operands gives the same numbers ((A^T B)^T == B^T A) already contiguous.
        transpose_w = meta["transpose_w"]
        gw1 = gw2 = None
        grad_pl: List[Optional[torch.Tensor]] = [None] * nc
        for k in range(nc):
            if on_cuda:
                ms.wait_event(comb_evt[k])
            x, h_pre = saved[2 * k], saved[2 * k + 1]
            order, flat, valid_sorted, offs = ctx.idx[k]
            T = gy[k].shape[0]

            if offs is not None:
                g = mask_rows(gy[k].index_select(0, order), valid_sorted)
            else:
                overflow = n_slot * cap
                g_ext = gy[k].new_zeros((overflow + 1, H))
                g_ext.index_add_(0, flat, gy[k].index_select(0, order))
                g = g_ext[:overflow].view(n_slot, cap, H)

            # --- Dgrad chain: needs the replica weights, so this is where a late pull would stall ---
            w1_eff, w2_eff = lease.wait()
            with profiling.record(
                "apply/expert_bwd_dgrad",
                time_it=True,
                device=device,
                stream=ms,
            ):
                grad_a = ragged_mm(g, w2_eff.transpose(1, 2), offs)
            with profiling.record(
                "apply/activation_bwd",
                time_it=True,
                device=device,
                stream=ms,
            ):
                with torch.enable_grad():
                    hp = h_pre.detach().requires_grad_(True)
                    a_g = _activation(meta, hp)
                    (grad_h_pre,) = torch.autograd.grad(a_g, hp, grad_a)
            with profiling.record(
                "apply/expert_bwd_dgrad",
                time_it=True,
                device=device,
                stream=ms,
            ):
                gx_out = ragged_mm(grad_h_pre, w1_eff.transpose(1, 2), offs)
            if offs is not None:
                # Rows past the last offset are slots the GEMM never wrote; select, do not multiply.
                gx_sorted = mask_rows(gx_out, valid_sorted)
            else:
                overflow = n_slot * cap
                gx_sorted = gx_out.reshape(overflow, H)[flat.clamp(max=overflow - 1)]
                gx_sorted = gx_sorted * valid_sorted.unsqueeze(1).to(gx_sorted.dtype)
            gx = scatter_rows(gx_sorted, order, T)
            # the payload's trailing columns carried the physical ids, a constant -> zero grad
            g_rp = torch.cat([gx, gx.new_zeros((T, pad_cols))], dim=1)
            comp_evt = _event(on_cuda, ms)
            if on_cuda:
                _rec(g_rp, cs)
            with _sctx(cs):
                if on_cuda:
                    cs.wait_event(comp_evt)
                with profiling.record(
                    "apply/dispatch_bwd",
                    time_it=True,
                    device=device,
                    stream=cs,
                    payload_bytes=dispatch_bytes[k],
                ):
                    grad_pl[k] = adapter.dispatch_chunk_bwd(g_rp, state[k], group)
                if on_cuda:
                    _rec(grad_pl[k], ms)

            # --- Wgrads: nothing downstream waits on them, so they run under dispatch^-1(k) ---
            with profiling.record(
                "apply/expert_bwd_wgrad",
                time_it=True,
                device=device,
                stream=ms,
            ):
                a = a_g.detach()                        # the activation the Dgrad chain already built
                g2 = ragged_mm_tn(g, a, offs) if transpose_w else ragged_mm_tn(a, g, offs)
                gw2 = g2 if gw2 is None else gw2.add_(g2)  # matmul output, unaliased -> safe in place
                g1 = ragged_mm_tn(grad_h_pre, x, offs) if transpose_w \
                    else ragged_mm_tn(x, grad_h_pre, offs)
                gw1 = g1 if gw1 is None else gw1.add_(g1)

        # 4. every chunk's Wgrad is in -> reduce to main(e), on the weight stream and behind the last
        #    dispatch^-1, which is still draining. Nothing in this layer's backward waits on it.
        if on_cuda:
            ws.wait_stream(ms)
            _rec(gw1, ws)
            _rec(gw2, ws)
        with _sctx(ws):
            with profiling.record(
                "apply/grad_move",
                time_it=True,
                device=device,
                stream=ws,
                payload_bytes=_remote_transfer_bytes(meta),
            ):
                grads = meta["transport"].reduce_grads(meta, gw1, gw2, dtype, device)
            if on_cuda:
                for g in grads:
                    _rec(g, ms)

        lease.release()
        if on_cuda:
            ms.wait_stream(cs)
            ms.wait_stream(ws)
        profiling.finish_debug_interval(
            "apply/backward_total", lease.debug_backward_start, stream=ms
        )
        lease.debug_backward_start = None
        profiling.emit_backward_debug(cfg.get("debug_context", ""))
        return (None, *grad_pl, *grads)


class ManualMoEBlock:
    """Driver for :class:`_ManualExpertPipeline`: owns the layer's metadata, lease and weight list."""

    def __init__(
        self,
        *,
        weights_local: Dict[int, Tuple[torch.Tensor, ...]],
        slot_to_e: torch.Tensor,
        main_rank: torch.Tensor,
        replicated: Sequence[int],
        weight_shapes: Sequence[torch.Size],
        gated: bool,
        act,
        transpose_w: bool,
        my_rank: int,
        n_slot: int,
        dtype: torch.dtype,
        device,
        group=None,
        transport=None,
        debug_context: str = "",
    ) -> None:
        self.meta, self.main_w = build_meta(
            weights_local=weights_local, slot_to_e=slot_to_e, main_rank=main_rank,
            replicated=replicated, weight_shapes=weight_shapes, gated=gated, act=act,
            transpose_w=transpose_w, my_rank=my_rank, n_slot=n_slot, group=group,
            transport=transport,
        )
        self.lease = _ReplicaLease(self.meta, self.main_w, dtype, device)
        self.lease.debug_interval_enabled = True
        self.debug_context = str(debug_context)

    def prefetch_on_backward_of(self, out: torch.Tensor) -> torch.Tensor:
        """Tag the block output so its backward starts this layer's weight pull before anything else."""
        return _PrefetchOnBackward.apply(out, self.lease)

    def run(
        self,
        payloads: Sequence[torch.Tensor],
        sent: Sequence[torch.Tensor],
        recv: Sequence[torch.Tensor],
        routes: Sequence[torch.Tensor],
        dispatch_bytes: Sequence[torch.Tensor],
        combine_bytes: Sequence[torch.Tensor],
        *,
        adapter,
        group,
        cap: Optional[int],
        n_slot: int,
        H: int,
        pad_cols: int,
        my_rank: int,
        max_recv_rows: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        cfg = {
            "nc": len(payloads), "meta": self.meta, "adapter": adapter, "group": group,
            "cap": cap, "max_recv_rows": max_recv_rows,
            "n_slot": n_slot, "H": H, "pad_cols": pad_cols, "my_rank": my_rank,
            "sent": list(sent), "recv": list(recv), "routes": list(routes),
            "dispatch_bytes": list(dispatch_bytes), "combine_bytes": list(combine_bytes),
            "lease": self.lease, "state": [], "debug_context": self.debug_context,
        }
        # one consumer for the lease: the single node now owns every chunk's backward
        self.lease.expect_consumer()
        return _ManualExpertPipeline.apply(cfg, *payloads, *self.main_w)

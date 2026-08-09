"""EPLB apply-mode manager: the sync-free Phase C MoE forward and its swappable transport.

Pipeline (per MoE layer, per forward). Two concerns are kept orthogonal:

  * TRANSPORT  - moving tokens between ranks (all-to-all), behind :class:`CommAdapter`
                 (``AllToAllAdapter`` fallback / ``DeepEPAdapter``).
  * REPLICATION - broadcasting a replicated expert's weight from its main owner and
                 reducing that expert's grad back to main. This lives in the compute
                 stage (``broadcast_from_main`` / overlap) and is adapter-independent.
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from . import profiling
from ..plan import Plan
from ..problem import ProblemSpec
from .comm import a2a_raw, all_to_all_single, broadcast_from_main
from .gin_weights import GinReplicaTransport, GinWeightReplicator, gin_enabled
from .grouped_mlp import grouped_expert_mlp
from .manual_block import ManualMoEBlock
from .overlap import OverlappedExperts, _comm_stream, overlapped_grouped_expert_mlp
from .physical import assign_physical


def _env_truthy(name: str, default: str = "0") -> bool:
    """Truthy parse of an on/off environment toggle."""
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "no")


# Cache one GIN replicator per (group, layout, dtype) so the symmetric buffers + static main(e)
# layout are allocated once and recycled across layers/steps.
_GIN_REPLICATORS: Dict[tuple, GinWeightReplicator] = {}
_ELASTIC_BUFFERS: Dict[tuple, object] = {}


def _get_gin_replicator(group, spec, weight_shapes, dtype, device) -> GinWeightReplicator:
    key = (
        id(group), int(spec.n_slot), int(spec.num_experts),
        tuple(tuple(s) for s in weight_shapes), dtype,
    )
    r = _GIN_REPLICATORS.get(key)
    if r is None:
        r = GinWeightReplicator(
            group=group, num_experts=int(spec.num_experts), n_slot=int(spec.n_slot),
            main_rank=spec.main_rank, weight_shapes=weight_shapes, dtype=dtype, device=device,
        )
        _GIN_REPLICATORS[key] = r
    return r


# ============================== Transport adapters ==============================
# A CommAdapter is the only place a token-channel all-to-all happens. Swapping the
# adapter changes the transport but not the routing/compute math above it.


# ElasticBuffer kernels move int4 vectors, so every token row must be 16-byte aligned.
DEEPEP_ROW_ALIGN = 16


def _payload_pad_cols(h: int, elem: int) -> int:
    """Columns appended after the ``H`` token columns: they carry the physical id and pad the row.

    Always at least one column, and enough of them to land the row on ``DEEPEP_ROW_ALIGN``.
    """
    pad_bytes = (DEEPEP_ROW_ALIGN - (h * elem) % DEEPEP_ROW_ALIGN) % DEEPEP_ROW_ALIGN
    return (pad_bytes or DEEPEP_ROW_ALIGN) // elem


class CommAdapter(Protocol):
    """Differentiable all-to-all transport seam taking device-side split sizes."""

    def all_to_all(
        self,
        inp: torch.Tensor,
        out_splits: torch.Tensor,
        in_splits: torch.Tensor,
        group,
    ) -> torch.Tensor:
        ...


class AllToAllAdapter:
    """Tested fallback over ``torch.distributed.all_to_all_single`` (moves splits to host)."""

    def __init__(self) -> None:
        # per-chunk (out_splits, in_splits) of each leg, so a hand-written backward can run the
        # transpose without a second D2H; claimed via chunk_state(tag)
        self._state: Dict[int, Dict[str, Tuple[List[int], List[int]]]] = {}

    def all_to_all(self, inp, out_splits, in_splits, group) -> torch.Tensor:
        # NCCL/Gloo need host-side split lists; this .tolist() is the one allowed D2H here
        return all_to_all_single(inp, out_splits.tolist(), in_splits.tolist(), group)

    # ---- two-chunk pipeline hooks (symmetric: dispatch and combine are both a plain a2a) ----
    def needs_recv_counts(self) -> bool:
        """all_to_all_single needs the per-src recv counts on host, so the caller must supply them."""
        return True

    def dispatch_chunk(
        self, payload, sent_per_dst, recv_per_src, group, tag: int = 0,
        *, route_idx=None, n_slot=None, cap=None,
    ):
        splits = (recv_per_src.tolist(), sent_per_dst.tolist())
        self._state.setdefault(tag, {})["disp"] = splits
        return all_to_all_single(payload, splits[0], splits[1], group)

    def combine_chunk(
        self, y, sent_per_dst, recv_per_src, group, tag: int = 0,
        *, route_idx=None, n_slot=None, cap=None,
    ):
        # reverse leg: send back what we received, receive back what we sent
        splits = (sent_per_dst.tolist(), recv_per_src.tolist())
        self._state.setdefault(tag, {})["comb"] = splits
        return all_to_all_single(y, splits[0], splits[1], group)

    def uses_padded_layout(self) -> bool:
        """Whether dispatch returns a fixed worst-case tensor with invalid rows."""
        return False

    # ---- transpose legs for callers that schedule the backward themselves ----
    def chunk_state(self, tag: int):
        """Take this chunk's replay state; the caller keeps it alive until its backward runs."""
        return self._state.pop(tag, None)

    def dispatch_chunk_bwd(self, grad_recv, state, group):
        out_splits, in_splits = state["disp"]
        return a2a_raw(grad_recv, in_splits, out_splits, group)

    def combine_chunk_bwd(self, grad_comb, state, group):
        out_splits, in_splits = state["comb"]
        return a2a_raw(grad_comb, in_splits, out_splits, group)


class _ElasticDispatch(torch.autograd.Function):
    """ElasticBuffer dispatch whose transpose is combine."""

    @staticmethod
    def forward(ctx, inp, buffer, topk_idx, num_experts, max_tokens, holder):
        recv, recv_topk_idx, _, handle, _ = buffer.dispatch(
            x=inp.contiguous(),
            topk_idx=topk_idx,
            num_experts=int(num_experts),
            num_max_tokens_per_rank=int(max_tokens),
            expert_alignment=1,
            do_handle_copy=True,
            do_cpu_sync=False,
            do_expand=False,
        )
        if recv_topk_idx is None:
            raise RuntimeError("ElasticBuffer non-expand dispatch did not return expert indices")
        ctx.buffer = buffer
        ctx.handle = handle
        holder["handle"] = handle
        holder["recv_topk_idx"] = recv_topk_idx
        return recv

    @staticmethod
    def backward(ctx, grad_recv):
        grad_in, _, _ = ctx.buffer.combine(x=grad_recv.contiguous(), handle=ctx.handle)
        return grad_in, None, None, None, None, None


class _ElasticCombine(torch.autograd.Function):
    """ElasticBuffer combine whose transpose is cached non-expand dispatch."""

    @staticmethod
    def forward(ctx, inp, buffer, handle):
        out, _, _ = buffer.combine(x=inp.contiguous(), handle=handle)
        ctx.buffer = buffer
        ctx.handle = handle
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in, _, _, _, _ = ctx.buffer.dispatch(
            x=grad_out.contiguous(), handle=ctx.handle, do_cpu_sync=False, do_expand=False
        )
        return grad_in, None, None


class DeepEPAdapter:
    """Zero-sync token transport over DeepEP V2 ``ElasticBuffer``."""

    def __init__(self, max_tokens_per_rank: Optional[int] = None):
        try:
            import deep_ep
        except Exception as e:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "DeepEPAdapter requires DeepEP V2, PyTorch>=2.10, NCCL>=2.30.4 and SM90+"
            ) from e
        if not hasattr(deep_ep, "ElasticBuffer"):
            raise RuntimeError("installed DeepEP does not export ElasticBuffer")
        self._deep_ep = deep_ep
        env_max = os.environ.get("EPLB_DEEPEP_MAX_TOKENS_PER_RANK")
        self._requested_max_tokens = int(
            max_tokens_per_rank if max_tokens_per_rank is not None else (env_max or 0)
        )
        if self._requested_max_tokens < 0:
            raise ValueError("EPLB_DEEPEP_MAX_TOKENS_PER_RANK must be non-negative")
        self._allow_hybrid = _env_truthy("EPLB_DEEPEP_HYBRID", "1")
        self._buffer = None
        self._group = None
        self._hidden = 0
        self._max_tokens = 0
        self._handles: Dict[int, Dict[str, object]] = {}
        self._state: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _deepep_eligible(inp: torch.Tensor) -> bool:
        """Elastic dispatch accepts contiguous BF16 rows aligned to one int4."""
        return (
            inp.dim() == 2
            and inp.dtype == torch.bfloat16
            and inp.shape[1] * inp.element_size() % DEEPEP_ROW_ALIGN == 0
        )

    def _get_buffer(self, group, payload: torch.Tensor):
        if group is None and dist.is_initialized():
            group = dist.distributed_c10d._get_default_group()
        max_tokens = self._requested_max_tokens or int(payload.shape[0])
        hidden = int(payload.shape[1])
        if int(payload.shape[0]) > max_tokens:
            raise ValueError(
                f"ElasticBuffer input has {payload.shape[0]} rows, above max_tokens_per_rank={max_tokens}"
            )
        if self._buffer is not None:
            if group is not self._group or hidden != self._hidden:
                raise RuntimeError("DeepEPAdapter cannot change process group or hidden size after initialization")
            if int(payload.shape[0]) > self._max_tokens:
                raise ValueError(
                    f"ElasticBuffer input has {payload.shape[0]} rows, above initialized maximum {self._max_tokens}"
                )
            return self._buffer

        key = (
            id(group),
            payload.device.type,
            payload.device.index,
            hidden,
            max_tokens,
            self._allow_hybrid,
        )
        buffer = _ELASTIC_BUFFERS.get(key)
        if buffer is None:
            buffer = self._deep_ep.ElasticBuffer(
                group,
                num_max_tokens_per_rank=max_tokens,
                hidden=hidden,
                num_topk=1,
                deterministic=False,
                allow_hybrid_mode=self._allow_hybrid,
            )
            _ELASTIC_BUFFERS[key] = buffer
        self._buffer = buffer
        self._group = group
        self._hidden = hidden
        self._max_tokens = max_tokens
        return buffer

    def _dispatch(
        self,
        payload: torch.Tensor,
        route_idx: torch.Tensor,
        n_slot: int,
        cap: int,
        group,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if not self._deepep_eligible(payload):
            raise TypeError(
                "ElasticBuffer token rows must be contiguous BF16 with a 16-byte-aligned width"
            )
        if route_idx is None:
            raise ValueError("ElasticBuffer dispatch requires physical route indices")
        R = dist.get_world_size(group) if dist.is_initialized() else 1
        num_experts = R * int(n_slot)
        if num_experts > 2048 or int(n_slot) > 256:
            raise ValueError(
                f"ElasticBuffer synthetic expert layout exceeds DeepEP limits: "
                f"experts={num_experts}, per_rank={n_slot}"
            )
        topk_dtype = getattr(self._deep_ep, "topk_idx_t", torch.int64)
        topk_idx = route_idx.reshape(-1, 1).to(dtype=topk_dtype).contiguous()
        buffer = self._get_buffer(group, payload)
        holder: Dict[str, object] = {}
        recv = _ElasticDispatch.apply(
            payload,
            buffer,
            topk_idx,
            num_experts,
            self._max_tokens,
            holder,
        )

        handle = holder["handle"]
        psum_expert = handle.psum_num_recv_tokens_per_expert.to(torch.int64)
        starts = torch.zeros_like(psum_expert)
        if psum_expert.shape[0] > 1:
            starts[1:] = psum_expert[:-1]
        group_sizes = psum_expert - starts
        total_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].to(torch.int64)
        rows = torch.arange(recv.shape[0], device=recv.device, dtype=torch.int64)
        valid = rows < total_recv
        recv_topk_idx = holder["recv_topk_idx"]
        recv_slot = torch.where(valid, recv_topk_idx[:, 0].to(torch.int64), 0)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(
                torch.all(group_sizes <= int(cap)),
                "ElasticBuffer: EPLB_CAP is below a physical slot's received-token count",
            )
        holder.update(
            recv_slot=recv_slot,
            valid=valid,
            group_sizes=group_sizes,
        )
        return recv, holder

    def all_to_all(self, inp, out_splits, in_splits, group):
        raise RuntimeError("ElasticBuffer requires physical route indices; use dispatch_chunk/combine_chunk")

    def needs_recv_counts(self) -> bool:
        """ElasticBuffer derives receive counts on device."""
        return False

    def uses_padded_layout(self) -> bool:
        """ElasticBuffer returns a fixed worst-case row count."""
        return True

    def recv_layout(self, tag: int):
        """Return device-only ``(slot, valid, group_sizes)`` for a dispatched chunk."""
        holder = self._handles[tag]
        return holder["recv_slot"], holder["valid"], holder["group_sizes"]

    def dispatch_chunk(
        self, payload, sent_per_dst, recv_per_src, group, tag: int = 0,
        *, route_idx=None, n_slot=None, cap=None,
    ):
        if n_slot is None or cap is None:
            raise ValueError("ElasticBuffer dispatch requires n_slot and cap")
        recv, holder = self._dispatch(payload, route_idx, int(n_slot), int(cap), group)
        self._handles[tag] = holder
        self._state.setdefault(tag, {})["disp"] = holder
        return recv

    def combine_chunk(
        self, y, sent_per_dst, recv_per_src, group, tag: int = 0,
        *, route_idx=None, n_slot=None, cap=None,
    ):
        holder = self._handles.pop(tag)
        holder["combine_hidden"] = int(y.shape[1])
        if y.shape[1] > self._hidden:
            raise ValueError(
                f"ElasticBuffer combine width {y.shape[1]} exceeds dispatch width {self._hidden}"
            )
        if y.shape[1] < self._hidden:
            padded = y.new_zeros((y.shape[0], self._hidden))
            padded[:, :y.shape[1]] = y
            y = padded
        self._state.setdefault(tag, {})["comb"] = holder
        combined = _ElasticCombine.apply(y, self._buffer, holder["handle"])
        return combined[:, :holder["combine_hidden"]]

    def chunk_state(self, tag: int):
        """Take this chunk's handles for a manually scheduled backward."""
        return self._state.pop(tag, None)

    def dispatch_chunk_bwd(self, grad_recv, state, group):
        return self._buffer.combine(
            x=grad_recv.contiguous(), handle=state["disp"]["handle"]
        )[0]

    def combine_chunk_bwd(self, grad_comb, state, group):
        hidden = int(state["comb"]["combine_hidden"])
        if hidden < self._hidden:
            padded = grad_comb.new_zeros((grad_comb.shape[0], self._hidden))
            padded[:, :hidden] = grad_comb
            grad_comb = padded
        grad_recv = self._buffer.dispatch(
            x=grad_comb.contiguous(), handle=state["comb"]["handle"],
            do_cpu_sync=False, do_expand=False,
        )[0]
        return grad_recv[:, :hidden]


# ============================ Routing & grouping helpers ============================


def _split_sizes(plan: Plan, my_rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device-side split sizes for this rank from the routing quota ``plan.q``.

    Returns:
        ``(sent_per_dst, recv_per_src, recv_per_expert)`` int64 tensors:
        tokens this rank sends to each dst (sums to ``U``), receives from each src,
        and receives for each logical expert it hosts.
    """
    sent_per_dst = plan.q[my_rank].sum(dim=0).to(torch.int64)         # [R], sums to U
    recv_per_src = plan.q[:, :, my_rank].sum(dim=1).to(torch.int64)   # [R]
    recv_per_expert = plan.q[:, :, my_rank].sum(dim=0).to(torch.int64)  # [E]
    return sent_per_dst, recv_per_src, recv_per_expert


def _slot_tables(
    x: torch.Tensor, my_rank: int, n_slot: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sync-free slot tables for ``my_rank`` (no ``nonzero``/host ops).

    Returns three device tensors:
      * ``slot_to_e`` int64 ``[n_slot]``: logical expert at each local slot (-1 if empty),
        experts placed in ascending id order (matches the old ``nonzero`` ordering).
      * ``slot_of_e`` int64 ``[E]``: local slot index of expert ``e`` (valid only where hosted).
      * ``hosted``    bool  ``[E]``: whether ``my_rank`` holds an instance of expert ``e``.

    Built purely with ``cumsum``/``scatter``/``where`` so the whole thing stays on device.
    """
    E = x.shape[0]
    device = x.device
    col = x[:, my_rank].to(torch.int64)            # [E] in {0, 1}
    hosted = col.bool()                            # [E]
    slot_of_e = (col.cumsum(0) - 1).clamp_(min=0)  # [E]; ascending-id slot for hosted experts
    e_ids = torch.arange(E, device=device, dtype=torch.int64)
    # scatter each expert id into its slot; non-hosted experts go to an overflow bucket (index n_slot)
    dump = torch.full_like(slot_of_e, n_slot)
    tgt = torch.where(hosted, slot_of_e, dump)     # [E]
    ext = torch.full((n_slot + 1,), -1, dtype=torch.int64, device=device)
    ext.scatter_(0, tgt, e_ids)
    slot_to_e = ext[:n_slot].clone()
    return slot_to_e, slot_of_e, hosted


def _group_sizes_by_slot(
    slot_to_e: torch.Tensor, recv_per_expert: torch.Tensor, n_slot: int, device
) -> torch.Tensor:
    """int64 ``[n_slot]``: number of received tokens landing in each local slot."""
    valid_slot = slot_to_e >= 0
    group_sizes = torch.zeros(n_slot, dtype=torch.int64, device=device)
    group_sizes[valid_slot] = recv_per_expert[slot_to_e[valid_slot]]
    return group_sizes


# ===================== Expert compute (replication + grouped MLP) =====================


def _make_materialize_and_compute(
    *,
    replicated: Sequence[int],
    replicated_main: Sequence[int],
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    weight_shapes: Sequence[torch.Size],
    slot_of_e: torch.Tensor,
    hosted: torch.Tensor,
    recv_slot: torch.Tensor,
    group_sizes: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    batched_mlp_fn: Callable,
    cap: int,
    n_slot: int,
    dtype: torch.dtype,
    device,
    group,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the plain/Level-A compute closure: broadcast replica weights then run one batched MLP.

    Returned as a closure so :func:`torch.utils.checkpoint` can recompute the broadcasts in backward
    (Level A re-materialisation). Backward grads reduce to main(e) via ``broadcast_from_main``.

    Sync-free: the per-slot weight stack is assembled with device ``index_copy_`` using the
    precomputed ``slot_of_e`` / ``hosted`` tensors, so there is no per-slot ``.item()``. The only
    host-side inputs are the (already materialised) ``replicated`` / ``replicated_main`` control
    lists, which drive the per-expert broadcast collectives (``dist.broadcast`` needs a host ``src``).
    """
    replicated_set = set(replicated)

    def _materialize_and_compute(recv_tokens: torch.Tensor) -> torch.Tensor:
        # replicate: materialise each replicated expert's weight from its (static) main owner.
        # every rank enters these collectives in the same ascending-id order -> grads reduce to main.
        We: Dict[int, Tuple[torch.Tensor, ...]] = {}
        # Under Level A (checkpoint) this closure also runs in backward, so the recorded count is
        # forward + recompute -- that double cost is exactly what re-materialisation trades away.
        with profiling.record("apply/weight_move", time_it=True, device=device):
            for e, main_local in zip(replicated, replicated_main):
                local_w = weights_local.get(e)
                We[e] = tuple(
                    broadcast_from_main(
                        local_w[j] if local_w is not None else None,
                        weight_shapes[j], dtype, device, main_local, group,
                    )
                    for j in range(len(weight_shapes))
                )
        # stack per-slot weights via device scatter into a [n_slot + 1, *w] buffer; row n_slot is the
        # overflow bucket for weights not hosted on this rank (dropped by the trailing [:n_slot] slice).
        overflow = torch.full((1,), n_slot, dtype=torch.int64, device=device)
        w_stacked = []
        for j in range(len(weight_shapes)):
            buf = torch.zeros((n_slot + 1, *weight_shapes[j]), dtype=dtype, device=device)
            # resident (non-replicated) mains: keys are static, slot index is a device tensor
            for e, wt in weights_local.items():
                if e in replicated_set:
                    continue  # placed via the broadcast path below (grads must reduce to main)
                buf.index_copy_(0, slot_of_e[e:e + 1], wt[j].to(dtype).unsqueeze(0))
            # replicated experts: place into this rank's slot iff hosted, else into the overflow row
            for e in replicated:
                tgt = torch.where(hosted[e], slot_of_e[e:e + 1], overflow)
                buf.index_copy_(0, tgt, We[e][j].to(dtype).unsqueeze(0))
            w_stacked.append(buf[:n_slot])
        out_units_recv = grouped_expert_mlp(
            recv_tokens, recv_slot, group_sizes, tuple(w_stacked), batched_mlp_fn, cap,
            valid_mask=valid_mask,
        )
        # keepalive: make output depend on every broadcast so all ranks hit the matching reduce in backward
        keep = recv_tokens.sum() * 0.0
        for e in replicated:
            for w in We[e]:
                keep = keep + w.sum() * 0.0
        return out_units_recv + keep

    return _materialize_and_compute


# ============================ Two-chunk fine-grained overlap ============================
# Split this rank's routing units into ``EPLB_CHUNKS`` token-chunks and pipeline
# dispatch(comm) / expert-GEMM(compute) / combine(comm) across a compute stream and a comm
# side stream, so dispatch(c2) overlaps compute(c1) and combine(c1) overlaps compute(c2)
# (the SCALE-EPLB forward timeline). The backward overlap is obtained for free: PyTorch's
# autograd engine runs each grad_fn on the stream its forward ran on, so combine^-1 / dispatch^-1
# / Wgrad-reduce land on the comm stream and Dgrad/Wgrad on the compute stream -- mirroring the
# backward timeline without a hand-written backward. Replica weights are re-materialised ONCE and
# shared by both chunks (grads from both accumulate, then reduce to main once).
#
# Correctness is order-invariant: the final output is ``sum_units prob * expert(token)`` scattered
# by ``index_add`` on the token index, so any disjoint partition of the units yields the same result.


def _sctx(stream):
    """Context manager that enqueues on ``stream`` (no-op / current stream on CPU)."""
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def _rec(t: torch.Tensor, stream) -> None:
    """Mark ``t`` as used on ``stream`` so the caching allocator won't free/reuse it too early."""
    if stream is not None and t.is_cuda:
        t.record_stream(stream)


def _exchange_recv_counts(send_counts: torch.Tensor, group) -> torch.Tensor:
    """All-to-all the per-dst send counts -> per-src recv counts (uniform 1-int-per-rank split).

    Only used by adapters whose ``needs_recv_counts()`` is True (``AllToAllAdapter``); DeepEP sizes
    the recv buffer statically and skips this. ``send_counts`` is int64 ``[R]`` (== world size).
    """
    if not dist.is_initialized():
        return send_counts
    recv = torch.empty_like(send_counts)
    dist.all_to_all_single(recv, send_counts.contiguous(), group=group)
    return recv


def _materialize_w_stacked(
    *,
    replicated: Sequence[int],
    replicated_main: Sequence[int],
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    weight_shapes: Sequence[torch.Size],
    slot_of_e: torch.Tensor,
    hosted: torch.Tensor,
    n_slot: int,
    dtype: torch.dtype,
    device,
    group,
) -> Tuple[Tuple[torch.Tensor, ...], Optional[torch.Tensor]]:
    """Broadcast-path replica weights, materialised ONCE for the whole layer (shared across chunks).

    Same per-slot assembly as :func:`_make_materialize_and_compute`, but returned as a standalone
    stacked-weight tuple (not a recompute closure) so both chunks' MLPs read the same tensors; grads
    from both chunks accumulate into each replica broadcast, whose backward reduces to ``main(e)`` once.

    Returns ``(w_stacked, keepalive)`` where ``keepalive`` is a scalar tied to every replica broadcast
    so all ranks hit the matching reduce in backward even for replicas that receive no tokens.
    """
    replicated_set = set(replicated)
    We: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for e, main_local in zip(replicated, replicated_main):
        local_w = weights_local.get(e)
        We[e] = tuple(
            broadcast_from_main(
                local_w[j] if local_w is not None else None,
                weight_shapes[j], dtype, device, main_local, group,
            )
            for j in range(len(weight_shapes))
        )
    overflow = torch.full((1,), n_slot, dtype=torch.int64, device=device)
    w_stacked: List[torch.Tensor] = []
    for j in range(len(weight_shapes)):
        buf = torch.zeros((n_slot + 1, *weight_shapes[j]), dtype=dtype, device=device)
        for e, wt in weights_local.items():
            if e in replicated_set:
                continue
            buf.index_copy_(0, slot_of_e[e:e + 1], wt[j].to(dtype).unsqueeze(0))
        for e in replicated:
            tgt = torch.where(hosted[e], slot_of_e[e:e + 1], overflow)
            buf.index_copy_(0, tgt, We[e][j].to(dtype).unsqueeze(0))
        w_stacked.append(buf[:n_slot])
    keep: Optional[torch.Tensor] = None
    for e in replicated:
        for w in We[e]:
            term = w.sum() * 0.0
            keep = term if keep is None else keep + term
    return tuple(w_stacked), keep


def _moe_forward_two_chunks(
    *,
    tokens: torch.Tensor,
    unit_token_idx: torch.Tensor,
    unit_prob: torch.Tensor,
    phys_id: torch.Tensor,
    dst_rank: torch.Tensor,
    w_stacked: Optional[Tuple[torch.Tensor, ...]],
    keepalive: Optional[torch.Tensor],
    experts: "Optional[OverlappedExperts]",
    block: "Optional[ManualMoEBlock]",
    batched_mlp_fn: Callable,
    cap: int,
    group,
    adapter,
    my_rank: int,
    n_slot: int,
    H: int,
    dtype: torch.dtype,
    device,
    R: int,
    num_chunks: int,
) -> torch.Tensor:
    """Token-chunked dispatch/compute/combine pipeline over compute + comm streams (weights pre-acquired).

    ``phys_id`` / ``dst_rank`` are the full-set routing results (STAGE 1 must run on all units so the
    quota-based physical assignment is consistent); we split the units by contiguous halves here.

    Exactly one of three expert paths is passed. ``block`` (:class:`ManualMoEBlock`) folds the whole
    dispatch/GEMM/combine pipeline into one autograd node and hand-schedules both directions -- the
    default whenever the weights move over a transport. ``experts``
    (:class:`OverlappedExperts`) is the same thing left to autograd's node ordering, kept as the
    reference implementation. ``w_stacked`` is the pre-materialised stack consumed by plain autograd.
    """
    U = int(unit_token_idx.shape[0])
    cs = _comm_stream(device)                                            # comm side stream (None on CPU)
    ms = torch.cuda.current_stream(device) if device.type == "cuda" else None
    on_cuda = cs is not None

    # ---- per-chunk static prep (cheap, on the default stream) ----------------------------------
    chunk_units = torch.chunk(torch.arange(U, device=device), num_chunks)
    prep: List[Dict[str, torch.Tensor]] = []
    for idx in chunk_units:
        dst_c = dst_rank.index_select(0, idx)
        perm_c = torch.argsort(dst_c, stable=True)
        idx_p = idx.index_select(0, perm_c)                              # unit ids in send (by-dst) order
        sent_c = torch.bincount(dst_c, minlength=R).to(torch.int64)      # [R] tokens sent to each dst
        utok_c = unit_token_idx.index_select(0, idx_p)                   # [Uc] owning token of each sent unit
        prob_c = unit_prob.index_select(0, idx_p)                        # [Uc] gate weight
        phys_c = phys_id.index_select(0, idx_p)                          # [Uc] target physical id
        send_tokens_c = tokens.index_select(0, utok_c)                   # [Uc, H]
        elem = send_tokens_c.element_size()
        pad_cols = _payload_pad_cols(H, elem)
        m = send_tokens_c.new_zeros((send_tokens_c.shape[0], pad_cols))
        m[:, 0] = phys_c.to(dtype)                                       # phys id carried through the token dtype
        payload_c = torch.cat([send_tokens_c, m], dim=1)
        recv_c = _exchange_recv_counts(sent_c, group) if adapter.needs_recv_counts() else sent_c
        prep.append({
            "sent": sent_c, "recv": recv_c, "payload": payload_c, "route": phys_c,
            "utok": utok_c, "prob": prob_c,
        })

    nc = len(prep)

    if block is not None:
        # One node for dispatch/GEMM/combine, both directions hand-scheduled: the forward's weight pull
        # hides behind dispatch(c1) and the backward's reduce-to-main overlaps the last dispatch^-1.
        comb = block.run(
            [p["payload"] for p in prep], [p["sent"] for p in prep], [p["recv"] for p in prep],
            [p["route"] for p in prep],
            adapter=adapter, group=group, cap=cap, n_slot=n_slot, H=H, pad_cols=pad_cols,
            my_rank=my_rank,
        )
        return block.prefetch_on_backward_of(
            _scatter_to_tokens(comb, prep, tokens.shape[0], H, device, keepalive=None)
        )

    recv: List[Optional[torch.Tensor]] = [None] * nc
    disp_evt: List[Optional[torch.cuda.Event]] = [None] * nc

    # ---- issue every chunk's dispatch on the comm stream ---------------------------------------
    # The payloads and split counts were built on the compute stream just above; the comm stream has
    # to be ordered after those writes before it reads them.
    if on_cuda:
        cs.wait_stream(ms)
    with _sctx(cs):
        for k in range(nc):
            recv[k] = adapter.dispatch_chunk(
                prep[k]["payload"], prep[k]["sent"], prep[k]["recv"], group, tag=k,
                route_idx=prep[k]["route"], n_slot=n_slot, cap=cap,
            )
            if on_cuda:
                _rec(recv[k], ms)
                disp_evt[k] = torch.cuda.Event()
                disp_evt[k].record(cs)

    # ---- interleave: compute(k) on compute stream, combine(k) on comm stream -------------------
    # compute(k) waits only for dispatch(k) (not later dispatches) so dispatch(k+1) overlaps compute(k),
    # and combine(k) (comm) overlaps compute(k+1).
    comb: List[Optional[torch.Tensor]] = [None] * nc
    for k in range(nc):
        if on_cuda:
            ms.wait_event(disp_evt[k])
        rp = recv[k]
        recv_tokens_k = rp[:, :H].contiguous()
        valid_k = None
        if adapter.uses_padded_layout():
            recv_slot_k, valid_k, group_sizes_k = adapter.recv_layout(k)
        else:
            recv_phys_k = rp[:, H].round().to(torch.int64)
            recv_slot_k = recv_phys_k - my_rank * n_slot
            group_sizes_k = torch.bincount(
                recv_slot_k.clamp(min=0, max=n_slot - 1), minlength=n_slot
            ).to(torch.int64)
        if experts is not None:
            y_k = experts.chunk(recv_tokens_k, recv_slot_k, group_sizes_k, cap, valid_k)
        else:
            y_k = grouped_expert_mlp(
                recv_tokens_k, recv_slot_k, group_sizes_k, w_stacked, batched_mlp_fn, cap,
                valid_mask=valid_k,
            )
        if on_cuda:
            comp_evt = torch.cuda.Event()
            comp_evt.record(ms)
            _rec(y_k, cs)
        with _sctx(cs):
            if on_cuda:
                cs.wait_event(comp_evt)
            comb[k] = adapter.combine_chunk(y_k, prep[k]["sent"], prep[k]["recv"], group, tag=k)
            adapter.chunk_state(k)
            if on_cuda:
                _rec(comb[k], ms)

    if on_cuda:
        ms.wait_stream(cs)                                               # gather after all combines land

    result = _scatter_to_tokens(comb, prep, tokens.shape[0], H, device, keepalive=keepalive)
    # Start the backward weight pull here rather than inside a chunk's backward, so it is in flight
    # across the scatter backward and both chunks' reverse combines.
    return experts.prefetch_on_backward_of(result) if experts is not None else result


def _scatter_to_tokens(comb, prep, n_tok: int, H: int, device, *, keepalive) -> torch.Tensor:
    """Gate-weight each chunk's combined output and additively scatter it onto its owning tokens."""
    out_dtype = comb[0].dtype
    result = torch.zeros((n_tok, H), dtype=out_dtype, device=device)
    for k in range(len(comb)):
        result = result.index_add(
            0, prep[k]["utok"], prep[k]["prob"].unsqueeze(1).to(out_dtype) * comb[k]
        )
    if keepalive is not None:
        result = result + keepalive.to(out_dtype)
    return result


# ================================= Public forward =================================


def sync_free_moe_forward(
    tokens: torch.Tensor,
    unit_token_idx: torch.Tensor,
    unit_expert: torch.Tensor,
    unit_prob: torch.Tensor,
    plan: Plan,
    spec: ProblemSpec,
    weights_local: Dict[int, Tuple[torch.Tensor, ...]],
    weight_shapes: Sequence[torch.Size],
    batched_mlp_fn: Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor],
    cap: Optional[int] = None,
    group=None,
    adapter: Optional[CommAdapter] = None,
    rematerialize: bool = False,
    overlap: bool = False,
    gated: bool = False,
    act: Callable = torch.relu,
    transpose_w: bool = False,
) -> torch.Tensor:
    """Replication-aware MoE forward via physical-id routing + grouped compute (the Phase C dispatch path; only the adapter may sync).

    Args:
        tokens: float ``[Ntok, H]`` hidden states for this rank's tokens.
        unit_token_idx: int64 ``[U]`` token index of each routing unit.
        unit_expert: int64 ``[U]`` logical expert id of each routing unit.
        unit_prob: float ``[U]`` gate weight of each routing unit.
        plan: Solved plan (global ``x`` / ``q``).
        spec: Problem spec (``num_experts``, ``main_rank`` as group-local ranks, ``n_slot``).
        weights_local: ``{e: weight_tuple}`` for experts whose ``main(e)`` is this rank.
        weight_shapes: Shape of each weight tensor in an expert's tuple.
        batched_mlp_fn: ``(x[S, cap, H], stacked_weights) -> y[S, cap, H]`` batched expert forward.
        cap: Per-slot capacity. ElasticBuffer requires a host-static value or ``EPLB_CAP``.
        group: EP process group.
        adapter: Transport backend (defaults to :class:`AllToAllAdapter`).
        rematerialize: ``dist.broadcast`` path only: if True, checkpoint the replication + expert GEMM
            so backward re-broadcasts instead of holding the stack. Ignored when the GIN path is
            selected, which never holds it in the first place.
        overlap: If True, use the Level-B custom backward that re-acquires replica weights on a side
            stream overlapped with Wgrad (needs ``gated``/``act``). ``EPLB_WEIGHT_COMM=gin`` always
            takes this path, re-pulling with device-initiated GIN ``get``/``put``; the
            ``dist.broadcast`` transport opts in here and re-broadcasts instead.
        gated: Whether GEMM-1 is gated (used on the overlap/GIN path).
        act: Activation function (used on the overlap/GIN path).
        transpose_w: True if weights are ``[out, in]`` (Megatron) and used as ``x @ W.t()``
            (overlap/GIN path only).

    Returns:
        float ``[Ntok, H]`` combined MoE output for this rank's tokens.
    """
    adapter = adapter or AllToAllAdapter()
    device = tokens.device
    dtype = tokens.dtype
    H = tokens.shape[1]
    n_slot = int(spec.n_slot)
    my_rank = dist.get_rank(group) if dist.is_initialized() else 0

    # Plan-derived tables. Device-only and independent of the dispatch, so they are built up front:
    # that keeps the host reads below off the critical path (a D2H placed after the token all-to-all
    # would block the host from enqueuing the rest of the layer until that collective retires).
    sent_per_dst, recv_per_src, recv_per_expert = _split_sizes(plan, my_rank)
    slot_to_e, slot_of_e, hosted = _slot_tables(plan.x, my_rank, n_slot)
    group_sizes = _group_sizes_by_slot(slot_to_e, recv_per_expert, n_slot, device)

    if cap is None and os.environ.get("EPLB_CAP"):
        cap = int(os.environ["EPLB_CAP"])
    elastic = adapter.uses_padded_layout()
    if cap is None and elastic:
        raise ValueError(
            "ElasticBuffer needs a host-static per-slot cap: pass cap=... or set EPLB_CAP"
        )

    # Control plane for the replica broadcasts. `dist.broadcast` needs a host-side `src`, so the set of
    # replicated experts and their (static) main ranks are read to host in ONE consolidated D2H here --
    # ~E ints, not per-slot/per-token -- carrying `cap` along when it still has to be derived.
    # Everything else on this path stays on device. The GIN weight backend needs none of this: its
    # get_batched/put_batched schedule (weight pull and grad reduce-to-main alike) is device-resident,
    # so on that path the only host read left here is `cap` -- and that is a compute-side quantity,
    # not part of the weight channel.
    #
    # `cap` sizes the dense [n_slot, cap, H] batch `grouped_expert_mlp` pads to, whose activations
    # dominate this path's memory -- far more than the replica weights. `group_sizes` is the exact
    # per-slot token count under this plan, so its max is the tightest correct cap and ~n_slot x below
    # the naive "every received token could land in one slot" bound; a better-balanced plan is
    # therefore also a cheaper one.
    replicated, replicated_main = [], []
    if not gin_enabled():
        rep_e = (plan.num_replicas() > 1).nonzero(as_tuple=False).flatten()   # ascending expert ids
        n_rep = int(rep_e.shape[0])   # host-known: nonzero already resolved its own output size
        parts = [rep_e, spec.main_rank.index_select(0, rep_e).to(torch.int64)]
        if cap is None:
            parts.append(group_sizes.max().reshape(1))
        flat = torch.cat(parts).tolist()
        replicated, replicated_main = flat[:n_rep], flat[n_rep:2 * n_rep]
        if cap is None:
            cap = max(flat[-1], 1)
    elif cap is None:
        cap = max(int(group_sizes.max()), 1)

    # --- STAGE 1: ROUTE each unit -> (physical id, dst rank); order by dst; split sizes ------
    # Physical assignment must see ALL units (it distributes them by the quota plan.q), so STAGE 1
    # always runs on the full set even when the transport is later chunked.
    phys_id, dst_rank = assign_physical(unit_expert, plan, spec, my_rank)

    # EPLB_CHUNKS >= 2: token-side pipeline -- the dispatch/compute/combine of each chunk is interleaved
    # across the compute and comm streams. Orthogonal to the weight channel: the stacks are acquired
    # once per layer and shared by every chunk, whichever backward strategy is in play.
    num_chunks = int(os.environ.get("EPLB_CHUNKS", "1") or "1")
    if num_chunks >= 2:
        w_stacked = keepalive = experts = block = None
        with profiling.record("apply/weight_move", time_it=True, device=device):
            if overlap or gin_enabled():
                # The weight stacks are acquired once per layer and shared by every chunk, in both
                # directions, so chunking costs nothing extra on the weight channel and the
                # reduce-to-main still happens once per layer.
                transport = None
                if gin_enabled():
                    replicator = _get_gin_replicator(group, spec, weight_shapes, dtype, device)
                    transport = GinReplicaTransport(replicator, plan.x)
                kw = dict(
                    weights_local=weights_local, slot_to_e=slot_to_e, main_rank=spec.main_rank,
                    replicated=replicated, weight_shapes=weight_shapes, gated=gated, act=act,
                    transpose_w=transpose_w, my_rank=my_rank, n_slot=n_slot, dtype=dtype,
                    device=device, group=group, transport=transport,
                )
                if _env_truthy("EPLB_MANUAL_BWD", "1"):
                    block = ManualMoEBlock(**kw)
                else:
                    experts = OverlappedExperts(**kw)
            else:
                w_stacked, keepalive = _materialize_w_stacked(
                    replicated=replicated, replicated_main=replicated_main, weights_local=weights_local,
                    weight_shapes=weight_shapes, slot_of_e=slot_of_e, hosted=hosted,
                    n_slot=n_slot, dtype=dtype, device=device, group=group,
                )
        # `cap` (resolved above) bounds each chunk too: a chunk holds a subset of this rank's units, so
        # its per-slot count cannot exceed the full-set `group_sizes` the cap was taken from.
        return _moe_forward_two_chunks(
            tokens=tokens, unit_token_idx=unit_token_idx, unit_prob=unit_prob,
            phys_id=phys_id, dst_rank=dst_rank, w_stacked=w_stacked, keepalive=keepalive,
            experts=experts, block=block, batched_mlp_fn=batched_mlp_fn, cap=cap, group=group,
            adapter=adapter, my_rank=my_rank, n_slot=n_slot, H=H, dtype=dtype, device=device,
            R=int(plan.q.shape[0]), num_chunks=num_chunks,
        )

    perm = torch.argsort(dst_rank, stable=True)

    # --- STAGE 2: DISPATCH tokens (+ their physical ids) to the owning ranks -----------------
    send_tokens = tokens[unit_token_idx][perm]
    send_phys = phys_id[perm]
    elem = send_tokens.element_size()
    pad_cols = _payload_pad_cols(H, elem)
    meta = send_tokens.new_zeros((send_tokens.shape[0], pad_cols))
    meta[:, 0] = send_phys.to(dtype)                             # phys carried (rounded) through the token dtype
    send_payload = torch.cat([send_tokens, meta], dim=1)
    with profiling.record("apply/dispatch", time_it=True, device=device):
        recv_payload = adapter.dispatch_chunk(
            send_payload, sent_per_dst, recv_per_src, group, tag=0,
            route_idx=send_phys, n_slot=n_slot, cap=cap,
        )
    recv_tokens = recv_payload[:, :H].contiguous()

    # --- STAGE 3: GROUP received tokens by local physical slot -------------------------------
    # `slot_to_e` / `slot_of_e` / `hosted` / `group_sizes` are plan-derived and already built above.
    valid_mask = None
    if elastic:
        recv_slot, valid_mask, group_sizes = adapter.recv_layout(0)
    else:
        recv_phys = recv_payload[:, H].round().to(torch.int64)
        recv_slot = recv_phys - my_rank * n_slot

    # --- STAGE 4: COMPUTE (replicate weights + batched expert MLP) ---------------------------
    # This region's per-rank latency is the straggler signal: a synchronous EP step is paced by the
    # max over ranks, so compare max-vs-mean across ranks (EPLB_PROFILE_ALL_RANKS=1).
    experts = None
    with profiling.record("apply/expert_compute", time_it=True, device=device):
        if overlap or gin_enabled():
            # Replica weights are not held across forward->backward: the custom backward re-acquires them
            # from the schedule the transport cached at construction (a few [n_slot] index tensors), so it
            # never re-derives the routing. GIN always lands here and re-pulls with get_batched; the
            # broadcast path opts in with EPLB_OVERLAP=1 and re-broadcasts instead (transport=None).
            transport = None
            if gin_enabled():
                replicator = _get_gin_replicator(group, spec, weight_shapes, dtype, device)
                transport = GinReplicaTransport(replicator, plan.x)
            experts = OverlappedExperts(
                weights_local=weights_local, slot_to_e=slot_to_e, main_rank=spec.main_rank,
                replicated=replicated, weight_shapes=weight_shapes, gated=gated, act=act,
                transpose_w=transpose_w, my_rank=my_rank, n_slot=n_slot, dtype=dtype,
                device=device, group=group, transport=transport,
            )
            out_units_recv = experts.chunk(
                recv_tokens, recv_slot, group_sizes, cap, valid_mask
            )
        else:
            compute = _make_materialize_and_compute(
                replicated=replicated, replicated_main=replicated_main, weights_local=weights_local,
                weight_shapes=weight_shapes, slot_of_e=slot_of_e, hosted=hosted, recv_slot=recv_slot,
                group_sizes=group_sizes, valid_mask=valid_mask,
                batched_mlp_fn=batched_mlp_fn, cap=cap,
                n_slot=n_slot, dtype=dtype, device=device, group=group,
            )
            if rematerialize:  # Level A: recompute the broadcasts in backward instead of holding them
                out_units_recv = checkpoint(compute, recv_tokens, use_reentrant=False, preserve_rng_state=False)
            else:
                out_units_recv = compute(recv_tokens)

    # --- STAGE 5: COMBINE outputs back, invert the permutation, gate-weight, scatter ---------
    with profiling.record("apply/combine", time_it=True, device=device):
        combined_back = adapter.combine_chunk(
            out_units_recv, sent_per_dst, recv_per_src, group, tag=0,
            route_idx=send_phys, n_slot=n_slot, cap=cap,
        )
    adapter.chunk_state(0)
    out_per_unit = combined_back[torch.argsort(perm)]
    result = torch.zeros((tokens.shape[0], H), dtype=out_per_unit.dtype, device=device)
    result = result.index_add(
        0, unit_token_idx, unit_prob.unsqueeze(1).to(result.dtype) * out_per_unit
    )
    # Start the backward weight pull here rather than inside the expert backward, so it is in flight
    # across the scatter backward and the reverse combine above it.
    return experts.prefetch_on_backward_of(result) if experts is not None else result

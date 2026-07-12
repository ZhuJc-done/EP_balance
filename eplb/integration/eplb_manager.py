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

from ..plan import Plan
from ..problem import ProblemSpec
from .comm import all_to_all_single, broadcast_from_main
from .gin_weights import GinWeightReplicator, gin_enabled
from .grouped_mlp import grouped_expert_mlp
from .overlap import _comm_stream, overlapped_grouped_expert_mlp
from .physical import assign_physical


def _env_truthy(name: str) -> bool:
    """Truthy parse of an on/off environment toggle."""
    return os.environ.get(name, "0").strip().lower() not in ("0", "", "false", "no")


# Cache one GIN replicator per (group, layout, dtype) so the symmetric buffers + static main(e)
# layout are allocated once and recycled across layers/steps.
_GIN_REPLICATORS: Dict[tuple, GinWeightReplicator] = {}


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

    def all_to_all(self, inp, out_splits, in_splits, group) -> torch.Tensor:
        # NCCL/Gloo need host-side split lists; this .tolist() is the one allowed D2H here
        return all_to_all_single(inp, out_splits.tolist(), in_splits.tolist(), group)

    # ---- two-chunk pipeline hooks (symmetric: dispatch and combine are both a plain a2a) ----
    def needs_recv_counts(self) -> bool:
        """all_to_all_single needs the per-src recv counts on host, so the caller must supply them."""
        return True

    def dispatch_chunk(self, payload, sent_per_dst, recv_per_src, group, tag: int = 0):
        return all_to_all_single(payload, recv_per_src.tolist(), sent_per_dst.tolist(), group)

    def combine_chunk(self, y, sent_per_dst, recv_per_src, group, tag: int = 0):
        # reverse leg: send back what we received, receive back what we sent
        return all_to_all_single(y, sent_per_dst.tolist(), recv_per_src.tolist(), group)


class _DeepEPDispatch(torch.autograd.Function):
    """Forward all-to-all (scatter) via DeepEP ``dispatch``; backward is the paired ``combine`` (gather)."""

    @staticmethod
    def forward(ctx, inp, buffer, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, num_worst_tokens, holder):
        # num_worst_tokens > 0 statically sizes the recv buffer to a host-known worst case, so DeepEP
        # skips the D2H that would read the actual recv count -> no CPU sync, CUDA-graph capturable.
        recv, _, _, _, handle, _ = buffer.dispatch(
            x=inp.contiguous(),
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            num_worst_tokens=int(num_worst_tokens),
        )
        ctx.buffer = buffer
        ctx.handle = handle
        holder["handle"] = handle          # expose to the adapter so the paired combine reuses this layout
        return recv

    @staticmethod
    def backward(ctx, grad_recv):
        # each token is routed to exactly one rank, so the gather (combine) is the exact transpose
        grad_in, _, _ = ctx.buffer.combine(x=grad_recv.contiguous(), handle=ctx.handle)
        return grad_in, None, None, None, None, None, None


class _DeepEPCombine(torch.autograd.Function):
    """Reverse all-to-all (gather) via DeepEP ``combine`` reusing a dispatch handle; backward is the cached ``dispatch``."""

    @staticmethod
    def forward(ctx, inp, buffer, handle):
        out, _, _ = buffer.combine(x=inp.contiguous(), handle=handle)
        ctx.buffer = buffer
        ctx.handle = handle
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_in, _, _, _, _, _ = ctx.buffer.dispatch(x=grad_out.contiguous(), handle=ctx.handle)
        return grad_in, None, None


class DeepEPAdapter:
    """Sync-free transport over DeepEP ``dispatch``/``combine`` (device-side counts, no D2H on the token channel).

    Drop-in for :class:`AllToAllAdapter` inside :func:`sync_free_moe_forward`, which issues, per forward,
    a dispatch (tokens), a same-direction metadata send (phys ids), and a reverse combine. The heavy
    ``[Ntok, H]`` token channel goes through DeepEP (NVLink, fully on-device); narrow / non-16B-aligned
    metadata (e.g. ``[U, 1]`` phys ids) transparently falls back to ``all_to_all_single`` whose ordering is
    bit-identical to DeepEP dispatch, so the two channels stay consistent.

    Sync-free knob: ``max_recv_tokens`` is a host-static worst-case bound (``= n_slot * cap``, guaranteed
    tight by the EPLB capacity policy). When set, dispatch runs with ``num_worst_tokens=max_recv_tokens``
    so the recv buffer is statically sized and DeepEP never does the D2H that reads the true recv count
    -> zero CPU sync, CUDA-graph capturable. Left unset, it uses the plain dynamic path (one recv-count
    D2H per dispatch, same behaviour as Megatron's flex+DeepEP integration).
    """

    def __init__(
        self,
        num_nvl_bytes: Optional[int] = None,
        num_qps_per_rank: int = 1,
        max_recv_tokens: Optional[int] = None,
    ):
        try:
            import deep_ep  # noqa: F401
        except Exception as e:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "DeepEPAdapter requires the 'deep_ep' package (PyTorch>=2.10, NCCL>=2.30.4, "
                "SM90+ cluster). Use AllToAllAdapter for CPU/single-GPU testing."
            ) from e
        import os

        self._deep_ep = deep_ep
        self._num_nvl_bytes = int(num_nvl_bytes or os.environ.get("EPLB_DEEPEP_NVL_BYTES", 1_000_000_000))
        self._num_qps_per_rank = int(num_qps_per_rank)
        env_max = os.environ.get("EPLB_DEEPEP_MAX_RECV")
        self._max_recv_tokens = int(max_recv_tokens if max_recv_tokens is not None else (env_max or 0))
        self._buffer = None
        self._group = None
        self._pending = None  # holder dict of the in-flight dispatch handle, consumed by the paired combine
        self._handles: Dict[int, Dict[str, object]] = {}  # per-chunk dispatch handles (two-chunk pipeline)

    def set_max_recv_tokens(self, n: int) -> None:
        """Set the host-static worst-case recv bound (``n_slot * cap``) that enables the sync-free path."""
        self._max_recv_tokens = int(n)

    def _get_buffer(self, group):
        if self._buffer is None:
            self._buffer = self._deep_ep.Buffer(
                group, self._num_nvl_bytes, 0,
                low_latency_mode=False, num_qps_per_rank=self._num_qps_per_rank, allow_mnnvl=False,
            )
            self._group = group
        return self._buffer

    @staticmethod
    def _deepep_eligible(inp: torch.Tensor) -> bool:
        # DeepEP kernels move 16B (int4) chunks of bf16/fp16 rows; everything else uses the exact fallback
        return (
            inp.dim() == 2
            and inp.dtype in (torch.bfloat16, torch.float16)
            and (inp.shape[1] * inp.element_size()) % 16 == 0
        )

    def _dispatch(self, inp, in_splits, group) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Run one DeepEP dispatch keyed by device-side per-dst counts; return ``(recv, handle_holder)``."""
        buffer = self._get_buffer(group)
        # forward dispatch: rows arrive pre-grouped by destination rank, so split sizes define the layout.
        # DeepEP models one "expert" per destination rank here; we regroup received tokens by physical
        # slot ourselves afterwards, so num_tokens_per_expert == num_tokens_per_rank is consistent.
        R = int(in_splits.shape[0])
        device = inp.device
        counts = in_splits.to(torch.long)
        # output_size == number of local rows (host-static): lets repeat_interleave skip the
        # sum().item() it would otherwise do to size its output -> no D2H on the token channel.
        dst = torch.repeat_interleave(
            torch.arange(R, device=device), counts, output_size=inp.shape[0]
        )
        is_token_in_rank = torch.zeros(inp.shape[0], R, dtype=torch.bool, device=device)
        is_token_in_rank[torch.arange(inp.shape[0], device=device), dst] = True
        npr = in_splits.to(torch.int32)
        holder: Dict[str, object] = {}
        recv = _DeepEPDispatch.apply(
            inp, buffer, npr, is_token_in_rank, npr, self._max_recv_tokens, holder
        )
        return recv, holder

    def all_to_all(self, inp, out_splits, in_splits, group):
        if not self._deepep_eligible(inp):
            # metadata channel: exact, ordering matches DeepEP dispatch (one small D2H on the splits)
            return all_to_all_single(inp, out_splits.tolist(), in_splits.tolist(), group)

        buffer = self._get_buffer(group)
        if self._pending is not None:
            # second aligned call of this forward == the reverse leg -> combine reusing the dispatch layout
            handle = self._pending["handle"]
            self._pending = None
            return _DeepEPCombine.apply(inp, buffer, handle)

        recv, holder = self._dispatch(inp, in_splits, group)
        self._pending = holder  # consumed by the paired combine (reverse leg) above
        return recv

    # ---- two-chunk pipeline hooks (explicit per-chunk handles so 2 dispatch + 2 combine don't collide) ----
    def needs_recv_counts(self) -> bool:
        """DeepEP sizes recv statically (``num_worst_tokens``); it does not need host-side recv counts."""
        return False

    def dispatch_chunk(self, payload, sent_per_dst, recv_per_src, group, tag: int = 0):
        if not self._deepep_eligible(payload):
            return all_to_all_single(payload, recv_per_src.tolist(), sent_per_dst.tolist(), group)
        recv, holder = self._dispatch(payload, sent_per_dst, group)
        self._handles[tag] = holder  # combine_chunk(tag) reuses this exact dispatch layout
        return recv

    def combine_chunk(self, y, sent_per_dst, recv_per_src, group, tag: int = 0):
        if not self._deepep_eligible(y):
            return all_to_all_single(y, sent_per_dst.tolist(), recv_per_src.tolist(), group)
        buffer = self._get_buffer(group)
        holder = self._handles.pop(tag)
        return _DeepEPCombine.apply(y, buffer, holder["handle"])


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
            recv_tokens, recv_slot, group_sizes, tuple(w_stacked), batched_mlp_fn, cap
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
    w_stacked: Tuple[torch.Tensor, ...],
    keepalive: Optional[torch.Tensor],
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
    """Token-chunked dispatch/compute/combine pipeline over compute + comm streams (weights pre-materialised).

    ``phys_id`` / ``dst_rank`` are the full-set routing results (STAGE 1 must run on all units so the
    quota-based physical assignment is consistent); we split the units by contiguous halves here.
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
        pad_bytes = (16 - (H * elem) % 16) % 16 or 16                    # keep payload 16B-aligned for DeepEP
        pad_cols = pad_bytes // elem
        m = send_tokens_c.new_zeros((send_tokens_c.shape[0], pad_cols))
        m[:, 0] = phys_c.to(dtype)                                       # phys id carried through the token dtype
        payload_c = torch.cat([send_tokens_c, m], dim=1)
        recv_c = _exchange_recv_counts(sent_c, group) if adapter.needs_recv_counts() else sent_c
        prep.append({"sent": sent_c, "recv": recv_c, "payload": payload_c, "utok": utok_c, "prob": prob_c})

    nc = len(prep)
    recv: List[Optional[torch.Tensor]] = [None] * nc
    disp_evt: List[Optional[torch.cuda.Event]] = [None] * nc

    # ---- issue every chunk's dispatch on the comm stream ---------------------------------------
    with _sctx(cs):
        for k in range(nc):
            recv[k] = adapter.dispatch_chunk(prep[k]["payload"], prep[k]["sent"], prep[k]["recv"], group, tag=k)
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
        recv_phys_k = rp[:, H].round().to(torch.int64)
        recv_slot_k = recv_phys_k - my_rank * n_slot                     # local slot in [0, n_slot)
        group_sizes_k = torch.bincount(
            recv_slot_k.clamp(min=0, max=n_slot - 1), minlength=n_slot
        ).to(torch.int64)
        y_k = grouped_expert_mlp(recv_tokens_k, recv_slot_k, group_sizes_k, w_stacked, batched_mlp_fn, cap)
        if on_cuda:
            comp_evt = torch.cuda.Event()
            comp_evt.record(ms)
            _rec(y_k, cs)
        with _sctx(cs):
            if on_cuda:
                cs.wait_event(comp_evt)
            comb[k] = adapter.combine_chunk(y_k, prep[k]["sent"], prep[k]["recv"], group, tag=k)
            if on_cuda:
                _rec(comb[k], ms)

    if on_cuda:
        ms.wait_stream(cs)                                               # gather after all combines land

    # ---- scatter each chunk's gate-weighted output back to its owning tokens (additive) --------
    out_dtype = comb[0].dtype
    result = torch.zeros((tokens.shape[0], H), dtype=out_dtype, device=device)
    for k in range(nc):
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
        cap: Per-slot capacity (host-static). If None, derived as this rank's received-token count
            (safe upper bound for the all-to-all fallback; a DeepEP adapter would pass a static value).
            Required to be host-static (pass ``cap`` or set ``EPLB_CAP``) when ``EPLB_DEEPEP_STATIC=1``.
        group: EP process group.
        adapter: Transport backend (defaults to :class:`AllToAllAdapter`).
        rematerialize: If True, free replica weights after forward and re-broadcast them in
            backward (weight recompute) instead of holding them across fwd->bwd.
        overlap: If True, use the Level-B custom backward that re-broadcasts replica weights on a
            side stream overlapped with Wgrad (implies re-materialisation; needs ``gated``/``act``).
        gated: Whether GEMM-1 is gated (only used when ``overlap``).
        act: Activation function (only used when ``overlap``).
        transpose_w: True if weights are ``[out, in]`` (Megatron) and used as ``x @ W.t()`` (overlap only).

    Returns:
        float ``[Ntok, H]`` combined MoE output for this rank's tokens.
    """
    adapter = adapter or AllToAllAdapter()
    device = tokens.device
    dtype = tokens.dtype
    H = tokens.shape[1]
    n_slot = int(spec.n_slot)
    my_rank = dist.get_rank(group) if dist.is_initialized() else 0

    # Control plane for the replica broadcasts. `dist.broadcast` needs a host-side `src`, so the set of
    # replicated experts and their (static) main ranks are read to host in ONE consolidated D2H here --
    # ~E ints, not per-slot/per-token. Everything else on this path stays on device. The GIN weight
    # backend drives its own device-side schedule instead, so this D2H is skipped there.
    replicated, replicated_main = [], []
    if overlap or not gin_enabled():
        rep_e = (plan.num_replicas() > 1).nonzero(as_tuple=False).flatten()   # ascending expert ids
        rep_pairs = torch.stack([rep_e, spec.main_rank.index_select(0, rep_e)]).tolist()
        replicated, replicated_main = (rep_pairs[0], rep_pairs[1]) if rep_pairs else ([], [])

    # EPLB_DEEPEP_STATIC: statically size the DeepEP recv buffer (num_worst_tokens = n_slot * cap) so the
    # token channel does no recv-count D2H and stays CUDA-graph capturable. Needs a host-static `cap`.
    if _env_truthy("EPLB_DEEPEP_STATIC"):
        if cap is None:
            env_cap = os.environ.get("EPLB_CAP")
            if env_cap is None:
                raise ValueError(
                    "EPLB_DEEPEP_STATIC needs a host-static per-slot cap: pass cap=... or set EPLB_CAP"
                )
            cap = int(env_cap)
        if hasattr(adapter, "set_max_recv_tokens"):
            adapter.set_max_recv_tokens(n_slot * cap)

    # --- STAGE 1: ROUTE each unit -> (physical id, dst rank); order by dst; split sizes ------
    # Physical assignment must see ALL units (it distributes them by the quota plan.q), so STAGE 1
    # always runs on the full set even when the transport is later chunked.
    phys_id, dst_rank = assign_physical(unit_expert, plan, spec, my_rank)

    # EPLB_CHUNKS >= 2: fine-grained two-chunk overlap. Weights are re-materialised once (shared by
    # both chunks) and the dispatch/compute/combine pipeline is interleaved on compute + comm streams.
    # Mutually exclusive with the Level-B `overlap` path (which owns its own custom backward).
    num_chunks = int(os.environ.get("EPLB_CHUNKS", "1") or "1")
    if num_chunks >= 2 and not overlap:
        _, slot_of_e, hosted = _slot_tables(plan.x, my_rank, n_slot)
        if gin_enabled():  # device-initiated replication; its backward reduces to main on its own stream
            replicator = _get_gin_replicator(group, spec, weight_shapes, dtype, device)
            w_stacked = replicator.materialize(plan.x, weights_local)
            keepalive = None
        else:
            w_stacked, keepalive = _materialize_w_stacked(
                replicated=replicated, replicated_main=replicated_main, weights_local=weights_local,
                weight_shapes=weight_shapes, slot_of_e=slot_of_e, hosted=hosted,
                n_slot=n_slot, dtype=dtype, device=device, group=group,
            )
        if cap is None:  # per-slot upper bound: no slot can hold more than this rank's total units
            env_cap = os.environ.get("EPLB_CAP")
            cap = int(env_cap) if env_cap else max(int(unit_token_idx.shape[0]), 1)
        return _moe_forward_two_chunks(
            tokens=tokens, unit_token_idx=unit_token_idx, unit_prob=unit_prob,
            phys_id=phys_id, dst_rank=dst_rank, w_stacked=w_stacked, keepalive=keepalive,
            batched_mlp_fn=batched_mlp_fn, cap=cap, group=group, adapter=adapter,
            my_rank=my_rank, n_slot=n_slot, H=H, dtype=dtype, device=device,
            R=int(plan.q.shape[0]), num_chunks=num_chunks,
        )

    perm = torch.argsort(dst_rank, stable=True)
    sent_per_dst, recv_per_src, recv_per_expert = _split_sizes(plan, my_rank)

    # --- STAGE 2: DISPATCH tokens (+ their physical ids) to the owning ranks -----------------
    send_tokens = tokens[unit_token_idx][perm]
    send_phys = phys_id[perm]
    elem = send_tokens.element_size()
    pad_bytes = (16 - (H * elem) % 16) % 16 or 16                 # >=1 col; makes (H+pad)*elem 16B-aligned
    pad_cols = pad_bytes // elem
    meta = send_tokens.new_zeros((send_tokens.shape[0], pad_cols))
    meta[:, 0] = send_phys.to(dtype)                             # phys carried (rounded) through the token dtype
    send_payload = torch.cat([send_tokens, meta], dim=1)
    recv_payload = adapter.all_to_all(send_payload, recv_per_src, sent_per_dst, group)
    recv_tokens = recv_payload[:, :H].contiguous()
    recv_phys = recv_payload[:, H].round().to(torch.int64)
    if cap is None:  # host-static upper bound: all received tokens could land in one slot
        cap = max(int(recv_tokens.shape[0]), 1)

    # --- STAGE 3: GROUP received tokens by local physical slot -------------------------------
    recv_slot = recv_phys - my_rank * n_slot                          # local slot in [0, n_slot)
    slot_to_e, slot_of_e, hosted = _slot_tables(plan.x, my_rank, n_slot)
    group_sizes = _group_sizes_by_slot(slot_to_e, recv_per_expert, n_slot, device)

    # --- STAGE 4: COMPUTE (replicate weights + batched expert MLP) ---------------------------
    if overlap:  # Level B: async re-materialisation overlapped with Wgrad, grads reduce to main
        out_units_recv = overlapped_grouped_expert_mlp(
            recv_tokens, recv_slot, group_sizes, weights_local, slot_to_e,
            spec.main_rank, replicated, weight_shapes, cap,
            gated=gated, act=act, transpose_w=transpose_w,
            my_rank=my_rank, n_slot=n_slot, group=group,
        )
    elif gin_enabled():  # device-initiated replication over NCCL GIN (no dist.broadcast / host weight bytes)
        replicator = _get_gin_replicator(group, spec, weight_shapes, dtype, device)
        w_stacked = replicator.materialize(plan.x, weights_local)  # (W_j[n_slot, *shape_j], ...)
        out_units_recv = grouped_expert_mlp(
            recv_tokens, recv_slot, group_sizes, w_stacked, batched_mlp_fn, cap
        )
    else:
        compute = _make_materialize_and_compute(
            replicated=replicated, replicated_main=replicated_main, weights_local=weights_local,
            weight_shapes=weight_shapes, slot_of_e=slot_of_e, hosted=hosted, recv_slot=recv_slot,
            group_sizes=group_sizes, batched_mlp_fn=batched_mlp_fn, cap=cap,
            n_slot=n_slot, dtype=dtype, device=device, group=group,
        )
        if rematerialize:  # Level A: recompute the broadcasts in backward instead of holding them
            out_units_recv = checkpoint(compute, recv_tokens, use_reentrant=False, preserve_rng_state=False)
        else:
            out_units_recv = compute(recv_tokens)

    # --- STAGE 5: COMBINE outputs back, invert the permutation, gate-weight, scatter ---------
    combined_back = adapter.all_to_all(out_units_recv, sent_per_dst, recv_per_src, group)
    out_per_unit = combined_back[torch.argsort(perm)]
    result = torch.zeros((tokens.shape[0], H), dtype=out_per_unit.dtype, device=device)
    result = result.index_add(0, unit_token_idx, unit_prob.unsqueeze(1) * out_per_unit)
    return result

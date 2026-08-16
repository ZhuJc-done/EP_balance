"""Device-initiated EPLB weight replication over NCCL symmetric memory.

Sync-free (Level 2): the per-slot replication schedule is built and consumed entirely on device.
``materialize`` derives ``slot_to_e`` / ``main_of_slot`` with the same cumsum/scatter tables as the
token path (no ``nonzero``), turns them into device-resident descriptor arrays
``(remote_off, local_off, nbytes, peers)`` and drives one batched get (forward) / put (backward)
per weight tensor -- the kernel skips ``peers < 0`` on device, so empty and local slots need no host
branch and there is **no D2H** to read the schedule. Local (``main(e) == my_rank``) slots are filled
with a vectorised on-device gather instead of a self-routed transfer.

Transport is per peer, chosen inside the batched kernel. One ``ncclCommWindowRegister`` covers both
paths, so a slot whose ``main(e)`` shares this node is read/written over NVLink with an SM90+ TMA
copy, and only genuinely cross-node slots go through GIN's network RDMA. This matters at the EP
sizes we run:
with the expert-parallel group inside a node, routing every slot through ``gin.put`` would send the
whole weight channel out to the NIC and back at a fraction of NVLink bandwidth. ``EPLB_GIN_LSA=0``
forces the network path for every peer, for A/B measurement only.

Two ordering fences per direction remain (homes-visible-before-get, reads-done-before-recycle, and the
backward puts-landed-before-sum). They default to ``dist.barrier`` (host-launched); set
``EPLB_GIN_FENCE=signal`` for the device-stream ``ncclSignal``/``ncclWaitSignal`` fence that is CUDA-graph
capturable (cluster-validate the signal-counter epochs before relying on it in capture).

Replica weights are never held across forward->backward. :class:`GinReplicaTransport` (plugged into
``overlap.overlapped_grouped_expert_mlp``) is the only consumption path: forward pulls the replica slots
it needs and keeps no clone, and backward re-acquires them with a second ``get_batched`` on a side
stream, overlapping the pull with the weight-free Wgrad before consuming them for Dgrad. Grad reduction
to ``main(e)`` is ``put_batched`` into the owner's scratch column plus a local sum.

The backward re-pull costs no re-derivation of the routing: a transport instance is bound to one
micro-batch's plan at construction and caches the resulting device schedule (a handful of ``[n_slot]``
index tensors), which both directions then reuse. Holding the stack instead would trade that second
pull for ``n_layers x n_slot x |W_e|`` of resident memory, which is the larger cost at scale.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist


def gin_enabled() -> bool:
    """Whether the GIN weight-replication backend is selected (``EPLB_WEIGHT_COMM=gin``)."""
    return os.environ.get("EPLB_WEIGHT_COMM", "").strip().lower() in ("gin", "nccl_gin", "devnccl")


def _nbytes(shape: torch.Size, elem_size: int) -> int:
    return int(math.prod(shape)) * int(elem_size)


def _replica_schedule(
    slot_to_e: torch.Tensor,      # int64 [S], logical expert at each local slot (-1 if empty)
    main_of_slot: torch.Tensor,   # int64 [S], main(e) of that slot's expert (-1 if empty)
    local_slot_of_e: torch.Tensor,  # int64 [E], index of e inside main(e)'s home buffer
    my_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device-resident replication schedule (no host loop, no D2H).

    Returns three device tensors over the ``S`` local slots:
      * ``peers`` int32 ``[S]``: source/destination rank for each slot, or ``-1`` for slots the
        batched GIN kernel must skip (empty slots and slots whose main is this rank).
      * ``ls``    int64 ``[S]``: ``local_slot(e)`` -- the row of ``e`` in ``main(e)``'s home buffer.
      * ``is_local`` bool ``[S]``: slots hosted by this rank's own main copy (filled by a local gather).
    """
    valid = slot_to_e >= 0
    ls = local_slot_of_e[slot_to_e.clamp(min=0)]
    is_remote = valid & (main_of_slot != my_rank)
    is_local = valid & (main_of_slot == my_rank)
    peers = torch.where(is_remote, main_of_slot, torch.full_like(main_of_slot, -1)).to(torch.int32)
    return peers, ls, is_local


class GinWeightReplicator:
    """Owns the NCCL GIN comm + symmetric buffers and the static ``main(e)`` layout.

    One instance is reused across layers/micro-batches for a given (EP group, weight
    shapes, dtype, ``n_slot``, ``E``); buffers are allocated once and recycled.
    """

    def __init__(
        self,
        *,
        group,
        num_experts: int,
        n_slot: int,
        main_rank: torch.Tensor,   # int64 [E], group-local main(e) (static, C7)
        weight_shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        device,
    ) -> None:
        import nccl_gin  # local import: only needed on the cluster

        self._gin = nccl_gin
        self.group = group
        self.E = int(num_experts)
        self.n_slot = int(n_slot)
        self.dtype = dtype
        self.device = device
        self.weight_shapes = [torch.Size(s) for s in weight_shapes]
        self.J = len(self.weight_shapes)
        self.grid = int(os.environ.get("EPLB_GIN_GRID", "8"))

        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        self.my_rank = dist.get_rank(group) if dist.is_initialized() else 0

        elem = torch.empty((), dtype=dtype).element_size()
        self.wb = [_nbytes(s, elem) for s in self.weight_shapes]  # bytes per weight tensor j

        # static per-expert layout: local_slot(e) = index of e within main(e)'s home buffer,
        # in ascending expert-id order. Computed once on host from the immutable main_rank.
        main_host = main_rank.to(torch.int64).tolist()
        self.main_host: List[int] = [int(m) for m in main_host]
        counts = [0] * self.world
        local_slot_of_e: List[int] = [0] * self.E
        for e in range(self.E):
            m = self.main_host[e]
            local_slot_of_e[e] = counts[m]
            counts[m] += 1
        self.home_cap = max(counts) if counts else 0  # max experts any rank is main of

        # device-resident copies of the static tables, so the per-step schedule never touches host
        self.main_dev = torch.as_tensor(self.main_host, dtype=torch.int64, device=device)
        self.local_slot_dev = torch.as_tensor(local_slot_of_e, dtype=torch.int64, device=device)
        self.local_slot_of_e = local_slot_of_e  # host list (used only for the static home-publish loop)
        self.slot_arange = torch.arange(self.n_slot, dtype=torch.int64, device=device)
        # descriptor columns that never depend on the plan: the slot buffer is packed, so a slot's
        # own offset and every transfer's length are the same in both directions and every layer
        self.slot_off = [self.slot_arange * self.wb[j] for j in range(self.J)]
        self.nbytes_const = [
            torch.full((self.n_slot,), self.wb[j], dtype=torch.int64, device=device)
            for j in range(self.J)
        ]

        # fence: default host barrier (tested); 'signal' = device-stream barrier (capturable)
        self._fence_mode = os.environ.get("EPLB_GIN_FENCE", "barrier").strip().lower()
        # resolved once, not per transfer: the batched calls sit in the per-layer path
        self.use_lsa = self._gin._lsa_default()

        # lazily initialise the GIN comm (idempotent guard on the module singleton)
        try:
            self._gin.get_rank()
        except Exception:
            self._gin.init(group)
        self._log_transport()

        # symmetric byte buffers (allocated once, recycled every step)
        self.home = [self._gin.create_tensor(self.home_cap * self.wb[j], torch.uint8)
                     for j in range(self.J)]
        self.slot = [self._gin.create_tensor(self.n_slot * self.wb[j], torch.uint8)
                     for j in range(self.J)]
        # one row past the real columns: the local grad scatter sends the rows it must not write
        # there, which keeps it a fixed-shape index_copy_ (see GinReplicaTransport.grad_row)
        self.scratch = [self._gin.create_tensor((self.home_cap * self.world + 1) * self.wb[j],
                                                torch.uint8)
                        for j in range(self.J)]

    def _log_transport(self) -> None:
        """Report the LSA/network split once, from rank 0.

        ``lsaSize == 1`` means no peer is TMA/LSA reachable and every replica transfer takes the
        network path -- worth seeing, because on a single-node EP group that is the whole weight
        channel going through the NIC and it costs bandwidth rather than correctness.
        """
        if self.my_rank != 0:
            return
        try:
            props = self._gin.get_comm_properties()
            lsa = int(props.get("lsaSize", 1))
        except Exception:
            return
        on = self.use_lsa
        tma = os.environ.get("EPLB_GIN_LSA_TMA", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
        path = (
            ("tma" if tma else "vector (EPLB_GIN_LSA_TMA=0)")
            if on else "off (EPLB_GIN_LSA=0)"
        )
        print(
            f"[eplb-gin] world={self.world} lsa_team={lsa} lsa_path={path}"
            f" -> up to {lsa - 1 if on else 0} of {self.world - 1} peers over NVLink,"
            f" the rest over GIN",
            flush=True,
        )

    # -- schedule ---------------------------------------------------------------
    def slot_tables(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Device tables ``(slot_to_e[S], main_of_slot[S])`` for ``my_rank`` (no D2H, no ``nonzero``).

        ``slot_to_e[s]`` is the logical expert in local slot ``s`` (-1 if empty), placed in ascending
        expert-id order; ``main_of_slot[s] = main(slot_to_e[s])`` (-1 if empty).
        """
        from .eplb_manager import _slot_tables  # reuse the sync-free cumsum/scatter tables

        slot_to_e, _, _ = _slot_tables(x, self.my_rank, self.n_slot)  # device, -1 where empty
        main_of_slot = torch.where(
            slot_to_e >= 0,
            self.main_dev[slot_to_e.clamp(min=0)],
            torch.full((self.n_slot,), -1, dtype=torch.int64, device=x.device),
        )
        return slot_to_e, main_of_slot

    # -- ordering fence ---------------------------------------------------------
    def fence(self, sig_idx: int) -> None:
        """Group-wide ordering fence. ``dist.barrier`` (default) or a device-stream barrier.

        The ``signal`` mode enqueues :func:`nccl_gin.world_fence` on the current CUDA stream, so
        unlike ``dist.barrier`` it is stream-ordered (the overlap in backward depends on that) and
        capture-safe. It reaches intra-node peers over the LSA barrier and the rest over the GIN
        rail barrier, which is why it is not a plain ``ncclSignal`` mesh: signals need a GIN
        connection and there is none to a peer inside the node.
        """
        if not dist.is_initialized() or self.world <= 1:
            return
        if self._fence_mode != "signal":
            dist.barrier(self.group)
            return
        self._gin.world_fence(index=sig_idx)

    # -- byte helpers -----------------------------------------------------------
    def _check_dtype(self, t: torch.Tensor, what: str) -> None:
        """Buffer strides come from ``dtype`` (the token dtype), so anything reinterpreted must match.

        A mismatch is not caught by the byte views below in any useful way: a wider tensor would run
        past its slot into the next one (silent corruption for interior slots), so it is rejected here.
        """
        if t.dtype != self.dtype:
            raise TypeError(
                f"GIN symmetric buffers are laid out for {self.dtype} (taken from the token dtype), "
                f"but {what} is {t.dtype}. Expert parameters and their gradients must share the "
                "activation dtype on this path."
            )

    def _copy_into(self, buf: torch.Tensor, off_bytes: int, src: torch.Tensor, stride: int) -> None:
        self._check_dtype(src, "an expert weight")
        b = src.detach().contiguous().view(torch.uint8).reshape(-1)
        if b.numel() != stride:
            raise ValueError(f"expert weight is {b.numel()} B but the slot stride is {stride} B")
        buf.narrow(0, off_bytes, stride).copy_(b)


class GinReplicaTransport:
    """GIN ``get``/``put`` replica transport for :func:`overlap.overlapped_grouped_expert_mlp`.

    Replica weights are never cloned or held across forward: the overlapped Function re-pulls them in
    its backward on a side stream and consumes them for Dgrad, with the weight-free Wgrad overlapping
    the pull. That spends a second ``get_batched`` (bandwidth) instead of ``n_layers x n_slot x |W_e|``
    of resident memory.

    A transport instance is bound to one micro-batch's plan (``plan_x``) and derives the device schedule
    once in ``__init__``: ``peers``/``ls``/``is_local``/``is_remote``/``col``/``grad_row``, six
    ``[n_slot]`` index tensors (order 1 KiB). Both directions then read those, so the backward re-pull
    neither re-derives
    the routing nor risks disagreeing with the forward get about which slot came from where. Both ops
    fill/consume the effective-layout stacks the overlap skeleton owns; only replica (non-main) slots are
    touched here -- main-owned slots are filled locally by ``overlap._fill_main_slots`` and their grads
    stay local.

    Note (perf, cluster-validate): the get is enqueued on the side stream ``cs`` for overlap, but the
    ordering fence defaults to a host ``dist.barrier`` which is not stream-ordered; set
    ``EPLB_GIN_FENCE=signal`` for a device-stream fence so the pull truly overlaps Wgrad.
    """

    def __init__(self, replicator: GinWeightReplicator, plan_x: torch.Tensor) -> None:
        self.r = replicator
        slot_to_e, main_of_slot = replicator.slot_tables(plan_x)
        peers, ls, is_local = _replica_schedule(
            slot_to_e, main_of_slot, replicator.local_slot_dev, replicator.my_rank
        )
        self.peers, self.ls, self.is_local = peers, ls, is_local
        self.is_remote = peers >= 0
        self._remote_payload_bytes = None
        self.remote_col = self.is_remote.view(-1, *([1] * len(replicator.weight_shapes[0])))
        self.col = ls * replicator.world + replicator.my_rank  # scratch column of each slot's put
        # Rows the local grad scatter must not touch (remote and empty slots) go to a dump row past
        # the real columns, so the scatter is one fixed-shape index_copy_ instead of a boolean mask
        # -- which would cost a D2H per tensor per layer. Their `col` would otherwise collide with a
        # real column: two experts main'd by different ranks can share a home index.
        dump = replicator.home_cap * replicator.world
        self.grad_row = torch.where(is_local, self.col, torch.full_like(self.col, dump))

    def remote_payload_bytes(self):
        """Logical bytes transferred by one get/put over this plan's remote replica slots."""
        if self._remote_payload_bytes is None:
            self._remote_payload_bytes = (
                self.is_remote.to(torch.int64).sum() * sum(self.r.wb)
            )
        return self._remote_payload_bytes

    @staticmethod
    def _sctx(cs):
        return torch.cuda.stream(cs) if cs is not None else contextlib.nullcontext()

    def fill_main_slots(self, meta, main_of, w1_eff, w2_eff, dtype, device) -> None:
        # No host-side fill: GIN gathers local (main==my_rank) slots from the just-published home
        # buffer inside `materialize_replicas`, using the device schedule (is_local/ls). This is what
        # lets the GIN overlap path skip the slot_to_e/main_rank .tolist() D2H entirely (0 D2H).
        return

    def materialize_replicas(self, meta, main_of, w1_eff, w2_eff, dtype, device, cs) -> None:
        r = self.r
        J, wb, nb = r.J, r.wb, r.n_slot
        w_eff = (w1_eff, w2_eff)
        main_expert_ids = meta["main_experts"]
        with self._sctx(cs):
            # publish this rank's resident (main-owned) weights into the symmetric home buffers
            # (static host loop over the few experts this rank owns -- data-independent, no D2H)
            for e in main_expert_ids:
                base = r.local_slot_of_e[e]
                for j in range(J):
                    r._copy_into(r.home[j], base * wb[j], main_of[e][j], wb[j])
            r.fence(0)  # homes globally visible before any get

            for j in range(J):
                # replica slots only: peers<0 (empty / main==me) skipped on device
                r._gin.get_batched(
                    r.home[j], r.slot[j], self.ls * wb[j], r.slot_off[j], r.nbytes_const[j],
                    self.peers, blocks_per_desc=r.grid, use_lsa=r.use_lsa,
                )
                # pulled bytes -> effective-layout stack (transpose for Megatron)
                slot_param = r.slot[j].view(dtype).reshape(nb, *r.weight_shapes[j])
                eff = slot_param.transpose(1, 2) if meta["transpose_w"] else slot_param
                if r.home_cap > 0:
                    # Local (main==me) slots come from the home buffer just published above, gathered
                    # on device. Selecting them with a boolean mask would be a D2H per tensor per
                    # layer -- `t[mask]` has to know how many rows it produced -- so every slot is
                    # gathered and `where` picks the side. `ls` is clamped into range for empty slots
                    # too, and their rows are then real weights that no token reaches, which the
                    # padded GEMM multiplies by an all-zero x_pad.
                    home_param = r.home[j].view(dtype).reshape(r.home_cap, *r.weight_shapes[j])
                    home_eff = home_param.transpose(1, 2) if meta["transpose_w"] else home_param
                    local_src = home_eff.index_select(0, self.ls)
                    torch.where(self.remote_col, eff, local_src, out=w_eff[j])
                else:
                    w_eff[j].copy_(eff)  # this rank is main of nothing: every filled slot is a pull
            r.fence(1)  # peers finished reading my home before it is recycled

    def reduce_grads(self, meta, grad_w1_slot, grad_w2_slot, dtype, device) -> List[torch.Tensor]:
        r = self.r
        J, wb, nb, world = r.J, r.wb, r.n_slot, r.world
        grad_slot = (grad_w1_slot, grad_w2_slot)
        g_bytes = []
        for j in range(J):
            r.scratch[j].zero_()
            r._check_dtype(grad_slot[j], "an expert weight gradient")
            g_bytes.append(grad_slot[j].detach().contiguous().view(torch.uint8).reshape(nb, wb[j]))
            r.slot[j].view(nb, wb[j]).copy_(g_bytes[j])
        # Clearing the scratch is itself a cross-rank ordering point: the columns a peer pushes into
        # are mine to clear, so its put has to wait for my zero. Without this the reduction drops
        # whichever replica arrived early -- masked by the host-barrier fence, which drains the group
        # anyway, and wrong under the device one.
        r.fence(2)

        for j in range(J):
            # replica slots: put grad -> main(e)'s scratch column (peers<0 skipped on device)
            r._gin.put_batched(
                r.slot[j], r.scratch[j], r.slot_off[j], self.col * wb[j], r.nbytes_const[j],
                self.peers, blocks_per_desc=r.grid, use_lsa=r.use_lsa,
            )
            # local slots: on-device copy grad -> my own scratch column; the rest land on the dump row
            scratch_view = r.scratch[j].view(r.home_cap * world + 1, wb[j])
            scratch_view.index_copy_(0, self.grad_row, g_bytes[j])
        r.fence(3)  # all remote grad puts landed before local sum

        grads: List[torch.Tensor] = []
        for e in meta["main_experts"]:
            base = r.local_slot_of_e[e]
            for j in range(J):
                seg = r.scratch[j].narrow(0, base * world * wb[j], world * wb[j])
                stacked = seg.view(dtype).reshape(world, *r.weight_shapes[j])
                grads.append(stacked.sum(dim=0))
        return grads

"""Sync-free Phase C forward: physical-id routing + grouped compute behind a swappable comm adapter (AllToAll fallback / DeepEP)."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from ..plan import Plan
from ..problem import ProblemSpec
from .comm import all_to_all_single, broadcast_from_main
from .grouped_mlp import grouped_expert_mlp
from .overlap import overlapped_grouped_expert_mlp
from .physical import assign_physical, build_phys_slot_table


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


class DeepEPAdapter:
    """Production drop-in backed by DeepEP's ``ElasticBuffer`` (device-side counts, no D2H)."""

    def __init__(self, buffer=None, num_sms: int = 0):
        try:
            import deep_ep  # noqa: F401
        except Exception as e:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "DeepEPAdapter requires the 'deep_ep' package (PyTorch>=2.10, NCCL>=2.30.4, "
                "SM90+ cluster). Use AllToAllAdapter for CPU/single-GPU testing."
            ) from e
        self.buffer = buffer
        self.num_sms = num_sms

    def all_to_all(self, inp, out_splits, in_splits, group):  # pragma: no cover
        # wire ElasticBuffer.dispatch/combine here on a DeepEP-capable cluster (counts stay on device)
        raise NotImplementedError(
            "Wire ElasticBuffer.dispatch/combine here on a DeepEP-capable cluster."
        )


def _slot_to_expert(x: torch.Tensor, my_rank: int, n_slot: int) -> torch.Tensor:
    """int64 ``[n_slot]``: logical expert hosted at each local slot on ``my_rank`` (-1 if empty)."""
    E, R = x.shape
    col = x[:, my_rank].to(torch.bool)  # [E]
    experts = torch.nonzero(col, as_tuple=False).flatten()  # ascending e (fallback: host op)
    out = torch.full((n_slot,), -1, dtype=torch.int64, device=x.device)
    out[: experts.numel()] = experts
    return out


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
    E = int(spec.num_experts)
    R = int(plan.num_ranks)
    n_slot = int(spec.n_slot)
    my_rank = dist.get_rank(group) if dist.is_initialized() else 0

    num_replicas = plan.num_replicas()
    replicated = (num_replicas > 1).nonzero(as_tuple=False).flatten().tolist()

    # --- routing: each unit -> physical instance id + destination rank (sync-free) ----------
    phys_id, dst_rank = assign_physical(unit_expert, plan, spec, my_rank)
    perm = torch.argsort(dst_rank, stable=True)

    # split sizes as device tensors: recv_per_src = output split, sent_per_dst = input split
    sent_per_dst = plan.q[my_rank].sum(dim=0).to(torch.int64)        # sums to U
    recv_per_src = plan.q[:, :, my_rank].sum(dim=1).to(torch.int64)

    send_tokens = tokens[unit_token_idx][perm]
    send_phys = phys_id[perm]
    recv_tokens = adapter.all_to_all(send_tokens, recv_per_src, sent_per_dst, group)
    recv_phys = adapter.all_to_all(
        send_phys.unsqueeze(1).to(dtype), recv_per_src, sent_per_dst, group
    ).squeeze(1).round().to(torch.int64)
    if cap is None:                                                  # host-static upper bound: all received tokens could land in one slot
        cap = max(int(recv_tokens.shape[0]), 1)

    # --- group received tokens by local physical slot, run one batched MLP -----------------
    recv_slot = recv_phys - my_rank * n_slot                         # local slot in [0, n_slot)
    slot_to_e = _slot_to_expert(plan.x, my_rank, n_slot)
    recv_per_expert = plan.q[:, :, my_rank].sum(dim=0).to(torch.int64)  # tokens per expert on this rank
    valid_slot = slot_to_e >= 0
    group_sizes = torch.zeros(n_slot, dtype=torch.int64, device=device)
    group_sizes[valid_slot] = recv_per_expert[slot_to_e[valid_slot]]

    def _materialize_and_compute(recv_tokens: torch.Tensor) -> torch.Tensor:
        # materialise replica weights from main(e) (collective; recomputed in backward when checkpointed)
        We: Dict[int, Tuple[torch.Tensor, ...]] = {}
        for e in replicated:
            main_local = int(spec.main_rank[e].item())
            local_w = weights_local.get(e)
            We[e] = tuple(
                broadcast_from_main(
                    local_w[j] if local_w is not None else None,
                    weight_shapes[j], dtype, device, main_local, group,
                )
                for j in range(len(weight_shapes))
            )
        # stacked per-slot weights (zeros for empty slots)
        w_stacked = []
        for j in range(len(weight_shapes)):
            buf = torch.zeros((n_slot, *weight_shapes[j]), dtype=dtype, device=device)
            for s in range(n_slot):
                e = int(slot_to_e[s].item())
                if e < 0:
                    continue
                w = We[e][j] if e in We else weights_local[e][j]
                buf[s] = w
            w_stacked.append(buf)
        out_units_recv = grouped_expert_mlp(
            recv_tokens, recv_slot, group_sizes, tuple(w_stacked), batched_mlp_fn, cap
        )
        # keepalive: make output depend on every broadcast so all ranks hit the matching reduce in backward
        keep = recv_tokens.sum() * 0.0
        for e in replicated:
            for w in We[e]:
                keep = keep + w.sum() * 0.0
        return out_units_recv + keep

    if overlap:
        out_units_recv = overlapped_grouped_expert_mlp(
            recv_tokens, recv_slot, group_sizes, weights_local, slot_to_e,
            spec.main_rank, replicated, weight_shapes, cap,
            gated=gated, act=act, transpose_w=transpose_w,
            my_rank=my_rank, n_slot=n_slot, group=group,
        )
    elif rematerialize:
        out_units_recv = checkpoint(_materialize_and_compute, recv_tokens, use_reentrant=False, preserve_rng_state=False)
    else:
        out_units_recv = _materialize_and_compute(recv_tokens)

    # --- combine: send outputs back, invert the dst-grouping permutation -------------------
    combined_back = adapter.all_to_all(out_units_recv, sent_per_dst, recv_per_src, group)
    out_per_unit = combined_back[torch.argsort(perm)]

    result = torch.zeros((tokens.shape[0], H), dtype=out_per_unit.dtype, device=device)
    result = result.index_add(0, unit_token_idx, unit_prob.unsqueeze(1) * out_per_unit)
    return result

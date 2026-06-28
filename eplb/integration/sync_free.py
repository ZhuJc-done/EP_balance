"""Sync-free Phase C forward: physical-id routing + grouped compute behind a swappable comm adapter (AllToAll fallback / DeepEP)."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist

from ..plan import Plan
from ..problem import ProblemSpec
from .comm import all_to_all_single, broadcast_from_main
from .grouped_mlp import grouped_expert_mlp
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
    cap: int,
    group=None,
    adapter: Optional[CommAdapter] = None,
) -> torch.Tensor:
    """Replication-aware MoE forward via physical-id routing + grouped compute (matches ``replicated_moe_forward``; only the adapter may sync).

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
        cap: Per-slot capacity (host-static; must cover the busiest physical instance).
        group: EP process group.
        adapter: Transport backend (defaults to :class:`AllToAllAdapter`).

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

    # --- materialise replica weights from main(e) (collective; every rank participates) ----
    num_replicas = plan.num_replicas()
    replicated = (num_replicas > 1).nonzero(as_tuple=False).flatten().tolist()
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

    # --- group received tokens by local physical slot, run one batched MLP -----------------
    recv_slot = recv_phys - my_rank * n_slot                         # local slot in [0, n_slot)
    slot_to_e = _slot_to_expert(plan.x, my_rank, n_slot)
    recv_per_expert = plan.q[:, :, my_rank].sum(dim=0).to(torch.int64)  # tokens per expert on this rank
    valid_slot = slot_to_e >= 0
    group_sizes = torch.zeros(n_slot, dtype=torch.int64, device=device)
    group_sizes[valid_slot] = recv_per_expert[slot_to_e[valid_slot]]

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
    w_stacked = tuple(w_stacked)

    out_units_recv = grouped_expert_mlp(
        recv_tokens, recv_slot, group_sizes, w_stacked, batched_mlp_fn, cap
    )

    # --- combine: send outputs back, invert the dst-grouping permutation -------------------
    combined_back = adapter.all_to_all(out_units_recv, sent_per_dst, recv_per_src, group)
    out_per_unit = combined_back[torch.argsort(perm)]

    result = torch.zeros((tokens.shape[0], H), dtype=out_per_unit.dtype, device=device)
    result = result.index_add(0, unit_token_idx, unit_prob.unsqueeze(1) * out_per_unit)

    # keepalive: keep every rank on the same collective backward branches (avoid deadlock)
    keep = recv_tokens.sum() * 0.0
    for e in replicated:
        for w in We[e]:
            keep = keep + w.sum() * 0.0
    return result + keep

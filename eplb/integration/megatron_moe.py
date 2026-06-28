"""Phase C binding (import-safe, SequentialMLP only): drive Megatron-Core's MoELayer through the sync-free EPLB dispatcher."""

from __future__ import annotations

import os
import types
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from . import profiling
from ..problem import ProblemSpec
from .grouped_mlp import make_batched_gated_mlp
from .sync_free import AllToAllAdapter, DeepEPAdapter, sync_free_moe_forward


def _env_flag(name: str) -> bool:
    """Truthy parse of an on/off environment toggle."""
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no")


def _make_adapter():
    """Select the sync-free transport backend from ``EPLB_ADAPTER`` (``alltoall`` default | ``deepep``)."""
    name = os.environ.get("EPLB_ADAPTER", "alltoall").strip().lower()
    if name in ("deepep", "deep_ep"):
        return DeepEPAdapter()
    if name in ("", "alltoall", "all_to_all", "a2a"):
        return AllToAllAdapter()
    raise ValueError(f"unknown EPLB_ADAPTER={name!r} (expected 'alltoall' | 'deepep')")


def find_moe_layers(model, class_name: str = "MoELayer") -> List:
    """Collect every Megatron ``MoELayer`` instance in a model (by class name)."""
    return [m for _, m in model.named_modules() if type(m).__name__ == class_name]


def extract_local_expert_weights(
    experts_module,
) -> Tuple[List[Tuple[torch.Tensor, ...]], List[torch.Size]]:
    """Return ``[(fc1_weight, fc2_weight), ...]`` for each local expert, plus the shared shapes.

    Args:
        experts_module: A Megatron ``SequentialMLP`` (``.local_experts`` ModuleList).

    Returns:
        ``(weight_tuples, weight_shapes)`` ordered by local expert index.
    """
    local = getattr(experts_module, "local_experts", None)
    if local is None:
        raise NotImplementedError(
            "EPLB Phase C binding currently supports SequentialMLP only "
            "(experts.local_experts). For GroupedMLP, run without --moe-grouped-gemm, "
            "or extend extract_local_expert_weights to reshape weight1/weight2."
        )
    tuples: List[Tuple[torch.Tensor, ...]] = []
    for mlp in local:
        tuples.append((mlp.linear_fc1.weight, mlp.linear_fc2.weight))
    shapes = [t.shape for t in tuples[0]]
    return tuples, shapes


def _routing_to_units(
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    """Flatten a router's per-token top-k selection into flat routing units.

    Args:
        probs: float ``[N, E]`` gate weights (zero where not selected).
        routing_map: bool/int ``[N, E]`` selection mask.
        num_tokens: ``N`` (for validation).
        num_experts: ``E``.

    Returns:
        ``(unit_token_idx [U], unit_expert [U], unit_prob [U])`` in row-major (token, expert) order.
    """
    rmap = routing_map.bool().reshape(num_tokens, num_experts)
    nz = torch.nonzero(rmap, as_tuple=False)
    unit_token_idx = nz[:, 0].contiguous().to(torch.int64)
    unit_expert = nz[:, 1].contiguous().to(torch.int64)
    unit_prob = probs.reshape(num_tokens, num_experts)[unit_token_idx, unit_expert].contiguous()
    return unit_token_idx, unit_expert, unit_prob


def eplb_moe_forward(self, hidden_states, *args, **kwargs):
    """Drop-in ``MoELayer.forward`` using the sync-free EPLB dispatcher (bound via :func:`bind_eplb_to_moe_layer`)."""
    cfg = self._eplb
    reb = cfg["reb"]
    group = cfg["group"]
    spec: ProblemSpec = reb.spec

    in_shape = hidden_states.shape
    tokens = hidden_states.reshape(-1, in_shape[-1])

    with profiling.record("apply/route", time_it=True, device=tokens.device):
        probs, routing_map = self.router(hidden_states)
        unit_token_idx, unit_expert, unit_prob = _routing_to_units(
            probs, routing_map, tokens.shape[0], spec.num_experts
        )
        local_row = torch.bincount(unit_expert, minlength=spec.num_experts).to(torch.int64)

    mb = cfg["mb"]
    cfg["mb"] = mb + 1
    plan = reb.rebalance(local_row, cfg["layer_id"], mb, group=group).plan

    ep_rank = dist.get_rank(group) if dist.is_initialized() else 0
    weight_tuples, weight_shapes = extract_local_expert_weights(self.experts)
    num_local = len(weight_tuples)
    weights_local: Dict[int, Tuple[torch.Tensor, ...]] = {
        ep_rank * num_local + i: weight_tuples[i] for i in range(num_local)
    }

    out = sync_free_moe_forward(
        tokens=tokens,
        unit_token_idx=unit_token_idx,
        unit_expert=unit_expert,
        unit_prob=unit_prob.to(tokens.dtype),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=weight_shapes, batched_mlp_fn=cfg["batched_mlp_fn"], cap=None,
        group=group, adapter=cfg["adapter"],
        rematerialize=cfg["rematerialize"], overlap=cfg["overlap"],
        gated=cfg["gated"], act=cfg["act"], transpose_w=True,
    )
    if profiling.enabled():
        profiling.maybe_summary(print if ep_rank == 0 else None)
    return out.reshape(in_shape), None


def bind_eplb_to_moe_layer(moe_layer, rebalancer, ep_group, layer_id: int = 0) -> None:
    """Patch a Megatron ``MoELayer`` to dispatch through Scale-EPLB (Phase C apply; call once per layer, spec comes from ``rebalancer.spec``).

    Args:
        moe_layer: A Megatron ``MoELayer`` instance.
        rebalancer: An :class:`~eplb.integration.rebalancer.EPLBRebalancer` for this layer.
        ep_group: The expert-model-parallel process group.
        layer_id: Stable id used as the rebalancer's ring-buffer key.
    """
    config = moe_layer.config
    gated = bool(getattr(config, "gated_linear_unit", False))
    act = getattr(config, "activation_func", torch.nn.functional.gelu)
    moe_layer._eplb = {
        "reb": rebalancer,
        "group": ep_group,
        "layer_id": int(layer_id),
        "mb": 0,
        "gated": gated,
        "act": act,
        "batched_mlp_fn": make_batched_gated_mlp(gated, act),
        "adapter": _make_adapter(),
        "rematerialize": _env_flag("EPLB_REMATERIALIZE"),
        "overlap": _env_flag("EPLB_OVERLAP"),
    }
    moe_layer.forward = types.MethodType(eplb_moe_forward, moe_layer)

"""Phase C binding (import-safe, SequentialMLP only): drive Megatron-Core's MoELayer through the sync-free EPLB dispatcher."""

from __future__ import annotations

import os
import types
import warnings
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from . import profiling
from ..problem import ProblemSpec
from .grouped_mlp import make_batched_gated_mlp
from .eplb_manager import AllToAllAdapter, DeepEPAdapter, sync_free_moe_forward


def _env_flag(name: str) -> bool:
    """Truthy parse of an on/off environment toggle."""
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no")


def _make_adapter():
    """Select ``alltoall`` or the strict ElasticBuffer-backed ``deepep`` transport."""
    name = os.environ.get("EPLB_ADAPTER", "alltoall").strip().lower()
    if name in ("deepep", "deep_ep"):
        legacy = [
            key for key in (
                "EPLB_DEEPEP_STATIC", "EPLB_DEEPEP_ALLOW_MNNVL",
                "EPLB_DEEPEP_NVL_BYTES", "EPLB_DEEPEP_RDMA_BYTES", "EPLB_DEEPEP_MAX_RECV",
            )
            if key in os.environ
        ]
        if legacy:
            raise ValueError(
                "legacy DeepEP Buffer settings are not supported by ElasticBuffer: "
                + ", ".join(legacy)
            )
        cap = int(os.environ.get("EPLB_CAP", "0") or "0")
        if cap <= 0:
            raise ValueError("EPLB_ADAPTER=deepep requires a positive host-static EPLB_CAP")
        if os.environ.get("EPLB_WEIGHT_COMM", "").strip().lower() != "gin":
            raise ValueError("zero-sync ElasticBuffer mode requires EPLB_WEIGHT_COMM=gin")
        if os.environ.get("EPLB_GIN_FENCE", "").strip().lower() != "signal":
            raise ValueError("zero-sync ElasticBuffer mode requires EPLB_GIN_FENCE=signal")
        if _env_flag("EPLB_PROFILE") or _env_flag("PROFILE_TRACE"):
            raise ValueError("zero-sync ElasticBuffer mode requires EPLB_PROFILE=0 and PROFILE_TRACE=0")
        if _env_flag("EPLB_DEBUG_TIMING"):
            warnings.warn(
                "EPLB_DEBUG_TIMING synchronizes once per MoE invocation; use its phase timings "
                "for diagnosis only, not its end-to-end throughput",
                RuntimeWarning,
                stacklevel=2,
            )
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
    topk: int | None = None,
):
    """Flatten a router's per-token top-k selection into flat routing units.

    Args:
        probs: float ``[N, E]`` gate weights (zero where not selected).
        routing_map: bool/int ``[N, E]`` selection mask.
        num_tokens: ``N`` (for validation).
        num_experts: ``E``.
        topk: If given (the router's fixed top-k), take the sync-free path where ``U = N*topk``
            is host-static -- no ``torch.nonzero`` (whose output size would force a D2H). Leave
            ``None`` for the generic reference path (variable selections per row).

    Returns:
        ``(unit_token_idx [U], unit_expert [U], unit_prob [U])``. Unit order is irrelevant to
        the dispatcher (outputs are scattered back by an ``index_add`` over ``unit_token_idx``).
    """
    rmap = routing_map.bool().reshape(num_tokens, num_experts)
    if topk is None:
        # generic path: variable #selections per row; nonzero syncs to discover the count
        nz = torch.nonzero(rmap, as_tuple=False)
        unit_token_idx = nz[:, 0].contiguous().to(torch.int64)
        unit_expert = nz[:, 1].contiguous().to(torch.int64)
        unit_prob = probs.reshape(num_tokens, num_experts)[unit_token_idx, unit_expert].contiguous()
        return unit_token_idx, unit_expert, unit_prob

    # sync-free fixed-topk path: stable argsort puts the selected columns (key 0) first, in
    # ascending expert-id order; the first ``topk`` of them are this token's experts.
    k = int(topk)
    order = torch.argsort((~rmap).to(torch.int8), dim=1, stable=True)          # [N, E]
    sel = order[:, :k].contiguous().to(torch.int64)                           # [N, k] expert ids
    unit_expert = sel.reshape(-1).contiguous()
    unit_token_idx = (
        torch.arange(num_tokens, device=rmap.device, dtype=torch.int64)
        .view(num_tokens, 1).expand(num_tokens, k).reshape(-1).contiguous()
    )
    unit_prob = probs.reshape(num_tokens, num_experts).gather(1, sel).reshape(-1).contiguous()
    return unit_token_idx, unit_expert, unit_prob


def eplb_moe_forward(self, hidden_states, *args, **kwargs):
    """Drop-in ``MoELayer.forward`` using the sync-free EPLB dispatcher (bound via :func:`bind_eplb_to_moe_layer`)."""
    cfg = self._eplb
    reb = cfg["reb"]
    group = cfg["group"]
    spec: ProblemSpec = reb.spec
    profiling.begin_debug_window()

    in_shape = hidden_states.shape
    tokens = hidden_states.reshape(-1, in_shape[-1])
    if isinstance(cfg["adapter"], DeepEPAdapter) and tokens.dtype != torch.bfloat16:
        raise TypeError(f"ElasticBuffer apply mode requires BF16 hidden states, got {tokens.dtype}")

    shared_expert_output = None
    if getattr(self, "use_shared_expert", False):
        with profiling.record("apply/shared_expert", time_it=True, device=tokens.device):
            shared_expert_output = self.shared_experts_compute(hidden_states)

    with profiling.record("apply/route", time_it=True, device=tokens.device):
        probs, routing_map = self.router(hidden_states)
        unit_token_idx, unit_expert, unit_prob = _routing_to_units(
            probs, routing_map, tokens.shape[0], spec.num_experts,
            topk=getattr(self.config, "moe_router_topk", None),
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
        weight_shapes=weight_shapes, batched_mlp_fn=cfg["batched_mlp_fn"],
        cap=int(os.environ["EPLB_CAP"]) if isinstance(cfg["adapter"], DeepEPAdapter) else None,
        group=group, adapter=cfg["adapter"],
        rematerialize=cfg["rematerialize"], overlap=cfg["overlap"],
        gated=cfg["gated"], act=cfg["act"], transpose_w=True,
    )
    if profiling.enabled():
        profiling.maybe_summary(
            print if (ep_rank == 0 or profiling.all_ranks()) else None,
            context=f"mode=apply layer={cfg['layer_id']} mb={mb}",
        )
    out = out.reshape(in_shape)
    if shared_expert_output is not None:
        out = out + shared_expert_output
    return out, None


def bind_eplb_to_moe_layer(moe_layer, rebalancer, ep_group, layer_id: int = 0) -> None:
    """Patch a Megatron ``MoELayer`` to dispatch through Scale-EPLB (Phase C apply; call once per layer, spec comes from ``rebalancer.spec``).

    Args:
        moe_layer: A Megatron ``MoELayer`` instance.
        rebalancer: An :class:`~eplb.integration.rebalancer.EPLBRebalancer` for this layer.
        ep_group: The expert-model-parallel process group.
        layer_id: Stable id used as the rebalancer's ring-buffer key.
    """
    config = moe_layer.config
    if getattr(moe_layer, "use_shared_expert", False):
        if getattr(moe_layer, "shared_expert_overlap", False):
            raise NotImplementedError(
                "EPLB apply mode supports shared experts only with "
                "moe_shared_expert_overlap=False; the EPLB dispatcher replaces "
                "Megatron's token dispatcher and cannot drive its overlap state machine."
            )
        if not callable(getattr(moe_layer, "shared_experts_compute", None)):
            raise NotImplementedError(
                "this Megatron MoELayer exposes shared experts but no "
                "shared_experts_compute method"
            )
    adapter = _make_adapter()
    if isinstance(adapter, DeepEPAdapter):
        topk = getattr(config, "moe_router_topk", None)
        if topk is None or int(topk) <= 0:
            raise ValueError("ElasticBuffer apply mode requires a fixed positive moe_router_topk")
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
        "adapter": adapter,
        "rematerialize": _env_flag("EPLB_REMATERIALIZE"),
        "overlap": _env_flag("EPLB_OVERLAP"),
    }
    moe_layer.forward = types.MethodType(eplb_moe_forward, moe_layer)

"""Phase C binding (import-safe, SequentialMLP only): drive Megatron-Core's MoELayer through the EPLB replication dispatcher."""

from __future__ import annotations

import types
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist

from ..problem import ProblemSpec
from .dispatcher import replicated_moe_forward


def find_moe_layers(model, class_name: str = "MoELayer") -> List:
    """Collect every Megatron ``MoELayer`` instance in a model (by class name)."""
    return [m for _, m in model.named_modules() if type(m).__name__ == class_name]


def build_expert_mlp_fn(config) -> Callable[[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
    """Build an expert forward matching Megatron's MLP from its ``TransformerConfig``.

    Weights are Megatron ``Linear`` tensors ``[out, in]`` (compute is ``x @ W.t()``).
    Handles gated (SwiGLU-style) and plain activations.

    Args:
        config: Megatron ``TransformerConfig`` (uses ``gated_linear_unit`` + ``activation_func``).

    Returns:
        ``mlp_fn(x, (fc1_weight, fc2_weight)) -> y``.
    """
    gated = bool(getattr(config, "gated_linear_unit", False))
    act = getattr(config, "activation_func", torch.nn.functional.gelu)

    def mlp_fn(x: torch.Tensor, w: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        h = x @ w[0].t()
        if gated:
            gate, up = torch.chunk(h, 2, dim=-1)
            h = act(gate) * up
        else:
            h = act(h)
        return h @ w[1].t()

    return mlp_fn


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
    """Drop-in ``MoELayer.forward`` using the EPLB replication dispatcher (bound via :func:`bind_eplb_to_moe_layer`)."""
    cfg = self._eplb
    reb = cfg["reb"]
    group = cfg["group"]
    spec: ProblemSpec = reb.spec

    in_shape = hidden_states.shape
    tokens = hidden_states.reshape(-1, in_shape[-1])

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

    out = replicated_moe_forward(
        tokens=tokens,
        unit_token_idx=unit_token_idx,
        unit_expert=unit_expert,
        unit_prob=unit_prob.to(tokens.dtype),
        plan=plan, spec=spec, weights_local=weights_local,
        weight_shapes=weight_shapes, mlp_fn=cfg["mlp_fn"], group=group,
    )
    return out.reshape(in_shape), None


def bind_eplb_to_moe_layer(moe_layer, rebalancer, ep_group, layer_id: int = 0) -> None:
    """Patch a Megatron ``MoELayer`` instance to dispatch through Scale-EPLB (Phase C, apply).

    Call once per MoE layer after the model is built. The layer's ``ProblemSpec`` comes from
    ``rebalancer.spec`` (build it with :func:`build_spec_for_megatron`).

    Args:
        moe_layer: A Megatron ``MoELayer`` instance.
        rebalancer: An :class:`~eplb.integration.rebalancer.EPLBRebalancer` for this layer.
        ep_group: The expert-model-parallel process group.
        layer_id: Stable id used as the rebalancer's ring-buffer key.
    """
    moe_layer._eplb = {
        "reb": rebalancer,
        "group": ep_group,
        "layer_id": int(layer_id),
        "mb": 0,
        "mlp_fn": build_expert_mlp_fn(moe_layer.config),
    }
    moe_layer.forward = types.MethodType(eplb_moe_forward, moe_layer)

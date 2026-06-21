"""Integration glue for plugging Scale-EPLB into a real EP training framework (Megatron-LM + DeepEP)."""

from .rebalancer import EPLBRebalancer
from .hooks import (
    WeightMaterializer,
    NullWeightMaterializer,
    Dispatcher,
    RebalanceResult,
)
from .dispatcher import replicated_moe_forward, assign_unit_dst
from .comm import all_to_all_single, broadcast_from_main
from .megatron_moe import (
    bind_eplb_to_moe_layer,
    build_expert_mlp_fn,
    extract_local_expert_weights,
    find_moe_layers,
)

__all__ = [
    "EPLBRebalancer",
    "WeightMaterializer",
    "NullWeightMaterializer",
    "Dispatcher",
    "RebalanceResult",
    "replicated_moe_forward",
    "assign_unit_dst",
    "all_to_all_single",
    "broadcast_from_main",
    "bind_eplb_to_moe_layer",
    "build_expert_mlp_fn",
    "extract_local_expert_weights",
    "find_moe_layers",
]

"""Integration glue for plugging Scale-EPLB into a real EP training framework (Megatron-LM + DeepEP)."""

from .rebalancer import EPLBRebalancer, PlanSolver
from .hooks import (
    WeightMaterializer,
    NullWeightMaterializer,
    Dispatcher,
    RebalanceResult,
)
from .comm import all_to_all_single, broadcast_from_main
from .physical import assign_physical, build_phys_slot_table
from .grouped_mlp import grouped_expert_mlp, make_batched_gated_mlp
from .overlap import WeightPool, overlapped_grouped_expert_mlp
from .eplb_manager import (
    CommAdapter,
    AllToAllAdapter,
    DeepEPAdapter,
    sync_free_moe_forward,
)
from .megatron_moe import (
    bind_eplb_to_moe_layer,
    extract_local_expert_weights,
    find_moe_layers,
)
from .megatron_timing import NativeMoETimingBinding, bind_native_moe_timing

__all__ = [
    "EPLBRebalancer",
    "PlanSolver",
    "WeightMaterializer",
    "NullWeightMaterializer",
    "Dispatcher",
    "RebalanceResult",
    "all_to_all_single",
    "broadcast_from_main",
    "assign_physical",
    "build_phys_slot_table",
    "grouped_expert_mlp",
    "make_batched_gated_mlp",
    "WeightPool",
    "overlapped_grouped_expert_mlp",
    "CommAdapter",
    "AllToAllAdapter",
    "DeepEPAdapter",
    "sync_free_moe_forward",
    "bind_eplb_to_moe_layer",
    "bind_native_moe_timing",
    "extract_local_expert_weights",
    "find_moe_layers",
    "NativeMoETimingBinding",
]

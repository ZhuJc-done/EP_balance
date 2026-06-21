"""How to wire Scale-EPLB into Megatron-Core's MoE layer, plus a runnable CPU demo.

Run the demo (no GPU/Megatron needed):
    python -m examples.megatron_eplb_hook

================================================================================
PHASE B -- observe, ZERO Megatron-Core source edits (recommended)
================================================================================
After Megatron builds the model (e.g. in pretrain_gpt.py's model_provider, or
right after get_model()), attach forward-hook observers in one call:

    from eplb.integration.megatron import setup_eplb_observer, assert_plan_replicated

    hook, handles = setup_eplb_observer(
        model,                       # the GPTModel (nn.Module)
        num_experts=args.num_experts,
        weight_bytes_each=expert_param_bytes,   # ~ #params_per_expert * dtype_size
        s_tok=args.hidden_size * 2,             # bf16 activation bytes / token
        n_slot=2 * (args.num_experts // ep_size),
        gpus_per_node=8,
        logger=print,               # prints "[EPLB] layer=.. tau=.. imbalance=.." each fwd
    )
    # ... run a few training iters; logs show real-routing imbalance vs EPLB makespan ...
    assert assert_plan_replicated(hook.last_plan, hook.ep_group)   # E3 on real routing

This needs NO edits to Megatron because register_forward_hook is a built-in
PyTorch mechanism. It measures E2 (solver latency) + E3 (determinism) on real
routing without changing dispatch.

PHASE C -- apply, bind the MoELayer to the replication dispatcher
================================================================================
Patch each Megatron MoELayer to dispatch through Scale-EPLB (splits tokens across
replicas per plan.q, materialises replica weights from main(e), aggregates grads):

    from eplb import EPLBConfig, Topology
    from eplb.integration import EPLBRebalancer, bind_eplb_to_moe_layer
    from eplb.integration.megatron import build_spec_for_megatron
    from megatron.core import parallel_state as mpu

    ep_group = mpu.get_expert_model_parallel_group()
    ep_size = mpu.get_expert_model_parallel_world_size()
    for layer_id, moe in enumerate(find_moe_layers(model)):   # each MoELayer instance
        topo = Topology.from_nvlink_rdma(ep_size // 4, 4, 1, 8, device='cuda')
        spec = build_spec_for_megatron(num_experts, ep_size, w_bytes, s_tok, n_slot, 'cuda')
        reb = EPLBRebalancer(topo, spec, EPLBConfig())
        bind_eplb_to_moe_layer(moe, reb, ep_group, layer_id)

Supports SequentialMLP experts (run without --moe-grouped-gemm). The correctness of
the dispatch/combine + grad aggregation is verified on CPU in tests/test_phase_c.py.
For peak performance, a DeepEP-fused fast path replaces this reference dispatcher later.

For the standalone real-cluster determinism / latency runs (no Megatron):
    torchrun --nproc_per_node=8 -m sim.run_dist --backend nccl --experts 64
================================================================================
"""

from __future__ import annotations

import torch

from eplb import EPLBConfig, Topology, compute_metrics
from eplb.integration import EPLBRebalancer
from eplb.integration.megatron import (
    MegatronEPLBHook,
    build_spec_for_megatron,
    lambda_row_from_routing_map,
)
from eplb.loads import Loads


def _fake_router_routing_map(num_tokens: int, num_experts: int, top_k: int,
                             hot: int, seed: int) -> torch.Tensor:
    """Emulate Megatron's routing_map [num_tokens, num_experts] (bool), skewed to `hot`."""
    g = torch.Generator().manual_seed(seed)
    probs = torch.ones(num_experts, dtype=torch.float64)
    probs[:hot] *= 6.0  # a few hot experts
    probs /= probs.sum()
    rmap = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    for t in range(num_tokens):
        picks = torch.multinomial(probs, top_k, replacement=False, generator=g)
        rmap[t, picks] = True
    return rmap


def main() -> None:
    ep_size = 8           # one EP rank per GPU in a node
    num_experts = 64
    top_k = 6

    topo = Topology.from_nvlink_rdma(num_nodes=1, gpus_per_node=ep_size, intra_cost=1, inter_cost=8)
    spec = build_spec_for_megatron(
        num_experts=num_experts, ep_size=ep_size,
        weight_bytes_each=44_000_000, s_tok=7168 * 2, n_slot=16,
    )
    hook = MegatronEPLBHook(
        EPLBRebalancer(topo, spec, EPLBConfig()), mode="observe", logger=print
    )

    # single process: build each EP rank's Lambda row from a simulated routing_map,
    # then assemble the full matrix the all-gather would have produced.
    rows = []
    for r in range(ep_size):
        rmap = _fake_router_routing_map(num_tokens=2048, num_experts=num_experts,
                                        top_k=top_k, hot=4, seed=100 + r)
        rows.append(lambda_row_from_routing_map(rmap))
    loads = Loads(torch.stack(rows, dim=0))

    plan = hook.reb.plan_from_lambda(loads)
    m = compute_metrics(plan, loads, topo, spec, EPLBConfig())
    print(f"EP={ep_size} experts={num_experts}: tau={m.tau} "
          f"imbalance={m.imbalance:.3f} replicas={m.total_replicas} phi_token={m.phi_token}")
    print("In a real run, hook.step(counts, layer_id, mb_id) all-gathers over the EP "
          "group and returns this plan on every rank, bit-identically.")


if __name__ == "__main__":
    main()

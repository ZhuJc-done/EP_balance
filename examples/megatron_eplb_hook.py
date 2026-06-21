"""Runnable CPU demo of wiring Scale-EPLB into Megatron-Core MoE (run: python -m examples.megatron_eplb_hook)."""

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

    # single process: build each EP rank's Lambda row from a simulated routing_map, then assemble the full matrix
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

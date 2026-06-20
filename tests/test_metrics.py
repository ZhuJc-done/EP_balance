"""Unit tests for the waterfill primitive, metrics, and the rebalancer flow."""

import torch

from eplb import (
    EPLBConfig,
    ProblemSpec,
    Topology,
    compute_metrics,
    solve,
)
from eplb.algorithm import _waterfill
from eplb.distributed import local_counts_from_routing
from eplb.integration import EPLBRebalancer
from eplb.loads import Loads
from sim.workload import make_loads


def test_waterfill_conserves_and_levels():
    base = torch.tensor([0, 5, 2, 9], dtype=torch.int64)
    tie = torch.zeros(4, dtype=torch.int64)
    add = _waterfill(20, base, tie)
    assert int(add.sum().item()) == 20
    assert torch.all(add >= 0)
    final = base + add
    # leveling: the max should not exceed what's necessary; min raised first
    assert int(final.max().item()) - int(final.min().item()) <= 1


def test_waterfill_single_dest():
    add = _waterfill(7, torch.tensor([3], dtype=torch.int64),
                     torch.tensor([0], dtype=torch.int64))
    assert int(add.item()) == 7


def test_waterfill_tie_break_prefers_low_cost():
    base = torch.tensor([0, 0], dtype=torch.int64)
    tie = torch.tensor([5, 1], dtype=torch.int64)  # dest 1 is cheaper
    add = _waterfill(1, base, tie)
    assert add[1].item() == 1 and add[0].item() == 0


def test_metrics_phi_token_zero_when_no_movement():
    """If every expert's tokens stay on its main rank, Phi_token == 0."""
    topo = Topology.from_nvlink_rdma(1, 4, intra_cost=1, inter_cost=8)
    R = topo.num_ranks
    spec = ProblemSpec.uniform_main_placement(
        num_experts=4, num_ranks=R, weight_bytes_each=1000,
        s_tok=1, n_slot=4,
    )
    # balanced: each rank r routes only expert r's tokens to itself (its main)
    lam = torch.zeros((R, 4), dtype=torch.int64)
    for r in range(R):
        lam[r, r] = 100
    loads = Loads(lam)
    plan = solve(loads, topo, spec, EPLBConfig())
    metrics = compute_metrics(plan, loads, topo, spec)
    assert metrics.phi_token == 0
    assert metrics.tau == 100
    assert metrics.imbalance == 1.0


def test_local_counts_from_routing():
    ids = torch.tensor([0, 0, 1, 3, 3, 3])
    counts = local_counts_from_routing(ids, num_experts=4)
    assert counts.tolist() == [2, 1, 0, 3]


def test_rebalancer_single_process():
    topo = Topology.from_nvlink_rdma(2, 4, intra_cost=1, inter_cost=8)
    R = topo.num_ranks
    spec = ProblemSpec.uniform_main_placement(
        num_experts=32, num_ranks=R, weight_bytes_each=44_000_000,
        s_tok=7168 * 2, n_slot=6,
    )
    reb = EPLBRebalancer(topo, spec, EPLBConfig(), cache_plans=False, ring_size=8)
    loads = make_loads(R, 32, tokens_per_rank=2048, top_k=6, skew=1.5,
                       hotspot_ranks=0.5, seed=9)

    # single-process: emulate the all-gathered Lambda directly
    plan = reb.plan_from_lambda(loads)
    assert plan.num_replicas().sum().item() >= 32

    # forward via the already-gathered Lambda path, then check backward recompute
    res = reb.rebalance_from_lambda(loads, layer_id=3, micro_batch_id=7)
    assert res.plan.tau >= 0
    bwd_plan = reb.backward(layer_id=3, micro_batch_id=7)
    assert bwd_plan.tau == res.plan.tau
    assert bwd_plan.equals(res.plan)

"""C1-C7 satisfaction and load-balancing efficacy across many random workloads."""

import pytest
import torch

from eplb import (
    EPLBConfig,
    ProblemSpec,
    Topology,
    check_constraints,
    compute_metrics,
    solve,
)
from sim.workload import make_loads


def _setup(nodes, gpus, experts, n_slot, cross_domain=True, inter_cost=8):
    R = nodes * gpus
    topo = Topology.from_nvlink_rdma(nodes, gpus, intra_cost=1, inter_cost=inter_cost)
    spec = ProblemSpec.uniform_main_placement(
        num_experts=experts,
        num_ranks=R,
        weight_bytes_each=44_000_000,
        s_tok=7168 * 2,
        n_slot=n_slot,
    )
    cfg = EPLBConfig(allow_cross_domain=cross_domain)
    return topo, spec, cfg, R


@pytest.mark.parametrize("skew", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("nodes,gpus,experts,n_slot", [
    (1, 8, 64, 16),
    (4, 8, 64, 4),
    (2, 4, 32, 6),
])
def test_constraints_hold(skew, nodes, gpus, experts, n_slot):
    topo, spec, cfg, R = _setup(nodes, gpus, experts, n_slot)
    loads = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                       hotspot_ranks=0.25, seed=skew_seed(skew, nodes, gpus))
    plan = solve(loads, topo, spec, cfg)
    report = check_constraints(plan, loads, topo, spec, cfg)
    assert report.ok, report.violations


def skew_seed(skew, nodes, gpus):
    return int(skew * 100) + nodes * 7 + gpus


@pytest.mark.parametrize("skew", [1.0, 2.0])
def test_improves_imbalance(skew):
    """The plan's makespan must be <= the no-replication baseline."""
    topo, spec, cfg, R = _setup(4, 8, 64, n_slot=4)
    loads = make_loads(R, 64, tokens_per_rank=4096, top_k=6, skew=skew,
                       hotspot_ranks=0.25, seed=7)
    plan = solve(loads, topo, spec, cfg)
    metrics = compute_metrics(plan, loads, topo, spec, cfg)

    base_load = torch.zeros(R, dtype=torch.int64)
    base_load.index_add_(0, spec.main_rank, loads.expert_load())
    base_tau = int(base_load.max().item())

    assert metrics.tau <= base_tau
    # with meaningful skew, replication should give a real win
    assert metrics.tau < base_tau


def test_no_cross_domain_keeps_replicas_in_domain():
    topo, spec, cfg, R = _setup(4, 8, 64, n_slot=4, cross_domain=False)
    loads = make_loads(R, 64, tokens_per_rank=4096, top_k=6, skew=2.0,
                       hotspot_ranks=0.25, seed=3)
    plan = solve(loads, topo, spec, cfg)
    dom = topo.domain_of_rank
    main_dom = dom[spec.main_rank]
    for e in range(spec.num_experts):
        for r in plan.replicas_of(e).tolist():
            assert int(dom[r].item()) == int(main_dom[e].item()), (
                f"expert {e} replicated cross-domain with cross-domain disabled"
            )
    assert check_constraints(plan, loads, topo, spec, cfg).ok


def test_slot_budget_respected():
    topo, spec, cfg, R = _setup(4, 8, 64, n_slot=3)
    loads = make_loads(R, 64, tokens_per_rank=4096, top_k=6, skew=2.0,
                       hotspot_ranks=0.5, seed=11)
    plan = solve(loads, topo, spec, cfg)
    assert int(plan.slots_used().max().item()) <= 3

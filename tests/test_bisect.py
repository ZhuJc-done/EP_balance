"""tau-bisection backend: a valid, deterministic plan that does not regress against the greedy solver."""

import torch

from eplb import EPLBConfig, ProblemSpec, Topology, check_constraints, compute_metrics, solve
from eplb.algorithm import solve_bisect
from sim.workload import make_loads

import pytest

CASES = [
    (1, 8, 64, 16),
    (4, 4, 32, 4),
    (2, 4, 16, 6),
    (4, 8, 64, 4),
]


def _baseline_tau(loads, spec, R):
    load = torch.zeros(R, dtype=torch.int64)
    load.index_add_(0, spec.main_rank, loads.expert_load())
    return int(load.max().item())


def _build(nodes, gpus, experts, n_slot, skew):
    R = nodes * gpus
    seed = int(skew * 100) + nodes + gpus
    loads = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                       hotspot_ranks=0.25, seed=seed)
    topo = Topology.from_nvlink_rdma(nodes, gpus, 1, 8)
    spec = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot)
    return loads, topo, spec, EPLBConfig()


@pytest.mark.parametrize("skew", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("nodes,gpus,experts,n_slot", CASES)
def test_bisect_constraints_and_no_regression(skew, nodes, gpus, experts, n_slot):
    loads, topo, spec, cfg = _build(nodes, gpus, experts, n_slot, skew)

    plan = solve_bisect(loads, topo, spec, cfg)
    report = check_constraints(plan, loads, topo, spec, cfg)
    assert report.ok, f"bisect violated constraints: {report.violations}"

    greedy = solve(loads, topo, spec, cfg)  # CPU tensors -> reference greedy path
    bisect_tau = int(plan.tau)
    greedy_tau = int(greedy.tau)

    # within a small margin of the greedy reference: the bisection descent trades a little
    # makespan for O(log L) exact routes instead of the greedy O(#replicas) re-routes.
    assert bisect_tau <= int(greedy_tau * 1.15) + 1, (
        f"bisect tau={bisect_tau} regressed vs greedy tau={greedy_tau}"
    )


@pytest.mark.parametrize("nodes,gpus,experts,n_slot", CASES)
def test_bisect_deterministic(nodes, gpus, experts, n_slot):
    loads, topo, spec, cfg = _build(nodes, gpus, experts, n_slot, skew=1.5)
    p1 = solve_bisect(loads, topo, spec, cfg)
    p2 = solve_bisect(loads, topo, spec, cfg)
    assert p1.equals(p2)


def test_bisect_helps_on_skew():
    """On a hot-spotted load the bisection plan should cut the makespan substantially."""
    loads, topo, spec, cfg = _build(4, 4, 32, 8, skew=2.0)
    R = 16
    base_tau = _baseline_tau(loads, spec, R)
    plan = solve_bisect(loads, topo, spec, cfg)
    m = compute_metrics(plan, loads, topo, spec, cfg)
    assert int(m.tau) < base_tau

"""Acceptance for the per-domain tau-descent Stage 2 (the GPU-kernel golden reference):
constraints, determinism, monotonicity, and no quality regression vs the greedy solver."""

import pytest
import torch

from eplb import EPLBConfig, ProblemSpec, Topology, check_constraints, compute_metrics, solve
from eplb.algorithm import solve_bisect
from sim.workload import make_loads

CASES = [
    (1, 8, 64, 16),
    (4, 4, 32, 4),
    (2, 4, 16, 6),
    (4, 8, 64, 4),
    (4, 4, 32, 8),
    (2, 8, 32, 6),
]


def _build(nodes, gpus, experts, n_slot, skew):
    R = nodes * gpus
    seed = int(skew * 100) + nodes + gpus
    loads = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                       hotspot_ranks=0.25, seed=seed)
    topo = Topology.from_nvlink_rdma(nodes, gpus, 1, 8)
    spec = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot)
    return loads, topo, spec


@pytest.mark.parametrize("skew", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("nodes,gpus,experts,n_slot", CASES)
def test_bisect_constraints_and_no_regression(skew, nodes, gpus, experts, n_slot):
    loads, topo, spec = _build(nodes, gpus, experts, n_slot, skew)

    plan = solve_bisect(loads, topo, spec, EPLBConfig())
    report = check_constraints(plan, loads, topo, spec, EPLBConfig())
    assert report.ok, f"bisect violated constraints: {report.violations}"

    greedy = solve(loads, topo, spec, EPLBConfig(stage2_mode="greedy"))
    bisect_tau = int(plan.tau)
    greedy_tau = int(greedy.tau)

    # within a small margin of the greedy reference (per-domain decomposition + frozen
    # cross-domain floor trade a little makespan for domain-parallel, kernel-friendly solving)
    assert bisect_tau <= int(greedy_tau * 1.15) + 1, (
        f"bisect tau={bisect_tau} regressed vs greedy tau={greedy_tau}"
    )


@pytest.mark.parametrize("nodes,gpus,experts,n_slot", CASES)
def test_bisect_deterministic(nodes, gpus, experts, n_slot):
    loads, topo, spec = _build(nodes, gpus, experts, n_slot, skew=1.5)
    p1 = solve_bisect(loads, topo, spec, EPLBConfig())
    p2 = solve_bisect(loads, topo, spec, EPLBConfig())
    assert p1.equals(p2)


@pytest.mark.parametrize("nodes,gpus,experts,n_slot", CASES)
def test_bisect_monotone_in_iters(nodes, gpus, experts, n_slot):
    """More descent steps can only lower (never raise) the makespan: best-tracking over a
    deterministic trajectory means tau(N) is non-increasing in tau_bisect_iters."""
    loads, topo, spec = _build(nodes, gpus, experts, n_slot, skew=2.0)
    taus = [
        int(solve_bisect(loads, topo, spec, EPLBConfig(tau_bisect_iters=k)).tau)
        for k in (1, 2, 4, 8, 24)
    ]
    for a, b in zip(taus, taus[1:]):
        assert b <= a, f"tau increased with more iters: {taus}"


def test_bisect_via_config_switch():
    """stage2_mode='bisect' routes solve() to the per-domain descent reference."""
    loads, topo, spec = _build(4, 8, 64, 4, skew=2.0)
    via_cfg = solve(loads, topo, spec, EPLBConfig(stage2_mode="bisect"))
    direct = solve_bisect(loads, topo, spec, EPLBConfig())
    assert via_cfg.equals(direct)


def test_bisect_helps_on_skew():
    """On a hot-spotted load with free slots, the descent cuts the makespan substantially."""
    loads, topo, spec = _build(4, 4, 32, 8, skew=2.0)
    R = 16
    base = torch.zeros(R, dtype=torch.int64)
    base.index_add_(0, spec.main_rank, loads.expert_load())
    base_tau = int(base.max().item())
    m = compute_metrics(solve_bisect(loads, topo, spec, EPLBConfig()), loads, topo, spec)
    assert int(m.tau) < base_tau


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("skew", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("nodes,gpus,experts,n_slot", CASES)
def test_bisect_gpu_bit_identical(skew, nodes, gpus, experts, n_slot):
    """The per-domain Triton kernel is bit-identical to the CPU reference (x, q, tau)."""
    from eplb.triton_solve import solve_bisect_fused

    loads, topo, spec = _build(nodes, gpus, experts, n_slot, skew)
    cfg = EPLBConfig()
    plan_cpu = solve_bisect(loads, topo, spec, cfg)

    dev = torch.device("cuda")
    R = nodes * gpus
    seed = int(skew * 100) + nodes + gpus
    loads_g = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                         hotspot_ranks=0.25, seed=seed, device=dev)
    topo_g = Topology.from_nvlink_rdma(nodes, gpus, 1, 8, device=dev)
    spec_g = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot, device=dev)
    plan_g = solve_bisect_fused(loads_g, topo_g, spec_g, cfg)

    assert torch.equal(plan_cpu.x, plan_g.x.cpu())
    assert torch.equal(plan_cpu.q, plan_g.q.cpu())
    assert int(plan_cpu.tau) == int(plan_g.tau)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_bisect_gpu_zero_sync():
    """The steady-state kernel solve issues no blocking CPU<->GPU synchronization."""
    from eplb.triton_solve import solve_bisect_fused

    dev = torch.device("cuda")
    R, E, n_slot = 16, 32, 6
    loads = make_loads(R, E, tokens_per_rank=2048, top_k=6, skew=2.0,
                       hotspot_ranks=0.25, seed=7, device=dev)
    topo = Topology.from_nvlink_rdma(4, 4, 1, 8, device=dev)
    spec = ProblemSpec.uniform_main_placement(E, R, 44_000_000, 7168 * 2, n_slot, device=dev)
    cfg = EPLBConfig()
    solve_bisect_fused(loads, topo, spec, cfg)  # warm up Triton compilation
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        solve_bisect_fused(loads, topo, spec, cfg)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()

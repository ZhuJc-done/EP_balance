"""GPU solver equivalence: the Triton-backed solve is bit-identical to the CPU reference (skipped without CUDA)."""

import pytest
import torch

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.loads import Loads
from eplb.plan import Plan
from sim.workload import make_loads

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")


def _cpu_plan(p: Plan) -> Plan:
    return Plan(x=p.x.cpu(), q=p.q.cpu(), tau=p.tau)


@pytest.mark.parametrize("skew", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("nodes,gpus,experts,n_slot", [
    (1, 8, 64, 16),
    (4, 4, 32, 4),
    (2, 4, 16, 6),
    (4, 8, 64, 4),
])
def test_gpu_solve_bit_identical(skew, nodes, gpus, experts, n_slot):
    R = nodes * gpus
    loads_cpu = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                           hotspot_ranks=0.25, seed=int(skew * 100) + nodes + gpus)
    topo_cpu = Topology.from_nvlink_rdma(nodes, gpus, 1, 8)
    spec_cpu = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot)
    cfg = EPLBConfig()

    plan_cpu = solve(loads_cpu, topo_cpu, spec_cpu, cfg)

    dev = torch.device("cuda")
    loads_gpu = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                           hotspot_ranks=0.25, seed=int(skew * 100) + nodes + gpus, device=dev)
    topo_gpu = Topology.from_nvlink_rdma(nodes, gpus, 1, 8, device=dev)
    spec_gpu = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot, device=dev)
    plan_gpu = solve(loads_gpu, topo_gpu, spec_gpu, cfg)

    assert plan_cpu.equals(_cpu_plan(plan_gpu)), (
        f"GPU plan diverged: tau cpu={plan_cpu.tau} gpu={plan_gpu.tau}"
    )


@pytest.mark.parametrize("u_min", [1, 2, 4])
def test_gpu_incremental_transfer_respects_quota_floor(u_min):
    lam = torch.tensor([[16, 8], [8, 16], [12, 12]], dtype=torch.int64)
    cfg = EPLBConfig(u_min=u_min, max_stage2_iters=8)

    topo_cpu = Topology.from_nvlink_rdma(1, 3)
    spec_cpu = ProblemSpec(
        2, torch.tensor([0, 1]), torch.tensor([1, 1]), 1, 2,
    )
    plan_cpu = solve(Loads(lam), topo_cpu, spec_cpu, cfg)

    topo_gpu = Topology.from_nvlink_rdma(1, 3, device="cuda")
    spec_gpu = ProblemSpec(
        2,
        torch.tensor([0, 1], device="cuda"),
        torch.tensor([1, 1], device="cuda"),
        1,
        2,
    )
    plan_gpu = solve(Loads(lam.cuda()), topo_gpu, spec_gpu, cfg)

    assert plan_cpu.equals(_cpu_plan(plan_gpu))
    nonzero = plan_gpu.q[plan_gpu.q > 0]
    assert nonzero.numel() == 0 or int(nonzero.min()) >= u_min


def test_fused_solver_has_no_host_sync_after_warmup():
    dev = torch.device("cuda")
    lam = torch.tensor([[11], [0]], dtype=torch.int64, device=dev)
    topo = Topology.from_nvlink_rdma(1, 2, device=dev)
    spec = ProblemSpec(
        1,
        torch.tensor([0], device=dev),
        torch.tensor([1], device=dev),
        1,
        1,
    )
    cfg = EPLBConfig(max_stage2_iters=8)

    # Compile first; sync-debug then guards the steady-state solve path.
    solve(Loads(lam), topo, spec, cfg, validate=False)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan = solve(Loads(lam), topo, spec, cfg, validate=False)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert plan.tau.is_cuda


def test_gpu_floor_aware_cross_domain_fallback_matches_cpu():
    lam = torch.tensor([[20], [10], [6]], dtype=torch.int64)
    dom = torch.tensor([0, 1, 2], dtype=torch.int64)
    cost = torch.ones((3, 3), dtype=torch.int64)
    cost.fill_diagonal_(0)
    cfg = EPLBConfig(u_min=3)

    topo_cpu = Topology(dom, cost)
    spec_cpu = ProblemSpec(
        1, torch.tensor([0]), torch.tensor([15]), s_tok=1, n_slot=1,
    )
    plan_cpu = solve(Loads(lam), topo_cpu, spec_cpu, cfg)

    topo_gpu = Topology(dom.cuda(), cost.cuda())
    spec_gpu = ProblemSpec(
        1,
        torch.tensor([0], device="cuda"),
        torch.tensor([15], device="cuda"),
        s_tok=1,
        n_slot=1,
    )
    plan_gpu = solve(Loads(lam.cuda()), topo_gpu, spec_gpu, cfg)

    assert plan_cpu.equals(_cpu_plan(plan_gpu))
    assert int(plan_gpu.q[plan_gpu.q > 0].min()) >= cfg.u_min

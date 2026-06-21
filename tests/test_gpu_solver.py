"""GPU solver equivalence: the Triton-backed solve is bit-identical to the CPU reference (skipped without CUDA)."""

import pytest
import torch

from eplb import EPLBConfig, ProblemSpec, Topology, solve
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

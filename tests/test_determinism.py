"""Determinism: the plan is a bit-identical function of Lambda (enables CPU-sync-free E3)."""

import torch

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.loads import Loads
from sim.workload import make_loads


def _setup():
    topo = Topology.from_nvlink_rdma(4, 8, intra_cost=1, inter_cost=8)
    R = topo.num_ranks
    spec = ProblemSpec.uniform_main_placement(
        num_experts=64, num_ranks=R, weight_bytes_each=44_000_000,
        s_tok=7168 * 2, n_slot=4,
    )
    cfg = EPLBConfig()
    return topo, spec, cfg, R


def test_resolve_bit_identical():
    topo, spec, cfg, R = _setup()
    loads = make_loads(R, 64, tokens_per_rank=4096, top_k=6, skew=1.5,
                       hotspot_ranks=0.25, seed=42)
    p1 = solve(loads, topo, spec, cfg)
    p2 = solve(Loads(loads.lam.clone()), topo, spec, cfg)
    assert p1.equals(p2)


def test_invariant_to_input_memory_layout():
    """A value-identical but non-contiguous Lambda must yield a bit-identical plan."""
    topo, spec, cfg, R = _setup()
    loads = make_loads(R, 64, tokens_per_rank=4096, top_k=6, skew=1.5,
                       hotspot_ranks=0.25, seed=123)
    base_plan = solve(loads, topo, spec, cfg)

    # build a value-identical but non-contiguous tensor via a transpose round-trip
    noncontig = loads.lam.t().contiguous().t()
    assert not noncontig.is_contiguous()
    assert torch.equal(noncontig, loads.lam)
    alt_plan = solve(Loads(noncontig), topo, spec, cfg)

    assert alt_plan.equals(base_plan)


def test_repeated_solves_stable_over_many_workloads():
    """Re-solving is bit-identical across a sweep of seeds/skews."""
    topo, spec, cfg, R = _setup()
    for seed in range(6):
        for skew in (0.0, 1.0, 2.0):
            loads = make_loads(R, 64, tokens_per_rank=3072, top_k=6, skew=skew,
                               hotspot_ranks=0.25, seed=seed)
            p1 = solve(loads, topo, spec, cfg)
            p2 = solve(Loads(loads.lam.clone()), topo, spec, cfg)
            assert p1.equals(p2), f"non-deterministic at seed={seed} skew={skew}"

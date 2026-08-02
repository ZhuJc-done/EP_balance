"""Focused tests for monotonic, source-quota-preserving Stage 2 updates."""

import torch
import pytest

from eplb import EPLBConfig, ProblemSpec, Topology, check_constraints, solve
from eplb.loads import Loads


def test_half_gap_uses_floor_and_does_not_swap_bottleneck():
    topo = Topology.from_nvlink_rdma(1, 2)
    spec = ProblemSpec(
        num_experts=1,
        main_rank=torch.tensor([0]),
        weight_bytes=torch.tensor([1]),
        s_tok=1,
        n_slot=1,
    )
    loads = Loads(torch.tensor([[11], [0]], dtype=torch.int64))

    plan = solve(loads, topo, spec, EPLBConfig(max_stage2_iters=8))

    assert plan.rank_load().tolist() == [6, 5]
    assert plan.x.tolist() == [[1, 1]]
    assert check_constraints(plan, loads, topo, spec).ok


def test_existing_replica_can_absorb_quota_without_using_a_slot():
    topo = Topology.from_nvlink_rdma(1, 3)
    spec = ProblemSpec(
        num_experts=2,
        main_rank=torch.tensor([0, 1]),
        weight_bytes=torch.tensor([1, 1]),
        s_tok=1,
        n_slot=2,
    )
    # A initially overloads r0 and B overloads r1.  The first two iterations
    # create A@r2 and B@r0; iteration three must reuse A@r2.
    loads = Loads(torch.tensor([[100, 100], [0, 0], [0, 0]], dtype=torch.int64))
    cfg = EPLBConfig(max_stage2_iters=3)

    plan = solve(loads, topo, spec, cfg)
    contribution = plan.q.sum(dim=0)

    assert plan.x[0, 2].item() == 1
    assert contribution[0, 2].item() > 50  # the third action reused A@r2
    assert plan.slots_used().tolist() == [2, 1, 1]
    assert check_constraints(plan, loads, topo, spec, cfg).ok


def test_quota_floor_applies_to_initial_cross_domain_fallback():
    # Domain 2 does not pass C6, so its six tokens fall back to differently
    # loaded A replicas in domains 0/1.  Both non-zero pieces must meet u_min.
    dom = torch.tensor([0, 1, 2], dtype=torch.int64)
    cost = torch.ones((3, 3), dtype=torch.int64)
    cost.fill_diagonal_(0)
    topo = Topology(dom, cost)
    spec = ProblemSpec(
        1, torch.tensor([0]), torch.tensor([15]), s_tok=1, n_slot=1,
    )
    loads = Loads(torch.tensor([[20], [10], [6]], dtype=torch.int64))
    cfg = EPLBConfig(u_min=3)

    plan = solve(loads, topo, spec, cfg)

    nonzero = plan.q[plan.q > 0]
    assert int(nonzero.min()) >= 3
    assert check_constraints(plan, loads, topo, spec, cfg).ok


def test_floor_fragment_fallback_still_makes_progress():
    topo = Topology.from_nvlink_rdma(1, 2)
    spec = ProblemSpec(
        2,
        main_rank=torch.tensor([0, 0]),
        weight_bytes=torch.tensor([1, 1]),
        s_tok=1,
        n_slot=3,
    )
    loads = Loads(torch.tensor([[6, 4], [0, 0]], dtype=torch.int64))
    cfg = EPLBConfig(u_min=4, max_stage2_iters=1)

    plan = solve(loads, topo, spec, cfg)

    assert int(plan.theta) < 10
    assert check_constraints(plan, loads, topo, spec, cfg).ok


def test_rejects_intrinsically_infeasible_quota_floor():
    topo = Topology.from_nvlink_rdma(1, 2)
    spec = ProblemSpec(
        1, torch.tensor([0]), torch.tensor([1]), s_tok=1, n_slot=1,
    )
    loads = Loads(torch.tensor([[2], [0]], dtype=torch.int64))

    with pytest.raises(ValueError, match="u_min is infeasible"):
        solve(loads, topo, spec, EPLBConfig(u_min=3))


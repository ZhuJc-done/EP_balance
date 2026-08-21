"""Runtime-contract tests for baseline plan-solver plugins."""

import pytest
import torch

from baseline.training import (
    DeepSeekPlanSolver,
    FasterMoEPlanSolver,
    FlexMoEPlanSolver,
    LPLBPlanSolver,
    make_training_plan_solver,
)
from eplb import EPLBConfig, ProblemSpec, Topology, plan_from_placement
from eplb.integration.rebalancer import EPLBRebalancer
from eplb.loads import Loads
from eplb.plan import Plan


def _spec(
    num_experts: int,
    num_ranks: int,
    slots_per_rank: int,
    *,
    weight_bytes: int = 1000,
    s_tok: int = 100,
) -> ProblemSpec:
    mains_per_rank = num_experts // num_ranks
    return ProblemSpec(
        num_experts=num_experts,
        main_rank=torch.arange(num_experts) // mains_per_rank,
        weight_bytes=torch.full((num_experts,), weight_bytes, dtype=torch.int64),
        s_tok=s_tok,
        n_slot=slots_per_rank,
    )


def _assert_runtime_contract(
    plan: Plan,
    loads: Loads,
    spec: ProblemSpec,
) -> None:
    num_experts = spec.num_experts
    assert plan.x.dtype == torch.int8
    assert plan.q.dtype == torch.int64
    assert torch.equal(plan.q.sum(dim=2), loads.omega)
    assert not torch.any(
        plan.q * (plan.x == 0).to(plan.q.dtype).unsqueeze(0)
    )
    assert torch.all(plan.x.sum(dim=0) <= int(spec.n_slot))
    expert = torch.arange(num_experts)
    assert torch.all(plan.x[expert, spec.main_rank] == 1)
    assert int(plan.theta) == int(plan.q.sum(dim=(0, 1)).max())


def test_fixed_placement_quota_prefers_source_domain_and_conserves_tokens():
    loads = Loads(
        torch.tensor(
            [
                [11, 7],
                [9, 5],
                [13, 8],
                [6, 10],
            ],
            dtype=torch.int64,
        )
    )
    topology = Topology.from_nvlink_rdma(2, 2)
    spec = ProblemSpec(
        num_experts=2,
        main_rank=torch.tensor([0, 2]),
        weight_bytes=torch.ones(2, dtype=torch.int64),
        s_tok=1,
        n_slot=2,
    )
    placement = torch.tensor(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.int8,
    )

    plan = plan_from_placement(
        loads,
        topology,
        spec,
        placement,
        EPLBConfig(u_min=3),
    )

    _assert_runtime_contract(plan, loads, spec)
    assert int(plan.q[0, 0, 2]) == 0
    assert int(plan.q[2, 0, 2]) == 13
    nonzero = plan.q[plan.q > 0]
    assert int(nonzero.min()) >= 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fixed_placement_quota_has_no_cuda_host_sync_after_warmup():
    loads = Loads(torch.tensor([[8, 4], [4, 8]], device="cuda"))
    topology = Topology.from_nvlink_rdma(1, 2, device="cuda")
    spec = _spec(2, 2, 2)
    spec = ProblemSpec(
        spec.num_experts,
        spec.main_rank.cuda(),
        spec.weight_bytes.cuda(),
        spec.s_tok,
        spec.n_slot,
    )
    placement = torch.ones((2, 2), dtype=torch.int8, device="cuda")

    plan_from_placement(
        loads, topology, spec, placement, EPLBConfig(), validate=False
    )
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan = plan_from_placement(
            loads, topology, spec, placement, EPLBConfig(), validate=False
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert plan.theta.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fixed_placement_cuda_matches_cpu_reference():
    omega = torch.tensor(
        [
            [11, 7],
            [9, 5],
            [13, 8],
            [6, 10],
        ],
        dtype=torch.int64,
    )
    placement = torch.tensor(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.int8,
    )
    spec_cpu = ProblemSpec(
        2,
        torch.tensor([0, 2]),
        torch.ones(2, dtype=torch.int64),
        1,
        2,
    )
    cfg = EPLBConfig(u_min=3)
    cpu = plan_from_placement(
        Loads(omega),
        Topology.from_nvlink_rdma(2, 2),
        spec_cpu,
        placement,
        cfg,
    )
    gpu = plan_from_placement(
        Loads(omega.cuda()),
        Topology.from_nvlink_rdma(2, 2, device="cuda"),
        ProblemSpec(
            2,
            spec_cpu.main_rank.cuda(),
            spec_cpu.weight_bytes.cuda(),
            1,
            2,
        ),
        placement.cuda(),
        cfg,
    )
    torch.cuda.synchronize()

    assert torch.equal(gpu.x.cpu(), cpu.x)
    assert torch.equal(gpu.q.cpu(), cpu.q)
    assert int(gpu.theta) == int(cpu.theta)


def test_rebalancer_accepts_a_plan_solver_plugin():
    loads = Loads(torch.tensor([[3, 1], [1, 3]], dtype=torch.int64))
    topology = Topology.from_nvlink_rdma(1, 2)
    spec = _spec(2, 2, 1)
    calls = []

    def plugin(current, topo, current_spec, cfg):
        calls.append((current, topo, current_spec, cfg))
        return plan_from_placement(
            current,
            topo,
            current_spec,
            torch.eye(2, dtype=torch.int8),
            cfg,
        )

    rebalancer = EPLBRebalancer(topology, spec, plan_solver=plugin)
    plan = rebalancer.plan_from_omega(loads)

    assert len(calls) == 1
    _assert_runtime_contract(plan, loads, spec)


def test_fastermoe_plugin_keeps_shadowed_tokens_on_the_source_rank():
    omega = torch.tensor(
        [
            [100, 1, 1, 1],
            [100, 1, 1, 1],
            [100, 1, 1, 1],
            [100, 1, 1, 1],
        ],
        dtype=torch.int64,
    )
    loads = Loads(omega)
    topology = Topology.from_nvlink_rdma(1, 4)
    spec = _spec(4, 4, 2)

    plan = FasterMoEPlanSolver()(loads, topology, spec, EPLBConfig())

    _assert_runtime_contract(plan, loads, spec)
    assert torch.all(plan.x[0] == 1)
    assert torch.equal(
        plan.q[:, 0, :],
        torch.diag(omega[:, 0]),
    )


def test_main_fixed_deepseek_plugin_builds_a_valid_training_plan():
    omega = torch.full((4, 8), 2, dtype=torch.int64)
    omega[:, 0] += 200
    loads = Loads(omega)
    topology = Topology.from_nvlink_rdma(1, 4)
    spec = _spec(8, 4, 3)

    plan = DeepSeekPlanSolver(num_groups=1)(
        loads, topology, spec, EPLBConfig()
    )

    _assert_runtime_contract(plan, loads, spec)
    assert int(plan.x.sum()) == 12
    assert int(plan.x[0].sum()) > 1


def test_main_fixed_flexmoe_plugin_builds_a_valid_training_plan():
    omega = torch.full((4, 4), 1, dtype=torch.int64)
    omega[:, 0] += 200
    loads = Loads(omega)
    topology = Topology.from_nvlink_rdma(1, 4)
    spec = _spec(4, 4, 2)

    plan = FlexMoEPlanSolver()(loads, topology, spec, EPLBConfig())

    _assert_runtime_contract(plan, loads, spec)
    assert int(plan.x[0].sum()) > 1


def test_lplb_plugin_converts_lp_ratios_to_main_fixed_integer_quotas():
    class FakePlanner:
        group_size = 2
        n_group = 1
        num_redundants = 1
        combined_redundant_experts = 1
        phy2log = None

        def solve_probs(self, workload, available):
            del workload, available
            # Original rank 0 retains 25%; rank 1 retains 75%.
            return torch.tensor([[[0.25], [0.75]]])

    class FakeLPLB:
        planner = FakePlanner()
        r2o = torch.tensor([[1], [0]], dtype=torch.int32)

    loads = Loads(
        torch.tensor(
            [
                [8, 1, 4, 1],
                [4, 1, 8, 1],
            ],
            dtype=torch.int64,
        )
    )
    topology = Topology.from_nvlink_rdma(1, 2)
    spec = _spec(4, 2, 3)
    solver = LPLBPlanSolver()
    solver._planner = FakeLPLB()
    solver._shape = (4, 2, 1)

    plan = solver(loads, topology, spec, EPLBConfig())

    _assert_runtime_contract(plan, loads, spec)
    assert torch.all(plan.x[0] == 1)
    assert torch.all(plan.x[2] == 1)
    assert plan.q[0, 0].tolist() == [2, 6]
    assert plan.q[1, 0].tolist() == [1, 3]
    assert plan.q[0, 2].tolist() == [1, 3]
    assert plan.q[1, 2].tolist() == [2, 6]


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("scale-eplb", "scale"),
        ("faster-moe", "fastermoe"),
        ("deepseek-eplb", "deepseek"),
        ("flex-moe", "flexmoe"),
        ("lplb", "lplb"),
    ],
)
def test_training_solver_factory_aliases(alias, expected):
    assert make_training_plan_solver(alias).name == expected


def test_training_solver_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown EPLB plan solver"):
        make_training_plan_solver("not-a-solver")

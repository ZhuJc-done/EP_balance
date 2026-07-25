"""Correctness smoke tests for the comparison adapters."""

import argparse

import pytest
import torch

from baseline.adapters import (
    FlexMoECostModel,
    LPLBBaseline,
    LPLBUnavailableError,
    ShadowCostModel,
    cube8_topology,
    run_deepseek_eplb,
    run_fastermoe,
    run_flexmoe,
    run_scale_eplb,
)
from baseline.benchmark import benchmark_trace
from baseline.deepseek_eplb import rebalance_experts
from baseline.fastermoe import select_shadow_experts
from baseline.flexmoe import flexmoe_schedule
from eplb import EPLBConfig, ProblemSpec, Topology
from eplb.integration.megatron import MegatronEPLBHook
from eplb.integration.rebalancer import EPLBRebalancer
from eplb.loads import Loads


def test_vendored_deepseek_eplb_reference_example():
    weight = torch.tensor(
        [
            [90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
            [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27],
        ]
    )

    phy2log, log2phy, logcnt = rebalance_experts(weight, 16, 4, 2, 8)

    assert phy2log.tolist() == [
        [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
        [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1],
    ]
    assert torch.all(logcnt >= 1)
    assert torch.all(log2phy[:, :, 0] >= 0)


def test_deepseek_adapter_uses_all_slots_and_conserves_predicted_load():
    loads = Loads(
        torch.tensor(
            [
                [20, 10, 5, 5],
                [5, 20, 10, 5],
                [5, 5, 20, 10],
                [10, 5, 5, 20],
            ],
            dtype=torch.int64,
        )
    )

    result = run_deepseek_eplb(
        loads, num_nodes=1, num_gpus=4, n_slot=2, num_groups=1
    )

    assert result.placement is not None
    assert result.metadata["replicas"] == 8
    assert result.placement.sum().item() == result.metadata["replicas"]
    assert result.placement.sum(dim=1).min().item() >= 1
    assert torch.isclose(result.rank_load.sum(), loads.lam.sum().double())


def test_deepseek_adapter_places_from_history_not_current_load():
    current = Loads(torch.tensor([[1, 1, 1, 50]] * 4, dtype=torch.int64))
    history = Loads(torch.tensor([[50, 1, 1, 1]] * 4, dtype=torch.int64))

    result = run_deepseek_eplb(
        current,
        num_nodes=1,
        num_gpus=4,
        n_slot=2,
        num_groups=1,
        placement_loads=history,
    )

    assert result.placement is not None
    assert result.placement[0].sum().item() > result.placement[3].sum().item()


def test_scale_adapter_reports_realized_quota_load():
    loads = Loads(torch.tensor([[20, 0], [0, 20]], dtype=torch.int64))
    topology = Topology.from_nvlink_rdma(1, 2)
    spec = ProblemSpec(
        2,
        main_rank=torch.tensor([0, 1]),
        weight_bytes=torch.tensor([1, 1]),
        s_tok=1,
        n_slot=2,
    )

    result = run_scale_eplb(loads, topology, spec, EPLBConfig())

    assert result.metadata["load_kind"] == "realized quota load"
    assert result.rank_load.sum().item() == loads.lam.sum().item()


def test_fastermoe_shadows_hot_expert_and_conserves_load():
    # Expert 0 (home rank 0) is hammered by every rank; the rest are cool.
    lam = torch.tensor(
        [
            [100, 1, 1, 1],
            [100, 1, 1, 1],
            [100, 1, 1, 1],
            [100, 1, 1, 1],
        ],
        dtype=torch.int64,
    )
    loads = Loads(lam)
    spec = ProblemSpec(
        num_experts=4,
        main_rank=torch.tensor([0, 1, 2, 3]),
        weight_bytes=torch.tensor([1000, 1000, 1000, 1000]),
        s_tok=100,
        n_slot=4,
    )

    result = run_fastermoe(loads, spec, num_ranks=4)

    baseline_tau = float(lam.sum(dim=0).max().item())  # 400 tokens on rank 0
    assert result.metadata["shadowed_experts"] >= 1
    assert result.placement is not None
    # A shadowed expert is replicated onto every rank.
    assert bool(result.placement[0].all())
    # Redistribution strictly lowers the peak and conserves total token work.
    assert result.tau < baseline_tau
    assert torch.isclose(result.rank_load.sum(), loads.lam.sum().double())


def test_fastermoe_leaves_balanced_load_untouched():
    loads = Loads(torch.full((4, 4), 10, dtype=torch.int64))
    spec = ProblemSpec(
        num_experts=4,
        main_rank=torch.tensor([0, 1, 2, 3]),
        weight_bytes=torch.tensor([44_000_000] * 4),
        s_tok=14336,
        n_slot=4,
    )

    shadow_mask, rank_load = select_shadow_experts(
        loads.lam, spec.main_rank, spec.weight_bytes, spec.s_tok, 4, ShadowCostModel()
    )

    # Already balanced: broadcasting weights only adds overhead, so shadow nothing.
    assert int(shadow_mask.sum().item()) == 0
    assert torch.equal(rank_load, torch.full((4,), 40.0, dtype=torch.float64))


def test_flexmoe_replicates_hot_expert_and_conserves_load():
    # Expert 0 is extremely hot; the other three are cold. R=4, n_slot=2 -> 8 slots.
    lam = torch.tensor(
        [
            [200, 1, 1, 1],
            [200, 1, 1, 1],
            [200, 1, 1, 1],
            [200, 1, 1, 1],
        ],
        dtype=torch.int64,
    )
    loads = Loads(lam)
    spec = ProblemSpec(
        num_experts=4,
        main_rank=torch.tensor([0, 1, 2, 3]),
        weight_bytes=torch.tensor([1000, 1000, 1000, 1000]),
        s_tok=100,
        n_slot=2,
    )

    result = run_flexmoe(loads, spec, num_ranks=4, cost=FlexMoECostModel(threshold=1.2))

    baseline_tau = float(lam.sum(dim=0).max().item())  # 800 tokens if not replicated
    # The hot expert must receive multiple vExperts, cutting the peak load.
    assert result.metadata["replicas"] > spec.num_experts
    assert bool(result.placement[0].sum() >= 2)
    assert result.tau < baseline_tau
    assert torch.isclose(result.rank_load.sum(), loads.lam.sum().double())
    assert result.metadata["balance_ratio"] <= 1.2 + 1e-6


def test_flexmoe_threshold_controls_replication_effort():
    lam = torch.tensor([[100, 20, 5, 1]] * 4, dtype=torch.int64)
    weight_bytes = torch.tensor([1] * 4)

    tight = flexmoe_schedule(lam, weight_bytes, 4, 3, FlexMoECostModel(threshold=1.05))
    loose = flexmoe_schedule(lam, weight_bytes, 4, 3, FlexMoECostModel(threshold=2.0))

    tight_ratio = tight[1].max().item() / tight[1].mean().item()
    loose_ratio = loose[1].max().item() / loose[1].mean().item()

    # A tighter threshold spends more vExperts and reaches a flatter distribution.
    assert sum(tight[0]) >= sum(loose[0])
    assert tight_ratio <= loose_ratio + 1e-9


def test_cube8_topology_matches_lplb_shape_contract():
    topology = cube8_topology()

    assert topology.shape == (8, 2)
    assert topology.dtype == torch.int32
    assert torch.all((topology >= 0) & (topology < 8))


def test_lplb_without_cuda_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(LPLBUnavailableError, match="requires CUDA"):
        LPLBBaseline(num_experts=64, ep_size=8, n_slot=10)


def _write_trace(path, *, ranks=4, experts=8, n_slot=4, samples=3, seed=0):
    """Emit a self-describing routing trace matching observe mode's dump format."""
    topo = Topology.from_nvlink_rdma(1, ranks)
    num_local = experts // ranks
    meta = {
        "num_ranks": ranks,
        "num_experts": experts,
        "s_tok": 128,
        "n_slot": n_slot,
        "num_domains": topo.num_domains,
        "main_rank": torch.arange(experts) // num_local,
        "weight_bytes": torch.full((experts,), 1000, dtype=torch.int64),
        "domain_of_rank": topo.domain_of_rank,
        "cost": topo.cost,
    }
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(samples):
        lam = torch.randint(1, 40, (ranks, experts), generator=gen, dtype=torch.int64)
        lam[:, 0] += 300  # a persistent hot expert homed on rank 0
        rows.append({"layer": 0, "mb": i, "lam": lam})
    torch.save({"meta": meta, "samples": rows}, path)


def _trace_args(trace_path, strategies):
    return argparse.Namespace(
        strategies=strategies,
        trace=str(trace_path),
        trace_max_samples=0,
        num_groups=4,
        fastermoe_bw_net=50e9 / 8,
        fastermoe_bw_mm=11.5e12,
        flexmoe_threshold=1.2,
        lplb_root="/home/tiger/LPLB",
        require_lplb=False,
    )


def test_observer_hook_dumps_replayable_trace(tmp_path):
    # Single-process (R=1) exercise of the observe-mode trace dump, no Megatron needed.
    topo = Topology.from_nvlink_rdma(1, 1)
    spec = ProblemSpec.uniform_main_placement(
        num_experts=4, num_ranks=1, weight_bytes_each=1000, s_tok=128, n_slot=4
    )
    reb = EPLBRebalancer(topo, spec, EPLBConfig())
    out = tmp_path / "trace.pt"
    hook = MegatronEPLBHook(reb, mode="observe", logger=None, trace_out=str(out), trace_every=1)

    hook.step(torch.tensor([10, 3, 2, 1]), layer_id=0, micro_batch_id=0)
    hook.step(torch.tensor([1, 2, 3, 10]), layer_id=0, micro_batch_id=1)

    assert out.exists()
    blob = torch.load(out, weights_only=False)
    assert blob["meta"]["num_ranks"] == 1
    assert blob["meta"]["num_experts"] == 4
    assert blob["meta"]["n_slot"] == 4
    assert len(blob["samples"]) == 2
    assert blob["samples"][0]["lam"].shape == (1, 4)
    assert blob["samples"][0]["lam"].tolist() == [[10, 3, 2, 1]]


def test_observer_hook_respects_trace_max(tmp_path):
    topo = Topology.from_nvlink_rdma(1, 1)
    spec = ProblemSpec.uniform_main_placement(
        num_experts=4, num_ranks=1, weight_bytes_each=1000, s_tok=128, n_slot=4
    )
    reb = EPLBRebalancer(topo, spec, EPLBConfig())
    out = tmp_path / "trace.pt"
    hook = MegatronEPLBHook(
        reb, mode="observe", logger=None, trace_out=str(out), trace_max=1, trace_every=1
    )

    for mb in range(4):
        hook.step(torch.tensor([10, 3, 2, 1]), layer_id=0, micro_batch_id=mb)

    blob = torch.load(out, weights_only=False)
    assert len(blob["samples"]) == 1


def test_benchmark_trace_replays_all_baselines(tmp_path):
    trace = tmp_path / "trace.pt"
    _write_trace(trace, samples=3)

    rows = benchmark_trace(_trace_args(trace, "scale,eplb,fastermoe,flexmoe"))

    by_name = {r["strategy"]: r for r in rows}
    assert set(by_name) == {"scale-eplb", "deepseek-eplb", "fastermoe", "flexmoe"}

    # The naive home-rank placement piles the hot expert onto its home rank.
    blob = torch.load(trace, weights_only=False)
    main_rank = blob["meta"]["main_rank"]
    naive = []
    for sample in blob["samples"]:
        expert_load = sample["lam"].sum(0).double()
        rank_load = torch.zeros(4, dtype=torch.float64).index_add_(0, main_rank, expert_load)
        naive.append(rank_load.max().item())
    naive_tau = sum(naive) / len(naive)

    for row in rows:
        assert row["samples"] == 3
        assert row["solve_ms_mean"] >= 0.0
        assert row["quality_imbalance_mean"] >= 1.0 - 1e-6
        # Every balancer must beat the naive home-rank peak on this hot-expert trace.
        assert 0.0 < row["quality_tau_mean"] < naive_tau


def test_benchmark_trace_respects_max_samples(tmp_path):
    trace = tmp_path / "trace.pt"
    _write_trace(trace, samples=5)

    args = _trace_args(trace, "scale")
    args.trace_max_samples = 2
    rows = benchmark_trace(args)

    assert rows[0]["samples"] == 2


def test_benchmark_trace_skips_lplb_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trace = tmp_path / "trace.pt"
    _write_trace(trace, samples=2)

    rows = benchmark_trace(_trace_args(trace, "scale,lplb"))

    by_name = {r["strategy"]: r for r in rows}
    assert "skipped" in by_name["lplb"]
    assert by_name["scale-eplb"]["samples"] == 2

"""Correctness smoke tests for the comparison adapters."""

import pytest
import torch

from baseline.adapters import (
    LPLBBaseline,
    LPLBUnavailableError,
    cube8_topology,
    run_deepseek_eplb,
    run_scale_eplb,
)
from baseline.deepseek_eplb import rebalance_experts
from eplb import EPLBConfig, ProblemSpec, Topology
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


def test_cube8_topology_matches_lplb_shape_contract():
    topology = cube8_topology()

    assert topology.shape == (8, 2)
    assert topology.dtype == torch.int32
    assert torch.all((topology >= 0) & (topology < 8))


def test_lplb_without_cuda_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(LPLBUnavailableError, match="requires CUDA"):
        LPLBBaseline(num_experts=64, ep_size=8, n_slot=10)

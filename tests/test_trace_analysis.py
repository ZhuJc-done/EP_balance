"""Tests for offline expert-hotspot trace aggregation."""

import torch

from eplb.trace_analysis import (
    expert_count_cube,
    expert_max_mean_by_layer,
    load_routing_trace,
    metric_rows,
    normalize_expert_counts,
    rank_loads_from_expert_counts,
    select_expert_count_matrix,
)


def _trace():
    return {
        "meta": {
            "format_version": 3,
            "num_ranks": 2,
            "num_experts": 4,
            "main_rank": torch.tensor([0, 0, 1, 1]),
            "counts_reduced_over_tp_cp": True,
        },
        "samples": [
            {
                "layer": 0,
                "mb": 0,
                "ordinal": 0,
                "omega": torch.tensor([[5, 1, 0, 0], [0, 0, 2, 2]]),
            },
            {
                "layer": 1,
                "mb": 0,
                "ordinal": 1,
                "omega": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
            },
            {
                "layer": 0,
                "mb": 1,
                "ordinal": 2,
                "omega": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
            },
            {
                "layer": 1,
                "mb": 1,
                "ordinal": 3,
                "omega": torch.tensor([[0, 0, 4, 0], [0, 0, 4, 0]]),
            },
        ],
    }


def test_load_and_select_trace_views(tmp_path):
    path = tmp_path / "trace.pt"
    torch.save(_trace(), path)
    trace = load_routing_trace(path)

    layers, snapshot = select_expert_count_matrix(trace, view="snapshot", occurrence=0)
    assert layers == [0, 1]
    assert snapshot.tolist() == [[5, 1, 2, 2], [2, 2, 2, 2]]

    _, aggregate = select_expert_count_matrix(trace, view="aggregate")
    assert aggregate.tolist() == [[7, 3, 4, 4], [2, 2, 10, 2]]

    cube_layers, cube = expert_count_cube(trace)
    assert cube_layers == layers
    assert cube.shape == (2, 2, 4)
    assert cube[0].tolist() == [[5, 1, 2, 2], [2, 2, 2, 2]]

    ratio_layers, batch_ratios, aggregate_ratios = expert_max_mean_by_layer(trace)
    assert ratio_layers == layers
    assert batch_ratios.shape == (2, 2)
    assert torch.allclose(batch_ratios[0], torch.tensor([2.0, 1.0], dtype=torch.float64))
    assert torch.allclose(
        aggregate_ratios,
        torch.tensor([7 / 4.5, 10 / 4.0], dtype=torch.float64),
    )


def test_load_normalizes_legacy_lam_samples(tmp_path):
    legacy_trace = _trace()
    legacy_trace["meta"]["format_version"] = 2
    for sample in legacy_trace["samples"]:
        sample["lam"] = sample.pop("omega")
    path = tmp_path / "legacy_trace.pt"
    torch.save(legacy_trace, path)

    trace = load_routing_trace(path)

    assert "omega" in trace["samples"][0]
    layers, snapshot = select_expert_count_matrix(trace, view="snapshot")
    assert layers == [0, 1]
    assert snapshot.tolist() == [[5, 1, 2, 2], [2, 2, 2, 2]]


def test_normalization_and_original_rank_loads():
    trace = _trace()
    layers, aggregate = select_expert_count_matrix(trace, view="aggregate")
    shares = normalize_expert_counts(aggregate, "share")
    relative = normalize_expert_counts(aggregate, "relative")
    assert torch.allclose(shares.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(relative.mean(dim=1), torch.ones(2, dtype=torch.float64))

    rank_loads = rank_loads_from_expert_counts(aggregate, trace["meta"])
    assert rank_loads.tolist() == [[10, 8], [4, 12]]

    rows = metric_rows(
        label="synthetic",
        view="aggregate",
        layers=layers,
        expert_counts=aggregate,
        meta=trace["meta"],
    )
    assert rows[0]["hot_expert"] == 0
    assert rows[0]["hot_rank"] == 0
    assert rows[1]["hot_expert"] == 2
    assert rows[1]["hot_rank"] == 1

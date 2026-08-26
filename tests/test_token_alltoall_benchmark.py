"""Unit tests for the topology-aware pure token All-to-All baseline."""

from __future__ import annotations

import pytest

from eval.benchmark_token_alltoall import (
    make_destination_splits,
    payload_rows_by_scope,
    summarize_samples,
)


def test_balanced_splits_are_exact_for_model_sized_input():
    splits = make_destination_splits(32_768, 32, 1.0)

    assert splits == [1024] * 32


def test_hotspot_splits_have_requested_receive_imbalance():
    splits = make_destination_splits(32_768, 32, 4.0)

    assert sum(splits) == 32_768
    assert splits[0] == 4096
    assert splits[0] == max(splits)
    assert 32 * max(splits) / sum(splits) == 4.0


def test_non_divisible_balanced_split_keeps_destination_zero_hottest():
    splits = make_destination_splits(10, 3, 1.0)

    assert splits == [4, 3, 3]
    assert sum(splits) == 10


@pytest.mark.parametrize("ratio", [0.5, 9.0])
def test_invalid_imbalance_is_rejected(ratio):
    with pytest.raises(ValueError):
        make_destination_splits(100, 8, ratio)


def test_payload_scope_separates_intra_and_inter_node_rows():
    rows = payload_rows_by_scope(
        send_splits=[10] * 8,
        recv_splits=[10] * 8,
        group_ranks=list(range(8)),
        global_rank=1,
        local_ep_rank=1,
        local_world_size=4,
    )

    assert rows == [70, 70, 30, 40, 40]


def test_summary_uses_per_iteration_critical_rank():
    result = summarize_samples(
        [[1.0, 2.0], [3.0, 4.0]],
        row_bytes=1024,
        remote_send_rows=[1_000_000, 2_000_000],
        remote_recv_rows=[2_000_000, 1_000_000],
        intra_node_send_rows=[500_000, 500_000],
        inter_node_send_rows=[500_000, 1_500_000],
        inter_node_recv_rows=[1_500_000, 500_000],
        link_gbps=400.0,
    )

    assert result["critical_samples_ms"] == [3.0, 4.0]
    assert result["critical_ms_median"] == 3.5
    assert result["mean_rank_ms_median"] == 2.5
    assert result["aggregate_payload_GBps"] == pytest.approx(877.714, abs=1e-3)
    assert result["bottleneck_inter_node_GBps"] == pytest.approx(
        438.857, abs=1e-3
    )
    assert result["nominal_link_utilization"] == pytest.approx(
        8.777143, abs=1e-6
    )

import pytest

from eval import extract_eplb_debug as extract


def _line(rank, body):
    return f"[EPLB-debug r{rank}] {body}\n"


def test_extracts_per_layer_backward_and_true_critical_rank(tmp_path):
    log = tmp_path / "debug.log"
    log.write_text(
        "".join(
            [
                _line(
                    0,
                    "mode=apply layer=0 mb=101 moe_fwd_total=10.0ms "
                    "dispatch=2.0ms/10.00MiB/5.24GB/s",
                ),
                _line(
                    1,
                    "mode=apply layer=0 mb=101 moe_fwd_total=12.0ms "
                    "dispatch=4.0ms/12.00MiB/3.15GB/s",
                ),
                _line(
                    0,
                    "mode=apply layer=1 mb=101 moe_fwd_total=11.0ms "
                    "dispatch=3.0ms/11.00MiB/3.84GB/s",
                ),
                _line(
                    1,
                    "mode=apply layer=1 mb=101 moe_fwd_total=9.0ms "
                    "dispatch=2.0ms/9.00MiB/4.72GB/s",
                ),
                _line(
                    0,
                    "mode=apply direction=backward layer=1 mb=101 "
                    "moe_bwd_total=12.0ms combine_bwd=4.0ms(x2) "
                    "expert_dgrad=1.0ms(x4) dispatch_bwd=2.0ms(x2) "
                    "expert_wgrad=1.5ms(x2)",
                ),
                _line(
                    1,
                    "mode=apply direction=backward layer=1 mb=101 "
                    "moe_bwd_total=13.0ms combine_bwd=3.0ms(x2) "
                    "expert_dgrad=2.0ms(x4) dispatch_bwd=3.0ms(x2) "
                    "expert_wgrad=1.0ms(x2)",
                ),
                _line(
                    0,
                    "mode=apply direction=backward layer=0 mb=101 "
                    "moe_bwd_total=14.0ms combine_bwd=5.0ms(x2) "
                    "expert_dgrad=2.0ms(x4) dispatch_bwd=3.0ms(x2) "
                    "expert_wgrad=2.0ms(x2)",
                ),
                _line(
                    1,
                    "mode=apply direction=backward layer=0 mb=101 "
                    "moe_bwd_total=11.0ms combine_bwd=6.0ms(x2) "
                    "expert_dgrad=1.0ms(x4) dispatch_bwd=2.0ms(x2) "
                    "expert_wgrad=2.5ms(x2)",
                ),
            ]
        )
    )

    forward, backward = extract.parse([log])
    rows = extract.merge(forward, backward)

    combine_rows = [row for row in rows if row["phase"] == "combine_bwd"]
    assert len(combine_rows) == 4
    assert {row["layer"] for row in combine_rows} == {0, 1}
    assert {row["events"] for row in combine_rows} == {2}

    critical = extract.critical_table(rows)
    layer0_dispatch = next(
        row
        for row in critical
        if row["phase"] == "dispatch" and row["layer"] == 0
    )
    assert layer0_dispatch["critical_rank"] == 1
    assert layer0_dispatch["critical_ms"] == 4.0
    assert layer0_dispatch["mean_rank_ms"] == 3.0
    assert layer0_dispatch["ranks_reporting"] == 2
    assert layer0_dispatch["total_mib"] == 22.0

    summary = extract.summarize(
        rows, warmup=100, step_ms=None, expected_ranks=2
    )
    # Layer-critical dispatch is max(2,4) + max(3,2) = 7ms. Taking
    # max after summing per-rank values would incorrectly produce 6ms.
    assert summary["phases"]["dispatch"][
        "critical_rank_ms_per_iteration_median"
    ] == 7.0
    assert summary["phases"]["dispatch"][
        "mean_rank_ms_per_iteration_median"
    ] == 5.5
    assert summary["phases"]["combine_bwd"][
        "critical_rank_ms_per_iteration_median"
    ] == 10.0
    report = tmp_path / "README.md"
    extract.write_readme(report, "synthetic", summary, [str(log)])
    report_text = report.read_text()
    assert "critical ms/iter" in report_text
    assert "`combine_bwd`" in report_text
    assert "`expert_dgrad`" in report_text

    with pytest.raises(SystemExit, match="expected 3"):
        extract.summarize(
            rows, warmup=100, step_ms=None, expected_ranks=3
        )

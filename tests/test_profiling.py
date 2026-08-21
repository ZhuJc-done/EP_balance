from collections import OrderedDict

import torch

from eplb.integration import profiling
from eplb.integration.grouped_mlp import grouped_expert_mlp


def _enable_fresh_debug_window(monkeypatch):
    monkeypatch.setattr(profiling, "_PROFILE_ENABLED", False)
    monkeypatch.setattr(profiling, "_DEBUG_ENABLED", True)
    monkeypatch.setattr(profiling, "_ENABLED", True)
    monkeypatch.setattr(profiling, "_STATS", OrderedDict())
    monkeypatch.setattr(profiling, "_WINDOW_STATS", OrderedDict())
    monkeypatch.setattr(profiling, "_PENDING", [])
    monkeypatch.setattr(profiling, "_calls", 0)
    monkeypatch.setattr(profiling, "_debug_logger", None)
    profiling.begin_debug_window()


def test_debug_timing_prints_one_compact_invocation(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    with profiling.record("solve", time_it=True, device="cpu"):
        pass
    for _ in range(2):
        with profiling.record("apply/dispatch", time_it=True, device="cpu"):
            pass

    lines = []
    profiling.maybe_summary(lines.append, context="layer=3 mb=7")

    assert len(lines) == 1
    assert lines[0].startswith("[EPLB-debug] layer=3 mb=7 ")
    assert "solver=" in lines[0]
    assert "dispatch=" in lines[0]
    assert "ms(x2)" in lines[0]
    assert not profiling._WINDOW_STATS


def test_cross_stream_interval_helpers_report_layer_totals(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    start = profiling.start_debug_interval(device="cpu")
    profiling.finish_debug_interval("apply/moe_fwd_total", start)
    forward = profiling.debug_str("layer=2 mb=5")
    assert "moe_fwd_total=" in forward

    lines = []
    profiling._debug_logger = lines.append
    start = profiling.start_debug_interval(device="cpu")
    profiling.finish_debug_interval("apply/moe_bwd_total", start)
    profiling.emit_backward_debug("layer=2 mb=5")

    assert len(lines) == 1
    assert lines[0].startswith(
        "[EPLB-debug] mode=apply direction=backward layer=2 mb=5 "
    )
    assert "moe_bwd_total=" in lines[0]
    assert not profiling._WINDOW_STATS


def test_native_layer_totals_use_off_mode_labels(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    start = profiling.start_debug_interval(device="cpu")
    profiling.finish_debug_interval("native/moe_fwd_total", start)
    forward = profiling.debug_str("mode=off layer=1 mb=3")
    assert "moe_fwd_total=" in forward

    lines = []
    profiling._debug_logger = lines.append
    start = profiling.start_debug_interval(device="cpu")
    profiling.finish_debug_interval("native/moe_bwd_total", start)
    for phase in ("combine_bwd", "expert_bwd", "dispatch_bwd"):
        with profiling.record(f"native/{phase}", time_it=True, device="cpu"):
            pass
    profiling.emit_backward_debug("layer=1 mb=3", mode="off")

    assert len(lines) == 1
    assert lines[0].startswith(
        "[EPLB-debug] mode=off direction=backward layer=1 mb=3 "
    )
    assert "moe_bwd_total=" in lines[0]
    assert "combine_bwd=" in lines[0]
    assert "expert_bwd=" in lines[0]
    assert "dispatch_bwd=" in lines[0]


def test_apply_backward_breakdown_prints_compute_and_token_phases(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)
    lines = []
    profiling._debug_logger = lines.append

    phases = (
        "combine_bwd",
        "expert_dgrad",
        "activation_bwd",
        "dispatch_bwd",
        "expert_wgrad",
    )
    for phase in phases:
        with profiling.record(
            f"apply/{phase}",
            time_it=True,
            device="cpu",
            payload_bytes=1024 if phase in {"combine_bwd", "dispatch_bwd"} else 0,
        ):
            pass
    profiling.emit_backward_debug("layer=3 mb=9")

    assert len(lines) == 1
    for phase in phases:
        assert f"{phase}=" in lines[0]
    assert lines[0].count("/0.00MiB/") == 2


def test_grouped_mlp_records_pure_expert_gemm(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)
    tokens = torch.randn(2, 4)
    slots = torch.zeros(2, dtype=torch.int64)
    group_sizes = torch.tensor([2], dtype=torch.int64)

    output = grouped_expert_mlp(
        tokens,
        slots,
        group_sizes,
        (torch.empty(0),),
        lambda x, _weights: 2.0 * x,
        cap=2,
    )
    line = profiling.debug_str("layer=0 mb=0")

    assert torch.allclose(output, 2.0 * tokens)
    assert "expert_gemm=" in line


def test_transfer_timing_reports_payload_bandwidth(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    with profiling.record(
        "apply/dispatch",
        time_it=True,
        device="cpu",
        payload_bytes=8 * 1024 * 1024,
    ):
        pass
    line = profiling.debug_str("layer=0 mb=0")

    assert "dispatch=" in line
    assert "/8.00MiB/" in line
    assert "GB/s" in line


def test_wire_timing_reports_get_and_put_without_outer_phase(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    for payload in (6 * 1024 * 1024, 2 * 1024 * 1024):
        with profiling.record(
            "apply/weight_get_wire",
            time_it=True,
            device="cpu",
            payload_bytes=payload,
        ):
            pass
    forward = profiling.debug_str("layer=0 mb=0")

    assert "expert_transfer_wire=" in forward
    assert "ms(x2)/8.00MiB/" in forward

    lines = []
    profiling._debug_logger = lines.append
    with profiling.record(
        "apply/weight_get_wire",
        time_it=True,
        device="cpu",
        payload_bytes=8 * 1024 * 1024,
    ):
        pass
    with profiling.record(
        "apply/grad_put_wire",
        time_it=True,
        device="cpu",
        payload_bytes=8 * 1024 * 1024,
    ):
        pass
    profiling.begin_debug_window()

    assert len(lines) == 1
    assert "expert_repull_wire=" in lines[0]
    assert "expert_grad_put_wire=" in lines[0]

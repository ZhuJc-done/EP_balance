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


def test_debug_timing_prints_deferred_backward_breakdown(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    # Register the logger through the normal forward reporting path.
    with profiling.record("apply/expert_gemm", time_it=True, device="cpu"):
        pass
    lines = []
    profiling.maybe_summary(lines.append, context="mode=apply layer=0 mb=0")

    backward_regions = (
        "apply/backward_total",
        "apply/weight_repull",
        "apply/combine_bwd",
        "apply/expert_bwd_dgrad",
        "apply/expert_bwd_dgrad",
        "apply/activation_bwd",
        "apply/dispatch_bwd",
        "apply/expert_bwd_wgrad",
        "apply/grad_move",
    )
    for region in backward_regions:
        with profiling.record(region, time_it=True, device="cpu"):
            pass

    profiling.begin_debug_window()

    assert len(lines) == 2
    backward = lines[1]
    assert backward.startswith("[EPLB-debug] mode=apply direction=backward ")
    for field in (
        "backward_total=",
        "expert_repull=",
        "combine_bwd=",
        "expert_dgrad=",
        "activation_bwd=",
        "dispatch_bwd=",
        "expert_wgrad=",
        "expert_grad_reduce=",
    ):
        assert field in backward
    assert "expert_dgrad=" in backward and "ms(x2)" in backward


def test_debug_timing_prints_and_clears_each_layer_backward(monkeypatch):
    _enable_fresh_debug_window(monkeypatch)

    lines = []
    with profiling.record("apply/expert_gemm", time_it=True, device="cpu"):
        pass
    profiling.maybe_summary(lines.append, context="mode=apply layer=4 mb=9")

    with profiling.record("apply/backward_total", time_it=True, device="cpu"):
        with profiling.record("apply/expert_bwd_dgrad", time_it=True, device="cpu"):
            pass
    profiling.emit_backward_debug("layer=4 mb=9")

    with profiling.record("apply/backward_total", time_it=True, device="cpu"):
        for _ in range(2):
            with profiling.record("apply/expert_bwd_wgrad", time_it=True, device="cpu"):
                pass
    profiling.emit_backward_debug("layer=3 mb=9")
    profiling.begin_debug_window()

    assert len(lines) == 3
    assert lines[1].startswith(
        "[EPLB-debug] mode=apply direction=backward layer=4 mb=9 "
    )
    assert "backward_total=" in lines[1]
    assert "expert_dgrad=" in lines[1]
    assert "expert_wgrad=" not in lines[1]
    assert lines[2].startswith(
        "[EPLB-debug] mode=apply direction=backward layer=3 mb=9 "
    )
    assert "backward_total=" in lines[2]
    assert "expert_wgrad=" in lines[2] and "ms(x2)" in lines[2]
    assert "expert_dgrad=" not in lines[2]
    assert not profiling._WINDOW_STATS

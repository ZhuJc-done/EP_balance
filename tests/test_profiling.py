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

"""Optional EPLB instrumentation: name regions for torch profiler and (when EPLB_PROFILE=1) time them."""

from __future__ import annotations

import contextlib
import os
import time
from collections import OrderedDict
from typing import Callable, Optional

import torch

_ENABLED = os.environ.get("EPLB_PROFILE", "0").lower() not in ("0", "", "false", "no")
_PERIOD = int(os.environ.get("EPLB_PROFILE_EVERY", "20"))


def enabled() -> bool:
    """Whether EPLB_PROFILE timing is on (region labels are always emitted)."""
    return _ENABLED


class _Stat:
    __slots__ = ("count", "total_ms", "min_ms", "max_ms", "last_ms")

    def __init__(self) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0
        self.last_ms = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)


_STATS: "OrderedDict[str, _Stat]" = OrderedDict()
_calls = 0


@contextlib.contextmanager
def record(name: str, *, time_it: bool = False, device=None):
    """Annotate ``eplb/<name>`` for torch profiler; if profiling and ``time_it``, also record its latency.

    Args:
        name: Region label (shown as ``eplb/<name>`` in the trace).
        time_it: Accumulate latency stats for this region (only when EPLB_PROFILE=1).
        device: Device hint; CUDA regions are timed with CUDA events (forces a sync).
    """
    with torch.profiler.record_function(f"eplb/{name}"):
        if not (_ENABLED and time_it):
            yield
            return
        use_cuda = torch.cuda.is_available() and (
            device is None or torch.device(device).type == "cuda"
        )
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                torch.cuda.synchronize()
                _STATS.setdefault(name, _Stat()).add(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                _STATS.setdefault(name, _Stat()).add((time.perf_counter() - t0) * 1e3)


def last_ms(name: str) -> float:
    """Latency of the most recent ``name`` region in ms (0 if none/disabled)."""
    s = _STATS.get(name)
    return s.last_ms if s is not None else 0.0


def summary_str() -> str:
    """A one-block table of accumulated per-region latency stats."""
    if not _STATS:
        return "[EPLB-profile] no samples (set EPLB_PROFILE=1)"
    lines = ["[EPLB-profile] region                 count    mean(ms)   min(ms)   max(ms)"]
    for name, s in _STATS.items():
        mean = s.total_ms / max(s.count, 1)
        lines.append(
            f"[EPLB-profile] {name:<22} {s.count:>6}   {mean:>8.3f}  {s.min_ms:>8.3f}  {s.max_ms:>8.3f}"
        )
    return "\n".join(lines)


def maybe_summary(logger: Optional[Callable[[str], None]]) -> None:
    """Every EPLB_PROFILE_EVERY calls, emit the latency summary through ``logger`` (call on rank 0)."""
    global _calls
    if not _ENABLED or logger is None:
        return
    _calls += 1
    if _PERIOD > 0 and _calls % _PERIOD == 0:
        logger(summary_str())

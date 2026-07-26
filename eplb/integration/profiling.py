"""Optional EPLB instrumentation: name regions for torch profiler and (when EPLB_PROFILE=1) time them."""

from __future__ import annotations

import contextlib
import os
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch

_ENABLED = os.environ.get("EPLB_PROFILE", "0").lower() not in ("0", "", "false", "no")
_PERIOD = int(os.environ.get("EPLB_PROFILE_EVERY", "20"))
_ALL_RANKS = os.environ.get("EPLB_PROFILE_ALL_RANKS", "0").lower() not in ("0", "", "false", "no")
# Queued CUDA events are resolved in one batch; cap the queue so long runs stay bounded.
_MAX_PENDING = int(os.environ.get("EPLB_PROFILE_MAX_PENDING", "1024"))
# Summary emission count at which to reset the CUDA peak-memory counters (0 = never). Set this to 1
# so the reported peak reflects steady state rather than warmup//startup allocations.
_RESET_AT = int(os.environ.get("EPLB_PROFILE_RESET_AT", "0"))
_RANK = os.environ.get("RANK", "")
_PREFIX = f"[EPLB-profile r{_RANK}]" if _RANK else "[EPLB-profile]"


def enabled() -> bool:
    """Whether EPLB_PROFILE timing is on (region labels are always emitted)."""
    return _ENABLED


def all_ranks() -> bool:
    """Whether every rank -- not just rank 0 -- should emit its own summary (EPLB_PROFILE_ALL_RANKS=1).

    Per-rank summaries are what make straggler analysis possible: the quantity of interest is
    max-over-ranks vs mean-over-ranks of ``apply/expert_compute``, which a rank-0-only log cannot show.
    """
    return _ALL_RANKS


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
_PENDING: "List[Tuple[str, torch.cuda.Event, torch.cuda.Event]]" = []
_calls = 0


def _drain() -> None:
    """Resolve every queued CUDA-event pair into stats, paying a single device sync for the batch."""
    global _PENDING
    if not _PENDING:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for name, start, end in _PENDING:
            _STATS.setdefault(name, _Stat()).add(start.elapsed_time(end))
    _PENDING = []


@contextlib.contextmanager
def record(name: str, *, time_it: bool = False, device=None):
    """Annotate ``eplb/<name>`` for torch profiler; if profiling and ``time_it``, also record its latency.

    CUDA regions are timed with events that are *queued*, not synchronised on the spot: elapsed time is
    resolved later in one batch (:func:`_drain`, triggered by a summary/``last_ms`` read or a full
    queue). Timing therefore injects no host sync into the steady-state path, so per-region stats and
    an undisturbed end-to-end step time can be collected in the same run.

    Args:
        name: Region label (shown as ``eplb/<name>`` in the trace).
        time_it: Accumulate latency stats for this region (only when EPLB_PROFILE=1).
        device: Device hint; CUDA regions are timed with CUDA events.
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
                _PENDING.append((name, start, end))
                if len(_PENDING) >= _MAX_PENDING:
                    _drain()
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                _STATS.setdefault(name, _Stat()).add((time.perf_counter() - t0) * 1e3)


def last_ms(name: str) -> float:
    """Latency of the most recent ``name`` region in ms (0 if none/disabled)."""
    _drain()
    s = _STATS.get(name)
    return s.last_ms if s is not None else 0.0


def reset_peak_memory() -> None:
    """Reset the CUDA peak-memory counters, so the next reported peak excludes what came before."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def memory_str() -> str:
    """Peak CUDA memory since the last reset, in MiB.

    This is where an ``N_slot`` replica budget shows up: every extra slot holds one expert's weights,
    so sweeping ``EPLB_N_SLOT`` and reading this line gives the memory side of the balance/cost curve.
    """
    if not torch.cuda.is_available():
        return f"{_PREFIX} peak memory: n/a (no CUDA)"
    mib = 1024.0 * 1024.0
    return (
        f"{_PREFIX} peak memory  allocated={torch.cuda.max_memory_allocated() / mib:9.1f} MiB"
        f"   reserved={torch.cuda.max_memory_reserved() / mib:9.1f} MiB"
    )


def summary_str() -> str:
    """A one-block table of accumulated per-region latency stats, plus peak memory."""
    _drain()
    if not _STATS:
        return f"{_PREFIX} no samples (set EPLB_PROFILE=1)"
    lines = [f"{_PREFIX} region                 count    mean(ms)   min(ms)   max(ms)"]
    for name, s in _STATS.items():
        mean = s.total_ms / max(s.count, 1)
        lines.append(
            f"{_PREFIX} {name:<22} {s.count:>6}   {mean:>8.3f}  {s.min_ms:>8.3f}  {s.max_ms:>8.3f}"
        )
    lines.append(memory_str())
    return "\n".join(lines)


def maybe_summary(logger: Optional[Callable[[str], None]]) -> None:
    """Every EPLB_PROFILE_EVERY calls, emit the latency summary through ``logger``.

    Pass a real logger on the ranks that should report: rank 0 normally, or every rank when
    :func:`all_ranks` is set (each line is tagged with the rank, so logs stay separable).
    """
    global _calls
    if not _ENABLED or logger is None:
        return
    _calls += 1
    if _PERIOD > 0 and _calls % _PERIOD == 0:
        logger(summary_str())
    if _RESET_AT and _calls == _RESET_AT:
        reset_peak_memory()

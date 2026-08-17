"""Optional EPLB instrumentation for aggregate profiles and per-invocation debug timing."""

from __future__ import annotations

import atexit
import contextlib
import os
import time
from collections import OrderedDict
from typing import Callable, List, Mapping, Optional, Tuple

import torch


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no")


_PROFILE_ENABLED = _env_flag("EPLB_PROFILE")
_DEBUG_ENABLED = _env_flag("EPLB_DEBUG_TIMING")
_ENABLED = _PROFILE_ENABLED or _DEBUG_ENABLED
_PERIOD = int(os.environ.get("EPLB_PROFILE_EVERY", "20"))
_ALL_RANKS = _env_flag("EPLB_PROFILE_ALL_RANKS")
# Queued CUDA events are resolved in one batch; cap the queue so long runs stay bounded.
_MAX_PENDING = int(os.environ.get("EPLB_PROFILE_MAX_PENDING", "1024"))
# Summary emission count at which to reset the CUDA peak-memory counters (0 = never). Set this to 1
# so the reported peak reflects steady state rather than warmup//startup allocations.
_RESET_AT = int(os.environ.get("EPLB_PROFILE_RESET_AT", "0"))
_RANK = os.environ.get("RANK", "")
_PREFIX = f"[EPLB-profile r{_RANK}]" if _RANK else "[EPLB-profile]"
_DEBUG_PREFIX = f"[EPLB-debug r{_RANK}]" if _RANK else "[EPLB-debug]"
_DEBUG_PHASES = (
    ("solver", ("solve",)),
    ("omega_gather", ("all_gather_omega",)),
    ("router", ("apply/route", "native/route")),
    ("shared_expert", ("apply/shared_expert", "native/shared_expert")),
    ("expert_transfer", ("apply/weight_move",)),
    ("dispatch", ("apply/dispatch", "native/dispatch")),
    ("expert_gemm", ("apply/expert_gemm", "native/expert_gemm")),
    ("combine", ("apply/combine", "native/combine")),
)
_DEBUG_BACKWARD_PHASES = (
    ("backward_total", ("apply/backward_total",)),
    ("expert_repull", ("apply/weight_repull",)),
    ("combine_bwd", ("apply/combine_bwd",)),
    ("expert_dgrad", ("apply/expert_bwd_dgrad",)),
    ("activation_bwd", ("apply/activation_bwd",)),
    ("dispatch_bwd", ("apply/dispatch_bwd",)),
    ("expert_wgrad", ("apply/expert_bwd_wgrad",)),
    ("expert_grad_reduce", ("apply/grad_move",)),
)


def enabled() -> bool:
    """Whether aggregate profiling or per-invocation debug timing is on."""
    return _ENABLED


def regions_requested() -> bool:
    """Whether timing or a PyTorch trace needs EPLB/native region bindings."""
    return _ENABLED or _env_flag("PROFILE_TRACE")


def debug_enabled() -> bool:
    """Whether ``EPLB_DEBUG_TIMING=1`` per-invocation reporting is on."""
    return _DEBUG_ENABLED


def all_ranks() -> bool:
    """Whether every rank -- not just rank 0 -- should emit timing output.

    Per-rank summaries are what make straggler analysis possible: the quantity of interest is
    max-over-ranks vs mean-over-ranks of ``apply/expert_compute``, which a rank-0-only log cannot show.
    """
    return _ALL_RANKS


class _Stat:
    __slots__ = (
        "count",
        "total_ms",
        "min_ms",
        "max_ms",
        "last_ms",
        "total_bytes",
        "last_bytes",
    )

    def __init__(self) -> None:
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0
        self.last_ms = 0.0
        self.total_bytes = 0
        self.last_bytes = 0

    def add(self, ms: float, payload_bytes: int = 0) -> None:
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)
        self.total_bytes += payload_bytes
        self.last_bytes = payload_bytes


_STATS: "OrderedDict[str, _Stat]" = OrderedDict()
_WINDOW_STATS: "OrderedDict[str, _Stat]" = OrderedDict()
_PENDING: "List[Tuple[str, torch.cuda.Event, torch.cuda.Event, object]]" = []
_calls = 0
_debug_logger: Optional[Callable[[str], None]] = None


def _resolve_payload_bytes(value) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("profiling payload_bytes tensor must contain exactly one value")
        value = value.item()
    result = int(value)
    if result < 0:
        raise ValueError("profiling payload_bytes must be non-negative")
    return result


def _add_sample(name: str, ms: float, payload_bytes=0) -> None:
    resolved_bytes = _resolve_payload_bytes(payload_bytes)
    _STATS.setdefault(name, _Stat()).add(ms, resolved_bytes)
    if _DEBUG_ENABLED:
        _WINDOW_STATS.setdefault(name, _Stat()).add(ms, resolved_bytes)


def _drain() -> None:
    """Resolve every queued CUDA-event pair into stats, paying a single device sync for the batch."""
    global _PENDING
    if not _PENDING:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for name, start, end, payload_bytes in _PENDING:
            _add_sample(name, start.elapsed_time(end), payload_bytes)
    _PENDING = []


def start_debug_interval(*, device=None, stream=None):
    """Start a debug-only interval that may span multiple autograd nodes."""
    if not _DEBUG_ENABLED:
        return None
    use_cuda = torch.cuda.is_available() and (
        device is None or torch.device(device).type == "cuda"
    )
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        return "cuda", start
    return "cpu", time.perf_counter()


def finish_debug_interval(name: str, marker, *, stream=None) -> None:
    """Finish an interval returned by :func:`start_debug_interval`."""
    if marker is None:
        return
    kind, start = marker
    if kind == "cuda":
        end = torch.cuda.Event(enable_timing=True)
        end.record(stream)
        _PENDING.append((name, start, end, 0))
        if len(_PENDING) >= _MAX_PENDING:
            _drain()
        return
    _add_sample(name, (time.perf_counter() - start) * 1e3)


def _format_fields(phases, missing: Optional[Mapping[str, str]] = None):
    fields = []
    missing = missing or {}
    for label, names in phases:
        stats = [_WINDOW_STATS[name] for name in names if name in _WINDOW_STATS]
        if not stats:
            if label in missing:
                fields.append(f"{label}={missing[label]}")
            continue
        count_value = sum(stat.count for stat in stats)
        total_ms = sum(stat.total_ms for stat in stats)
        total_bytes = sum(stat.total_bytes for stat in stats)
        count = f"(x{count_value})" if count_value > 1 else ""
        field = f"{label}={total_ms:.3f}ms{count}"
        if total_bytes:
            mib = total_bytes / (1024.0 * 1024.0)
            gbps = total_bytes / (max(total_ms, 1e-12) * 1e6)
            field += f"/{mib:.2f}MiB/{gbps:.2f}GB/s"
        fields.append(field)
    return fields


def _emit_deferred_backward(context: str = "") -> None:
    fields = _format_fields(_DEBUG_BACKWARD_PHASES)
    if fields and _debug_logger is not None:
        where = f" {context}" if context else ""
        _debug_logger(
            f"{_DEBUG_PREFIX} mode=apply direction=backward{where} "
            + " ".join(fields)
        )


def emit_backward_debug(context: str = "") -> None:
    """Synchronize and print one completed apply-layer backward timing window.

    The manual apply path calls this at the end of every layer backward. It intentionally pays a
    full device synchronization when debug timing is enabled, so the reported fields contain only
    that layer's two token chunks instead of accumulating all MoE layers until the next forward.
    """
    if not _DEBUG_ENABLED:
        return
    _drain()
    _emit_deferred_backward(context)
    _WINDOW_STATS.clear()


def begin_debug_window() -> None:
    """Start one forward/observe timing window after reporting deferred backward transfers.

    Debug reporting intentionally synchronizes at invocation boundaries. This keeps a line scoped to
    one MoE invocation, but makes ``EPLB_DEBUG_TIMING`` unsuitable for end-to-end throughput numbers.
    """
    if not _DEBUG_ENABLED:
        return
    emit_backward_debug()


@contextlib.contextmanager
def record(
    name: str,
    *,
    time_it: bool = False,
    device=None,
    stream=None,
    payload_bytes=0,
):
    """Annotate ``eplb/<name>`` for torch profiler; if profiling and ``time_it``, also record its latency.

    CUDA regions are timed with events that are *queued*, not synchronised on the spot: elapsed time is
    resolved later in one batch (:func:`_drain`, triggered by a summary/``last_ms`` read or a full
    queue). ``EPLB_PROFILE`` therefore injects no per-region host sync. Debug timing deliberately
    drains the queue at each MoE boundary to produce one self-contained invocation line.

    Args:
        name: Region label (shown as ``eplb/<name>`` in the trace).
        time_it: Accumulate latency stats for this region when either timing mode is enabled.
        device: Device hint; CUDA regions are timed with CUDA events.
        stream: Optional CUDA stream on which to place the timing events. This is needed for
            hand-scheduled communication and weight-transfer regions that run on side streams.
        payload_bytes: Logical remote payload moved by the region. May be an integer or a
            one-element device tensor; it is resolved only after the queued CUDA events complete.
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
            start.record(stream)
            try:
                yield
            finally:
                end.record(stream)
                _PENDING.append((name, start, end, payload_bytes))
                if len(_PENDING) >= _MAX_PENDING:
                    _drain()
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                _add_sample(name, (time.perf_counter() - t0) * 1e3, payload_bytes)


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
    lines = [
        f"{_PREFIX} region                 count    mean(ms)   min(ms)   max(ms)"
        "  mean(MiB)  payload(GB/s)"
    ]
    for name, s in _STATS.items():
        mean = s.total_ms / max(s.count, 1)
        mean_mib = s.total_bytes / max(s.count, 1) / (1024.0 * 1024.0)
        gbps = s.total_bytes / (max(s.total_ms, 1e-12) * 1e6)
        lines.append(
            f"{_PREFIX} {name:<22} {s.count:>6}   {mean:>8.3f}  {s.min_ms:>8.3f}"
            f"  {s.max_ms:>8.3f}  {mean_mib:>9.2f}  {gbps:>12.2f}"
        )
    lines.append(memory_str())
    return "\n".join(lines)


def debug_str(
    context: str = "", *, missing: Optional[Mapping[str, str]] = None
) -> str:
    """Return and clear one invocation's compact timing breakdown."""
    _drain()
    fields = _format_fields(_DEBUG_PHASES, missing)
    _WINDOW_STATS.clear()
    where = f" {context}" if context else ""
    if not fields:
        return f"{_DEBUG_PREFIX}{where} no timing samples"
    return f"{_DEBUG_PREFIX}{where} " + " ".join(fields)


def maybe_summary(
    logger: Optional[Callable[[str], None]],
    *,
    context: str = "",
    missing: Optional[Mapping[str, str]] = None,
) -> None:
    """Emit per-invocation debug timing and/or the periodic aggregate summary.

    Pass a real logger on the ranks that should report: rank 0 normally, or every rank when
    :func:`all_ranks` is set (each line is tagged with the rank, so logs stay separable).
    """
    global _calls, _debug_logger
    if not _ENABLED:
        return
    if _DEBUG_ENABLED:
        _debug_logger = logger
        line = debug_str(context, missing=missing)
        if logger is not None:
            logger(line)
    if not _PROFILE_ENABLED or logger is None:
        return
    _calls += 1
    if _PERIOD > 0 and _calls % _PERIOD == 0:
        logger(summary_str())
    if _RESET_AT and _calls == _RESET_AT:
        reset_peak_memory()


def _flush_debug_at_exit() -> None:
    if not _DEBUG_ENABLED:
        return
    try:
        _drain()
        _emit_deferred_backward()
        _WINDOW_STATS.clear()
    except Exception:
        # CUDA may already be shutting down; runtime logs from earlier iterations remain valid.
        pass


if _DEBUG_ENABLED:
    atexit.register(_flush_debug_at_exit)

"""Diagnostic capture of raw router assignments and the applied physical plan."""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path

import torch

from ..loads import Loads
from ..plan import Plan
from ..problem import ProblemSpec
from ..topology import Topology


def rank_local_trace_path(path: str | Path, global_rank: int, num_writers: int) -> Path:
    """Resolve one writer's path without allowing PP/EDP leaders to collide."""

    raw = str(path)
    if "{rank}" in raw:
        return Path(raw.replace("{rank}", str(global_rank)))
    resolved = Path(raw)
    if num_writers <= 1:
        return resolved
    return resolved.with_name(
        f"{resolved.stem}.rank{global_rank}{resolved.suffix}"
    )


class PlanTraceWriter:
    """Persist ``omega``, placement ``x`` and physical quota ``q`` on one rank.

    Capturing a sample copies the complete plan to CPU and synchronizes the
    device.  This is deliberately an opt-in diagnostic and must not be enabled
    in throughput measurements.
    """

    def __init__(
        self,
        path: str | Path,
        topology: Topology,
        spec: ProblemSpec,
        *,
        solver: str,
        global_rank: int,
        pipeline_rank: int,
        max_samples: int = 0,
        flush_every: int = 25,
    ) -> None:
        self.path = Path(path)
        self.max_samples = int(max_samples)
        self.flush_every = max(1, int(flush_every))
        if self.max_samples < 0:
            raise ValueError("trace max_samples must be non-negative")
        self.samples: list[dict] = []
        self._dirty = 0
        self._ordinal = 0
        self.meta = {
            "format_version": 4,
            "trace_kind": "applied_plan",
            "plan_solver": str(solver),
            "writer_global_rank": int(global_rank),
            "pipeline_rank": int(pipeline_rank),
            "num_ranks": topology.num_ranks,
            "num_experts": int(spec.num_experts),
            "omega_semantics": (
                "source_ep_rank_by_logical_expert_token_assignments"
            ),
            "q_semantics": (
                "source_ep_rank_by_logical_expert_by_physical_destination_quota"
            ),
            "s_tok": int(spec.s_tok),
            "n_slot": int(spec.n_slot),
            "num_domains": topology.num_domains,
            "main_rank": spec.main_rank.detach().cpu().clone(),
            "weight_bytes": spec.weight_bytes.detach().cpu().clone(),
            "domain_of_rank": topology.domain_of_rank.detach().cpu().clone(),
            "cost": topology.cost.detach().cpu().clone(),
        }
        atexit.register(self.flush)

    def append(
        self,
        loads: Loads,
        plan: Plan,
        layer_id: int,
        micro_batch_id: int,
    ) -> None:
        if self.max_samples and len(self.samples) >= self.max_samples:
            return
        theta = torch.as_tensor(plan.theta).detach().to("cpu", torch.int64)
        self.samples.append(
            {
                "layer": int(layer_id),
                "mb": int(micro_batch_id),
                "ordinal": self._ordinal,
                "omega": loads.omega.detach().to("cpu", torch.int64).clone(),
                "x": plan.x.detach().to("cpu", torch.int8).clone(),
                "q": plan.q.detach().to("cpu", torch.int64).clone(),
                "theta": int(theta),
            }
        )
        self._ordinal += 1
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.samples or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, suffix=".tmp"
        )
        os.close(fd)
        try:
            torch.save({"meta": self.meta, "samples": self.samples}, temporary)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._dirty = 0


__all__ = ["PlanTraceWriter", "rank_local_trace_path"]

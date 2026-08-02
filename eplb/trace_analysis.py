"""Offline analysis helpers for real Megatron MoE routing traces."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch


def load_routing_trace(path: str | Path) -> dict:
    """Load and validate an ``EPLB_TRACE_OUT`` file."""
    trace = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(trace, dict) or not isinstance(trace.get("meta"), dict):
        raise ValueError(f"{path}: expected a trace dict with a 'meta' mapping")
    if not isinstance(trace.get("samples"), list) or not trace["samples"]:
        raise ValueError(f"{path}: trace has no samples")

    meta = trace["meta"]
    num_ranks = int(meta["num_ranks"])
    num_experts = int(meta["num_experts"])
    for index, sample in enumerate(trace["samples"]):
        if (
            not isinstance(sample, dict)
            or "layer" not in sample
            or ("omega" not in sample and "lam" not in sample)
        ):
            raise ValueError(f"{path}: malformed sample {index}")
        # Trace v2 originally named the assignment matrix ``lam``. Normalize
        # legacy files in memory so current analysis consistently reads omega.
        if "omega" not in sample:
            sample["omega"] = sample["lam"]
        omega = torch.as_tensor(sample["omega"])
        if omega.shape != (num_ranks, num_experts):
            raise ValueError(
                f"{path}: sample {index} Ω shape {tuple(omega.shape)} != "
                f"({num_ranks}, {num_experts})"
            )
        if torch.any(omega < 0):
            raise ValueError(f"{path}: sample {index} contains negative token counts")
    return trace


def expert_counts_by_layer(trace: dict) -> dict[int, list[torch.Tensor]]:
    """Group chronological logical-expert counts by MoE layer."""
    grouped: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for fallback_ordinal, sample in enumerate(trace["samples"]):
        layer = int(sample["layer"])
        ordinal = int(sample.get("ordinal", fallback_ordinal))
        counts = torch.as_tensor(sample["omega"], dtype=torch.int64).sum(dim=0)
        grouped[layer].append((ordinal, counts))
    return {
        layer: [counts for _, counts in sorted(records, key=lambda item: item[0])]
        for layer, records in grouped.items()
    }


def expert_count_cube(
    trace: dict,
    *,
    max_occurrences: int = 0,
) -> tuple[list[int], torch.Tensor]:
    """Return complete routing counts as ``[layer, occurrence, expert]``."""
    if max_occurrences < 0:
        raise ValueError("max_occurrences must be non-negative")
    grouped = expert_counts_by_layer(trace)
    layers = sorted(grouped)
    occurrences = min(len(grouped[layer]) for layer in layers)
    if max_occurrences:
        occurrences = min(occurrences, max_occurrences)
    if occurrences <= 0:
        raise ValueError("trace has no complete layer occurrences")
    counts = torch.stack(
        [
            torch.stack(grouped[layer][:occurrences], dim=0)
            for layer in layers
        ],
        dim=0,
    )
    return layers, counts


def expert_max_mean_by_layer(
    trace: dict,
    *,
    max_occurrences: int = 0,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Return per-occurrence and aggregate expert max/mean ratios by layer."""
    layers, counts = expert_count_cube(trace, max_occurrences=max_occurrences)
    counts = counts.to(torch.float64)

    batch_means = counts.mean(dim=-1)
    batch_ratios = torch.where(
        batch_means > 0,
        counts.max(dim=-1).values / batch_means,
        torch.zeros_like(batch_means),
    )

    aggregate_counts = counts.sum(dim=1)
    aggregate_means = aggregate_counts.mean(dim=-1)
    aggregate_ratios = torch.where(
        aggregate_means > 0,
        aggregate_counts.max(dim=-1).values / aggregate_means,
        torch.zeros_like(aggregate_means),
    )
    return layers, batch_ratios, aggregate_ratios


def select_expert_count_matrix(
    trace: dict,
    *,
    view: str,
    occurrence: int = 0,
    max_occurrences: int = 0,
) -> tuple[list[int], torch.Tensor]:
    """Return raw ``[layer, expert]`` assignment counts for one requested view."""
    if view not in {"snapshot", "aggregate"}:
        raise ValueError("view must be 'snapshot' or 'aggregate'")
    if max_occurrences < 0:
        raise ValueError("max_occurrences must be non-negative")
    grouped = expert_counts_by_layer(trace)
    layers = sorted(grouped)
    aggregate_occurrences = min(len(grouped[layer]) for layer in layers)
    if max_occurrences:
        aggregate_occurrences = min(aggregate_occurrences, max_occurrences)
    rows = []
    for layer in layers:
        values = grouped[layer]
        if view == "snapshot":
            try:
                counts = values[occurrence]
            except IndexError as exc:
                raise ValueError(
                    f"layer {layer} has {len(values)} occurrences, cannot select {occurrence}"
                ) from exc
        else:
            # A trace can end between layer hooks. Use only the common complete
            # occurrence window so every heatmap row covers identical requests.
            selected = values[:aggregate_occurrences]
            if not selected:
                raise ValueError(f"layer {layer} has no selected occurrences")
            counts = torch.stack(selected, dim=0).sum(dim=0)
        rows.append(counts)
    return layers, torch.stack(rows, dim=0)


def normalize_expert_counts(counts: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize each layer by share, by its mean, or not at all."""
    values = counts.to(torch.float64)
    if mode == "share":
        return values / values.sum(dim=1, keepdim=True).clamp_min(1)
    if mode == "relative":
        return values / values.mean(dim=1, keepdim=True).clamp_min(1)
    if mode == "count":
        return values
    raise ValueError("normalization mode must be share, relative, or count")


def resolve_main_rank(meta: dict) -> torch.Tensor:
    """Read the logical-expert home placement, with a contiguous fallback."""
    num_ranks = int(meta["num_ranks"])
    num_experts = int(meta["num_experts"])
    if "main_rank" in meta:
        main_rank = torch.as_tensor(meta["main_rank"], dtype=torch.int64)
    else:
        if num_experts % num_ranks:
            raise ValueError("trace lacks main_rank and experts are not evenly divisible by ranks")
        main_rank = torch.arange(num_experts, dtype=torch.int64) // (num_experts // num_ranks)
    if main_rank.shape != (num_experts,):
        raise ValueError(f"main_rank shape {tuple(main_rank.shape)} != ({num_experts},)")
    if torch.any(main_rank < 0) or torch.any(main_rank >= num_ranks):
        raise ValueError("main_rank contains a rank outside the trace topology")
    return main_rank


def rank_loads_from_expert_counts(counts: torch.Tensor, meta: dict) -> torch.Tensor:
    """Map logical-expert counts to no-balancing receiving-rank loads."""
    main_rank = resolve_main_rank(meta)
    num_ranks = int(meta["num_ranks"])
    rank_loads = torch.zeros(
        (counts.shape[0], num_ranks),
        dtype=counts.dtype,
        device=counts.device,
    )
    for layer_index in range(counts.shape[0]):
        rank_loads[layer_index].index_add_(0, main_rank.to(counts.device), counts[layer_index])
    return rank_loads


def _max_mean(values: torch.Tensor) -> float:
    mean = values.to(torch.float64).mean()
    return float(values.max().to(torch.float64) / mean) if mean > 0 else 0.0


def metric_rows(
    *,
    label: str,
    view: str,
    layers: Iterable[int],
    expert_counts: torch.Tensor,
    meta: dict,
) -> list[dict]:
    """Compute raw expert and original-placement rank imbalance per layer."""
    rank_loads = rank_loads_from_expert_counts(expert_counts, meta)
    rows = []
    for row_index, layer in enumerate(layers):
        experts = expert_counts[row_index]
        ranks = rank_loads[row_index]
        total = int(experts.sum())
        hot_expert = int(experts.argmax()) if total else -1
        hot_rank = int(ranks.argmax()) if total else -1
        rows.append(
            {
                "label": label,
                "view": view,
                "layer": int(layer),
                "total_assignments": total,
                "expert_max_mean": _max_mean(experts),
                "rank_max_mean": _max_mean(ranks),
                "hot_expert": hot_expert,
                "hot_expert_share": (
                    float(experts[hot_expert].to(torch.float64) / total) if total else 0.0
                ),
                "hot_rank": hot_rank,
            }
        )
    return rows

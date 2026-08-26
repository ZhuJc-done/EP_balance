#!/usr/bin/env python3
"""Extract and plot PP-aware critical-rank MoE latency breakdowns."""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

from eval.extract_eplb_debug import merge, parse


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["hatch.linewidth"] = 0.7
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    key: str
    model: str
    model_title: str
    method: str
    log_stem: str
    ep_size: int


RUNS = (
    RunSpec(
        key="qwen_native",
        model="qwen",
        model_title="Qwen3-30B-A3B",
        method="baseline",
        log_stem="debug_qwen3_5L_ep32_native_skew_m40",
        ep_size=32,
    ),
    RunSpec(
        key="qwen_scale",
        model="qwen",
        model_title="Qwen3-30B-A3B",
        method="scale_eplb",
        log_stem="debug_qwen3_5L_ep32_scale_ns6_skew_m40",
        ep_size=32,
    ),
    RunSpec(
        key="glm_native",
        model="glm",
        model_title="GLM-4.5-Air",
        method="baseline",
        log_stem="debug_glm45_air_5moe_ep32_native_skew_m40",
        ep_size=16,
    ),
    RunSpec(
        key="glm_scale",
        model="glm",
        model_title="GLM-4.5-Air",
        method="scale_eplb",
        log_stem="debug_glm45_air_5moe_ep32_scale_ns8_skew_m40",
        ep_size=16,
    ),
)

METHOD_ORDER = ("baseline", "scale_eplb")
METHOD_LABEL = {
    "baseline": "Megatron-LM\nBaseline",
    "scale_eplb": "Scale-EPLB",
}
DIRECTION_ORDER = ("forward", "backward")
DIRECTION_LABEL = {"forward": "Forward", "backward": "Backward"}

# Replica movement uses nested wire-only timers so buffer materialization,
# staging and fences in the parent operation are not charged to this category.
# Token transport likewise uses events placed directly around NCCL or the fused
# DeepEP transport primitive, excluding permutation, allocation and count exchange.
WIRE_CATEGORY_PHASES = {
    "forward": {
        "expert_compute": ("expert_gemm",),
        "token_all_to_all": ("dispatch_wire", "combine_wire"),
        "replica_management": ("solver", "expert_transfer_wire"),
        "other": ("router", "shared_expert"),
    },
    "backward": {
        "expert_compute": (
            "expert_bwd",
            "expert_dgrad",
            "activation_bwd",
            "expert_wgrad",
        ),
        "token_all_to_all": ("combine_bwd_wire", "dispatch_bwd_wire"),
        "replica_management": ("expert_repull_wire", "expert_grad_put_wire"),
        "other": (),
    },
}
DISPATCHER_CATEGORY_PHASES = {
    direction: {
        **categories,
        "token_all_to_all": (
            ("dispatch", "combine")
            if direction == "forward"
            else ("combine_bwd", "dispatch_bwd")
        ),
    }
    for direction, categories in WIRE_CATEGORY_PHASES.items()
}
TOTAL_PHASE = {"forward": "moe_fwd_total", "backward": "moe_bwd_total"}
CATEGORY_STYLE = {
    "expert_compute": {
        "label": "Expert Compute",
        "color": "#8DA0CB",
        "text_color": "#111111",
        "hatch": "///",
    },
    "token_all_to_all": {
        "label": "Token Transport Kernel",
        "color": "#66C2A5",
        "text_color": "#111111",
        "hatch": "\\\\",
    },
    "replica_management": {
        "label": "Replica Management",
        "color": "#FC8D62",
        "text_color": "#111111",
        "hatch": "xx",
    },
    "other": {
        "label": "Other",
        "color": "#E5E5E5",
        "text_color": "#111111",
        "hatch": "..",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--rank-policy",
        choices=("critical", "best-communication"),
        default="critical",
        help=(
            "critical takes max rank for every phase; best-communication keeps "
            "expert compute at max rank but takes min rank for token transport "
            "and replica management, and hides Other/MoE Total"
        ),
    )
    parser.add_argument(
        "--token-scope",
        choices=("wire", "dispatcher"),
        default="wire",
        help=(
            "wire requires nested transport timers; dispatcher explicitly uses "
            "legacy method-wide dispatch/combine intervals"
        ),
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def source_paths(logs_dir: Path, spec: RunSpec) -> list[Path]:
    paths = [logs_dir / f"{spec.log_stem}_node{node}.log" for node in range(4)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{spec.key}: missing logs: {missing}")
    return paths


def load_critical_phase_series(
    logs_dir: Path,
    spec: RunSpec,
    *,
    warmup: int,
    min_rank_phases: frozenset[str] = frozenset(),
) -> tuple[dict[tuple[str, str], list[tuple[int, float]]], dict[str, Any]]:
    rows = [
        row
        for row in merge(*parse(source_paths(logs_dir, spec)))
        if row["iteration"] > warmup
    ]
    if not rows:
        raise ValueError(f"{spec.key}: no timing records after warm-up {warmup}")

    grouped: dict[tuple[int, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pipeline_stage = int(row["rank"]) // spec.ep_size
        key = (
            int(row["iteration"]),
            pipeline_stage,
            int(row["layer"]),
            str(row["direction"]),
            str(row["phase"]),
        )
        grouped[key].append(row)

    signatures = {
        (stage, layer, direction, phase)
        for (_iteration, stage, layer, direction, phase) in grouped
    }
    iterations = sorted({iteration for (iteration, *_rest) in grouped})
    bad_iterations = []
    for iteration in iterations:
        iteration_keys = {
            (stage, layer, direction, phase)
            for (current, stage, layer, direction, phase) in grouped
            if current == iteration
        }
        complete = iteration_keys == signatures
        if complete:
            complete = all(
                len(
                    {
                        int(item["rank"])
                        for item in grouped[(iteration, stage, layer, direction, phase)]
                    }
                )
                == spec.ep_size
                for stage, layer, direction, phase in signatures
            )
        if not complete:
            bad_iterations.append(iteration)

    good_iterations = [iteration for iteration in iterations if iteration not in bad_iterations]
    if not good_iterations:
        raise ValueError(f"{spec.key}: no rank-complete iterations")

    layer_keys = sorted(
        {
            (stage, layer)
            for (_iteration, stage, layer, _direction, _phase) in grouped
        }
    )
    rank_reduced: dict[tuple[int, int, int, str, str], float] = {}
    for key, items in grouped.items():
        if key[0] in bad_iterations:
            continue
        phase = key[-1]
        reduce_rank = min if phase in min_rank_phases else max
        rank_reduced[key] = reduce_rank(float(item["ms"]) for item in items)

    by_phase_iteration: dict[tuple[str, str, int], float] = defaultdict(float)
    for (
        iteration,
        _stage,
        _layer,
        direction,
        phase,
    ), value in rank_reduced.items():
        by_phase_iteration[(direction, phase, iteration)] += value / len(layer_keys)

    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (direction, phase, iteration), value in sorted(by_phase_iteration.items()):
        series[(direction, phase)].append((iteration, value))

    metadata = {
        "first_iteration": good_iterations[0],
        "last_iteration": good_iterations[-1],
        "iterations": len(good_iterations),
        "dropped_iterations": bad_iterations,
        "layers": len(layer_keys),
        "layer_keys": layer_keys,
        "ep_size": spec.ep_size,
        "pipeline_stages": len({stage for stage, _layer in layer_keys}),
        "sources": [str(path) for path in source_paths(logs_dir, spec)],
        "min_rank_phases": sorted(min_rank_phases),
    }
    return dict(series), metadata


def phase_values(
    series: dict[tuple[str, str], list[tuple[int, float]]],
    direction: str,
    phase: str,
) -> dict[int, float]:
    return dict(series.get((direction, phase), ()))


def aggregate_run(
    logs_dir: Path,
    spec: RunSpec,
    *,
    warmup: int,
    category_phases: dict[str, dict[str, tuple[str, ...]]],
    min_rank_phases: frozenset[str] = frozenset(),
    include_other: bool = True,
    include_total: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase_series, metadata = load_critical_phase_series(
        logs_dir,
        spec,
        warmup=warmup,
        min_rank_phases=min_rank_phases,
    )
    summary_rows = []
    per_iteration_rows = []

    for direction in DIRECTION_ORDER:
        total_by_iteration = phase_values(
            phase_series, direction, TOTAL_PHASE[direction]
        )
        if not total_by_iteration:
            raise ValueError(f"{spec.key}: missing {TOTAL_PHASE[direction]}")
        iterations = sorted(total_by_iteration)

        category_by_iteration: dict[str, dict[int, float]] = {}
        active_categories = [
            category
            for category in CATEGORY_STYLE
            if include_other or category != "other"
        ]
        for category in active_categories:
            phases = category_phases[direction][category]
            if category == "token_all_to_all":
                missing = [
                    phase
                    for phase in phases
                    if not phase_values(phase_series, direction, phase)
                ]
                if missing:
                    raise ValueError(
                        f"{spec.key}: logs lack selected token phases {missing}; "
                        "choose --token-scope dispatcher only for explicitly "
                        "labelled legacy diagnostics, or rerun debug timing"
                    )
            values = {iteration: 0.0 for iteration in iterations}
            for phase in phases:
                for iteration, value in phase_values(
                    phase_series, direction, phase
                ).items():
                    if iteration in values:
                        values[iteration] += value
            category_by_iteration[category] = values

        for category in active_categories:
            values = [
                category_by_iteration[category][iteration]
                for iteration in iterations
            ]
            summary_rows.append(
                _summary_row(
                    spec,
                    direction=direction,
                    category=category,
                    kind="stream_occupancy",
                    phases=category_phases[direction][category],
                    values=values,
                    metadata=metadata,
                    warmup=warmup,
                    rank_reduction=(
                        "min"
                        if category_phases[direction][category]
                        and all(
                            phase in min_rank_phases
                            for phase in category_phases[direction][category]
                        )
                        else "max"
                    ),
                )
            )
            per_iteration_rows.extend(
                {
                    "model": spec.model,
                    "method": spec.method,
                    "direction": direction,
                    "iteration": iteration,
                    "category": category,
                    "kind": "stream_occupancy",
                    "rank_reduction": (
                        "min"
                        if category_phases[direction][category]
                        and all(
                            phase in min_rank_phases
                            for phase in category_phases[direction][category]
                        )
                        else "max"
                    ),
                    "ms_per_layer": round(category_by_iteration[category][iteration], 6),
                }
                for iteration in iterations
            )

        if include_total:
            total_values = [total_by_iteration[iteration] for iteration in iterations]
            summary_rows.append(
                _summary_row(
                    spec,
                    direction=direction,
                    category="moe_total",
                    kind="wall_time",
                    phases=(TOTAL_PHASE[direction],),
                    values=total_values,
                    metadata=metadata,
                    warmup=warmup,
                    rank_reduction=(
                        "min"
                        if TOTAL_PHASE[direction] in min_rank_phases
                        else "max"
                    ),
                )
            )
            per_iteration_rows.extend(
                {
                    "model": spec.model,
                    "method": spec.method,
                    "direction": direction,
                    "iteration": iteration,
                    "category": "moe_total",
                    "kind": "wall_time",
                    "rank_reduction": (
                        "min"
                        if TOTAL_PHASE[direction] in min_rank_phases
                        else "max"
                    ),
                    "ms_per_layer": round(total_by_iteration[iteration], 6),
                }
                for iteration in iterations
            )
    return summary_rows, per_iteration_rows, metadata


def _summary_row(
    spec: RunSpec,
    *,
    direction: str,
    category: str,
    kind: str,
    phases: tuple[str, ...],
    values: list[float],
    metadata: dict[str, Any],
    warmup: int,
    rank_reduction: str,
) -> dict[str, Any]:
    return {
        "model": spec.model,
        "model_title": spec.model_title,
        "method": spec.method,
        "direction": direction,
        "category": category,
        "kind": kind,
        "rank_reduction": rank_reduction,
        "phases": "+".join(phases),
        "median_ms_per_layer": round(st.median(values), 6),
        "p25_ms_per_layer": round(percentile(values, 0.25), 6),
        "p75_ms_per_layer": round(percentile(values, 0.75), 6),
        "samples": len(values),
        "warmup_excluded": warmup,
        "first_iteration": metadata["first_iteration"],
        "last_iteration": metadata["last_iteration"],
        "moe_layers": metadata["layers"],
        "ep_size": metadata["ep_size"],
        "pipeline_stages": metadata["pipeline_stages"],
        "dropped_iterations": ";".join(
            str(value) for value in metadata["dropped_iterations"]
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def visible_categories(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row["category"]) for row in rows}
    return [category for category in CATEGORY_STYLE if category in present]


def has_moe_total(rows: list[dict[str, Any]]) -> bool:
    return any(row["category"] == "moe_total" for row in rows)


def rank_axis_label(rows: list[dict[str, Any]]) -> str:
    reductions = {
        str(row["rank_reduction"])
        for row in rows
        if row["category"] != "moe_total"
    }
    if reductions == {"min"}:
        return "Best-rank time per MoE layer (ms)"
    if reductions == {"max"}:
        return "Critical-rank time per MoE layer (ms)"
    return "Mixed-rank time per MoE layer (ms)"


def legend_handles(rows: list[dict[str, Any]]) -> list[Any]:
    handles: list[Any] = [
        Patch(
            facecolor=CATEGORY_STYLE[category]["color"],
            edgecolor="white",
            hatch=CATEGORY_STYLE[category]["hatch"],
            label=CATEGORY_STYLE[category]["label"],
        )
        for category in visible_categories(rows)
    ]
    if has_moe_total(rows):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#202020",
                markeredgecolor="white",
                label="Measured MoE Total",
                markersize=5.5,
            )
        )
    return handles


def draw_panel(
    axis: Any,
    model_rows: list[dict[str, Any]],
    direction: str,
    *,
    panel_label: str,
    show_ylabel: bool = True,
) -> None:
    x_positions = list(range(len(METHOD_ORDER)))
    bottoms = [0.0] * len(METHOD_ORDER)
    total_values = []

    for category in visible_categories(model_rows):
        style = CATEGORY_STYLE[category]
        values = [
            _lookup(model_rows, method, direction, category)
            for method in METHOD_ORDER
        ]
        bars = axis.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
            hatch=style["hatch"],
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 0.7:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=style["text_color"],
                    fontweight="medium",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    if has_moe_total(model_rows):
        for method_index, method in enumerate(METHOD_ORDER):
            total = _lookup(model_rows, method, direction, "moe_total")
            total_values.append(total)
            total_x = method_index + 0.20
            axis.plot(
                total_x,
                total,
                marker="D",
                markersize=5.0,
                color="#202020",
                markeredgecolor="white",
                markeredgewidth=0.5,
                linestyle="none",
                zorder=5,
            )
            axis.annotate(
                f"{total:.2f}",
                (total_x, total),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7.8,
                color="#202020",
            )

    axis.set_xticks(x_positions, [METHOD_LABEL[item] for item in METHOD_ORDER])
    axis.set_ylabel(
        rank_axis_label(model_rows) if show_ylabel else "",
        fontsize=9.5,
    )
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=8.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(0, 1.16 * max([*bottoms, *total_values]))
    axis.text(
        0.5,
        -0.28,
        panel_label,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )


def plot_model(
    model: str,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    dpi: int,
) -> Path:
    model_rows = [row for row in rows if row["model"] == model]
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.65))
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.22,
        top=0.80,
        wspace=0.20,
    )

    for panel_index, direction in enumerate(DIRECTION_ORDER):
        draw_panel(
            axes[panel_index],
            model_rows,
            direction,
            panel_label=(
                f"({chr(ord('a') + panel_index)}) {DIRECTION_LABEL[direction]}"
            ),
        )

    figure.legend(
        handles=legend_handles(model_rows),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(legend_handles(model_rows)),
        frameon=False,
        fontsize=7.1,
        columnspacing=0.9,
        handlelength=1.6,
    )

    output_stem = output_dir / f"{model}_latency_breakdown"
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_stem


def plot_combined(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 6.15))
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.88,
        wspace=0.20,
        hspace=0.66,
    )
    panel_index = 0
    for model_index, model in enumerate(("qwen", "glm")):
        model_rows = [row for row in rows if row["model"] == model]
        model_title = str(model_rows[0]["model_title"])
        for direction_index, direction in enumerate(DIRECTION_ORDER):
            draw_panel(
                axes[model_index][direction_index],
                model_rows,
                direction,
                panel_label=(
                    f"({chr(ord('a') + panel_index)}) {model_title} "
                    f"{DIRECTION_LABEL[direction]}"
                ),
                show_ylabel=direction_index == 0,
            )
            panel_index += 1

    figure.legend(
        handles=legend_handles(rows),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(legend_handles(rows)),
        frameon=False,
        fontsize=7.1,
        columnspacing=0.9,
        handlelength=1.6,
    )

    output_stem = output_dir / "latency_breakdown"
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_stem


def _lookup(
    rows: list[dict[str, Any]],
    method: str,
    direction: str,
    category: str,
) -> float:
    matches = [
        float(row["median_ms_per_layer"])
        for row in rows
        if row["method"] == method
        and row["direction"] == direction
        and row["category"] == category
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one value for {method}/{direction}/{category}, found {len(matches)}"
        )
    return matches[0]


def write_readme(
    path: Path,
    *,
    warmup: int,
    metadata: dict[str, dict[str, Any]],
    rank_policy: str,
    token_scope: str,
) -> None:
    best_communication = rank_policy == "best-communication"
    lines = [
        "# End-to-end MoE latency breakdown",
        "",
        "The plots use the median rank-reduced phase time per MoE layer after "
        f"excluding the first {warmup} iterations.",
        "",
        "For PP=2 GLM runs, local layer IDs are disambiguated by pipeline stage "
        "(`global_rank // EP_size`), yielding five distinct MoE layers. Each phase "
        + (
            "takes the maximum rank for Expert Compute and the minimum rank for "
            "Token Transport and Replica Management, then averages over MoE layers "
            "inside an iteration. These independently selected ranks form a best-case "
            "communication envelope, not one rank's additive execution path."
            if best_communication
            else "first takes the maximum over the corresponding EP group, then "
            "averages over MoE layers inside an iteration."
        ),
        "",
        "Colored stacks are CUDA-stream occupancy diagnostics. Their height is not "
        "wall-clock latency because communication and compute overlap."
        + (
            " Other and the measured MoE total are intentionally omitted."
            if best_communication
            else " Black diamonds show the directly measured MoE forward/backward total."
        ),
        "",
        "Replica movement uses its nested wire-only timers; parent operation timers "
        "that also include buffer materialization, staging and fences are excluded.",
        "",
        "Forward replica management includes the CUDA-event solver time and "
        "expert-parameter get kernels; `omega_gather` is deliberately excluded. "
        "Backward replica management includes only expert-weight get and replica-gradient "
        "put kernels.",
        "",
        (
            "Token Transport Kernel uses nested CUDA events placed directly around "
            "the communication primitive. Native Megatron measures each "
            "`all_to_all_single` call (two in dispatch: hidden states and "
            "probabilities; one in combine), excluding allocation, permutation, "
            "count exchange and barriers. DeepEP is one fused transport kernel, so "
            "NIC-only time cannot be separated externally."
            if token_scope == "wire"
            else "Token Transport uses legacy dispatcher-wide dispatch/combine "
            "intervals because the source logs predate nested wire timers. These "
            "values include dispatcher overhead and are diagnostic only; they must "
            "not be described as pure All-to-All communication."
        ),
        "",
        "## Run coverage",
        "",
    ]
    for spec in RUNS:
        item = metadata[spec.key]
        dropped = item["dropped_iterations"] or "none"
        lines.append(
            f"- `{spec.key}`: iterations {item['first_iteration']}–"
            f"{item['last_iteration']} ({item['iterations']} complete), "
            f"{item['layers']} MoE layers, EP={item['ep_size']}, "
            f"dropped incomplete iterations: {dropped}."
        )
    lines += [
        "",
        "## Files",
        "",
        "- `latency_breakdown.csv`: plotted medians and interquartile ranges.",
        "- `latency_breakdown_per_iteration.csv`: underlying per-iteration values.",
        "- `qwen_latency_breakdown.{png,pdf}` and "
        "`glm_latency_breakdown.{png,pdf}`: per-model publication figures.",
        "- `latency_breakdown.{png,pdf}`: combined four-panel publication figure.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    logs_dir = args.logs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    category_phases = (
        WIRE_CATEGORY_PHASES
        if args.token_scope == "wire"
        else DISPATCHER_CATEGORY_PHASES
    )
    CATEGORY_STYLE["token_all_to_all"]["label"] = (
        "Token Transport Kernel"
        if args.token_scope == "wire"
        else "Token Dispatch/Combine"
    )
    best_communication = args.rank_policy == "best-communication"
    min_rank_phases = (
        frozenset(
            phase
            for direction in DIRECTION_ORDER
            for category in ("token_all_to_all", "replica_management")
            for phase in category_phases[direction][category]
        )
        if best_communication
        else frozenset()
    )

    summary_rows = []
    per_iteration_rows = []
    metadata = {}
    for spec in RUNS:
        run_summary, run_per_iteration, run_metadata = aggregate_run(
            logs_dir,
            spec,
            warmup=args.warmup,
            category_phases=category_phases,
            min_rank_phases=min_rank_phases,
            include_other=not best_communication,
            include_total=not best_communication,
        )
        for row in (*run_summary, *run_per_iteration):
            row["rank_policy_mode"] = args.rank_policy
            row["token_scope"] = args.token_scope
        summary_rows.extend(run_summary)
        per_iteration_rows.extend(run_per_iteration)
        metadata[spec.key] = run_metadata
        print(
            f"{spec.key}: {run_metadata['iterations']} complete iterations, "
            f"{run_metadata['layers']} MoE layers, "
            f"dropped={run_metadata['dropped_iterations']}"
        )

    write_csv(output_dir / "latency_breakdown.csv", summary_rows)
    write_csv(
        output_dir / "latency_breakdown_per_iteration.csv",
        per_iteration_rows,
    )
    stems = [
        plot_model(model, summary_rows, output_dir=output_dir, dpi=args.dpi)
        for model in ("qwen", "glm")
    ]
    stems.append(
        plot_combined(summary_rows, output_dir=output_dir, dpi=args.dpi)
    )
    write_readme(
        output_dir / "README.md",
        warmup=args.warmup,
        metadata=metadata,
        rank_policy=args.rank_policy,
        token_scope=args.token_scope,
    )

    for stem in stems:
        print(f"saved {stem.with_suffix('.png')}")
        print(f"saved {stem.with_suffix('.pdf')}")
    print(f"saved {output_dir / 'latency_breakdown.csv'}")


if __name__ == "__main__":
    main()

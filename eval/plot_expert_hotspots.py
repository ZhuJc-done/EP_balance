#!/usr/bin/env python3
"""Plot per-layer logical-expert hotspots from one or more Megatron routing traces."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eplb.trace_analysis import (  # noqa: E402
    load_routing_trace,
    metric_rows,
    normalize_expert_counts,
    select_expert_count_matrix,
)


def _trace_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label.strip():
            raise argparse.ArgumentTypeError("trace label cannot be empty")
    else:
        raw_path = value
        label = Path(value).stem
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"trace not found: {path}")
    return label.strip(), path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        type=_trace_arg,
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each workload; LABEL= is optional",
    )
    parser.add_argument(
        "--view",
        choices=["snapshot", "aggregate", "both"],
        default="both",
        help="One microbatch occurrence, whole-trace aggregate, or two columns",
    )
    parser.add_argument("--occurrence", type=int, default=0, help="Per-layer snapshot occurrence")
    parser.add_argument(
        "--max-occurrences",
        type=int,
        default=0,
        help="Aggregate at most this many occurrences per layer; 0 means all",
    )
    parser.add_argument(
        "--normalization",
        choices=["share", "relative", "count"],
        default="share",
        help="Expert share, ratio to layer mean, or raw assignment count",
    )
    parser.add_argument("--output", default="expert_hotspots.png")
    parser.add_argument("--csv", dest="csv_path", help="Metrics CSV; defaults beside the image")
    parser.add_argument("--title")
    parser.add_argument("--cmap", default="Reds")
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.5,
        help="Shared color maximum percentile when --vmax is absent",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _view_names(requested: str) -> list[str]:
    return ["snapshot", "aggregate"] if requested == "both" else [requested]


def _colorbar_label(normalization: str) -> str:
    return {
        "share": "Expert Token-Assignment Ratio",
        "relative": "Load / Layer Mean",
        "count": "Token-Expert Assignments",
    }[normalization]


def main() -> None:
    args = parse_args()
    if not 0 < args.color_percentile <= 100:
        raise ValueError("--color-percentile must be in (0, 100]")
    labels = [label for label, _ in args.trace]
    if len(set(labels)) != len(labels):
        raise ValueError("every --trace label must be unique")

    views = _view_names(args.view)
    panels = []
    all_metric_rows = []
    for label, path in args.trace:
        trace = load_routing_trace(path)
        meta = trace["meta"]
        if not meta.get("counts_reduced_over_tp_cp", False):
            print(
                f"warning: {path} does not declare TP×CP-reduced counts; "
                "use TP=CP=1 or recapture with trace format v2",
                file=sys.stderr,
            )
        for view in views:
            layers, raw_counts = select_expert_count_matrix(
                trace,
                view=view,
                occurrence=args.occurrence,
                max_occurrences=args.max_occurrences,
            )
            normalized = normalize_expert_counts(raw_counts, args.normalization)
            panels.append(
                {
                    "label": label,
                    "path": path,
                    "view": view,
                    "layers": layers,
                    "raw": raw_counts,
                    "matrix": normalized.numpy(),
                }
            )
            all_metric_rows.extend(
                metric_rows(
                    label=label,
                    view=view,
                    layers=layers,
                    expert_counts=raw_counts,
                    meta=meta,
                )
            )

    flattened = np.concatenate([panel["matrix"].ravel() for panel in panels])
    vmax = (
        args.vmax
        if args.vmax is not None
        else float(np.percentile(flattened, args.color_percentile))
    )
    if vmax <= 0:
        vmax = 1.0

    num_rows = len(args.trace)
    num_cols = len(views)
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(7.0 * num_cols, 3.2 * num_rows),
        squeeze=False,
        constrained_layout=True,
    )

    image = None
    panel_by_key = {(panel["label"], panel["view"]): panel for panel in panels}
    for row, (label, _path) in enumerate(args.trace):
        for col, view in enumerate(views):
            panel = panel_by_key[(label, view)]
            matrix = panel["matrix"]
            layers = panel["layers"]
            axis = axes[row, col]
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=args.cmap,
                vmin=0,
                vmax=vmax,
            )
            view_title = (
                f"Micro-batch occurrence {args.occurrence}"
                if view == "snapshot"
                else "Trace aggregate"
            )
            axis.set_title(f"{label} — {view_title}")
            axis.set_ylabel("MoE Layer")
            if row == num_rows - 1:
                axis.set_xlabel("Logical Expert ID")

            y_step = max(1, len(layers) // 8)
            y_positions = np.arange(0, len(layers), y_step)
            axis.set_yticks(y_positions)
            axis.set_yticklabels([str(layers[position] + 1) for position in y_positions])

            num_experts = matrix.shape[1]
            x_step = max(1, num_experts // 8)
            x_positions = np.arange(0, num_experts, x_step)
            axis.set_xticks(x_positions)
            axis.set_xticklabels([str(position) for position in x_positions])

    assert image is not None
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label=_colorbar_label(args.normalization),
        shrink=0.92,
    )
    if args.title:
        figure.suptitle(args.title)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)

    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output.with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "view",
        "layer",
        "total_assignments",
        "expert_max_mean",
        "rank_max_mean",
        "hot_expert",
        "hot_expert_share",
        "hot_rank",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metric_rows)

    print(f"[plot_expert_hotspots] image: {output}")
    print(f"[plot_expert_hotspots] metrics: {csv_path}")
    for label, _path in args.trace:
        for view in views:
            rows = [
                row
                for row in all_metric_rows
                if row["label"] == label and row["view"] == view
            ]
            expert_mean = sum(row["expert_max_mean"] for row in rows) / len(rows)
            rank_mean = sum(row["rank_max_mean"] for row in rows) / len(rows)
            print(
                f"[plot_expert_hotspots] {label}/{view}: "
                f"mean expert max/mean={expert_mean:.3f}, "
                f"mean original-rank max/mean={rank_mean:.3f}"
            )


if __name__ == "__main__":
    main()

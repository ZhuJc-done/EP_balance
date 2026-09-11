#!/usr/bin/env python3
"""Plot rank-load variation across layers, ranks, and micro-batch occurrences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from eplb.trace_analysis import (  # noqa: E402
    expert_count_cube,
    load_routing_trace,
    rank_loads_from_expert_counts,
)


def _trace_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(value).stem
    if not label.strip():
        raise argparse.ArgumentTypeError("trace label cannot be empty")
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
        help="Exactly two workload traces",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="One-based representative MoE layer; 0 selects the common layer with highest mean imbalance",
    )
    parser.add_argument(
        "--max-occurrences",
        type=int,
        default=0,
        help="Use at most this many complete occurrences; 0 means all",
    )
    parser.add_argument("--output", default="rank_dynamics.pdf")
    parser.add_argument("--csv", dest="csv_path")
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _rank_panel(label: str, path: Path, max_occurrences: int) -> dict:
    trace = load_routing_trace(path)
    layers, expert_counts = expert_count_cube(
        trace,
        max_occurrences=max_occurrences,
    )
    num_layers, occurrences, num_experts = expert_counts.shape
    flat_counts = expert_counts.reshape(num_layers * occurrences, num_experts)
    rank_loads = rank_loads_from_expert_counts(flat_counts, trace["meta"]).reshape(
        num_layers,
        occurrences,
        int(trace["meta"]["num_ranks"]),
    )
    means = rank_loads.to(torch.float64).mean(dim=-1, keepdim=True)
    relative = torch.where(
        means > 0,
        rank_loads.to(torch.float64) / means,
        torch.zeros_like(rank_loads, dtype=torch.float64),
    )
    return {
        "label": label,
        "path": path,
        "layers": layers,
        "rank_loads": rank_loads,
        "relative": relative,
        "max_mean": relative.max(dim=-1).values,
    }


def _representative_layer(panels: list[dict], requested_layer: int) -> int:
    common_layers = set(panels[0]["layers"])
    for panel in panels[1:]:
        common_layers.intersection_update(panel["layers"])
    if not common_layers:
        raise ValueError("traces have no common MoE layer")
    if requested_layer:
        layer = requested_layer - 1
        if layer not in common_layers:
            available = ", ".join(str(value + 1) for value in sorted(common_layers))
            raise ValueError(f"layer {requested_layer} is unavailable; common layers: {available}")
        return layer

    def combined_mean(layer: int) -> float:
        values = []
        for panel in panels:
            index = panel["layers"].index(layer)
            values.append(float(panel["max_mean"][index].mean()))
        return sum(values) / len(values)

    return max(sorted(common_layers), key=combined_mean)


def _set_occurrence_ticks(axis, occurrences: int) -> None:
    step = max(1, occurrences // 8)
    positions = np.arange(0, occurrences, step)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(position) for position in positions])


def main() -> None:
    args = parse_args()
    if len(args.trace) != 2:
        raise ValueError("the four-panel figure requires exactly two --trace arguments")
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    if args.max_occurrences < 0:
        raise ValueError("--max-occurrences must be non-negative")
    labels = [label for label, _ in args.trace]
    if len(set(labels)) != len(labels):
        raise ValueError("trace labels must be unique")

    panels = [
        _rank_panel(label, path, args.max_occurrences)
        for label, path in args.trace
    ]
    representative_layer = _representative_layer(panels, args.layer)

    top_values = np.concatenate(
        [panel["max_mean"].numpy().ravel() for panel in panels]
    )
    top_vmax = max(1.0001, float(np.percentile(top_values, 99.5)))

    representative_matrices = []
    for panel in panels:
        layer_index = panel["layers"].index(representative_layer)
        representative_matrices.append(panel["relative"][layer_index].T.numpy())
    relative_values = np.concatenate([matrix.ravel() for matrix in representative_matrices])
    max_deviation = max(
        0.05,
        float(np.percentile(np.abs(relative_values - 1.0), 99.5)),
    )
    bottom_norm = TwoSlopeNorm(
        vmin=max(0.0, 1.0 - max_deviation),
        vcenter=1.0,
        vmax=1.0 + max_deviation,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 8.2),
        constrained_layout=True,
    )
    top_images = []
    bottom_images = []
    for column, panel in enumerate(panels):
        max_mean = panel["max_mean"].numpy()
        top_axis = axes[0, column]
        top_images.append(
            top_axis.imshow(
                max_mean,
                aspect="auto",
                interpolation="nearest",
                cmap="YlOrRd",
                vmin=1.0,
                vmax=top_vmax,
            )
        )
        top_axis.set_title(f"({chr(ord('a') + column)}) {panel['label']}")
        top_axis.set_xlabel("Micro-batch occurrence")
        top_axis.set_ylabel("MoE layer")
        _set_occurrence_ticks(top_axis, max_mean.shape[1])
        y_step = max(1, len(panel["layers"]) // 8)
        y_positions = np.arange(0, len(panel["layers"]), y_step)
        top_axis.set_yticks(y_positions)
        top_axis.set_yticklabels(
            [str(panel["layers"][position] + 1) for position in y_positions]
        )

        rank_matrix = representative_matrices[column]
        bottom_axis = axes[1, column]
        bottom_images.append(
            bottom_axis.imshow(
                rank_matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                norm=bottom_norm,
            )
        )
        bottom_axis.set_title(
            f"({chr(ord('c') + column)}) {panel['label']}, layer {representative_layer + 1}"
        )
        bottom_axis.set_xlabel("Micro-batch occurrence")
        bottom_axis.set_ylabel("Expert-parallel rank")
        _set_occurrence_ticks(bottom_axis, rank_matrix.shape[1])
        rank_step = max(1, rank_matrix.shape[0] // 8)
        rank_positions = np.arange(0, rank_matrix.shape[0], rank_step)
        bottom_axis.set_yticks(rank_positions)
        bottom_axis.set_yticklabels([str(position) for position in rank_positions])

    figure.colorbar(
        top_images[0],
        ax=axes[0, :].tolist(),
        label="Rank max / mean load",
        shrink=0.92,
    )
    figure.colorbar(
        bottom_images[0],
        ax=axes[1, :].tolist(),
        label="Rank load / mean load",
        shrink=0.92,
    )
    if args.title:
        figure.suptitle(args.title)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output.with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=(
                "label",
                "layer",
                "occurrence",
                "rank_max_mean",
                "hot_rank",
            ),
        )
        writer.writeheader()
        for panel in panels:
            for layer_index, layer in enumerate(panel["layers"]):
                for occurrence in range(panel["max_mean"].shape[1]):
                    rank_values = panel["rank_loads"][layer_index, occurrence]
                    writer.writerow(
                        {
                            "label": panel["label"],
                            "layer": layer + 1,
                            "occurrence": occurrence,
                            "rank_max_mean": float(
                                panel["max_mean"][layer_index, occurrence]
                            ),
                            "hot_rank": int(rank_values.argmax()),
                        }
                    )

    print(f"[plot_rank_dynamics] representative layer: {representative_layer + 1}")
    print(f"[plot_rank_dynamics] image: {output}")
    print(f"[plot_rank_dynamics] metrics: {csv_path}")
    for panel in panels:
        values = panel["max_mean"]
        print(
            f"[plot_rank_dynamics] {panel['label']}: "
            f"occurrences={values.shape[1]} mean={float(values.mean()):.3f} "
            f"max={float(values.max()):.3f}"
        )


if __name__ == "__main__":
    main()

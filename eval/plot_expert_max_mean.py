#!/usr/bin/env python3
"""Plot expert max/mean routing imbalance for every MoE layer and micro-batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eplb.trace_analysis import expert_count_cube, load_routing_trace  # noqa: E402


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
        "--max-occurrences",
        type=int,
        default=0,
        help="Plot at most this many complete micro-batch occurrences; 0 means all",
    )
    parser.add_argument("--output", default="expert_max_mean.png")
    parser.add_argument("--csv", dest="csv_path", help="Metrics CSV; defaults beside the image")
    parser.add_argument("--title")
    parser.add_argument("--cmap", default="YlOrRd")
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.5,
        help="Shared color maximum percentile when --vmax is absent",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _layer_occurrence_stats(
    trace: dict,
    max_occurrences: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-[layer, occurrence] max, mean, max/mean, and hot expert."""
    layers, counts = expert_count_cube(trace, max_occurrences=max_occurrences)
    counts = counts.to(torch.float64)
    max_tokens, hot_experts = counts.max(dim=-1)
    mean_tokens = counts.mean(dim=-1)
    max_mean = torch.where(
        mean_tokens > 0,
        max_tokens / mean_tokens,
        torch.zeros_like(mean_tokens),
    )
    return layers, max_tokens, mean_tokens, max_mean, hot_experts


def main() -> None:
    args = parse_args()
    if not 0 < args.color_percentile <= 100:
        raise ValueError("--color-percentile must be in (0, 100]")
    labels = [label for label, _ in args.trace]
    if len(set(labels)) != len(labels):
        raise ValueError("every --trace label must be unique")

    panels = []
    csv_rows = []
    for label, path in args.trace:
        trace = load_routing_trace(path)
        layers, max_tokens, mean_tokens, max_mean, hot_experts = _layer_occurrence_stats(
            trace,
            args.max_occurrences,
        )
        panels.append(
            {
                "label": label,
                "layers": layers,
                "max_mean": max_mean.numpy(),
            }
        )
        for layer_index, layer in enumerate(layers):
            for occurrence in range(max_mean.shape[1]):
                csv_rows.append(
                    {
                        "label": label,
                        "occurrence": occurrence,
                        "layer": layer,
                        "max_tokens": int(max_tokens[layer_index, occurrence]),
                        "mean_tokens": float(mean_tokens[layer_index, occurrence]),
                        "max_mean": float(max_mean[layer_index, occurrence]),
                        "hot_expert": int(hot_experts[layer_index, occurrence]),
                    }
                )

    positive_values = np.concatenate(
        [panel["max_mean"][panel["max_mean"] > 0] for panel in panels]
    )
    vmax = (
        args.vmax
        if args.vmax is not None
        else float(np.percentile(positive_values, args.color_percentile))
    )
    vmax = max(vmax, 1.0001)

    figure, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10.0, 3.4 * len(panels)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row, panel in enumerate(panels):
        matrix = panel["max_mean"]
        layers = panel["layers"]
        axis = axes[row, 0]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=args.cmap,
            vmin=1.0,
            vmax=vmax,
        )
        axis.set_title(panel["label"])
        axis.set_ylabel("MoE Layer")
        axis.set_xlabel("Micro-batch Occurrence")

        y_step = max(1, len(layers) // 8)
        y_positions = np.arange(0, len(layers), y_step)
        axis.set_yticks(y_positions)
        axis.set_yticklabels([str(layers[position] + 1) for position in y_positions])

        num_occurrences = matrix.shape[1]
        x_step = max(1, num_occurrences // 10)
        x_positions = np.arange(0, num_occurrences, x_step)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(position) for position in x_positions])

    assert image is not None
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Max Expert Tokens / Mean Expert Tokens",
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
        "occurrence",
        "layer",
        "max_tokens",
        "mean_tokens",
        "max_mean",
        "hot_expert",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[plot_expert_max_mean] image: {output}")
    print(f"[plot_expert_max_mean] metrics: {csv_path}")
    for panel in panels:
        matrix = panel["max_mean"]
        max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
        print(
            f"[plot_expert_max_mean] {panel['label']}: "
            f"mean={matrix.mean():.3f}, max={matrix[max_index]:.3f} "
            f"at layer={panel['layers'][max_index[0]]}, occurrence={max_index[1]}"
        )


if __name__ == "__main__":
    main()

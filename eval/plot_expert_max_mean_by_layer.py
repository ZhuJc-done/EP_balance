#!/usr/bin/env python3
"""Plot expert max/mean routing imbalance as a function of MoE layer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from eplb.trace_analysis import expert_max_mean_by_layer, load_routing_trace  # noqa: E402


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
        help="Use at most this many complete micro-batch occurrences; 0 means all",
    )
    parser.add_argument(
        "--band-percentiles",
        type=float,
        nargs=2,
        default=(10.0, 90.0),
        metavar=("LOW", "HIGH"),
        help="Percentile band across micro-batches (default: 10 90)",
    )
    parser.add_argument(
        "--show-aggregate",
        action="store_true",
        help="Also draw a dashed ratio computed after summing all occurrences",
    )
    parser.add_argument("--output", default="expert_max_mean_by_layer.png")
    parser.add_argument("--csv", dest="csv_path", help="CSV path; defaults beside the image")
    parser.add_argument("--title", default="Expert routing imbalance by MoE layer")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _layer_summary(
    trace: dict,
    *,
    max_occurrences: int,
    low_percentile: float,
    high_percentile: float,
) -> dict:
    layers, batch_ratios, aggregate_ratio = expert_max_mean_by_layer(
        trace,
        max_occurrences=max_occurrences,
    )
    quantiles = torch.quantile(
        batch_ratios,
        torch.tensor(
            [low_percentile / 100.0, 0.5, high_percentile / 100.0],
            dtype=batch_ratios.dtype,
        ),
        dim=1,
    )
    return {
        "layers": layers,
        "occurrences": batch_ratios.shape[1],
        "mean": batch_ratios.mean(dim=1).numpy(),
        "median": quantiles[1].numpy(),
        "band_low": quantiles[0].numpy(),
        "band_high": quantiles[2].numpy(),
        "minimum": batch_ratios.min(dim=1).values.numpy(),
        "maximum": batch_ratios.max(dim=1).values.numpy(),
        "aggregate": aggregate_ratio.numpy(),
    }


def main() -> None:
    args = parse_args()
    if args.max_occurrences < 0:
        raise ValueError("--max-occurrences must be non-negative")
    low_percentile, high_percentile = args.band_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("--band-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    labels = [label for label, _ in args.trace]
    if len(set(labels)) != len(labels):
        raise ValueError("every --trace label must be unique")

    summaries = []
    csv_rows = []
    for label, path in args.trace:
        trace = load_routing_trace(path)
        summary = _layer_summary(
            trace,
            max_occurrences=args.max_occurrences,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )
        summary["label"] = label
        summaries.append(summary)

        for index, layer in enumerate(summary["layers"]):
            csv_rows.append(
                {
                    "label": label,
                    "layer": layer,
                    "display_layer": layer + 1,
                    "occurrences": summary["occurrences"],
                    "mean_batch_max_mean": summary["mean"][index],
                    "median_batch_max_mean": summary["median"][index],
                    "band_low_percentile": low_percentile,
                    "band_low_max_mean": summary["band_low"][index],
                    "band_high_percentile": high_percentile,
                    "band_high_max_mean": summary["band_high"][index],
                    "min_batch_max_mean": summary["minimum"][index],
                    "max_batch_max_mean": summary["maximum"][index],
                    "aggregate_max_mean": summary["aggregate"][index],
                }
            )

    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for summary in summaries:
        x_values = np.asarray(summary["layers"]) + 1
        (line,) = axis.plot(
            x_values,
            summary["mean"],
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label=summary["label"],
        )
        axis.fill_between(
            x_values,
            summary["band_low"],
            summary["band_high"],
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )
        if args.show_aggregate:
            axis.plot(
                x_values,
                summary["aggregate"],
                color=line.get_color(),
                linewidth=1.4,
                linestyle="--",
                alpha=0.8,
                label=f"{summary['label']} (trace aggregate)",
            )

    axis.axhline(1.0, color="#555555", linewidth=1.0, linestyle=":", alpha=0.8)
    axis.set_xlabel("MoE layer (1-based)")
    axis.set_ylabel("Max expert tokens / mean expert tokens")
    axis.set_title(args.title)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    axis.grid(axis="both", alpha=0.22)
    axis.set_ylim(bottom=0.95)
    axis.legend(frameon=False, ncol=2 if len(summaries) > 1 else 1)

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
        "layer",
        "display_layer",
        "occurrences",
        "mean_batch_max_mean",
        "median_batch_max_mean",
        "band_low_percentile",
        "band_low_max_mean",
        "band_high_percentile",
        "band_high_max_mean",
        "min_batch_max_mean",
        "max_batch_max_mean",
        "aggregate_max_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[plot_expert_max_mean_by_layer] image: {output}")
    print(f"[plot_expert_max_mean_by_layer] metrics: {csv_path}")
    for summary in summaries:
        peak_index = int(np.argmax(summary["mean"]))
        print(
            f"[plot_expert_max_mean_by_layer] {summary['label']}: "
            f"layer-mean={summary['mean'].mean():.3f}, "
            f"peak={summary['mean'][peak_index]:.3f} "
            f"at layer={summary['layers'][peak_index] + 1}, "
            f"occurrences={summary['occurrences']}"
        )


if __name__ == "__main__":
    main()

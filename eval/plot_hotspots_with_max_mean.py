#!/usr/bin/env python3
"""Combine snapshot expert hotspots with a max/mean-by-layer line plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from eplb.trace_analysis import (  # noqa: E402
    expert_max_mean_by_layer,
    load_routing_trace,
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
        help="Repeat for each workload; each trace becomes one top-row heatmap",
    )
    parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="Micro-batch occurrence shown in the hotspot panels",
    )
    parser.add_argument(
        "--max-occurrences",
        type=int,
        default=0,
        help="Use at most this many complete occurrences for the line plot; 0 means all",
    )
    parser.add_argument(
        "--band-percentiles",
        type=float,
        nargs=2,
        default=(10.0, 90.0),
        metavar=("LOW", "HIGH"),
        help="Line-plot percentile band across micro-batches (default: 10 90)",
    )
    parser.add_argument(
        "--normalization",
        choices=["share", "relative", "count"],
        default="share",
        help="Hotspot normalization",
    )
    parser.add_argument(
        "--output",
        default="hotspots_with_max_mean.pdf",
        help="Output figure; the extension selects PDF, PNG, or SVG",
    )
    parser.add_argument("--cmap", default="Reds")
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.5,
        help="Shared heatmap color maximum percentile when --vmax is absent",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-size",
        type=float,
        default=26.0,
        help="Base size in points for labels, ticks, legends, and subcaptions",
    )
    return parser.parse_args()


def _colorbar_label(normalization: str) -> str:
    return {
        "share": "",
        "relative": "Load / Layer Mean",
        "count": "Token-Expert Assignments",
    }[normalization]


def _line_summary(
    trace: dict,
    *,
    max_occurrences: int,
    low_percentile: float,
    high_percentile: float,
) -> dict:
    layers, batch_ratios, _aggregate = expert_max_mean_by_layer(
        trace,
        max_occurrences=max_occurrences,
    )
    quantiles = torch.quantile(
        batch_ratios,
        torch.tensor(
            [low_percentile / 100.0, high_percentile / 100.0],
            dtype=batch_ratios.dtype,
        ),
        dim=1,
    )
    return {
        "layers": layers,
        "occurrences": batch_ratios.shape[1],
        "mean": batch_ratios.mean(dim=1).numpy(),
        "band_low": quantiles[0].numpy(),
        "band_high": quantiles[1].numpy(),
    }


def main() -> None:
    args = parse_args()
    if args.occurrence < 0:
        raise ValueError("--occurrence must be non-negative")
    if args.max_occurrences < 0:
        raise ValueError("--max-occurrences must be non-negative")
    if args.font_size <= 0:
        raise ValueError("--font-size must be positive")
    if not 0 < args.color_percentile <= 100:
        raise ValueError("--color-percentile must be in (0, 100]")
    low_percentile, high_percentile = args.band_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("--band-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    labels = [label for label, _ in args.trace]
    if len(set(labels)) != len(labels):
        raise ValueError("every --trace label must be unique")

    matplotlib.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
            "legend.fontsize": args.font_size,
        }
    )

    panels = []
    for label, path in args.trace:
        trace = load_routing_trace(path)
        layers, raw_counts = select_expert_count_matrix(
            trace,
            view="snapshot",
            occurrence=args.occurrence,
        )
        panels.append(
            {
                "label": label,
                "layers": layers,
                "matrix": normalize_expert_counts(
                    raw_counts,
                    args.normalization,
                ).numpy(),
                "line": _line_summary(
                    trace,
                    max_occurrences=args.max_occurrences,
                    low_percentile=low_percentile,
                    high_percentile=high_percentile,
                ),
            }
        )

    flattened = np.concatenate([panel["matrix"].ravel() for panel in panels])
    vmax = (
        args.vmax
        if args.vmax is not None
        else float(np.percentile(flattened, args.color_percentile))
    )
    if vmax <= 0:
        vmax = 1.0

    num_panels = len(panels)
    figure = plt.figure(
        figsize=(6.2 * num_panels, 8.2),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        4,
        num_panels,
        height_ratios=(1.12, 0.07, 0.88, 0.07),
    )
    heatmap_axes = [
        figure.add_subplot(grid[0, column])
        for column in range(num_panels)
    ]
    heatmap_caption_axes = [
        figure.add_subplot(grid[1, column])
        for column in range(num_panels)
    ]
    for panel_index, (caption_axis, panel) in enumerate(
        zip(heatmap_caption_axes, panels)
    ):
        caption_axis.axis("off")
        caption_axis.text(
            0.5,
            0.5,
            f"({chr(ord('a') + panel_index)}) {panel['label']} routing hotspots",
            ha="center",
            va="center",
            fontsize=args.font_size,
        )
    line_axis = figure.add_subplot(grid[2, :])
    bottom_caption_axis = figure.add_subplot(grid[3, :])
    bottom_caption_axis.axis("off")
    bottom_caption_axis.text(
        0.5,
        0.5,
        f"({chr(ord('a') + num_panels)}) Layer-wise expert load imbalance",
        ha="center",
        va="center",
        fontsize=args.font_size,
    )

    image = None
    for axis, panel in zip(heatmap_axes, panels):
        matrix = panel["matrix"]
        layers = panel["layers"]
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=args.cmap,
            vmin=0,
            vmax=vmax,
        )
        axis.set_xlabel("Logical Expert ID")
        axis.set_ylabel("MoE Layer")

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
        ax=heatmap_axes,
        label=_colorbar_label(args.normalization),
        shrink=0.92,
        pad=0.02,
    )

    for panel in panels:
        summary = panel["line"]
        x_values = np.asarray(summary["layers"]) + 1
        (line,) = line_axis.plot(
            x_values,
            summary["mean"],
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label=panel["label"],
        )
        line_axis.fill_between(
            x_values,
            summary["band_low"],
            summary["band_high"],
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )

    line_axis.axhline(
        1.0,
        color="#555555",
        linewidth=1.0,
        linestyle=":",
        alpha=0.8,
    )
    line_axis.set_xlabel("MoE layer")
    line_axis.set_ylabel("Max / mean expert tokens")
    line_axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    line_axis.grid(axis="both", alpha=0.22)
    line_axis.set_ylim(bottom=0.95)
    line_axis.legend(frameon=False, ncol=2 if num_panels > 1 else 1)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"[plot_hotspots_with_max_mean] image: {output}")
    for panel in panels:
        summary = panel["line"]
        peak_index = int(np.argmax(summary["mean"]))
        print(
            f"[plot_hotspots_with_max_mean] {panel['label']}: "
            f"peak mean max/mean={summary['mean'][peak_index]:.3f} "
            f"at layer={summary['layers'][peak_index] + 1}, "
            f"occurrences={summary['occurrences']}"
        )


if __name__ == "__main__":
    main()

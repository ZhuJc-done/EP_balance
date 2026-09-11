#!/usr/bin/env python3
"""Plot a truncated 10k-step loss comparison for Megatron-LM and Scale-EPLB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402


ITERATION_RE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<total>\d+).*?"
    r"lm loss:\s*(?P<loss>[\dEe.+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--scale-log", type=Path)
    parser.add_argument(
        "--curve-input",
        type=Path,
        help="Saved loss_comparison_50step.csv for an offline figure-only replot",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdf-output", type=Path)
    parser.add_argument("--curve-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--bucket",
        type=int,
        default=50,
        help="Number of iterations averaged into each plotted point.",
    )
    parser.add_argument("--early-end", type=int, default=3000)
    parser.add_argument("--late-start", type=int, default=7000)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_loss(path: Path) -> dict[int, float]:
    losses: dict[int, float] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            match = ITERATION_RE.search(line)
            if match is None:
                continue
            iteration = int(match.group("iteration"))
            if iteration in losses:
                raise ValueError(f"{path}: duplicate iteration {iteration}")
            losses[iteration] = float(match.group("loss"))
    if not losses:
        raise ValueError(f"{path}: no iteration losses found")
    return losses


def bucket_means(
    losses: dict[int, float],
    *,
    first: int,
    last: int,
    bucket: int,
) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    means: list[float] = []
    for start in range(first, last + 1, bucket):
        stop = min(start + bucket - 1, last)
        window = [(step, losses[step]) for step in range(start, stop + 1)]
        steps.append(statistics.mean(step for step, _ in window))
        means.append(statistics.mean(loss for _, loss in window))
    return steps, means


def comparison_rows(
    baseline: dict[int, float],
    scale: dict[int, float],
    *,
    bucket: int,
) -> list[dict[str, float | int]]:
    final_step = max(set(baseline).intersection(scale))
    rows: list[dict[str, float | int]] = []
    for first in range(1, final_step + 1, bucket):
        last = min(first + bucket - 1, final_step)
        baseline_mean = statistics.mean(baseline[step] for step in range(first, last + 1))
        scale_mean = statistics.mean(scale[step] for step in range(first, last + 1))
        absolute_difference = abs(scale_mean - baseline_mean)
        rows.append(
            {
                "first_iteration": first,
                "last_iteration": last,
                "baseline_mean_loss": baseline_mean,
                "scale_eplb_mean_loss": scale_mean,
                "absolute_difference": absolute_difference,
                "relative_difference_percent": (
                    100.0 * absolute_difference / baseline_mean
                ),
            }
        )
    return rows


def load_comparison_rows(path: Path) -> list[dict[str, float | int]]:
    required = {
        "first_iteration",
        "last_iteration",
        "baseline_mean_loss",
        "scale_eplb_mean_loss",
        "absolute_difference",
        "relative_difference_percent",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing curve columns: {', '.join(sorted(missing))}"
            )
        rows = [
            {
                "first_iteration": int(row["first_iteration"]),
                "last_iteration": int(row["last_iteration"]),
                "baseline_mean_loss": float(row["baseline_mean_loss"]),
                "scale_eplb_mean_loss": float(row["scale_eplb_mean_loss"]),
                "absolute_difference": float(row["absolute_difference"]),
                "relative_difference_percent": float(
                    row["relative_difference_percent"]
                ),
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path}: empty loss curve")
    rows.sort(key=lambda row: int(row["first_iteration"]))
    expected_first = 1
    for row in rows:
        first = int(row["first_iteration"])
        last = int(row["last_iteration"])
        if first != expected_first or last < first:
            raise ValueError(f"{path}: curve buckets are not contiguous")
        expected_first = last + 1
    return rows


def nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def comparison_summary(
    baseline: dict[int, float],
    scale: dict[int, float],
    *,
    bucket: int,
    early_end: int,
    late_start: int,
) -> dict[str, object]:
    common = sorted(set(baseline).intersection(scale))
    signed = [scale[step] - baseline[step] for step in common]
    absolute = [abs(value) for value in signed]
    relative = [
        absolute_difference / baseline[step]
        for step, absolute_difference in zip(common, absolute)
    ]
    tail = common[-1000:]
    return {
        "iterations": len(common),
        "baseline_mean_loss": statistics.mean(baseline[step] for step in common),
        "scale_eplb_mean_loss": statistics.mean(scale[step] for step in common),
        "mean_signed_difference": statistics.mean(signed),
        "mean_absolute_difference": statistics.mean(absolute),
        "rmse": math.sqrt(statistics.mean(value * value for value in signed)),
        "mean_relative_difference_percent": 100.0 * statistics.mean(relative),
        "p95_relative_difference_percent": 100.0 * nearest_rank(relative, 0.95),
        "maximum_relative_difference_percent": 100.0 * max(relative),
        "baseline_final_loss": baseline[common[-1]],
        "scale_eplb_final_loss": scale[common[-1]],
        "baseline_final_1000_mean_loss": statistics.mean(
            baseline[step] for step in tail
        ),
        "scale_eplb_final_1000_mean_loss": statistics.mean(
            scale[step] for step in tail
        ),
        "plot": {
            "aggregation": f"non-overlapping {bucket}-iteration means",
            "shown_iteration_ranges": [[1, early_end], [late_start, common[-1]]],
            "omitted_iteration_range": [early_end + 1, late_start - 1],
        },
    }


def write_curve(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
    axis.tick_params(axis="both", labelsize=8)
    axis.spines["top"].set_visible(False)


def format_step(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return f"{step:,}"


def add_break_marks(left: plt.Axes, right: plt.Axes) -> None:
    size = 0.015
    kwargs = {"color": "#333333", "clip_on": False, "linewidth": 0.9}
    left.plot((1 - size, 1 + size), (-size, +size), transform=left.transAxes, **kwargs)
    left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=left.transAxes, **kwargs)
    right.plot((-size, +size), (-size, +size), transform=right.transAxes, **kwargs)
    right.plot((-size, +size), (1 - size, 1 + size), transform=right.transAxes, **kwargs)


def plot(
    rows: list[dict[str, float | int]],
    *,
    output: Path,
    pdf_output: Path,
    early_end: int,
    late_start: int,
    dpi: int,
) -> None:
    final_step = int(rows[-1]["last_iteration"])
    if early_end >= late_start:
        raise ValueError("--early-end must be smaller than --late-start")
    if not (1 <= early_end < late_start <= final_step):
        raise ValueError("truncated ranges fall outside the logged iterations")
    if dpi <= 0:
        raise ValueError("--dpi must be positive")

    figure, (left, right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(4.2, 2.25),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.07},
    )
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.23, top=0.77)

    styles = {
        "Megatron-LM": {
            "column": "baseline_mean_loss",
            "color": "#555555",
            "linestyle": "--",
            "marker": "s",
            "markevery": (0, 4),
            "zorder": 5,
        },
        "Scale-EPLB": {
            "column": "scale_eplb_mean_loss",
            "color": "#E15759",
            "linestyle": "-",
            "marker": "o",
            "markevery": (2, 4),
            "zorder": 4,
        },
    }
    ranges = ((left, 1, early_end), (right, late_start, final_step))
    for axis, first, last in ranges:
        for label, style in styles.items():
            visible = [
                row
                for row in rows
                if int(row["first_iteration"]) >= first
                and int(row["last_iteration"]) <= last
            ]
            if not visible:
                raise ValueError(
                    f"no complete curve buckets in iteration range {first}-{last}"
                )
            steps = [
                statistics.mean(
                    (int(row["first_iteration"]), int(row["last_iteration"]))
                )
                for row in visible
            ]
            means = [float(row[str(style["column"])]) for row in visible]
            axis.plot(
                steps,
                means,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                marker=style["marker"],
                markersize=2.4,
                markeredgewidth=0.45,
                markevery=style["markevery"],
                zorder=style["zorder"],
            )
        axis.set_xlim(first - 0.03 * (last - first), last + 0.03 * (last - first))
        configure_axis(axis)

    left.set_ylabel("Language-model loss", fontsize=8)
    left_ticks = [1, *range(1000, early_end + 1, 1000)]
    right_ticks = list(range(late_start, final_step + 1, 1000))
    left.set_xticks(left_ticks)
    left.set_xticklabels([format_step(step) for step in left_ticks])
    right.set_xticks(right_ticks)
    right.set_xticklabels([format_step(step) for step in right_ticks])
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.spines["right"].set_visible(False)
    right.tick_params(axis="y", left=False, labelleft=False)
    add_break_marks(left, right)

    handles, labels = left.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.98),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=2.5,
        columnspacing=1.6,
    )
    figure.text(0.54, 0.055, "Training step", ha="center", fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.bucket <= 0:
        raise ValueError("--bucket must be positive")
    if args.curve_input is not None:
        if args.baseline_log is not None or args.scale_log is not None:
            raise ValueError(
                "--curve-input cannot be combined with --baseline-log/--scale-log"
            )
        if args.summary_output is not None:
            raise ValueError(
                "--summary-output requires raw logs; preserve the existing summary "
                "when replotting from --curve-input"
            )
        curve_input = args.curve_input.expanduser().resolve()
        rows = load_comparison_rows(curve_input)
        baseline = None
        scale = None
    else:
        if args.baseline_log is None or args.scale_log is None:
            raise ValueError(
                "provide --curve-input, or provide both --baseline-log and --scale-log"
            )
        baseline = load_loss(args.baseline_log.expanduser().resolve())
        scale = load_loss(args.scale_log.expanduser().resolve())
        common = sorted(set(baseline).intersection(scale))
        if common != list(range(1, common[-1] + 1)):
            raise ValueError("logs must contain the same contiguous iteration range")
        rows = comparison_rows(baseline, scale, bucket=args.bucket)

    output = args.output.expanduser().resolve()
    pdf_output = (
        args.pdf_output.expanduser().resolve()
        if args.pdf_output
        else output.with_suffix(".pdf")
    )
    plot(
        rows,
        output=output,
        pdf_output=pdf_output,
        early_end=args.early_end,
        late_start=args.late_start,
        dpi=args.dpi,
    )
    if args.curve_output:
        curve_output = args.curve_output.expanduser().resolve()
        write_curve(rows, curve_output)
        print(f"saved {curve_output}")
    if args.summary_output:
        assert baseline is not None and scale is not None
        summary_output = args.summary_output.expanduser().resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(
                comparison_summary(
                    baseline,
                    scale,
                    bucket=args.bucket,
                    early_end=args.early_end,
                    late_start=args.late_start,
                ),
                indent=2,
            )
            + "\n"
        )
        print(f"saved {summary_output}")
    if args.curve_input is not None:
        print(
            f"loaded {len(rows)} loss buckets covering iterations "
            f"1-{rows[-1]['last_iteration']} from {curve_input}"
        )
    else:
        assert baseline is not None and scale is not None
        print(f"loaded {len(baseline)} baseline and {len(scale)} Scale-EPLB losses")
    print(f"saved {output}")
    print(f"saved {pdf_output}")


if __name__ == "__main__":
    main()

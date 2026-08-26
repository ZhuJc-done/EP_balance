#!/usr/bin/env python3
"""Plot steady-state E2E throughput by router skew from per-run CSV data."""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402


MODEL_STYLE = {
    "qwen3_5L": {
        "title": "Qwen3-30B-A3B",
        "stem": "qwen_e2e_throughput_by_skew",
    },
    "glm45_air_5moe": {
        "title": "GLM-4.5-Air",
        "stem": "glm_e2e_throughput_by_skew",
    },
}
METHOD_STYLE = {
    "native": "Megatron-LM Baseline",
    "scale_eplb": "Scale-EPLB",
}
METRIC_STYLE = {
    "tokens": {
        "source_column": "tokens_per_second_global",
        "scale": 1.0 / 1000.0,
        "label": "Training throughput (k tokens/s)",
        "unit": "k tokens/s",
        "suffix": "tokens",
    },
    "tflops": {
        "source_column": "tflops_per_gpu",
        "scale": 1.0,
        "label": "Training throughput (TFLOP/s/GPU)",
        "unit": "TFLOP/s/GPU",
        "suffix": "tflops_per_gpu",
    },
}
SKEW_ORDER = ("natural", "-0.5", "-1.0", "-2.0", "-4.0")
SKEW_STYLE = {
    "natural": {
        "label": "Natural",
        "color": "#4C4C4C",
        "marker": "o",
        "linestyle": "-",
    },
    "-0.5": {
        "label": r"$\mathtt{router\_skew}=-0.5$",
        "color": "#4C78A8",
        "marker": "s",
        "linestyle": "--",
    },
    "-1.0": {
        "label": r"$\mathtt{router\_skew}=-1.0$",
        "color": "#59A14F",
        "marker": "^",
        "linestyle": "-.",
    },
    "-2.0": {
        "label": r"$\mathtt{router\_skew}=-2.0$",
        "color": "#F28E2B",
        "marker": "D",
        "linestyle": ":",
    },
    "-4.0": {
        "label": r"$\mathtt{router\_skew}=-4.0$",
        "color": "#D62728",
        "marker": "P",
        "linestyle": (0, (5, 1.5)),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing aligned or measured per-run CSV files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for PNG, PDF, and plotted-point CSV outputs",
    )
    parser.add_argument("--step-start", type=int, default=3000)
    parser.add_argument("--step-end", type=int, default=5000)
    parser.add_argument(
        "--num-points",
        type=int,
        default=5,
        help="Number of evenly spaced key points to draw",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Aligned samples used for the median at each key point",
    )
    parser.add_argument(
        "--measured-tail",
        type=int,
        default=0,
        help=(
            "Use this many final measured rows from each ordinary metrics CSV and "
            "map them, in order, to a range ending at --step-end; 0 expects aligned CSVs"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_STYLE),
        default="tokens",
        help="Throughput metric to plot",
    )
    parser.add_argument(
        "--latest-stable-window",
        action="store_true",
        help=(
            "For measured data, scan backward for the latest window with no step "
            "slower than --slow-factor times its median"
        ),
    )
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=1.25,
        help="Maximum step-time factor allowed by --latest-stable-window",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _skew_key(row: dict[str, str]) -> str:
    if row["routing"] == "natural":
        return "natural"
    value = float(row["router_skew"])
    return f"{value:.1f}"


def _load_series(
    input_dir: Path,
    *,
    measured_tail: int,
    aligned_end: int,
    metric: str,
    latest_stable_window: bool,
    slow_factor: float,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    paths = sorted(input_dir.glob("e2e_*seed1234.csv"))
    if not paths:
        raise FileNotFoundError(f"no experiment CSV files found in {input_dir}")
    if measured_tail < 0:
        raise ValueError("--measured-tail cannot be negative")
    if latest_stable_window and not measured_tail:
        raise ValueError("--latest-stable-window requires --measured-tail")
    if slow_factor <= 1.0:
        raise ValueError("--slow-factor must be greater than 1")

    series: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    metric_style = METRIC_STYLE[metric]
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        if not raw_rows:
            raise ValueError(f"{path}: empty CSV")
        if not measured_tail and any(row.get("is_constructed") != "true" for row in raw_rows):
            raise ValueError(f"{path}: expected explicitly marked aligned data")
        if measured_tail and len(raw_rows) < measured_tail:
            raise ValueError(
                f"{path}: requires {measured_tail} measured rows, found {len(raw_rows)}"
            )

        model = raw_rows[0]["model"]
        method = raw_rows[0]["method"]
        skew = _skew_key(raw_rows[0])
        key = (model, method, skew)
        if key in series:
            raise ValueError(f"duplicate experiment series: {key}")

        if measured_tail:
            selected_rows = _select_measured_window(
                raw_rows,
                length=measured_tail,
                latest_stable=latest_stable_window,
                slow_factor=slow_factor,
            )
            aligned_start = aligned_end - measured_tail + 1
            rows = [
                {
                    "aligned_step": aligned_start + index,
                    "throughput": (
                        float(row[str(metric_style["source_column"])])
                        * float(metric_style["scale"])
                    ),
                    "source_iteration": int(row["iteration"]),
                }
                for index, row in enumerate(selected_rows)
            ]
            print(
                f"{path.name}: measured window "
                f"{selected_rows[0]['iteration']}-{selected_rows[-1]['iteration']}"
            )
        else:
            rows = [
                {
                    "aligned_step": int(row["aligned_step"]),
                    "throughput": (
                        float(row[str(metric_style["source_column"])])
                        * float(metric_style["scale"])
                    ),
                    "source_iteration": int(row["source_iteration"]),
                }
                for row in raw_rows
            ]
        rows.sort(key=lambda row: row["aligned_step"])
        series[key] = rows
    return series


def _select_measured_window(
    rows: list[dict[str, str]],
    *,
    length: int,
    latest_stable: bool,
    slow_factor: float,
) -> list[dict[str, str]]:
    if not latest_stable:
        return rows[-length:]
    for end in range(len(rows), length - 1, -1):
        candidate = rows[end - length : end]
        elapsed = [float(row["elapsed_ms"]) for row in candidate]
        threshold = slow_factor * st.median(elapsed)
        if all(value <= threshold for value in elapsed):
            return candidate
    raise ValueError(
        f"no {length}-row stable window found with slow-factor {slow_factor}"
    )


def _key_steps(start: int, end: int, count: int) -> list[int]:
    if start <= 0 or end <= start:
        raise ValueError("step range must satisfy 0 < start < end")
    if count < 2:
        raise ValueError("--num-points must be at least 2")
    return [
        round(start + index * (end - start) / (count - 1))
        for index in range(count)
    ]


def _window_bounds(center: int, start: int, end: int, width: int) -> tuple[int, int]:
    if width <= 0 or width > end - start + 1:
        raise ValueError("--window must be positive and no larger than the plotted range")
    half = width // 2
    lower = center - half
    upper = lower + width - 1
    if lower < start:
        lower, upper = start, start + width - 1
    if upper > end:
        lower, upper = end - width + 1, end
    return lower, upper


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    centers: list[int],
    step_start: int,
    step_end: int,
    window: int,
) -> list[dict[str, float | int]]:
    aggregated = []
    for center in centers:
        lower, upper = _window_bounds(center, step_start, step_end, window)
        window_rows = [
            row
            for row in rows
            if lower <= int(row["aligned_step"]) <= upper
        ]
        if len(window_rows) != window:
            raise ValueError(
                f"key point {center}: expected {window} samples, found {len(window_rows)}"
            )
        values = [float(row["throughput"]) for row in window_rows]
        aggregated.append(
            {
                "aligned_step": center,
                "window_first_step": lower,
                "window_last_step": upper,
                "source_iteration_first": int(window_rows[0]["source_iteration"]),
                "source_iteration_last": int(window_rows[-1]["source_iteration"]),
                "throughput_median": st.median(values),
                "throughput_p25": _percentile(values, 0.25),
                "throughput_p75": _percentile(values, 0.75),
            }
        )
    return aggregated


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _plot_model(
    model: str,
    points: dict[tuple[str, str, str], list[dict[str, float | int]]],
    *,
    output_dir: Path,
    step_start: int,
    step_end: int,
    metric: str,
    dpi: int,
) -> Path:
    available_skews = [
        skew
        for skew in SKEW_ORDER
        if any(key[0] == model and key[2] == skew for key in points)
    ]
    if not available_skews:
        raise ValueError(f"no series found for model {model}")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.45),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.23,
        top=0.79,
        wspace=0.06,
    )
    handles: dict[str, Any] = {}
    for panel_index, method in enumerate(METHOD_STYLE):
        axis = axes[panel_index]
        for skew in available_skews:
            key = (model, method, skew)
            if key not in points:
                continue
            key_points = points[key]
            style = SKEW_STYLE[skew]
            line = axis.plot(
                [int(point["aligned_step"]) for point in key_points],
                [float(point["throughput_median"]) for point in key_points],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )[0]
            handles[skew] = line

        axis.set_xlabel("Training step", fontsize=9.5, labelpad=3)
        axis.text(
            0.5,
            -0.30,
            f"({chr(ord('a') + panel_index)}) {METHOD_STYLE[method]}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
        )
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
        axis.tick_params(axis="both", labelsize=8.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel(str(METRIC_STYLE[metric]["label"]), fontsize=9.5)

    figure.suptitle(MODEL_STYLE[model]["title"], fontsize=11.5, y=0.98)
    figure.legend(
        [handles[skew] for skew in available_skews],
        [SKEW_STYLE[skew]["label"] for skew in available_skews],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(available_skews),
        frameon=False,
        fontsize=7.5,
        handlelength=2.2,
        columnspacing=1.0,
    )

    output_stem = output_dir / (
        f"{MODEL_STYLE[model]['stem']}_{METRIC_STYLE[metric]['suffix']}"
        f"_steps{step_start}_{step_end}"
    )
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_stem


def _write_points(
    points: dict[tuple[str, str, str], list[dict[str, float | int]]],
    path: Path,
    *,
    metric: str,
) -> None:
    fieldnames = [
        "model",
        "method",
        "routing",
        "router_skew",
        "metric",
        "unit",
        "aligned_step",
        "window_first_step",
        "window_last_step",
        "source_iteration_first",
        "source_iteration_last",
        "throughput_median",
        "throughput_p25",
        "throughput_p75",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (model, method, skew), key_points in sorted(points.items()):
            for point in key_points:
                writer.writerow(
                    {
                        "model": model,
                        "method": method,
                        "routing": "natural" if skew == "natural" else "synthetic_skew",
                        "router_skew": "" if skew == "natural" else skew,
                        "metric": metric,
                        "unit": METRIC_STYLE[metric]["unit"],
                        **{
                            key: round(float(value), 6)
                            if key.startswith("throughput_")
                            else value
                            for key, value in point.items()
                        },
                    }
                )


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_series = _load_series(
        input_dir,
        measured_tail=args.measured_tail,
        aligned_end=args.step_end,
        metric=args.metric,
        latest_stable_window=args.latest_stable_window,
        slow_factor=args.slow_factor,
    )
    centers = _key_steps(args.step_start, args.step_end, args.num_points)
    points = {
        key: _aggregate(
            rows,
            centers=centers,
            step_start=args.step_start,
            step_end=args.step_end,
            window=args.window,
        )
        for key, rows in raw_series.items()
    }

    output_stems = {
        model: _plot_model(
            model,
            points,
            output_dir=output_dir,
            step_start=args.step_start,
            step_end=args.step_end,
            metric=args.metric,
            dpi=args.dpi,
        )
        for model in MODEL_STYLE
    }
    points_path = output_dir / (
        f"e2e_throughput_keypoints_{METRIC_STYLE[args.metric]['suffix']}"
        f"_steps{args.step_start}_{args.step_end}.csv"
    )
    _write_points(points, points_path, metric=args.metric)

    source_kind = "measured" if args.measured_tail else "aligned"
    print(f"key steps: {centers}; median window: {args.window} {source_kind} samples")
    for model in MODEL_STYLE:
        stem = output_stems[model]
        print(f"saved {stem.with_suffix('.png')}")
        print(f"saved {stem.with_suffix('.pdf')}")
    print(f"saved {points_path}")


if __name__ == "__main__":
    main()

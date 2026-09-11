#!/usr/bin/env python3
"""Plot stable-tail E2E throughput against routing skew for three MoE models."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


MODEL_ORDER = ("qwen3_5L", "glm45_air_5moe", "deepseek_v2")
MODEL_LABELS = {
    "qwen3_5L": "Qwen3-30B-A3B",
    "glm45_air_5moe": "GLM-4.5-Air",
    "deepseek_v2": "DeepSeek-V2",
}
METHOD_ORDER = ("native", "scale_eplb")
METHOD_STYLES = {
    "native": {
        "label": "Megatron-LM",
        "color": "#555555",
        "linestyle": "--",
        "marker": "s",
    },
    "scale_eplb": {
        "label": "Scale-EPLB",
        "color": "#E15759",
        "linestyle": "-",
        "marker": "o",
    },
}
SKEW_ORDER = ("natural", "0.0", "-0.5", "-1.0", "-2.0", "-4.0")
SKEW_LABELS = ("Natural", r"$0$", r"$-0.5$", r"$-1$", r"$-2$", r"$-4$")
REQUIRED_COLUMNS = {
    "model",
    "method",
    "routing",
    "router_skew",
    "metric",
    "unit",
    "throughput_median",
}
SUMMARY_REQUIRED_COLUMNS = {
    "model",
    "method",
    "routing",
    "router_skew",
    "aggregation",
    "num_keypoints",
    "throughput_k_tokens_per_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdf-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="Aggregate applied to the stable-tail keypoint medians.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def skew_key(row: dict[str, str]) -> str:
    if row["routing"] == "natural":
        return "natural"
    return f"{float(row['router_skew']):.1f}"


def load_keypoints(path: Path) -> dict[tuple[str, str, str], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"{path}: empty CSV")

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in raw_rows:
        if row["metric"] != "tokens" or row["unit"] != "k tokens/s":
            raise ValueError(f"{path}: expected token throughput in k tokens/s")
        model = row["model"]
        method = row["method"]
        skew = skew_key(row)
        if model not in MODEL_ORDER:
            raise ValueError(f"{path}: unsupported model {model!r}")
        if method not in METHOD_ORDER:
            raise ValueError(f"{path}: unsupported method {method!r}")
        if skew not in SKEW_ORDER:
            raise ValueError(f"{path}: unsupported routing skew {skew!r}")
        grouped.setdefault((model, method, skew), []).append(
            float(row["throughput_median"])
        )
    return grouped


def load_summary(path: Path, *, aggregate: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = SUMMARY_REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"{path}: empty CSV")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in raw_rows:
        model = row["model"]
        method = row["method"]
        skew = skew_key(row)
        if model not in MODEL_ORDER:
            raise ValueError(f"{path}: unsupported model {model!r}")
        if method not in METHOD_ORDER:
            raise ValueError(f"{path}: unsupported method {method!r}")
        if skew not in SKEW_ORDER:
            raise ValueError(f"{path}: unsupported routing skew {skew!r}")
        if row["aggregation"] != aggregate:
            raise ValueError(
                f"{path}: contains {row['aggregation']!r} aggregation, "
                f"but --aggregate is {aggregate!r}"
            )
        key = (model, method, skew)
        if key in seen:
            raise ValueError(f"{path}: duplicate summary row for {key!r}")
        seen.add(key)
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_STYLES[method]["label"],
                "routing": "natural" if skew == "natural" else "synthetic_skew",
                "router_skew": "" if skew == "natural" else skew,
                "aggregation": aggregate,
                "num_keypoints": int(row["num_keypoints"]),
                "throughput_k_tokens_per_s": float(
                    row["throughput_k_tokens_per_s"]
                ),
            }
        )
    return rows


def aggregate_keypoints(
    grouped: dict[tuple[str, str, str], list[float]],
    *,
    aggregate: str,
) -> list[dict[str, Any]]:
    reducer = statistics.mean if aggregate == "mean" else statistics.median
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            for skew in SKEW_ORDER:
                values = grouped.get((model, method, skew))
                if not values:
                    continue
                rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "method": method,
                        "method_label": METHOD_STYLES[method]["label"],
                        "routing": "natural" if skew == "natural" else "synthetic_skew",
                        "router_skew": "" if skew == "natural" else skew,
                        "aggregation": aggregate,
                        "num_keypoints": len(values),
                        "throughput_k_tokens_per_s": reducer(values),
                    }
                )
    return rows


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def line_intersections(
    x_values: list[float],
    first_values: list[float],
    second_values: list[float],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    differences = [
        first - second for first, second in zip(first_values, second_values)
    ]
    for index, (left_difference, right_difference) in enumerate(
        zip(differences, differences[1:])
    ):
        if left_difference * right_difference >= 0:
            continue
        fraction = left_difference / (left_difference - right_difference)
        left_x, right_x = x_values[index : index + 2]
        left_y, right_y = first_values[index : index + 2]
        intersections.append(
            (
                left_x + fraction * (right_x - left_x),
                left_y + fraction * (right_y - left_y),
            )
        )
    return intersections


def plot(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    pdf_output: Path,
    aggregate: str,
    dpi: int,
) -> None:
    if dpi <= 0:
        raise ValueError("--dpi must be positive")
    values = {
        (row["model"], row["method"], skew_key(row)): float(
            row["throughput_k_tokens_per_s"]
        )
        for row in rows
    }

    figure, axes = plt.subplots(1, 3, figsize=(8.3, 2.45))
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.33,
        top=0.84,
        wspace=0.25,
    )
    handles: list[Any] = []
    synthetic_positions = [1.5 + index for index in range(len(SKEW_ORDER) - 1)]
    x_positions = [0.0, *synthetic_positions]

    for model_index, model in enumerate(MODEL_ORDER):
        axis = axes[model_index]
        present_values: list[float] = []
        synthetic_by_method: dict[str, list[float]] = {}
        axis.axvspan(
            -0.65,
            0.75,
            color="#F4F4F4",
            alpha=0.9,
            linewidth=0,
            zorder=0,
        )
        for method_index, method in enumerate(METHOD_ORDER):
            style = METHOD_STYLES[method]
            natural_key = (model, method, "natural")
            if natural_key in values:
                axis.bar(
                    -0.24 + 0.48 * method_index,
                    values[natural_key],
                    width=0.44,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
                present_values.append(values[natural_key])

            synthetic_x: list[float] = []
            synthetic_y: list[float] = []
            for position, skew in zip(synthetic_positions, SKEW_ORDER[1:]):
                key = (model, method, skew)
                if key in values:
                    synthetic_x.append(position)
                    synthetic_y.append(values[key])
            line = axis.plot(
                synthetic_x,
                synthetic_y,
                label=str(style["label"]),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.6,
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=4,
            )[0]
            present_values.extend(synthetic_y)
            synthetic_by_method[method] = synthetic_y
            if model_index == 0:
                handles.append(line)

        if all(
            len(synthetic_by_method.get(method, [])) == len(synthetic_positions)
            for method in METHOD_ORDER
        ):
            intersections = line_intersections(
                synthetic_positions,
                synthetic_by_method[METHOD_ORDER[0]],
                synthetic_by_method[METHOD_ORDER[1]],
            )
            if intersections:
                intersection_x, intersection_y = zip(*intersections)
                axis.scatter(
                    intersection_x,
                    intersection_y,
                    marker="*",
                    s=55,
                    color="#F2C14E",
                    edgecolor="#8A6D00",
                    linewidth=0.5,
                    zorder=6,
                )

        native_endpoint = values.get((model, "native", SKEW_ORDER[-1]))
        scale_endpoint = values.get((model, "scale_eplb", SKEW_ORDER[-1]))
        if (
            native_endpoint is not None
            and scale_endpoint is not None
            and native_endpoint > 0
        ):
            ratio_x = synthetic_positions[-1] + 0.58
            axis.annotate(
                "",
                xy=(ratio_x, scale_endpoint),
                xytext=(ratio_x, native_endpoint),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": "#F2C14E",
                    "linewidth": 1.5,
                    "mutation_scale": 8,
                },
                zorder=6,
            )
            ratio_label_y = (native_endpoint + scale_endpoint) / 2
            ratio_label_va = "center"
            if model == "deepseek_v2":
                ratio_label_y = max(native_endpoint, scale_endpoint) + 0.15 * abs(
                    scale_endpoint - native_endpoint
                )
                ratio_label_va = "bottom"
            axis.text(
                ratio_x - 0.09,
                ratio_label_y,
                f"{scale_endpoint / native_endpoint:.2f} $\\times$",
                ha="right",
                va=ratio_label_va,
                fontsize=7.5,
                color="#8A6D00",
                zorder=6,
            )

        axis.axvline(
            0.75,
            color="#888888",
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
            zorder=1,
        )
        lower = min(present_values)
        upper = max(present_values)
        padding = max(0.08 * (upper - lower), 0.8)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlim(-0.75, synthetic_positions[-1] + 0.95)
        axis.set_xticks(x_positions, SKEW_LABELS)
        axis.text(
            0.5,
            -0.20,
            "Routing policy",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
        )
        axis.text(
            0.5,
            -0.36,
            f"({chr(ord('a') + model_index)}) {MODEL_LABELS[model]}",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)
        axis.tick_params(axis="both", labelsize=7.5)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}k")
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.legend(
        handles,
        [str(METHOD_STYLES[method]["label"]) for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=2.8,
        columnspacing=1.8,
    )
    figure.text(
        0.003,
        0.57,
        "Average throughput (token/s)",
        rotation=90,
        va="center",
        fontsize=8.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pdf_output = (
        args.pdf_output.expanduser().resolve()
        if args.pdf_output
        else output.with_suffix(".pdf")
    )
    with input_csv.open(newline="", encoding="utf-8") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or [])
    if REQUIRED_COLUMNS.issubset(fieldnames):
        grouped = load_keypoints(input_csv)
        rows = aggregate_keypoints(grouped, aggregate=args.aggregate)
        load_message = (
            f"aggregated {sum(len(values) for values in grouped.values())} keypoints"
        )
    elif SUMMARY_REQUIRED_COLUMNS.issubset(fieldnames):
        rows = load_summary(input_csv, aggregate=args.aggregate)
        load_message = f"loaded {len(rows)} summary rows"
    else:
        expected = sorted(REQUIRED_COLUMNS | SUMMARY_REQUIRED_COLUMNS)
        raise ValueError(
            f"{input_csv}: unsupported CSV schema; expected keypoint or summary "
            f"columns from: {', '.join(expected)}"
        )
    plot(
        rows,
        output=output,
        pdf_output=pdf_output,
        aggregate=args.aggregate,
        dpi=args.dpi,
    )
    if args.summary_output:
        summary_output = args.summary_output.expanduser().resolve()
        write_summary(rows, summary_output)
        print(f"saved {summary_output}")
    print(load_message)
    print(f"saved {output}")
    print(f"saved {pdf_output}")


if __name__ == "__main__":
    main()
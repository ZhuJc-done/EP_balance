#!/usr/bin/env python3
"""Plot mean E2E throughput across load-balancing methods at router_skew=-4.0."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["hatch.linewidth"] = 0.7
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "logs"
    / "e2e_baseline_comparison_skew_m40"
    / "average_throughput_skew_m40.csv"
)
MODEL_ORDER = ("qwen3_5L", "glm45_air_5moe", "deepseek_v2")
METHOD_ORDER = (
    "megatron_lm",
    "fastermoe",
    "deepseek_eplb",
    "flexmoe",
    "scale_eplb",
)
METHOD_STYLE = {
    "megatron_lm": {
        "label": "Megatron-LM",
        "color": "#8C8C8C",
        "hatch": "",
    },
    "fastermoe": {
        "label": "FasterMoE",
        "color": "#4C78A8",
        "hatch": "///",
    },
    "deepseek_eplb": {
        "label": "DeepSeek-EPLB",
        "color": "#F2CF5B",
        "hatch": "\\\\\\",
    },
    "flexmoe": {
        "label": "FlexMoE",
        "color": "#59A14F",
        "hatch": "xx",
    },
    "scale_eplb": {
        "label": "Scale-EPLB",
        "color": "#E15759",
        "hatch": "..",
    },
}
REQUIRED_COLUMNS = {
    "model",
    "model_label",
    "method",
    "method_label",
    "router_skew",
    "mean_k_tokens_per_second",
    "std_k_tokens_per_second",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output (defaults beside the input CSV)",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="PDF output (defaults to the PNG path with a .pdf suffix)",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _load(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"{path}: empty CSV")

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_rows:
        model = raw["model"]
        method = raw["method"]
        if model not in MODEL_ORDER:
            raise ValueError(f"{path}: unknown model {model!r}")
        if method not in METHOD_ORDER:
            raise ValueError(f"{path}: unknown method {method!r}")
        if float(raw["router_skew"]) != -4.0:
            raise ValueError(f"{path}: expected router_skew=-4.0")
        if raw["method_label"] != METHOD_STYLE[method]["label"]:
            raise ValueError(
                f"{path}: inconsistent method label for {method}: "
                f"{raw['method_label']!r}"
            )

        key = (model, method)
        if key in rows:
            raise ValueError(f"{path}: duplicate row for {key}")
        mean = float(raw["mean_k_tokens_per_second"])
        standard_deviation = float(raw["std_k_tokens_per_second"])
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError(f"{path}: invalid mean throughput for {key}")
        if not math.isfinite(standard_deviation) or standard_deviation < 0:
            raise ValueError(f"{path}: invalid throughput standard deviation for {key}")
        rows[key] = {
            **raw,
            "mean": mean,
            "standard_deviation": standard_deviation,
        }

    expected = {(model, method) for model in MODEL_ORDER for method in METHOD_ORDER}
    missing_rows = expected.difference(rows)
    if missing_rows:
        raise ValueError(f"{path}: missing model/method rows: {sorted(missing_rows)}")
    return rows


def plot(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    output: Path,
    pdf_output: Path,
    dpi: int,
) -> None:
    if dpi <= 0:
        raise ValueError("--dpi must be positive")

    figure, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(8.1, 3.9))
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.18,
        top=0.77,
        wspace=0.28,
    )

    bar_step = 0.14
    bar_width = 0.105
    x_positions = [
        (index - (len(METHOD_ORDER) - 1) / 2) * bar_step
        for index in range(len(METHOD_ORDER))
    ]
    legend_handles = []

    for model_index, (axis, model) in enumerate(zip(axes, MODEL_ORDER)):
        values = [float(rows[(model, method)]["mean"]) for method in METHOD_ORDER]
        for method_index, (method, value) in enumerate(zip(METHOD_ORDER, values)):
            style = METHOD_STYLE[method]
            bars = axis.bar(
                [x_positions[method_index]],
                [value],
                width=bar_width,
                color=str(style["color"]),
                edgecolor="#333333",
                linewidth=0.65,
                hatch=str(style["hatch"]),
                zorder=3,
            )
            if model_index == 0:
                legend_handles.append(bars[0])
            axis.bar_label(
                bars,
                labels=[f"{value:.1f}"],
                padding=3,
                fontsize=9.0,
                rotation=90,
                color="#111111",
            )

        model_label = str(rows[(model, METHOD_ORDER[0])]["model_label"])
        axis.set_xlim(-0.48, 0.48)
        axis.set_ylim(0, max(values) * 1.18)
        axis.set_xticks([])
        axis.set_xlabel(
            model_label,
            fontsize=11.0,
            labelpad=8,
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.35, zorder=0)
        axis.tick_params(axis="y", labelsize=10.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.text(
        0.012,
        0.485,
        "Average throughput (k tokens/s)",
        rotation=90,
        va="center",
        fontsize=12.0,
    )
    figure.legend(
        legend_handles,
        [str(METHOD_STYLE[method]["label"]) for method in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=len(METHOD_ORDER),
        frameon=False,
        fontsize=9.8,
        handlelength=2.1,
        columnspacing=1.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output = args.output or input_csv.parent / "e2e_baseline_throughput_skew_m40.png"
    pdf_output = args.pdf_output or output.with_suffix(".pdf")
    rows = _load(input_csv)
    plot(
        rows,
        output=output.expanduser().resolve(),
        pdf_output=pdf_output.expanduser().resolve(),
        dpi=args.dpi,
    )
    print(f"loaded {len(rows)} model/method averages from {input_csv}")
    print(f"saved {output.expanduser().resolve()}")
    print(f"saved {pdf_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
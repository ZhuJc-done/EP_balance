#!/usr/bin/env python3
"""Plot average rank-load imbalance across load-balancing methods."""

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
    / "e2e_imbalance_comparison_skew_m40"
    / "average_imbalance_skew_m40.csv"
)
MODEL_ORDER = ("qwen3_5L", "glm45_air_5moe", "deepseek_v2")
METHOD_ORDER = (
    "megatron_lm",
    "fastermoe",
    "flexmoe",
    "deepseek_eplb",
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
    "mean_rank_max_mean",
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
        key = (model, method)
        if key in rows:
            raise ValueError(f"{path}: duplicate row for {key}")
        if raw["method_label"] != METHOD_STYLE[method]["label"]:
            raise ValueError(f"{path}: inconsistent label for {method}")

        mean = float(raw["mean_rank_max_mean"])
        if not math.isfinite(mean) or mean < 1.0:
            raise ValueError(f"{path}: invalid mean imbalance for {method}")
        rows[key] = {
            **raw,
            "mean": mean,
        }

    expected = {
        (model, method) for model in MODEL_ORDER for method in METHOD_ORDER
    }
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

    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    figure.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.80)
    group_width = 0.84
    bar_width = group_width / len(METHOD_ORDER)
    model_centers = list(range(len(MODEL_ORDER)))
    all_upper_bounds = []

    for method_index, method in enumerate(METHOD_ORDER):
        style = METHOD_STYLE[method]
        x_positions = [
            center - group_width / 2 + (method_index + 0.5) * bar_width
            for center in model_centers
        ]
        values = [float(rows[(model, method)]["mean"]) for model in MODEL_ORDER]
        all_upper_bounds.extend(values)
        bars = axis.bar(
            x_positions,
            values,
            width=bar_width * 0.88,
            label=str(style["label"]),
            color=str(style["color"]),
            edgecolor="#333333",
            linewidth=0.65,
            hatch=str(style["hatch"]),
            zorder=3,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in values],
            padding=3,
            fontsize=9.0,
            rotation=90,
            color="#111111",
        )

    axis.axhline(
        1.0,
        color="#555555",
        linestyle="--",
        linewidth=0.9,
        alpha=0.75,
        zorder=2,
    )
    model_labels = [
        str(rows[(model, METHOD_ORDER[0])]["model_label"])
        for model in MODEL_ORDER
    ]
    axis.set_xticks(model_centers, model_labels)
    axis.set_xlabel("Model", fontsize=11.5)
    axis.set_ylabel("Average rank-load imbalance (max / mean)", fontsize=11.5)
    axis.set_ylim(0, max(all_upper_bounds) * 1.20)
    axis.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.35, zorder=0)
    axis.tick_params(axis="both", labelsize=10.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
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
    output = args.output or input_csv.parent / "e2e_baseline_imbalance_skew_m40.png"
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

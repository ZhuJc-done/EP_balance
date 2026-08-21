#!/usr/bin/env python3
"""Plot rank-load imbalance versus the per-rank replica-slot budget."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_DIR = Path(os.environ.get("EPLB_EXP_DIR", REPO_ROOT / "logs"))
DEFAULT_GLOB = str(DEFAULT_EXP_DIR / "slot_sweep/*_seed0.json")
DEFAULT_OUTPUT = DEFAULT_EXP_DIR / "slot_sweep/slot_imbalance.png"

STRATEGY_ORDER = (
    "lplb",
    "deepseek-eplb",
    "fastermoe",
    "flexmoe",
    "scale-eplb",
)
STYLE = {
    "scale-eplb": {
        "label": "Scale-EPLB",
        "color": "#D62728",
        "marker": "o",
        "linewidth": 2.8,
    },
    "deepseek-eplb": {
        "label": "DeepSeek EPLB",
        "color": "#1F77B4",
        "marker": "s",
        "linewidth": 2.2,
    },
    "fastermoe": {
        "label": "FasterMoE",
        "color": "#2CA02C",
        "marker": "^",
        "linewidth": 2.2,
    },
    "flexmoe": {
        "label": "FlexMoE",
        "color": "#9467BD",
        "marker": "D",
        "linewidth": 2.2,
    },
    "lplb": {
        "label": "LPLB",
        "color": "#FF7F0E",
        "marker": "P",
        "linewidth": 2.2,
    },
}
NO_BALANCE_STYLE = {
    "label": "No balancing",
    "color": "#7F7F7F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_GLOB,
        help="Glob selecting benchmark JSON files (default: logs/slot_sweep/*_seed0.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="High-resolution PNG output",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="Vector PDF output (default: same path as --output with .pdf suffix)",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Print each imbalance value next to its marker",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title (empty by default)",
    )
    parser.add_argument(
        "--allow-missing-strategies",
        action="store_true",
        help="Plot available strategies instead of requiring every configured baseline",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list")
    return rows


def load_series(
    pattern: str,
    *,
    require_all_strategies: bool,
) -> tuple[dict[str, list[tuple[int, float]]], float]:
    paths = [Path(raw) for raw in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"no benchmark JSON files matched: {pattern}")

    values: dict[str, dict[int, float]] = defaultdict(dict)
    no_balance_values: list[float] = []
    for path in paths:
        for row in _load_rows(path):
            if not isinstance(row, dict) or "skipped" in row:
                continue
            strategy = str(row["strategy"])
            if strategy == "no-balance":
                no_balance_values.append(float(row["quality_imbalance"]))
                continue
            if strategy not in STYLE:
                continue
            slot = int(row["replica_slots_per_rank"])
            imbalance = float(row["quality_imbalance"])
            if not math.isfinite(imbalance) or imbalance < 1.0:
                raise ValueError(
                    f"{path}: invalid quality_imbalance={imbalance} "
                    f"for {strategy}, slot={slot}"
                )
            if slot in values[strategy]:
                raise ValueError(
                    f"duplicate {strategy}, slot={slot}; narrow --input-glob to one seed"
                )
            values[strategy][slot] = imbalance

    missing = [strategy for strategy in STRATEGY_ORDER if strategy not in values]
    if missing and require_all_strategies:
        raise ValueError(f"missing strategy data: {missing}")
    if not values:
        raise ValueError("no recognized strategy data found")
    if not no_balance_values:
        raise ValueError("missing no-balance reference; rerun the slot sweep")
    no_balance = no_balance_values[0]
    if any(
        not math.isclose(value, no_balance, rel_tol=1e-9, abs_tol=1e-9)
        for value in no_balance_values[1:]
    ):
        raise ValueError("no-balance reference differs across slot files")
    series = {
        strategy: sorted(slot_values.items())
        for strategy, slot_values in values.items()
    }
    return series, no_balance


def plot(
    series: dict[str, list[tuple[int, float]]],
    *,
    no_balance: float,
    output: Path,
    pdf_output: Path,
    title: str,
    dpi: int,
    annotate: bool,
) -> None:
    if dpi <= 0:
        raise ValueError("--dpi must be positive")

    figure, axis = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
    all_slots = sorted({slot for points in series.values() for slot, _ in points})
    active_strategies = [
        strategy for strategy in STRATEGY_ORDER if strategy in series
    ]
    slot_positions = {
        slot: index + 1 for index, slot in enumerate(all_slots)
    }
    bar_width = 0.15
    num_strategies = len(active_strategies)
    no_balance_position = 0.4

    no_balance_bar = axis.bar(
        [no_balance_position],
        [no_balance],
        width=bar_width,
        label=NO_BALANCE_STYLE["label"],
        color=NO_BALANCE_STYLE["color"],
        edgecolor="white",
        linewidth=0.7,
    )
    bar = no_balance_bar[0]
    axis.annotate(
        f"{no_balance:.2f}",
        (bar.get_x() + bar.get_width() / 2, no_balance),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=NO_BALANCE_STYLE["color"],
    )

    for strategy_index, strategy in enumerate(active_strategies):
        points = series[strategy]
        offset = (strategy_index - (num_strategies - 1) / 2) * bar_width
        positions = [slot_positions[slot] + offset for slot, _ in points]
        imbalance = [value for _, value in points]
        style = STYLE[strategy]
        bars = axis.bar(
            positions,
            imbalance,
            width=bar_width,
            label=style["label"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
        )
        if annotate:
            for bar, (_, value) in zip(bars, points):
                axis.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=9,
                    color=style["color"],
                )

    axis.set_xticks(
        [no_balance_position, *range(1, len(all_slots) + 1)],
        ["No balance", *[str(slot) for slot in all_slots]],
    )
    axis.set_xlabel(r"Replica slots per rank ($N_{\mathrm{slot}}$)", fontsize=13)
    axis.set_ylabel("Rank-load imbalance (max / mean)", fontsize=13)
    if title:
        axis.set_title(title, fontsize=14, pad=12)
    axis.axhline(1.0, color="#666666", linestyle="--", linewidth=1.2, label="Ideal")
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.tick_params(axis="both", labelsize=11)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )

    output = output.expanduser().resolve()
    pdf_output = pdf_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    series, no_balance = load_series(
        args.input_glob,
        require_all_strategies=not args.allow_missing_strategies,
    )
    pdf_output = args.pdf_output or args.output.with_suffix(".pdf")
    plot(
        series,
        no_balance=no_balance,
        output=args.output,
        pdf_output=pdf_output,
        title=args.title,
        dpi=args.dpi,
        annotate=args.annotate,
    )
    print(f"No balancing: {no_balance:.4f}")
    for strategy in STRATEGY_ORDER:
        if strategy not in series:
            continue
        values = ", ".join(
            f"slot {slot}={imbalance:.4f}"
            for slot, imbalance in series[strategy]
        )
        print(f"{STYLE[strategy]['label']}: {values}")
    print(f"saved PNG to {args.output.expanduser().resolve()}")
    print(f"saved PDF to {pdf_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()

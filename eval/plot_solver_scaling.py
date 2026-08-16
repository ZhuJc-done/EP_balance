#!/usr/bin/env python3
"""Plot the minimum observed fast_solver.cu latency versus ranks and experts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "logs/solver_scaling"
COLOR = "#D62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory produced by scripts/run_solver_scaling.sh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="High-resolution PNG (default: INPUT_DIR/solver_scaling.png)",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        help="Vector PDF (default: same path as --output with .pdf suffix)",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--y-scale",
        choices=("linear", "log"),
        default="linear",
        help="Latency-axis scale",
    )
    return parser.parse_args()


def _read_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if not isinstance(report.get("config"), dict):
        raise ValueError(f"{path}: missing config")
    if not isinstance(report.get("kernel_only"), dict):
        raise ValueError(f"{path}: missing kernel_only statistics")
    return report


def _load_sweep(input_dir: Path, prefix: str) -> list[dict[str, float]]:
    paths = sorted(input_dir.glob(f"{prefix}_r*_e*.json"))
    if not paths:
        raise FileNotFoundError(f"no {prefix} JSON files found in {input_dir}")

    points = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        report = _read_report(path)
        config = report["config"]
        stats = report["kernel_only"]
        ranks = int(config["logical_ranks"])
        experts = int(config["experts"])
        key = (ranks, experts)
        if key in seen:
            raise ValueError(f"duplicate configuration R={ranks}, E={experts}")
        seen.add(key)

        point = {
            "ranks": ranks,
            "experts": experts,
            "mean_us": float(stats["mean_us"]),
            "p50_us": float(stats["p50_us"]),
            "p95_us": float(stats["p95_us"]),
            "min_us": float(stats["min_us"]),
            "max_us": float(stats["max_us"]),
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0
            for name, value in point.items()
            if name.endswith("_us")
        ):
            raise ValueError(f"{path}: latency values must be finite and positive")
        points.append(point)

    sort_key = "ranks" if prefix == "rank_scale" else "experts"
    return sorted(points, key=lambda point: point[sort_key])


def _draw_panel(
    axis: plt.Axes,
    points: list[dict[str, float]],
    *,
    x_key: str,
    x_label: str,
    title: str,
    y_scale: str,
) -> None:
    x = [int(point[x_key]) for point in points]
    minimum = [point["min_us"] for point in points]

    axis.plot(
        x,
        minimum,
        color=COLOR,
        marker="o",
        linewidth=2.4,
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    for index, (x_value, latency) in enumerate(zip(x, minimum)):
        annotate_above = index % 2 == 0
        label = (
            f"{latency / 1_000:.2f} ms"
            if latency >= 1_000
            else f"{latency:.0f} µs"
        )
        axis.annotate(
            label,
            (x_value, latency),
            xytext=(0, 9 if annotate_above else -11),
            textcoords="offset points",
            ha="center",
            va="bottom" if annotate_above else "top",
            fontsize=7,
            color=COLOR,
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale(y_scale)
    axis.set_xticks(x, [str(value) for value in x])
    axis.margins(y=0.18)
    axis.set_xlabel(x_label, fontsize=16)
    axis.text(
        0.5,
        -0.24,
        title,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=16,
    )
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.tick_params(axis="both", labelsize=10)
    axis.tick_params(axis="x", labelrotation=30)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot(
    rank_points: list[dict[str, float]],
    expert_points: list[dict[str, float]],
    *,
    output: Path,
    pdf_output: Path,
    dpi: int,
    y_scale: str,
) -> None:
    if dpi <= 0:
        raise ValueError("--dpi must be positive")

    rank_experts = {int(point["experts"]) for point in rank_points}
    expert_ranks = {int(point["ranks"]) for point in expert_points}
    if len(rank_experts) != 1:
        raise ValueError("rank sweep must hold the expert count constant")
    if len(expert_ranks) != 1:
        raise ValueError("expert sweep must hold the rank count constant")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.45),
        constrained_layout=True,
        sharey=True,
    )
    _draw_panel(
        axes[0],
        rank_points,
        x_key="ranks",
        x_label="EP ranks",
        title=f"(a) Rank scaling (E={next(iter(rank_experts))})",
        y_scale=y_scale,
    )
    _draw_panel(
        axes[1],
        expert_points,
        x_key="experts",
        x_label="Expert nums",
        title=f"(b) Expert scaling (R={next(iter(expert_ranks))})",
        y_scale=y_scale,
    )
    axes[0].set_ylabel("kernel solve latency (µs)", fontsize=16)

    output = output.expanduser().resolve()
    pdf_output = pdf_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_output, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output = args.output or input_dir / "solver_scaling.png"
    pdf_output = args.pdf_output or output.with_suffix(".pdf")
    rank_points = _load_sweep(input_dir, "rank_scale")
    expert_points = _load_sweep(input_dir, "expert_scale")
    plot(
        rank_points,
        expert_points,
        output=output,
        pdf_output=pdf_output,
        dpi=args.dpi,
        y_scale=args.y_scale,
    )

    for label, points, x_key in (
        ("rank", rank_points, "ranks"),
        ("expert", expert_points, "experts"),
    ):
        values = ", ".join(
            f"{x_key[0].upper()}={int(point[x_key])}: "
            f"min={point['min_us']:.2f}us"
            for point in points
        )
        print(f"{label} sweep: {values}")
    print(f"saved PNG to {output.expanduser().resolve()}")
    print(f"saved PDF to {pdf_output.expanduser().resolve()}")


if __name__ == "__main__":
    main()

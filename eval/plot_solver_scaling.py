#!/usr/bin/env python3
"""Plot the minimum observed fast_solver.cu latency versus ranks and experts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_DIR = Path(os.environ.get("EPLB_EXP_DIR", REPO_ROOT / "logs"))
DEFAULT_INPUT_DIR = DEFAULT_EXP_DIR / "solver_scaling"
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
        "--input-csv",
        type=Path,
        help="Saved solver-scaling plot data; overrides --input-dir",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        help="Optional CSV output for the loaded plotting values",
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


def _load_csv(
    path: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    required = {
        "sweep",
        "ranks",
        "experts",
        "mean_us",
        "p50_us",
        "p95_us",
        "min_us",
        "max_us",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing plot-data columns: {', '.join(sorted(missing))}"
            )
        grouped: dict[str, list[dict[str, float]]] = {
            "rank": [],
            "expert": [],
        }
        for row in reader:
            sweep = row["sweep"]
            if sweep not in grouped:
                raise ValueError(f"{path}: unsupported sweep {sweep!r}")
            grouped[sweep].append(
                {
                    "ranks": int(row["ranks"]),
                    "experts": int(row["experts"]),
                    "mean_us": float(row["mean_us"]),
                    "p50_us": float(row["p50_us"]),
                    "p95_us": float(row["p95_us"]),
                    "min_us": float(row["min_us"]),
                    "max_us": float(row["max_us"]),
                }
            )
    if not grouped["rank"] or not grouped["expert"]:
        raise ValueError(f"{path}: both rank and expert sweeps are required")
    return (
        sorted(grouped["rank"], key=lambda point: point["ranks"]),
        sorted(grouped["expert"], key=lambda point: point["experts"]),
    )


def _write_csv(
    path: Path,
    rank_points: list[dict[str, float]],
    expert_points: list[dict[str, float]],
) -> None:
    rows = [
        {"sweep": sweep, **point}
        for sweep, points in (("rank", rank_points), ("expert", expert_points))
        for point in points
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    annotation_layouts = {
        "ranks": (
            (6, -4, "left"),
            (0, 12, "center"),
            (0, 12, "center"),
            (0, -14, "center"),
            (-6, 12, "right"),
        ),
        "experts": (
            (10, 12, "left"),
            (0, -15, "center"),
            (0, 12, "center"),
            (0, -15, "center"),
            (0, -10, "left"),
            (-3, 2, "right"),
            (4, -2, "left"),
            (0, 4, "center"),
        ),
    }
    annotation_layout = annotation_layouts[x_key]
    for index, (x_value, latency) in enumerate(zip(x, minimum)):
        offset_x, offset_y, horizontal_alignment = annotation_layout[index]
        label = f"{latency:.0f} us"
        axis.annotate(
            label,
            (x_value, latency),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="bottom" if offset_y > 0 else "top",
            fontsize=10.5,
            color=COLOR,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.5,
            },
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale(y_scale)
    axis.set_xticks(x, [str(value) for value in x])
    axis.margins(y=0.18)
    axis.set_xlabel(x_label, fontsize=14.5)
    axis.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.tick_params(axis="both", labelsize=11.5)
    axis.tick_params(axis="x", labelrotation=30)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.5,
        -0.34,
        title,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=14.5,
    )


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
        figsize=(9.2, 4.2),
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.29,
        top=0.96,
        wspace=0.06,
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
    axes[0].set_ylabel("Kernel solve latency (us)", fontsize=14.5)

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
    output_dir = (
        args.input_csv.expanduser().resolve().parent
        if args.input_csv is not None
        else input_dir
    )
    output = args.output or output_dir / "solver_scaling.png"
    pdf_output = args.pdf_output or output.with_suffix(".pdf")
    if args.input_csv is not None:
        rank_points, expert_points = _load_csv(
            args.input_csv.expanduser().resolve()
        )
    else:
        rank_points = _load_sweep(input_dir, "rank_scale")
        expert_points = _load_sweep(input_dir, "expert_scale")
    if args.data_output is not None:
        data_output = args.data_output.expanduser().resolve()
        _write_csv(data_output, rank_points, expert_points)
        print(f"saved plot data to {data_output}")
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

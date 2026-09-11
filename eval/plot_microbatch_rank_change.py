#!/usr/bin/env python3
"""Plot adjacent-microbatch changes in fixed-placement rank-load distributions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

from eplb.trace_analysis import (  # noqa: E402
    expert_count_cube,
    load_routing_trace,
    rank_loads_from_expert_counts,
)


def _trace_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(value).stem
    path = Path(raw_path).expanduser().resolve()
    if not label.strip():
        raise argparse.ArgumentTypeError("trace label cannot be empty")
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
    )
    parser.add_argument("--target-ranks", type=int, default=32)
    parser.add_argument("--occurrence-group", type=int, default=8)
    parser.add_argument("--max-occurrences", type=int, default=0)
    parser.add_argument("--output", default="microbatch_rank_change.pdf")
    parser.add_argument("--csv", dest="csv_path")
    parser.add_argument("--synthetic-placeholder", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _adjacent_changes(
    trace: dict,
    *,
    target_ranks: int,
    occurrence_group: int,
    max_occurrences: int,
) -> tuple[list[int], torch.Tensor]:
    layers, expert_counts = expert_count_cube(
        trace,
        max_occurrences=max_occurrences,
    )
    num_layers, occurrences, num_experts = expert_counts.shape
    if num_experts % target_ranks:
        raise ValueError(
            f"{num_experts} experts cannot be divided evenly over {target_ranks} ranks"
        )
    groups = occurrences // occurrence_group
    if groups < 2:
        raise ValueError(
            f"need at least two groups of {occurrence_group} occurrences, got {occurrences}"
        )
    expert_counts = expert_counts[:, : groups * occurrence_group].reshape(
        num_layers,
        groups,
        occurrence_group,
        num_experts,
    ).sum(dim=2)

    experts_per_rank = num_experts // target_ranks
    placement_meta = {
        "num_ranks": target_ranks,
        "num_experts": num_experts,
        "main_rank": torch.arange(num_experts, dtype=torch.int64)
        // experts_per_rank,
    }
    rank_loads = rank_loads_from_expert_counts(
        expert_counts.reshape(num_layers * groups, num_experts),
        placement_meta,
    ).reshape(num_layers, groups, target_ranks)
    shares = rank_loads.to(torch.float64)
    shares /= shares.sum(dim=-1, keepdim=True).clamp_min(1)
    # Total-variation distance lies in [0, 1]. Zero means identical normalized
    # rank loads; one means that consecutive loads have disjoint support.
    changes = 0.5 * (shares[:, 1:] - shares[:, :-1]).abs().sum(dim=-1)
    return layers, changes


def main() -> None:
    args = parse_args()
    if args.target_ranks <= 0:
        raise ValueError("--target-ranks must be positive")
    if args.occurrence_group <= 0:
        raise ValueError("--occurrence-group must be positive")
    if args.max_occurrences < 0:
        raise ValueError("--max-occurrences must be non-negative")

    summaries = []
    rows = []
    for label, path in args.trace:
        trace = load_routing_trace(path)
        layers, changes = _adjacent_changes(
            trace,
            target_ranks=args.target_ranks,
            occurrence_group=args.occurrence_group,
            max_occurrences=args.max_occurrences,
        )
        quantiles = torch.quantile(
            changes,
            torch.tensor([0.1, 0.9], dtype=changes.dtype),
            dim=1,
        )
        summary = {
            "label": label,
            "layers": layers,
            "mean": changes.mean(dim=1).numpy(),
            "p10": quantiles[0].numpy(),
            "p90": quantiles[1].numpy(),
            "pairs": changes.shape[1],
        }
        summaries.append(summary)
        for index, layer in enumerate(layers):
            rows.append(
                {
                    "label": label,
                    "layer": layer + 1,
                    "mean_adjacent_tv": float(summary["mean"][index]),
                    "p10_adjacent_tv": float(summary["p10"][index]),
                    "p90_adjacent_tv": float(summary["p90"][index]),
                    "adjacent_pairs": summary["pairs"],
                }
            )

    matplotlib.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    figure, axis = plt.subplots(figsize=(6.4, 2.9), constrained_layout=True)
    for summary in summaries:
        x_values = np.asarray(summary["layers"]) + 1
        (line,) = axis.plot(
            x_values,
            summary["mean"],
            linewidth=2.0,
            label=summary["label"],
        )
        axis.fill_between(
            x_values,
            summary["p10"],
            summary["p90"],
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )

    axis.set_title("(d) Adjacent-microbatch rank-load variation")
    axis.set_xlabel("MoE layer")
    axis.set_ylabel("Rank-load change (TV distance)")
    axis.set_xlim(1, max(max(summary["layers"]) for summary in summaries) + 1)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=max(1, len(summaries)))
    if args.synthetic_placeholder:
        axis.text(
            0.99,
            0.03,
            "SYNTHETIC PLACEHOLDER",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#9b1c1c",
            fontsize=9,
            fontweight="bold",
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    csv_path = (
        Path(args.csv_path).expanduser().resolve()
        if args.csv_path
        else output.with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[plot_microbatch_rank_change] image: {output}")
    print(f"[plot_microbatch_rank_change] metrics: {csv_path}")
    for summary in summaries:
        print(
            f"[plot_microbatch_rank_change] {summary['label']}: "
            f"mean adjacent TV={summary['mean'].mean():.3f}, "
            f"pairs/layer={summary['pairs']}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Flatten slot-sweep and solver-scaling JSON reports into analysis-friendly CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SLOT_FILE_RE = re.compile(
    r"^baseline_skew(?P<skew>[^_]+)_slot(?P<slot>\d+)_seed(?P<seed>[^.]+)\.json$"
)

SLOT_COLUMNS = (
    "source_file",
    "seed",
    "skew",
    "requested_n_slot",
    "strategy",
    "skipped",
    "solve_ms",
    "placement_ms",
    "routing_ms",
    "total_ms",
    "quality_theta",
    "quality_mean_load",
    "quality_imbalance",
    "load_kind",
    "placement_kind",
    "main_slots_per_rank",
    "replica_slots_per_rank",
    "physical_slots_per_rank",
    "replica_budget",
    "physical_instances",
    "replicas",
    "num_groups",
    "shadowed_experts",
    "balance_ratio",
)

SLOT_SUMMARY_COLUMNS = (
    "strategy",
    "skew",
    "requested_n_slot",
    "runs",
    "seeds",
    "mean_quality_imbalance",
    "min_quality_imbalance",
    "max_quality_imbalance",
    "mean_quality_theta",
    "mean_solve_ms",
    "min_solve_ms",
    "max_solve_ms",
)

SOLVER_COLUMNS = (
    "source_file",
    "sweep",
    "solver",
    "nodes",
    "gpus_per_node",
    "logical_ranks",
    "experts",
    "tokens_per_rank",
    "top_k",
    "total_routes",
    "main_slots_per_rank",
    "extra_slots_per_rank",
    "total_slots_per_rank",
    "cross_domain",
    "skew",
    "hotspot_ranks",
    "max_stage2_iters",
    "max_fast_stage2_iters",
    "stage2_stagnation_patience",
    "stage2_patience_all_scales",
    "stage2_blocks",
    "stage2_budget_scope",
    "jit_compile_ms",
    "kernel_mean_us",
    "kernel_p50_us",
    "kernel_p95_us",
    "kernel_min_us",
    "kernel_max_us",
    "theta",
    "replicas",
    "routes",
    "mean_load",
    "baseline_theta",
    "baseline_imbalance",
    "solved_theta",
    "solved_imbalance",
    "theta_reduction_percent",
    "balance_speedup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("slot-sweep", "solver-scaling"),
        required=True,
        help="JSON report layout to flatten",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional aggregate CSV; supported only for --kind slot-sweep",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> int:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in writer.fieldnames})
            count += 1
    temporary.replace(path)
    return count


def _slot_rows(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("baseline_*.json"))
    if not paths:
        raise FileNotFoundError(f"no slot-sweep baseline JSON files found in {input_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        report = _read_json(path)
        if not isinstance(report, list):
            raise ValueError(f"{path}: expected a JSON list")
        match = SLOT_FILE_RE.match(path.name)
        filename_fields = match.groupdict() if match else {}
        for entry in report:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: expected every report entry to be an object")
            rows.append(
                {
                    "source_file": path.name,
                    "seed": filename_fields.get("seed", ""),
                    "skew": filename_fields.get("skew", ""),
                    "requested_n_slot": filename_fields.get("slot", ""),
                    **{column: entry.get(column) for column in SLOT_COLUMNS if column in entry},
                }
            )

    def order(row: dict[str, Any]) -> tuple[float, float, str]:
        def numeric(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return math.inf

        return numeric(row["seed"]), numeric(row["requested_n_slot"]), str(row.get("strategy", ""))

    return sorted(rows, key=order)


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _slot_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("skipped"):
            continue
        grouped[
            (
                str(row.get("strategy", "")),
                str(row.get("skew", "")),
                str(row.get("requested_n_slot", "")),
            )
        ].append(row)

    summary = []
    for (strategy, skew, slot), group in sorted(grouped.items()):
        imbalance = _finite_values(group, "quality_imbalance")
        theta = _finite_values(group, "quality_theta")
        solve = _finite_values(group, "solve_ms")
        summary.append(
            {
                "strategy": strategy,
                "skew": skew,
                "requested_n_slot": slot,
                "runs": len(group),
                "seeds": ",".join(sorted({str(row.get("seed", "")) for row in group})),
                "mean_quality_imbalance": statistics.fmean(imbalance) if imbalance else None,
                "min_quality_imbalance": min(imbalance) if imbalance else None,
                "max_quality_imbalance": max(imbalance) if imbalance else None,
                "mean_quality_theta": statistics.fmean(theta) if theta else None,
                "mean_solve_ms": statistics.fmean(solve) if solve else None,
                "min_solve_ms": min(solve) if solve else None,
                "max_solve_ms": max(solve) if solve else None,
            }
        )
    return summary


def _solver_rows(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("rank_scale_*.json")) + sorted(
        input_dir.glob("expert_scale_*.json")
    )
    if not paths:
        raise FileNotFoundError(f"no solver-scaling JSON files found in {input_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        report = _read_json(path)
        if not isinstance(report, dict):
            raise ValueError(f"{path}: expected a JSON object")
        config = report.get("config")
        kernel = report.get("kernel_only")
        result = report.get("result")
        quality = report.get("quality")
        if not all(isinstance(section, dict) for section in (config, kernel, result, quality)):
            raise ValueError(f"{path}: missing config, kernel_only, result, or quality report")
        sweep = "rank_scale" if path.name.startswith("rank_scale_") else "expert_scale"
        rows.append(
            {
                "source_file": path.name,
                "sweep": sweep,
                **{column: config.get(column) for column in SOLVER_COLUMNS if column in config},
                "jit_compile_ms": report.get("jit_compile_ms"),
                "kernel_mean_us": kernel.get("mean_us"),
                "kernel_p50_us": kernel.get("p50_us"),
                "kernel_p95_us": kernel.get("p95_us"),
                "kernel_min_us": kernel.get("min_us"),
                "kernel_max_us": kernel.get("max_us"),
                "theta": result.get("theta"),
                "replicas": result.get("replicas"),
                "routes": result.get("routes"),
                "mean_load": quality.get("mean_load"),
                "baseline_theta": quality.get("baseline_theta"),
                "baseline_imbalance": quality.get("baseline_imbalance"),
                "solved_theta": quality.get("solved_theta"),
                "solved_imbalance": quality.get("solved_imbalance"),
                "theta_reduction_percent": quality.get("theta_reduction_percent"),
                "balance_speedup": quality.get("balance_speedup"),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            str(row["sweep"]),
            int(row["logical_ranks"]),
            int(row["experts"]),
        ),
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    if args.kind == "slot-sweep":
        rows = _slot_rows(input_dir)
        count = _write_csv(args.output, SLOT_COLUMNS, rows)
        print(f"wrote {count} slot-sweep rows to {args.output.expanduser().resolve()}")
        if args.summary_output:
            summary = _slot_summary(rows)
            summary_count = _write_csv(args.summary_output, SLOT_SUMMARY_COLUMNS, summary)
            print(
                f"wrote {summary_count} slot-sweep summary rows to "
                f"{args.summary_output.expanduser().resolve()}"
            )
    else:
        if args.summary_output:
            raise ValueError("--summary-output is supported only for --kind slot-sweep")
        rows = _solver_rows(input_dir)
        count = _write_csv(args.output, SOLVER_COLUMNS, rows)
        print(f"wrote {count} solver-scaling rows to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()

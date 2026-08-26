#!/usr/bin/env python3
"""Analyze actual token rerouting captured from ``EPLB_MODE=apply``.

The input is an ``EPLB_TRACE_OUT`` v4 file containing the raw router matrix
``omega[src, expert]`` and the applied quota ``q[src, expert, dst]``.  The
report distinguishes local execution, remote execution inside one NVLink
domain, and cross-domain execution.  It also compares the applied plan with
Megatron's original home-expert placement.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from eplb.routing_stats import compute_routing_stats
from eplb.trace_analysis import load_routing_trace, resolve_main_rank

MIB = 1024**2
COUNT_KEYS = (
    "total_assignments",
    "baseline_local_assignments",
    "baseline_intra_domain_assignments",
    "baseline_inter_domain_assignments",
    "planned_local_assignments",
    "planned_intra_domain_assignments",
    "planned_inter_domain_assignments",
    "rerouted_assignments",
    "rerouted_local_assignments",
    "rerouted_intra_domain_assignments",
    "rerouted_inter_domain_assignments",
    "avoided_inter_domain_assignments",
    "remaining_inter_domain_assignments",
    "introduced_inter_domain_assignments",
)
BYTE_KEYS = (
    "baseline_one_way_remote_bytes",
    "baseline_one_way_inter_domain_bytes",
    "planned_one_way_remote_bytes",
    "planned_one_way_inter_domain_bytes",
    "baseline_training_remote_bytes",
    "baseline_training_inter_domain_bytes",
    "planned_training_remote_bytes",
    "planned_training_inter_domain_bytes",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    opener = gzip.open if path.suffix == ".gz" else open
    kwargs = {"encoding": "utf-8", "newline": ""}
    with opener(path, "wt", **kwargs) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return st.median(values) if values else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(int(row[key]) for row in rows) for key in COUNT_KEYS}
    byte_totals = {key: sum(int(row[key]) for row in rows) for key in BYTE_KEYS}
    total = counts["total_assignments"]
    baseline_inter = counts["baseline_inter_domain_assignments"]
    planned_inter = counts["planned_inter_domain_assignments"]
    return {
        "samples": len(rows),
        **counts,
        **byte_totals,
        "rerouted_fraction": counts["rerouted_assignments"] / total if total else 0.0,
        "baseline_inter_domain_fraction": baseline_inter / total if total else 0.0,
        "planned_inter_domain_fraction": planned_inter / total if total else 0.0,
        "inter_domain_reduction_fraction": (
            (baseline_inter - planned_inter) / baseline_inter
            if baseline_inter
            else 0.0
        ),
        "baseline_rank_max_mean_median": _median(
            rows, "baseline_rank_max_mean"
        ),
        "planned_rank_max_mean_median": _median(
            rows, "planned_rank_max_mean"
        ),
        "absorbed_excess_load_median": _median(
            rows, "absorbed_excess_load"
        ),
        "topology_cost_reduction_fraction_median": _median(
            rows, "topology_cost_reduction_fraction"
        ),
        "planned_one_way_remote_mib_per_sample": (
            byte_totals["planned_one_way_remote_bytes"] / MIB / len(rows)
            if rows
            else 0.0
        ),
        "planned_one_way_inter_domain_mib_per_sample": (
            byte_totals["planned_one_way_inter_domain_bytes"] / MIB / len(rows)
            if rows
            else 0.0
        ),
        "planned_training_remote_mib_per_sample": (
            byte_totals["planned_training_remote_bytes"] / MIB / len(rows)
            if rows
            else 0.0
        ),
        "planned_training_inter_domain_mib_per_sample": (
            byte_totals["planned_training_inter_domain_bytes"] / MIB / len(rows)
            if rows
            else 0.0
        ),
    }


def _flow_rows(
    *,
    baseline: torch.Tensor,
    planned: torch.Tensor,
    domain_of_rank: torch.Tensor,
    s_tok: int,
) -> list[dict[str, Any]]:
    rows = []
    for route, flow in (("home", baseline), ("planned", planned)):
        for source in range(flow.shape[0]):
            for destination in range(flow.shape[1]):
                assignments = int(flow[source, destination])
                if not assignments:
                    continue
                source_domain = int(domain_of_rank[source])
                destination_domain = int(domain_of_rank[destination])
                locality = (
                    "local"
                    if source == destination
                    else "intra_domain"
                    if source_domain == destination_domain
                    else "inter_domain"
                )
                rows.append(
                    {
                        "route": route,
                        "source_rank": source,
                        "destination_rank": destination,
                        "source_domain": source_domain,
                        "destination_domain": destination_domain,
                        "locality": locality,
                        "assignments": assignments,
                        "one_way_mib": round(assignments * s_tok / MIB, 6),
                        "training_mib": round(4 * assignments * s_tok / MIB, 6),
                    }
                )
    return rows


def _domain_flow_rows(
    baseline: torch.Tensor, planned: torch.Tensor, s_tok: int
) -> list[dict[str, Any]]:
    rows = []
    for route, flow in (("home", baseline), ("planned", planned)):
        for source in range(flow.shape[0]):
            for destination in range(flow.shape[1]):
                assignments = int(flow[source, destination])
                if not assignments:
                    continue
                rows.append(
                    {
                        "route": route,
                        "source_domain": source,
                        "destination_domain": destination,
                        "locality": (
                            "intra_domain" if source == destination else "inter_domain"
                        ),
                        "assignments": assignments,
                        "one_way_mib": round(assignments * s_tok / MIB, 6),
                        "training_mib": round(4 * assignments * s_tok / MIB, 6),
                    }
                )
    return rows


def _read_traces(paths: list[Path]) -> tuple[dict, list[tuple[dict, dict]]]:
    traces = [load_routing_trace(path) for path in paths]
    reference = traces[0]["meta"]
    for path, trace in zip(paths, traces):
        meta = trace["meta"]
        for key in ("num_ranks", "num_experts", "s_tok", "n_slot"):
            if int(meta[key]) != int(reference[key]):
                raise ValueError(f"{path}: incompatible {key}")
        if meta.get("plan_solver") != reference.get("plan_solver"):
            raise ValueError(f"{path}: incompatible plan_solver")
        for key in ("main_rank", "domain_of_rank", "cost"):
            if not torch.equal(
                torch.as_tensor(meta[key]), torch.as_tensor(reference[key])
            ):
                raise ValueError(f"{path}: incompatible {key}")
    records = []
    for path, trace in zip(paths, traces):
        meta = trace["meta"]
        if meta.get("trace_kind") != "applied_plan":
            raise ValueError(
                f"{path}: expected an apply-mode plan trace; "
                "set EPLB_MODE=apply and EPLB_TRACE_OUT"
            )
        for index, sample in enumerate(trace["samples"]):
            if "q" not in sample or "x" not in sample:
                raise ValueError(f"{path}: sample {index} has no applied q/x plan")
            records.append((meta, sample))
    return reference, records


def analyze(paths: list[Path], warmup_mb: int) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    meta, records = _read_traces(paths)
    selected = [
        (record_meta, sample)
        for record_meta, sample in records
        if int(sample["mb"]) >= warmup_mb
    ]
    if not selected:
        raise ValueError(
            f"no samples remain after --warmup-mb {warmup_mb}"
        )

    main_rank = resolve_main_rank(meta)
    domains = torch.as_tensor(meta["domain_of_rank"], dtype=torch.int64)
    cost = torch.as_tensor(meta["cost"], dtype=torch.int64)
    s_tok = int(meta["s_tok"])
    sample_rows = []
    baseline_rank_flow = torch.zeros(
        (int(meta["num_ranks"]), int(meta["num_ranks"])), dtype=torch.int64
    )
    planned_rank_flow = torch.zeros_like(baseline_rank_flow)
    baseline_domain_flow = torch.zeros(
        (int(meta["num_domains"]), int(meta["num_domains"])), dtype=torch.int64
    )
    planned_domain_flow = torch.zeros_like(baseline_domain_flow)
    quotas: dict[tuple[int, int, int, int], int] = defaultdict(int)
    placements: dict[tuple[int, int, int], int] = defaultdict(int)
    placement_samples: dict[int, int] = defaultdict(int)

    for record_meta, sample in selected:
        stats = compute_routing_stats(
            sample["omega"],
            sample["q"],
            main_rank,
            domains,
            s_tok=s_tok,
            cost=cost,
        )
        layer = int(sample["layer"])
        row = {
            "writer_global_rank": int(record_meta.get("writer_global_rank", 0)),
            "pipeline_rank": int(record_meta.get("pipeline_rank", 0)),
            "solver": record_meta.get("plan_solver", ""),
            "layer": layer,
            "mb": int(sample["mb"]),
            "ordinal": int(sample.get("ordinal", 0)),
            "replicas": int(torch.as_tensor(sample["x"]).sum()),
        }
        for key, value in stats.items():
            if not isinstance(value, torch.Tensor):
                row[key] = value
        sample_rows.append(row)
        baseline_rank_flow += stats["baseline_rank_flow"]
        planned_rank_flow += stats["planned_rank_flow"]
        baseline_domain_flow += stats["baseline_domain_flow"]
        planned_domain_flow += stats["planned_domain_flow"]

        q = torch.as_tensor(sample["q"], dtype=torch.int64)
        x = torch.as_tensor(sample["x"], dtype=torch.int64)
        placement_samples[layer] += 1
        for expert, destination in torch.nonzero(x, as_tuple=False).tolist():
            placements[(layer, expert, destination)] += 1
        for source, expert, destination in torch.nonzero(
            q, as_tuple=False
        ).tolist():
            quotas[(layer, source, expert, destination)] += int(
                q[source, expert, destination]
            )

    sample_rows.sort(
        key=lambda row: (
            row["writer_global_rank"],
            row["mb"],
            row["layer"],
            row["ordinal"],
        )
    )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[int(row["layer"])].append(row)
    by_layer = [{"layer": layer, **_aggregate(rows)} for layer, rows in sorted(grouped.items())]

    quota_rows = []
    for (layer, source, expert, destination), assignments in sorted(quotas.items()):
        source_domain = int(domains[source])
        destination_domain = int(domains[destination])
        home = int(main_rank[expert])
        quota_rows.append(
            {
                "layer": layer,
                "source_rank": source,
                "expert": expert,
                "main_rank": home,
                "destination_rank": destination,
                "source_domain": source_domain,
                "destination_domain": destination_domain,
                "locality": (
                    "local"
                    if source == destination
                    else "intra_domain"
                    if source_domain == destination_domain
                    else "inter_domain"
                ),
                "rerouted_from_main": int(destination != home),
                "assignments": assignments,
                "one_way_mib": round(assignments * s_tok / MIB, 6),
            }
        )
    placement_rows = []
    for (layer, expert, destination), occurrences in sorted(placements.items()):
        placement_rows.append(
            {
                "layer": layer,
                "expert": expert,
                "main_rank": int(main_rank[expert]),
                "host_rank": destination,
                "host_domain": int(domains[destination]),
                "is_main": int(destination == int(main_rank[expert])),
                "sample_occurrences": occurrences,
                "sample_fraction": round(
                    occurrences / placement_samples[layer], 6
                ),
            }
        )

    summary = {
        "topology": {
            "num_ranks": int(meta["num_ranks"]),
            "num_domains": int(meta["num_domains"]),
            "num_experts": int(meta["num_experts"]),
            "s_tok": s_tok,
        },
        "coverage": {
            "trace_files": [str(path) for path in paths],
            "samples": len(sample_rows),
            "layers": sorted(grouped),
            "warmup_mb": warmup_mb,
            "mb_first": min(row["mb"] for row in sample_rows),
            "mb_last": max(row["mb"] for row in sample_rows),
        },
        "aggregate": _aggregate(sample_rows),
        "interpretation": {
            "one_way_bytes": "one dispatch direction; hidden-state payload only",
            "training_bytes": (
                "4x one-way: forward dispatch/combine plus both backward transposes"
            ),
            "inter_domain": "source and physical destination are in different domains",
            "rerouted": "physical destination differs from the expert main rank",
        },
    }
    return (
        summary,
        sample_rows,
        by_layer,
        _flow_rows(
            baseline=baseline_rank_flow,
            planned=planned_rank_flow,
            domain_of_rank=domains,
            s_tok=s_tok,
        ),
        _domain_flow_rows(
            baseline_domain_flow, planned_domain_flow, s_tok
        ),
        quota_rows,
        placement_rows,
    )


def _readme(summary: dict[str, Any]) -> str:
    agg = summary["aggregate"]
    topology = summary["topology"]
    return f"""# EPLB token-routing report

- Coverage: {summary['coverage']['samples']} layer invocations, layers {summary['coverage']['layers']}, mb {summary['coverage']['mb_first']}..{summary['coverage']['mb_last']}
- Topology: {topology['num_ranks']} EP ranks in {topology['num_domains']} domains, {topology['num_experts']} logical experts
- Rank max/mean: {agg['baseline_rank_max_mean_median']:.3f} home placement -> {agg['planned_rank_max_mean_median']:.3f} applied plan
- Excess load absorbed: {agg['absorbed_excess_load_median']:.1%}
- Assignments sent to replicas: {agg['rerouted_fraction']:.1%}
- Cross-domain share: {agg['baseline_inter_domain_fraction']:.1%} home placement -> {agg['planned_inter_domain_fraction']:.1%} applied plan
- Cross-domain assignment reduction: {agg['inter_domain_reduction_fraction']:.1%}
- Applied one-way traffic per layer invocation: {agg['planned_one_way_remote_mib_per_sample']:.2f} MiB remote, {agg['planned_one_way_inter_domain_mib_per_sample']:.2f} MiB cross-domain
- Applied forward+backward logical traffic per layer invocation: {agg['planned_training_remote_mib_per_sample']:.2f} MiB remote, {agg['planned_training_inter_domain_mib_per_sample']:.2f} MiB cross-domain

`samples.csv.gz` contains exact per-layer metrics. `rank_flow.csv` and
`domain_flow.csv` compare original home routing with the applied physical plan.
`quota.csv.gz` exposes the aggregate `(layer, source, expert, destination)`
rerouting decisions. `placement.csv` shows where each expert replica existed.
Byte estimates cover hidden-state rows only; DeepEP metadata, alignment, padding
and protocol headers are intentionally excluded.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        nargs="+",
        required=True,
        help="one or more apply-mode EPLB_TRACE_OUT files",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--warmup-mb",
        type=int,
        default=0,
        help="discard samples with mb smaller than this value",
    )
    args = parser.parse_args()
    if args.warmup_mb < 0:
        parser.error("--warmup-mb must be non-negative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (
        summary,
        samples,
        by_layer,
        rank_flow,
        domain_flow,
        quota,
        placement,
    ) = analyze(args.trace, args.warmup_mb)
    _write_csv(args.out_dir / "samples.csv.gz", samples)
    _write_csv(args.out_dir / "by_layer.csv", by_layer)
    _write_csv(args.out_dir / "rank_flow.csv", rank_flow)
    _write_csv(args.out_dir / "domain_flow.csv", domain_flow)
    _write_csv(args.out_dir / "quota.csv.gz", quota)
    _write_csv(args.out_dir / "placement.csv", placement)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        _readme(summary), encoding="utf-8"
    )
    print(_readme(summary))


if __name__ == "__main__":
    main()

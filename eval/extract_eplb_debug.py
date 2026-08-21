#!/usr/bin/env python3
"""Aggregate ``EPLB_DEBUG_TIMING=1`` per-invocation lines into a phase budget.

``EPLB_DEBUG_TIMING=1`` prints one forward and one backward line per MoE-layer
invocation per reporting rank. This collapses them into per-layer records,
per-iteration phase costs, effective payload bandwidth, and the actual
critical-rank latency that a synchronous EP step is paced by.

Two normalisations matter and are handled here:

- Every line is keyed by ``(rank, mode, layer, micro-batch, direction)``. Under
  ``EPLB_CHUNKS=N`` the value already sums the ``N`` events shown by the ``(xN)``
  suffix; the extractor never divides by that count.
- Critical latency is computed for each matching layer invocation by taking the
  maximum over ranks first. Per-iteration critical cost then sums those layer
  maxima before taking a steady-state median.

Because dispatch, combine, expert GEMM and the weight transfers run on separate CUDA
streams and overlap by design, the per-phase totals are stream occupancy, not a time
budget: do not read their sum as the step time.

Writes into ``--out-dir``:

- ``phases.csv.gz``  one row per (rank, layer, iteration, direction, phase)
- ``by_bucket.csv``  phase medians per iteration bucket, for drift inspection
- ``by_rank.csv``    steady-state phase cost per rank, for straggler analysis
- ``critical_rank.csv`` one row per (layer, iteration, phase), max over ranks
- ``summary.json``   the phase budget, bandwidths and cross-rank spread
- ``README.md``      the same summary in readable form
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ranks interleave their prints without flushing, so a newline is not a reliable record
# separator -- split on the tag instead.
RECORD_SPLIT = re.compile(r"\[EPLB-debug r(\d+)\]\s*")
FORWARD_RE = re.compile(r"mode=(\w+)\s+layer=(\d+)\s+mb=(\d+)")
BACKWARD_RE = re.compile(
    r"mode=(\w+)\s+direction=backward"
    r"(?:\s+layer=(\d+)\s+mb=(\d+))?"
)
# name=1.234ms            name=1.234ms(x2)      name=1.234ms(x2)/64.00MiB/31.86GB/s
FIELD_RE = re.compile(
    r"(?P<name>\w+)=(?P<ms>[\d.]+)ms(?:\(x(?P<count>\d+)\))?"
    r"(?:/(?P<mib>[\d.]+)MiB/(?P<gbps>[\d.]+)GB/s)?"
)

FORWARD_PHASES = (
    "moe_fwd_total",
    "solver",
    "omega_gather",
    "router",
    "shared_expert",
    "expert_transfer",
    "expert_transfer_wire",
    "dispatch",
    "expert_gemm",
    "combine",
)
BACKWARD_PHASES = (
    "moe_bwd_total",
    "expert_repull",
    "expert_repull_wire",
    "combine_bwd",
    "expert_bwd",
    "expert_dgrad",
    "activation_bwd",
    "dispatch_bwd",
    "expert_wgrad",
    "expert_grad_reduce",
    "expert_grad_put_wire",
)
# Phases that exist only because EPLB replicates experts; the rest also run in EPLB_MODE=off.
EPLB_ONLY = (
    "solver",
    "omega_gather",
    "expert_transfer",
    "expert_transfer_wire",
    "expert_repull",
    "expert_repull_wire",
    "expert_grad_reduce",
    "expert_grad_put_wire",
)
EPLB_OVERHEAD_PHASES = (
    "solver",
    "omega_gather",
    "expert_transfer",
    "expert_repull",
    "expert_grad_reduce",
)
TOTAL_PHASES = ("moe_fwd_total", "moe_bwd_total")
WIRE_PHASES = (
    "expert_transfer_wire",
    "expert_repull_wire",
    "expert_grad_put_wire",
)
TRANSFER_PHASES = (
    "expert_transfer",
    "expert_repull",
    "expert_grad_reduce",
    "dispatch",
    "combine",
    "combine_bwd",
    "dispatch_bwd",
)


def parse(paths: list[Path]) -> tuple[dict, dict]:
    """Return phase maps keyed by ``(rank, mode, iteration, layer)``."""
    forward: dict = defaultdict(lambda: defaultdict(lambda: {"ms": 0.0, "mib": 0.0, "count": 0}))
    backward: dict = defaultdict(lambda: defaultdict(lambda: {"ms": 0.0, "mib": 0.0, "count": 0}))
    last_context: dict[int, tuple[str, int]] = {}

    for path in paths:
        text = path.read_text(errors="ignore").replace("\0", "")
        pieces = RECORD_SPLIT.split(text)
        # split() yields [prefix, rank, body, rank, body, ...]
        for index in range(1, len(pieces) - 1, 2):
            rank = int(pieces[index])
            body = pieces[index + 1]
            fields = {
                match.group("name"): match
                for match in FIELD_RE.finditer(body)
            }
            if not fields:
                continue
            backward_head = BACKWARD_RE.search(body)
            if backward_head:
                mode = backward_head.group(1)
                if backward_head.group(2) is not None:
                    layer = int(backward_head.group(2))
                    iteration = int(backward_head.group(3))
                else:
                    # Legacy aggregated backward lines had no layer/mb context.
                    previous = last_context.get(rank)
                    if previous is None:
                        continue
                    mode, iteration = previous
                    layer = -1
                for phase in BACKWARD_PHASES:
                    match = fields.get(phase)
                    if match is None:
                        continue
                    slot = backward[(rank, mode, iteration, layer)][phase]
                    slot["ms"] += float(match.group("ms"))
                    slot["mib"] += float(match.group("mib") or 0.0)
                    slot["count"] += int(match.group("count") or 1)
                continue
            head = FORWARD_RE.search(body)
            if head is None:
                continue
            mode = head.group(1)
            layer = int(head.group(2))
            iteration = int(head.group(3))
            last_context[rank] = (mode, iteration)
            for phase in FORWARD_PHASES:
                match = fields.get(phase)
                if match is None:
                    continue
                slot = forward[(rank, mode, iteration, layer)][phase]
                slot["ms"] += float(match.group("ms"))
                slot["mib"] += float(match.group("mib") or 0.0)
                slot["count"] += int(match.group("count") or 1)
    return forward, backward


def merge(forward: dict, backward: dict) -> list[dict[str, Any]]:
    """Flatten into one row per rank/layer/invocation/phase."""
    rows: list[dict[str, Any]] = []
    for direction, source in (("forward", forward), ("backward", backward)):
        for (rank, mode, iteration, layer), phases in source.items():
            for phase, value in phases.items():
                rows.append(
                    {
                        "rank": rank,
                        "mode": mode,
                        "iteration": iteration,
                        "layer": layer,
                        "direction": direction,
                        "phase": phase,
                        "ms": round(value["ms"], 4),
                        "mib": round(value["mib"], 3),
                        "events": value["count"],
                        "gbps": round(value["mib"] * 1.048576 / value["ms"], 3)
                        if value["mib"] and value["ms"]
                        else 0.0,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["iteration"],
            row["layer"],
            row["rank"],
            row["direction"],
            row["phase"],
        )
    )
    return rows


def critical_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Take max over ranks for every matching layer invocation and phase."""
    grouped: dict = defaultdict(list)
    for row in rows:
        key = (
            row["mode"],
            row["direction"],
            row["iteration"],
            row["layer"],
            row["phase"],
        )
        grouped[key].append(row)

    out = []
    for (mode, direction, iteration, layer, phase), items in sorted(grouped.items()):
        critical = max(items, key=lambda item: (item["ms"], -item["rank"]))
        critical_ms = critical["ms"]
        mean_ms = st.mean(item["ms"] for item in items)
        total_mib = sum(item["mib"] for item in items)
        out.append(
            {
                "mode": mode,
                "direction": direction,
                "iteration": iteration,
                "layer": layer,
                "phase": phase,
                "critical_rank": critical["rank"],
                "critical_ms": round(critical_ms, 4),
                "mean_rank_ms": round(mean_ms, 4),
                "critical_over_mean": round(critical_ms / mean_ms, 4)
                if mean_ms
                else 0.0,
                "ranks_reporting": len({item["rank"] for item in items}),
                "total_mib": round(total_mib, 3),
                "cluster_effective_gbps": round(
                    total_mib * 1.048576 / critical_ms, 3
                )
                if total_mib and critical_ms
                else 0.0,
            }
        )
    return out


def bucket_table(rows: list[dict[str, Any]], bucket: int) -> list[dict[str, Any]]:
    """Bucket already-critical per-layer samples to expose temporal drift."""
    grouped: dict = defaultdict(list)
    for row in rows:
        key = (
            row["mode"],
            row["direction"],
            row["layer"],
            (row["iteration"] - 1) // bucket * bucket + bucket,
            row["phase"],
        )
        grouped[key].append(row)
    out = []
    for (mode, direction, layer, edge, phase), items in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "direction": direction,
                "layer": layer,
                "bucket_end": edge,
                "phase": phase,
                "critical_ms_median": round(
                    st.median([item["critical_ms"] for item in items]), 4
                ),
                "mean_rank_ms_median": round(
                    st.median([item["mean_rank_ms"] for item in items]), 4
                ),
                "critical_over_mean_median": round(
                    st.median([item["critical_over_mean"] for item in items]), 4
                ),
                "cluster_effective_gbps_median": round(
                    st.median([item["cluster_effective_gbps"] for item in items]), 3
                ),
                "samples": len(items),
            }
        )
    return out


def steady_rows(rows: list[dict[str, Any]], warmup: int) -> list[dict[str, Any]]:
    return [row for row in rows if row["iteration"] > warmup]


def rank_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["mode"],
                row["direction"],
                row["rank"],
                row["layer"],
                row["phase"],
            )
        ].append(row["ms"])
    out = []
    for (mode, direction, rank, layer, phase), values in sorted(grouped.items()):
        out.append(
            {
                "mode": mode,
                "direction": direction,
                "rank": rank,
                "layer": layer,
                "phase": phase,
                "ms_median": round(st.median(values), 4),
                "samples": len(values),
            }
        )
    return out


def summarize(
    rows: list[dict[str, Any]],
    warmup: int,
    step_ms: float | None,
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    steady = steady_rows(rows, warmup)
    if not steady:
        raise SystemExit(f"no debug samples past iteration {warmup}")

    modes = sorted({row["mode"] for row in steady})
    if len(modes) != 1:
        raise SystemExit(
            f"expected logs from one mode, found {modes}; extract each configuration separately"
        )

    critical = critical_table(steady)
    if expected_ranks is not None:
        incomplete = [
            item
            for item in critical
            if item["ranks_reporting"] != expected_ranks
        ]
        if incomplete:
            first = incomplete[0]
            raise SystemExit(
                "critical-rank extraction is incomplete: "
                f"{first['phase']} layer={first['layer']} mb={first['iteration']} has "
                f"{first['ranks_reporting']} ranks, expected {expected_ranks}"
            )

    # Sum layer-critical values inside each micro-batch before taking the
    # steady-state median. Taking max after summing rank medians would be wrong
    # when the identity of the straggler changes from layer to layer.
    by_phase_iteration: dict = defaultdict(
        lambda: {
            "critical_ms": 0.0,
            "mean_rank_ms": 0.0,
            "total_mib": 0.0,
            "ranks_reporting": [],
        }
    )
    by_phase_layer: dict = defaultdict(list)
    for item in critical:
        slot = by_phase_iteration[(item["phase"], item["iteration"])]
        slot["critical_ms"] += item["critical_ms"]
        slot["mean_rank_ms"] += item["mean_rank_ms"]
        slot["total_mib"] += item["total_mib"]
        slot["ranks_reporting"].append(item["ranks_reporting"])
        by_phase_layer[item["phase"]].append(item)

    phases = {}
    by_phase: dict = defaultdict(list)
    for (phase, _iteration), value in by_phase_iteration.items():
        by_phase[phase].append(value)
    for phase, items in by_phase.items():
        critical_ms = [item["critical_ms"] for item in items]
        mean_rank_ms = [item["mean_rank_ms"] for item in items]
        critical_over_mean = [
            critical / mean
            for critical, mean in zip(critical_ms, mean_rank_ms)
            if mean
        ]
        mib = [item["total_mib"] for item in items if item["total_mib"]]
        layer_items = by_phase_layer[phase]
        ranks_reporting = [
            count for item in items for count in item["ranks_reporting"]
        ]
        entry = {
            # Keep the old key as an alias for downstream consumers. It now
            # correctly means sum(layer max-over-ranks), not median over rows.
            "ms_per_iteration_median": round(st.median(critical_ms), 3),
            "critical_rank_ms_per_iteration_median": round(
                st.median(critical_ms), 3
            ),
            "critical_rank_ms_p95": round(
                sorted(critical_ms)[int(0.95 * (len(critical_ms) - 1))], 3
            ),
            "mean_rank_ms_per_iteration_median": round(
                st.median(mean_rank_ms), 3
            ),
            "critical_over_mean_median": round(
                st.median(critical_over_mean), 4
            )
            if critical_over_mean
            else None,
            "per_layer_critical_ms_median": round(
                st.median(item["critical_ms"] for item in layer_items), 3
            ),
            "ranks_reporting_min": min(ranks_reporting),
            "ranks_reporting_max": max(ranks_reporting),
            "eplb_only": phase in EPLB_ONLY,
        }
        if mib:
            entry["mib_per_iteration_median"] = round(st.median(mib), 3)
            entry["cluster_effective_gbps_median"] = round(
                st.median(
                    item["total_mib"] * 1.048576 / item["critical_ms"]
                    for item in items
                    if item["total_mib"] and item["critical_ms"]
                ),
                2,
            )
        phases[phase] = entry

    component_phases = [
        phase
        for phase in phases
        if phase not in TOTAL_PHASES and phase not in WIRE_PHASES
    ]
    total = sum(phases[phase]["ms_per_iteration_median"] for phase in component_phases)
    eplb_only = sum(
        phases[phase]["ms_per_iteration_median"]
        for phase in EPLB_OVERHEAD_PHASES
        if phase in phases
    )
    transfer = sum(
        phases[phase]["ms_per_iteration_median"]
        for phase in ("expert_transfer", "expert_repull", "expert_grad_reduce")
        if phase in phases
    )
    iterations = sorted({row["iteration"] for row in rows})
    observed_ranks = sorted({row["rank"] for row in rows})
    summary: dict[str, Any] = {
        "mode": modes[0],
        "iterations": {
            "first": iterations[0],
            "last": iterations[-1],
            "warmup_excluded": warmup,
            "ranks_reporting": len(observed_ranks),
            "expected_ranks": expected_ranks,
        },
        "totals": {
            "component_occupancy_ms_per_iteration": round(total, 3),
            "eplb_only_ms_per_iteration": round(eplb_only, 3),
            "weight_movement_ms_per_iteration": round(transfer, 3),
            "eplb_only_share_of_instrumented": round(eplb_only / total, 4) if total else None,
        },
        "phases": dict(sorted(phases.items(), key=lambda kv: -kv[1]["ms_per_iteration_median"])),
        "caveats": [
            "Phases run on separate CUDA streams and overlap by design; their sum is stream "
            "occupancy, not the step time.",
            "Critical latency is max-over-ranks for each layer invocation, then summed over "
            "layers. It is not max of per-rank medians.",
            "EPLB_DEBUG_TIMING synchronizes at every invocation boundary, so these are "
            "unoverlapped phase costs and the run's own step time must not be quoted.",
            "Bandwidth is effective tensor payload: remote bytes only in the numerator, whole "
            "operation including fences and staging in the denominator.",
        ],
    }
    if step_ms:
        summary["totals"]["reference_step_ms"] = step_ms
        summary["totals"]["instrumented_share_of_step"] = round(total / step_ms, 4)
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    # The per-record table is one row per rank x iteration x phase and compresses ~5x,
    # so it is gzipped in place rather than left as tens of MB on the shared mount.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, name: str, summary: dict[str, Any], sources: list[str]) -> None:
    totals = summary["totals"]
    lines = [
        f"# {name} — MoE phase budget",
        "",
        f"Mode `{summary['mode']}`, iterations "
        f"{summary['iterations']['first']}–{summary['iterations']['last']}, "
        f"{summary['iterations']['ranks_reporting']} ranks reporting, "
        f"first {summary['iterations']['warmup_excluded']} iterations excluded.",
        "",
        "## Per-iteration phase cost",
        "",
        "| phase | critical ms/iter | mean-rank ms/iter | critical/mean | "
        "critical ms/layer | MiB/iter | cluster GB/s | EPLB-only |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for phase, entry in summary["phases"].items():
        lines.append(
            f"| `{phase}` | {entry['critical_rank_ms_per_iteration_median']} | "
            f"{entry['mean_rank_ms_per_iteration_median']} | "
            f"{entry['critical_over_mean_median']} | "
            f"{entry['per_layer_critical_ms_median']} | "
            f"{entry.get('mib_per_iteration_median', '—')} | "
            f"{entry.get('cluster_effective_gbps_median', '—')} | "
            f"{'yes' if entry['eplb_only'] else 'no'} |"
        )
    lines += [
        "",
        "## Totals",
        "",
        f"- Component stream occupancy: "
        f"{totals['component_occupancy_ms_per_iteration']} ms/iteration",
        f"- EPLB-only phases: {totals['eplb_only_ms_per_iteration']} ms/iteration "
        f"({100 * totals['eplb_only_share_of_instrumented']:.1f}% of instrumented)",
        f"- Expert weight and gradient movement: "
        f"{totals['weight_movement_ms_per_iteration']} ms/iteration",
    ]
    if "reference_step_ms" in totals:
        lines.append(
            f"- Against a {totals['reference_step_ms']} ms reference step: "
            f"{100 * totals['instrumented_share_of_step']:.1f}% instrumented"
        )
    lines += ["", "## Caveats", ""] + [f"- {item}" for item in summary["caveats"]]
    lines += ["", "## Files", "",
              "- `phases.csv.gz`: one row per (rank, layer, iteration, direction, phase).",
              "- `critical_rank.csv`: max-over-ranks for every layer invocation and phase.",
              "- `by_bucket.csv`: critical phase medians per iteration bucket and layer.",
              "- `by_rank.csv`: steady-state phase cost per rank and layer.",
              "- `summary.json`: the statistics above in machine-readable form.",
              "", "## Sources", ""]
    lines += [f"- `{source}`" for source in sources]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", required=True, type=Path, nargs="+",
                        help="per-node logs holding the [EPLB-debug r*] lines")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-name", help="defaults to the output directory name")
    parser.add_argument("--warmup", type=int, default=100,
                        help="iterations excluded from the steady-state statistics")
    parser.add_argument("--bucket", type=int, default=250, help="bucket width for by_bucket.csv")
    parser.add_argument("--step-ms", type=float,
                        help="reference step time from an uninstrumented run, for share-of-step")
    parser.add_argument(
        "--expected-ranks",
        type=int,
        help="fail unless every layer/phase sample contains exactly this many ranks",
    )
    args = parser.parse_args()

    forward, backward = parse(args.log)
    rows = merge(forward, backward)
    if not rows:
        raise SystemExit(f"no [EPLB-debug] records found in {args.log}")

    summary = summarize(
        rows,
        warmup=args.warmup,
        step_ms=args.step_ms,
        expected_ranks=args.expected_ranks,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "phases.csv.gz",
              ["rank", "mode", "iteration", "layer", "direction", "phase",
               "ms", "mib", "events", "gbps"])
    critical = critical_table(rows)
    write_csv(
        critical,
        args.out_dir / "critical_rank.csv",
        ["mode", "direction", "iteration", "layer", "phase", "critical_rank",
         "critical_ms", "mean_rank_ms", "critical_over_mean", "ranks_reporting",
         "total_mib", "cluster_effective_gbps"],
    )
    write_csv(bucket_table(critical, args.bucket), args.out_dir / "by_bucket.csv",
              ["mode", "direction", "layer", "bucket_end", "phase",
               "critical_ms_median", "mean_rank_ms_median",
               "critical_over_mean_median", "cluster_effective_gbps_median", "samples"])
    write_csv(rank_table(steady_rows(rows, args.warmup)), args.out_dir / "by_rank.csv",
              ["mode", "direction", "rank", "layer", "phase", "ms_median", "samples"])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(args.out_dir / "README.md", args.run_name or args.out_dir.name, summary,
                 [str(path) for path in args.log])
    print(f"wrote {len(rows)} phase records to {args.out_dir}")


if __name__ == "__main__":
    main()

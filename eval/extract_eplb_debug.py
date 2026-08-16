#!/usr/bin/env python3
"""Aggregate ``EPLB_DEBUG_TIMING=1`` per-invocation lines into a phase budget.

``EPLB_DEBUG_TIMING=1`` prints one line per MoE invocation per reporting rank, so a
short run produces hundreds of thousands of lines across the per-node logs. This
collapses them into per-iteration phase costs, effective payload bandwidth, and the
cross-rank spread that a synchronous EP step is actually paced by.

Two normalisations matter and are handled here:

- Forward fields are per invocation. Under ``EPLB_CHUNKS=N`` the printed value already
  sums the ``N`` chunk events (the ``(xN)`` suffix), but a single line still covers one
  layer, so per-iteration cost is the sum over that iteration's layers.
- Backward fields (``expert_repull``, ``expert_grad_reduce``) are emitted once per
  iteration with all layers already summed, and they arrive at the *next* forward
  boundary, so they are attributed to the most recent iteration seen on that rank.

Because dispatch, combine, expert GEMM and the weight transfers run on separate CUDA
streams and overlap by design, the per-phase totals are stream occupancy, not a time
budget: do not read their sum as the step time.

Writes into ``--out-dir``:

- ``phases.csv.gz``  one row per (rank, iteration, phase)
- ``by_bucket.csv``  phase medians per iteration bucket, for drift inspection
- ``by_rank.csv``    steady-state phase cost per rank, for straggler analysis
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
FORWARD_RE = re.compile(r"mode=(\w+) layer=(\d+) mb=(\d+)")
BACKWARD_RE = re.compile(r"mode=(\w+) direction=backward")
# name=1.234ms            name=1.234ms(x2)      name=1.234ms(x2)/64.00MiB/31.86GB/s
FIELD_RE = re.compile(
    r"(?P<name>\w+)=(?P<ms>[\d.]+)ms(?:\(x(?P<count>\d+)\))?"
    r"(?:/(?P<mib>[\d.]+)MiB/(?P<gbps>[\d.]+)GB/s)?"
)

FORWARD_PHASES = (
    "solver",
    "omega_gather",
    "router",
    "shared_expert",
    "expert_transfer",
    "dispatch",
    "expert_gemm",
    "combine",
)
BACKWARD_PHASES = ("expert_repull", "expert_grad_reduce")
# Phases that exist only because EPLB replicates experts; the rest also run in EPLB_MODE=off.
EPLB_ONLY = ("solver", "omega_gather", "expert_transfer", "expert_repull", "expert_grad_reduce")
TRANSFER_PHASES = ("expert_transfer", "expert_repull", "expert_grad_reduce", "dispatch", "combine")


def parse(paths: list[Path]) -> tuple[dict, dict]:
    """Return (forward, backward) as {(rank, iteration): {phase: {ms, mib, count}}}."""
    forward: dict = defaultdict(lambda: defaultdict(lambda: {"ms": 0.0, "mib": 0.0, "count": 0}))
    backward: dict = defaultdict(dict)
    last_iteration: dict[int, int] = {}

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
            if BACKWARD_RE.search(body):
                iteration = last_iteration.get(rank)
                if iteration is None:
                    continue
                for phase in BACKWARD_PHASES:
                    match = fields.get(phase)
                    if match is None:
                        continue
                    backward[(rank, iteration)][phase] = {
                        "ms": float(match.group("ms")),
                        "mib": float(match.group("mib") or 0.0),
                        "count": int(match.group("count") or 1),
                    }
                continue
            head = FORWARD_RE.search(body)
            if head is None:
                continue
            iteration = int(head.group(3))
            last_iteration[rank] = iteration
            for phase in FORWARD_PHASES:
                match = fields.get(phase)
                if match is None:
                    continue
                slot = forward[(rank, iteration)][phase]
                slot["ms"] += float(match.group("ms"))
                slot["mib"] += float(match.group("mib") or 0.0)
                slot["count"] += int(match.group("count") or 1)
    return forward, backward


def merge(forward: dict, backward: dict) -> list[dict[str, Any]]:
    """Flatten into one row per (rank, iteration, phase), already per-iteration."""
    rows: list[dict[str, Any]] = []
    for source in (forward, backward):
        for (rank, iteration), phases in source.items():
            for phase, value in phases.items():
                rows.append(
                    {
                        "rank": rank,
                        "iteration": iteration,
                        "phase": phase,
                        "ms": round(value["ms"], 4),
                        "mib": round(value["mib"], 3),
                        "events": value["count"],
                        "gbps": round(value["mib"] * 1.048576 / value["ms"], 3)
                        if value["mib"] and value["ms"]
                        else 0.0,
                    }
                )
    rows.sort(key=lambda row: (row["iteration"], row["rank"], row["phase"]))
    return rows


def bucket_table(rows: list[dict[str, Any]], bucket: int) -> list[dict[str, Any]]:
    grouped: dict = defaultdict(list)
    for row in rows:
        key = ((row["iteration"] - 1) // bucket * bucket + bucket, row["phase"])
        grouped[key].append(row)
    out = []
    for (edge, phase), items in sorted(grouped.items()):
        out.append(
            {
                "bucket_end": edge,
                "phase": phase,
                "ms_median": round(st.median([item["ms"] for item in items]), 4),
                "mib_median": round(st.median([item["mib"] for item in items]), 3),
                "gbps_median": round(st.median([item["gbps"] for item in items]), 3),
                "samples": len(items),
            }
        )
    return out


def steady_rows(rows: list[dict[str, Any]], warmup: int) -> list[dict[str, Any]]:
    return [row for row in rows if row["iteration"] > warmup]


def rank_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[(row["rank"], row["phase"])].append(row["ms"])
    out = []
    for (rank, phase), values in sorted(grouped.items()):
        out.append(
            {
                "rank": rank,
                "phase": phase,
                "ms_median": round(st.median(values), 4),
                "samples": len(values),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], warmup: int, step_ms: float | None) -> dict[str, Any]:
    steady = steady_rows(rows, warmup)
    if not steady:
        raise SystemExit(f"no debug samples past iteration {warmup}")

    by_phase: dict = defaultdict(list)
    for row in steady:
        by_phase[row["phase"]].append(row)

    phases = {}
    for phase, items in by_phase.items():
        ms = [item["ms"] for item in items]
        per_rank: dict = defaultdict(list)
        for item in items:
            per_rank[item["rank"]].append(item["ms"])
        rank_medians = [st.median(values) for values in per_rank.values()]
        mib = [item["mib"] for item in items if item["mib"]]
        entry = {
            "ms_per_iteration_median": round(st.median(ms), 3),
            "ms_p95": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 3),
            "ranks_reporting": len(per_rank),
            "max_over_mean_across_ranks": round(max(rank_medians) / st.mean(rank_medians), 4),
            "eplb_only": phase in EPLB_ONLY,
        }
        if mib:
            entry["mib_per_iteration_median"] = round(st.median(mib), 3)
            entry["effective_gbps_median"] = round(
                st.median([item["gbps"] for item in items if item["mib"]]), 2
            )
        phases[phase] = entry

    total = sum(entry["ms_per_iteration_median"] for entry in phases.values())
    eplb_only = sum(
        entry["ms_per_iteration_median"] for phase, entry in phases.items() if phase in EPLB_ONLY
    )
    transfer = sum(
        entry["ms_per_iteration_median"]
        for phase in ("expert_transfer", "expert_repull", "expert_grad_reduce")
        if (entry := phases.get(phase))
    )
    iterations = sorted({row["iteration"] for row in rows})
    summary: dict[str, Any] = {
        "iterations": {
            "first": iterations[0],
            "last": iterations[-1],
            "warmup_excluded": warmup,
            "ranks_reporting": len({row["rank"] for row in rows}),
        },
        "totals": {
            "instrumented_ms_per_iteration": round(total, 3),
            "eplb_only_ms_per_iteration": round(eplb_only, 3),
            "weight_movement_ms_per_iteration": round(transfer, 3),
            "eplb_only_share_of_instrumented": round(eplb_only / total, 4) if total else None,
        },
        "phases": dict(sorted(phases.items(), key=lambda kv: -kv[1]["ms_per_iteration_median"])),
        "caveats": [
            "Phases run on separate CUDA streams and overlap by design; their sum is stream "
            "occupancy, not the step time.",
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
        f"# {name} — EPLB phase budget",
        "",
        f"Iterations {summary['iterations']['first']}–{summary['iterations']['last']}, "
        f"{summary['iterations']['ranks_reporting']} ranks reporting, "
        f"first {summary['iterations']['warmup_excluded']} iterations excluded.",
        "",
        "## Per-iteration phase cost (median over ranks and iterations)",
        "",
        "| phase | ms/iter | MiB/iter | effective GB/s | max/mean over ranks | EPLB-only |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for phase, entry in summary["phases"].items():
        lines.append(
            f"| `{phase}` | {entry['ms_per_iteration_median']} | "
            f"{entry.get('mib_per_iteration_median', '—')} | "
            f"{entry.get('effective_gbps_median', '—')} | "
            f"{entry['max_over_mean_across_ranks']} | "
            f"{'yes' if entry['eplb_only'] else 'no'} |"
        )
    lines += [
        "",
        "## Totals",
        "",
        f"- Instrumented phases: {totals['instrumented_ms_per_iteration']} ms/iteration",
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
              "- `phases.csv.gz`: one row per (rank, iteration, phase).",
              "- `by_bucket.csv`: phase medians per iteration bucket, for drift inspection.",
              "- `by_rank.csv`: steady-state phase cost per rank.",
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
    args = parser.parse_args()

    forward, backward = parse(args.log)
    rows = merge(forward, backward)
    if not rows:
        raise SystemExit(f"no [EPLB-debug] records found in {args.log}")

    summary = summarize(rows, warmup=args.warmup, step_ms=args.step_ms)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "phases.csv.gz",
              ["rank", "iteration", "phase", "ms", "mib", "events", "gbps"])
    write_csv(bucket_table(rows, args.bucket), args.out_dir / "by_bucket.csv",
              ["bucket_end", "phase", "ms_median", "mib_median", "gbps_median", "samples"])
    write_csv(rank_table(steady_rows(rows, args.warmup)), args.out_dir / "by_rank.csv",
              ["rank", "phase", "ms_median", "samples"])
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(args.out_dir / "README.md", args.run_name or args.out_dir.name, summary,
                 [str(path) for path in args.log])
    print(f"wrote {len(rows)} phase records to {args.out_dir}")


if __name__ == "__main__":
    main()

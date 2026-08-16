#!/usr/bin/env python3
"""Report the *raw* routing imbalance from an ``EPLB_TRACE_OUT`` capture.

The ``[EPLB] ... imbalance=`` line an observe run prints is ``plan.rank_load().max() /
mean_load`` -- the imbalance left over *after* the solver has placed replicas. It says how
good the plan is, not how skewed the router was, so it cannot be used on its own to show
that a balancer is needed. The raw skew only exists in the gathered ``Ω[R, E]``, which is
written when ``EPLB_TRACE_OUT`` is set.

This computes, per captured ``(layer, mb)``:

- ``expert_max_mean`` -- skew across logical experts, ``Ω`` summed over source ranks.
- ``rank_max_mean``   -- skew across receiving ranks with every expert on its home rank
  (Megatron's contiguous ``main(e)`` split), i.e. the imbalance with no balancer at all.
- ``source_max_mean`` -- skew across *sending* ranks; every rank routes the same number of
  tokens, so this is a well-formedness check on the trace and should sit at 1.0.

Pass ``--log`` as well and the observe run's residual imbalance is joined on ``(layer, mb)``.
Both quantities divide by the same ``mean_load = Ω.sum() / R`` (plan load is conserved), so
they are exactly comparable and the decomposition is not an approximation:

    absorbed = (rank_max_mean - residual_imbalance) / (rank_max_mean - 1)

is the share of the excess load the solver removed at the configured ``N_slot``.

Writes into ``--out-dir``:

- ``samples.csv.gz``  one row per captured (layer, mb)
- ``by_layer.csv``    medians per MoE layer
- ``by_bucket.csv``   medians per iteration bucket, for drift inspection
- ``summary.json``    headline raw/residual imbalance and the absorbed share
- ``README.md``       the same summary in readable form
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

import torch

from eplb.trace_analysis import load_routing_trace, resolve_main_rank

EPLB_LINE = re.compile(
    r"\[EPLB\] layer=(?P<layer>\d+) mb=(?P<mb>\d+) theta=(?P<theta>\d+) "
    r"imbalance=(?P<imbalance>[\d.]+) replicas=(?P<replicas>\d+) phi_token=(?P<phi_token>\d+)"
)


def _max_mean(values: torch.Tensor) -> float:
    mean = values.mean()
    return float(values.max() / mean) if mean > 0 else 0.0


def raw_metrics(trace: dict) -> list[dict[str, Any]]:
    """Per-sample raw imbalance, before any replica placement."""
    meta = trace["meta"]
    main_rank = resolve_main_rank(meta)
    num_ranks = int(meta["num_ranks"])
    rows = []
    for fallback, sample in enumerate(trace["samples"]):
        omega = torch.as_tensor(sample["omega"], dtype=torch.int64)
        expert_load = omega.sum(dim=0).to(torch.float64)
        source_load = omega.sum(dim=1).to(torch.float64)
        rank_load = torch.zeros(num_ranks, dtype=torch.float64).index_add_(
            0, main_rank, expert_load
        )
        total = float(expert_load.sum())
        hot_expert = int(expert_load.argmax())
        rows.append(
            {
                "layer": int(sample["layer"]),
                "mb": int(sample["mb"]),
                "ordinal": int(sample.get("ordinal", fallback)),
                "total_assignments": int(total),
                "expert_max_mean": round(_max_mean(expert_load), 4),
                "rank_max_mean": round(_max_mean(rank_load), 4),
                "source_max_mean": round(_max_mean(source_load), 4),
                "hot_expert": hot_expert,
                "hot_expert_share": round(float(expert_load[hot_expert]) / total, 5)
                if total
                else 0.0,
                "hot_rank": int(rank_load.argmax()),
                "idle_experts": int((expert_load == 0).sum()),
            }
        )
    rows.sort(key=lambda row: (row["mb"], row["layer"]))
    return rows


def join_residual(rows: list[dict[str, Any]], paths: list[Path]) -> int:
    """Attach the observe run's post-solve imbalance, matched on (layer, mb)."""
    residual: dict[tuple[int, int], dict[str, float]] = {}
    for path in paths:
        text = path.read_text(errors="ignore").replace("\0", "")
        for match in EPLB_LINE.finditer(text):
            residual[(int(match.group("layer")), int(match.group("mb")))] = {
                "residual_imbalance": float(match.group("imbalance")),
                "theta": int(match.group("theta")),
                "replicas": int(match.group("replicas")),
            }
    matched = 0
    for row in rows:
        found = residual.get((row["layer"], row["mb"]))
        if found is None:
            continue
        matched += 1
        row.update(found)
        excess = row["rank_max_mean"] - 1.0
        row["absorbed"] = (
            round((row["rank_max_mean"] - found["residual_imbalance"]) / excess, 4)
            if excess > 1e-9
            else None
        )
    return matched


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(st.median(values), 4) if values else None


def group_table(
    rows: list[dict[str, Any]], key_fn, key_name: str
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        out.append(
            {
                key_name: key,
                "samples": len(items),
                "expert_max_mean_median": _median(items, "expert_max_mean"),
                "rank_max_mean_median": _median(items, "rank_max_mean"),
                "residual_imbalance_median": _median(items, "residual_imbalance"),
                "absorbed_median": _median(items, "absorbed"),
                "replicas_median": _median(items, "replicas"),
                "idle_experts_median": _median(items, "idle_experts"),
            }
        )
    return out


def summarize(
    rows: list[dict[str, Any]], trace: dict, warmup: int, matched: int
) -> dict[str, Any]:
    meta = trace["meta"]
    steady = [row for row in rows if row["mb"] > warmup] or rows
    source = [row["source_max_mean"] for row in steady]
    summary: dict[str, Any] = {
        "topology": {
            "num_ranks": int(meta["num_ranks"]),
            "num_experts": int(meta["num_experts"]),
            "n_slot": int(meta["n_slot"]),
            "experts_per_rank": int(meta["num_experts"]) // int(meta["num_ranks"]),
        },
        "coverage": {
            "samples": len(rows),
            "layers": sorted({row["layer"] for row in rows}),
            "mb_first": min(row["mb"] for row in rows),
            "mb_last": max(row["mb"] for row in rows),
            "warmup_excluded": warmup,
            "residual_matched": matched,
        },
        "raw_imbalance": {
            "expert_max_mean_median": _median(steady, "expert_max_mean"),
            "rank_max_mean_median": _median(steady, "rank_max_mean"),
            "rank_max_mean_p95": round(
                sorted(row["rank_max_mean"] for row in steady)[
                    int(0.95 * (len(steady) - 1))
                ],
                4,
            ),
            "idle_experts_median": _median(steady, "idle_experts"),
        },
        "after_solver": {
            "residual_imbalance_median": _median(steady, "residual_imbalance"),
            "absorbed_share_median": _median(steady, "absorbed"),
            "replicas_median": _median(steady, "replicas"),
        },
        "trace_wellformed": {
            "source_max_mean_median": round(st.median(source), 4),
            "source_max_mean_worst": round(max(source), 4),
            "note": "every rank routes the same token count, so this should be ~1.0",
        },
        "caveats": [
            "rank_max_mean is the imbalance with no balancer: every expert on its home "
            "rank under Megatron's contiguous main(e) split.",
            "residual_imbalance comes from the observe log and already includes the "
            "solver's replica placement at the configured N_slot.",
            "Both use mean_load = sum(omega) / R, so they are directly comparable.",
        ],
    }
    if not matched:
        summary["after_solver"]["note"] = (
            "no observe log supplied or no (layer, mb) matched; raw skew only"
        )
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path, name: str, summary: dict[str, Any], by_layer: list[dict[str, Any]],
    sources: list[str],
) -> None:
    topology, coverage = summary["topology"], summary["coverage"]
    raw, solved = summary["raw_imbalance"], summary["after_solver"]
    lines = [
        f"# {name} — raw routing imbalance",
        "",
        f"{coverage['samples']} captures over layers {coverage['layers']}, "
        f"iterations {coverage['mb_first']}–{coverage['mb_last']} "
        f"(first {coverage['warmup_excluded']} excluded). "
        f"E={topology['num_experts']} on R={topology['num_ranks']} ranks "
        f"({topology['experts_per_rank']} experts/rank), N_slot={topology['n_slot']}.",
        "",
        "## Headline",
        "",
        f"- Expert-level skew (max/mean): **{raw['expert_max_mean_median']}**",
        f"- Rank-level skew with no balancer: **{raw['rank_max_mean_median']}** "
        f"(p95 {raw['rank_max_mean_p95']})",
        f"- Experts receiving nothing: {raw['idle_experts_median']} of "
        f"{topology['num_experts']}",
    ]
    if solved.get("residual_imbalance_median") is not None:
        lines += [
            f"- Left over after the solver: **{solved['residual_imbalance_median']}** "
            f"using {solved['replicas_median']} replicas",
            f"- Share of the excess absorbed: "
            f"**{100 * solved['absorbed_share_median']:.1f}%**",
        ]
    else:
        lines.append(f"- After the solver: {solved.get('note', 'not available')}")
    lines += [
        "",
        "## By layer",
        "",
        "| layer | expert max/mean | rank max/mean | residual | absorbed | idle experts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in by_layer:
        absorbed = row["absorbed_median"]
        lines.append(
            f"| {row['layer']} | {row['expert_max_mean_median']} | "
            f"{row['rank_max_mean_median']} | "
            f"{row['residual_imbalance_median'] if row['residual_imbalance_median'] else '—'} | "
            f"{f'{100 * absorbed:.1f}%' if absorbed else '—'} | "
            f"{row['idle_experts_median']} |"
        )
    well = summary["trace_wellformed"]
    lines += [
        "",
        "## Trace check",
        "",
        f"- Sending-rank skew: {well['source_max_mean_median']} median, "
        f"{well['source_max_mean_worst']} worst ({well['note']}).",
        "",
        "## Caveats",
        "",
    ]
    lines += [f"- {item}" for item in summary["caveats"]]
    lines += [
        "",
        "## Files",
        "",
        "- `samples.csv.gz`: one row per captured (layer, mb).",
        "- `by_layer.csv`: medians per MoE layer.",
        "- `by_bucket.csv`: medians per iteration bucket, for drift inspection.",
        "- `summary.json`: the statistics above in machine-readable form.",
        "",
        "## Sources",
        "",
    ]
    lines += [f"- `{source}`" for source in sources]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trace", required=True, type=Path,
                        help="EPLB_TRACE_OUT file holding the gathered omega")
    parser.add_argument("--log", type=Path, nargs="*", default=[],
                        help="observe logs, to join the post-solve residual imbalance")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-name", help="defaults to the output directory name")
    parser.add_argument("--warmup", type=int, default=100,
                        help="iterations excluded from the headline statistics")
    parser.add_argument("--bucket", type=int, default=100,
                        help="iteration bucket width for by_bucket.csv")
    args = parser.parse_args()

    trace = load_routing_trace(args.trace)
    rows = raw_metrics(trace)
    matched = join_residual(rows, args.log) if args.log else 0
    if args.log and not matched:
        print("warning: no [EPLB] line matched a trace sample; reporting raw skew only")

    summary = summarize(rows, trace, args.warmup, matched)
    steady = [row for row in rows if row["mb"] > args.warmup] or rows
    by_layer = group_table(steady, lambda row: row["layer"], "layer")
    by_bucket = group_table(
        rows, lambda row: row["mb"] // args.bucket * args.bucket + args.bucket, "bucket_end"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "samples.csv.gz",
              ["layer", "mb", "ordinal", "total_assignments", "expert_max_mean",
               "rank_max_mean", "source_max_mean", "residual_imbalance", "absorbed",
               "theta", "replicas", "hot_expert", "hot_expert_share", "hot_rank",
               "idle_experts"])
    group_columns = ["samples", "expert_max_mean_median", "rank_max_mean_median",
                     "residual_imbalance_median", "absorbed_median", "replicas_median",
                     "idle_experts_median"]
    write_csv(by_layer, args.out_dir / "by_layer.csv", ["layer"] + group_columns)
    write_csv(by_bucket, args.out_dir / "by_bucket.csv", ["bucket_end"] + group_columns)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(args.out_dir / "README.md", args.run_name or args.out_dir.name, summary,
                 by_layer, [str(args.trace)] + [str(path) for path in args.log])
    print(f"wrote {len(rows)} trace samples ({matched} with residual) to {args.out_dir}")


if __name__ == "__main__":
    main()

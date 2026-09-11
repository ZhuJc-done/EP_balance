#!/usr/bin/env python3
"""Extract paired rank-load imbalance statistics from apply-mode EPLB traces."""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path
from typing import Any

import torch

from eplb.routing_stats import compute_routing_stats
from eplb.trace_analysis import load_routing_trace, resolve_main_rank


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = Path("/mnt/hdfs/__MERLIN_USER_DIR__/eplb_data/logs")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "e2e_imbalance_comparison_skew_m40"
MODEL_SPECS = {
    "qwen3_5L": {
        "label": "Qwen3-30B-A3B",
        "trace_template": (
            "imbalance_qwen_"
            "plan-{solver}_skew_m40_seed1234.pt"
        ),
    },
    "glm45_air_5moe": {
        "label": "GLM-4.5-Air",
        "trace_template": (
            "imbalance_glm45_air_5moe_ep16pp2_"
            "plan-{solver}_skew_m40_seed1234.pt"
        ),
    },
    "deepseek_v2": {
        "label": "DeepSeek-V2",
        "trace_template": (
            "imbalance_deepseek_v2_pp2ep16_"
            "plan-{solver}_skew_m40_seed1234.pt"
        ),
    },
}
PLANNERS = (
    ("fastermoe", "FasterMoE", "fastermoe"),
    ("flexmoe", "FlexMoE", "flexmoe"),
    ("deepseek_eplb", "DeepSeek-EPLB", "deepseek"),
    ("scale_eplb", "Scale-EPLB", "scale"),
)
METHOD_ORDER = (
    ("megatron_lm", "Megatron-LM"),
    *((method, label) for method, label, _ in PLANNERS),
)
META_INTEGER_KEYS = (
    "num_ranks",
    "num_experts",
    "s_tok",
    "n_slot",
    "num_domains",
)
META_TENSOR_KEYS = ("main_rank", "domain_of_rank", "cost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=tuple(MODEL_SPECS),
        help="Models to extract (defaults to every configured model)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--router-skew", type=float, default=-4.0)
    return parser.parse_args()


def _round(value: float) -> float:
    return round(float(value), 6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sample_map(trace: dict[str, Any], path: Path) -> dict[tuple[int, int], dict]:
    samples: dict[tuple[int, int], dict] = {}
    for fallback, sample in enumerate(trace["samples"]):
        key = (int(sample["layer"]), int(sample["mb"]))
        if key in samples:
            raise ValueError(f"{path}: duplicate layer/microbatch sample {key}")
        if "omega" not in sample or "x" not in sample or "q" not in sample:
            raise ValueError(f"{path}: sample {fallback} lacks omega, x, or q")
        samples[key] = sample
    if not samples:
        raise ValueError(f"{path}: trace contains no samples")
    return samples


def _load_planner_traces(
    trace_dir: Path, trace_template: str
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, dict[tuple[int, int], dict]],
    list[tuple[int, int]],
]:
    if trace_template.count("{solver}") != 1:
        raise ValueError("--trace-template must contain exactly one {solver} placeholder")

    traces: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    sample_maps: dict[str, dict[tuple[int, int], dict]] = {}
    for method, _, solver in PLANNERS:
        path = trace_dir / trace_template.format(solver=solver)
        if not path.is_file():
            raise FileNotFoundError(path)
        trace = load_routing_trace(path)
        meta = trace["meta"]
        if meta.get("trace_kind") != "applied_plan":
            raise ValueError(f"{path}: expected trace_kind=applied_plan")
        if meta.get("plan_solver") != solver:
            raise ValueError(
                f"{path}: plan_solver={meta.get('plan_solver')!r}, expected {solver!r}"
            )
        traces[method] = trace
        paths[method] = path
        sample_maps[method] = _sample_map(trace, path)

    reference_method = "scale_eplb"
    reference = traces[reference_method]
    reference_meta = reference["meta"]
    reference_samples = sample_maps[reference_method]
    sample_keys = sorted(reference_samples, key=lambda key: (key[1], key[0]))

    for method, trace in traces.items():
        path = paths[method]
        meta = trace["meta"]
        for key in META_INTEGER_KEYS:
            if int(meta[key]) != int(reference_meta[key]):
                raise ValueError(f"{path}: incompatible metadata field {key}")
        for key in META_TENSOR_KEYS:
            if not torch.equal(
                torch.as_tensor(meta[key]), torch.as_tensor(reference_meta[key])
            ):
                raise ValueError(f"{path}: incompatible metadata tensor {key}")

        samples = sample_maps[method]
        if set(samples) != set(reference_samples):
            raise ValueError(f"{path}: layer/microbatch coverage differs")
        for sample_key in sample_keys:
            if not torch.equal(
                torch.as_tensor(samples[sample_key]["omega"]),
                torch.as_tensor(reference_samples[sample_key]["omega"]),
            ):
                raise ValueError(
                    f"{path}: omega differs at layer/microbatch {sample_key}"
                )

    return traces, paths, sample_maps, sample_keys


def _compute_rows(
    *,
    traces: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    sample_maps: dict[str, dict[tuple[int, int], dict]],
    sample_keys: list[tuple[int, int]],
    model: str,
    model_label: str,
    router_skew: float,
) -> list[dict[str, Any]]:
    reference_meta = traces["scale_eplb"]["meta"]
    main_rank = resolve_main_rank(reference_meta)
    domains = torch.as_tensor(reference_meta["domain_of_rank"], dtype=torch.int64)
    cost = torch.as_tensor(reference_meta["cost"], dtype=torch.int64)
    s_tok = int(reference_meta["s_tok"])

    planner_stats: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for method, _, _ in PLANNERS:
        stats_by_sample = {}
        for sample_key in sample_keys:
            sample = sample_maps[method][sample_key]
            stats_by_sample[sample_key] = compute_routing_stats(
                sample["omega"],
                sample["q"],
                main_rank,
                domains,
                s_tok=s_tok,
                cost=cost,
            )
        planner_stats[method] = stats_by_sample

    rows = []
    reference_method = "scale_eplb"
    for sample_index, (layer, microbatch) in enumerate(sample_keys):
        reference_sample = sample_maps[reference_method][(layer, microbatch)]
        reference_stats = planner_stats[reference_method][(layer, microbatch)]
        baseline_imbalance = float(reference_stats["baseline_rank_max_mean"])
        baseline_inter_fraction = float(
            reference_stats["baseline_inter_domain_fraction"]
        )

        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "method": "megatron_lm",
                "method_label": "Megatron-LM",
                "router_skew": router_skew,
                "sample_index": sample_index,
                "layer": layer,
                "microbatch": microbatch,
                "ordinal": int(reference_sample.get("ordinal", sample_index)),
                "num_ranks": int(reference_meta["num_ranks"]),
                "num_experts": int(reference_meta["num_experts"]),
                "n_slot": int(reference_meta["n_slot"]),
                "num_domains": int(reference_meta["num_domains"]),
                "total_assignments": int(reference_stats["total_assignments"]),
                "baseline_rank_max_mean": _round(baseline_imbalance),
                "method_rank_max_mean": _round(baseline_imbalance),
                "absorbed_excess_load": 0.0,
                "rerouted_fraction": 0.0,
                "baseline_inter_domain_fraction": _round(
                    baseline_inter_fraction
                ),
                "method_inter_domain_fraction": _round(baseline_inter_fraction),
                "inter_domain_reduction_fraction": 0.0,
                "source_kind": "home_expert_placement",
                "source_trace": str(paths[reference_method]),
            }
        )

        for method, method_label, _ in PLANNERS:
            stats = planner_stats[method][(layer, microbatch)]
            rows.append(
                {
                    "model": model,
                    "model_label": model_label,
                    "method": method,
                    "method_label": method_label,
                    "router_skew": router_skew,
                    "sample_index": sample_index,
                    "layer": layer,
                    "microbatch": microbatch,
                    "ordinal": int(
                        sample_maps[method][(layer, microbatch)].get(
                            "ordinal", sample_index
                        )
                    ),
                    "num_ranks": int(reference_meta["num_ranks"]),
                    "num_experts": int(reference_meta["num_experts"]),
                    "n_slot": int(reference_meta["n_slot"]),
                    "num_domains": int(reference_meta["num_domains"]),
                    "total_assignments": int(stats["total_assignments"]),
                    "baseline_rank_max_mean": _round(
                        stats["baseline_rank_max_mean"]
                    ),
                    "method_rank_max_mean": _round(
                        stats["planned_rank_max_mean"]
                    ),
                    "absorbed_excess_load": _round(
                        stats["absorbed_excess_load"]
                    ),
                    "rerouted_fraction": _round(stats["rerouted_fraction"]),
                    "baseline_inter_domain_fraction": _round(
                        stats["baseline_inter_domain_fraction"]
                    ),
                    "method_inter_domain_fraction": _round(
                        stats["planned_inter_domain_fraction"]
                    ),
                    "inter_domain_reduction_fraction": _round(
                        stats["inter_domain_reduction_fraction"]
                    ),
                    "source_kind": "applied_plan",
                    "source_trace": str(paths[method]),
                }
            )
    return rows


def _summarize(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    model_order = list(dict.fromkeys(str(row["model"]) for row in sample_rows))
    for model in model_order:
        model_rows = [row for row in sample_rows if row["model"] == model]
        rows_by_method = {
            method: [row for row in model_rows if row["method"] == method]
            for method, _ in METHOD_ORDER
        }
        for method, method_label in METHOD_ORDER:
            method_rows = rows_by_method[method]
            if not method_rows:
                raise ValueError(f"no sample rows for {model}/{method}")
            first = method_rows[0]
            summaries.append(
                {
                    "model": first["model"],
                    "model_label": first["model_label"],
                    "method": method,
                    "method_label": method_label,
                    "mean_rank_max_mean": _round(
                        st.mean(
                            float(row["method_rank_max_mean"])
                            for row in method_rows
                        )
                    ),
                    "mean_rerouted_fraction": _round(
                        st.mean(
                            float(row["rerouted_fraction"])
                            for row in method_rows
                        )
                    ),
                    "mean_method_inter_domain_fraction": _round(
                        st.mean(
                            float(row["method_inter_domain_fraction"])
                            for row in method_rows
                        )
                    ),
                }
            )
    return summaries


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sample_rows = []
    for model in args.models:
        spec = MODEL_SPECS[model]
        traces, paths, sample_maps, sample_keys = _load_planner_traces(
            trace_dir, str(spec["trace_template"])
        )
        sample_rows.extend(
            _compute_rows(
                traces=traces,
                paths=paths,
                sample_maps=sample_maps,
                sample_keys=sample_keys,
                model=model,
                model_label=str(spec["label"]),
                router_skew=args.router_skew,
            )
        )
        print(
            f"{model}: validated {len(paths)} planner traces with "
            f"{len(sample_keys)} identical routing samples"
        )
    summary_rows = _summarize(sample_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "average_imbalance_skew_m40.csv"
    sample_path = output_dir / "routing_sample_imbalance_skew_m40.csv"
    _write_csv(summary_path, summary_rows)
    _write_csv(sample_path, sample_rows)
    print(f"saved {summary_path}")
    print(f"saved {sample_path}")


if __name__ == "__main__":
    main()

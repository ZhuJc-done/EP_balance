#!/usr/bin/env python3
"""Extract comparable router-skew=-4.0 throughput tails for MoE baselines."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
from pathlib import Path
from typing import Any

from extract_train_metrics import ARG_RE, parse_config, parse_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = Path(
    os.environ.get(
        "EPLB_LOG_DIR",
        "/mnt/hdfs/__MERLIN_USER_DIR__/eplb_data/logs",
    )
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "e2e_baseline_comparison_skew_m40"
ROUTER_SKEW = -4.0
METHOD_ORDER = (
    "megatron_lm",
    "fastermoe",
    "deepseek_eplb",
    "flexmoe",
    "scale_eplb",
)
METHOD_LABEL = {
    "megatron_lm": "Megatron-LM",
    "fastermoe": "FasterMoE",
    "deepseek_eplb": "DeepSeek-EPLB",
    "flexmoe": "FlexMoE",
    "scale_eplb": "Scale-EPLB",
}
MODEL_LABEL = {
    "qwen3_5L": "Qwen3-30B-A3B",
    "glm45_air_5moe": "GLM-4.5-Air",
    "deepseek_v2": "DeepSeek-V2",
}
RUNS = {
    "qwen3_5L": {
        "megatron_lm": (
            "e2e_qwen3_5L_ep32_native_skew_m40_adam_noaux_seed1234"
        ),
        "fastermoe": (
            "e2e_qwen3_5L_ep32_plan-fastermoe_ns6_c2_skew_m40_seed1234"
        ),
        "deepseek_eplb": (
            "e2e_qwen3_5L_ep32_plan-deepseek_ns6_c2_skew_m40_seed1234"
        ),
        "flexmoe": (
            "e2e_qwen3_5L_ep32_plan-flexmoe_ns6_c2_skew_m40_seed1234"
        ),
        "scale_eplb": (
            "e2e_qwen3_5L_ep32_scale_eplb_ns6_c2_skew40_adam_noaux_seed1234"
        ),
    },
    "glm45_air_5moe": {
        "megatron_lm": (
            "e2e_glm45_air_5moe_ep32_native_skew_m40_adam_noaux_seed1234"
        ),
        "fastermoe": (
            "e2e_glm45_air_5moe_ep16pp2_plan-fastermoe_"
            "ns10_c2_skew_m40_seed1234"
        ),
        "deepseek_eplb": (
            "e2e_glm45_air_5moe_ep16pp2_plan-deepseek_"
            "ns10_c2_skew_m40_seed1234"
        ),
        "flexmoe": (
            "e2e_glm45_air_5moe_ep16pp2_plan-flexmoe_"
            "ns10_c2_skew_m40_seed1234"
        ),
        "scale_eplb": (
            "e2e_glm45_air_5moe_ep32_scale_eplb_"
            "ns6_c2_skew_m40_adam_noaux_seed1234"
        ),
    },
    "deepseek_v2": {
        "megatron_lm": "e2e_deepseek_v2_pp2ep16_native_skew_m40_seed1234",
        "fastermoe": "e2e_deepseek_v2_pp2ep16_fastermoe_skew_m40_seed1234",
        "deepseek_eplb": (
            "e2e_deepseek_v2_pp2ep16_deepseek_skew_m40_seed1234"
        ),
        "flexmoe": "e2e_deepseek_v2_pp2ep16_flexmoe_skew_m40_seed1234",
        "scale_eplb": "e2e_deepseek_v2_pp2ep16_scale_skew_m40_seed1234",
    },
}
COMPARABILITY_KEYS = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_experts",
    "moe_ffn_hidden_size",
    "moe_router_topk",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "seq_length",
    "micro_batch_size",
    "global_batch_size",
    "world_size",
    "optimizer",
    "seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=101,
        help="Final consecutive measured iterations used for each average",
    )
    return parser.parse_args()


def _all_arguments(path: Path) -> dict[str, str]:
    arguments: dict[str, str] = {}
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = ARG_RE.match(line)
            if match and match.group("key") not in arguments:
                arguments[match.group("key")] = match.group("value")
    return arguments


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_log_and_rows(
    log_dir: Path,
    stem: str,
) -> tuple[list[Path], Path, list[dict[str, Any]]]:
    node_logs = sorted(log_dir.glob(f"{stem}_node*.log"))
    if not node_logs:
        raise FileNotFoundError(f"no node logs found for {stem}")

    metric_sources = []
    for path in node_logs:
        rows = parse_metrics(path)
        if rows:
            metric_sources.append((path, rows))
    if len(metric_sources) != 1:
        names = ", ".join(path.name for path, _ in metric_sources) or "none"
        raise ValueError(f"{stem}: expected one metric-bearing node log, found {names}")

    metric_path, rows = metric_sources[0]
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != list(range(iterations[0], iterations[-1] + 1)):
        raise ValueError(f"{metric_path}: iterations are not unique and contiguous")
    return node_logs, metric_path, rows


def _round(value: float) -> float:
    return round(value, 6)


def main() -> None:
    args = parse_args()
    if args.tail_steps <= 1:
        raise ValueError("--tail-steps must be greater than 1")

    log_dir = args.log_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []
    configs_by_model: dict[str, dict[str, dict[str, Any]]] = {}

    for model, methods in RUNS.items():
        configs_by_model[model] = {}
        for method in METHOD_ORDER:
            stem = methods[method]
            config_path = log_dir / f"{stem}_node0.log"
            if not config_path.is_file():
                raise FileNotFoundError(config_path)

            arguments = _all_arguments(config_path)
            observed_skew = float(arguments.get("moe_router_force_biased", "nan"))
            if observed_skew != ROUTER_SKEW:
                raise ValueError(
                    f"{config_path}: expected moe_router_force_biased="
                    f"{ROUTER_SKEW}, found {observed_skew}"
                )

            config = parse_config(config_path)
            configs_by_model[model][method] = config
            node_logs, metric_path, rows = _metric_log_and_rows(log_dir, stem)
            if len(rows) < args.tail_steps:
                raise ValueError(
                    f"{metric_path}: requires {args.tail_steps} measured rows, "
                    f"found {len(rows)}"
                )

            selected = rows[-args.tail_steps :]
            seq_length = int(config["seq_length"])
            values = [
                float(row["global_batch_size"])
                * seq_length
                / float(row["elapsed_ms"])
                for row in selected
            ]
            elapsed = [float(row["elapsed_ms"]) for row in selected]
            mean = st.mean(values)
            baseline_fields = {
                "model": model,
                "model_label": MODEL_LABEL[model],
                "method": method,
                "method_label": METHOD_LABEL[method],
                "router_skew": ROUTER_SKEW,
            }
            for row, throughput in zip(selected, values):
                stable_rows.append(
                    {
                        **baseline_fields,
                        "iteration": int(row["iteration"]),
                        "elapsed_ms": _round(float(row["elapsed_ms"])),
                        "tokens_per_second_global": _round(throughput * 1000.0),
                        "k_tokens_per_second_global": _round(throughput),
                        "global_batch_size": int(row["global_batch_size"]),
                        "seq_length": seq_length,
                        "source_metrics_log": metric_path.name,
                    }
                )

            summaries.append(
                {
                    **baseline_fields,
                    "window_policy": f"final_{args.tail_steps}_consecutive_steps",
                    "iteration_first": int(selected[0]["iteration"]),
                    "iteration_last": int(selected[-1]["iteration"]),
                    "num_steps": len(selected),
                    "mean_k_tokens_per_second": _round(mean),
                    "std_k_tokens_per_second": _round(st.pstdev(values)),
                    "median_k_tokens_per_second": _round(st.median(values)),
                    "p25_k_tokens_per_second": _round(_percentile(values, 0.25)),
                    "p75_k_tokens_per_second": _round(_percentile(values, 0.75)),
                    "mean_elapsed_ms": _round(st.mean(elapsed)),
                    "seq_length": seq_length,
                    "global_batch_size": int(config["global_batch_size"]),
                    "world_size": int(config["world_size"]),
                    "expert_parallel_size": int(config["expert_model_parallel_size"]),
                    "pipeline_parallel_size": int(
                        config["pipeline_model_parallel_size"]
                    ),
                    "source_metrics_log": metric_path.name,
                    "source_config_log": config_path.name,
                    "available_node_logs": len(node_logs),
                }
            )

    for model, method_configs in configs_by_model.items():
        reference = method_configs["megatron_lm"]
        for method, config in method_configs.items():
            differences = {
                key: (reference.get(key), config.get(key))
                for key in COMPARABILITY_KEYS
                if reference.get(key) != config.get(key)
            }
            if differences:
                raise ValueError(
                    f"{model}/{method}: non-comparable training configuration "
                    f"relative to Megatron-LM: {differences}"
                )

    baseline_means = {
        str(row["model"]): float(row["mean_k_tokens_per_second"])
        for row in summaries
        if row["method"] == "megatron_lm"
    }
    for row in summaries:
        relative = float(row["mean_k_tokens_per_second"]) / baseline_means[str(row["model"])]
        row["throughput_vs_megatron"] = _round(relative)
        row["change_vs_megatron_percent"] = _round((relative - 1.0) * 100.0)

    summary_path = output_dir / "average_throughput_skew_m40.csv"
    samples_path = output_dir / "tail_step_throughput_skew_m40.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stable_rows[0]))
        writer.writeheader()
        writer.writerows(stable_rows)

    print(
        f"wrote {len(summaries)} averages from "
        f"{len(stable_rows)} measured steps to {summary_path}"
    )
    print(f"wrote tail-step data to {samples_path}")


if __name__ == "__main__":
    main()

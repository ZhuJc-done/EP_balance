#!/usr/bin/env python3
"""Add stable-tail router-skew 0 and -0.5 measurements to E2E plot data."""

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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "e2e_analyse"
MODEL_ORDER = ("qwen3_5L", "glm45_air_5moe", "deepseek_v2")
MODEL_LABELS = {
    "qwen3_5L": "Qwen3-30B-A3B",
    "glm45_air_5moe": "GLM-4.5-Air",
    "deepseek_v2": "DeepSeek-V2",
}
METHOD_ORDER = ("native", "scale_eplb")
METHOD_LABELS = {
    "native": "Megatron-LM",
    "scale_eplb": "Scale-EPLB",
}
PLOT_SKEW_ORDER = ("natural", "0.0", "-0.5", "-1.0", "-2.0", "-4.0")
RUNS = {
    ("qwen3_5L", "native", "0.0"): (
        "e2e_qwen3_5L_ep32_native_skew_m00_adam_noaux_seed1234"
    ),
    ("qwen3_5L", "scale_eplb", "0.0"): (
        "e2e_qwen3_5L_ep32_scale_eplb_ns6_c2_skew_m00_adam_noaux_seed1234"
    ),
    ("qwen3_5L", "native", "-0.5"): (
        "e2e_qwen3_5L_ep32_native_skew_m05_adam_noaux_seed1234"
    ),
    ("qwen3_5L", "scale_eplb", "-0.5"): (
        "e2e_qwen3_5L_ep32_scale_eplb_ns6_c2_skew_m05_adam_noaux_seed1234"
    ),
    ("glm45_air_5moe", "native", "0.0"): (
        "e2e_glm45_air_5moe_ep32_native_skew_m00_adam_noaux_seed1234"
    ),
    ("glm45_air_5moe", "scale_eplb", "0.0"): (
        "e2e_glm45_air_5moe_ep32_scale_eplb_ns6_c2_skew_m00_adam_noaux_seed1234"
    ),
    ("glm45_air_5moe", "native", "-0.5"): (
        "e2e_glm45_air_5moe_ep32_native_skew_m05_adam_noaux_seed1234"
    ),
    ("glm45_air_5moe", "scale_eplb", "-0.5"): (
        "e2e_glm45_air_5moe_ep32_scale_eplb_ns6_c2_skew_m05_adam_noaux_seed1234"
    ),
    ("deepseek_v2", "native", "0.0"): (
        "e2e_deepseek_v2_pp2ep16_native_skew_m00_seed1234"
    ),
    ("deepseek_v2", "scale_eplb", "0.0"): (
        "e2e_deepseek_v2_pp2ep16_scale_skew_m00_seed1234"
    ),
    ("deepseek_v2", "native", "-0.5"): (
        "e2e_deepseek_v2_pp2ep16_native_skew_m05_seed1234"
    ),
    ("deepseek_v2", "scale_eplb", "-0.5"): (
        "e2e_deepseek_v2_pp2ep16_scale_skew_m05_seed1234"
    ),
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
KEYPOINT_FIELDS = (
    "model",
    "method",
    "routing",
    "router_skew",
    "metric",
    "unit",
    "aligned_step",
    "window_first_step",
    "window_last_step",
    "source_iteration_first",
    "source_iteration_last",
    "throughput_median",
    "throughput_p25",
    "throughput_p75",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keypoints-csv",
        type=Path,
        help="Existing keypoint CSV to update; defaults inside --output-dir",
    )
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=101,
        help="Number of consecutive measured iterations in each stable tail",
    )
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=1.25,
        help="Maximum elapsed-time multiple allowed in a selected stable window",
    )
    parser.add_argument("--aligned-end", type=int, default=5000)
    parser.add_argument("--num-points", type=int, default=11)
    parser.add_argument("--point-window", type=int, default=3)
    return parser.parse_args()


def _all_arguments(path: Path) -> dict[str, str]:
    arguments: dict[str, str] = {}
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = ARG_RE.match(line)
            if match and match.group("key") not in arguments:
                arguments[match.group("key")] = match.group("value")
    return arguments


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


def _select_latest_stable_window(
    rows: list[dict[str, Any]],
    *,
    length: int,
    slow_factor: float,
) -> list[dict[str, Any]]:
    for end in range(len(rows), length - 1, -1):
        candidate = rows[end - length : end]
        iterations = [int(row["iteration"]) for row in candidate]
        if iterations != list(range(iterations[0], iterations[-1] + 1)):
            continue
        elapsed = [float(row["elapsed_ms"]) for row in candidate]
        threshold = slow_factor * st.median(elapsed)
        if all(value <= threshold for value in elapsed):
            return candidate
    raise ValueError(
        f"no {length}-step stable window found with slow-factor {slow_factor}"
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _point_centers(start: int, end: int, count: int) -> list[int]:
    return [
        round(start + index * (end - start) / (count - 1))
        for index in range(count)
    ]


def _window_bounds(center: int, start: int, end: int, width: int) -> tuple[int, int]:
    half = width // 2
    lower = center - half
    upper = lower + width - 1
    if lower < start:
        lower, upper = start, start + width - 1
    if upper > end:
        lower, upper = end - width + 1, end
    return lower, upper


def _build_keypoints(
    samples: list[dict[str, Any]],
    *,
    model: str,
    method: str,
    skew: str,
    aligned_start: int,
    aligned_end: int,
    num_points: int,
    point_window: int,
) -> list[dict[str, Any]]:
    by_step = {int(row["aligned_step"]): row for row in samples}
    points = []
    for center in _point_centers(aligned_start, aligned_end, num_points):
        lower, upper = _window_bounds(
            center,
            aligned_start,
            aligned_end,
            point_window,
        )
        window = [by_step[step] for step in range(lower, upper + 1)]
        values = [float(row["throughput_k_tokens_per_s"]) for row in window]
        points.append(
            {
                "model": model,
                "method": method,
                "routing": "synthetic_skew",
                "router_skew": skew,
                "metric": "tokens",
                "unit": "k tokens/s",
                "aligned_step": center,
                "window_first_step": lower,
                "window_last_step": upper,
                "source_iteration_first": window[0]["iteration"],
                "source_iteration_last": window[-1]["iteration"],
                "throughput_median": round(st.median(values), 6),
                "throughput_p25": round(_percentile(values, 0.25), 6),
                "throughput_p75": round(_percentile(values, 0.75), 6),
            }
        )
    return points


def _skew_key(row: dict[str, Any]) -> str:
    if row["routing"] == "natural":
        return "natural"
    return f"{float(row['router_skew']):.1f}"


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merge_keypoints(
    path: Path,
    new_rows: list[dict[str, Any]],
    replaced_keys: set[tuple[str, str, str]],
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    kept = [
        row
        for row in existing
        if (row["model"], row["method"], _skew_key(row)) not in replaced_keys
    ]
    combined: list[dict[str, Any]] = [*kept, *new_rows]
    model_index = {value: index for index, value in enumerate(MODEL_ORDER)}
    method_index = {value: index for index, value in enumerate(METHOD_ORDER)}
    skew_index = {value: index for index, value in enumerate(PLOT_SKEW_ORDER)}
    combined.sort(
        key=lambda row: (
            model_index[row["model"]],
            method_index[row["method"]],
            skew_index[_skew_key(row)],
            int(row["aligned_step"]),
        )
    )
    _write_csv(combined, path, list(KEYPOINT_FIELDS))


def main() -> None:
    args = parse_args()
    if args.tail_steps < 3:
        raise ValueError("--tail-steps must be at least 3")
    if args.slow_factor <= 1.0:
        raise ValueError("--slow-factor must be greater than 1")
    if args.num_points < 2:
        raise ValueError("--num-points must be at least 2")
    if not 1 <= args.point_window <= args.tail_steps:
        raise ValueError("--point-window must be between 1 and --tail-steps")

    log_dir = args.log_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints_path = (
        args.keypoints_csv.expanduser().resolve()
        if args.keypoints_csv
        else output_dir / "e2e_throughput_keypoints_tokens_steps4900_5000.csv"
    )
    if not keypoints_path.is_file():
        raise FileNotFoundError(keypoints_path)

    aligned_start = args.aligned_end - args.tail_steps + 1
    all_samples: list[dict[str, Any]] = []
    all_keypoints: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    configs: dict[tuple[str, str, str], dict[str, Any]] = {}

    for (model, method, skew), stem in RUNS.items():
        config_path = log_dir / f"{stem}_node0.log"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        arguments = _all_arguments(config_path)
        observed_skew = f"{float(arguments.get('moe_router_force_biased', 'nan')):.1f}"
        if observed_skew != skew:
            raise ValueError(
                f"{config_path}: expected moe_router_force_biased={skew}, "
                f"found {observed_skew}"
            )

        config = parse_config(config_path)
        configs[(model, method, skew)] = config
        node_logs, metric_path, rows = _metric_log_and_rows(log_dir, stem)
        if len(rows) < args.tail_steps:
            raise ValueError(
                f"{metric_path}: requires {args.tail_steps} measured rows, "
                f"found {len(rows)}"
            )
        selected = _select_latest_stable_window(
            rows,
            length=args.tail_steps,
            slow_factor=args.slow_factor,
        )

        seq_length = int(config["seq_length"])
        run_samples = []
        for index, row in enumerate(selected):
            throughput = (
                float(row["global_batch_size"])
                * seq_length
                / float(row["elapsed_ms"])
            )
            run_samples.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "routing": "synthetic_skew",
                    "router_skew": skew,
                    "aligned_step": aligned_start + index,
                    "iteration": int(row["iteration"]),
                    "elapsed_ms": round(float(row["elapsed_ms"]), 6),
                    "throughput_k_tokens_per_s": round(throughput, 6),
                    "global_batch_size": int(row["global_batch_size"]),
                    "seq_length": seq_length,
                    "source_metrics_log": metric_path.name,
                    "source_config_log": config_path.name,
                }
            )
        all_samples.extend(run_samples)
        all_keypoints.extend(
            _build_keypoints(
                run_samples,
                model=model,
                method=method,
                skew=skew,
                aligned_start=aligned_start,
                aligned_end=args.aligned_end,
                num_points=args.num_points,
                point_window=args.point_window,
            )
        )

        throughputs = [
            float(row["throughput_k_tokens_per_s"]) for row in run_samples
        ]
        elapsed = [float(row["elapsed_ms"]) for row in selected]
        elapsed_median = st.median(elapsed)
        summaries.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "router_skew": skew,
                "logged_iterations": len(rows),
                "logged_iteration_first": int(rows[0]["iteration"]),
                "logged_iteration_last": int(rows[-1]["iteration"]),
                "selected_iteration_first": int(selected[0]["iteration"]),
                "selected_iteration_last": int(selected[-1]["iteration"]),
                "selected_steps": len(selected),
                "mean_k_tokens_per_s": round(st.mean(throughputs), 6),
                "median_k_tokens_per_s": round(st.median(throughputs), 6),
                "std_k_tokens_per_s": round(st.pstdev(throughputs), 6),
                "p25_k_tokens_per_s": round(_percentile(throughputs, 0.25), 6),
                "p75_k_tokens_per_s": round(_percentile(throughputs, 0.75), 6),
                "median_elapsed_ms": round(elapsed_median, 6),
                "maximum_elapsed_over_median": round(
                    max(elapsed) / elapsed_median,
                    6,
                ),
                "window_policy": (
                    f"latest_{args.tail_steps}_consecutive_steps_with_"
                    f"max_elapsed_le_{args.slow_factor:g}x_median"
                ),
                "available_node_logs": len(node_logs),
                "source_metrics_log": metric_path.name,
                "source_config_log": config_path.name,
            }
        )

    for model in MODEL_ORDER:
        for skew in ("0.0", "-0.5"):
            reference = configs[(model, "native", skew)]
            candidate = configs[(model, "scale_eplb", skew)]
            differences = {
                key: (reference.get(key), candidate.get(key))
                for key in COMPARABILITY_KEYS
                if reference.get(key) != candidate.get(key)
            }
            if differences:
                raise ValueError(
                    f"{model}/{skew}: non-comparable Scale-EPLB configuration: "
                    f"{differences}"
                )

    sample_path = output_dir / "e2e_new_skews_stable_samples.csv"
    summary_path = output_dir / "e2e_new_skews_stable_windows.csv"
    _write_csv(all_samples, sample_path, list(all_samples[0]))
    _write_csv(summaries, summary_path, list(summaries[0]))
    _merge_keypoints(keypoints_path, all_keypoints, set(RUNS))

    for row in summaries:
        print(
            f"{row['model']}/{row['method']}/skew={row['router_skew']}: "
            f"iterations {row['selected_iteration_first']}-"
            f"{row['selected_iteration_last']}, "
            f"mean {row['mean_k_tokens_per_s']:.3f} k tokens/s"
        )
    print(f"saved {sample_path}")
    print(f"saved {summary_path}")
    print(f"updated {keypoints_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract loss and throughput series from a Megatron training log.

Megatron prints the per-iteration line only on the last rank, so ``--log``
normally points at the highest-numbered node's log while ``--config-log``
points at rank 0's log, which is where the argument dump and the memory
report land.

Writes into ``--out-dir``:

- ``metrics.csv``    one row per iteration
- ``curve.csv``      bucket-averaged series for plotting
- ``summary.json``   aggregate loss / throughput / stability statistics
- ``config.json``    the training arguments that identify the run
- ``README.md``      the same summary in readable form
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path
from typing import Any

ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+)\s*\|"
    r"\s*consumed samples:\s*(?P<samples>\d+)\s*\|"
    r"\s*elapsed time per iteration \(ms\):\s*(?P<ms>[\d.]+)\s*\|"
    r"(?:\s*throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[\d.]+)\s*\|)?"
    r"\s*learning rate:\s*(?P<lr>[\dEe.+-]+)\s*\|"
    r"\s*global batch size:\s*(?P<gbs>\d+)\s*\|"
    r"\s*lm loss:\s*(?P<loss>[\dEe.+-]+)\s*\|"
)
GRAD_NORM_RE = re.compile(r"grad norm:\s*(?P<gn>[\d.]+|nan)")
TIMESTAMP_RE = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
ARG_RE = re.compile(r"^\s{2}(?P<key>\w+)\s\.{2,}\s(?P<value>.*?)\s*$")
MEMORY_RE = re.compile(
    r"\(after (?P<iter>\d+) iterations\) memory \(MB\).*?max reserved:\s*(?P<max_reserved>[\d.]+)"
)
PARAM_RE = re.compile(r"^(Total number of (?:active )?parameters in billions|"
                      r"Number of parameters in most loaded shard in billions):\s*([\d.]+)")

# Arguments worth keeping next to the metrics; anything else is noise for
# run-to-run comparison.
CONFIG_KEYS = (
    "num_layers hidden_size ffn_hidden_size num_attention_heads num_experts "
    "moe_ffn_hidden_size moe_router_topk moe_router_dtype "
    "moe_router_load_balancing_type moe_aux_loss_coeff moe_token_dispatcher_type "
    "moe_grouped_gemm moe_permute_fusion moe_expert_capacity_factor "
    "tensor_model_parallel_size pipeline_model_parallel_size "
    "expert_model_parallel_size world_size "
    "seq_length micro_batch_size global_batch_size train_iters "
    "optimizer lr min_lr lr_decay_style lr_warmup_iters lr_warmup_fraction "
    "adam_beta1 adam_beta2 adam_eps weight_decay clip_grad "
    "use_distributed_optimizer bf16 fp16 seed init_method_std "
    "data_path split tokenizer_type save load eval_iters eval_interval"
).split()


def parse_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = ITER_RE.search(line)
            if match is None:
                continue
            grad_norm = GRAD_NORM_RE.search(line)
            timestamp = TIMESTAMP_RE.search(line)
            tflops = match.group("tflops")
            rows.append(
                {
                    "iteration": int(match.group("iter")),
                    "timestamp": timestamp.group("ts") if timestamp else "",
                    "consumed_samples": int(match.group("samples")),
                    "lm_loss": float(match.group("loss")),
                    "learning_rate": float(match.group("lr")),
                    "grad_norm": float(grad_norm.group("gn")) if grad_norm and grad_norm.group("gn") != "nan" else math.nan,
                    "elapsed_ms": float(match.group("ms")),
                    "tflops_per_gpu": float(tflops) if tflops else math.nan,
                    "global_batch_size": int(match.group("gbs")),
                }
            )
    rows.sort(key=lambda row: row["iteration"])
    return rows


def parse_config(path: Path) -> dict[str, Any]:
    wanted = set(CONFIG_KEYS)
    config: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    with path.open(errors="ignore") as handle:
        for line in handle:
            match = ARG_RE.match(line)
            if match and match.group("key") in wanted and match.group("key") not in config:
                config[match.group("key")] = match.group("value")
                continue
            mem = MEMORY_RE.search(line)
            if mem:
                extras["max_reserved_mb"] = max(
                    float(mem.group("max_reserved")), extras.get("max_reserved_mb", 0.0)
                )
                continue
            param = PARAM_RE.match(line)
            if param:
                extras[param.group(1)] = float(param.group(2))
    ordered = {key: config[key] for key in CONFIG_KEYS if key in config}
    ordered.update(extras)
    return ordered


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return st.mean(finite) if finite else math.nan


def _pct(values: list[float], q: float) -> float:
    finite = sorted(v for v in values if not math.isnan(v))
    if not finite:
        return math.nan
    return finite[min(int(q * len(finite)), len(finite) - 1)]


def _slope_per_1k(rows: list[dict[str, Any]]) -> float:
    xs = [row["iteration"] for row in rows]
    ys = [row["lm_loss"] for row in rows]
    if len(xs) < 2:
        return math.nan
    mx, my = st.mean(xs), st.mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return math.nan
    return 1000.0 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def summarize(rows: list[dict[str, Any]], warmup: int, block: int, slow_factor: float) -> dict[str, Any]:
    steady = [row for row in rows if row["iteration"] > warmup]
    tail = rows[-500:]
    ms = [row["elapsed_ms"] for row in steady]
    tflops = [row["tflops_per_gpu"] for row in steady]
    grad_norms = [row["grad_norm"] for row in rows]
    # Relative to this run's own median, so the list stays meaningful when two runs have
    # different steady-state step times.
    slow_threshold = slow_factor * st.median(ms)

    blocks = []
    previous = None
    for start in range(rows[0]["iteration"], rows[-1]["iteration"] + 1, block):
        window = [row["lm_loss"] for row in rows if start <= row["iteration"] < start + block]
        if not window:
            continue
        mean = st.mean(window)
        blocks.append(
            {
                "first_iteration": start,
                "last_iteration": min(start + block - 1, rows[-1]["iteration"]),
                "lm_loss": round(mean, 4),
                "delta": None if previous is None else round(mean - previous, 4),
            }
        )
        previous = mean

    return {
        "iterations": {
            "logged": len(rows),
            "first": rows[0]["iteration"],
            "last": rows[-1]["iteration"],
            "warmup_excluded_from_perf": warmup,
        },
        "wall_clock": {
            "first_timestamp": rows[0]["timestamp"],
            "last_timestamp": rows[-1]["timestamp"],
            "sum_elapsed_minutes": round(sum(row["elapsed_ms"] for row in rows) / 60000.0, 2),
        },
        "loss": {
            "first": rows[0]["lm_loss"],
            "final": rows[-1]["lm_loss"],
            "last500_mean": round(_mean([row["lm_loss"] for row in tail]), 4),
            "last500_std": round(st.pstdev([row["lm_loss"] for row in tail]), 4),
            "last500_min": min(row["lm_loss"] for row in tail),
            "last500_max": max(row["lm_loss"] for row in tail),
            "final_perplexity": round(math.exp(_mean([row["lm_loss"] for row in tail])), 2),
            "slope_per_1k_last2000": round(_slope_per_1k(rows[-2000:]), 4),
            "slope_per_1k_last1000": round(_slope_per_1k(rows[-1000:]), 4),
            "by_block": blocks,
        },
        "throughput": {
            "tflops_per_gpu_mean": round(_mean(tflops), 2),
            "tflops_per_gpu_median": round(st.median(tflops), 2),
            "tflops_per_gpu_p5": round(_pct(tflops, 0.05), 2),
            "tflops_per_gpu_p95": round(_pct(tflops, 0.95), 2),
            "elapsed_ms_mean": round(_mean(ms), 1),
            "elapsed_ms_median": round(st.median(ms), 1),
            "elapsed_ms_p5": round(_pct(ms, 0.05), 1),
            "elapsed_ms_p95": round(_pct(ms, 0.95), 1),
            "elapsed_ms_max": round(max(ms), 1),
            "slow_iteration_threshold_ms": round(slow_threshold, 1),
            "slow_iteration_threshold_factor": slow_factor,
            "slow_iterations": [
                {"iteration": row["iteration"], "elapsed_ms": row["elapsed_ms"]}
                for row in steady
                if row["elapsed_ms"] > slow_threshold
            ],
        },
        "stability": {
            "nan_grad_norms": sum(1 for gn in grad_norms if math.isnan(gn)),
            "grad_norm_max": round(max(gn for gn in grad_norms if not math.isnan(gn)), 3),
            "grad_norm_mean_last2000": round(_mean([row["grad_norm"] for row in rows[-2000:]]), 3),
        },
        "data": {
            "consumed_samples": rows[-1]["consumed_samples"],
            "global_batch_size": rows[-1]["global_batch_size"],
        },
    }


def write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_curve_csv(rows: list[dict[str, Any]], path: Path, bucket: int) -> None:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["iteration"] - 1) // bucket, []).append(row)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "lm_loss", "tflops_per_gpu", "elapsed_ms", "samples_in_bucket"])
        for key in sorted(grouped):
            window = grouped[key]
            writer.writerow(
                [
                    window[-1]["iteration"],
                    round(_mean([r["lm_loss"] for r in window]), 4),
                    round(_mean([r["tflops_per_gpu"] for r in window]), 2),
                    round(_mean([r["elapsed_ms"] for r in window]), 1),
                    len(window),
                ]
            )


def write_readme(
    path: Path, run: str, summary: dict[str, Any], config: dict[str, Any], sources: list[str], bucket: int
) -> None:
    loss, thr, stab = summary["loss"], summary["throughput"], summary["stability"]
    lines = [
        f"# {run}",
        "",
        f"- iterations: {summary['iterations']['first']}–{summary['iterations']['last']} "
        f"({summary['iterations']['logged']} logged)",
        f"- wall clock: {summary['wall_clock']['first_timestamp']} → {summary['wall_clock']['last_timestamp']} "
        f"({summary['wall_clock']['sum_elapsed_minutes']} min of summed step time)",
        f"- final lm loss: {loss['last500_mean']} (mean of last 500 iterations, perplexity {loss['final_perplexity']})",
        f"- loss slope over the last 1000 iterations: {loss['slope_per_1k_last1000']} per 1k iterations",
        f"- throughput: {thr['tflops_per_gpu_median']} TFLOP/s/GPU median, "
        f"{thr['elapsed_ms_median']} ms per iteration median "
        f"(iterations after {summary['iterations']['warmup_excluded_from_perf']} only)",
        f"- stability: {stab['nan_grad_norms']} NaN grad norms, max grad norm {stab['grad_norm_max']}",
        "",
        "## Loss by block",
        "",
        "| iterations | mean lm loss | delta |",
        "| --- | --- | --- |",
    ]
    for entry in loss["by_block"]:
        delta = "—" if entry["delta"] is None else f"{entry['delta']:+.4f}"
        lines.append(
            f"| {entry['first_iteration']}–{entry['last_iteration']} | {entry['lm_loss']:.4f} | {delta} |"
        )
    lines += [
        "",
        "## Key configuration",
        "",
        "| argument | value |",
        "| --- | --- |",
    ]
    for key, value in config.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Files",
        "",
        "- `metrics.csv`: one row per iteration (loss, lr, grad norm, step time, throughput).",
        f"- `curve.csv`: the same series averaged over {bucket}-iteration buckets, for plotting.",
        "- `summary.json`: the statistics above in machine-readable form.",
        "- `config.json`: the training arguments identifying this run.",
        "",
        "## Sources",
        "",
    ]
    lines += [f"- `{source}`" for source in sources]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, type=Path, help="log containing the per-iteration lines (last rank)")
    parser.add_argument("--config-log", type=Path, help="log containing the argument dump (rank 0)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-name", help="defaults to the output directory name")
    parser.add_argument("--bucket", type=int, default=250, help="bucket width for curve.csv")
    parser.add_argument("--warmup", type=int, default=100, help="iterations excluded from throughput statistics")
    parser.add_argument("--block", type=int, default=1000, help="block width for the loss table")
    parser.add_argument(
        "--slow-factor",
        type=float,
        default=1.5,
        help="an iteration is listed as slow above this multiple of the run's median step time",
    )
    args = parser.parse_args()

    rows = parse_metrics(args.log)
    if not rows:
        raise SystemExit(f"no iteration lines found in {args.log}")

    config = parse_config(args.config_log) if args.config_log else {}
    summary = summarize(rows, warmup=args.warmup, block=args.block, slow_factor=args.slow_factor)
    seq_length = config.get("seq_length")
    if seq_length:
        summary["data"]["total_tokens"] = rows[-1]["consumed_samples"] * int(seq_length)

    run_name = args.run_name or args.out_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(rows, args.out_dir / "metrics.csv")
    write_curve_csv(rows, args.out_dir / "curve.csv", args.bucket)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    sources = [str(args.log)] + ([str(args.config_log)] if args.config_log else [])
    write_readme(args.out_dir / "README.md", run_name, summary, config, sources, args.bucket)
    print(f"wrote {len(rows)} iterations to {args.out_dir}")


if __name__ == "__main__":
    main()

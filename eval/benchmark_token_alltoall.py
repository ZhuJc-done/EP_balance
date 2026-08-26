#!/usr/bin/env python3
"""Benchmark pure NCCL All-to-All-v kernels at controlled routing imbalance.

The benchmark pre-allocates input/output buffers and split vectors before the
timed region. CUDA events enclose only ``dist.all_to_all_single``; barriers,
split exchange, allocation, packing and validation stay outside the interval.
Each result reports the maximum rank time because a synchronous MoE layer is
paced by the slowest rank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def parse_ratios(value: str) -> list[float]:
    ratios = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if not math.isfinite(ratio) or ratio < 1.0:
            raise argparse.ArgumentTypeError(
                f"max-over-mean ratio must be >= 1, got {ratio}"
            )
        ratios.append(ratio)
    if not ratios:
        raise argparse.ArgumentTypeError("at least one ratio is required")
    return ratios


def make_destination_splits(
    total_rows: int,
    world_size: int,
    max_over_mean: float,
) -> list[int]:
    """Return one source rank's splits with destination zero as the hotspot."""
    if total_rows <= 0:
        raise ValueError("total_rows must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 1.0 <= max_over_mean <= world_size:
        raise ValueError(
            f"max_over_mean must be in [1, {world_size}], got {max_over_mean}"
        )
    if world_size == 1:
        return [total_rows]

    hot_rows = max(
        math.ceil(total_rows / world_size),
        round(total_rows * max_over_mean / world_size),
    )
    hot_rows = min(total_rows, max(0, hot_rows))
    remaining = total_rows - hot_rows
    cold_rows, extra = divmod(remaining, world_size - 1)
    splits = [hot_rows]
    splits.extend(
        cold_rows + (1 if index < extra else 0)
        for index in range(world_size - 1)
    )
    assert sum(splits) == total_rows
    return splits


def summarize_samples(
    rank_samples_ms: list[list[float]],
    *,
    row_bytes: int,
    remote_send_rows: list[int],
    remote_recv_rows: list[int],
    intra_node_send_rows: list[int],
    inter_node_send_rows: list[int],
    inter_node_recv_rows: list[int],
    link_gbps: float,
) -> dict[str, Any]:
    """Summarize per-rank samples into synchronous critical-path metrics."""
    if not rank_samples_ms or not rank_samples_ms[0]:
        raise ValueError("rank_samples_ms cannot be empty")
    repeats = len(rank_samples_ms[0])
    if any(len(values) != repeats for values in rank_samples_ms):
        raise ValueError("all ranks must report the same sample count")
    rank_count = len(rank_samples_ms)
    vectors = (
        remote_send_rows,
        remote_recv_rows,
        intra_node_send_rows,
        inter_node_send_rows,
        inter_node_recv_rows,
    )
    if any(len(values) != rank_count for values in vectors):
        raise ValueError("payload vectors must contain one value per rank")

    critical_ms = [
        max(rank_samples_ms[rank][sample] for rank in range(rank_count))
        for sample in range(repeats)
    ]
    mean_rank_ms = [
        st.mean(rank_samples_ms[rank][sample] for rank in range(rank_count))
        for sample in range(repeats)
    ]
    critical_median = st.median(critical_ms)
    cluster_remote_bytes = sum(remote_send_rows) * row_bytes
    cluster_intra_node_bytes = sum(intra_node_send_rows) * row_bytes
    cluster_inter_node_bytes = sum(inter_node_send_rows) * row_bytes
    bottleneck_remote_bytes = max(
        max(remote_send_rows), max(remote_recv_rows)
    ) * row_bytes
    bottleneck_inter_node_bytes = max(
        max(inter_node_send_rows), max(inter_node_recv_rows)
    ) * row_bytes
    aggregate_GBps = cluster_remote_bytes / (critical_median * 1e6)
    bottleneck_GBps = bottleneck_remote_bytes / (critical_median * 1e6)
    bottleneck_inter_node_GBps = (
        bottleneck_inter_node_bytes / (critical_median * 1e6)
    )
    nominal_link_GBps = link_gbps / 8.0

    return {
        "samples": repeats,
        "critical_ms_min": round(min(critical_ms), 6),
        "critical_ms_p25": round(percentile(critical_ms, 0.25), 6),
        "critical_ms_median": round(critical_median, 6),
        "critical_ms_p75": round(percentile(critical_ms, 0.75), 6),
        "critical_ms_p95": round(percentile(critical_ms, 0.95), 6),
        "critical_ms_max": round(max(critical_ms), 6),
        "mean_rank_ms_median": round(st.median(mean_rank_ms), 6),
        "critical_over_mean_median": round(
            st.median(
                critical / mean
                for critical, mean in zip(critical_ms, mean_rank_ms)
                if mean
            ),
            6,
        ),
        "cluster_remote_mib": round(cluster_remote_bytes / 2**20, 3),
        "cluster_intra_node_mib": round(cluster_intra_node_bytes / 2**20, 3),
        "cluster_inter_node_mib": round(
            cluster_inter_node_bytes / 2**20, 3
        ),
        "bottleneck_remote_mib": round(
            bottleneck_remote_bytes / 2**20, 3
        ),
        "bottleneck_inter_node_mib": round(
            bottleneck_inter_node_bytes / 2**20, 3
        ),
        "aggregate_payload_GBps": round(aggregate_GBps, 3),
        "bottleneck_payload_GBps": round(bottleneck_GBps, 3),
        "bottleneck_inter_node_GBps": round(
            bottleneck_inter_node_GBps, 3
        ),
        "nominal_link_utilization": round(
            bottleneck_inter_node_GBps / nominal_link_GBps, 6
        )
        if nominal_link_GBps and bottleneck_inter_node_bytes
        else 0.0,
        "critical_samples_ms": [round(value, 6) for value in critical_ms],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument(
        "--tokens-per-rank",
        type=int,
        default=4096,
        help="input tokens before Top-k expansion",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--row-elements",
        type=int,
        default=2048,
        help="elements in one transported routing-unit row",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=parse_ratios("1,2,4,8"),
        help="comma-separated destination max/mean receive ratios",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--link-gbps",
        type=float,
        default=400.0,
        help="nominal per-GPU inter-node link rate in gigabits/s",
    )
    parser.add_argument(
        "--max-buffer-mib",
        type=float,
        default=8192.0,
        help="fail before allocating a larger per-rank input or output buffer",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("token_alltoall"),
        help="writes PREFIX.json and PREFIX.csv on global rank zero",
    )
    return parser.parse_args()


def make_ep_group(
    rank: int,
    world_size: int,
    ep_size: int,
) -> tuple[Any, list[int], int, int]:
    if world_size % ep_size:
        raise ValueError(
            f"world size {world_size} must be divisible by EP size {ep_size}"
        )
    selected = None
    selected_ranks = None
    selected_index = -1
    selected_local_rank = -1
    for group_index, start in enumerate(range(0, world_size, ep_size)):
        ranks = list(range(start, start + ep_size))
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            selected = group
            selected_ranks = ranks
            selected_index = group_index
            selected_local_rank = rank - start
    assert selected is not None and selected_ranks is not None
    return selected, selected_ranks, selected_index, selected_local_rank


def exchange_recv_splits(
    send_splits: list[int],
    *,
    group: Any,
    local_rank: int,
    device: torch.device,
) -> list[int]:
    send_tensor = torch.tensor(send_splits, dtype=torch.int64, device=device)
    gathered = [torch.empty_like(send_tensor) for _ in send_splits]
    dist.all_gather(gathered, send_tensor, group=group)
    return [int(source[local_rank].item()) for source in gathered]


def payload_rows_by_scope(
    *,
    send_splits: list[int],
    recv_splits: list[int],
    group_ranks: list[int],
    global_rank: int,
    local_ep_rank: int,
    local_world_size: int,
) -> list[int]:
    source_node = global_rank // local_world_size
    remote_send = 0
    intra_send = 0
    inter_send = 0
    for destination_local, rows in enumerate(send_splits):
        destination_global = group_ranks[destination_local]
        if destination_global == global_rank:
            continue
        remote_send += rows
        if destination_global // local_world_size == source_node:
            intra_send += rows
        else:
            inter_send += rows

    remote_recv = 0
    inter_recv = 0
    for source_local, rows in enumerate(recv_splits):
        source_global = group_ranks[source_local]
        if source_global == global_rank:
            continue
        remote_recv += rows
        if source_global // local_world_size != source_node:
            inter_recv += rows
    assert send_splits[local_ep_rank] + remote_send == sum(send_splits)
    return [remote_send, remote_recv, intra_send, inter_send, inter_recv]


def gather_rank_vectors(local: torch.Tensor) -> list[list[float]]:
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [item.cpu().tolist() for item in gathered]


@torch.inference_mode()
def run_case(
    *,
    ratio: float,
    total_rows: int,
    row_elements: int,
    dtype: torch.dtype,
    group: Any,
    group_ranks: list[int],
    local_ep_rank: int,
    global_rank: int,
    device: torch.device,
    local_world_size: int,
    warmup: int,
    repeats: int,
    max_buffer_mib: float,
    link_gbps: float,
) -> dict[str, Any] | None:
    ep_size = len(group_ranks)
    send_splits = make_destination_splits(total_rows, ep_size, ratio)
    recv_splits = exchange_recv_splits(
        send_splits,
        group=group,
        local_rank=local_ep_rank,
        device=device,
    )
    input_rows = sum(send_splits)
    output_rows = sum(recv_splits)
    row_bytes = row_elements * torch.empty((), dtype=dtype).element_size()
    max_output_rows = ep_size * max(send_splits)
    largest_buffer_mib = max(input_rows, max_output_rows) * row_bytes / 2**20
    if largest_buffer_mib > max_buffer_mib:
        raise ValueError(
            f"ratio {ratio:g} needs a {largest_buffer_mib:.1f} MiB buffer on "
            f"rank {global_rank}, above --max-buffer-mib={max_buffer_mib:g}"
        )

    source = torch.empty(
        (input_rows, row_elements), dtype=dtype, device=device
    )
    destination = torch.empty(
        (output_rows, row_elements), dtype=dtype, device=device
    )
    source.fill_(global_rank % 17)

    def collective() -> None:
        dist.all_to_all_single(
            destination,
            source,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=group,
            async_op=False,
        )

    for _ in range(warmup):
        # Align distinct EP groups so they exert concurrent network pressure.
        dist.barrier()
        torch.cuda.synchronize(device)
        collective()
        torch.cuda.synchronize(device)

    samples = []
    for _ in range(repeats):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        collective()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    rank_samples = gather_rank_vectors(
        torch.tensor(samples, dtype=torch.float64, device=device)
    )
    payload_rows = payload_rows_by_scope(
        send_splits=send_splits,
        recv_splits=recv_splits,
        group_ranks=group_ranks,
        global_rank=global_rank,
        local_ep_rank=local_ep_rank,
        local_world_size=local_world_size,
    )
    gathered_payload = gather_rank_vectors(
        torch.tensor(payload_rows, dtype=torch.int64, device=device)
    )
    del source, destination
    torch.cuda.empty_cache()

    if global_rank != 0:
        return None
    payload_columns = list(zip(*gathered_payload))
    summary = summarize_samples(
        rank_samples,
        row_bytes=row_bytes,
        remote_send_rows=[int(value) for value in payload_columns[0]],
        remote_recv_rows=[int(value) for value in payload_columns[1]],
        intra_node_send_rows=[int(value) for value in payload_columns[2]],
        inter_node_send_rows=[int(value) for value in payload_columns[3]],
        inter_node_recv_rows=[int(value) for value in payload_columns[4]],
        link_gbps=link_gbps,
    )
    actual_ratio = ep_size * max(send_splits) / total_rows
    summary.update(
        {
            "max_over_mean_requested": ratio,
            "max_over_mean_actual": round(actual_ratio, 6),
            "send_splits_rows": send_splits,
            "hot_destination": 0,
            "max_input_rows": input_rows,
            "max_output_rows": max_output_rows,
        }
    )
    return summary


def write_outputs(
    prefix: Path,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    prefix = prefix.expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    columns = [
        "max_over_mean_requested",
        "max_over_mean_actual",
        "samples",
        "critical_ms_min",
        "critical_ms_p25",
        "critical_ms_median",
        "critical_ms_p75",
        "critical_ms_p95",
        "critical_ms_max",
        "mean_rank_ms_median",
        "critical_over_mean_median",
        "cluster_remote_mib",
        "cluster_intra_node_mib",
        "cluster_inter_node_mib",
        "bottleneck_remote_mib",
        "bottleneck_inter_node_mib",
        "aggregate_payload_GBps",
        "bottleneck_payload_GBps",
        "bottleneck_inter_node_GBps",
        "nominal_link_utilization",
        "max_input_rows",
        "max_output_rows",
        "hot_destination",
        "send_splits_rows",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in report["results"]:
            writer.writerow(
                {
                    column: json.dumps(result[column])
                    if column == "send_splits_rows"
                    else result[column]
                    for column in columns
                }
            )
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.tokens_per_rank <= 0 or args.topk <= 0 or args.row_elements <= 0:
        raise ValueError("tokens, topk and row elements must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        group, group_ranks, group_index, local_ep_rank = make_ep_group(
            global_rank, world_size, args.ep_size
        )
        total_rows = args.tokens_per_rank * args.topk
        dtype = DTYPES[args.dtype]
        row_bytes = args.row_elements * torch.empty(
            (), dtype=dtype
        ).element_size()
        results = []
        for ratio in args.ratios:
            if ratio > args.ep_size:
                raise ValueError(
                    f"ratio {ratio:g} exceeds EP size {args.ep_size}"
                )
            result = run_case(
                ratio=ratio,
                total_rows=total_rows,
                row_elements=args.row_elements,
                dtype=dtype,
                group=group,
                group_ranks=group_ranks,
                local_ep_rank=local_ep_rank,
                global_rank=global_rank,
                device=device,
                local_world_size=local_world_size,
                warmup=args.warmup,
                repeats=args.repeats,
                max_buffer_mib=args.max_buffer_mib,
                link_gbps=args.link_gbps,
            )
            if result is not None:
                results.append(result)
                print(
                    f"ratio={result['max_over_mean_actual']:.3f} "
                    f"critical_p50={result['critical_ms_median']:.3f}ms "
                    f"critical_p95={result['critical_ms_p95']:.3f}ms "
                    f"aggregate={result['aggregate_payload_GBps']:.1f}GB/s "
                    f"inter_node_bottleneck="
                    f"{result['bottleneck_inter_node_GBps']:.1f}GB/s"
                )

        if global_rank == 0:
            report = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "measurement": {
                    "timed_region": "torch.distributed.all_to_all_single only",
                    "timer": "CUDA events",
                    "allocation_in_timed_region": False,
                    "barrier_in_timed_region": False,
                    "packing_in_timed_region": False,
                    "critical_definition": "max rank latency for every repeat",
                },
                "configuration": {
                    "world_size": world_size,
                    "local_world_size": local_world_size,
                    "nodes": world_size // local_world_size,
                    "ep_size": args.ep_size,
                    "ep_groups": world_size // args.ep_size,
                    "group_index_of_rank_zero": group_index,
                    "tokens_per_rank": args.tokens_per_rank,
                    "topk": args.topk,
                    "routing_rows_per_rank": total_rows,
                    "row_elements": args.row_elements,
                    "dtype": args.dtype,
                    "row_bytes": row_bytes,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "nominal_link_gbps": args.link_gbps,
                    "device": torch.cuda.get_device_name(device),
                    "torch_version": torch.__version__,
                    "nccl_version": torch.cuda.nccl.version(),
                },
                "results": results,
            }
            json_path, csv_path = write_outputs(args.output_prefix, report)
            print(f"saved {json_path}")
            print(f"saved {csv_path}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

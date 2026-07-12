"""Benchmark Scale-EPLB against DeepSeek EPLB and optional compiled LPLB.

Run from the repository root:
    python -m baseline.benchmark --strategies scale,eplb,lplb
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import torch

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from sim.workload import make_loads

from .adapters import (
    LPLBBaseline,
    LPLBUnavailableError,
    run_deepseek_eplb,
    run_scale_eplb,
)
from .deepseek_eplb import rebalance_experts


def _time_cpu(fn: Callable[[], Any], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) * 1_000 / iterations


def _time_cuda(fn: Callable[[], Any], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _time_mixed_cuda(fn: Callable[[], Any], warmup: int, iterations: int) -> float:
    """Wall-time a path containing both host work and asynchronous CUDA work."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000 / iterations


def _row(name: str, result, **timings: float | None) -> dict[str, Any]:
    return {
        "strategy": name,
        **timings,
        "quality_tau": result.tau,
        "quality_mean_load": result.mean_load,
        "quality_imbalance": result.imbalance,
        **result.metadata,
    }


def benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    strategies = {x.strip().lower() for x in args.strategies.split(",") if x.strip()}
    unknown = strategies - {"scale", "eplb", "lplb"}
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")

    ranks = args.nodes * args.gpus_per_node
    if "lplb" in strategies and args.experts % ranks != 0:
        raise ValueError("--experts must be divisible by total ranks for a fair LPLB layout")
    if args.n_slot * ranks < args.experts:
        raise ValueError("--n-slot is too small to place every logical expert")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loads = make_loads(
        ranks,
        args.experts,
        args.tokens_per_rank,
        args.top_k,
        args.skew,
        args.hotspot_ranks,
        args.seed,
        device=device,
    )
    history_loads = make_loads(
        ranks,
        args.experts,
        args.tokens_per_rank,
        args.top_k,
        args.history_skew,
        args.hotspot_ranks,
        args.history_seed,
        device=device,
    )
    topology = Topology.from_nvlink_rdma(
        args.nodes,
        args.gpus_per_node,
        args.intra_cost,
        args.inter_cost,
        device=device,
    )
    spec = ProblemSpec.uniform_main_placement(
        args.experts,
        ranks,
        args.weight_bytes,
        args.s_tok,
        args.n_slot,
        device=device,
    )
    cfg = EPLBConfig()
    rows: list[dict[str, Any]] = []

    if "scale" in strategies:
        timer = _time_cuda if device.type == "cuda" else _time_cpu
        latency = timer(
            lambda: solve(loads, topology, spec, cfg, validate=False),
            args.warmup,
            args.iterations,
        )
        result = run_scale_eplb(loads, topology, spec, cfg)
        rows.append(
            _row(
                "scale-eplb",
                result,
                solve_ms=latency,
                placement_ms=None,
                routing_ms=None,
                total_ms=latency,
            )
        )

    if "eplb" in strategies:
        num_physical = ranks * args.n_slot

        def deepseek_placement():
            # Include reduction and GPU->CPU transfer when the common input is CUDA.
            history_expert_load = history_loads.expert_load().cpu().unsqueeze(0)
            return rebalance_experts(
                history_expert_load,
                num_physical,
                args.num_groups,
                args.nodes,
                ranks,
            )

        placement_ms = (
            _time_mixed_cuda(deepseek_placement, args.warmup, args.iterations)
            if device.type == "cuda"
            else _time_cpu(deepseek_placement, args.warmup, args.iterations)
        )
        result = run_deepseek_eplb(
            loads,
            num_nodes=args.nodes,
            num_gpus=ranks,
            n_slot=args.n_slot,
            num_groups=args.num_groups,
            placement_loads=history_loads,
        )
        rows.append(
            _row(
                "deepseek-eplb",
                result,
                solve_ms=None,
                placement_ms=placement_ms,
                routing_ms=None,
                total_ms=None,
            )
        )

    if "lplb" in strategies:
        try:
            planner = LPLBBaseline(
                num_experts=args.experts,
                ep_size=ranks,
                n_slot=args.n_slot,
                lplb_root=args.lplb_root,
            )
        except LPLBUnavailableError as exc:
            if args.require_lplb:
                raise
            rows.append({"strategy": "lplb", "skipped": str(exc)})
        else:
            workload = loads.expert_load().to(device="cuda", dtype=torch.int32)
            history_workload = history_loads.expert_load().to(
                device="cuda", dtype=torch.int32
            )
            mapping_ms = _time_mixed_cuda(
                lambda: planner.update_mapping(history_workload),
                min(args.warmup, 1),
                max(1, min(args.iterations, args.mapping_iterations)),
            )
            planner.update_mapping(history_workload)
            available = torch.zeros((), dtype=torch.int32, device="cuda")

            def solve_lplb():
                available.zero_()
                return planner.planner.solve_probs(workload, available)

            lp_solve_ms = _time_cuda(
                solve_lplb,
                args.warmup,
                args.iterations,
            )
            result = planner.solve(workload)
            rows.append(
                _row(
                    "lplb",
                    result,
                    solve_ms=lp_solve_ms,
                    placement_ms=mapping_ms,
                    routing_ms=None,
                    total_ms=None,
                )
            )

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", default="scale,eplb,lplb")
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--n-slot", type=int, default=4)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--skew", type=float, default=1.5)
    parser.add_argument("--history-skew", type=float, default=1.3)
    parser.add_argument("--hotspot-ranks", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-seed", type=int, default=1)
    parser.add_argument("--intra-cost", type=int, default=1)
    parser.add_argument("--inter-cost", type=int, default=8)
    parser.add_argument("--s-tok", type=int, default=7168 * 2)
    parser.add_argument("--weight-bytes", type=int, default=44_000_000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mapping-iterations", type=int, default=3)
    parser.add_argument("--lplb-root", default="/home/tiger/LPLB")
    parser.add_argument("--require-lplb", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = benchmark(args)
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for row in rows:
        if "skipped" in row:
            print(f"{row['strategy']:<16} skipped: {row['skipped']}")
            continue
        timings = ", ".join(
            f"{key}={row[key]:.3f} ms"
            for key in ("placement_ms", "solve_ms", "routing_ms", "total_ms")
            if row.get(key) is not None
        )
        print(
            f"{row['strategy']:<16} {timings}, "
            f"quality_tau={row['quality_tau']:.1f}, "
            f"quality_imbalance={row['quality_imbalance']:.4f}, "
            f"quality={row['load_kind']}"
        )


if __name__ == "__main__":
    main()

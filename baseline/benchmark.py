"""Benchmark Scale-EPLB against DeepSeek EPLB, FasterMoE, FlexMoE, and optional LPLB.

Run from the repository root on a synthetic workload:
    python -m baseline.benchmark --strategies scale,eplb,fastermoe,flexmoe,lplb

Or replay a real routing trace captured during observe-mode training
(``EPLB_TRACE_OUT=trace.pt EPLB_MODE=observe bash scripts/run_real_moe.sh``):
    python -m baseline.benchmark --trace trace.pt --strategies scale,eplb,fastermoe,flexmoe,lplb
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import torch

from eplb import EPLBConfig, ProblemSpec, Topology, solve
from eplb.loads import Loads
from sim.workload import make_loads

from .adapters import (
    FlexMoECostModel,
    LPLBBaseline,
    LPLBUnavailableError,
    ShadowCostModel,
    run_deepseek_eplb,
    run_fastermoe,
    run_flexmoe,
    run_scale_eplb,
)
from .deepseek_eplb import rebalance_experts
from .fastermoe import select_shadow_experts
from .flexmoe import flexmoe_schedule


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
        "quality_theta": result.theta,
        "quality_mean_load": result.mean_load,
        "quality_imbalance": result.imbalance,
        **result.metadata,
    }


def benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    strategies = {x.strip().lower() for x in args.strategies.split(",") if x.strip()}
    unknown = strategies - {"scale", "eplb", "fastermoe", "flexmoe", "lplb"}
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

    if "fastermoe" in strategies:
        fm_cost = ShadowCostModel(
            bw_net=args.fastermoe_bw_net,
            bw_mm=args.fastermoe_bw_mm,
        )

        def fastermoe_selection():
            # Include the GPU->CPU transfer of the load matrix when the input is CUDA.
            return select_shadow_experts(
                loads.omega,
                spec.main_rank,
                spec.weight_bytes,
                spec.s_tok,
                ranks,
                fm_cost,
            )

        selection_ms = (
            _time_mixed_cuda(fastermoe_selection, args.warmup, args.iterations)
            if device.type == "cuda"
            else _time_cpu(fastermoe_selection, args.warmup, args.iterations)
        )
        result = run_fastermoe(loads, spec, num_ranks=ranks, cost=fm_cost)
        rows.append(
            _row(
                "fastermoe",
                result,
                solve_ms=selection_ms,
                placement_ms=None,
                routing_ms=None,
                total_ms=selection_ms,
            )
        )

    if "flexmoe" in strategies:
        fm_cost = FlexMoECostModel(threshold=args.flexmoe_threshold)

        def flexmoe_scheduling():
            # Include the GPU->CPU transfer of the load matrix when the input is CUDA.
            return flexmoe_schedule(
                loads.omega, spec.weight_bytes, ranks, args.n_slot, fm_cost
            )

        schedule_ms = (
            _time_mixed_cuda(flexmoe_scheduling, args.warmup, args.iterations)
            if device.type == "cuda"
            else _time_cpu(flexmoe_scheduling, args.warmup, args.iterations)
        )
        result = run_flexmoe(loads, spec, num_ranks=ranks, cost=fm_cost)
        rows.append(
            _row(
                "flexmoe",
                result,
                solve_ms=schedule_ms,
                placement_ms=None,
                routing_ms=None,
                total_ms=schedule_ms,
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


# --- real-trace replay --------------------------------------------------------
def _load_trace(
    path: str, device: torch.device
) -> tuple[dict[str, Any], list[dict[str, Any]], Topology, ProblemSpec]:
    """Load a routing trace dumped by observe mode and rebuild its Topology/ProblemSpec."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    meta, samples = blob["meta"], blob["samples"]
    topo = Topology(
        meta["domain_of_rank"].to(device),
        meta["cost"].to(device),
        num_domains_hint=int(meta["num_domains"]),
    )
    spec = ProblemSpec(
        int(meta["num_experts"]),
        meta["main_rank"].to(device),
        meta["weight_bytes"].to(device),
        int(meta["s_tok"]),
        int(meta["n_slot"]),
    )
    topo.validate()
    spec.validate(topo.num_ranks)
    return meta, samples, topo, spec


def _safe_num_groups(num_experts: int, num_nodes: int, requested: int) -> int:
    """Pick a DeepSeek group count that satisfies its divisibility constraints for this trace."""
    if requested and num_experts % requested == 0 and requested % num_nodes == 0:
        return requested
    if num_nodes >= 1 and num_experts % num_nodes == 0:
        return num_nodes
    return 1


def _run_timed(fn: Callable[[], Any], device: torch.device) -> tuple[Any, float]:
    """Run ``fn`` once, returning its result and wall time in ms (CUDA-synced)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1_000


def _agg(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def q(p: float) -> float:
        return ordered[min(n - 1, max(0, round(p * (n - 1))))]

    return {"mean": sum(values) / n, "p50": q(0.5), "p90": q(0.9)}


def benchmark_trace(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Replay a real routing trace through every selected baseline and aggregate quality/latency.

    Each sample's own ``Ω`` is fed to every strategy (i.e. every baseline replans
    per micro-batch from that batch's load), so the reported numbers are apples-to-apples
    load-balance quality on the real routing distribution.
    """
    strategies = {x.strip().lower() for x in args.strategies.split(",") if x.strip()}
    unknown = strategies - {"scale", "eplb", "fastermoe", "flexmoe", "lplb"}
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, samples, topo, spec = _load_trace(args.trace, device)
    ranks, experts, n_slot = topo.num_ranks, spec.num_experts, spec.n_slot
    num_nodes = topo.num_domains
    if args.trace_max_samples:
        samples = samples[: args.trace_max_samples]
    if not samples:
        raise ValueError(f"trace {args.trace!r} contains no samples")

    cfg = EPLBConfig()
    fm_cost = ShadowCostModel(bw_net=args.fastermoe_bw_net, bw_mm=args.fastermoe_bw_mm)
    flex_cost = FlexMoECostModel(threshold=args.flexmoe_threshold)
    num_groups = _safe_num_groups(experts, num_nodes, args.num_groups)

    lplb_planner = None
    lplb_skip: str | None = None
    if "lplb" in strategies:
        try:
            lplb_planner = LPLBBaseline(
                num_experts=experts, ep_size=ranks, n_slot=n_slot, lplb_root=args.lplb_root
            )
        except (LPLBUnavailableError, ValueError) as exc:
            if args.require_lplb:
                raise
            lplb_skip = str(exc)

    acc: dict[str, dict[str, Any]] = {}

    def record(key: str, result, ms: float) -> None:
        bucket = acc.setdefault(
            key, {"theta": [], "imb": [], "ms": [], "name": key, "meta": {}}
        )
        bucket["theta"].append(result.theta)
        bucket["imb"].append(result.imbalance)
        bucket["ms"].append(ms)
        bucket["name"] = result.name
        bucket["meta"] = result.metadata

    for sample in samples:
        loads = Loads(sample["omega"].to(device))
        if "scale" in strategies:
            res, ms = _run_timed(lambda ld=loads: run_scale_eplb(ld, topo, spec, cfg), device)
            record("scale", res, ms)
        if "eplb" in strategies:
            res, ms = _run_timed(
                lambda ld=loads: run_deepseek_eplb(
                    ld,
                    num_nodes=num_nodes,
                    num_gpus=ranks,
                    n_slot=n_slot,
                    num_groups=num_groups,
                    placement_loads=ld,
                ),
                device,
            )
            record("eplb", res, ms)
        if "fastermoe" in strategies:
            res, ms = _run_timed(
                lambda ld=loads: run_fastermoe(ld, spec, num_ranks=ranks, cost=fm_cost), device
            )
            record("fastermoe", res, ms)
        if "flexmoe" in strategies:
            res, ms = _run_timed(
                lambda ld=loads: run_flexmoe(ld, spec, num_ranks=ranks, cost=flex_cost), device
            )
            record("flexmoe", res, ms)
        if "lplb" in strategies and lplb_planner is not None:
            workload = loads.expert_load().to(device="cuda", dtype=torch.int32)

            def solve_lplb(wl=workload, planner=lplb_planner) -> Any:
                planner.update_mapping(wl)
                return planner.solve(wl)

            res, ms = _run_timed(solve_lplb, device)
            record("lplb", res, ms)

    rows: list[dict[str, Any]] = []
    for key in ("scale", "eplb", "fastermoe", "flexmoe", "lplb"):
        if key not in strategies:
            continue
        if key == "lplb" and lplb_planner is None:
            rows.append({"strategy": "lplb", "skipped": lplb_skip})
            continue
        bucket = acc[key]
        theta, imb, ms = (
            _agg(bucket["theta"]),
            _agg(bucket["imb"]),
            _agg(bucket["ms"]),
        )
        rows.append(
            {
                "strategy": bucket["name"],
                "samples": len(bucket["theta"]),
                "solve_ms_mean": ms["mean"],
                "quality_theta_mean": theta["mean"],
                "quality_imbalance_mean": imb["mean"],
                "quality_imbalance_p90": imb["p90"],
                "load_kind": bucket["meta"].get("load_kind", ""),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", default="scale,eplb,fastermoe,flexmoe,lplb")
    parser.add_argument(
        "--trace",
        default=None,
        help="Replay a real routing trace (dumped by EPLB_TRACE_OUT in observe mode) "
        "through every strategy instead of a synthetic workload.",
    )
    parser.add_argument(
        "--trace-max-samples",
        type=int,
        default=0,
        help="Cap the number of trace (layer, micro-batch) samples replayed (0 = all).",
    )
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
    parser.add_argument("--fastermoe-bw-net", type=float, default=50e9 / 8)
    parser.add_argument("--fastermoe-bw-mm", type=float, default=11.5e12)
    parser.add_argument("--flexmoe-threshold", type=float, default=1.2)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _print_trace_rows(rows: list[dict[str, Any]]) -> None:
    non_skipped = [r for r in rows if "skipped" not in r]
    samples = non_skipped[0]["samples"] if non_skipped else 0
    print(f"replaying {samples} trace sample(s) through {len(rows)} strategy(ies):")
    for row in rows:
        if "skipped" in row:
            print(f"{row['strategy']:<16} skipped: {row['skipped']}")
            continue
        print(
            f"{row['strategy']:<16} solve_ms(mean)={row['solve_ms_mean']:.3f}, "
            f"theta(mean)={row['quality_theta_mean']:.1f}, "
            f"imbalance(mean)={row['quality_imbalance_mean']:.4f}, "
            f"imbalance(p90)={row['quality_imbalance_p90']:.4f}, "
            f"quality={row['load_kind']}"
        )


def main() -> None:
    args = _parse_args()
    if args.trace:
        rows = benchmark_trace(args)
        if args.json:
            print(json.dumps(rows, indent=2))
            return
        _print_trace_rows(rows)
        return

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
            f"quality_theta={row['quality_theta']:.1f}, "
            f"quality_imbalance={row['quality_imbalance']:.4f}, "
            f"quality={row['load_kind']}"
        )


if __name__ == "__main__":
    main()

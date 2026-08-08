"""Benchmark Scale-EPLB against DeepSeek EPLB, FasterMoE, FlexMoE, and optional LPLB.

Run from the repository root on a synthetic workload:
    python -m baseline.benchmark --strategies scale,eplb,fastermoe,flexmoe,lplb

Or replay a real routing trace captured during observe-mode training
(``EPLB_TRACE_OUT=trace.pt EPLB_MODE=observe bash scripts/run_real_moe.sh``):
    python -m baseline.benchmark --trace trace.pt --strategies scale,eplb,fastermoe,flexmoe,lplb
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
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
        # Element i is the token-expert assignment load evaluated for rank i.
        "quality_rank_load_by_rank": [
            float(value)
            for value in result.rank_load.detach()
            .to(device="cpu", dtype=torch.float64)
            .tolist()
        ],
        **result.metadata,
    }


def _slot_counts(
    num_experts: int, num_ranks: int, n_slot: int
) -> tuple[int, int]:
    """Return main and total physical slots per rank.

    ``n_slot`` is the benchmark-wide budget of *additional replica experts* on
    each rank. All compared constrained strategies receive this same budget.
    """
    if num_ranks <= 0 or num_experts % num_ranks != 0:
        raise ValueError("--experts must be divisible by total ranks")
    if int(n_slot) < 0:
        raise ValueError("--n-slot must be non-negative")
    main_slots = num_experts // num_ranks
    return main_slots, main_slots + int(n_slot)


def _no_balance_row(loads: Loads, spec: ProblemSpec, num_ranks: int) -> dict[str, Any]:
    """Evaluate the fixed main-expert placement without any balancing."""
    expert_load = loads.expert_load()
    rank_load = torch.zeros(
        num_ranks, dtype=expert_load.dtype, device=expert_load.device
    )
    rank_load.index_add_(0, spec.main_rank, expert_load)
    rank_load_f64 = rank_load.to(torch.float64)
    theta = float(rank_load_f64.max().item())
    mean_load = float(rank_load_f64.mean().item())
    main_slots, _ = _slot_counts(spec.num_experts, num_ranks, 0)
    return {
        "strategy": "no-balance",
        "solve_ms": None,
        "placement_ms": None,
        "routing_ms": None,
        "total_ms": None,
        "quality_theta": theta,
        "quality_mean_load": mean_load,
        "quality_imbalance": theta / mean_load if mean_load > 0 else 1.0,
        "quality_rank_load_by_rank": [
            float(value) for value in rank_load_f64.cpu().tolist()
        ],
        "load_kind": "static main-rank token load",
        "placement_kind": "fixed main experts only",
        "main_slots_per_rank": main_slots,
        "replica_slots_per_rank": 0,
        "physical_slots_per_rank": main_slots,
        "replica_budget": 0,
        "physical_instances": spec.num_experts,
        "replicas": 0,
    }


def _placement_details(
    result: Any,
    loads: Loads,
    spec: ProblemSpec,
) -> dict[str, Any]:
    """Convert a placement matrix into explicit relocation and replica events.

    These are planning-derived logical transfers, not measurements from the
    communication backend. For strategies that do not retain the common
    ``main(e)`` placement, one hosted instance is classified as the relocated
    base and all remaining instances are classified as replicas.
    """
    if result.placement is None:
        return {
            "available": False,
            "reason": "strategy did not expose a physical placement",
        }

    placement = result.placement.detach().to(device="cpu", dtype=torch.int64)
    num_experts, num_ranks = placement.shape
    if num_experts != spec.num_experts or num_ranks != loads.num_ranks:
        raise ValueError(
            "placement shape does not match the common expert/rank dimensions"
        )
    main_rank = spec.main_rank.detach().to(device="cpu", dtype=torch.int64)
    per_expert_instances = placement.sum(dim=1)
    if torch.any(per_expert_instances < 1):
        missing = torch.nonzero(per_expert_instances < 1).flatten().tolist()
        raise ValueError(f"placement omitted logical expert(s): {missing}")

    relocation_events: list[dict[str, Any]] = []
    replica_events: list[dict[str, Any]] = []
    changed_experts: list[dict[str, Any]] = []
    replica_instances_per_rank = [0] * num_ranks
    relocated_bases_per_rank = [0] * num_ranks

    for expert in range(num_experts):
        source = int(main_rank[expert].item())
        counts = [int(x) for x in placement[expert].tolist()]
        if counts[source] > 0:
            base_rank = source
        else:
            # Reporting-only classification for reordering baselines: choose a
            # deterministic hosted instance as the logical base.
            base_rank = max(range(num_ranks), key=lambda rank: (counts[rank], -rank))
            relocation_events.append(
                {
                    "expert": expert,
                    "source_main_rank": source,
                    "destination_rank": base_rank,
                }
            )
            relocated_bases_per_rank[base_rank] += 1

        replica_counts = counts.copy()
        replica_counts[base_rank] -= 1
        for destination, instances in enumerate(replica_counts):
            if instances <= 0:
                continue
            replica_events.append(
                {
                    "expert": expert,
                    "source_main_rank": source,
                    "destination_rank": destination,
                    "instances": instances,
                    "local_to_main": destination == source,
                }
            )
            replica_instances_per_rank[destination] += instances

        if base_rank != source or int(per_expert_instances[expert].item()) > 1:
            changed_experts.append(
                {
                    "expert": expert,
                    "source_main_rank": source,
                    "base_rank": base_rank,
                    "hosts": [
                        {"rank": rank, "instances": count}
                        for rank, count in enumerate(counts)
                        if count > 0
                    ],
                }
            )

    physical_instances = int(placement.sum().item())
    replica_instances = sum(
        int(event["instances"]) for event in replica_events
    )
    expected_replicas = physical_instances - num_experts
    if replica_instances != expected_replicas:
        raise RuntimeError(
            "replica event accounting does not match physical placement"
        )

    expert_load = loads.expert_load().detach().to(device="cpu", dtype=torch.float64)
    static_main_load = torch.zeros(num_ranks, dtype=torch.float64)
    static_main_load.index_add_(0, main_rank, expert_load)
    planned_load = result.rank_load.detach().to(device="cpu", dtype=torch.float64)
    rank_summary = []
    for rank in range(num_ranks):
        main_experts = torch.nonzero(main_rank == rank).flatten().tolist()
        hosted = torch.nonzero(placement[:, rank] > 0).flatten().tolist()
        rank_summary.append(
            {
                "rank": rank,
                "main_experts": [int(expert) for expert in main_experts],
                "hosted_instances": [
                    {
                        "expert": int(expert),
                        "instances": int(placement[expert, rank].item()),
                    }
                    for expert in hosted
                ],
                "physical_instances": int(placement[:, rank].sum().item()),
                "replica_instances": replica_instances_per_rank[rank],
                "relocated_bases_in": relocated_bases_per_rank[rank],
                "static_main_load_before": float(static_main_load[rank].item()),
                "planned_load_after": float(planned_load[rank].item()),
            }
        )

    return {
        "available": True,
        "semantics": (
            "Derived from the planner placement; source_main_rank is the common "
            "initial main(e), not an observed communication trace."
        ),
        "summary": {
            "logical_experts": num_experts,
            "physical_instances": physical_instances,
            "replica_instances": replica_instances,
            "relocated_logical_experts": len(relocation_events),
            "remote_replica_instances": sum(
                int(event["instances"])
                for event in replica_events
                if not event["local_to_main"]
            ),
            "local_replica_instances": sum(
                int(event["instances"])
                for event in replica_events
                if event["local_to_main"]
            ),
        },
        "relocation_events": relocation_events,
        "replica_transfer_events": replica_events,
        "changed_experts": changed_experts,
        "rank_summary": rank_summary,
    }


def _attach_placement_details(
    result: Any,
    args: argparse.Namespace,
    loads: Loads,
    spec: ProblemSpec,
) -> None:
    if getattr(args, "details", False):
        result.metadata["placement_details"] = _placement_details(
            result, loads, spec
        )


def benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    strategies = {x.strip().lower() for x in args.strategies.split(",") if x.strip()}
    unknown = strategies - {"scale", "eplb", "fastermoe", "flexmoe", "lplb"}
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")

    ranks = args.nodes * args.gpus_per_node
    _, physical_slots_per_rank = _slot_counts(
        args.experts, ranks, args.n_slot
    )

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
        physical_slots_per_rank,
        device=device,
    )
    cfg = EPLBConfig()
    rows: list[dict[str, Any]] = []
    if getattr(args, "include_no_balance", False):
        rows.append(_no_balance_row(loads, spec, ranks))

    if "scale" in strategies:
        timer = _time_cuda if device.type == "cuda" else _time_cpu
        latency = timer(
            lambda: solve(loads, topology, spec, cfg, validate=False),
            args.warmup,
            args.iterations,
        )
        result = run_scale_eplb(loads, topology, spec, cfg)
        _attach_placement_details(result, args, loads, spec)
        rows.append(
            _row(
                "scale-eplb",
                result,
                solve_ms=latency,
                placement_ms=None,
                routing_ms=None,
                total_ms=(
                    None
                    if getattr(args, "solver_only_timing", False)
                    else latency
                ),
            )
        )

    if "eplb" in strategies:
        num_physical = args.experts + ranks * args.n_slot
        history_expert_load = (
            history_loads.expert_load().detach().cpu().unsqueeze(0)
        )

        def deepseek_placement():
            return rebalance_experts(
                history_expert_load,
                num_physical,
                args.num_groups,
                args.nodes,
                ranks,
            )

        deepseek_solve_ms = _time_cpu(
            deepseek_placement,
            args.warmup,
            args.iterations,
        )
        result = run_deepseek_eplb(
            loads,
            num_nodes=args.nodes,
            num_gpus=ranks,
            n_slot=args.n_slot,
            num_groups=args.num_groups,
            placement_loads=history_loads,
        )
        _attach_placement_details(result, args, loads, spec)
        rows.append(
            _row(
                "deepseek-eplb",
                result,
                solve_ms=deepseek_solve_ms,
                placement_ms=None,
                routing_ms=None,
                total_ms=None,
            )
        )

    if "fastermoe" in strategies:
        fm_cost = ShadowCostModel(
            bw_net=args.fastermoe_bw_net,
            bw_mm=args.fastermoe_bw_mm,
        )
        fastermoe_omega = loads.omega.detach().to(
            device="cpu", dtype=torch.float64
        )
        fastermoe_main_rank = spec.main_rank.detach().to(
            device="cpu", dtype=torch.int64
        )
        fastermoe_weight_bytes = spec.weight_bytes.detach().to(
            device="cpu", dtype=torch.float64
        )

        def fastermoe_selection():
            return select_shadow_experts(
                fastermoe_omega,
                fastermoe_main_rank,
                fastermoe_weight_bytes,
                spec.s_tok,
                ranks,
                fm_cost,
                args.n_slot,
            )

        selection_ms = _time_cpu(
            fastermoe_selection,
            args.warmup,
            args.iterations,
        )
        result = run_fastermoe(
            loads, spec, num_ranks=ranks, n_slot=args.n_slot, cost=fm_cost
        )
        _attach_placement_details(result, args, loads, spec)
        rows.append(
            _row(
                "fastermoe",
                result,
                solve_ms=selection_ms,
                placement_ms=None,
                routing_ms=None,
                total_ms=(
                    None
                    if getattr(args, "solver_only_timing", False)
                    else selection_ms
                ),
            )
        )

    if "flexmoe" in strategies:
        fm_cost = FlexMoECostModel(threshold=args.flexmoe_threshold)
        flexmoe_omega = loads.omega.detach().to(
            device="cpu", dtype=torch.float64
        )
        flexmoe_weight_bytes = spec.weight_bytes.detach().to(
            device="cpu", dtype=torch.float64
        )

        def flexmoe_scheduling():
            return flexmoe_schedule(
                flexmoe_omega,
                flexmoe_weight_bytes,
                ranks,
                args.n_slot,
                fm_cost,
            )

        schedule_ms = _time_cpu(
            flexmoe_scheduling,
            args.warmup,
            args.iterations,
        )
        result = run_flexmoe(
            loads, spec, num_ranks=ranks, n_slot=args.n_slot, cost=fm_cost
        )
        _attach_placement_details(result, args, loads, spec)
        rows.append(
            _row(
                "flexmoe",
                result,
                solve_ms=schedule_ms,
                placement_ms=None,
                routing_ms=None,
                total_ms=(
                    None
                    if getattr(args, "solver_only_timing", False)
                    else schedule_ms
                ),
            )
        )

    if "lplb" in strategies:
        try:
            stdout_context = (
                contextlib.redirect_stdout(sys.stderr)
                if getattr(args, "json", False)
                else contextlib.nullcontext()
            )
            with stdout_context:
                planner = LPLBBaseline(
                    num_experts=args.experts,
                    ep_size=ranks,
                    n_slot=args.n_slot,
                    lplb_root=args.lplb_root,
                    topology_name=getattr(args, "lplb_topology", "auto"),
                )
        except (LPLBUnavailableError, ValueError) as exc:
            if args.require_lplb:
                raise
            rows.append({"strategy": "lplb", "skipped": str(exc)})
        else:
            workload = loads.expert_load().to(device="cuda", dtype=torch.int32)
            history_workload = history_loads.expert_load().to(
                device="cuda", dtype=torch.int32
            )
            if getattr(args, "solver_only_timing", False):
                planner.update_mapping(history_workload)
                mapping_ms = None
            else:
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
            _attach_placement_details(result, args, loads, spec)
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
    if getattr(args, "details", False):
        raise ValueError(
            "--details currently describes one synthetic placement; "
            "trace replay contains one placement per sample"
        )
    strategies = {x.strip().lower() for x in args.strategies.split(",") if x.strip()}
    unknown = strategies - {"scale", "eplb", "fastermoe", "flexmoe", "lplb"}
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, samples, topo, spec = _load_trace(args.trace, device)
    ranks, experts = topo.num_ranks, spec.num_experts
    main_slots_per_rank, _ = _slot_counts(experts, ranks, 0)
    # Trace format v3 stored the solver's *total physical* slot count as
    # ``n_slot``. New traces may additionally spell out the replica budget.
    n_slot = int(
        meta.get(
            "replica_slots_per_rank",
            int(spec.n_slot) - main_slots_per_rank,
        )
    )
    _, expected_physical_slots = _slot_counts(experts, ranks, n_slot)
    if expected_physical_slots != int(spec.n_slot):
        raise ValueError(
            "trace slot metadata is inconsistent: physical slots must equal "
            "main slots plus replica slots"
        )
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
            stdout_context = (
                contextlib.redirect_stdout(sys.stderr)
                if getattr(args, "json", False)
                else contextlib.nullcontext()
            )
            with stdout_context:
                lplb_planner = LPLBBaseline(
                    num_experts=experts,
                    ep_size=ranks,
                    n_slot=n_slot,
                    lplb_root=args.lplb_root,
                    topology_name=getattr(args, "lplb_topology", "auto"),
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
                lambda ld=loads: run_fastermoe(
                    ld,
                    spec,
                    num_ranks=ranks,
                    n_slot=n_slot,
                    cost=fm_cost,
                ),
                device,
            )
            record("fastermoe", res, ms)
        if "flexmoe" in strategies:
            res, ms = _run_timed(
                lambda ld=loads: run_flexmoe(
                    ld,
                    spec,
                    num_ranks=ranks,
                    n_slot=n_slot,
                    cost=flex_cost,
                ),
                device,
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
        row = {
            "strategy": bucket["name"],
            "samples": len(bucket["theta"]),
            "solve_ms_mean": ms["mean"],
            "quality_theta_mean": theta["mean"],
            "quality_imbalance_mean": imb["mean"],
            "quality_imbalance_p90": imb["p90"],
            "load_kind": bucket["meta"].get("load_kind", ""),
        }
        for field in (
            "main_slots_per_rank",
            "replica_slots_per_rank",
            "physical_slots_per_rank",
            "replica_budget",
            "physical_instances",
            "replicas",
        ):
            if field in bucket["meta"]:
                row[field] = bucket["meta"][field]
        rows.append(row)
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
    parser.add_argument("--experts", type=int, default=640)
    parser.add_argument(
        "--n-slot",
        type=int,
        default=4,
        help="additional expert replica slots available per rank (excludes main experts)",
    )
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=8)
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
    parser.add_argument(
        "--include-no-balance",
        action="store_true",
        help="prepend a fixed-main-placement reference row with no solver",
    )
    parser.add_argument(
        "--solver-only-timing",
        action="store_true",
        help=(
            "populate only solve_ms; leave placement_ms, routing_ms, and "
            "total_ms null"
        ),
    )
    parser.add_argument("--lplb-root", default="/home/tiger/LPLB")
    parser.add_argument(
        "--lplb-topology",
        choices=("auto", "cube", "ring"),
        default="auto",
        help=(
            "LPLB redundancy topology: auto uses Ring for odd N_slot and Cube "
            "for even N_slot; use ring for a topology-controlled slot sweep"
        ),
    )
    parser.add_argument("--require-lplb", action="store_true")
    parser.add_argument("--fastermoe-bw-net", type=float, default=50e9 / 8)
    parser.add_argument("--fastermoe-bw-mm", type=float, default=11.5e12)
    parser.add_argument("--flexmoe-threshold", type=float, default=1.2)
    parser.add_argument(
        "--details",
        action="store_true",
        help=(
            "include per-rank placement, expert relocation, and replica transfer "
            "details for a synthetic benchmark"
        ),
    )
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


def _print_placement_details(row: dict[str, Any]) -> None:
    details = row.get("placement_details")
    if not details:
        return
    if not details.get("available", False):
        print(f"  placement details unavailable: {details.get('reason', 'unknown')}")
        return

    summary = details["summary"]
    print(
        "  placement: "
        f"{summary['logical_experts']} logical, "
        f"{summary['physical_instances']} physical, "
        f"{summary['replica_instances']} replicas, "
        f"{summary['relocated_logical_experts']} relocated bases"
    )
    for rank in details["rank_summary"]:
        print(
            f"  rank {rank['rank']:>2}: "
            f"physical={rank['physical_instances']}, "
            f"replicas={rank['replica_instances']}, "
            f"relocated_in={rank['relocated_bases_in']}, "
            f"load={rank['static_main_load_before']:.1f}"
            f"->{rank['planned_load_after']:.1f}"
        )
    for event in details["relocation_events"]:
        print(
            f"  relocate expert {event['expert']}: "
            f"rank {event['source_main_rank']} -> "
            f"rank {event['destination_rank']}"
        )
    for event in details["replica_transfer_events"]:
        print(
            f"  replicate expert {event['expert']}: "
            f"rank {event['source_main_rank']} -> "
            f"rank {event['destination_rank']} "
            f"(instances={event['instances']})"
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
            f"replicas={row.get('replicas', 'n/a')}/"
            f"{row.get('replica_budget', 'n/a')}, "
            f"N_slot={row.get('replica_slots_per_rank', 'n/a')}, "
            f"quality={row['load_kind']}"
        )
        _print_placement_details(row)


if __name__ == "__main__":
    main()

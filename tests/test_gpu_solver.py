"""GPU solver correctness tests and a parameterized performance driver.

Run the performance driver from the repository root, for example::

    python tests/test_gpu_solver.py
    python tests/test_gpu_solver.py --nodes 2 --gpus-per-node 8 --experts 64

The default benchmark models 32 nodes x 8 GPUs, 640 experts, 6000 tokens per
rank and top-k=8. All logical ranks are simulated by one physical CUDA device;
the benchmark measures solver kernels, not distributed communication.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from eplb import EPLBConfig, ProblemSpec, Topology, check_constraints, solve
from eplb.loads import Loads
from eplb import cuda_solve as cuda_solver
from sim.workload import make_loads

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("skew", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("nodes,gpus,experts,n_slot", [
    (1, 8, 64, 16),
    (4, 4, 32, 4),
    (2, 4, 16, 6),
    (4, 8, 64, 4),
])
def test_gpu_solver_matches_reference_quality(skew, nodes, gpus, experts, n_slot):
    R = nodes * gpus
    loads_cpu = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                           hotspot_ranks=0.25, seed=int(skew * 100) + nodes + gpus)
    topo_cpu = Topology.from_nvlink_rdma(nodes, gpus, 1, 8)
    spec_cpu = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot)
    cfg = EPLBConfig()
    plan_cpu = solve(loads_cpu, topo_cpu, spec_cpu, cfg)  # serial CPU reference

    dev = torch.device("cuda")
    loads_gpu = make_loads(R, experts, tokens_per_rank=2048, top_k=6, skew=skew,
                           hotspot_ranks=0.25, seed=int(skew * 100) + nodes + gpus, device=dev)
    topo_gpu = Topology.from_nvlink_rdma(nodes, gpus, 1, 8, device=dev)
    spec_gpu = ProblemSpec.uniform_main_placement(experts, R, 44_000_000, 7168 * 2, n_slot, device=dev)
    plan_gpu = solve(loads_gpu, topo_gpu, spec_gpu, cfg)  # default -> parallel CUDA kernel

    # The parallel CUDA kernel is deterministic and constraint-preserving, but
    # trades a little makespan for speed vs the serial reference: its bounded
    # repair leaves more residual imbalance on near-uniform loads.  Require a
    # valid, route-conserving plan within a generous makespan tolerance.
    report = check_constraints(plan_gpu, loads_gpu, topo_gpu, spec_gpu, cfg)
    assert report.ok, report.violations
    assert int(plan_gpu.q.sum()) == int(loads_gpu.omega.sum())
    theta_cpu, theta_gpu = int(plan_cpu.theta), int(plan_gpu.theta)
    assert theta_gpu <= theta_cpu * 1.25 + 1, (
        f"CUDA theta {theta_gpu} exceeds reference {theta_cpu} beyond tolerance"
    )


def test_stage1_batched_sort_preserves_domain_local_candidate_order():
    ranks, experts, domains = 8, 32, 2
    loads = make_loads(
        ranks,
        experts,
        tokens_per_rank=2048,
        top_k=6,
        skew=1.5,
        hotspot_ranks=0.25,
        seed=7,
        device="cuda",
    )
    topology = Topology.from_nvlink_rdma(domains, ranks // domains, 1, 8, "cuda")
    spec = ProblemSpec.uniform_main_placement(
        experts,
        ranks,
        1_000_000,
        4096,
        experts // ranks + 2,
        device="cuda",
    )
    actual = cuda_solver._stage1_candidates(loads, topology, spec)

    dom = topology.domain_of_rank
    demand = loads.domain_demand(dom, domains)
    expert = torch.arange(experts, device="cuda", dtype=torch.int64).repeat_interleave(
        domains
    )
    domain = torch.arange(domains, device="cuda", dtype=torch.int64).repeat(experts)
    tokens = demand[domain, expert]
    weight = spec.weight_bytes[expert]
    benefit = 2 * tokens * int(spec.s_tok) - weight
    main_domain = dom[spec.main_rank]
    valid = (
        (domain != main_domain[expert])
        & (tokens > 0)
        & (benefit > 0)
    )
    legacy_order = torch.arange(experts * domains, device="cuda")
    for key in (domain, expert, -benefit, (~valid).to(torch.int64)):
        legacy_order = legacy_order[
            torch.argsort(key[legacy_order], stable=True)
        ]
    for domain_id in range(domains):
        begin = domain_id * experts
        end = begin + experts
        expected = legacy_order[
            valid[legacy_order] & (domain[legacy_order] == domain_id)
        ]
        valid_count = expected.numel()

        assert torch.equal(
            actual[0][begin : begin + valid_count],
            expert[expected],
        )
        assert torch.all(actual[1][begin:end] == domain_id)
        assert torch.all(actual[2][begin : begin + valid_count] == 1)
        assert torch.all(actual[2][begin + valid_count : end] == 0)


@pytest.mark.parametrize("u_min", [1, 2, 4])
def test_gpu_incremental_transfer_respects_quota_floor(u_min):
    omega = torch.tensor([[16, 8], [8, 16], [12, 12]], dtype=torch.int64)
    cfg = EPLBConfig(u_min=u_min, max_stage2_iters=8)

    topo_gpu = Topology.from_nvlink_rdma(1, 3, device="cuda")
    spec_gpu = ProblemSpec(
        2,
        torch.tensor([0, 1], device="cuda"),
        torch.tensor([1, 1], device="cuda"),
        1,
        2,
    )
    loads_gpu = Loads(omega.cuda())
    plan_gpu = solve(loads_gpu, topo_gpu, spec_gpu, cfg)

    nonzero = plan_gpu.q[plan_gpu.q > 0]
    assert nonzero.numel() == 0 or int(nonzero.min()) >= u_min
    report = check_constraints(plan_gpu, loads_gpu, topo_gpu, spec_gpu, cfg)
    assert report.ok, report.violations


def test_default_gpu_solver_has_no_host_sync_after_warmup():
    dev = torch.device("cuda")
    omega = torch.tensor([[11], [0]], dtype=torch.int64, device=dev)
    topo = Topology.from_nvlink_rdma(1, 2, device=dev)
    spec = ProblemSpec(
        1,
        torch.tensor([0], device=dev),
        torch.tensor([1], device=dev),
        1,
        1,
    )
    cfg = EPLBConfig(max_stage2_iters=8)

    # Compile first; sync-debug then guards the steady-state solve path.
    solve(Loads(omega), topo, spec, cfg, validate=False)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan = solve(Loads(omega), topo, spec, cfg, validate=False)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert plan.theta.is_cuda


def test_gpu_floor_aware_cross_domain_fallback():
    omega = torch.tensor([[20], [10], [6]], dtype=torch.int64)
    dom = torch.tensor([0, 1, 2], dtype=torch.int64)
    cost = torch.ones((3, 3), dtype=torch.int64)
    cost.fill_diagonal_(0)
    cfg = EPLBConfig(u_min=3)

    topo_gpu = Topology(dom.cuda(), cost.cuda())
    spec_gpu = ProblemSpec(
        1,
        torch.tensor([0], device="cuda"),
        torch.tensor([15], device="cuda"),
        s_tok=1,
        n_slot=1,
    )
    loads_gpu = Loads(omega.cuda())
    plan_gpu = solve(loads_gpu, topo_gpu, spec_gpu, cfg)

    assert int(plan_gpu.q[plan_gpu.q > 0].min()) >= cfg.u_min
    report = check_constraints(plan_gpu, loads_gpu, topo_gpu, spec_gpu, cfg)
    assert report.ok, report.violations


def test_cuda_solver_is_deterministic_and_constraint_safe():
    ranks, experts = 8, 32
    loads = Loads(
        make_loads(
            ranks,
            experts,
            tokens_per_rank=1024,
            top_k=4,
            skew=1.5,
            seed=17,
            device="cuda",
        ).omega
        * 2
    )
    topo = Topology.from_nvlink_rdma(2, 4, device="cuda")
    spec = ProblemSpec.uniform_main_placement(
        experts, ranks, 1_000_000, 4096, 6, device="cuda"
    )
    cfg = EPLBConfig(
        u_min=2,
        max_stage2_iters=64,
        stage2_patience_all_scales=True,
    )
    first = solve(loads, topo, spec, cfg, validate=False)
    second = solve(loads, topo, spec, cfg, validate=False)
    torch.cuda.synchronize()

    assert torch.equal(first.x, second.x)
    assert torch.equal(first.q, second.q)
    assert torch.equal(first.theta, second.theta)
    report = check_constraints(first, loads, topo, spec, cfg)
    assert report.ok, report.violations


def test_stage2_stagnation_probe_is_bitwise_deterministic():
    ranks, experts = 32, 128
    loads = Loads(
        make_loads(
            ranks,
            experts,
            tokens_per_rank=4096,
            top_k=8,
            skew=1.5,
            hotspot_ranks=0.25,
            seed=0,
            device="cuda",
        ).omega
    )
    topo = Topology.from_nvlink_rdma(4, 8, device="cuda")
    spec = ProblemSpec.uniform_main_placement(
        experts, ranks, 44_000_000, 7168 * 2, 6, device="cuda"
    )
    early_cfg = EPLBConfig(
        stage2_stagnation_patience=8,
        stage2_patience_all_scales=True,
    )
    full_cfg = EPLBConfig(stage2_stagnation_patience=64)

    first = solve(loads, topo, spec, early_cfg, validate=False)
    second = solve(loads, topo, spec, early_cfg, validate=False)
    full = solve(loads, topo, spec, full_cfg, validate=False)
    torch.cuda.synchronize()

    assert torch.equal(first.x, second.x)
    assert torch.equal(first.q, second.q)
    assert torch.equal(first.theta, second.theta)
    assert torch.equal(first.theta, full.theta)
    report = check_constraints(first, loads, topo, spec, early_cfg)
    assert report.ok, report.violations


def test_cuda_solver_has_no_host_sync_after_warmup():
    loads = Loads(torch.tensor([[16, 8], [8, 16]], device="cuda"))
    topo = Topology.from_nvlink_rdma(1, 2, device="cuda")
    spec = ProblemSpec.uniform_main_placement(2, 2, 1, 1, 2, device="cuda")
    cfg = EPLBConfig(
        max_stage2_iters=16,
        stage2_patience_all_scales=True,
    )
    solve(loads, topo, spec, cfg, validate=False)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan = solve(loads, topo, spec, cfg, validate=False)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert plan.theta.is_cuda


# ---------------------------------------------------------------------------
# Standalone performance driver


@dataclass
class _PreparedKernel:
    omega: torch.Tensor
    cost: torch.Tensor
    dom: torch.Tensor
    cand_e: torch.Tensor
    cand_d: torch.Tensor
    cand_valid: torch.Tensor
    x_initial: torch.Tensor
    slot_initial: torch.Tensor
    x: torch.Tensor
    slot_used: torch.Tensor
    q: torch.Tensor
    u: torch.Tensor
    load_out: torch.Tensor
    stuck: torch.Tensor
    stage2_control: torch.Tensor
    num_ranks: int
    num_experts: int
    num_domains: int
    n_slot: int

    def reset_outputs(self) -> None:
        """Restore mutable buffers; callers place the timing event afterwards."""
        self.x.copy_(self.x_initial)
        self.slot_used.copy_(self.slot_initial)
        self.q.zero_()
        self.u.zero_()
        self.load_out.zero_()
        self.stuck.zero_()
        self.stage2_control.zero_()


def _prepare_kernel_inputs(
    loads: Loads,
    topology: Topology,
    spec: ProblemSpec,
) -> _PreparedKernel:
    """Build the device-side solver inputs once for kernel-only timing."""
    device = loads.device
    num_ranks = topology.num_ranks
    num_experts = spec.num_experts

    omega = loads.omega.to(torch.int64).contiguous()
    dom = topology.domain_of_rank.to(torch.int64).contiguous()
    cost = topology.cost.to(torch.int64).contiguous()
    main_rank = spec.main_rank.to(torch.int64)

    x_initial = torch.zeros(
        (num_experts, num_ranks), dtype=torch.int8, device=device
    )
    x_initial.scatter_(1, main_rank.view(num_experts, 1), 1)
    slot_initial = x_initial.sum(0).to(torch.int64)

    candidate_expert, candidate_domain, candidate_valid = (
        cuda_solver._stage1_candidates(loads, topology, spec)
    )

    return _PreparedKernel(
        omega=omega,
        cost=cost,
        dom=dom,
        cand_e=candidate_expert,
        cand_d=candidate_domain,
        cand_valid=candidate_valid,
        x_initial=x_initial,
        slot_initial=slot_initial,
        x=x_initial.clone(),
        slot_used=slot_initial.clone(),
        q=torch.zeros(
            (num_ranks, num_experts, num_ranks),
            dtype=torch.int64,
            device=device,
        ),
        u=torch.zeros(
            (num_experts, num_ranks), dtype=torch.int64, device=device
        ),
        load_out=torch.zeros(num_ranks, dtype=torch.int64, device=device),
        stuck=torch.zeros(num_ranks, dtype=torch.uint8, device=device),
        stage2_control=torch.zeros(3, dtype=torch.int64, device=device),
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_domains=topology.sync_free_num_domains,
        n_slot=int(spec.n_slot),
    )


def _launch_prepared_cuda(
    prepared: _PreparedKernel,
    cfg: EPLBConfig,
) -> None:
    """Launch the fine-grained CUDA admission, routing, and repair kernels."""
    cuda_solver._get_backend().fast_solve(
        prepared.omega,
        prepared.x,
        prepared.cost,
        prepared.dom,
        prepared.q,
        prepared.u,
        prepared.cand_e,
        prepared.cand_d,
        prepared.cand_valid,
        prepared.slot_used,
        prepared.load_out,
        prepared.stuck,
        prepared.stage2_control,
        prepared.num_domains,
        prepared.n_slot,
        min(cfg.max_stage2_iters, cfg.max_fast_stage2_iters),
        cfg.stage2_stagnation_patience,
        cfg.stage2_patience_all_scales,
        cfg.u_min,
        cfg.allow_cross_domain,
    )


def _measure_cuda(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    before_each: Callable[[], None] | None = None,
) -> tuple[list[float], object | None]:
    """Measure individual CUDA launches in milliseconds with CUDA events."""
    result = None
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        result = fn()
    torch.cuda.synchronize()

    samples: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iterations):
        if before_each is not None:
            before_each()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # When before_each resets kernel state, this event is enqueued after the
        # reset, so reset time is excluded from the measured interval.
        start.record()
        result = fn()
        end.record()
        samples.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in samples], result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(samples_ms) * 1_000,
        "p50_us": _percentile(samples_ms, 0.50) * 1_000,
        "p95_us": _percentile(samples_ms, 0.95) * 1_000,
        "min_us": min(samples_ms) * 1_000,
        "max_us": max(samples_ms) * 1_000,
    }


def _parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one Scale-EPLB GPU solver configuration."
    )
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--experts", type=int, default=640)
    parser.add_argument("--tokens-per-rank", type=int, default=6000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--extra-slots",
        type=int,
        default=2,
        help="Replica slots in addition to the main experts on each rank.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--stage2-patience-all-scales",
        action="store_true",
        help="Force the deterministic Stage 2 stagnation probe at every scale",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the benchmark report as machine-readable JSON",
    )
    return parser.parse_args()


def _validate_benchmark_args(args: argparse.Namespace) -> None:
    positive = (
        "nodes",
        "gpus_per_node",
        "experts",
        "tokens_per_rank",
        "top_k",
        "iterations",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.extra_slots < 0:
        raise ValueError("--extra-slots must be non-negative")


def run_gpu_solver_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Construct a realistic synthetic Ω and benchmark the CUDA solver."""
    _validate_benchmark_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(args.cuda_device)
    device = torch.device("cuda", args.cuda_device)
    num_ranks = args.nodes * args.gpus_per_node
    main_slots = (args.experts + num_ranks - 1) // num_ranks
    n_slot = main_slots + int(args.extra_slots)

    loads = make_loads(
        num_ranks,
        args.experts,
        tokens_per_rank=args.tokens_per_rank,
        top_k=args.top_k,
        skew=1.5,
        hotspot_ranks=0.25,
        seed=0,
        device=device,
    )
    topology = Topology.from_nvlink_rdma(
        args.nodes,
        args.gpus_per_node,
        intra_cost=1,
        inter_cost=8,
        device=device,
    )
    spec = ProblemSpec.uniform_main_placement(
        args.experts,
        num_ranks,
        44_000_000,
        7168 * 2,
        n_slot,
        device=device,
    )
    cfg = EPLBConfig(
        stage2_patience_all_scales=args.stage2_patience_all_scales
    )
    baseline_rank_load = torch.zeros(
        num_ranks, dtype=torch.int64, device=device
    )
    baseline_rank_load.index_add_(
        0, spec.main_rank, loads.expert_load()
    )
    baseline_theta = int(baseline_rank_load.max().item())
    mean_load = float(loads.omega.sum().item()) / num_ranks
    prepared = _prepare_kernel_inputs(loads, topology, spec)

    launch = _launch_prepared_cuda

    # Compile/load the selected specialization once and report it separately.
    prepared.reset_outputs()
    compile_start = time.perf_counter()
    launch(prepared, cfg)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - compile_start) * 1_000

    report: dict[str, object] = {
        "config": {
            "solver": "cuda",
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "logical_ranks": num_ranks,
            "experts": args.experts,
            "tokens_per_rank": args.tokens_per_rank,
            "top_k": args.top_k,
            "total_routes": num_ranks * args.tokens_per_rank * args.top_k,
            "main_slots_per_rank": main_slots,
            "extra_slots_per_rank": n_slot - main_slots,
            "total_slots_per_rank": n_slot,
            "cross_domain": True,
            "skew": 1.5,
            "hotspot_ranks": 0.25,
            "max_stage2_iters": cfg.max_stage2_iters,
            "max_fast_stage2_iters": cfg.max_fast_stage2_iters,
            "stage2_stagnation_patience": cfg.stage2_stagnation_patience,
            "stage2_patience_all_scales": cfg.stage2_patience_all_scales,
            "stage2_blocks": topology.sync_free_num_domains,
            "stage2_budget_scope": "per_domain",
        },
        "jit_compile_ms": compile_ms,
    }

    samples, _ = _measure_cuda(
        lambda: launch(prepared, cfg),
        warmup=args.warmup,
        iterations=args.iterations,
        before_each=prepared.reset_outputs,
    )
    report["kernel_only"] = _latency_summary(samples)

    expected_routes = num_ranks * args.tokens_per_rank * args.top_k
    actual_routes = int(prepared.q.sum().item())
    if actual_routes != expected_routes:
        raise AssertionError(
            f"kernel lost routes: expected {expected_routes}, got {actual_routes}"
        )
    solved_theta = int(prepared.load_out.max().item())
    report["result"] = {
        "theta": solved_theta,
        "replicas": int(prepared.x.sum().item()),
        "routes": actual_routes,
    }

    baseline_imbalance = baseline_theta / mean_load if mean_load else 1.0
    solved_imbalance = solved_theta / mean_load if mean_load else 1.0
    report["quality"] = {
        "mean_load": mean_load,
        "baseline_theta": baseline_theta,
        "baseline_imbalance": baseline_imbalance,
        "solved_theta": solved_theta,
        "solved_imbalance": solved_imbalance,
        "theta_reduction_percent": (
            100.0 * (baseline_theta - solved_theta) / baseline_theta
            if baseline_theta
            else 0.0
        ),
        "balance_speedup": (
            baseline_theta / solved_theta if solved_theta else float("inf")
        ),
    }
    return report


def _print_benchmark_report(report: dict[str, object]) -> None:
    config = report["config"]
    assert isinstance(config, dict)
    print(
        f"{config['solver'].upper()} solver benchmark: "
        f"R={config['logical_ranks']}, E={config['experts']}, "
        f"tokens/rank={config['tokens_per_rank']}, top-k={config['top_k']}, "
        f"domains={config['nodes']}, slots={config['total_slots_per_rank']} "
        f"({config['main_slots_per_rank']} main + "
        f"{config['extra_slots_per_rank']} extra)"
    )
    print(f"JIT compile: {report['jit_compile_ms']:.3f} ms")
    stats = report["kernel_only"]
    assert isinstance(stats, dict)
    print(
        f"kernel_only  mean={stats['mean_us']:.2f} us  "
        f"p50={stats['p50_us']:.2f} us  p95={stats['p95_us']:.2f} us  "
        f"min={stats['min_us']:.2f} us  max={stats['max_us']:.2f} us"
    )
    result = report["result"]
    assert isinstance(result, dict)
    print(
        f"result       theta={result['theta']}  replicas={result['replicas']}  "
        f"routes={result['routes']}"
    )
    quality = report["quality"]
    assert isinstance(quality, dict)
    print(
        "balance      "
        f"theta {quality['baseline_theta']} -> {quality['solved_theta']}  "
        f"imbalance {quality['baseline_imbalance']:.4f} -> "
        f"{quality['solved_imbalance']:.4f}  "
        f"reduction={quality['theta_reduction_percent']:.2f}%  "
        f"speedup={quality['balance_speedup']:.3f}x"
    )


def main() -> None:
    args = _parse_benchmark_args()
    report = run_gpu_solver_benchmark(args)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        _print_benchmark_report(report)


if __name__ == "__main__":
    main()

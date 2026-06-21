"""End-to-end CPU simulation of one Scale-EPLB rebalance (run: python -m sim.run_sim)."""

from __future__ import annotations

import argparse

import torch

from eplb import (
    EPLBConfig,
    ProblemSpec,
    Topology,
    check_constraints,
    compute_metrics,
    solve,
)
from sim.workload import make_loads


def _baseline_load(loads, spec, num_ranks):
    """Per-rank load with no replication: every token goes to ``main(e)``."""
    load = torch.zeros(num_ranks, dtype=torch.int64)
    lam_e = loads.expert_load()
    load.index_add_(0, spec.main_rank, lam_e)
    return load


def main() -> None:
    ap = argparse.ArgumentParser(description="Scale-EPLB CPU simulation")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--tokens-per-rank", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--skew", type=float, default=1.5)
    ap.add_argument("--hotspot-ranks", type=float, default=0.25)
    ap.add_argument("--n-slot", type=int, default=4)
    ap.add_argument("--intra-cost", type=int, default=1)
    ap.add_argument("--inter-cost", type=int, default=8)
    ap.add_argument("--s-tok", type=int, default=7168 * 2)  # hidden*bf16 bytes
    ap.add_argument("--weight-bytes", type=int, default=44_000_000)
    ap.add_argument("--no-cross-domain", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    R = args.nodes * args.gpus
    topo = Topology.from_nvlink_rdma(
        args.nodes, args.gpus, intra_cost=args.intra_cost, inter_cost=args.inter_cost
    )
    spec = ProblemSpec.uniform_main_placement(
        num_experts=args.experts,
        num_ranks=R,
        weight_bytes_each=args.weight_bytes,
        s_tok=args.s_tok,
        n_slot=args.n_slot,
    )
    cfg = EPLBConfig(allow_cross_domain=not args.no_cross_domain)
    loads = make_loads(
        num_ranks=R,
        num_experts=args.experts,
        tokens_per_rank=args.tokens_per_rank,
        top_k=args.top_k,
        skew=args.skew,
        hotspot_ranks=args.hotspot_ranks,
        seed=args.seed,
    )

    plan = solve(loads, topo, spec, cfg)
    report = check_constraints(plan, loads, topo, spec, cfg)
    metrics = compute_metrics(plan, loads, topo, spec, cfg)

    total = int(loads.lam.sum().item())
    mean = total / R
    base_load = _baseline_load(loads, spec, R)
    base_tau = int(base_load.max().item())

    print("=" * 64)
    print(f"Topology : {args.nodes} nodes x {args.gpus} gpus = {R} ranks, "
          f"{topo.num_domains} NVLink domains")
    print(f"Experts  : {args.experts}, N_slot={args.n_slot}, "
          f"cross-domain={'off' if args.no_cross_domain else 'on'}")
    print(f"Tokens   : {total} total, mean/rank={mean:.1f}")
    print("-" * 64)
    print(f"Baseline (no replication):  tau={base_tau:>8}  "
          f"imbalance={base_tau / mean:6.3f}")
    print(f"Scale-EPLB plan          :  tau={metrics.tau:>8}  "
          f"imbalance={metrics.imbalance:6.3f}")
    speedup = base_tau / metrics.tau if metrics.tau else float("inf")
    print(f"Makespan reduction       :  {speedup:6.3f}x")
    print("-" * 64)
    print(f"Total replicas : {metrics.total_replicas} "
          f"(experts={args.experts}, extra={metrics.total_replicas - args.experts})")
    print(f"Max slots/rank : {metrics.max_slots_used} / {args.n_slot}")
    print(f"Phi_token      : {metrics.phi_token}")
    print(f"Phi_weight     : {metrics.phi_weight}")
    print(f"Objective      : {metrics.objective}")
    print("-" * 64)
    print(f"Constraints C1-C7: {'OK' if report.ok else 'VIOLATED'}")
    for msg in report.violations:
        print(f"  - {msg}")
    print("=" * 64)


if __name__ == "__main__":
    main()

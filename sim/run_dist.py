"""Multi-process verification of the no-CPU-sync, bit-identical-plan property.

Each process is one EP rank. Every rank builds *only its own* row of ``Lambda``,
all-gathers, solves locally, and we verify all ranks produced a bit-identical
plan (experiment E3) -- with no broadcast of the plan itself.

Run on CPU (gloo) locally::

    python -m sim.run_dist --world-size 8 --experts 64 --skew 1.5

Run on GPUs (nccl) via torchrun::

    torchrun --nproc_per_node=8 -m sim.run_dist --backend nccl --experts 64
"""

from __future__ import annotations

import argparse
import hashlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from eplb import EPLBConfig, ProblemSpec, Topology, check_constraints, solve
from eplb.distributed import all_gather_lambda
from sim.workload import make_loads


def _plan_hash(plan) -> str:
    h = hashlib.sha256()
    h.update(plan.x.cpu().numpy().tobytes())
    h.update(plan.q.cpu().numpy().tobytes())
    h.update(str(plan.tau).encode())
    return h.hexdigest()[:16]


def _worker(rank: int, args, full_lam_bytes: bytes, lam_shape):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    dist.init_process_group(
        backend=args.backend, rank=rank, world_size=args.world_size
    )
    device = "cuda" if args.backend == "nccl" else "cpu"
    if args.backend == "nccl":
        torch.cuda.set_device(rank)

    # reconstruct the full reference Lambda (same on every rank) and take our row
    full_lam = torch.frombuffer(bytearray(full_lam_bytes), dtype=torch.int64).reshape(lam_shape)
    local_row = full_lam[rank].to(device)

    topo = Topology.from_nvlink_rdma(
        args.nodes, args.gpus, intra_cost=1, inter_cost=8, device=device
    )
    spec = ProblemSpec.uniform_main_placement(
        num_experts=args.experts, num_ranks=args.world_size,
        weight_bytes_each=44_000_000, s_tok=7168 * 2, n_slot=args.n_slot,
        device=device,
    )
    cfg = EPLBConfig()

    # the ONLY communication: all-gather the integer load rows
    loads = all_gather_lambda(local_row)
    plan = solve(loads, topo, spec, cfg)

    report = check_constraints(plan, loads, topo, spec, cfg)
    digest = _plan_hash(plan)

    # gather digests to rank 0 to confirm bit-identical plans
    obj = [None] * args.world_size
    dist.all_gather_object(obj, (rank, digest, plan.tau, report.ok))
    if rank == 0:
        digests = {d for (_, d, _, _) in obj}
        all_ok = all(ok for (_, _, _, ok) in obj)
        print(f"world_size={args.world_size} backend={args.backend} "
              f"experts={args.experts}")
        for (rk, d, tau, ok) in sorted(obj):
            print(f"  rank {rk:>2}: plan_hash={d} tau={tau} constraints={'OK' if ok else 'BAD'}")
        print("-" * 50)
        print(f"  bit-identical across ranks: {len(digests) == 1}")
        print(f"  all constraints satisfied : {all_ok}")
        assert len(digests) == 1, "plans diverged across ranks!"
        assert all_ok, "constraints violated on some rank!"
        print("  E3 determinism check: PASS")

    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description="Distributed Scale-EPLB determinism check")
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--n-slot", type=int, default=16)
    ap.add_argument("--skew", type=float, default=1.5)
    ap.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    ap.add_argument("--port", type=int, default=29501)
    args = ap.parse_args()

    if args.nodes * args.gpus != args.world_size:
        # default to single domain spanning all ranks if mismatch
        args.nodes, args.gpus = 1, args.world_size

    ref = make_loads(args.world_size, args.experts, tokens_per_rank=4096,
                     top_k=6, skew=args.skew, hotspot_ranks=0.25, seed=0)
    full_lam = ref.lam.to(torch.int64).contiguous()

    if "RANK" in os.environ:  # launched by torchrun
        rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        _worker(rank, args, full_lam.numpy().tobytes(), tuple(full_lam.shape))
    else:  # spawn locally (gloo)
        mp.spawn(
            _worker,
            args=(args, full_lam.numpy().tobytes(), tuple(full_lam.shape)),
            nprocs=args.world_size,
            join=True,
        )


if __name__ == "__main__":
    main()

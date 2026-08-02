"""Parallel CUDA backend for the latency-sensitive Scale-EPLB solve path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from .plan import Plan

_BACKEND: Any | None = None


def _get_backend():
    """Build the CUDA extension lazily and cache the loaded module."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if not torch.cuda.is_available():
        raise RuntimeError("the fast Scale-EPLB backend requires CUDA")

    source = Path(__file__).with_name("csrc") / "fast_solver.cu"
    _BACKEND = load(
        name="_scale_eplb_fast_cuda",
        sources=[os.fspath(source)],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("EPLB_CUDA_BUILD_VERBOSE", "0") == "1",
    )
    return _BACKEND


def _stage1_candidates(loads, topo, spec):
    """Create the benefit-ordered cross-domain candidate arrays on device."""
    device = loads.device
    experts = spec.num_experts
    domains = topo.sync_free_num_domains
    if not domains:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty

    dom = topo.domain_of_rank.to(torch.int64).contiguous()
    main_rank = spec.main_rank.to(torch.int64)
    demand = loads.domain_demand(dom, domains)
    expert = torch.arange(experts, device=device, dtype=torch.int64).repeat_interleave(
        domains
    )
    domain = torch.arange(domains, device=device, dtype=torch.int64).repeat(experts)
    tokens = demand[domain, expert]
    weight = spec.weight_bytes.to(torch.int64)[expert]
    benefit = 2 * tokens * int(spec.s_tok) - weight
    main_domain = dom[main_rank]
    valid = (
        (domain != main_domain[expert])
        & (tokens > 0)
        & (weight < 2 * tokens * int(spec.s_tok))
    )

    order = torch.arange(experts * domains, device=device, dtype=torch.int64)
    for key in (domain, expert, -benefit, (~valid).to(torch.int64)):
        order = order[torch.argsort(key[order], stable=True)]
    return (
        expert[order].contiguous(),
        domain[order].contiguous(),
        valid[order].to(torch.int64).contiguous(),
    )


def solve_cuda(loads, topo, spec, cfg) -> Plan:
    """Solve with parallel routing plus a deterministic block-parallel repair."""
    backend = _get_backend()
    device = loads.device
    ranks = topo.num_ranks
    experts = spec.num_experts
    omega = loads.omega.to(torch.int64).contiguous()
    dom = topo.domain_of_rank.to(torch.int64).contiguous()
    cost = topo.cost.to(torch.int64).contiguous()
    main_rank = spec.main_rank.to(torch.int64)

    x = torch.zeros((experts, ranks), dtype=torch.int8, device=device)
    x.scatter_(1, main_rank.view(experts, 1), 1)
    slot_used = x.sum(0).to(torch.int64)
    candidate_expert, candidate_domain, candidate_valid = _stage1_candidates(
        loads, topo, spec
    )

    q = torch.zeros((ranks, experts, ranks), dtype=torch.int64, device=device)
    expert_rank_load = torch.zeros(
        (experts, ranks), dtype=torch.int64, device=device
    )
    rank_load = torch.zeros(ranks, dtype=torch.int64, device=device)
    stuck = torch.zeros(ranks, dtype=torch.uint8, device=device)

    backend.fast_solve(
        omega,
        x,
        cost,
        dom,
        q,
        expert_rank_load,
        candidate_expert,
        candidate_domain,
        candidate_valid,
        slot_used,
        rank_load,
        stuck,
        int(spec.n_slot),
        min(int(cfg.max_stage2_iters), int(cfg.max_fast_stage2_iters)),
        int(cfg.u_min),
        bool(cfg.allow_cross_domain),
    )
    return Plan(x=x, q=q, theta=rank_load.max())

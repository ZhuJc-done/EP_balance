"""Evaluation metrics and C1-C7 constraint verification for a :class:`Plan`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from .config import EPLBConfig
from .loads import Loads
from .plan import Plan
from .problem import ProblemSpec
from .topology import Topology


@dataclass
class Metrics:
    """Quantitative quality of a plan."""

    tau: int  # makespan (max per-rank token load)
    mean_load: float  # mean per-rank load
    imbalance: float  # tau / mean_load (1.0 is perfect)
    phi_token: int  # token comm cost sum c[r,r'] q[r,e,r']
    phi_weight: int  # weight movement cost sum_{e,r!=main} 2 c[main,r] |W_e| x
    objective: int  # alpha tau + beta phi_token + gamma phi_weight
    total_replicas: int  # total physical instances across all experts
    max_slots_used: int  # max instances hosted on any single rank


def compute_metrics(
    plan: Plan,
    loads: Loads,
    topo: Topology,
    spec: ProblemSpec,
    cfg: EPLBConfig | None = None,
) -> Metrics:
    cfg = cfg or EPLBConfig()
    cost = topo.cost
    R = topo.num_ranks

    load = plan.rank_load()  # [R]
    tau = int(load.max().item()) if R > 0 else 0
    total_tokens = int(loads.lam.sum().item())
    mean_load = total_tokens / R if R > 0 else 0.0
    imbalance = (tau / mean_load) if mean_load > 0 else 1.0

    # Phi_token = sum_{r,e,r'} c[r,r'] q[r,e,r']; elementwise reduce (not einsum) for int64-on-CUDA support
    cost_i64 = cost.to(torch.int64)  # [R, R] over (src, dst)
    phi_token = int((plan.q * cost_i64.unsqueeze(1)).sum().item())

    # Phi_weight = sum_{e, r!=main(e)} 2 c[main(e), r] |W_e| x[e, r]
    main_rank = spec.main_rank
    c_main = cost[main_rank]
    x = plan.x.to(torch.int64)
    # zero out the main column so r==main contributes nothing
    not_main = torch.ones_like(x)
    not_main[torch.arange(spec.num_experts, device=x.device), main_rank] = 0
    phi_weight = int(
        (2 * c_main * spec.weight_bytes.unsqueeze(1) * x * not_main).sum().item()
    )

    objective = cfg.alpha * tau + cfg.beta * phi_token + cfg.gamma * phi_weight

    return Metrics(
        tau=tau,
        mean_load=mean_load,
        imbalance=imbalance,
        phi_token=phi_token,
        phi_weight=phi_weight,
        objective=objective,
        total_replicas=int(plan.num_replicas().sum().item()),
        max_slots_used=int(plan.slots_used().max().item()) if R > 0 else 0,
    )


@dataclass
class ConstraintReport:
    """Result of checking C1-C7 against a plan."""

    ok: bool
    violations: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def check_constraints(
    plan: Plan,
    loads: Loads,
    topo: Topology,
    spec: ProblemSpec,
    cfg: EPLBConfig | None = None,
) -> ConstraintReport:
    """Verify the plan satisfies constraints C1-C7 from the problem definition."""
    cfg = cfg or EPLBConfig()
    v: List[str] = []
    lam = loads.lam
    x = plan.x
    q = plan.q
    R, E = topo.num_ranks, spec.num_experts
    dom = topo.domain_of_rank
    main_rank = spec.main_rank
    main_dom = dom[main_rank]
    n_slot = int(spec.n_slot)
    s_tok = int(spec.s_tok)

    # C1 conservation: sum_{r'} q[r,e,r'] == lam[r,e]
    served = q.sum(dim=2)  # [R, E]
    if not torch.equal(served, lam):
        bad = int((served != lam).sum().item())
        v.append(f"C1 conservation violated for {bad} (r,e) pairs")

    # C2 reachability: q[r,e,r'] > 0 only where x[e,r']==1 (forbidden marks ranks without an instance)
    forbidden = (x == 0)  # [E, R] over (e, dst)
    if torch.any((q * forbidden.unsqueeze(0).to(q.dtype)) != 0):
        v.append("C2 reachability violated: quota routed to a rank without an instance")

    # C3 makespan: check plan.tau matches the realised max rank load
    load = q.sum(dim=(0, 1))
    if int(load.max().item()) != int(plan.tau):
        v.append(
            f"C3/tau mismatch: plan.tau={plan.tau} but max rank load={int(load.max().item())}"
        )

    # C4 slot budget: sum_e x[e,r] <= N_slot
    slots = x.sum(dim=0)
    if torch.any(slots > n_slot):
        worst = int(slots.max().item())
        v.append(f"C4 slot budget violated: a rank uses {worst} > N_slot={n_slot}")

    # C5 quota granularity: q == 0 or q >= u_min
    if cfg.u_min > 1:
        nz = q[q > 0]
        if nz.numel() and int(nz.min().item()) < cfg.u_min:
            v.append(f"C5 granularity violated: a nonzero quota < u_min={cfg.u_min}")

    # C6 cross-domain gate: any cross-domain replica must satisfy |W_e| < 2 T[dom(r),e] s_tok
    Tde = loads.domain_demand(dom, topo.num_domains)
    for e in range(E):
        hosts = torch.nonzero(x[e] == 1, as_tuple=False).flatten()
        for r in hosts.tolist():
            d = int(dom[r].item())
            if d != int(main_dom[e].item()):
                lhs = int(spec.weight_bytes[e].item())
                rhs = 2 * int(Tde[d, e].item()) * s_tok
                if not (lhs < rhs):
                    v.append(
                        f"C6 violated: cross-domain replica e={e} on rank {r} "
                        f"(|W_e|={lhs} !< 2*T[d,e]*s_tok={rhs})"
                    )

    # C7 main placement fixed: x[e, main(e)] == 1
    if not torch.all(x[torch.arange(E, device=x.device), main_rank] == 1):
        v.append("C7 violated: some main instance is not present")

    return ConstraintReport(ok=(len(v) == 0), violations=v)

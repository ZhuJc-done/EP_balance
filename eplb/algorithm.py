"""The Scale-EPLB deterministic solver: Stage 0 precompute, Stage 1 cross-domain gate, Stage 2 intra-domain balancing."""

from __future__ import annotations

import os

import torch

from .config import EPLBConfig
from .loads import Loads
from .plan import Plan
from .problem import ProblemSpec
from .topology import Topology
from .triton_solve import HAS_TRITON


def _lexsort_keys(*keys: torch.Tensor) -> torch.Tensor:
    """Return indices sorting by ``keys[0]`` then ``keys[1]`` ... ascending (all 1-D int64)."""
    n = keys[0].numel()
    order = torch.arange(n, device=keys[0].device, dtype=torch.int64)
    # apply stable sorts least- to most-significant, so keys[0] is applied last
    for key in reversed(keys):
        k = key[order]
        perm = torch.argsort(k, stable=True)
        order = order[perm]
    return order


def _waterfill(need: int, base: torch.Tensor, tie: torch.Tensor) -> torch.Tensor:
    """Distribute ``need`` units to minimise max(base+add), tie-broken by ``tie`` then index.

    Args:
        need: Total units to distribute (>= 0).
        base: int64 ``[D]`` current load of each destination.
        tie: int64 ``[D]`` secondary key (e.g. communication cost) for tie-break.

    Returns:
        int64 ``[D]`` non-negative additions summing exactly to ``need``.
    """
    D = base.numel()
    add = torch.zeros(D, dtype=torch.int64, device=base.device)
    if need <= 0 or D == 0:
        return add
    if D == 1:
        add[0] = need
        return add

    # order destinations by (current load, tie key, original index)
    idx = torch.arange(D, dtype=torch.int64, device=base.device)
    order = _lexsort_keys(base, tie, idx)
    b = base[order]

    rem = int(need)
    # find the largest prefix of sorted dests we can fully level within rem
    k = 1
    while k < D:
        cost_to_next = int((b[k] * k - torch.sum(b[:k])).item())
        if cost_to_next > rem:
            break
        k += 1
    # level the first k destinations evenly with whatever remains
    level_floor = int(b[k - 1].item())
    base_cost = int((level_floor * k - torch.sum(b[:k])).item())
    rem_after = rem - base_cost
    add_sorted = torch.zeros(D, dtype=torch.int64, device=base.device)
    add_sorted[:k] = level_floor - b[:k]
    share = rem_after // k
    extra = rem_after - share * k
    add_sorted[:k] += share
    add_sorted[:extra] += 1  # first `extra` in sorted order get one more
    add[order] = add_sorted  # scatter back to original indices
    return add


def _assign_quota(
    lam: torch.Tensor,
    x: torch.Tensor,
    cost: torch.Tensor,
    dom: torch.Tensor,
):
    """Route tokens under strict domain-local serving (cross-domain only when no in-domain instance).

    Args:
        lam: int64 ``[R, E]`` load matrix.
        x: int8 ``[E, R]`` placement.
        cost: int64 ``[R, R]`` per-token comm cost.
        dom: int64 ``[R]`` domain id per rank.

    Returns:
        ``(q, load)`` where ``q`` is int64 ``[R, E, R]`` and ``load`` is
        int64 ``[R]`` per-destination token counts.
    """
    R = lam.shape[0]
    E = lam.shape[1]
    device = lam.device
    q = torch.zeros((R, E, R), dtype=torch.int64, device=device)
    load = torch.zeros(R, dtype=torch.int64, device=device)

    # process (r, e) pairs in descending token count (LPT), tie by (e, r)
    rr, ee = torch.meshgrid(
        torch.arange(R, device=device, dtype=torch.int64),
        torch.arange(E, device=device, dtype=torch.int64),
        indexing="ij",
    )
    flat_r = rr.reshape(-1)
    flat_e = ee.reshape(-1)
    flat_lam = lam.reshape(-1)
    neg_lam = -flat_lam
    order = _lexsort_keys(neg_lam, flat_e, flat_r)

    for idx in order.tolist():
        need = int(flat_lam[idx].item())
        if need == 0:
            continue
        r = int(flat_r[idx].item())
        e = int(flat_e[idx].item())
        d = int(dom[r].item())
        inst = torch.nonzero(x[e] == 1, as_tuple=False).flatten()
        # prefer in-domain instances, fall back cross-domain if none
        in_domain = inst[dom[inst] == d]
        dests = in_domain if in_domain.numel() > 0 else inst
        add = _waterfill(need, load[dests], cost[r, dests])
        q[r, e, dests] += add
        load[dests] += add

    return q, load


def solve(
    loads: Loads,
    topo: Topology,
    spec: ProblemSpec,
    cfg: EPLBConfig | None = None,
    *,
    validate: bool = True,
) -> Plan:
    """Compute a deterministic Scale-EPLB plan for one (layer, micro-batch).

    Args:
        loads: Dynamic load matrix ``Lambda`` (already all-gathered).
        topo: Cluster topology.
        spec: Static problem spec (main placement, weights, slot budget).
        cfg: Solver configuration (defaults to :class:`EPLBConfig`).
        validate: Run input validation first (disable in hot loops once trusted).

    Returns:
        A :class:`~eplb.plan.Plan` with placement ``x``, routing quota ``q`` and
        makespan ``tau``.
    """
    cfg = cfg or EPLBConfig()
    R = topo.num_ranks
    E = spec.num_experts
    device = loads.device

    if validate:
        topo.validate()
        spec.validate(R)
        loads.validate(R, E)

    backend = os.environ.get("EPLB_SOLVER_BACKEND", "auto")

    # bisect backend: tau-bisection heuristic (separate plan, not bit-identical to the greedy)
    if backend == "bisect":
        return solve_bisect(loads, topo, spec, cfg)

    # GPU fast path: run the whole solver in one Triton launch (zero host sync), bit-identical to the reference
    if (
        R > 0
        and E > 0
        and loads.lam.is_cuda
        and HAS_TRITON
        and backend in ("auto", "fused")
    ):
        from .triton_solve import solve_fused

        return solve_fused(loads, topo, spec, cfg)

    lam = loads.lam
    dom = topo.domain_of_rank
    cost = topo.cost
    main_rank = spec.main_rank
    main_dom = dom[main_rank]
    W = spec.weight_bytes
    s_tok = int(spec.s_tok)
    n_slot = int(spec.n_slot)
    M = topo.num_domains

    # Stage 0: per-domain demand T[d,e]
    Tde = loads.domain_demand(dom, M)

    # placement init: main fixed (C7)
    x = torch.zeros((E, R), dtype=torch.int8, device=device)
    x[torch.arange(E, device=device, dtype=torch.int64), main_rank] = 1
    slot_used = x.sum(dim=0).to(torch.int64)

    # Stage 1: admit a cross-domain replica iff C6 holds, greedily by benefit, within slots
    if cfg.allow_cross_domain and M > 1:
        cand_e, cand_d, cand_benefit = [], [], []
        for e in range(E):
            mde = int(main_dom[e].item())
            we = int(W[e].item())
            for d in range(M):
                if d == mde:
                    continue
                t = int(Tde[d, e].item())
                if t == 0:
                    continue
                if we < 2 * t * s_tok:  # C6 gate
                    cand_e.append(e)
                    cand_d.append(d)
                    cand_benefit.append(2 * t * s_tok - we)
        if cand_e:
            ce = torch.tensor(cand_e, dtype=torch.int64, device=device)
            cd = torch.tensor(cand_d, dtype=torch.int64, device=device)
            cb = torch.tensor(cand_benefit, dtype=torch.int64, device=device)
            order = _lexsort_keys(-cb, ce, cd)  # benefit desc, then e, then d
            for idx in order.tolist():
                e = int(ce[idx].item())
                d = int(cd[idx].item())
                ranks_d = topo.ranks_in_domain(d)
                if int(x[e, ranks_d].sum().item()) > 0:
                    continue  # already has an instance in this domain
                free = ranks_d[slot_used[ranks_d] < n_slot]
                if free.numel() == 0:
                    continue
                # least-loaded rank in the domain (argmax slack), tie by id
                chosen_order = _lexsort_keys(slot_used[free], free)
                chosen = int(free[chosen_order[0]].item())
                x[e, chosen] = 1
                slot_used[chosen] += 1

    # Stage 2: relieve the busiest rank by replicating its top expert inside its own domain
    q, load = _assign_quota(lam, x, cost, dom)
    tau = int(load.max().item()) if R > 0 else 0
    stuck = torch.zeros(R, dtype=torch.bool, device=device)

    iters = 0
    while iters < cfg.max_stage2_iters:
        iters += 1
        if int((slot_used < n_slot).sum().item()) == 0:
            break  # no free slots anywhere (C4 saturated)

        # busiest rank we have not already failed to relieve
        masked = load.clone()
        masked[stuck] = -1
        max_load = int(masked.max().item())
        if max_load <= 0:
            break
        bottleneck_ranks = torch.nonzero(masked == max_load, as_tuple=False).flatten()
        r_star = int(bottleneck_ranks.min().item())
        d_star = int(dom[r_star].item())

        # expert contributing the most tokens to r_star, tie by expert id
        contrib = q[:, :, r_star].sum(dim=0)
        max_contrib = int(contrib.max().item())
        if max_contrib == 0:
            stuck[r_star] = True
            continue
        cand_experts = torch.nonzero(contrib == max_contrib, as_tuple=False).flatten()
        e_star = int(cand_experts.min().item())

        # target: same domain, free slot, not already hosting e_star; pick max slack
        hosts = x[e_star] == 1
        eligible_mask = (slot_used < n_slot) & (~hosts) & (dom == d_star)
        eligible = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        if eligible.numel() == 0:
            stuck[r_star] = True
            continue

        place_order = _lexsort_keys(load[eligible], cost[r_star, eligible], eligible)
        target = int(eligible[place_order[0]].item())

        # tentatively add the intra-domain replica and re-route domain-locally
        x[e_star, target] = 1
        new_q, new_load = _assign_quota(lam, x, cost, dom)
        new_tau = int(new_load.max().item())
        if int(new_load[r_star].item()) < int(load[r_star].item()) and new_tau <= tau:
            slot_used[target] += 1
            q, load, tau = new_q, new_load, new_tau
            stuck[:] = False  # loads changed; re-evaluate every rank
        else:
            x[e_star, target] = 0  # revert; did not relieve r_star
            stuck[r_star] = True

    return Plan(x=x, q=q, tau=tau)


def _stage01_placement(loads: Loads, topo: Topology, spec: ProblemSpec, cfg: EPLBConfig):
    """Stage 0 (per-domain demand) + Stage 1 (C6-gated cross-domain admission), shared placement init.

    Args:
        loads: Dynamic load matrix ``Lambda``.
        topo: Cluster topology.
        spec: Static problem spec.
        cfg: Solver configuration.

    Returns:
        ``(x, slot_used)`` where ``x`` is int8 ``[E, R]`` placement (mains + Stage 1
        cross-domain replicas) and ``slot_used`` is int64 ``[R]`` instances per rank.
    """
    R = topo.num_ranks
    E = spec.num_experts
    device = loads.device
    dom = topo.domain_of_rank
    main_rank = spec.main_rank
    main_dom = dom[main_rank]
    W = spec.weight_bytes
    s_tok = int(spec.s_tok)
    n_slot = int(spec.n_slot)
    M = topo.num_domains

    Tde = loads.domain_demand(dom, M)
    x = torch.zeros((E, R), dtype=torch.int8, device=device)
    x[torch.arange(E, device=device, dtype=torch.int64), main_rank] = 1
    slot_used = x.sum(dim=0).to(torch.int64)

    if cfg.allow_cross_domain and M > 1:
        cand_e, cand_d, cand_benefit = [], [], []
        for e in range(E):
            mde = int(main_dom[e].item())
            we = int(W[e].item())
            for d in range(M):
                if d == mde:
                    continue
                t = int(Tde[d, e].item())
                if t == 0:
                    continue
                if we < 2 * t * s_tok:  # C6 gate
                    cand_e.append(e)
                    cand_d.append(d)
                    cand_benefit.append(2 * t * s_tok - we)
        if cand_e:
            ce = torch.tensor(cand_e, dtype=torch.int64, device=device)
            cd = torch.tensor(cand_d, dtype=torch.int64, device=device)
            cb = torch.tensor(cand_benefit, dtype=torch.int64, device=device)
            order = _lexsort_keys(-cb, ce, cd)
            for idx in order.tolist():
                e = int(ce[idx].item())
                d = int(cd[idx].item())
                ranks_d = topo.ranks_in_domain(d)
                if int(x[e, ranks_d].sum().item()) > 0:
                    continue
                free = ranks_d[slot_used[ranks_d] < n_slot]
                if free.numel() == 0:
                    continue
                chosen_order = _lexsort_keys(slot_used[free], free)
                chosen = int(free[chosen_order[0]].item())
                x[e, chosen] = 1
                slot_used[chosen] += 1

    return x, slot_used


def solve_bisect(
    loads: Loads,
    topo: Topology,
    spec: ProblemSpec,
    cfg: EPLBConfig | None = None,
) -> Plan:
    """tau-bisection solver (CPU reference): Stage 0/1 then a makespan-bisection descent (a different heuristic, not bit-identical to :func:`solve`).

    Args:
        loads: Dynamic load matrix ``Lambda`` (already all-gathered).
        topo: Cluster topology.
        spec: Static problem spec.
        cfg: Solver configuration (uses ``cfg.tau_bisect_iters`` probe budget).

    Returns:
        A :class:`~eplb.plan.Plan` with placement ``x``, routing quota ``q`` and makespan ``tau``.
    """
    cfg = cfg or EPLBConfig()
    R = topo.num_ranks
    E = spec.num_experts
    device = loads.device
    dom = topo.domain_of_rank
    cost = topo.cost
    n_slot = int(spec.n_slot)

    if R == 0 or E == 0:
        q = torch.zeros((R, E, R), dtype=torch.int64, device=device)
        return Plan(x=torch.zeros((E, R), dtype=torch.int8, device=device), q=q, tau=0)

    lam = loads.lam
    dom_list = dom.tolist()
    x_cur, su = _stage01_placement(loads, topo, spec, cfg)
    su = su.tolist()

    def _propose(load: list, contrib: torch.Tensor, x: torch.Tensor, target: int) -> list:
        """Propose intra-domain replicas to push every rank's load <= ``target`` (seeded from the exact ``load``).

        Args:
            load: Exact per-rank token load (length ``R``) from the latest route.
            contrib: int64 ``[E, R]`` per-(expert, dst) token contribution from that route.
            x: int8 ``[E, R]`` current placement.
            target: Makespan threshold to drive every rank under.

        Returns:
            List of ``(expert, rank)`` intra-domain replicas to add.
        """
        load = list(load)
        cell = []
        for r in range(R):
            nz = torch.nonzero(contrib[:, r] > 0, as_tuple=False).flatten().tolist()
            cell.append({e: int(contrib[e, r].item()) for e in nz})
        hosts: dict[int, set] = {}
        for r in range(R):
            for e in torch.nonzero(x[:, r] == 1, as_tuple=False).flatten().tolist():
                hosts.setdefault(e, set()).add(r)
        su_local = list(su)
        stuck: set = set()
        added: list = []
        while True:
            cand = [r for r in range(R) if load[r] > target and r not in stuck]
            if not cand:
                return added
            r_star = min(cand, key=lambda r: (-load[r], r))
            d_star = dom_list[r_star]
            ccell = cell[r_star]
            max_c = max(ccell.values()) if ccell else 0
            if max_c == 0:
                stuck.add(r_star)
                continue
            e_star = min(e for e, v in ccell.items() if v == max_c)
            hosts_e = hosts.get(e_star, set())
            elig = [t for t in range(R)
                    if dom_list[t] == d_star and su_local[t] < n_slot and t not in hosts_e]
            if not elig:
                stuck.add(r_star)
                continue
            t = min(elig, key=lambda t: (load[t], int(cost[r_star, t].item()), t))
            # re-waterfill e_star's in-domain(d*) tokens over its d*-instances incl. the new replica
            H = sorted(h for h in hosts_e if dom_list[h] == d_star)
            H.append(t)
            base_vals = [load[h] - cell[h].get(e_star, 0) for h in H]
            pool = sum(cell[h].get(e_star, 0) for h in H)
            base_t = torch.tensor(base_vals, dtype=torch.int64, device=device)
            tie_t = torch.zeros(len(H), dtype=torch.int64, device=device)
            addv = _waterfill(pool, base_t, tie_t).tolist()
            for h, a, b in zip(H, addv, base_vals):
                load[h] = b + a
                cell[h][e_star] = a
            hosts.setdefault(e_star, set()).add(t)
            su_local[t] += 1
            added.append((e_star, t))
            stuck = set()  # loads changed; re-evaluate every rank

    # makespan-bisection descent: each step halves the gap to the even-split lower bound (O(log L) routes)
    lo = (int(lam.sum().item()) + R - 1) // R
    best_x = x_cur.clone()
    best_q, best_load = _assign_quota(lam, best_x, cost, dom)
    best_tau = int(best_load.max().item())
    for _ in range(int(cfg.tau_bisect_iters)):
        q, load = _assign_quota(lam, x_cur, cost, dom)
        tau = int(load.max().item())
        if tau < best_tau:
            best_tau, best_x, best_q = tau, x_cur.clone(), q
        if tau <= lo:
            break
        target = (lo + tau) // 2
        added = _propose(load.tolist(), q.sum(dim=0).to(torch.int64), x_cur, target)
        if not added:
            break
        for (e, t) in added:
            x_cur[e, t] = 1
            su[t] += 1

    return Plan(x=best_x, q=best_q, tau=best_tau)

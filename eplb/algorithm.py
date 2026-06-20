"""The Scale-EPLB deterministic solver.

Structure (mirrors the problem-definition doc's "algorithm design"):

    Stage 0  precompute        lambda_e, T[d,e], break-even threshold T*[e]
    Stage 1  cross-domain gate  per (e, d!=dom(main)) admit a replica iff C6 holds,
                                greedily by marginal benefit, respecting slots
    Stage 2  intra balancing    lower the makespan tau by iteratively adding
                                replicas (argmax-slack / LPT placement) until no
                                free slot helps; assign the routing quota q

Determinism contract
--------------------
Every input is an integer tensor and every decision uses integer arithmetic with
a fully specified, rank-independent tie-break order (value, then expert id, then
rank id). Therefore all ranks that all-gather the same ``Lambda`` compute a
bit-identical :class:`~eplb.plan.Plan`. No floating point is used on the decision
path; no broadcast or CPU synchronisation is required.

Replication, not rearrangement
------------------------------
``main(e)`` is fixed (C7); ``x[e, main(e)] = 1`` always. The solver only *adds*
replicas, so the logical->physical mapping never changes and gradients can be
aggregated back to the single optimizer owner ``main(e)``.

.. note::
   This is the *reference* implementation: correct, deterministic and
   constraint-satisfying, optimised for clarity over latency. The hot path
   (``_assign_quota`` + the Stage 2 loop) is what the production single-SM CUDA
   kernel replaces; the Python here is the oracle used to validate that kernel.
"""

from __future__ import annotations

import torch

from .config import EPLBConfig
from .loads import Loads
from .plan import Plan
from .problem import ProblemSpec
from .topology import Topology


def _lexsort_keys(*keys: torch.Tensor) -> torch.Tensor:
    """Return indices that sort by ``keys[0]``, then ``keys[1]``, ... ascending.

    All keys must be 1-D int64 tensors of equal length and non-negative. Uses
    successive stable sorts (last key is the most significant), which keeps the
    ordering deterministic and independent of any prior ordering.
    """
    n = keys[0].numel()
    order = torch.arange(n, device=keys[0].device, dtype=torch.int64)
    # Apply stable sorts from least- to most-significant key. ``keys[0]`` is the
    # primary key, so it must be applied last.
    for key in reversed(keys):
        k = key[order]
        perm = torch.argsort(k, stable=True)
        order = order[perm]
    return order


def _waterfill(need: int, base: torch.Tensor, tie: torch.Tensor) -> torch.Tensor:
    """Distribute ``need`` integer units across destinations to minimise the
    resulting maximum of ``base + add``, breaking ties by ``tie`` then index.

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

    # Order destinations by (current load, tie key, original index).
    idx = torch.arange(D, dtype=torch.int64, device=base.device)
    order = _lexsort_keys(base, tie, idx)
    b = base[order]  # ascending by load (with tie order)

    rem = int(need)
    # Raise the "water level" across a growing prefix of the sorted destinations.
    # Cost to lift the first k destinations up to b[k] is sum_{j<k}(b[k]-b[j]).
    # Find the largest prefix length k we can fully level within `rem`.
    k = 1
    while k < D:
        # cost to bring first k dests from their levels up to b[k]
        cost_to_next = int((b[k] * k - torch.sum(b[:k])).item())
        if cost_to_next > rem:
            break
        k += 1
    # Now level the first k destinations evenly with whatever remains.
    level_floor = int(b[k - 1].item())
    base_cost = int((level_floor * k - torch.sum(b[:k])).item())
    rem_after = rem - base_cost  # >= 0 by construction
    add_sorted = torch.zeros(D, dtype=torch.int64, device=base.device)
    # bring the first k up to level_floor
    add_sorted[:k] = level_floor - b[:k]
    # spread rem_after evenly across the k destinations (deterministic remainder)
    share = rem_after // k
    extra = rem_after - share * k
    add_sorted[:k] += share
    add_sorted[:extra] += 1  # first `extra` in sorted order get one more
    # scatter back to original destination indices
    add[order] = add_sorted
    return add


def _assign_quota(
    lam: torch.Tensor,
    x: torch.Tensor,
    cost: torch.Tensor,
):
    """Given a fixed placement ``x``, route every token to an instance so as to
    minimise the makespan, using communication cost as a tie-break.

    Args:
        lam: int64 ``[R, E]`` load matrix.
        x: int8 ``[E, R]`` placement.
        cost: int64 ``[R, R]`` per-token comm cost.

    Returns:
        ``(q, load)`` where ``q`` is int64 ``[R, E, R]`` and ``load`` is
        int64 ``[R]`` per-destination token counts.
    """
    R = lam.shape[0]
    E = lam.shape[1]
    device = lam.device
    q = torch.zeros((R, E, R), dtype=torch.int64, device=device)
    load = torch.zeros(R, dtype=torch.int64, device=device)

    # Process (r, e) pairs in descending token count (LPT) for better packing;
    # ties broken by (e, r) for determinism.
    rr, ee = torch.meshgrid(
        torch.arange(R, device=device, dtype=torch.int64),
        torch.arange(E, device=device, dtype=torch.int64),
        indexing="ij",
    )
    flat_r = rr.reshape(-1)
    flat_e = ee.reshape(-1)
    flat_lam = lam.reshape(-1)
    neg_lam = -flat_lam  # ascending sort on -lam == descending lam
    order = _lexsort_keys(neg_lam, flat_e, flat_r)

    for idx in order.tolist():
        need = int(flat_lam[idx].item())
        if need == 0:
            continue
        r = int(flat_r[idx].item())
        e = int(flat_e[idx].item())
        dests = torch.nonzero(x[e] == 1, as_tuple=False).flatten()
        # waterfill onto destination loads, tie-break by comm cost from r
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

    lam = loads.lam
    dom = topo.domain_of_rank
    cost = topo.cost
    main_rank = spec.main_rank
    main_dom = dom[main_rank]  # [E]
    W = spec.weight_bytes
    s_tok = int(spec.s_tok)
    n_slot = int(spec.n_slot)
    M = topo.num_domains

    # ---- Stage 0: precompute -------------------------------------------------
    Tde = loads.domain_demand(dom, M)  # [M, E]

    # ---- placement init: main fixed (C7) ------------------------------------
    x = torch.zeros((E, R), dtype=torch.int8, device=device)
    x[torch.arange(E, device=device, dtype=torch.int64), main_rank] = 1
    slot_used = x.sum(dim=0).to(torch.int64)  # [R]

    # ---- Stage 1: cross-domain replication gate (C6) ------------------------
    # Admit a cross-domain replica of e in domain d only when the one-time weight
    # move (counted x2 for the gradient return in training) is cheaper than
    # repeatedly shipping that domain's tokens for e across the boundary:
    #     |W_e| < 2 * T[d,e] * s_tok                                       (C6)
    # Greedily admit in descending marginal benefit, respecting the slot budget.
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
                    cand_benefit.append(2 * t * s_tok - we)  # token bytes saved
        if cand_e:
            ce = torch.tensor(cand_e, dtype=torch.int64, device=device)
            cd = torch.tensor(cand_d, dtype=torch.int64, device=device)
            cb = torch.tensor(cand_benefit, dtype=torch.int64, device=device)
            # order: benefit desc, then expert id asc, then domain id asc
            order = _lexsort_keys(-cb, ce, cd)
            for idx in order.tolist():
                e = int(ce[idx].item())
                d = int(cd[idx].item())
                ranks_d = topo.ranks_in_domain(d)
                if int(x[e, ranks_d].sum().item()) > 0:
                    continue  # already has an instance in this domain
                free_mask = slot_used[ranks_d] < n_slot
                free = ranks_d[free_mask]
                if free.numel() == 0:
                    continue
                # choose the least-loaded rank in the domain (argmax slack), tie by id
                chosen_order = _lexsort_keys(slot_used[free], free)
                chosen = int(free[chosen_order[0]].item())
                x[e, chosen] = 1
                slot_used[chosen] += 1

    # ---- Stage 2: intra-domain balancing via iterative replica insertion ----
    # Assign quota for the current placement, then repeatedly add the replica
    # that most relieves the makespan rank, until no free slot can help. Each
    # accepted replica strictly lowers (or holds) tau; we stop when it no longer
    # improves. This is the deterministic stand-in for the tau-bisection + LPT
    # described in the doc.
    q, load = _assign_quota(lam, x, cost)
    tau = int(load.max().item()) if R > 0 else 0

    iters = 0
    while iters < cfg.max_stage2_iters:
        iters += 1
        if int((slot_used < n_slot).sum().item()) == 0:
            break  # no free slots anywhere -> C4 saturated

        # makespan rank (smallest id among the maxima for determinism)
        max_load = int(load.max().item())
        if max_load == 0:
            break
        bottleneck_ranks = torch.nonzero(load == max_load, as_tuple=False).flatten()
        r_star = int(bottleneck_ranks.min().item())

        # the expert contributing the most tokens *to* r_star (tie by expert id)
        contrib = q[:, :, r_star].sum(dim=0)  # [E]
        max_contrib = int(contrib.max().item())
        if max_contrib == 0:
            break
        cand_experts = torch.nonzero(contrib == max_contrib, as_tuple=False).flatten()
        e_star = int(cand_experts.min().item())

        # place a new replica of e_star on a free-slot rank with maximum slack
        # (lowest current load), preferring an already-cheap-to-reach rank; never
        # a rank that already hosts e_star.
        hosts = x[e_star] == 1
        eligible_mask = (slot_used < n_slot) & (~hosts)
        if not cfg.allow_cross_domain:
            # keep all instances of e_star inside its home (main) NVLink domain
            home = int(main_dom[e_star].item())
            eligible_mask = eligible_mask & (dom == home)
        eligible = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        if eligible.numel() == 0:
            # Cannot place another replica of this bottleneck expert (it already
            # occupies every rank with a free slot, or none remain in-domain).
            # Stop rather than loop without progress.
            break

        place_order = _lexsort_keys(
            load[eligible], cost[r_star, eligible], eligible
        )
        target = int(eligible[place_order[0]].item())

        # tentatively add and re-assign; accept only if tau does not increase
        x[e_star, target] = 1
        new_q, new_load = _assign_quota(lam, x, cost)
        new_tau = int(new_load.max().item())
        if new_tau < tau:
            slot_used[target] += 1
            q, load, tau = new_q, new_load, new_tau
        else:
            x[e_star, target] = 0  # revert; this replica did not help
            break

    return Plan(x=x, q=q, tau=tau)

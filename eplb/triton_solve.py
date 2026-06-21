"""Fully-fused single Triton kernel for the whole solver (Stage 1 + Stage 2), zero host sync, bit-identical to the CPU reference."""

from __future__ import annotations

import torch

from .plan import Plan

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # triton missing or unimportable -> caller falls back to the reference
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr,
               R, E, WRITE_Q: tl.constexpr, BLOCK_R: tl.constexpr):
        """Serial LPT routing over the precomputed (r,e) order; returns load[BLOCK_R], writes q iff WRITE_Q."""
        lane = tl.arange(0, BLOCK_R)
        lane_mask = lane < R
        lane64 = lane.to(tl.int64)
        dom_all = tl.load(dom_ptr + lane, mask=lane_mask, other=0)
        load = tl.zeros([BLOCK_R], tl.int64)
        ids = lane64
        for t in range(R * E):
            idx = tl.load(order_ptr + t)
            need = tl.load(lam_ptr + idx)
            r = idx // E
            e = idx - r * E
            dr = tl.load(dom_ptr + r)
            hosts = tl.load(x_ptr + e * R + lane64, mask=lane_mask, other=0)
            hosts_b = (hosts != 0) & lane_mask
            cost_r = tl.load(cost_ptr + r * R + lane64, mask=lane_mask, other=0)
            in_dom = hosts_b & (dom_all == dr)
            has_in = tl.sum(in_dom.to(tl.int64)) > 0
            active = tl.where(has_in, in_dom, hosts_b)

            base = load
            tie = cost_r
            a_base = base[:, None]; b_base = base[None, :]
            a_tie = tie[:, None]; b_tie = tie[None, :]
            a_id = ids[:, None]; b_id = ids[None, :]
            act_a = active[:, None]; act_b = active[None, :]
            before = ((b_base < a_base)
                      | ((b_base == a_base) & (b_tie < a_tie))
                      | ((b_base == a_base) & (b_tie == a_tie) & (b_id < a_id)))
            before = before & act_a & act_b
            before_i = before.to(tl.int64)
            pos = tl.sum(before_i, axis=1)
            s_excl = tl.sum(before_i * b_base, axis=1)
            cond = active & (pos >= 1) & (base * pos - s_excl <= need)
            k = 1 + tl.sum(cond.to(tl.int64))
            level_floor = tl.sum(tl.where(active & (pos == (k - 1)), base, 0))
            s_lt_k = tl.sum(tl.where(active & (pos < k), base, 0))
            base_cost = level_floor * k - s_lt_k
            rem_after = need - base_cost
            share = rem_after // k
            extra = rem_after - share * k
            first_k = active & (pos < k)
            zeros = tl.zeros([BLOCK_R], tl.int64)
            add = tl.where(first_k, (level_floor - base) + share, zeros)
            add = tl.where(first_k & (pos < extra), add + 1, add)
            add = tl.where(need > 0, add, zeros)
            if WRITE_Q:
                tl.store(q_ptr + (r * E + e) * R + lane64, add, mask=lane_mask)
            load = load + add
        return load

    @triton.jit
    def _solve_kernel(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr,
                      cand_e_ptr, cand_d_ptr, cand_valid_ptr, slot_used_ptr,
                      load_out_ptr,
                      R, E, M, EM, n_slot, allow_cd, max_iters,
                      BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr):
        """One program: Stage 1 admission scan + Stage 2 relief loop entirely on-device (no host sync)."""
        BIG = 1 << 62
        lane = tl.arange(0, BLOCK_R)
        lane_mask = lane < R
        lane_i = lane.to(tl.int64)
        lane64 = lane_i
        dom_all = tl.load(dom_ptr + lane, mask=lane_mask, other=0)
        slot_used = tl.load(slot_used_ptr + lane, mask=lane_mask, other=0)

        # Stage 1: serial admission over benefit-sorted cross-domain candidates
        if (allow_cd != 0) and (M > 1):
            for c in range(EM):
                valid = tl.load(cand_valid_ptr + c)
                e = tl.load(cand_e_ptr + c)
                d = tl.load(cand_d_ptr + c)
                x_row = tl.load(x_ptr + e * R + lane64, mask=lane_mask, other=0)
                in_d = (dom_all == d) & lane_mask
                has_inst = tl.sum(tl.where(in_d, x_row.to(tl.int64), 0)) > 0
                free = in_d & (slot_used < n_slot)
                has_free = tl.sum(free.to(tl.int64)) > 0
                do_admit = (valid != 0) & (not has_inst) & has_free
                m1 = tl.min(tl.where(free, slot_used, BIG))
                c1 = free & (slot_used == m1)
                chosen = tl.min(tl.where(c1, lane_i, BIG))
                sel = lane_mask & (lane_i == chosen) & do_admit
                tl.store(x_ptr + e * R + lane64, tl.full([BLOCK_R], 1, tl.int8), mask=sel)
                slot_used = tl.where(sel, slot_used + 1, slot_used)

        # commit initial routing
        load = _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, R, E, True, BLOCK_R)
        tau = tl.max(tl.where(lane_mask, load, -BIG))
        stuck = lane < 0  # all False

        ce = tl.arange(0, BLOCK_E)
        ce_mask = ce < E
        ce_i = ce.to(tl.int64)
        TRUE_VEC = lane >= 0
        FALSE_VEC = lane < 0

        done = False
        for _it in range(max_iters):
            if not done:
                has_free_any = tl.sum(tl.where(lane_mask & (slot_used < n_slot), 1, 0)) > 0
                masked = tl.where(stuck, tl.full([BLOCK_R], -1, tl.int64), load)
                max_load = tl.max(tl.where(lane_mask, masked, -BIG))
                if (not has_free_any) or (max_load <= 0):
                    done = True
                else:
                    r_star = tl.min(tl.where(lane_mask & (masked == max_load), lane_i, BIG))
                    d_star = tl.load(dom_ptr + r_star)
                    contrib = tl.zeros([BLOCK_E], tl.int64)
                    for r in range(R):
                        off = ce_i * R + (r * E * R + r_star)
                        contrib += tl.load(q_ptr + off, mask=ce_mask, other=0)
                    max_contrib = tl.max(tl.where(ce_mask, contrib, -BIG))
                    if max_contrib == 0:
                        stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)
                    else:
                        e_star = tl.min(tl.where(ce_mask & (contrib == max_contrib), ce_i, BIG))
                        x_estar = tl.load(x_ptr + e_star * R + lane64, mask=lane_mask, other=0)
                        hosts_b = x_estar != 0
                        elig = lane_mask & (slot_used < n_slot) & (~hosts_b) & (dom_all == d_star)
                        has_elig = tl.sum(elig.to(tl.int64)) > 0
                        if not has_elig:
                            stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)
                        else:
                            cost_rstar = tl.load(cost_ptr + r_star * R + lane64, mask=lane_mask, other=0)
                            m1 = tl.min(tl.where(elig, load, BIG))
                            e1 = elig & (load == m1)
                            m2 = tl.min(tl.where(e1, cost_rstar, BIG))
                            e2 = e1 & (cost_rstar == m2)
                            target = tl.min(tl.where(e2, lane_i, BIG))
                            sel_t = lane_mask & (lane_i == target)
                            tl.store(x_ptr + e_star * R + lane64, tl.full([BLOCK_R], 1, tl.int8), mask=sel_t)
                            new_load = _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, R, E, False, BLOCK_R)
                            new_tau = tl.max(tl.where(lane_mask, new_load, -BIG))
                            nl_rstar = tl.sum(tl.where(lane_i == r_star, new_load, 0))
                            l_rstar = tl.sum(tl.where(lane_i == r_star, load, 0))
                            accept = (nl_rstar < l_rstar) & (new_tau <= tau)
                            if accept:
                                slot_used = tl.where(sel_t, slot_used + 1, slot_used)
                                load = _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, R, E, True, BLOCK_R)
                                tau = new_tau
                                stuck = FALSE_VEC
                            else:
                                tl.store(x_ptr + e_star * R + lane64, tl.zeros([BLOCK_R], tl.int8), mask=sel_t)
                                stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)

        tl.store(load_out_ptr + lane, load, mask=lane_mask)


def solve_fused(loads, topo, spec, cfg) -> Plan:
    """Run the entire Scale-EPLB solver in one Triton launch (Stage 1 + Stage 2), bit-identical to :func:`eplb.algorithm.solve`.

    Args:
        loads: Dynamic load matrix ``Lambda`` on a CUDA device.
        topo: Cluster topology.
        spec: Static problem spec (main placement, weights, slot budget).
        cfg: Solver configuration.

    Returns:
        A :class:`~eplb.plan.Plan` with placement ``x``, routing quota ``q`` and makespan ``tau``.
    """
    dev = loads.device
    R = topo.num_ranks
    E = spec.num_experts
    # sync-free upper bound on domain count (domains <= ranks); empty padding domains
    # yield zero-demand candidates that the C6 gate filters out, so the plan is unchanged.
    M = R
    lam = loads.lam.to(torch.int64)
    dom = topo.domain_of_rank.to(torch.int64).contiguous()
    cost = topo.cost.to(torch.int64).contiguous()
    main_rank = spec.main_rank.to(torch.int64)
    W = spec.weight_bytes.to(torch.int64)
    s_tok = int(spec.s_tok)
    n_slot = int(spec.n_slot)

    x = torch.zeros((E, R), dtype=torch.int8, device=dev)
    x.scatter_(1, main_rank.view(E, 1), 1)
    slot_used = x.sum(0).to(torch.int64)

    # static (r,e) LPT order by (-lam, e, r) -- depends only on lam, reused for every re-route
    flat_lam = lam.reshape(-1).contiguous()
    idx = torch.arange(R * E, device=dev, dtype=torch.int64)
    fe = idx % E
    fr = idx // E
    order = idx.clone()
    for key in (fr, fe, -flat_lam):
        order = order[torch.argsort(key[order], stable=True)]
    order = order.contiguous()

    # Stage 1 candidates over (e,d), sorted by (valid desc, benefit desc, e asc, d asc)
    Tde = loads.domain_demand(dom, M)
    main_dom = dom[main_rank]
    ee = torch.arange(E, device=dev, dtype=torch.int64).repeat_interleave(M)
    dd = torch.arange(M, device=dev, dtype=torch.int64).repeat(E)
    t = Tde[dd, ee]
    we = W[ee]
    benefit = 2 * t * s_tok - we
    valid = (dd != main_dom[ee]) & (t > 0) & (we < 2 * t * s_tok)
    cand_order = torch.arange(E * M, device=dev, dtype=torch.int64)
    for key in (dd, ee, -benefit, (~valid).to(torch.int64)):
        cand_order = cand_order[torch.argsort(key[cand_order], stable=True)]
    cand_e = ee[cand_order].contiguous()
    cand_d = dd[cand_order].contiguous()
    cand_valid = valid[cand_order].to(torch.int64).contiguous()

    q = torch.zeros((R, E, R), dtype=torch.int64, device=dev)
    load_out = torch.zeros(R, dtype=torch.int64, device=dev)
    BLOCK_R = triton.next_power_of_2(max(R, 1))
    BLOCK_E = triton.next_power_of_2(max(E, 1))
    _solve_kernel[(1,)](
        flat_lam, x, cost, dom, order, q,
        cand_e, cand_d, cand_valid, slot_used, load_out,
        R, E, M, E * M, n_slot, 1 if cfg.allow_cross_domain else 0, cfg.max_stage2_iters,
        BLOCK_R=BLOCK_R, BLOCK_E=BLOCK_E,
    )
    # fully sync-free: tau stays a 0-dim device tensor (tau == max committed rank load).
    # the hot dispatch path never reads tau; consumers needing a Python int coerce lazily via int(plan.tau).
    return Plan(x=x, q=q, tau=load_out.max())


if HAS_TRITON:

    @triton.jit
    def _wf(base, tie, ids, active, need, BLOCK_R: tl.constexpr):
        """Integer waterfill: add ``need`` over ``active`` lanes minimising max(base+add), tie by (tie, id).

        Returns the per-lane additions (0 on inactive lanes / when ``need<=0``); matches the
        CPU ``_waterfill`` exactly (rank-independent tie-break)."""
        a_base = base[:, None]; b_base = base[None, :]
        a_tie = tie[:, None]; b_tie = tie[None, :]
        a_id = ids[:, None]; b_id = ids[None, :]
        act_a = active[:, None]; act_b = active[None, :]
        before = ((b_base < a_base)
                  | ((b_base == a_base) & (b_tie < a_tie))
                  | ((b_base == a_base) & (b_tie == a_tie) & (b_id < a_id)))
        before = before & act_a & act_b
        before_i = before.to(tl.int64)
        pos = tl.sum(before_i, axis=1)
        s_excl = tl.sum(before_i * b_base, axis=1)
        cond = active & (pos >= 1) & (base * pos - s_excl <= need)
        k = 1 + tl.sum(cond.to(tl.int64))
        level_floor = tl.sum(tl.where(active & (pos == (k - 1)), base, 0))
        s_lt_k = tl.sum(tl.where(active & (pos < k), base, 0))
        base_cost = level_floor * k - s_lt_k
        rem_after = need - base_cost
        share = rem_after // k
        extra = rem_after - share * k
        first_k = active & (pos < k)
        zeros = tl.zeros([BLOCK_R], tl.int64)
        add = tl.where(first_k, (level_floor - base) + share, zeros)
        add = tl.where(first_k & (pos < extra), add + 1, add)
        add = tl.where(need > 0, add, zeros)
        return add

    @triton.jit
    def _route_dom(lam_ptr, x_ptr, cost_ptr, order_ptr, cell_ptr,
                   in_dom, ids, load_seed, R, E, BLOCK_R: tl.constexpr):
        """In-domain route for one domain: seed load with floor, waterfill in-domain tokens over
        in-domain instances, write per-(expert,dst) tokens into ``cell``; return per-lane load."""
        lane = tl.arange(0, BLOCK_R)
        lane_mask = lane < R
        lane64 = lane.to(tl.int64)
        load = load_seed
        # zero this domain's columns of cell
        for ez in range(E):
            tl.store(cell_ptr + ez * R + lane64, tl.zeros([BLOCK_R], tl.int64), mask=in_dom)
        for t in range(R * E):
            idx = tl.load(order_ptr + t)
            rr = idx // E
            ev = idx - rr * E
            need = tl.load(lam_ptr + idx)
            hosts = tl.load(x_ptr + ev * R + lane64, mask=lane_mask, other=0) != 0
            cost_r = tl.load(cost_ptr + rr * R + lane64, mask=lane_mask, other=0)
            active = hosts & in_dom
            r_in = tl.sum(tl.where(lane64 == rr, in_dom.to(tl.int64), 0)) > 0  # rr belongs to this domain
            add = _wf(load, cost_r, ids, active, need, BLOCK_R)
            add = tl.where(r_in, add, tl.zeros([BLOCK_R], tl.int64))
            cur = tl.load(cell_ptr + ev * R + lane64, mask=lane_mask, other=0)
            tl.store(cell_ptr + ev * R + lane64, cur + add, mask=lane_mask)
            load = load + add
        return load

    @triton.jit
    def _seed_kernel(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr,
                     cand_e_ptr, cand_d_ptr, cand_valid_ptr, slot_used_ptr, load_out_ptr,
                     R, E, M, EM, n_slot, allow_cd, BLOCK_R: tl.constexpr):
        """Stage 1 admission + one global seed route (writes q_seed, load0, updated slot_used)."""
        BIG = 1 << 62
        lane = tl.arange(0, BLOCK_R)
        lane_mask = lane < R
        lane_i = lane.to(tl.int64)
        lane64 = lane_i
        dom_all = tl.load(dom_ptr + lane, mask=lane_mask, other=0)
        slot_used = tl.load(slot_used_ptr + lane, mask=lane_mask, other=0)
        if (allow_cd != 0) and (M > 1):
            for c in range(EM):
                valid = tl.load(cand_valid_ptr + c)
                e = tl.load(cand_e_ptr + c)
                d = tl.load(cand_d_ptr + c)
                x_row = tl.load(x_ptr + e * R + lane64, mask=lane_mask, other=0)
                in_d = (dom_all == d) & lane_mask
                has_inst = tl.sum(tl.where(in_d, x_row.to(tl.int64), 0)) > 0
                free = in_d & (slot_used < n_slot)
                has_free = tl.sum(free.to(tl.int64)) > 0
                do_admit = (valid != 0) & (not has_inst) & has_free
                m1 = tl.min(tl.where(free, slot_used, BIG))
                c1 = free & (slot_used == m1)
                chosen = tl.min(tl.where(c1, lane_i, BIG))
                sel = lane_mask & (lane_i == chosen) & do_admit
                tl.store(x_ptr + e * R + lane64, tl.full([BLOCK_R], 1, tl.int8), mask=sel)
                slot_used = tl.where(sel, slot_used + 1, slot_used)
        load = _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, R, E, True, BLOCK_R)
        tl.store(load_out_ptr + lane, load, mask=lane_mask)
        tl.store(slot_used_ptr + lane, slot_used, mask=lane_mask)

    @triton.jit
    def _bisect_kernel(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, cell_ptr, floor_ptr,
                       slot_used_ptr, xbest_ptr,
                       R, E, n_slot, iters, max_propose,
                       BLOCK_R: tl.constexpr, BLOCK_E: tl.constexpr):
        """One program per domain: makespan tau-descent on the frozen-floor in-domain load model."""
        BIG = 1 << 62
        d = tl.program_id(0)
        lane = tl.arange(0, BLOCK_R)
        lane_mask = lane < R
        lane_i = lane.to(tl.int64)
        lane64 = lane_i
        ids = lane64
        dom_all = tl.load(dom_ptr + lane, mask=lane_mask, other=0)
        in_dom = (dom_all == d) & lane_mask
        dom_cnt = tl.sum(in_dom.to(tl.int64))
        TRUE_VEC = lane >= 0
        FALSE_VEC = lane < 0
        ce = tl.arange(0, BLOCK_E)
        ce_mask = ce < E
        ce_i = ce.to(tl.int64)

        if dom_cnt > 0:
            floor_v = tl.load(floor_ptr + lane, mask=lane_mask, other=0)
            slot_used = tl.load(slot_used_ptr + lane, mask=lane_mask, other=0)
            best_tau = tl.full((), BIG, tl.int64)
            lo = tl.full((), -1, tl.int64)
            done = False
            for _it in range(iters):
                if not done:
                    seed = tl.where(in_dom, floor_v, tl.zeros([BLOCK_R], tl.int64))
                    load = _route_dom(lam_ptr, x_ptr, cost_ptr, order_ptr, cell_ptr,
                                      in_dom, ids, seed, R, E, BLOCK_R)
                    tau = tl.max(tl.where(in_dom, load, -BIG))
                    if lo < 0:
                        tot = tl.sum(tl.where(in_dom, load, 0))
                        lo = (tot + dom_cnt - 1) // dom_cnt
                    if tau < best_tau:
                        best_tau = tau
                        for e in range(E):
                            xv = tl.load(x_ptr + e * R + lane64, mask=in_dom, other=0)
                            tl.store(xbest_ptr + e * R + lane64, xv, mask=in_dom)
                    if tau <= lo:
                        done = True
                    else:
                        target = (lo + tau) // 2
                        load_p = load
                        stuck = FALSE_VEC
                        pdone = False
                        for _p in range(max_propose):
                            if not pdone:
                                cand = in_dom & (load_p > target) & (~stuck)
                                if tl.sum(cand.to(tl.int64)) == 0:
                                    pdone = True
                                else:
                                    maxl = tl.max(tl.where(cand, load_p, -BIG))
                                    r_star = tl.min(tl.where(cand & (load_p == maxl), lane_i, BIG))
                                    col = tl.load(cell_ptr + ce_i * R + r_star, mask=ce_mask, other=0)
                                    maxc = tl.max(tl.where(ce_mask, col, -BIG))
                                    if maxc == 0:
                                        stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)
                                    else:
                                        e_star = tl.min(tl.where(ce_mask & (col == maxc), ce_i, BIG))
                                        xrow = tl.load(x_ptr + e_star * R + lane64, mask=lane_mask, other=0) != 0
                                        elig = in_dom & (slot_used < n_slot) & (~xrow)
                                        if tl.sum(elig.to(tl.int64)) == 0:
                                            stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)
                                        else:
                                            cost_rs = tl.load(cost_ptr + r_star * R + lane64, mask=lane_mask, other=0)
                                            m1 = tl.min(tl.where(elig, load_p, BIG))
                                            e1 = elig & (load_p == m1)
                                            m2 = tl.min(tl.where(e1, cost_rs, BIG))
                                            e2 = e1 & (cost_rs == m2)
                                            t_sel = tl.min(tl.where(e2, lane_i, BIG))
                                            cell_es = tl.load(cell_ptr + e_star * R + lane64, mask=lane_mask, other=0)
                                            host_es = (xrow & in_dom) | (lane_i == t_sel)
                                            base = load_p - cell_es
                                            ztie = tl.zeros([BLOCK_R], tl.int64)
                                            pool = tl.sum(tl.where(host_es, cell_es, 0))
                                            add = _wf(base, ztie, ids, host_es, pool, BLOCK_R)
                                            load_p = tl.where(host_es, base + add, load_p)
                                            tl.store(cell_ptr + e_star * R + lane64, add, mask=host_es)
                                            sel_t = (lane_i == t_sel) & lane_mask
                                            tl.store(x_ptr + e_star * R + lane64,
                                                     tl.full([BLOCK_R], 1, tl.int8), mask=sel_t)
                                            slot_used = tl.where(sel_t, slot_used + 1, slot_used)
                                            stuck = FALSE_VEC
            # restore the best placement seen for this domain
            for e in range(E):
                bv = tl.load(xbest_ptr + e * R + lane64, mask=in_dom, other=0)
                tl.store(x_ptr + e * R + lane64, bv, mask=in_dom)

    @triton.jit
    def _route_only_kernel(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, load_out_ptr,
                           R, E, BLOCK_R: tl.constexpr):
        """Standalone global route (writes q + load) for exact final accounting."""
        lane = tl.arange(0, BLOCK_R)
        lane_mask = lane < R
        load = _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, R, E, True, BLOCK_R)
        tl.store(load_out_ptr + lane, load, mask=lane_mask)


def solve_bisect_fused(loads, topo, spec, cfg) -> Plan:
    """GPU per-domain tau-bisection in three async Triton launches (zero host sync), aligned
    with the :func:`eplb.algorithm.solve_bisect` CPU reference.

    Args:
        loads: Dynamic load matrix ``Lambda`` on a CUDA device.
        topo: Cluster topology.
        spec: Static problem spec.
        cfg: Solver configuration (uses ``cfg.tau_bisect_iters`` per-domain step budget).

    Returns:
        A :class:`~eplb.plan.Plan` with placement ``x``, routing quota ``q`` and 0-dim device ``tau``.
    """
    dev = loads.device
    R = topo.num_ranks
    E = spec.num_experts
    M = R  # sync-free upper bound on domain count; empty padding domains are skipped in-kernel
    lam = loads.lam.to(torch.int64)
    dom = topo.domain_of_rank.to(torch.int64).contiguous()
    cost = topo.cost.to(torch.int64).contiguous()
    main_rank = spec.main_rank.to(torch.int64)
    W = spec.weight_bytes.to(torch.int64)
    s_tok = int(spec.s_tok)
    n_slot = int(spec.n_slot)

    x = torch.zeros((E, R), dtype=torch.int8, device=dev)
    x.scatter_(1, main_rank.view(E, 1), 1)
    slot_used = x.sum(0).to(torch.int64)

    flat_lam = lam.reshape(-1).contiguous()
    idx = torch.arange(R * E, device=dev, dtype=torch.int64)
    fe = idx % E
    fr = idx // E
    order = idx.clone()
    for key in (fr, fe, -flat_lam):
        order = order[torch.argsort(key[order], stable=True)]
    order = order.contiguous()

    Tde = loads.domain_demand(dom, M)
    main_dom = dom[main_rank]
    ee = torch.arange(E, device=dev, dtype=torch.int64).repeat_interleave(M)
    dd = torch.arange(M, device=dev, dtype=torch.int64).repeat(E)
    t = Tde[dd, ee]
    we = W[ee]
    benefit = 2 * t * s_tok - we
    valid = (dd != main_dom[ee]) & (t > 0) & (we < 2 * t * s_tok)
    cand_order = torch.arange(E * M, device=dev, dtype=torch.int64)
    for key in (dd, ee, -benefit, (~valid).to(torch.int64)):
        cand_order = cand_order[torch.argsort(key[cand_order], stable=True)]
    cand_e = ee[cand_order].contiguous()
    cand_d = dd[cand_order].contiguous()
    cand_valid = valid[cand_order].to(torch.int64).contiguous()

    BLOCK_R = triton.next_power_of_2(max(R, 1))
    BLOCK_E = triton.next_power_of_2(max(E, 1))

    # launch 1: Stage 1 admission + seed route
    q_seed = torch.zeros((R, E, R), dtype=torch.int64, device=dev)
    load0 = torch.zeros(R, dtype=torch.int64, device=dev)
    _seed_kernel[(1,)](
        flat_lam, x, cost, dom, order, q_seed,
        cand_e, cand_d, cand_valid, slot_used, load0,
        R, E, M, E * M, n_slot, 1 if cfg.allow_cross_domain else 0,
        BLOCK_R=BLOCK_R,
    )

    # frozen cross-domain inbound floor (pure tensor ops, no host sync)
    same = dom.unsqueeze(1) == dom.unsqueeze(0)
    in_dom_load = (q_seed.sum(dim=1) * same).sum(dim=0)
    floor = (load0 - in_dom_load).contiguous()

    # launch 2: per-domain tau-descent (one block per domain)
    cell = torch.zeros((E, R), dtype=torch.int64, device=dev)
    xbest = x.clone()
    max_propose = R * (n_slot + 1) + R
    _bisect_kernel[(M,)](
        flat_lam, x, cost, dom, order, cell, floor, slot_used, xbest,
        R, E, n_slot, int(cfg.tau_bisect_iters), max_propose,
        BLOCK_R=BLOCK_R, BLOCK_E=BLOCK_E,
    )

    # launch 3: exact final route over the committed placement
    q = torch.zeros((R, E, R), dtype=torch.int64, device=dev)
    load_out = torch.zeros(R, dtype=torch.int64, device=dev)
    _route_only_kernel[(1,)](
        flat_lam, x, cost, dom, order, q, load_out, R, E, BLOCK_R=BLOCK_R,
    )
    return Plan(x=x, q=q, tau=load_out.max())

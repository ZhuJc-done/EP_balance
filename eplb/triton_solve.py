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
    def _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, u_ptr,
               R, E, WRITE_Q: tl.constexpr, U_MIN: tl.constexpr,
               BLOCK_R: tl.constexpr):
        """Serial LPT route; optionally commit both source quotas and expert contributions."""
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
            if U_MIN > 1:
                # Reserve a legal floor on the least-loaded prefix, then
                # water-fill the remainder over that prefix.  If need<U_MIN,
                # C1 is preserved; such an input has no C5-feasible split.
                active_count = tl.sum(active.to(tl.int64))
                floor_k = tl.minimum(active_count, need // U_MIN)
                floor_active = active & (pos < floor_k)
                rem = need - floor_k * U_MIN
                base2 = base + U_MIN
                s_excl2 = tl.sum(before_i * base2[None, :], axis=1)
                cond2 = (floor_active & (pos >= 1)
                         & (base2 * pos - s_excl2 <= rem))
                k2 = 1 + tl.sum(cond2.to(tl.int64))
                level2 = tl.sum(tl.where(floor_active & (pos == (k2 - 1)),
                                         base2, 0))
                sum2 = tl.sum(tl.where(floor_active & (pos < k2), base2, 0))
                base_cost2 = level2 * k2 - sum2
                rem2 = rem - base_cost2
                share2 = rem2 // k2
                extra2 = rem2 - share2 * k2
                first2 = floor_active & (pos < k2)
                extra_add = tl.where(first2, (level2 - base2) + share2, zeros)
                extra_add = tl.where(first2 & (pos < extra2), extra_add + 1, extra_add)
                floor_add = tl.where(floor_active, U_MIN, zeros) + extra_add
                add = tl.where(need >= U_MIN, floor_add, add)
            if WRITE_Q:
                tl.store(q_ptr + (r * E + e) * R + lane64, add, mask=lane_mask)
                old_u = tl.load(u_ptr + e * R + lane64, mask=lane_mask, other=0)
                tl.store(u_ptr + e * R + lane64, old_u + add, mask=lane_mask)
            load = load + add
        return load

    @triton.jit
    def _safe_move(src_q, dst_q, remaining, u_min):
        """Largest source-local move within ``remaining`` that preserves the quota floor."""
        upper = tl.minimum(src_q, remaining)
        empty_donor = (upper == src_q) & (dst_q + upper >= u_min)
        partial_cap = tl.minimum(upper, tl.maximum(src_q - u_min, 0))
        min_partial = tl.where(dst_q > 0, 1, u_min)
        partial = tl.where(partial_cap >= min_partial, partial_cap, 0)
        return tl.where((upper > 0) & empty_donor, upper, tl.where(upper > 0, partial, 0))

    @triton.jit
    def _solve_kernel(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, u_ptr,
                      cand_e_ptr, cand_d_ptr, cand_valid_ptr, slot_used_ptr,
                      load_out_ptr,
                      R, E, M, EM, n_slot, allow_cd, max_iters,
                      U_MIN: tl.constexpr, BLOCK_R: tl.constexpr):
        """One program: Stage 1, one full route, then monotonic incremental Stage 2."""
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

        # The only full route.  It commits Q and U[e,dst] on device.
        load = _route(lam_ptr, x_ptr, cost_ptr, dom_ptr, order_ptr, q_ptr, u_ptr,
                      R, E, True, U_MIN, BLOCK_R)
        stuck = lane < 0  # all False

        TRUE_VEC = lane >= 0
        FALSE_VEC = lane < 0
        ZERO = tl.sum(tl.zeros([BLOCK_R], tl.int64))

        done = False
        for _it in range(max_iters):
            if not done:
                masked = tl.where(stuck, tl.full([BLOCK_R], -1, tl.int64), load)
                max_load = tl.max(tl.where(lane_mask, masked, -BIG))
                if max_load <= 0:
                    done = True
                else:
                    r_star = tl.min(tl.where(lane_mask & (masked == max_load), lane_i, BIG))
                    d_star = tl.load(dom_ptr + r_star)
                    cost_rstar = tl.load(cost_ptr + r_star * R + lane64,
                                         mask=lane_mask, other=BIG)

                    # Search every (expert, same-domain target).  Loop experts
                    # serially to avoid materialising an E x R register tensor;
                    # target ranks are compared in parallel across BLOCK_R.
                    best_delta = ZERO
                    best_load = BIG
                    best_cost = BIG
                    best_e = BIG
                    best_t = BIG
                    for e in range(E):
                        available = tl.load(u_ptr + e * R + r_star)
                        hosts = tl.load(x_ptr + e * R + lane64,
                                        mask=lane_mask, other=0) != 0
                        legal_slot = hosts | (slot_used < n_slot)
                        gap = max_load - load
                        delta = tl.minimum(available, gap // 2)
                        eligible = (lane_mask
                                    & (dom_all == d_star)
                                    & (load < max_load)
                                    & legal_slot
                                    & (delta >= U_MIN))
                        cand_delta = tl.where(eligible, delta, 0)
                        e_delta = tl.max(cand_delta)
                        d1 = eligible & (cand_delta == e_delta)
                        e_load = tl.min(tl.where(d1, load, BIG))
                        d2 = d1 & (load == e_load)
                        e_cost = tl.min(tl.where(d2, cost_rstar, BIG))
                        d3 = d2 & (cost_rstar == e_cost)
                        e_target = tl.min(tl.where(d3, lane_i, BIG))

                        better = ((e_delta > best_delta)
                                  | ((e_delta == best_delta) & (e_load < best_load))
                                  | ((e_delta == best_delta) & (e_load == best_load)
                                     & (e_cost < best_cost))
                                  | ((e_delta == best_delta) & (e_load == best_load)
                                     & (e_cost == best_cost) & (e < best_e))
                                  | ((e_delta == best_delta) & (e_load == best_load)
                                     & (e_cost == best_cost) & (e == best_e)
                                     & (e_target < best_t)))
                        better = better & (e_delta > 0)
                        best_delta = tl.where(better, e_delta, best_delta)
                        best_load = tl.where(better, e_load, best_load)
                        best_cost = tl.where(better, e_cost, best_cost)
                        best_e = tl.where(better, e, best_e)
                        best_t = tl.where(better, e_target, best_t)

                    if best_delta <= 0:
                        stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)
                    else:
                        # First pass computes the exact floor-safe transfer.
                        transfer_limit = best_delta
                        remaining = transfer_limit
                        actual = ZERO
                        for src in range(R):
                            from_off = (src * E + best_e) * R + r_star
                            to_off = (src * E + best_e) * R + best_t
                            src_q = tl.load(q_ptr + from_off)
                            dst_q = tl.load(q_ptr + to_off)
                            move = _safe_move(src_q, dst_q, remaining, U_MIN)
                            remaining -= move
                            actual += move

                        if actual < U_MIN:
                            # If a source quota straddles the half-gap, allow a
                            # whole-fragment move below the full gap.  This may
                            # reverse the pair's order, but strictly decreases
                            # its squared-load potential and cannot oscillate.
                            transfer_limit = max_load - best_load - 1
                            remaining = transfer_limit
                            actual = ZERO
                            for src in range(R):
                                from_off = (src * E + best_e) * R + r_star
                                to_off = (src * E + best_e) * R + best_t
                                src_q = tl.load(q_ptr + from_off)
                                dst_q = tl.load(q_ptr + to_off)
                                move = _safe_move(src_q, dst_q, remaining, U_MIN)
                                remaining -= move
                                actual += move

                        if actual < U_MIN:
                            stuck = tl.where(lane_i == r_star, TRUE_VEC, stuck)
                        else:
                            was_host = tl.load(x_ptr + best_e * R + best_t) != 0
                            is_new = not was_host
                            tl.store(x_ptr + best_e * R + best_t, 1, mask=is_new)
                            slot_used = tl.where(
                                lane_mask & (lane_i == best_t) & is_new,
                                slot_used + 1,
                                slot_used,
                            )

                            # Second pass commits exactly the transfer measured
                            # above, preserving C1 and the quota floor.
                            remaining = transfer_limit
                            for src in range(R):
                                from_off = (src * E + best_e) * R + r_star
                                to_off = (src * E + best_e) * R + best_t
                                src_q = tl.load(q_ptr + from_off)
                                dst_q = tl.load(q_ptr + to_off)
                                move = _safe_move(src_q, dst_q, remaining, U_MIN)
                                tl.store(q_ptr + from_off, src_q - move)
                                tl.store(q_ptr + to_off, dst_q + move)
                                remaining -= move

                            from_u = tl.load(u_ptr + best_e * R + r_star)
                            to_u = tl.load(u_ptr + best_e * R + best_t)
                            tl.store(u_ptr + best_e * R + r_star, from_u - actual)
                            tl.store(u_ptr + best_e * R + best_t, to_u + actual)
                            load = (load
                                    - tl.where(lane_i == r_star, actual, ZERO)
                                    + tl.where(lane_i == best_t, actual, ZERO))

                            # Only this destination domain changed.
                            stuck = tl.where(dom_all == d_star, FALSE_VEC, stuck)

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
    # Constructors such as ``from_nvlink_rdma`` carry the host-side domain count.
    # Custom CUDA-resident topologies fall back to the sync-free upper bound R.
    M = topo.sync_free_num_domains
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
    u = torch.zeros((E, R), dtype=torch.int64, device=dev)
    load_out = torch.zeros(R, dtype=torch.int64, device=dev)
    BLOCK_R = triton.next_power_of_2(max(R, 1))
    _solve_kernel[(1,)](
        flat_lam, x, cost, dom, order, q, u,
        cand_e, cand_d, cand_valid, slot_used, load_out,
        R, E, M, E * M, n_slot,
        1 if cfg.allow_cross_domain else 0, cfg.max_stage2_iters,
        U_MIN=int(cfg.u_min), BLOCK_R=BLOCK_R,
    )
    # fully sync-free: tau stays a 0-dim device tensor (tau == max committed rank load).
    # the hot dispatch path never reads tau; consumers needing a Python int coerce lazily via int(plan.tau).
    return Plan(x=x, q=q, tau=load_out.max())

"""Vendored FasterMoE dynamic-shadowing load balancer (PPoPP'22, Algorithm 1).

Adapted from the reference prototype at https://github.com/thu-pacman/FasterMoE
(``fmoe/transformer.py::_global_policy``).  Only the *load-balancing* decision is
reproduced here -- which hot experts to replicate ("shadow") on every worker so
their parameters, rather than their tokens, travel over the network.

The selection is expressed on Scale-EPLB's per-source load matrix ``Ω[r, e]``
(tokens sent from source rank ``r`` to expert ``e``).  Shadowing expert ``e`` means
every source rank keeps and computes its own ``Ω[r, e]`` tokens locally instead
of shipping them to ``e``'s home rank, which spreads a hot expert's load across all
ranks.  The greedy is guided by FasterMoE's performance model (Eqs. 7-8): an expert
is shadowed only while the predicted end-to-end latency keeps dropping, because each
extra shadow adds a fixed weight-broadcast (counted x2 for the backward gradient
reduce) whose benefit shrinks as the load flattens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ShadowCostModel:
    """FasterMoE performance-model constants (Eqs. 7-8 of the paper).

    Defaults mirror ``fmoe/transformer.py::_global_policy`` (the ``johnny`` V100 +
    50 Gb/s IB cluster).  The absolute units are irrelevant: only the *relative*
    balance between the compute and communication terms drives the decision.
    """

    bw_net: float = 50e9 / 8  # inter-worker bandwidth (bytes/s)
    bw_mm: float = 11.5e12  # GeMM throughput used as the compute rate
    grad_factor: float = 2.0  # weight broadcast (fwd) + gradient reduce (bwd)
    fwd_bwd_gemms: int = 3  # 1 forward + 2 backward GeMM rounds
    comm_rounds: int = 4  # token all-to-all rounds per iteration (fwd + bwd)


def select_shadow_experts(
    omega: torch.Tensor,
    main_rank: torch.Tensor,
    weight_bytes: torch.Tensor,
    s_tok: int,
    num_ranks: int,
    cost: ShadowCostModel | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the shadowed experts and return the resulting per-rank token load.

    Args:
        omega: int ``[R, E]`` load matrix, ``Ω[r, e]`` tokens from rank ``r``
            to expert ``e`` (this repository's :class:`~eplb.loads.Loads.omega`).
        main_rank: int ``[E]`` home rank ``main(e)`` of each expert.
        weight_bytes: numeric ``[E]`` byte size of each expert's parameters.
        s_tok: bytes of one token's activation vector (``H * dtype_size``).
        num_ranks: number of ranks/workers ``R``.
        cost: performance-model constants; defaults to :class:`ShadowCostModel`.

    Returns:
        ``(shadow_mask, rank_load)`` where ``shadow_mask`` is a bool ``[E]`` tensor
        (``True`` == replicated on every rank) and ``rank_load`` is a float ``[R]``
        tensor of realized per-rank token load after redistribution.
    """
    cost = cost or ShadowCostModel()

    # FasterMoE runs this policy on host-side integer counts; do the same on CPU.
    omega_f = omega.detach().to(device="cpu", dtype=torch.float64)
    home = main_rank.detach().to(device="cpu", dtype=torch.int64)
    w_bytes = weight_bytes.detach().to(device="cpu", dtype=torch.float64)
    num_experts = omega_f.shape[1]

    expert_load = omega_f.sum(dim=0)  # ω_e = sum_r Ω[r, e] -> [E]

    # Baseline: each expert's home rank aggregates the tokens from every source.
    rank_load = torch.zeros(num_ranks, dtype=torch.float64)
    rank_load.index_add_(0, home, expert_load)

    # Per-token compute cost 12 * alpha*H^2 / P.  With expert params ~= 2*alpha*H^2
    # elements, |W_e| bytes == 2 * (alpha*H^2 * dtype), so alpha*H^2*dtype == |W_e|/2.
    # Weights are uniform in the default harness, so the mean is exact there.
    alpha_h_sq = w_bytes / 2.0
    comp_per_token = cost.fwd_bwd_gemms * 4.0 * alpha_h_sq.mean().item() / cost.bw_mm

    b_max = rank_load.max().item()
    # Eq. 7: imbalanced compute + 4 rounds of token all-to-all (no shadowing).
    c_min = comp_per_token * b_max + cost.comm_rounds * b_max * s_tok / cost.bw_net

    order = torch.argsort(expert_load, descending=True).tolist()
    expert_load_l = expert_load.tolist()
    home_l = home.tolist()
    w_bytes_l = w_bytes.tolist()

    shadow_mask = torch.zeros(num_experts, dtype=torch.bool)
    broadcast_bytes = 0.0

    for e in order:
        h = home_l[e]
        # Redistribute: home keeps only its own tokens (Ω[h, e]); every source
        # rank now computes the Ω[r, e] tokens it used to ship (Alg. 1, l.6-8).
        rank_load[h] -= expert_load_l[e]
        rank_load += omega_f[:, e]

        broadcast_try = broadcast_bytes + w_bytes_l[e]
        # Eq. 8: shadowed compute + per-shadow weight broadcast (x2 for gradient).
        c = comp_per_token * rank_load.max().item() + cost.grad_factor * broadcast_try / cost.bw_net

        if c < c_min:
            c_min = c
            broadcast_bytes = broadcast_try
            shadow_mask[e] = True
        else:
            # Diminishing returns: this shadow no longer helps -> undo it and stop.
            rank_load -= omega_f[:, e]
            rank_load[h] += expert_load_l[e]
            break

    return shadow_mask, rank_load


__all__ = ["ShadowCostModel", "select_shadow_experts"]

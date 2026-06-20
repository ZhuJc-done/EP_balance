"""Synthetic load-matrix (``Lambda``) generators for experiments and tests.

Real MoE routing is heavily skewed (a few hot experts), which is exactly what
EPLB must fix. These generators produce integer ``[R, E]`` matrices with
controllable skew so the solver can be stressed without a real model.
"""

from __future__ import annotations

import torch

from eplb.loads import Loads


def make_loads(
    num_ranks: int,
    num_experts: int,
    tokens_per_rank: int,
    top_k: int = 1,
    skew: float = 0.0,
    hotspot_ranks: float = 1.0,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> Loads:
    """Generate a synthetic ``Lambda``.

    Args:
        num_ranks: ``R``.
        num_experts: ``E``.
        tokens_per_rank: Tokens emitted per source rank (before top_k).
        top_k: Experts activated per token (load is ``tokens_per_rank * top_k``).
        skew: 0.0 = uniform expert popularity; larger = more concentrated
            (Zipf-like exponent). Try 0.0, 1.0, 2.0.
        hotspot_ranks: Fraction of ranks (from rank 0) whose tokens all pile onto
            the hottest experts (simulates domain-local hotspots). 1.0 = all
            ranks share the same skewed distribution.
        seed: RNG seed (per-call deterministic).
        device: Tensor device.

    Returns:
        :class:`~eplb.loads.Loads` with an int64 ``[R, E]`` matrix.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # base popularity over experts (Zipf-ish): p_e ~ 1 / (rank_e + 1)^skew
    ranks_e = torch.arange(num_experts, dtype=torch.float64)
    base = 1.0 / torch.pow(ranks_e + 1.0, float(skew))
    base = base / base.sum()

    lam = torch.zeros((num_ranks, num_experts), dtype=torch.int64)
    total_per_rank = int(tokens_per_rank * top_k)
    n_hot = max(1, int(round(num_ranks * float(hotspot_ranks))))

    for r in range(num_ranks):
        if r < n_hot:
            probs = base
        else:
            # cooler ranks: flatten the distribution toward uniform
            probs = 0.5 * base + 0.5 * (torch.ones_like(base) / num_experts)
        # shuffle which expert ids are "hot" per rank a little, deterministically
        counts = torch.multinomial(
            probs, total_per_rank, replacement=True, generator=g
        )
        lam[r] = torch.bincount(counts, minlength=num_experts).to(torch.int64)

    return Loads(lam.to(device))

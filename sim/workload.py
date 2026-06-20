"""Synthetic, controllably-skewed load-matrix (``Lambda``) generators for experiments and tests."""

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
        skew: Zipf-like exponent; 0.0 = uniform popularity, larger = more concentrated.
        hotspot_ranks: Fraction of ranks (from rank 0) sharing the skewed distribution (1.0 = all).
        seed: RNG seed (per-call deterministic).
        device: Tensor device.

    Returns:
        :class:`~eplb.loads.Loads` with an int64 ``[R, E]`` matrix.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # base popularity p_e ~ 1 / (rank_e + 1)^skew
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
            # cooler ranks: flatten toward uniform
            probs = 0.5 * base + 0.5 * (torch.ones_like(base) / num_experts)
        counts = torch.multinomial(
            probs, total_per_rank, replacement=True, generator=g
        )
        lam[r] = torch.bincount(counts, minlength=num_experts).to(torch.int64)

    return Loads(lam.to(device))

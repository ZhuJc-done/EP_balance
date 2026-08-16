#!/usr/bin/env python3
"""Pick a ``ROUTER_SKEW`` value that produces a wanted routing imbalance.

``ROUTER_SKEW=<std>`` becomes Megatron's ``--moe-router-force-biased``, which builds the
router logits synthetically::

    logits[t, e] = eps[t, e] + b[e]      eps ~ N(0, 1) per (token, expert), per-rank seed
                                         b   ~ N(0, |std|) per expert, seed shared by all ranks

``std < 0`` draws ``b`` once per layer and reuses it for the whole run (stationary skew);
``std > 0`` redraws it every forward pass. Note that Megatron *replaces* the real logits
with noise rather than adding to them, so the loss is meaningless under this flag.

This reproduces that sampler offline so a target imbalance can be dialled in without
burning a training run. Two numbers come out, and the gap between them is the point:

- ``expert max/mean`` -- skew across logical experts, what the router produces.
- ``rank max/mean``   -- skew across receiving ranks under Megatron's contiguous
  ``main(e) = e // (E / R)`` placement, what an expert load balancer actually has to fix.

Because ``b`` is drawn independently per expert, hot experts land on unrelated ranks and
each rank averages ``E / R`` of them, so rank skew is heavily diluted relative to expert
skew. ``--bias-mode block`` correlates the bias inside each rank's expert block instead,
bounding how much a co-located hotspot pattern would add; Megatron cannot express it
today, so treat that column as motivation for a patch rather than something reachable.

Since ``b`` is a single draw per run, the spread across draws is as important as the
median: two runs with the same ``ROUTER_SKEW`` but different ``--seed`` land in different
places. Compare modes at a fixed seed so both see the identical ``b``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

DEFAULT_STDS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def draw_bias(std: float, num_experts: int, main_rank: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-expert logit bias, either i.i.d. (Megatron today) or correlated per rank block."""
    if mode == "iid":
        return torch.empty(num_experts).normal_(std=abs(std))
    if mode == "block":
        num_ranks = int(main_rank.max()) + 1
        per_rank = torch.empty(num_ranks).normal_(std=abs(std))
        return per_rank[main_rank]
    raise ValueError("bias mode must be 'iid' or 'block'")


def realise(
    std: float,
    *,
    num_experts: int,
    topk: int,
    main_rank: torch.Tensor,
    num_ranks: int,
    tokens: int,
    draws: int,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Sample the imbalance this ``std`` yields, once per independent bias draw."""
    expert_mm = torch.empty(draws)
    rank_mm = torch.empty(draws)
    for index in range(draws):
        bias = draw_bias(std, num_experts, main_rank, mode)
        chosen = (torch.randn(tokens, num_experts) + bias).topk(topk, dim=-1).indices
        counts = torch.bincount(chosen.reshape(-1), minlength=num_experts).to(torch.float64)
        rank_load = torch.zeros(num_ranks, dtype=torch.float64).index_add_(0, main_rank, counts)
        expert_mm[index] = counts.max() / counts.mean()
        rank_mm[index] = rank_load.max() / rank_load.mean()
    return {"expert_max_mean": expert_mm, "rank_max_mean": rank_mm}


def quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64)
    probs = torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype)
    p10, p50, p90 = torch.quantile(values, probs).tolist()
    return {
        "p10": round(p10, 4),
        "median": round(p50, 4),
        "p90": round(p90, 4),
        "max": round(float(values.max()), 4),
    }


def sweep(stds: tuple[float, ...], **kwargs: Any) -> list[dict[str, Any]]:
    rows = []
    for std in stds:
        got = realise(std, **kwargs)
        rows.append(
            {
                "std": std,
                "router_skew_stationary": -abs(std) if std else 0.0,
                "expert_max_mean": quantiles(got["expert_max_mean"]),
                "rank_max_mean": quantiles(got["rank_max_mean"]),
            }
        )
    return rows


def solve_for_target(target: float, tolerance: float, **kwargs: Any) -> dict[str, Any] | None:
    """Bisect |std| for a target median rank imbalance."""
    low, high = 0.0, 4.0
    at_high = float(torch.quantile(realise(high, **kwargs)["rank_max_mean"], 0.5))
    if at_high < target:
        return None
    best: dict[str, Any] | None = None
    for _ in range(18):
        mid = 0.5 * (low + high)
        got = realise(mid, **kwargs)
        median = float(torch.quantile(got["rank_max_mean"], 0.5))
        best = {
            "router_skew": round(-mid, 4),
            "rank_max_mean": quantiles(got["rank_max_mean"]),
            "expert_max_mean": quantiles(got["expert_max_mean"]),
        }
        if abs(median - target) <= tolerance:
            break
        low, high = (mid, high) if median < target else (low, mid)
    return best


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'std':>6}  {'ROUTER_SKEW':>11}  "
          f"{'rank max/mean':^33}  {'expert max/mean':^24}")
    print(f"{'':>6}  {'':>11}  "
          f"{'p10':>8}{'median':>9}{'p90':>8}{'max':>8}  {'median':>11}{'p90':>11}")
    for row in rows:
        rank, expert = row["rank_max_mean"], row["expert_max_mean"]
        print(
            f"{row['std']:6.2f}  {row['router_skew_stationary']:11.2f}  "
            f"{rank['p10']:8.3f}{rank['median']:9.3f}{rank['p90']:8.3f}{rank['max']:8.3f}  "
            f"{expert['median']:11.3f}{expert['p90']:11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experts", type=int, default=128, help="total routed experts E")
    parser.add_argument("--topk", type=int, default=8, help="router top-k")
    parser.add_argument("--ranks", type=int, default=16, help="expert-parallel world size R")
    parser.add_argument("--tokens", type=int, default=16384,
                        help="tokens sampled per bias draw (only sets estimator noise)")
    parser.add_argument("--draws", type=int, default=128,
                        help="independent bias draws; a real run gets exactly one of these")
    parser.add_argument("--bias-mode", choices=("iid", "block"), default="iid",
                        help="iid reproduces Megatron; block correlates bias within a rank")
    parser.add_argument("--std", type=float, nargs="+", default=list(DEFAULT_STDS),
                        help="magnitudes to sweep (pass the value as positive)")
    parser.add_argument("--target-rank-imbalance", type=float,
                        help="bisect for the ROUTER_SKEW giving this median rank max/mean")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.experts % args.ranks:
        raise SystemExit("experts must be divisible by ranks for Megatron's contiguous split")
    torch.manual_seed(args.seed)

    main_rank = torch.arange(args.experts, dtype=torch.int64) // (args.experts // args.ranks)
    shared = {
        "num_experts": args.experts,
        "topk": args.topk,
        "main_rank": main_rank,
        "num_ranks": args.ranks,
        "tokens": args.tokens,
        "draws": args.draws,
        "mode": args.bias_mode,
    }

    rows = sweep(tuple(args.std), **shared)
    print(f"E={args.experts} top-{args.topk} R={args.ranks} "
          f"bias-mode={args.bias_mode} draws={args.draws}\n")
    print_table(rows)
    print("\nstd=0 is the sampling floor: any imbalance below it is noise, not skew.")

    payload: dict[str, Any] = {"config": {k: v for k, v in vars(args).items()
                                          if k not in ("out_json",)}, "sweep": rows}
    payload["config"]["out_json"] = str(args.out_json) if args.out_json else None

    if args.target_rank_imbalance:
        found = solve_for_target(
            args.target_rank_imbalance, args.tolerance, **shared
        )
        payload["target"] = {
            "requested_rank_max_mean": args.target_rank_imbalance,
            "solution": found,
        }
        if found is None:
            print(f"\nno |std| <= 4.0 reaches rank max/mean {args.target_rank_imbalance}; "
                  f"with i.i.d. bias the rank-level skew saturates well below the expert-level "
                  f"skew — consider --bias-mode block.")
        else:
            print(f"\nfor median rank max/mean {args.target_rank_imbalance}: "
                  f"ROUTER_SKEW={found['router_skew']} "
                  f"(realised {found['rank_max_mean']['median']}, "
                  f"p10–p90 {found['rank_max_mean']['p10']}–{found['rank_max_mean']['p90']}, "
                  f"expert max/mean {found['expert_max_mean']['median']})")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

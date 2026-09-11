#!/usr/bin/env python3
"""Generate clearly marked synthetic routing traces for paper-layout prototyping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--raw-occurrences", type=int, default=128)
    parser.add_argument("--occurrence-group", type=int, default=8)
    parser.add_argument("--source-ranks", type=int, default=4)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--virtual-ranks", type=int, default=32)
    parser.add_argument("--tokens-per-source-rank", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    return parser.parse_args()


def _domain_trace(
    *,
    label: str,
    seed: int,
    layers: int,
    raw_occurrences: int,
    occurrence_group: int,
    source_ranks: int,
    experts: int,
    virtual_ranks: int,
    assignments_per_source_rank: int,
    temporal_strength: float,
    persistent_strength: float,
) -> dict:
    rng = np.random.default_rng(seed)
    experts_per_virtual_rank = experts // virtual_ranks

    expert_bias = rng.normal(0.0, 0.42, size=(layers, experts))
    persistent_rank_bias = rng.normal(
        0.0,
        persistent_strength,
        size=(layers, virtual_ranks),
    )
    layer_strength = np.clip(
        0.78
        + 0.20 * np.sin(np.linspace(0.0, 3.5 * np.pi, layers))
        + rng.normal(0.0, 0.09, size=layers),
        0.55,
        1.20,
    )
    # Keep the representative middle layer visually informative.
    layer_strength[min(23, layers - 1)] = 1.18

    groups = raw_occurrences // occurrence_group
    temporal_rank_bias = np.zeros((layers, groups, virtual_ranks), dtype=np.float64)
    for layer in range(layers):
        state = rng.normal(0.0, 0.08, size=virtual_ranks)
        for group in range(groups):
            state = 0.38 * state + rng.normal(0.0, 0.10, size=virtual_ranks)
            hot_ranks = rng.choice(virtual_ranks, size=3, replace=False)
            state[hot_ranks[0]] += temporal_strength
            state[hot_ranks[1]] += 0.72 * temporal_strength
            state[hot_ranks[2]] += 0.48 * temporal_strength
            temporal_rank_bias[layer, group] = state

    samples = []
    for occurrence in range(raw_occurrences):
        group = occurrence // occurrence_group
        for layer in range(layers):
            rank_logits = (
                persistent_rank_bias[layer]
                + layer_strength[layer] * temporal_rank_bias[layer, group]
            )
            logits = (
                expert_bias[layer]
                + np.repeat(rank_logits, experts_per_virtual_rank)
                + rng.normal(0.0, 0.055, size=experts)
            )
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()

            omega = np.stack(
                [
                    rng.multinomial(assignments_per_source_rank, probabilities)
                    for _ in range(source_ranks)
                ],
                axis=0,
            )
            samples.append(
                {
                    "layer": layer,
                    "mb": occurrence,
                    "ordinal": occurrence * layers + layer,
                    "omega": torch.from_numpy(omega).to(torch.int32),
                }
            )

    return {
        "meta": {
            "format_version": 3,
            "synthetic_placeholder": True,
            "warning": "Layout prototype only; do not report as a measured result.",
            "generator_seed": seed,
            "workload_label": label,
            "num_ranks": source_ranks,
            "num_experts": experts,
            "main_rank": torch.arange(experts, dtype=torch.int64)
            // (experts // source_ranks),
            "counts_reduced_over_tp_cp": True,
        },
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    if args.layers <= 0 or args.raw_occurrences <= 0:
        raise ValueError("--layers and --raw-occurrences must be positive")
    if args.occurrence_group <= 0 or args.raw_occurrences % args.occurrence_group:
        raise ValueError("--raw-occurrences must be divisible by --occurrence-group")
    if args.source_ranks <= 0 or args.experts % args.source_ranks:
        raise ValueError("--source-ranks must divide --experts")
    if args.virtual_ranks <= 0 or args.experts % args.virtual_ranks:
        raise ValueError("--virtual-ranks must divide --experts")
    if args.tokens_per_source_rank <= 0 or args.topk <= 0:
        raise ValueError("token and Top-k counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = args.tokens_per_source_rank * args.topk
    specifications = (
        ("dapo_math", args.seed, 0.92, 0.30),
        ("starcoder", args.seed + 1009, 0.78, 0.27),
    )
    for label, seed, temporal_strength, persistent_strength in specifications:
        trace = _domain_trace(
            label=label,
            seed=seed,
            layers=args.layers,
            raw_occurrences=args.raw_occurrences,
            occurrence_group=args.occurrence_group,
            source_ranks=args.source_ranks,
            experts=args.experts,
            virtual_ranks=args.virtual_ranks,
            assignments_per_source_rank=assignments,
            temporal_strength=temporal_strength,
            persistent_strength=persistent_strength,
        )
        output = args.output_dir / f"{label}_synthetic_placeholder.pt"
        torch.save(trace, output)
        print(f"[placeholder] SYNTHETIC, NOT MEASURED: {output}")


if __name__ == "__main__":
    main()

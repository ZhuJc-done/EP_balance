"""CLI smoke test for the layer-by-batch expert max/mean heatmap."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_plot_expert_max_mean_cli(tmp_path):
    trace_path = tmp_path / "trace.pt"
    image_path = tmp_path / "max_mean.png"
    torch.save(
        {
            "meta": {
                "format_version": 3,
                "num_ranks": 2,
                "num_experts": 4,
                "counts_reduced_over_tp_cp": True,
            },
            "samples": [
                {
                    "layer": layer,
                    "mb": occurrence,
                    "ordinal": occurrence * 2 + layer,
                    "omega": torch.tensor(
                        [[5 + occurrence, 1, 0, 0], [0, 0, 2, 2 + layer]]
                    ),
                }
                for occurrence in range(2)
                for layer in range(2)
            ],
        },
        trace_path,
    )

    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "eval/plot_expert_max_mean.py",
            "--trace",
            f"Demo={trace_path}",
            "--output",
            str(image_path),
        ],
        cwd=repository,
        check=True,
    )

    assert image_path.stat().st_size > 0
    with image_path.with_suffix(".csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    first = next(
        row for row in rows if row["layer"] == "0" and row["occurrence"] == "0"
    )
    assert first["max_tokens"] == "5"
    assert float(first["mean_tokens"]) == pytest.approx(2.5)
    assert float(first["max_mean"]) == pytest.approx(2.0)
    assert first["hot_expert"] == "0"

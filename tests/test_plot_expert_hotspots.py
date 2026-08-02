"""CLI smoke test for the publication-style hotspot plot."""

import csv
import subprocess
import sys
from pathlib import Path

import torch


def test_plot_expert_hotspots_cli(tmp_path):
    trace_path = tmp_path / "trace.pt"
    image_path = tmp_path / "hotspots.png"
    torch.save(
        {
            "meta": {
                "format_version": 3,
                "num_ranks": 2,
                "num_experts": 4,
                "main_rank": torch.tensor([0, 0, 1, 1]),
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
            "eval/plot_expert_hotspots.py",
            "--trace",
            f"Demo={trace_path}",
            "--view",
            "both",
            "--output",
            str(image_path),
        ],
        cwd=repository,
        check=True,
    )

    assert image_path.stat().st_size > 0
    csv_path = image_path.with_suffix(".csv")
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert {row["view"] for row in rows} == {"snapshot", "aggregate"}

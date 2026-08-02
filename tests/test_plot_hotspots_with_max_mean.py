"""CLI smoke test for the combined hotspot and layer-line figure."""

import subprocess
import sys
from pathlib import Path

import torch


def test_plot_hotspots_with_max_mean_cli(tmp_path):
    trace_path = tmp_path / "trace.pt"
    image_path = tmp_path / "combined.pdf"
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
            "eval/plot_hotspots_with_max_mean.py",
            "--trace",
            f"Math={trace_path}",
            "--trace",
            f"Code={trace_path}",
            "--output",
            str(image_path),
        ],
        cwd=repository,
        check=True,
    )

    assert image_path.stat().st_size > 0
    assert image_path.read_bytes().startswith(b"%PDF")

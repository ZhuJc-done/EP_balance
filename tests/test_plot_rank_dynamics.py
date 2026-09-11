"""CLI smoke test for the four-panel rank-dynamics figure."""

import subprocess
import sys
from pathlib import Path

import torch


def test_plot_rank_dynamics_cli(tmp_path):
    traces = []
    for trace_index in range(2):
        trace_path = tmp_path / f"trace_{trace_index}.pt"
        samples = []
        for occurrence in range(3):
            for layer in range(2):
                omega = torch.zeros((4, 8), dtype=torch.int64)
                for source_rank in range(4):
                    omega[source_rank, (source_rank + occurrence + trace_index) % 8] += 4
                    omega[source_rank, (layer + 2 * source_rank) % 8] += 2
                samples.append(
                    {
                        "layer": layer,
                        "mb": occurrence,
                        "ordinal": occurrence * 2 + layer,
                        "omega": omega,
                    }
                )
        torch.save(
            {
                "meta": {
                    "format_version": 3,
                    "num_ranks": 4,
                    "num_experts": 8,
                    "main_rank": torch.arange(8) // 2,
                    "counts_reduced_over_tp_cp": True,
                },
                "samples": samples,
            },
            trace_path,
        )
        traces.append(trace_path)

    image_path = tmp_path / "rank_dynamics.pdf"
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "eval/plot_rank_dynamics.py",
            "--trace",
            f"Math={traces[0]}",
            "--trace",
            f"Code={traces[1]}",
            "--output",
            str(image_path),
        ],
        cwd=repository,
        check=True,
    )

    assert image_path.stat().st_size > 0
    assert image_path.read_bytes().startswith(b"%PDF")
    assert image_path.with_suffix(".csv").is_file()

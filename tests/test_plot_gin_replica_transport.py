"""CLI smoke test for the GIN replica transport schematic."""

import subprocess
import sys
from pathlib import Path


def test_plot_gin_replica_transport_cli(tmp_path):
    image_path = tmp_path / "gin.pdf"

    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "eval/plot_gin_replica_transport.py",
            "--output",
            str(image_path),
        ],
        cwd=repository,
        check=True,
    )

    assert image_path.stat().st_size > 0
    assert image_path.read_bytes().startswith(b"%PDF")

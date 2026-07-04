from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_continuous_smoke() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmao.cli",
            "continuous",
            "--problem",
            "sphere",
            "--dimension",
            "3",
            "--iterations",
            "5",
            "--seed",
            "1",
            "--summary-only",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "best_fitness" in payload


def test_cli_output_file_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "sphere.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mmao.cli",
            "continuous",
            "--problem",
            "sphere",
            "--dimension",
            "3",
            "--iterations",
            "5",
            "--seed",
            "1",
            "--output",
            str(output),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "history" in payload


def test_cli_bundled_benchmarks_smoke() -> None:
    repo = Path(__file__).resolve().parents[1]
    tsp_result = subprocess.run(
        [sys.executable, "-m", "mmao.cli", "tsp", "--benchmark", "berlin52", "--iterations", "10", "--summary-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    tsp_payload = json.loads(tsp_result.stdout)
    assert tsp_payload["problem"].lower() == "berlin52"

    mkp_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmao.cli",
            "knapsack",
            "--benchmark",
            "mknap2",
            "--instance-index",
            "0",
            "--iterations",
            "4",
            "--summary-only",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    mkp_payload = json.loads(mkp_result.stdout)
    assert mkp_payload["problem"].startswith("ORL-MKP-")

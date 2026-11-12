# Copyright (c) 2026 Jinliang Xu
# Author: Jinliang Xu
# Email: jlxufly@gmail.com
# Email: xujinliang@caict.ac.cn
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    assert payload["known_optimum"] == 0
    assert payload["metadata"]["seed"] == 1
    assert payload["nfev"] > 0


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


def test_cli_classification_smoke() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmao.cli",
            "classification",
            "--dataset",
            "breast_cancer",
            "--classifier",
            "svm_rbf",
            "--iterations",
            "4",
            "--summary-only",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "test_metric" in payload

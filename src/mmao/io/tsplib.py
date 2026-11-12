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

from importlib.resources import files
from pathlib import Path

from ..discrete.tsp import TSPProblem


def _resolve_tsplib_path(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.exists():
        return file_path

    normalized = str(path).strip().lower().replace("\\", "/")
    if normalized in {"berlin52", "berlin52.tsp"}:
        return Path(str(files("mmao.data.tsplib").joinpath("berlin52.tsp")))

    raise FileNotFoundError(
        f"TSPLIB file not found: {path}. Provide a real local .tsp file path, use "
        "'load_bundled_tsplib_problem(\"berlin52\")' in Python, or use "
        "'mmao tsp --benchmark berlin52 --summary-only' for the bundled example."
    )


def load_tsplib_problem(path: str | Path, *, known_optimum: float | None = None) -> TSPProblem:
    file_path = _resolve_tsplib_path(path)
    if str(path).strip().lower() in {"berlin52", "berlin52.tsp"} and not Path(path).exists():
        from .bundled import load_bundled_tsplib_problem
        from dataclasses import replace
        bundled = load_bundled_tsplib_problem("berlin52")
        return bundled if known_optimum is None else replace(bundled, known_optimum=known_optimum)
    try:
        import tsplib95
    except ImportError as exc:
        raise ImportError("tsplib95 is required to load TSPLIB files. Install mmao-opt[io].") from exc

    problem = tsplib95.load(str(file_path))
    nodes = sorted(problem.get_nodes())
    matrix: list[list[float]] = []
    for i in nodes:
        row: list[float] = []
        for j in nodes:
            row.append(float(problem.get_weight(i, j)))
        matrix.append(row)

    coordinates = None
    if getattr(problem, "node_coords", None) and all(len(problem.node_coords[node]) == 2 for node in nodes):
        coordinates = [(float(problem.node_coords[node][0]), float(problem.node_coords[node][1])) for node in nodes]

    return TSPProblem(
        name=str(problem.name or str(file_path)),
        distance_matrix=matrix,
        coordinates=coordinates,
        known_optimum=known_optimum,
    )

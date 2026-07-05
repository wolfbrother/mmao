from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..discrete.tsp import TSPProblem


def _resolve_tsplib_path(path: str) -> Path:
    file_path = Path(path)
    if file_path.exists():
        return file_path

    normalized = path.strip().lower().replace("\\", "/")
    if normalized in {"berlin52", "berlin52.tsp"}:
        return Path(str(files("mmao.data.tsplib").joinpath("berlin52.tsp")))

    raise FileNotFoundError(
        f"TSPLIB file not found: {path}. Provide a real local .tsp file path, use "
        "'load_bundled_tsplib_problem(\"berlin52\")' in Python, or use "
        "'mmao tsp --benchmark berlin52 --summary-only' for the bundled example."
    )


def load_tsplib_problem(path: str) -> TSPProblem:
    try:
        import tsplib95
    except ImportError as exc:
        raise ImportError("tsplib95 is required to load TSPLIB files. Install mmao-opt[io].") from exc

    file_path = _resolve_tsplib_path(path)

    problem = tsplib95.load(str(file_path))
    nodes = sorted(problem.get_nodes())
    matrix: list[list[float]] = []
    for i in nodes:
        row: list[float] = []
        for j in nodes:
            row.append(float(problem.get_weight(i, j)))
        matrix.append(row)

    coordinates = None
    if getattr(problem, "node_coords", None):
        coordinates = []
        for node in nodes:
            x, y = problem.node_coords[node]
            coordinates.append((float(x), float(y)))

    known_optimum = None
    if getattr(problem, "tours", None):
        try:
            best_tour = problem.tours[0]
            known_optimum = float(sum(problem.get_weight(best_tour[idx], best_tour[(idx + 1) % len(best_tour)]) for idx in range(len(best_tour))))
        except Exception:
            known_optimum = None

    return TSPProblem(
        name=str(problem.name or str(file_path)),
        distance_matrix=matrix,
        coordinates=coordinates,
        known_optimum=known_optimum,
    )

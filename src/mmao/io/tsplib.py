from __future__ import annotations

from ..discrete.tsp import TSPProblem


def load_tsplib_problem(path: str) -> TSPProblem:
    try:
        import tsplib95
    except ImportError as exc:
        raise ImportError("tsplib95 is required to load TSPLIB files. Install mmao-opt[io].") from exc

    problem = tsplib95.load(path)
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
        name=str(problem.name or path),
        distance_matrix=matrix,
        coordinates=coordinates,
        known_optimum=known_optimum,
    )

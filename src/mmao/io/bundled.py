from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..discrete.tsp import TSPProblem
from .orlib import load_orlib_mkp_instance
from .tsplib import load_tsplib_problem


def bundled_tsplib_path(name: str = "berlin52") -> Path:
    normalized = name.lower()
    if normalized != "berlin52":
        raise ValueError(f"Unsupported bundled TSPLIB benchmark: {name}")
    return Path(str(files("mmao.data.tsplib").joinpath("berlin52.tsp")))


def bundled_orlib_path(name: str = "mknap2") -> Path:
    normalized = name.lower()
    if normalized != "mknap2":
        raise ValueError(f"Unsupported bundled OR-Library benchmark: {name}")
    return Path(str(files("mmao.data.orlib").joinpath("mknap2.txt")))


def load_bundled_tsplib_problem(name: str = "berlin52"):
    normalized = name.lower()
    if normalized == "berlin52":
        path = bundled_tsplib_path(name)
        lines = path.read_text(encoding="utf-8").splitlines()
        coordinates: list[tuple[float, float]] = []
        reading_coords = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper() == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            if line.upper() == "EOF":
                break
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    coordinates.append((float(parts[1]), float(parts[2])))
        matrix: list[list[float]] = []
        for x1, y1 in coordinates:
            row: list[float] = []
            for x2, y2 in coordinates:
                row.append(round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
            matrix.append(row)
        return TSPProblem(
            name="berlin52",
            distance_matrix=matrix,
            coordinates=coordinates,
            known_optimum=7542.0,
        )
    return load_tsplib_problem(str(bundled_tsplib_path(name)))


def load_bundled_orlib_mkp_instance(name: str = "mknap2", index: int = 0):
    return load_orlib_mkp_instance(str(bundled_orlib_path(name)), index=index)

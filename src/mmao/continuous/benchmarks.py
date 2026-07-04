from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


Vector = list[float]


@dataclass(frozen=True)
class ContinuousProblem:
    name: str
    dimension: int
    bounds: list[tuple[float, float]]
    objective: Callable[[Vector], float]
    optimum_value: float = 0.0

    def evaluate(self, position: Vector) -> float:
        return float(self.objective(position))


def sphere(vector: Vector) -> float:
    return sum(value * value for value in vector)


def rastrigin(vector: Vector) -> float:
    return 10.0 * len(vector) + sum(
        value * value - 10.0 * math.cos(2.0 * math.pi * value) for value in vector
    )


def sphere_problem(dimension: int = 10, lower: float = -5.0, upper: float = 5.0) -> ContinuousProblem:
    bounds = [(lower, upper) for _ in range(dimension)]
    return ContinuousProblem(name=f"sphere-{dimension}d", dimension=dimension, bounds=bounds, objective=sphere)


def rastrigin_problem(dimension: int = 10, lower: float = -5.12, upper: float = 5.12) -> ContinuousProblem:
    bounds = [(lower, upper) for _ in range(dimension)]
    return ContinuousProblem(name=f"rastrigin-{dimension}d", dimension=dimension, bounds=bounds, objective=rastrigin)

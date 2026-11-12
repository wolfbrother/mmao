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

from dataclasses import dataclass
import math
from typing import Callable
from ..common.validation import finite_number, integer, validate_bounds


Vector = list[float]


@dataclass(frozen=True)
class ContinuousProblem:
    name: str
    dimension: int
    bounds: list[tuple[float, float]]
    objective: Callable[[Vector], float]
    optimum_value: float | None = None

    def __post_init__(self) -> None:
        validate_bounds(self.dimension, self.bounds)
        if not callable(self.objective):
            raise ValueError("objective must be callable.")
        if self.optimum_value is not None:
            finite_number("optimum_value", self.optimum_value)

    def evaluate(self, position: Vector) -> float:
        return float(self.objective(position))


def sphere(vector: Vector) -> float:
    return sum(value * value for value in vector)


def rastrigin(vector: Vector) -> float:
    return 10.0 * len(vector) + sum(
        value * value - 10.0 * math.cos(2.0 * math.pi * value) for value in vector
    )


def sphere_problem(dimension: int = 10, lower: float = -5.0, upper: float = 5.0) -> ContinuousProblem:
    integer("dimension", dimension)
    bounds = [(lower, upper) for _ in range(dimension)]
    optimum = dimension * min(upper, max(lower, 0.0)) ** 2
    return ContinuousProblem(name=f"sphere-{dimension}d", dimension=dimension, bounds=bounds, objective=sphere, optimum_value=optimum)


def rastrigin_problem(dimension: int = 10, lower: float = -5.12, upper: float = 5.12) -> ContinuousProblem:
    integer("dimension", dimension)
    bounds = [(lower, upper) for _ in range(dimension)]
    optimum = 0.0 if lower <= 0 <= upper else None
    return ContinuousProblem(name=f"rastrigin-{dimension}d", dimension=dimension, bounds=bounds, objective=rastrigin, optimum_value=optimum)

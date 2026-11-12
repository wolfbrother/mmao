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
from ..common.validation import finite_number, integer


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    dimension: int
    change_frequency: int
    change_severity: float
    shift_scale: float

    def __post_init__(self) -> None:
        integer("dimension", self.dimension)
        integer("change_frequency", self.change_frequency)
        if self.name not in {"dyn-sphere", "dyn-rastrigin", "dyn-ackley"}:
            raise ValueError(f"Unsupported dynamic objective: {self.name}")
        for name in ("change_severity", "shift_scale"):
            value = getattr(self, name)
            finite_number(name, value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")


def make_dynamic_problem(
    name: str = "dyn-sphere",
    dimension: int = 10,
    change_frequency: int = 25,
    change_severity: float = 1.0,
) -> tuple[DynamicScenario, list[tuple[float, float]]]:
    bounds = [(-5.0, 5.0) for _ in range(dimension)]
    shift_scale = 0.55 * change_severity
    return (
        DynamicScenario(
            name=name,
            dimension=dimension,
            change_frequency=change_frequency,
            change_severity=change_severity,
            shift_scale=shift_scale,
        ),
        bounds,
    )

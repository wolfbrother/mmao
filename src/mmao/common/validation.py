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

"""Validation shared by public APIs and direct optimizer entry points."""

import math
from dataclasses import fields
from numbers import Integral, Real


def integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def finite_number(name: str, value: Real) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def validate_config(config, *, minimum_iterations: int = 0) -> None:
    integer("iterations", config.iterations, minimum_iterations)
    integer("seed", config.seed, 0)
    for name in ("minimum_agents", "initial_agents", "maximum_agents"):
        integer(name, getattr(config, name))
    if not config.minimum_agents <= config.initial_agents <= config.maximum_agents:
        raise ValueError("Require minimum_agents <= initial_agents <= maximum_agents.")
    count_minimums = {
        "progress_window": 1, "branch_children": 0, "respawn_floor": 0,
        "local_search_trials": 1, "kick_strength": 1, "recovery_window": 1,
        "bitflip_k_min": 1, "bitflip_k_base": 0,
    }
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name in {"return_model", "use_guided_memory"}:
            if not isinstance(value, bool):
                raise ValueError(f"{field.name} must be a boolean.")
            continue
        if value is None and field.name in {"fixed_role", "max_evaluations"}:
            continue
        finite_number(field.name, value)
        if field.name in count_minimums:
            integer(field.name, value, count_minimums[field.name])
        if value < 0:
            raise ValueError(f"{field.name} must be nonnegative.")
        if (field.name.endswith(("share", "rate", "probability")) or field.name == "fixed_role") and not 0 <= value <= 1:
            raise ValueError(f"{field.name} must be between 0 and 1.")
    if hasattr(config, "max_energy") and not 0 < config.initial_energy <= config.max_energy:
        raise ValueError("Require 0 < initial_energy <= max_energy.")


def validate_bounds(dimension: int, bounds) -> None:
    integer("dimension", dimension)
    if len(bounds) != dimension:
        raise ValueError("bounds must contain one pair per dimension.")
    for pair in bounds:
        if len(pair) != 2:
            raise ValueError("Each bound must contain finite lower < upper values.")
        for value in pair:
            finite_number("bound", value)
        if pair[0] >= pair[1]:
            raise ValueError("Each bound must contain finite lower < upper values.")

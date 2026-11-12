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

from dataclasses import asdict
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
import sklearn
from .version import __version__
from .results import to_jsonable

from .classification.core import ClassificationProblem, MMAOClassificationConfig, optimize_classification_problem
from .continuous.benchmarks import ContinuousProblem
from .continuous.core import MMAOContinuousConfig, optimize_continuous_problem
from .discrete.knapsack import KnapsackProblem, MMAOKnapsackConfig, optimize_knapsack_problem
from .discrete.tsp import TSPProblem, MMAOTSPConfig, optimize_tsp_problem
from .dynamic.core import DynamicOptimizationProblem, MMAODynamicConfig, optimize_dynamic_problem


def _attach_known_optimum(result: dict[str, object], known_optimum: float | int | None, value_key: str, direction: str = "minimize") -> dict[str, object]:
    result["optimization_direction"] = direction
    if known_optimum is None or value_key not in result:
        return result
    observed = float(result[value_key])  # type: ignore[arg-type]
    optimum = float(known_optimum)
    gap = optimum - observed if direction == "maximize" else observed - optimum
    denominator = max(abs(optimum), 1e-12)
    result["known_optimum"] = optimum
    result["gap"] = gap
    result["relative_gap"] = gap / denominator
    return result


def _run(optimizer, problem, config) -> dict[str, object]:
    started = perf_counter()
    result = optimizer(problem, config)
    elapsed = perf_counter() - started
    description = {"name": problem.name, "type": type(problem).__name__}
    for key in ("dimension", "bounds", "distance_matrix", "profits", "weights", "capacities", "dataset_name", "classifier_name", "metric", "known_optimum", "optimum_value"):
        if hasattr(problem, key):
            description[key] = getattr(problem, key)
    if isinstance(problem, DynamicOptimizationProblem):
        description["scenario"] = asdict(problem.scenario)
    if isinstance(problem, ClassificationProblem):
        x, y = problem.load()
        description["data_sha256"] = hashlib.sha256(
            np.ascontiguousarray(x).tobytes() + json.dumps(y.tolist()).encode("utf-8")
        ).hexdigest()
        description["data_shape"] = list(x.shape)
        description["data_dtype"] = str(x.dtype)
    fingerprint = hashlib.sha256(json.dumps(to_jsonable(description), sort_keys=True).encode("utf-8")).hexdigest()
    result["metadata"] = {
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "config": asdict(config),
        "seed": config.seed,
        "elapsed_seconds": elapsed,
        "problem_fingerprint": fingerprint,
    }
    return result


def optimize_continuous(
    problem: ContinuousProblem,
    config: MMAOContinuousConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        _run(optimize_continuous_problem, problem, config or MMAOContinuousConfig()),
        problem.optimum_value,
        "best_fitness",
    )


def optimize_tsp(
    problem: TSPProblem,
    config: MMAOTSPConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        _run(optimize_tsp_problem, problem, config or MMAOTSPConfig()),
        problem.known_optimum,
        "best_distance",
    )


def optimize_knapsack(
    problem: KnapsackProblem,
    config: MMAOKnapsackConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        _run(optimize_knapsack_problem, problem, config or MMAOKnapsackConfig()),
        problem.known_optimum,
        "best_profit",
        "maximize",
    )


def optimize_dynamic(
    problem: DynamicOptimizationProblem,
    config: MMAODynamicConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        _run(optimize_dynamic_problem, problem, config or MMAODynamicConfig()),
        problem.optimum_value,
        "best_fitness",
    )


def optimize_classification(
    problem: ClassificationProblem,
    config: MMAOClassificationConfig | None = None,
) -> dict[str, object]:
    result = _run(optimize_classification_problem, problem, config or MMAOClassificationConfig())
    result["optimization_direction"] = "maximize"
    return result

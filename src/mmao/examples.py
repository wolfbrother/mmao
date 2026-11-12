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

import math

from .classification import ClassificationProblem, MMAOClassificationConfig, optimize_classification_problem
from .continuous import MMAOContinuousConfig, optimize_continuous_problem, rastrigin_problem
from .discrete import KnapsackProblem, MMAOKnapsackConfig, MMAOTSPConfig, TSPProblem, optimize_knapsack_problem, optimize_tsp_problem
from .dynamic import DynamicOptimizationProblem, MMAODynamicConfig, make_dynamic_problem, optimize_dynamic_problem


def demo_tsp_problem() -> TSPProblem:
    points = [
        (0.0, 0.0),
        (1.2, 3.8),
        (2.5, 1.1),
        (4.7, 4.2),
        (5.8, 0.5),
        (7.0, 3.5),
        (8.3, 1.2),
        (6.1, 6.4),
        (3.4, 6.7),
        (0.9, 6.1),
    ]
    matrix = []
    for x1, y1 in points:
        row = []
        for x2, y2 in points:
            row.append(math.hypot(x2 - x1, y2 - y1))
        matrix.append(row)
    return TSPProblem(name="demo-tsp-10", distance_matrix=matrix, coordinates=points)


def demo_knapsack_problem() -> KnapsackProblem:
    profits = [24, 13, 23, 15, 16, 28, 12, 20]
    weights = [
        [12, 7, 11, 8, 9, 13, 6, 10],
        [8, 5, 7, 6, 6, 9, 4, 7],
    ]
    capacities = [35, 23]
    return KnapsackProblem(name="demo-mkp-8", profits=profits, weights=weights, capacities=capacities)


def run_continuous_demo() -> dict[str, object]:
    problem = rastrigin_problem(dimension=10)
    return optimize_continuous_problem(problem, MMAOContinuousConfig(iterations=80, seed=7))


def run_tsp_demo() -> dict[str, object]:
    return optimize_tsp_problem(demo_tsp_problem(), MMAOTSPConfig(iterations=100, seed=11))


def run_knapsack_demo() -> dict[str, object]:
    return optimize_knapsack_problem(demo_knapsack_problem(), MMAOKnapsackConfig(iterations=60, seed=23))


def run_dynamic_demo() -> dict[str, object]:
    scenario, bounds = make_dynamic_problem(name="dyn-rastrigin", dimension=10, change_frequency=20, change_severity=1.0)
    problem = DynamicOptimizationProblem(name=scenario.name, scenario=scenario, bounds=bounds)
    return optimize_dynamic_problem(problem, MMAODynamicConfig(iterations=80, seed=7))


def run_classification_demo() -> dict[str, object]:
    problem = ClassificationProblem(name="breast-cancer-svm", dataset_name="breast_cancer", classifier_name="svm_rbf")
    return optimize_classification_problem(problem, MMAOClassificationConfig(iterations=12, seed=3))

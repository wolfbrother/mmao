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

"""MMAO: a metabolic multi-agent optimization framework."""

from .api import (
    optimize_classification,
    optimize_continuous,
    optimize_dynamic,
    optimize_knapsack,
    optimize_tsp,
)
from .classification import ClassificationProblem, MMAOClassificationConfig, load_builtin_dataset
from .continuous import ContinuousProblem, MMAOContinuousConfig, rastrigin_problem, sphere_problem
from .discrete import KnapsackProblem, MMAOKnapsackConfig, MMAOTSPConfig, TSPProblem
from .dynamic import DynamicOptimizationProblem, MMAODynamicConfig, make_dynamic_problem
from .io import load_orlib_mkp_instance, load_orlib_mkp_instances, load_tsplib_problem
from .io import (
    bundled_orlib_path,
    bundled_tsplib_path,
    load_bundled_orlib_mkp_instance,
    load_bundled_tsplib_problem,
)
from .results import save_classification_model, summarize_result, write_result_json
from .version import __version__

__all__ = [
    "__version__",
    "bundled_orlib_path",
    "bundled_tsplib_path",
    "ClassificationProblem",
    "ContinuousProblem",
    "DynamicOptimizationProblem",
    "KnapsackProblem",
    "MMAOClassificationConfig",
    "MMAOContinuousConfig",
    "MMAODynamicConfig",
    "MMAOKnapsackConfig",
    "MMAOTSPConfig",
    "TSPProblem",
    "load_builtin_dataset",
    "load_bundled_orlib_mkp_instance",
    "load_bundled_tsplib_problem",
    "load_orlib_mkp_instance",
    "load_orlib_mkp_instances",
    "load_tsplib_problem",
    "make_dynamic_problem",
    "optimize_continuous",
    "optimize_tsp",
    "optimize_knapsack",
    "optimize_dynamic",
    "optimize_classification",
    "rastrigin_problem",
    "sphere_problem",
    "summarize_result",
    "save_classification_model",
    "write_result_json",
]

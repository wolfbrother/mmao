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
from .results import summarize_result, write_result_json
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
    "write_result_json",
]

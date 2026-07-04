from __future__ import annotations

from .classification.core import ClassificationProblem, MMAOClassificationConfig, optimize_classification_problem
from .continuous.benchmarks import ContinuousProblem
from .continuous.core import MMAOContinuousConfig, optimize_continuous_problem
from .discrete.knapsack import KnapsackProblem, MMAOKnapsackConfig, optimize_knapsack_problem
from .discrete.tsp import TSPProblem, MMAOTSPConfig, optimize_tsp_problem
from .dynamic.core import DynamicOptimizationProblem, MMAODynamicConfig, optimize_dynamic_problem


def _attach_known_optimum(result: dict[str, object], known_optimum: float | int | None, value_key: str) -> dict[str, object]:
    if known_optimum is None or value_key not in result:
        return result
    observed = float(result[value_key])  # type: ignore[arg-type]
    optimum = float(known_optimum)
    gap = observed - optimum
    denominator = max(abs(optimum), 1e-12)
    result["known_optimum"] = optimum
    result["gap"] = gap
    result["relative_gap"] = gap / denominator
    return result


def optimize_continuous(
    problem: ContinuousProblem,
    config: MMAOContinuousConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        optimize_continuous_problem(problem, config or MMAOContinuousConfig()),
        problem.optimum_value,
        "best_fitness",
    )


def optimize_tsp(
    problem: TSPProblem,
    config: MMAOTSPConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        optimize_tsp_problem(problem, config or MMAOTSPConfig()),
        problem.known_optimum,
        "best_distance",
    )


def optimize_knapsack(
    problem: KnapsackProblem,
    config: MMAOKnapsackConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        optimize_knapsack_problem(problem, config or MMAOKnapsackConfig()),
        problem.known_optimum,
        "best_profit",
    )


def optimize_dynamic(
    problem: DynamicOptimizationProblem,
    config: MMAODynamicConfig | None = None,
) -> dict[str, object]:
    return _attach_known_optimum(
        optimize_dynamic_problem(problem, config or MMAODynamicConfig()),
        problem.optimum_value,
        "best_fitness",
    )


def optimize_classification(
    problem: ClassificationProblem,
    config: MMAOClassificationConfig | None = None,
) -> dict[str, object]:
    return optimize_classification_problem(problem, config or MMAOClassificationConfig())

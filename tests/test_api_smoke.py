from __future__ import annotations

from mmao import (
    optimize_classification,
    optimize_continuous,
    optimize_dynamic,
    optimize_knapsack,
    optimize_tsp,
)
from mmao.classification import ClassificationProblem, MMAOClassificationConfig
from mmao.continuous import MMAOContinuousConfig, rastrigin_problem
from mmao.dynamic import DynamicOptimizationProblem, MMAODynamicConfig, make_dynamic_problem
from mmao.examples import demo_knapsack_problem, demo_tsp_problem


def test_continuous_smoke() -> None:
    result = optimize_continuous(rastrigin_problem(dimension=5), MMAOContinuousConfig(iterations=15, seed=1))
    assert "best_fitness" in result
    assert len(result["history"]) == 16


def test_tsp_smoke() -> None:
    result = optimize_tsp(demo_tsp_problem())
    assert result["best_distance"] > 0.0
    assert len(result["best_route"]) == demo_tsp_problem().city_count


def test_knapsack_smoke() -> None:
    result = optimize_knapsack(demo_knapsack_problem())
    assert result["best_profit"] > 0


def test_dynamic_smoke() -> None:
    scenario, bounds = make_dynamic_problem(name="dyn-sphere", dimension=5, change_frequency=10, change_severity=1.0)
    result = optimize_dynamic(
        DynamicOptimizationProblem(name=scenario.name, scenario=scenario, bounds=bounds),
        MMAODynamicConfig(iterations=20, seed=2),
    )
    assert "offline_error" in result
    assert len(result["history"]) == 21


def test_classification_smoke() -> None:
    result = optimize_classification(
        ClassificationProblem(name="iris-knn", dataset_name="iris", classifier_name="knn"),
        MMAOClassificationConfig(iterations=4, seed=3),
    )
    assert 0.0 <= result["test_metric"] <= 1.0

from __future__ import annotations

from mmao.classification import ClassificationProblem, MMAOClassificationConfig
from mmao.continuous import MMAOContinuousConfig, rastrigin_problem
from mmao.discrete import MMAOTSPConfig
from mmao.dynamic import DynamicOptimizationProblem, MMAODynamicConfig, make_dynamic_problem
from mmao.examples import demo_tsp_problem
from mmao.api import optimize_classification, optimize_continuous, optimize_dynamic, optimize_tsp


def main() -> None:
    continuous = optimize_continuous(rastrigin_problem(dimension=10), MMAOContinuousConfig(iterations=60, seed=7))
    print("Continuous best fitness:", continuous["best_fitness"])

    tsp = optimize_tsp(demo_tsp_problem(), MMAOTSPConfig(iterations=80, seed=11))
    print("TSP best distance:", tsp["best_distance"])

    scenario, bounds = make_dynamic_problem(name="dyn-rastrigin", dimension=10, change_frequency=20, change_severity=1.0)
    dynamic = optimize_dynamic(
        DynamicOptimizationProblem(name=scenario.name, scenario=scenario, bounds=bounds),
        MMAODynamicConfig(iterations=60, seed=7),
    )
    print("Dynamic offline error:", dynamic["offline_error"])

    classification = optimize_classification(
        ClassificationProblem(name="breast-cancer-svm", dataset_name="breast_cancer", classifier_name="svm_rbf"),
        MMAOClassificationConfig(iterations=10, seed=3),
    )
    print("Classification test metric:", classification["test_metric"])


if __name__ == "__main__":
    main()

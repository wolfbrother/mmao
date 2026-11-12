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

import json

import joblib
import numpy as np
import pytest
from sklearn.datasets import load_iris

from mmao import (
    ClassificationProblem, ContinuousProblem, KnapsackProblem,
    MMAOClassificationConfig, MMAOContinuousConfig, MMAOKnapsackConfig,
    MMAODynamicConfig, DynamicOptimizationProblem, make_dynamic_problem,
    optimize_continuous, optimize_knapsack, optimize_classification, optimize_dynamic,
    save_classification_model, sphere_problem, write_result_json,
)
from mmao.api import _attach_known_optimum
from mmao.continuous import core as continuous_core
from mmao.classification import core as classification_core
from mmao.discrete import tsp as tsp_core


def test_continuous_archives_last_generation_children(monkeypatch):
    births = []
    original = continuous_core.make_agent

    def tracked(*args, **kwargs):
        agent = original(*args, **kwargs)
        births.append(agent.best_fitness)
        return agent

    monkeypatch.setattr(continuous_core, "make_agent", tracked)
    result = optimize_continuous(sphere_problem(2), MMAOContinuousConfig(
        iterations=1, seed=2, initial_agents=4, minimum_agents=4, maximum_agents=8,
        initial_pool=100, branch_pool_threshold=0, branch_energy_threshold=0,
    ))
    assert len(births) > 4
    assert result["best_fitness"] <= min(births)
    assert sphere_problem(2).evaluate(result["best_position"]) == result["best_fitness"]


def test_tsp_retains_best_route_after_agent_removal(monkeypatch):
    matrix = [[0, 1, 3, 1], [1, 0, 1, 3], [3, 1, 0, 1], [1, 3, 1, 0]]
    problem = tsp_core.TSPProblem("square", matrix)
    optimal = [0, 1, 2, 3]
    worse = [0, 2, 1, 3]
    original = tsp_core.make_agent
    calls = []

    def make_agent(problem, rng, config, **kwargs):
        route = optimal if not calls else worse
        calls.append(route)
        return original(problem, rng, config, route=route)

    monkeypatch.setattr(tsp_core, "make_agent", make_agent)
    monkeypatch.setattr(tsp_core, "sampled_two_opt", lambda route, problem, *args: (route, problem.route_length(route)))
    monkeypatch.setattr(tsp_core, "mutate_route", lambda route, *args, **kwargs: route)
    monkeypatch.setattr(tsp_core, "guided_route", lambda *args: worse)
    result = tsp_core.optimize_tsp_problem(problem, tsp_core.MMAOTSPConfig(
        iterations=1, initial_agents=2, minimum_agents=1, maximum_agents=2, initial_energy=0.01,
    ))
    assert result["best_distance"] == 4
    assert problem.route_length(result["best_route"]) == 4


@pytest.mark.parametrize("budget", [8, 9, 10, 11, 32, 100])
def test_evaluation_budget_counts_every_objective_call(budget):
    calls = []

    def objective(x):
        value = sum(item * item for item in x)
        calls.append(value)
        return value

    problem = ContinuousProblem("counted", 2, [(-5, 5)] * 2, objective)
    result = optimize_continuous(problem, MMAOContinuousConfig(iterations=100, max_evaluations=budget))
    assert result["nfev"] == len(calls) <= budget
    assert result["best_fitness"] == min(calls)
    assert result["history"][-1]["nfev"] == len(calls)
    assert result["termination_reason"] == "max_evaluations"
    assert "known_optimum" not in result


def test_knapsack_population_limit_and_valid_solution():
    problem = KnapsackProblem("small", [10, 9], [[1, 1]], [1], known_optimum=10)
    result = optimize_knapsack(problem, MMAOKnapsackConfig(
        iterations=4, seed=0, initial_agents=6, minimum_agents=2, maximum_agents=6,
        initial_energy=18, initial_pool_ratio=2,
    ))
    assert all(2 <= row["population"] <= 6 for row in result["history"])
    assert problem.feasible(result["best_solution"])
    assert problem.profit(result["best_solution"]) == result["best_profit"]


def test_dynamic_population_limit():
    scenario, bounds = make_dynamic_problem("dyn-sphere", 3, 2, 1)
    result = optimize_dynamic(DynamicOptimizationProblem(scenario.name, scenario, bounds), MMAODynamicConfig(
        iterations=10, initial_agents=4, minimum_agents=2, maximum_agents=4,
    ))
    assert all(2 <= row["population"] <= 4 for row in result["history"])


def test_classification_reuses_repaired_mask_and_exports_pipeline(monkeypatch, tmp_path):
    x, y = load_iris(return_X_y=True)
    monkeypatch.setattr(classification_core, "derive_feature_budget", lambda priors: (3, 3))
    monkeypatch.setattr(classification_core, "mutate_position", lambda *args: np.array([-1.] * 4 + [0., 0.]))
    monkeypatch.setattr(classification_core, "elite_refine", lambda candidate, outcome, *args: (candidate, outcome))
    result = optimize_classification(
        ClassificationProblem("custom", classifier_name="knn", x=x, y=y),
        MMAOClassificationConfig(iterations=1, initial_agents=2, minimum_agents=1, maximum_agents=3, return_model=True),
    )
    assert result["selected_features"] == result["test_selected_features"] == 3
    assert result["history"][-1]["mean_feature_ratio"] == 0.75
    assert result["history"][-1]["evaluated_population"] == 2
    assert result["selected_feature_indices"] == np.flatnonzero(result["mask"]).tolist()
    path = save_classification_model(result, tmp_path / "model.joblib")
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict(x), result["model"].predict(x))
    write_result_json(result, tmp_path / "result.json")
    payload = json.loads((tmp_path / "result.json").read_text())
    assert "model" not in payload
    assert payload["metadata"]["config"]["return_model"] is True


def test_gap_direction():
    assert _attach_known_optimum({"best_profit": 8}, 10, "best_profit", "maximize")["gap"] == 2
    assert _attach_known_optimum({"best_fitness": 12}, 10, "best_fitness")["gap"] == 2


def test_invalid_problem_and_config_values():
    with pytest.raises(ValueError, match="dimension"):
        sphere_problem(0)
    with pytest.raises(ValueError, match="lower"):
        ContinuousProblem("bad", 1, [(1, -1)], lambda x: 0)
    with pytest.raises(ValueError, match="weight row"):
        KnapsackProblem("bad", [1, 2], [[1]], [1])
    with pytest.raises(ValueError, match="minimum_agents"):
        optimize_continuous(sphere_problem(2), MMAOContinuousConfig(initial_agents=30))
    with pytest.raises(ValueError, match="max_evaluations"):
        optimize_continuous(sphere_problem(2), MMAOContinuousConfig(max_evaluations=1))
    with pytest.raises(ValueError, match="finite"):
        optimize_continuous(ContinuousProblem("bad", 1, [(0, 1)], lambda x: float("nan")))


def test_metadata_and_seed_reproducibility():
    config = MMAOContinuousConfig(iterations=2, seed=12)
    first = optimize_continuous(sphere_problem(2), config)
    second = optimize_continuous(sphere_problem(2), config)
    assert first["history"] == second["history"]
    assert first["metadata"]["config"]["seed"] == 12
    assert first["metadata"]["problem_fingerprint"] == second["metadata"]["problem_fingerprint"]
    assert first["metadata"]["elapsed_seconds"] >= 0


def test_plot_resource_history(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mmao.plotting import plot_history

    result = optimize_continuous(sphere_problem(2), MMAOContinuousConfig(iterations=2))
    figure = plot_history(result, tmp_path / "history.png")
    assert len(figure.axes) == 4
    assert (tmp_path / "history.png").stat().st_size > 0
    plt.close(figure)

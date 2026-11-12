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

"""Release regressions for user inputs, loaders and durable result export."""

import json
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from mmao import (
    ClassificationProblem, MMAOClassificationConfig, MMAOContinuousConfig,
    MMAODynamicConfig, MMAOKnapsackConfig, MMAOTSPConfig,
    DynamicOptimizationProblem, KnapsackProblem, TSPProblem,
    make_dynamic_problem, optimize_classification, optimize_continuous,
    optimize_dynamic, optimize_knapsack, optimize_tsp, rastrigin_problem,
    sphere_problem, save_classification_model, write_result_json,
)
from mmao.common.validation import validate_config
from mmao.io import load_orlib_mkp_instance, load_orlib_mkp_instances, load_tsplib_problem


@pytest.mark.parametrize("config_class", [MMAOContinuousConfig, MMAODynamicConfig, MMAOKnapsackConfig, MMAOTSPConfig, MMAOClassificationConfig])
@pytest.mark.parametrize("value", ["bad", None, True, float("nan"), float("inf"), -0.1])
def test_invalid_numeric_parameters_fail_cleanly(config_class, value):
    config = config_class()
    attribute = "feature_penalty" if config_class is MMAOClassificationConfig else "initial_energy"
    setattr(config, attribute, value)
    with pytest.raises(ValueError):
        validate_config(config)


def test_zero_branching_is_a_valid_configuration():
    result = optimize_continuous(sphere_problem(2), MMAOContinuousConfig(
        iterations=2, branch_children=0, respawn_floor=0,
    ))
    assert len(result["history"]) == 3


def test_numpy_problem_inputs_and_seed_are_json_serializable(tmp_path):
    matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float64)
    tsp = optimize_tsp(TSPProblem("numpy-tsp", matrix), MMAOTSPConfig(iterations=1, seed=np.int64(7)))
    mkp = optimize_knapsack(KnapsackProblem(
        "numpy-mkp", np.array([10, 9]), np.array([[1, 1]]), np.array([1]),
    ), MMAOKnapsackConfig(iterations=1, seed=np.int64(7)))
    for name, result in [("tsp", tsp), ("mkp", mkp)]:
        write_result_json(result, tmp_path / f"{name}.json")
        payload = json.loads((tmp_path / f"{name}.json").read_text())
        assert payload["metadata"]["seed"] == 7


def test_json_export_preserves_existing_file_when_serialization_fails(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"valid": true}')
    with pytest.raises(ValueError):
        write_result_json({"bad": np.array([float("nan")])}, output)
    assert json.loads(output.read_text()) == {"valid": True}
    assert len(list(tmp_path.iterdir())) == 1


def test_json_export_cleans_temporary_files_when_replace_fails(tmp_path, monkeypatch):
    import mmao.results as results

    output = tmp_path / "result.json"
    output.write_text('{"valid": true}')
    monkeypatch.setattr(results.os, "replace", Mock(side_effect=PermissionError("locked")))
    with pytest.raises(PermissionError):
        write_result_json({"new": True}, output)
    assert json.loads(output.read_text()) == {"valid": True}
    assert len(list(tmp_path.iterdir())) == 1


def test_model_export_preserves_existing_file_when_serialization_fails(tmp_path, monkeypatch):
    import joblib

    output = tmp_path / "model.joblib"
    output.write_bytes(b"existing")
    monkeypatch.setattr(joblib, "dump", Mock(side_effect=RuntimeError("serialization failed")))
    with pytest.raises(RuntimeError):
        save_classification_model({"model": Mock(predict=Mock())}, output)
    assert output.read_bytes() == b"existing"
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("content", ["", "a description without data", "2 5\n10 20", "0 5\n"])
def test_malformed_orlib_files_fail_cleanly(tmp_path, content):
    path = tmp_path / "invalid.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_orlib_mkp_instances(path)


def test_orlib_limit_and_index_validation():
    with pytest.raises(ValueError):
        load_orlib_mkp_instances("mknap2", limit=0)
    with pytest.raises(IndexError):
        load_orlib_mkp_instance("mknap2", index=-1)
    assert len(load_orlib_mkp_instances("mknap2", limit=2)) == 2


def test_tsplib_path_objects_and_explicit_optimum(tmp_path):
    pytest.importorskip("tsplib95")
    path = tmp_path / "triangle.tsp"
    path.write_text(
        "NAME: triangle\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\n"
        "NODE_COORD_SECTION\n1 0 0\n2 3 0\n3 0 4\nTOUR_SECTION\n1\n2\n3\n-1\nEOF\n"
    )
    assert load_tsplib_problem(path).known_optimum is None
    assert load_tsplib_problem(path, known_optimum=12).known_optimum == 12


def test_known_optima_follow_custom_bounds():
    assert sphere_problem(2, lower=1, upper=2).optimum_value == 2
    assert rastrigin_problem(2, lower=1.1, upper=1.2).optimum_value is None


def test_dynamic_zero_iterations_reports_initial_error():
    scenario, bounds = make_dynamic_problem("dyn-sphere", 2)
    result = optimize_dynamic(DynamicOptimizationProblem(scenario.name, scenario, bounds), MMAODynamicConfig(iterations=0))
    assert result["offline_error"] == result["best_fitness"]
    assert result["offline_error"] > 0


@pytest.mark.parametrize("kwargs", [{"dimension": 0}, {"change_frequency": 0}, {"change_severity": -1}, {"name": "unknown"}])
def test_dynamic_scenario_validation(kwargs):
    with pytest.raises(ValueError):
        make_dynamic_problem(**kwargs)


def test_small_custom_classification_keeps_neighbor_count_consistent(monkeypatch):
    import mmao.classification.core as core

    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(20, 3)), np.array([0, 1] * 10)
    monkeypatch.setattr(core, "mutate_position", lambda *args: np.array([1., 1., 1., 2., 0.]))
    result = optimize_classification(ClassificationProblem("small", classifier_name="knn", x=x, y=y), MMAOClassificationConfig(
        iterations=1, initial_agents=1, minimum_agents=1, maximum_agents=1, return_model=True,
    ))
    assert result["classifier_parameters"]["n_neighbors"] == 11
    assert result["model"].predict(x[:2]).shape == (2,)


@pytest.mark.parametrize("args", [
    ["continuous", "--dimension", "0"],
    ["tsp", "--benchmark", "berlin52", "--tsplib", "file.tsp"],
    ["dynamic", "--change-frequency", "0"],
])
def test_cli_invalid_input_has_no_traceback(args):
    process = subprocess.run([sys.executable, "-m", "mmao.cli", *args], capture_output=True, text=True)
    assert process.returncode == 2
    assert "error:" in process.stderr
    assert "Traceback" not in process.stderr


def test_version_metadata_agree():
    import mmao
    from importlib.metadata import version

    assert mmao.__version__ == version("mmao-opt")

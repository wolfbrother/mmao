# Copyright (c) 2026 Jinliang Xu
# Author: Jinliang Xu
# Email: jlxufly@gmail.com and xujinliang@caict.ac.cn
# SPDX-License-Identifier: Apache-2.0
"""Budget and resource invariants for the additive research API."""

from dataclasses import replace
import numpy as np
import pytest

from mmao import ResourceControlConfig, optimize_resource_controlled, sphere_problem
from mmao.continuous.controlled import allocation_probabilities


@pytest.mark.parametrize("policy", ["F", "I", "M", "D"])
@pytest.mark.parametrize("kernel", ["probe", "gaussian"])
def test_budget_ledger_and_population(policy, kernel):
    config = ResourceControlConfig(policy=policy, kernel=kernel, max_evaluations=97)
    result = optimize_resource_controlled(sphere_problem(3), config)
    assert result["nfev"] <= 97
    assert result["nfev"] == len(result["evaluations"])
    assert result["evaluations"][-1][1] == result["best_fitness"]
    assert all(4 <= row["population"] <= 12 for row in result["history"])
    assert all(row["pool"] >= 0 and abs(row["ledger_residual"]) < 1e-9 for row in result["history"])


def test_common_initialization_and_repeatability():
    config = ResourceControlConfig(max_evaluations=100)
    left = optimize_resource_controlled(sphere_problem(3), config)
    right = optimize_resource_controlled(sphere_problem(3), config)
    assert left == right
    for policy in ["F", "I", "D"]:
        other = optimize_resource_controlled(sphere_problem(3), replace(config, policy=policy))
        assert left["evaluations"][:8] == other["evaluations"][:8]


def test_softmax_floor_and_shift_invariance():
    probabilities = allocation_probabilities([0, 2, 5], 0.1, 2)
    assert np.isclose(probabilities.sum(), 1)
    assert np.all(probabilities >= 0.1 / 3)
    assert np.all(np.diff(probabilities) > 0)
    np.testing.assert_allclose(probabilities, allocation_probabilities([10, 12, 15], 0.1, 2))


def test_validation_before_objective_calls():
    with pytest.raises(ValueError):
        optimize_resource_controlled(sphere_problem(3), ResourceControlConfig(policy="unknown"))

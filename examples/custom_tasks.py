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

"""Custom continuous objective and custom classification data."""

from sklearn.datasets import load_iris
from mmao import (
    ClassificationProblem, ContinuousProblem, MMAOClassificationConfig,
    MMAOContinuousConfig, optimize_classification, optimize_continuous,
    save_classification_model, write_result_json,
)


def objective(x):
    return (x[0] - 1.5) ** 2 + (x[1] + 0.5) ** 2


def main():
    problem = ContinuousProblem("custom-quadratic", 2, [(-5, 5)] * 2, objective, optimum_value=0.0)
    result = optimize_continuous(problem, MMAOContinuousConfig(iterations=100, max_evaluations=500))
    write_result_json(result, "outputs/custom-continuous.json")
    print("Continuous", result["best_fitness"], "evaluations", result["nfev"])

    x, y = load_iris(return_X_y=True)
    result = optimize_classification(
        ClassificationProblem("custom-iris", classifier_name="knn", x=x, y=y),
        MMAOClassificationConfig(iterations=4, return_model=True),
    )
    write_result_json(result, "outputs/custom-classification.json")
    save_classification_model(result, "outputs/custom-classification.joblib")
    print("Features", result["selected_feature_indices"], "predictions", result["model"].predict(x[:3]))


if __name__ == "__main__":
    main()

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

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
            if not (key == "model" and hasattr(item, "predict"))
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "problem",
        "best_fitness",
        "best_distance",
        "best_profit",
        "offline_error",
        "test_metric",
        "feature_ratio",
        "test_feature_ratio",
        "known_optimum",
        "gap",
        "relative_gap",
        "optimization_direction",
        "nfev",
        "termination_reason",
        "metadata",
    ):
        if key in result:
            summary[key] = result[key]

    history = result.get("history")
    if isinstance(history, list):
        summary["iterations_recorded"] = len(history)
        if history:
            summary["final_state"] = history[-1]
    return summary


def write_result_json(result: dict[str, Any], output_path: str | Path) -> Path:
    text = json.dumps(to_jsonable(result), indent=2, allow_nan=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def save_classification_model(result: dict[str, Any], output_path: str | Path) -> Path:
    """Save the fitted pipeline produced with return_model=True."""
    import joblib

    if "model" not in result or not hasattr(result["model"], "predict"):
        raise ValueError("Run classification with return_model=True before saving a model.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(result["model"], temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path

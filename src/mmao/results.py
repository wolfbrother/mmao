from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(result), indent=2), encoding="utf-8")
    return path

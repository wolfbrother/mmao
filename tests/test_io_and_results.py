from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmao.api import optimize_knapsack
from mmao.examples import demo_knapsack_problem
from mmao.io import (
    bundled_orlib_path,
    bundled_tsplib_path,
    load_bundled_orlib_mkp_instance,
    load_bundled_tsplib_problem,
    load_orlib_mkp_instance,
    load_orlib_mkp_instances,
    load_tsplib_problem,
)
from mmao.results import summarize_result, to_jsonable, write_result_json


def test_orlib_loader_smoke() -> None:
    sample = Path(__file__).parent / "data" / "sample_mkp.txt"
    instances = load_orlib_mkp_instances(sample)
    assert len(instances) == 1
    instance = load_orlib_mkp_instance(sample, 0)
    assert instance.item_count == 5
    assert instance.known_optimum == 36


def test_bundled_benchmarks_smoke() -> None:
    tsp = load_bundled_tsplib_problem("berlin52")
    mkp = load_bundled_orlib_mkp_instance("mknap2", 0)
    assert bundled_tsplib_path("berlin52").name == "berlin52.tsp"
    assert bundled_orlib_path("mknap2").name == "mknap2.txt"
    assert tsp.city_count == 52
    assert mkp.item_count > 0


def test_short_benchmark_names_resolve_to_bundled_files() -> None:
    tsp = load_tsplib_problem("berlin52.tsp")
    mkp = load_orlib_mkp_instance("mknap2.txt", 0)
    assert tsp.name.lower() == "berlin52"
    assert mkp.name.startswith("ORL-MKP-")


def test_result_summary_and_export(tmp_path: Path) -> None:
    result = optimize_knapsack(demo_knapsack_problem())
    summary = summarize_result(result)
    assert "best_profit" in summary

    output = tmp_path / "result.json"
    write_result_json(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_profit"] == result["best_profit"]


def test_to_jsonable_handles_numpy_values() -> None:
    payload = {
        "vector": np.array([1, 2, 3]),
        "scalar": np.float64(1.5),
        "nested": {"mask": np.array([0, 1])},
    }
    converted = to_jsonable(payload)
    assert converted == {"vector": [1, 2, 3], "scalar": 1.5, "nested": {"mask": [0, 1]}}


def test_missing_file_errors_are_user_friendly() -> None:
    with pytest.raises(FileNotFoundError, match="Provide a real local \\.tsp file path"):
        load_tsplib_problem("path/to/berlin52.tsp")

    with pytest.raises(FileNotFoundError, match="Provide a real local \\.txt file path"):
        load_orlib_mkp_instance("path/to/mknap2.txt", 0)

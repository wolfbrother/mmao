from __future__ import annotations

import json
from pathlib import Path

from mmao.api import optimize_knapsack
from mmao.examples import demo_knapsack_problem
from mmao.io import (
    bundled_orlib_path,
    bundled_tsplib_path,
    load_bundled_orlib_mkp_instance,
    load_bundled_tsplib_problem,
    load_orlib_mkp_instance,
    load_orlib_mkp_instances,
)
from mmao.results import summarize_result, write_result_json


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


def test_result_summary_and_export(tmp_path: Path) -> None:
    result = optimize_knapsack(demo_knapsack_problem())
    summary = summarize_result(result)
    assert "best_profit" in summary

    output = tmp_path / "result.json"
    write_result_json(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["best_profit"] == result["best_profit"]

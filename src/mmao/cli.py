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

import argparse
import json
from .api import optimize_classification, optimize_continuous, optimize_dynamic, optimize_knapsack, optimize_tsp
from .version import __version__

from .classification import ClassificationProblem, MMAOClassificationConfig
from .continuous import MMAOContinuousConfig, rastrigin_problem, sphere_problem
from .discrete import MMAOKnapsackConfig, MMAOTSPConfig
from .dynamic import DynamicOptimizationProblem, MMAODynamicConfig, make_dynamic_problem
from .examples import demo_knapsack_problem, demo_tsp_problem
from .results import summarize_result, to_jsonable, write_result_json


def add_common_output_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--output", type=str, default=None, help="Optional path to save the full JSON result.")
    subparser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a compact summary instead of the full optimization history.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMAO examples and lightweight benchmark tasks.")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cont = subparsers.add_parser("continuous", help="Run a continuous optimization task.")
    cont.add_argument("--problem", choices=["sphere", "rastrigin"], default="rastrigin")
    cont.add_argument("--dimension", type=int, default=10)
    cont.add_argument("--iterations", type=int, default=80)
    cont.add_argument("--seed", type=int, default=7)
    cont.add_argument("--max-evaluations", type=int, default=None)
    add_common_output_args(cont)

    tsp = subparsers.add_parser("tsp", help="Run a TSP example or TSPLIB file.")
    tsp_input = tsp.add_mutually_exclusive_group()
    tsp_input.add_argument("--benchmark", choices=["berlin52"], default=None, help="Run a bundled TSPLIB benchmark.")
    tsp_input.add_argument("--tsplib", type=str, default=None)
    tsp.add_argument("--iterations", type=int, default=100)
    tsp.add_argument("--seed", type=int, default=11)
    add_common_output_args(tsp)

    mkp = subparsers.add_parser("knapsack", help="Run a multidimensional knapsack example.")
    mkp_input = mkp.add_mutually_exclusive_group()
    mkp_input.add_argument("--benchmark", choices=["mknap2"], default=None, help="Run a bundled OR-Library benchmark collection.")
    mkp_input.add_argument("--orlib", type=str, default=None, help="Optional OR-Library MKP file path.")
    mkp.add_argument("--instance-index", type=int, default=0, help="Instance index in the OR-Library file.")
    mkp.add_argument("--iterations", type=int, default=60)
    mkp.add_argument("--seed", type=int, default=23)
    add_common_output_args(mkp)

    dyn = subparsers.add_parser("dynamic", help="Run a dynamic optimization task.")
    dyn.add_argument("--problem", choices=["dyn-sphere", "dyn-rastrigin", "dyn-ackley"], default="dyn-rastrigin")
    dyn.add_argument("--dimension", type=int, default=10)
    dyn.add_argument("--iterations", type=int, default=80)
    dyn.add_argument("--change-frequency", type=int, default=20)
    dyn.add_argument("--change-severity", type=float, default=1.0)
    dyn.add_argument("--seed", type=int, default=7)
    add_common_output_args(dyn)

    cls = subparsers.add_parser("classification", help="Run a classification-oriented MMAO task.")
    cls.add_argument("--dataset", choices=["breast_cancer", "wine", "digits", "iris"], default="breast_cancer")
    cls.add_argument("--classifier", choices=["svm_rbf", "knn", "logreg"], default="svm_rbf")
    cls.add_argument("--metric", choices=["balanced_accuracy", "f1_macro"], default="balanced_accuracy")
    cls.add_argument("--iterations", type=int, default=12)
    cls.add_argument("--seed", type=int, default=3)
    add_common_output_args(cls)

    return parser


def run_task(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "continuous":
        problem = sphere_problem(args.dimension) if args.problem == "sphere" else rastrigin_problem(args.dimension)
        result = optimize_continuous(problem, MMAOContinuousConfig(iterations=args.iterations, seed=args.seed, max_evaluations=args.max_evaluations))
    elif args.command == "tsp":
        if args.benchmark:
            from .io import load_bundled_tsplib_problem

            problem = load_bundled_tsplib_problem(args.benchmark)
        elif args.tsplib:
            from .io import load_tsplib_problem

            problem = load_tsplib_problem(args.tsplib)
        else:
            problem = demo_tsp_problem()
        result = optimize_tsp(problem, MMAOTSPConfig(iterations=args.iterations, seed=args.seed))
    elif args.command == "knapsack":
        if args.benchmark:
            from .io import load_bundled_orlib_mkp_instance

            problem = load_bundled_orlib_mkp_instance(args.benchmark, index=args.instance_index)
        elif args.orlib:
            from .io import load_orlib_mkp_instance

            problem = load_orlib_mkp_instance(args.orlib, index=args.instance_index)
        else:
            problem = demo_knapsack_problem()
        result = optimize_knapsack(problem, MMAOKnapsackConfig(iterations=args.iterations, seed=args.seed))
    elif args.command == "dynamic":
        scenario, bounds = make_dynamic_problem(args.problem, args.dimension, args.change_frequency, args.change_severity)
        problem = DynamicOptimizationProblem(name=scenario.name, scenario=scenario, bounds=bounds)
        result = optimize_dynamic(problem, MMAODynamicConfig(iterations=args.iterations, seed=args.seed))
    else:
        problem = ClassificationProblem(
            name=f"{args.dataset}-{args.classifier}",
            dataset_name=args.dataset,
            classifier_name=args.classifier,
            metric=args.metric,
        )
        result = optimize_classification(problem, MMAOClassificationConfig(iterations=args.iterations, seed=args.seed))
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_task(args)
        if args.output:
            write_result_json(result, args.output)
        payload = summarize_result(result) if args.summary_only else result
        print(json.dumps(to_jsonable(payload), indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

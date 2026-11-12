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

"""Small fixed-budget seed sweep with descriptive statistics."""

import argparse
from statistics import mean, stdev
from math import sqrt
from scipy.stats import t

from mmao import MMAOContinuousConfig, optimize_continuous, sphere_problem, rastrigin_problem, write_result_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["sphere", "rastrigin"], default="sphere")
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", default="outputs/batch.json")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide distinct seeds.")
    factory = sphere_problem if args.problem == "sphere" else rastrigin_problem
    runs = [
        optimize_continuous(factory(args.dimension), MMAOContinuousConfig(
            iterations=args.iterations, seed=seed, max_evaluations=args.max_evaluations,
        ))
        for seed in args.seeds
    ]
    values = [run["best_fitness"] for run in runs]
    deviation = stdev(values) if len(values) > 1 else 0.0
    statistics = {"count": len(values), "mean": mean(values), "sample_std": deviation}
    if len(values) > 1:
        margin = float(t.ppf(0.975, len(values) - 1)) * deviation / sqrt(len(values))
        statistics["mean_ci95_student_t"] = [mean(values) - margin, mean(values) + margin]
    write_result_json({"runs": runs, "statistics": statistics}, args.output)
    print(statistics)


if __name__ == "__main__":
    main()

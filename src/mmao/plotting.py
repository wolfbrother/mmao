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

"""Plot optimization and resource trajectories with the optional plot extra."""

from pathlib import Path


def plot_history(result: dict, output_path: str | Path | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install mmao-opt[plot].") from exc

    history = result.get("history", [])
    if not history:
        raise ValueError("A nonempty optimization history is required.")
    metric = next((key for key in ("best_fitness", "best_distance", "best_profit", "best_objective") if key in history[0]), None)
    if metric is None:
        raise ValueError("History has no supported objective metric.")
    pool = "resource_pool" if "resource_pool" in history[0] else "communal_budget"
    keys = [key for key in (metric, pool, "population", "mean_role") if key in history[0]]
    figure, axes = plt.subplots(len(keys), 1, figsize=(8, 2.5 * len(keys)), squeeze=False)
    iterations = [record["iteration"] for record in history]
    for axis, key in zip(axes[:, 0], keys):
        axis.plot(iterations, [record[key] for record in history])
        axis.set_ylabel(key.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("iteration")
    figure.suptitle(str(result.get("problem", "MMAO")))
    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    return figure

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


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def median_upper_scale(recent_gains: list[float], fallback: float) -> float:
    positives = [value for value in recent_gains if value > 1e-12]
    if not positives:
        return max(1e-9, fallback)
    positives.sort()
    median = positives[len(positives) // 2]
    upper = positives[min(len(positives) - 1, int(0.8 * (len(positives) - 1)))]
    return max(1e-9, 0.65 * median + 0.35 * upper)

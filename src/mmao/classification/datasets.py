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

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.preprocessing import LabelEncoder


def _encode_labels(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(np.asarray(y).dtype, np.number):
        return np.asarray(y)
    return LabelEncoder().fit_transform(np.asarray(y))


def load_builtin_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    key = name.lower()
    if key == "breast_cancer":
        ds = load_breast_cancer()
        return np.asarray(ds.data, dtype=float), _encode_labels(ds.target)
    if key == "wine":
        ds = load_wine()
        return np.asarray(ds.data, dtype=float), _encode_labels(ds.target)
    if key == "digits":
        ds = load_digits()
        return np.asarray(ds.data, dtype=float), _encode_labels(ds.target)
    if key == "iris":
        ds = load_iris()
        return np.asarray(ds.data, dtype=float), _encode_labels(ds.target)
    raise ValueError(f"Unsupported built-in dataset: {name}")

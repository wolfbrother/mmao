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

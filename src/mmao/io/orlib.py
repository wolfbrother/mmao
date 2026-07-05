from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ..discrete.knapsack import KnapsackProblem


def _resolve_orlib_path(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.exists():
        return file_path

    normalized = str(path).strip().lower().replace("\\", "/")
    if normalized in {"mknap2", "mknap2.txt"}:
        return Path(str(files("mmao.data.orlib").joinpath("mknap2.txt")))

    raise FileNotFoundError(
        f"OR-Library file not found: {path}. Provide a real local .txt file path, use "
        "'load_bundled_orlib_mkp_instance(\"mknap2\", index=0)' in Python, or use "
        "'mmao knapsack --benchmark mknap2 --instance-index 0 --summary-only' for the bundled example."
    )


def load_orlib_mkp_instances(path: str | Path, *, limit: int | None = None) -> list[KnapsackProblem]:
    file_path = _resolve_orlib_path(path)
    lines = file_path.read_text(encoding="utf-8").splitlines()
    cleaned_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    tokens = text.split()
    index = 0
    instances: list[KnapsackProblem] = []
    instance_id = 1

    while index < len(tokens):
        try:
            int(tokens[index])
            int(tokens[index + 1])
            break
        except Exception:
            index += 1

    while index + 2 <= len(tokens):
        try:
            constraint_count = int(tokens[index])
            item_count = int(tokens[index + 1])
        except ValueError:
            index += 1
            continue
        index += 2
        if constraint_count <= 0 or item_count <= 0:
            break

        profits = [int(tokens[index + offset]) for offset in range(item_count)]
        index += item_count
        capacities = [int(tokens[index + offset]) for offset in range(constraint_count)]
        index += constraint_count
        weights: list[list[int]] = []
        for _ in range(constraint_count):
            weights.append([int(tokens[index + offset]) for offset in range(item_count)])
            index += item_count
        optimum = int(tokens[index])
        index += 1

        instances.append(
            KnapsackProblem(
                name=f"ORL-MKP-{instance_id:02d}",
                profits=profits,
                weights=weights,
                capacities=capacities,
                known_optimum=optimum,
            )
        )
        instance_id += 1
        if limit is not None and len(instances) >= limit:
            break

    return instances


def load_orlib_mkp_instance(path: str | Path, index: int = 0) -> KnapsackProblem:
    instances = load_orlib_mkp_instances(path, limit=None)
    if index < 0 or index >= len(instances):
        raise IndexError(f"Requested OR-Library instance index {index}, but only {len(instances)} instances were loaded.")
    return instances[index]

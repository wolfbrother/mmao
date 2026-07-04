from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicScenario:
    name: str
    dimension: int
    change_frequency: int
    change_severity: float
    shift_scale: float


def make_dynamic_problem(
    name: str = "dyn-sphere",
    dimension: int = 10,
    change_frequency: int = 25,
    change_severity: float = 1.0,
) -> tuple[DynamicScenario, list[tuple[float, float]]]:
    bounds = [(-5.0, 5.0) for _ in range(dimension)]
    shift_scale = 0.55 * change_severity
    return (
        DynamicScenario(
            name=name,
            dimension=dimension,
            change_frequency=change_frequency,
            change_severity=change_severity,
            shift_scale=shift_scale,
        ),
        bounds,
    )

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

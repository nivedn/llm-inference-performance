"""Distribution statistics for latency samples."""

from __future__ import annotations

import math
from collections.abc import Sequence

PERCENTILES = (50, 90, 95, 99)


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not samples:
        raise ValueError("percentile of empty sample")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be in [0, 100], got {q}")
    xs = sorted(samples)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def distribution(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Summary of a latency distribution.

    `n` is always reported: a P99 from 20 samples is just the max, not a tail estimate.
    Aim for >= 100 samples per configuration before quoting P99.
    """
    if not samples:
        return {"n": 0, "mean": None, "min": None, "max": None} | {
            f"p{q}": None for q in PERCENTILES
        }
    return {
        "n": len(samples),
        "mean": sum(samples) / len(samples),
        "min": min(samples),
        "max": max(samples),
    } | {f"p{q}": percentile(samples, q) for q in PERCENTILES}

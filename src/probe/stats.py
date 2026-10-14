"""Summary statistics for timing samples.

These definitions match probe_core's C++ implementation (stats.hpp) exactly,
and both are tested against tests/fixtures/stats_golden.txt:

- p50/p95: linear interpolation between closest ranks (NumPy ``method="linear"``,
  Hyndman-Fan type 7).
- stddev: sample standard deviation (``ddof=1``); ``None`` for a single sample,
  where spread is undefined.
- Empty input, non-numbers, and any NaN, infinite or negative duration are
  rejected.
"""

from collections.abc import Sequence

import numpy as np

STAT_FIELDS = ("count", "min", "max", "mean", "p50", "p95", "stddev")


def as_durations(samples: Sequence[float]) -> np.ndarray:
    """Validate a series of durations and return it as a float64 array."""
    if isinstance(samples, str | bytes) or not isinstance(samples, Sequence | np.ndarray):
        raise ValueError("samples must be a list of numbers")
    for i, value in enumerate(samples):
        # bool is a subclass of int; strings would be silently parsed by NumPy.
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            raise ValueError(f"sample {i} is not a number")
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("no samples")
    bad = ~np.isfinite(x) | (x < 0)
    if bad.any():
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(f"sample {index} is not a finite non-negative duration")
    return x


def summarize(samples: Sequence[float]) -> dict[str, float | int | None]:
    x = as_durations(samples)
    return {
        "count": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50, method="linear")),
        "p95": float(np.percentile(x, 95, method="linear")),
        "stddev": float(np.std(x, ddof=1)) if x.size > 1 else None,
    }


def mad(samples: Sequence[float]) -> float:
    """Median absolute deviation: median(|x - median(x)|).

    A robust measure of spread: unlike the standard deviation, a few extreme
    iterations (a context switch, a clock dip) barely move it.
    """
    x = as_durations(samples)
    median = np.median(x)
    return float(np.median(np.abs(x - median)))

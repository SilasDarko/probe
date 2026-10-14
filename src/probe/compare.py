"""The regression decision rule.

For one metric (e.g. the kernel phase) and one statistic (p50 by default),
Probe compares the baseline's samples with the current run's samples and
returns one of four verdicts:

  REGRESSION    slower by more than the threshold AND by more than the noise floor
  IMPROVEMENT   faster by more than the threshold AND by more than the noise floor
  INCONCLUSIVE  the change exceeds the threshold but not the noise floor, or
                there are too few samples to estimate the noise
  PASS          everything else: no change beyond noise, or a real change that
                is within the threshold

Two separate questions, two separate tests:

1. "Is the change big enough to matter?"  -> the user's percentage threshold.
   This also absorbs run-to-run drift (clocks, temperature, other processes),
   which the samples of a single run cannot reveal.

2. "Is the change bigger than chance?"    -> the noise floor:

       noise_floor = max(Z * sqrt(se_b^2 + se_c^2), TIMER_RESOLUTION_MS)
       se          = factor(stat) * sigma / sqrt(n)
       sigma       = MAD_TO_SIGMA * MAD

   se is the standard error of the statistic: how much the median (or p95,
   or mean) of n samples would wobble between repeated runs if nothing had
   changed. sigma is estimated from the MAD so that a few outlier iterations
   do not inflate it. A difference of two independent statistics has
   standard error sqrt(se_b^2 + se_c^2). Requiring Z = 3 standard errors means
   that, under a normal model, an unchanged workload exceeds the floor about
   0.3% of the time.

The noise floor shrinks as iterations increase: more samples pin the median
down more precisely. A rule based on raw spread alone (for example "3 x MAD")
would not: with 1,000 iterations and 2% per-iteration jitter it would ignore a
5% median shift that is, statistically, unmistakable.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

import numpy as np

from probe.stats import as_durations, mad

NOISE_Z = 3.0
# For normally distributed data, sigma = 1.4826 * MAD.
MAD_TO_SIGMA = 1.4826
# Below this many samples on either side, MAD is too unstable to trust.
MIN_SAMPLES = 10
# CUDA events resolve time to about 0.5 microseconds (CUDA Runtime API docs,
# cudaEventElapsedTime); differences below that are not measurable.
TIMER_RESOLUTION_MS = 0.0005
# Used to classify changes when no --fail-if-regression threshold is given.
DEFAULT_THRESHOLD_PCT = 5.0

SUPPORTED_STATS = ("p50", "p95", "mean")


class Verdict(StrEnum):
    PASS = "PASS"
    INCONCLUSIVE = "INCONCLUSIVE"
    REGRESSION = "REGRESSION"
    IMPROVEMENT = "IMPROVEMENT"


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    stat: str
    baseline: float  # the chosen statistic
    current: float
    delta: float  # current - baseline, ms
    percent: float  # 100 * delta / baseline; +/-inf if baseline is 0 and current is not
    baseline_p95: float
    current_p95: float
    noise_floor: float  # ms
    verdict: Verdict
    reason: str


def statistic(samples, stat: str) -> float:
    x = as_durations(samples)
    if stat == "p50":
        return float(np.percentile(x, 50, method="linear"))
    if stat == "p95":
        return float(np.percentile(x, 95, method="linear"))
    if stat == "mean":
        return float(x.mean())
    raise ValueError(f"unsupported statistic '{stat}'")


def standard_error_factor(stat: str) -> float:
    """c such that SE(stat) ~= c * sigma / sqrt(n) for normally distributed data.

    For the mean c = 1. For the q-quantile, asymptotic theory gives
    SE = sqrt(q(1 - q) / n) / f(x_q); with a normal density this is
    c = sqrt(q(1 - q)) / phi(z_q): about 1.2533 for the median and 2.11 for
    p95, which is why tail percentiles need larger changes to be trusted.
    """
    if stat == "mean":
        return 1.0
    q = {"p50": 0.50, "p95": 0.95}.get(stat)
    if q is None:
        raise ValueError(f"unsupported statistic '{stat}'")
    normal = NormalDist()
    return math.sqrt(q * (1.0 - q)) / normal.pdf(normal.inv_cdf(q))


def noise_floor(baseline, current, stat: str) -> float:
    factor = standard_error_factor(stat)

    def standard_error(samples) -> float:
        sigma = MAD_TO_SIGMA * mad(samples)
        return factor * sigma / math.sqrt(len(samples))

    combined = math.hypot(standard_error(baseline), standard_error(current))
    return max(NOISE_Z * combined, TIMER_RESOLUTION_MS)


def percent_change(baseline: float, current: float) -> float:
    if baseline == 0.0:
        if current == 0.0:
            return 0.0
        return math.copysign(math.inf, current)
    return 100.0 * (current - baseline) / baseline


def decide(
    delta: float,
    percent: float,
    floor: float,
    threshold_pct: float,
    n_baseline: int,
    n_current: int,
) -> tuple[Verdict, str]:
    """The decision rule itself, separated from the statistics so it can be
    tested exactly. "Exceeds" is strict: a change equal to the threshold, or
    equal to the noise floor, does not count."""
    if threshold_pct < 0:
        raise ValueError("threshold must be non-negative")
    if min(n_baseline, n_current) < MIN_SAMPLES:
        return Verdict.INCONCLUSIVE, f"fewer than {MIN_SAMPLES} samples on one side"

    above_threshold = abs(percent) > threshold_pct
    above_noise = abs(delta) > floor

    if not above_noise:
        if above_threshold:
            return Verdict.INCONCLUSIVE, "change exceeds the threshold but not the noise floor"
        return Verdict.PASS, "no change beyond the noise floor"
    if not above_threshold:
        return Verdict.PASS, "measurable change, within the threshold"
    if delta > 0:
        return Verdict.REGRESSION, "slower beyond both the threshold and the noise floor"
    return Verdict.IMPROVEMENT, "faster beyond both the threshold and the noise floor"


def compare_series(
    metric: str, baseline, current, stat: str, threshold_pct: float
) -> MetricComparison:
    b = statistic(baseline, stat)
    c = statistic(current, stat)
    delta = c - b
    percent = percent_change(b, c)
    floor = noise_floor(baseline, current, stat)
    verdict, reason = decide(delta, percent, floor, threshold_pct, len(baseline), len(current))
    return MetricComparison(
        metric=metric,
        stat=stat,
        baseline=b,
        current=c,
        delta=delta,
        percent=percent,
        baseline_p95=statistic(baseline, "p95"),
        current_p95=statistic(current, "p95"),
        noise_floor=floor,
        verdict=verdict,
        reason=reason,
    )

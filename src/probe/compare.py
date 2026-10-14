"""The regression decision rule.

For one metric (e.g. the kernel phase) and one statistic (p50 by default),
Probe compares the baseline's samples with the current run's samples and
returns one of four verdicts:

  REGRESSION    slower by more than the threshold AND by more than the smallest
                detectable change
  IMPROVEMENT   faster by more than the threshold AND by more than the smallest
                detectable change
  INCONCLUSIVE  the change exceeds the threshold but is not detectable, or
                there are too few samples for the chosen statistic
  PASS          everything else: no change beyond noise, or a real change that
                is within the threshold

Two separate questions, two separate tests:

1. "Is the change big enough to matter?"  -> the user's percentage threshold.
   This also absorbs run-to-run drift (clocks, temperature, other processes),
   which the samples of a single run cannot reveal.

2. "Is the change bigger than chance?"    -> the smallest detectable change:

       min_detectable = max(noise_floor, timer_resolution)
       noise_floor    = Z * sqrt(se_b^2 + se_c^2)      estimated from the samples
       se             = factor(stat) * sigma / sqrt(n)
       sigma          = MAD_TO_SIGMA * MAD

   Two different things, kept apart on purpose:

   noise_floor       how much the statistic varies by chance, estimated from
                     the observed samples. This is the only noise estimate.
   timer_resolution  a property of the clock, not of the workload: changes
                     smaller than it cannot be measured, however quiet the
                     samples look (e.g. when every sample is identical and
                     MAD = 0). It is not a noise estimate.

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

from probe.result_schema import HOST_END_TO_END
from probe.stats import as_durations, mad

NOISE_Z = 3.0
# For normally distributed data, sigma = 1.4826 * MAD.
MAD_TO_SIGMA = 1.4826
# Minimum samples on EACH side before a statistic is trusted; fewer gives
# INCONCLUSIVE. See docs/regression-model.md for the derivation.
#   p95:  a distribution-free 95% confidence interval for the 95th percentile
#         only ends at an observed sample once n >= 72; below that, p95 is
#         essentially the slowest one or two iterations. 100 leaves about five
#         observations above the estimate.
#   p50, mean: the binding constraint is the MAD-based noise floor, whose sigma
#         estimate is uncertain by about 1.17 / sqrt(n): 37% at n = 10, 26% at
#         n = 20. 20 is also probe-workloads' default iteration count.
MIN_SAMPLES = {"mean": 20, "p50": 20, "p95": 100}
# Clock resolutions: approximate smallest measurable differences. These are
# properties of the clocks, NOT estimates of execution noise.
# CUDA events: "resolution of around 0.5 microseconds" (CUDA Runtime API,
# cudaEventElapsedTime). Applies to every GPU phase and to gpu_total.
EVENT_TIMER_RESOLUTION_MS = 0.0005
# std::chrono::steady_clock: nominal 1 ns tick on Linux/libstdc++. The real
# precision of a host measurement is worse; that shows up in the samples' MAD.
HOST_CLOCK_RESOLUTION_MS = 0.000001
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
    noise_floor: float  # ms, estimated from the samples (may be 0)
    timer_resolution: float  # ms, property of the clock that measured this metric
    min_detectable: float  # ms, max(noise_floor, timer_resolution)
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


def timer_resolution_ms(metric: str) -> float:
    """Resolution of the clock that measured `metric`."""
    return HOST_CLOCK_RESOLUTION_MS if metric == HOST_END_TO_END else EVENT_TIMER_RESOLUTION_MS


def noise_floor(baseline, current, stat: str) -> float:
    """NOISE_Z standard errors of the difference between the two statistics,
    estimated from the samples alone. Zero if neither side shows any spread."""
    factor = standard_error_factor(stat)

    def standard_error(samples) -> float:
        sigma = MAD_TO_SIGMA * mad(samples)
        return factor * sigma / math.sqrt(len(samples))

    combined = math.hypot(standard_error(baseline), standard_error(current))
    return NOISE_Z * combined


def percent_change(baseline: float, current: float) -> float:
    if baseline == 0.0:
        if current == 0.0:
            return 0.0
        return math.copysign(math.inf, current)
    return 100.0 * (current - baseline) / baseline


def decide(
    delta: float,
    percent: float,
    noise: float,
    resolution: float,
    threshold_pct: float,
    n_baseline: int,
    n_current: int,
    min_samples: int,
) -> tuple[Verdict, str]:
    """The decision rule itself, separated from the statistics so it can be
    tested exactly. `noise` is the sample-derived noise floor, `resolution`
    the clock's resolution; a change must exceed the larger of the two.
    "Exceeds" is strict: a change equal to the threshold or to that bound
    does not count."""
    if threshold_pct < 0:
        raise ValueError("threshold must be non-negative")
    if min(n_baseline, n_current) < min_samples:
        return Verdict.INCONCLUSIVE, f"fewer than {min_samples} samples on one side"

    min_detectable = max(noise, resolution)
    bound = "the noise floor" if noise >= resolution else "the timer resolution"
    above_threshold = abs(percent) > threshold_pct
    detectable = abs(delta) > min_detectable

    if not detectable:
        if above_threshold:
            return Verdict.INCONCLUSIVE, f"change exceeds the threshold but not {bound}"
        return Verdict.PASS, f"no change beyond {bound}"
    if not above_threshold:
        return Verdict.PASS, "measurable change, within the threshold"
    if delta > 0:
        return Verdict.REGRESSION, f"slower beyond both the threshold and {bound}"
    return Verdict.IMPROVEMENT, f"faster beyond both the threshold and {bound}"


def compare_series(
    metric: str, baseline, current, stat: str, threshold_pct: float
) -> MetricComparison:
    b = statistic(baseline, stat)
    c = statistic(current, stat)
    delta = c - b
    percent = percent_change(b, c)
    noise = noise_floor(baseline, current, stat)
    resolution = timer_resolution_ms(metric)
    verdict, reason = decide(
        delta,
        percent,
        noise,
        resolution,
        threshold_pct,
        len(baseline),
        len(current),
        MIN_SAMPLES[stat],
    )
    return MetricComparison(
        metric=metric,
        stat=stat,
        baseline=b,
        current=c,
        delta=delta,
        percent=percent,
        baseline_p95=statistic(baseline, "p95"),
        current_p95=statistic(current, "p95"),
        noise_floor=noise,
        timer_resolution=resolution,
        min_detectable=max(noise, resolution),
        verdict=verdict,
        reason=reason,
    )

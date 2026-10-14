"""Tests for the regression decision rule (probe.compare)."""

import math

import pytest

from probe.compare import (
    MIN_SAMPLES,
    NOISE_Z,
    TIMER_RESOLUTION_MS,
    Verdict,
    compare_series,
    decide,
    noise_floor,
    percent_change,
    standard_error_factor,
)
from probe.stats import mad

from synthetic import jitter

# A small hand-checkable distribution: median 10, MAD 1.
SPREAD_10 = [10.0, 11.0, 9.0, 10.0, 12.0, 8.0, 10.0, 10.0, 11.0, 9.0]


# --- the pure decision rule ------------------------------------------------------


def test_regression_needs_threshold_and_noise():
    assert decide(0.3, 15.0, 0.1, 10.0, 20, 20)[0] is Verdict.REGRESSION


def test_improvement_is_symmetric():
    assert decide(-0.3, -15.0, 0.1, 10.0, 20, 20)[0] is Verdict.IMPROVEMENT


def test_change_equal_to_threshold_is_not_a_regression():
    # "exceeds" is strict
    assert decide(0.3, 10.0, 0.1, 10.0, 20, 20)[0] is Verdict.PASS
    assert decide(0.3, 10.000001, 0.1, 10.0, 20, 20)[0] is Verdict.REGRESSION
    assert decide(-0.3, -10.0, 0.1, 10.0, 20, 20)[0] is Verdict.PASS


def test_change_equal_to_noise_floor_counts_as_noise():
    assert decide(0.1, 20.0, 0.1, 10.0, 20, 20)[0] is Verdict.INCONCLUSIVE
    assert decide(0.1000001, 20.0, 0.1, 10.0, 20, 20)[0] is Verdict.REGRESSION


def test_above_threshold_but_within_noise_is_inconclusive():
    verdict, reason = decide(0.05, 15.0, 0.1, 10.0, 20, 20)
    assert verdict is Verdict.INCONCLUSIVE
    assert "noise floor" in reason
    assert decide(-0.05, -15.0, 0.1, 10.0, 20, 20)[0] is Verdict.INCONCLUSIVE


def test_real_but_small_change_passes():
    verdict, reason = decide(0.3, 4.0, 0.1, 10.0, 20, 20)
    assert verdict is Verdict.PASS
    assert "within the threshold" in reason


def test_no_change_passes():
    assert decide(0.0, 0.0, 0.1, 10.0, 20, 20)[0] is Verdict.PASS


def test_too_few_samples_is_inconclusive_even_for_huge_changes():
    assert decide(5.0, 500.0, 0.1, 10.0, MIN_SAMPLES - 1, 20)[0] is Verdict.INCONCLUSIVE
    assert decide(5.0, 500.0, 0.1, 10.0, 20, MIN_SAMPLES - 1)[0] is Verdict.INCONCLUSIVE
    assert decide(5.0, 500.0, 0.1, 10.0, MIN_SAMPLES, MIN_SAMPLES)[0] is Verdict.REGRESSION


def test_zero_threshold_flags_any_change_beyond_noise():
    assert decide(0.3, 0.5, 0.1, 0.0, 20, 20)[0] is Verdict.REGRESSION


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError):
        decide(0.3, 15.0, 0.1, -1.0, 20, 20)


# --- the noise floor -----------------------------------------------------------------


def test_standard_error_factors():
    assert standard_error_factor("mean") == 1.0
    assert standard_error_factor("p50") == pytest.approx(math.sqrt(math.pi / 2))  # 1.2533
    assert standard_error_factor("p95") == pytest.approx(2.113, abs=1e-3)
    with pytest.raises(ValueError):
        standard_error_factor("p99")


def test_noise_floor_formula_on_a_hand_checked_case():
    assert mad(SPREAD_10) == 1.0
    # sigma = 1.4826; se = 1.2533 * 1.4826 / sqrt(10) = 0.5876 per side;
    # combined = 0.5876 * sqrt(2) = 0.8310; floor = 3 * 0.8310 = 2.493
    assert noise_floor(SPREAD_10, SPREAD_10, "p50") == pytest.approx(2.493, abs=1e-3)


def test_noise_floor_shrinks_with_more_samples():
    small = noise_floor(SPREAD_10, SPREAD_10, "p50")
    large = noise_floor(SPREAD_10 * 4, SPREAD_10 * 4, "p50")  # 4x samples, same spread
    assert large == pytest.approx(small / 2)


def test_noise_floor_never_below_timer_resolution():
    constant = [1.0] * 20
    assert mad(constant) == 0.0
    assert noise_floor(constant, constant, "p50") == TIMER_RESOLUTION_MS


def test_noise_floor_ignores_a_single_outlier():
    with_outlier = [*SPREAD_10[:-1], 1000.0]
    assert noise_floor(with_outlier, SPREAD_10, "p50") == pytest.approx(
        noise_floor(SPREAD_10, SPREAD_10, "p50")
    )


def test_three_standard_errors_are_required():
    # floor for SPREAD_10 vs a shifted copy is 2.493 (= NOISE_Z standard errors)
    assert NOISE_Z == 3.0
    within = compare_series("kernel", SPREAD_10, [x + 2.0 for x in SPREAD_10], "p50", 10.0)
    beyond = compare_series("kernel", SPREAD_10, [x + 3.0 for x in SPREAD_10], "p50", 10.0)
    assert within.verdict is Verdict.INCONCLUSIVE  # +20%, but only ~2.4 standard errors
    assert beyond.verdict is Verdict.REGRESSION  # +30%, ~3.6 standard errors


# --- whole-series comparisons -------------------------------------------------------------


def test_threshold_boundary_with_exact_values():
    base, cur = [4.0] * 20, [4.5] * 20  # exactly +12.5%, zero spread
    assert compare_series("kernel", base, cur, "p50", 12.5).verdict is Verdict.PASS
    assert compare_series("kernel", base, cur, "p50", 12.4).verdict is Verdict.REGRESSION


def test_clear_regression_and_improvement():
    base = jitter(1.80, 0.01, 30)
    slower = compare_series("kernel", base, [t * 1.2 for t in base], "p50", 10.0)
    faster = compare_series("kernel", base, [t * 0.8 for t in base], "p50", 10.0)
    assert slower.verdict is Verdict.REGRESSION
    assert slower.percent == pytest.approx(20.0)
    assert slower.delta == pytest.approx(0.2 * slower.baseline)
    assert faster.verdict is Verdict.IMPROVEMENT


def test_noisy_measurements_are_inconclusive():
    base = jitter(2.40, 1.0, 20)
    cur = [t + 0.36 for t in jitter(2.40, 1.0, 20, phase=4)]  # +15%, but very noisy
    result = compare_series("gpu_total", base, cur, "p50", 10.0)
    assert result.percent > 10.0
    assert result.verdict is Verdict.INCONCLUSIVE


def test_p95_needs_a_larger_change_than_p50():
    base = jitter(2.0, 0.2, 20)
    cur = [t + 0.2 for t in base]
    p50 = compare_series("x", base, cur, "p50", 5.0)
    p95 = compare_series("x", base, cur, "p95", 5.0)
    assert p95.noise_floor > p50.noise_floor


def test_zero_baseline():
    assert percent_change(0.0, 0.0) == 0.0
    assert percent_change(0.0, 1.0) == math.inf
    zeros = [0.0] * 20
    assert compare_series("kernel", zeros, zeros, "p50", 10.0).verdict is Verdict.PASS
    grew = compare_series("kernel", zeros, [0.01] * 20, "p50", 10.0)
    assert grew.percent == math.inf and grew.verdict is Verdict.REGRESSION
    # Below the event timer's resolution: not measurable either way.
    tiny = compare_series("kernel", zeros, [0.0001] * 20, "p50", 10.0)
    assert tiny.verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize(
    "bad",
    [[1.0, math.nan] * 10, [1.0, math.inf] * 10, [1.0, -1.0] * 10, ["1.0"] * 20, [True] * 20, []],
)
def test_malformed_samples_are_rejected(bad):
    with pytest.raises(ValueError):
        compare_series("kernel", bad, [1.0] * 20, "p50", 10.0)
    with pytest.raises(ValueError):
        compare_series("kernel", [1.0] * 20, bad, "p50", 10.0)

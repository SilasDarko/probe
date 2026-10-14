"""Tests for probe.stats, including agreement with the C++ implementation via
the shared golden file (the C++ side reads the same file in ctest)."""

import math
from pathlib import Path

import pytest

from probe.stats import STAT_FIELDS, summarize

GOLDEN = Path(__file__).parent / "fixtures" / "stats_golden.txt"


def load_golden():
    cases = []
    for line in GOLDEN.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        samples_text, expected_text = line.split("|")
        samples = [float(t) for t in samples_text.split()]
        expected = dict(zip(STAT_FIELDS, (float(t) for t in expected_text.split()), strict=True))
        cases.append(pytest.param(samples, expected, id=samples_text.strip()[:40]))
    return cases


@pytest.mark.parametrize(("samples", "expected"), load_golden())
def test_matches_golden(samples, expected):
    stats = summarize(samples)
    assert stats["count"] == expected["count"]
    for field in ("min", "max", "mean", "p50", "p95"):
        assert stats[field] == pytest.approx(expected[field], rel=1e-12, abs=1e-12), field
    if math.isnan(expected["stddev"]):
        assert stats["stddev"] is None
    else:
        assert stats["stddev"] == pytest.approx(expected["stddev"], rel=1e-12, abs=1e-12)


def test_golden_file_has_enough_cases():
    assert len(load_golden()) >= 5


def test_percentile_is_linear_interpolation():
    # rank h = (5 - 1) * 0.95 = 3.8  ->  40 + 0.8 * (50 - 40) = 48
    assert summarize([10, 20, 30, 40, 50])["p95"] == pytest.approx(48.0)
    assert summarize([50, 10, 40, 20, 30])["p95"] == pytest.approx(48.0)  # order-independent


def test_single_sample():
    stats = summarize([2.5])
    assert stats == {
        "count": 1,
        "min": 2.5,
        "max": 2.5,
        "mean": 2.5,
        "p50": 2.5,
        "p95": 2.5,
        "stddev": None,
    }


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([], "no samples"),
        ([1.0, math.nan], "sample 1 is not a finite non-negative duration"),
        ([math.inf], "sample 0 is not a finite non-negative duration"),
        ([1.0, 2.0, -0.5], "sample 2 is not a finite non-negative duration"),
    ],
)
def test_rejects_malformed_samples(samples, message):
    with pytest.raises(ValueError, match=message):
        summarize(samples)

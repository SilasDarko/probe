"""Schema tests against real C++ output.

probe_example_report serializes synthetic data through the same probe_core
code path probe-workloads uses, so these tests exercise the actual C++ JSON
without needing a GPU.
"""

import json
import subprocess

import pytest

from probe.result_schema import RUN_SCHEMA_VERSION, check_run_result
from probe.stats import summarize


def emit(example_report_binary, mode):
    result = subprocess.run(
        [str(example_report_binary), mode], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(result.stdout)


def test_valid_example_matches_schema(example_report_binary):
    doc = emit(example_report_binary, "valid")
    assert check_run_result(doc) == []
    assert doc["schema_version"] == RUN_SCHEMA_VERSION
    assert doc["valid"] is True
    assert doc["timing"]["gpu"]["phase_order"] == ["h2d", "kernel", "d2h"]


def test_cpp_statistics_agree_with_python(example_report_binary):
    doc = emit(example_report_binary, "valid")
    gpu = doc["timing"]["gpu"]
    series = {
        f"gpu.{name}": (gpu["phases"][name], doc["samples"]["gpu"][name]) for name in gpu["phases"]
    }
    series["gpu.total"] = (gpu["total"], doc["samples"]["gpu"]["total"])
    series["host.end_to_end"] = (
        doc["timing"]["host"]["end_to_end"],
        doc["samples"]["host"]["end_to_end"],
    )
    for label, (cpp_stats, raw) in series.items():
        python_stats = summarize(raw)
        for field, value in python_stats.items():
            assert cpp_stats[field] == pytest.approx(value, rel=1e-12), f"{label}.{field}"


def test_single_iteration_has_undefined_stddev(example_report_binary):
    doc = emit(example_report_binary, "single")
    assert check_run_result(doc) == []
    assert doc["timing"]["gpu"]["phases"]["kernel"]["stddev"] is None
    assert doc["timing"]["gpu"]["phases"]["kernel"]["count"] == 1


def test_invalid_example_is_marked_and_explained(example_report_binary):
    doc = emit(example_report_binary, "invalid")
    assert check_run_result(doc) == []
    assert doc["valid"] is False
    assert doc["validation"]["passed"] is False
    assert doc["validation"]["first_failed_iteration"] == 3
    assert doc["invalid_reasons"] == ["validation failed in 1 of 6 checked iterations"]


def test_checker_rejects_wrong_version_and_inconsistency(example_report_binary):
    doc = emit(example_report_binary, "valid")
    doc["schema_version"] = 1
    assert any("schema_version" in p for p in check_run_result(doc))

    doc = emit(example_report_binary, "valid")
    doc["invalid_reasons"] = ["something"]
    assert "valid result lists invalid_reasons" in check_run_result(doc)

    doc = emit(example_report_binary, "valid")
    del doc["timing"]["gpu"]["phases"]["kernel"]["p95"]
    assert any("timing.gpu.phases.kernel" in p for p in check_run_result(doc))

    assert check_run_result([]) == ["result is not a JSON object"]

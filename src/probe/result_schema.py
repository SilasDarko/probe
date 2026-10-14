"""Structural and consistency checks for ``probe.run`` result documents.

The schema is produced by probe_core (report.hpp); the Python CLI adds a
``provenance`` section and ``environment.host``. This module checks the
structure Probe relies on and, for results that carry raw samples, that the
stored statistics really follow from those samples.
"""

import math
from typing import Any

from probe.stats import STAT_FIELDS, summarize

RUN_SCHEMA = "probe.run"
RUN_SCHEMA_VERSION = 2

# Metric names used by the CLI: the workload's GPU phases plus these two.
GPU_TOTAL = "gpu_total"
HOST_END_TO_END = "host_end_to_end"

_TOP_LEVEL_KEYS = (
    "schema",
    "schema_version",
    "workload",
    "valid",
    "invalid_reasons",
    "parameters",
    "config",
    "build",
    "environment",
    "validation",
    "timing",
)


def _check_stats(label: str, stats: Any, problems: list[str]) -> None:
    if stats is None:
        return  # allowed only in invalid results; checked by the caller
    if not isinstance(stats, dict) or set(stats) != set(STAT_FIELDS):
        problems.append(f"{label}: expected statistics with keys {STAT_FIELDS}")


def check_run_result(doc: Any) -> list[str]:
    """Return a list of structural problems; empty if the document is usable."""
    if not isinstance(doc, dict):
        return ["result is not a JSON object"]
    problems = [f"missing key '{k}'" for k in _TOP_LEVEL_KEYS if k not in doc]
    if problems:
        return problems

    try:
        if doc["schema"] != RUN_SCHEMA:
            problems.append(f"schema is {doc['schema']!r}, expected {RUN_SCHEMA!r}")
        if doc["schema_version"] != RUN_SCHEMA_VERSION:
            problems.append(
                f"schema_version is {doc['schema_version']!r}, expected {RUN_SCHEMA_VERSION}"
            )
            return problems  # a different version may lay out everything else differently

        gpu = doc["timing"]["gpu"]
        host = doc["timing"]["host"]
        order = gpu["phase_order"]
        phases = gpu["phases"]
        if list(phases) != list(order):
            problems.append("timing.gpu.phases does not match timing.gpu.phase_order")
        for name in order:
            _check_stats(f"timing.gpu.phases.{name}", phases.get(name), problems)
        _check_stats("timing.gpu.total", gpu.get("total"), problems)
        _check_stats("timing.host.end_to_end", host.get("end_to_end"), problems)

        for key in ("warmup", "iterations", "seed"):
            if not isinstance(doc["config"].get(key), int):
                problems.append(f"config.{key} must be an integer")
        if not isinstance(doc["parameters"], dict):
            problems.append("parameters must be an object")

        if doc["valid"] is not True and doc["valid"] is not False:
            problems.append("valid must be true or false")
        if doc["valid"] and doc["invalid_reasons"]:
            problems.append("valid result lists invalid_reasons")
        if not doc["valid"] and not doc["invalid_reasons"]:
            problems.append("invalid result gives no reason")
        if doc["valid"] and not doc["validation"].get("passed"):
            problems.append("result marked valid but validation did not pass")
    except (KeyError, TypeError, AttributeError) as exc:
        problems.append(f"malformed result: {type(exc).__name__}: {exc}")
    return problems


def metric_names(doc: dict) -> list[str]:
    return [*doc["timing"]["gpu"]["phase_order"], GPU_TOTAL, HOST_END_TO_END]


def raw_series(doc: dict) -> dict[str, list]:
    """Raw per-iteration samples keyed by metric name."""
    raw = doc["samples"]
    series = {name: raw["gpu"][name] for name in doc["timing"]["gpu"]["phase_order"]}
    series[GPU_TOTAL] = raw["gpu"]["total"]
    series[HOST_END_TO_END] = raw["host"]["end_to_end"]
    return series


def stored_stats(doc: dict) -> dict[str, Any]:
    """Aggregated statistics keyed by metric name, as stored in the document."""
    gpu = doc["timing"]["gpu"]
    stats = {name: gpu["phases"][name] for name in gpu["phase_order"]}
    stats[GPU_TOTAL] = gpu["total"]
    stats[HOST_END_TO_END] = doc["timing"]["host"]["end_to_end"]
    return stats


def _same(stored: Any, recomputed: float | int | None) -> bool:
    if recomputed is None:
        return stored is None
    if isinstance(stored, bool) or not isinstance(stored, int | float):
        return False
    return math.isclose(stored, recomputed, rel_tol=1e-9, abs_tol=1e-12)


def check_raw_samples(doc: dict) -> list[str]:
    """Check that raw samples exist for every metric, have one entry per
    measured iteration, contain only finite non-negative numbers, and
    reproduce the stored statistics. A mismatch means the data is corrupted or
    internally inconsistent; it does not say how it got that way."""
    if "samples" not in doc:
        return ["result has no raw samples (re-run with raw samples recorded)"]
    try:
        series = raw_series(doc)
        stored = stored_stats(doc)
        iterations = doc["config"]["iterations"]
    except (KeyError, TypeError) as exc:
        return [f"raw samples incomplete: missing {exc}"]

    problems = []
    for name, samples in series.items():
        if not isinstance(samples, list) or len(samples) != iterations:
            problems.append(f"{name}: expected {iterations} raw samples")
            continue
        try:
            recomputed = summarize(samples)
        except ValueError as exc:
            problems.append(f"{name}: {exc}")
            continue
        if not isinstance(stored[name], dict):
            problems.append(f"{name}: stored statistics missing")
            continue
        for field in STAT_FIELDS:
            if not _same(stored[name].get(field), recomputed[field]):
                problems.append(
                    f"{name}.{field}: stored {stored[name].get(field)!r} does not match "
                    f"{recomputed[field]!r} recomputed from raw samples"
                )
    return problems

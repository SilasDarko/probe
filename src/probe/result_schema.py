"""Structural checks for ``probe.run`` result documents.

The schema is produced by probe_core (report.hpp). This module only checks the
structure Probe relies on; it does not re-validate the measurements.
"""

from typing import Any

from probe.stats import STAT_FIELDS

RUN_SCHEMA = "probe.run"
RUN_SCHEMA_VERSION = 2

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

    if doc["schema"] != RUN_SCHEMA:
        problems.append(f"schema is {doc['schema']!r}, expected {RUN_SCHEMA!r}")
    if doc["schema_version"] != RUN_SCHEMA_VERSION:
        problems.append(
            f"schema_version is {doc['schema_version']!r}, expected {RUN_SCHEMA_VERSION}"
        )

    gpu = doc["timing"].get("gpu", {})
    host = doc["timing"].get("host", {})
    order = gpu.get("phase_order", [])
    phases = gpu.get("phases", {})
    if list(phases) != list(order):
        problems.append("timing.gpu.phases does not match timing.gpu.phase_order")
    for name in order:
        _check_stats(f"timing.gpu.phases.{name}", phases.get(name), problems)
    _check_stats("timing.gpu.total", gpu.get("total"), problems)
    _check_stats("timing.host.end_to_end", host.get("end_to_end"), problems)

    if doc["valid"] and doc["invalid_reasons"]:
        problems.append("valid result lists invalid_reasons")
    if not doc["valid"] and not doc["invalid_reasons"]:
        problems.append("invalid result gives no reason")
    if doc["valid"] and not doc["validation"].get("passed"):
        problems.append("result marked valid but validation did not pass")
    return problems

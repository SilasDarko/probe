"""Synthetic result documents for CPU-only tests.

Every value here is invented test data, labeled as such in the device name.
None of it is a measurement and none of it may appear in benchmark results.
"""

import copy

from probe.stats import summarize

SYNTHETIC_GPU = "SYNTHETIC TEST GPU (not a measurement)"


def jitter(center: float, spread: float, n: int, phase: int = 0) -> list[float]:
    """Deterministic samples spread evenly over [center - spread, center + spread]."""
    return [center + spread * (((i * 7 + phase) % 21) - 10) / 10 for i in range(n)]


def default_phases(iterations: int, kernel_scale: float = 1.0) -> dict[str, list[float]]:
    return {
        "h2d": jitter(0.40, 0.004, iterations, 1),
        "kernel": [t * kernel_scale for t in jitter(1.80, 0.009, iterations, 3)],
        "d2h": jitter(0.20, 0.002, iterations, 5),
    }


def make_result(
    *,
    workload: str = "vector_add",
    phases: dict[str, list[float]] | None = None,
    iterations: int = 20,
    valid: bool = True,
    n: int = 1048576,
    gpu: str = SYNTHETIC_GPU,
    build_type: str = "Release",
    driver: str = "12.2",
    commit: str = "0123456789abcdef0123456789abcdef01234567",
) -> dict:
    """A complete, internally consistent result document, as `probe run`
    would write it: statistics are computed from the raw samples."""
    if phases is None:
        phases = default_phases(iterations)
    iterations = len(next(iter(phases.values())))
    total = [sum(values) for values in zip(*phases.values(), strict=True)]
    host = [t + 0.05 for t in total]

    doc = {
        "schema": "probe.run",
        "schema_version": 2,
        "workload": workload,
        "valid": valid,
        "invalid_reasons": [] if valid else ["validation failed in 1 of 20 checked iterations"],
        "parameters": {"n": n, "block_size": 256, "dtype": "float32"},
        "config": {"warmup": 3, "iterations": iterations, "seed": 42},
        "build": {
            "probe_version": "0.1.0",
            "build_type": build_type,
            "cuda_architectures": "75-real,80-real,86-real,89-real,90",
            "cuda_compiler": "NVIDIA 12.0.140",
            "cxx_compiler": "GNU 13.3.0",
            "cuda_flags": "-O3 -DNDEBUG",
        },
        "environment": {
            "device": {
                "index": 0,
                "count": 1,
                "name": gpu,
                "compute_capability": "8.6",
                "sm_count": 1,
                "global_mem_bytes": 1,
                "cuda_runtime_version": "12.0",
                "cuda_driver_version": driver,
            },
            "host": {
                "os": "Linux",
                "os_release": "test",
                "machine": "x86_64",
                "cpu_model": "synthetic",
                "python": "3.13",
            },
        },
        "validation": {
            "passed": valid,
            "method": "synthetic",
            "rtol": 0,
            "atol": 0,
            "iterations_checked": iterations,
            "iterations_failed": 0 if valid else 1,
            "total_mismatches": 0 if valid else 1,
            "max_abs_error": 0,
            "first_failed_iteration": -1 if valid else 0,
        },
        "timing": {
            "unit": "ms",
            "gpu": {
                "method": "cuda_events",
                "phase_order": list(phases),
                "phases": {name: summarize(values) for name, values in phases.items()},
                "total": summarize(total),
            },
            "host": {"method": "steady_clock", "end_to_end": summarize(host)},
        },
        "samples": {
            "gpu": {**copy.deepcopy(phases), "total": total},
            "host": {"end_to_end": host},
        },
        "provenance": {
            "probe_cli_version": "0.1.0",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "command": ["probe", "run", workload.replace("_", "-")],
            "git_commit": commit,
            "git_dirty": False,
        },
    }
    return doc

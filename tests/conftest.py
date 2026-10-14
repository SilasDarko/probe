"""Shared pytest configuration.

Tests marked ``@pytest.mark.gpu`` need real NVIDIA hardware. They are skipped,
not failed, when no GPU is detected, so the CPU-only suite runs unchanged on
ordinary CI runners.
"""

import os
from pathlib import Path

import pytest

from probe.hardware import nvidia_gpu_available

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("PROBE_BUILD_DIR", REPO_ROOT / "build"))
DEFAULT_BINARY = BUILD_DIR / "workloads" / "probe-workloads"


def pytest_collection_modifyitems(config, items):
    if nvidia_gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="no NVIDIA GPU detected (nvidia-smi -L)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def workloads_binary() -> Path:
    """Path to the built probe-workloads binary; skips if it has not been built."""
    override = os.environ.get("PROBE_WORKLOADS_BIN")
    path = Path(override) if override else DEFAULT_BINARY
    if not path.is_file():
        pytest.skip(f"probe-workloads not built at {path}")
    return path


@pytest.fixture
def example_report_binary() -> Path:
    """CPU-only helper that prints a result JSON from synthetic data."""
    path = BUILD_DIR / "workloads" / "probe_example_report"
    if not path.is_file():
        pytest.skip(f"probe_example_report not built at {path}")
    return path

"""Contract tests for the probe-workloads executable.

These need the binary to be built (they skip otherwise). Only the device-info
test needs an actual GPU.
"""

import json
import subprocess

import pytest

from probe.hardware import nvidia_gpu_available

EXIT_USAGE = 2
EXIT_NO_DEVICE = 3


def run(binary, *args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=60)


def test_version_reports_build_type(workloads_binary):
    result = run(workloads_binary, "--version")
    assert result.returncode == 0
    assert result.stdout.startswith("probe-workloads 0.1.0 (")


def test_unknown_flag_is_a_usage_error(workloads_binary):
    assert run(workloads_binary, "--bogus").returncode == EXIT_USAGE


@pytest.mark.skipif(nvidia_gpu_available(), reason="checks the no-GPU path")
def test_device_info_without_gpu_exits_cleanly(workloads_binary):
    result = run(workloads_binary, "--device-info")
    assert result.returncode == EXIT_NO_DEVICE
    assert result.stdout == ""
    assert "no usable CUDA device" in result.stderr


@pytest.mark.gpu
def test_device_info_reports_gpu(workloads_binary):
    result = run(workloads_binary, "--device-info")
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["schema"] == "probe.device_info/1"
    assert info["device_count"] >= 1
    assert info["name"]
    major, minor = (int(part) for part in info["compute_capability"].split("."))
    assert major >= 5 and minor >= 0
    assert info["global_mem_bytes"] > 0

"""Contract tests for the probe-workloads executable.

All tests need the binary to be built (they skip otherwise). Argument
validation and the no-GPU path run on any machine; tests marked ``gpu`` need
real NVIDIA hardware.
"""

import json
import subprocess

import pytest

from probe.hardware import nvidia_gpu_available
from probe.result_schema import check_run_result
from probe.stats import summarize

EXIT_USAGE = 2
EXIT_NO_DEVICE = 3

WORKLOADS = ("vector_add", "reduction")

no_gpu_only = pytest.mark.skipif(nvidia_gpu_available(), reason="checks the no-GPU path")


def run(binary, *args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=300)


def run_json(binary, *args):
    result = run(binary, *args, "--json", "--raw-samples")
    assert result.returncode == 0, result.stdout + result.stderr
    doc = json.loads(result.stdout)
    assert check_run_result(doc) == []
    return doc


# --- any machine ---------------------------------------------------------------


def test_version_reports_build_and_architectures(workloads_binary):
    result = run(workloads_binary, "--version")
    assert result.returncode == 0
    assert result.stdout.startswith("probe-workloads 0.1.0 (")
    assert "cuda architectures" in result.stdout


def test_unknown_command_is_a_usage_error(workloads_binary):
    assert run(workloads_binary, "--bogus").returncode == EXIT_USAGE
    assert run(workloads_binary, "no_such_workload").returncode == EXIT_USAGE


@pytest.mark.parametrize("workload", WORKLOADS)
@pytest.mark.parametrize(
    "args",
    [
        ["--n", "0"],
        ["--n", "abc"],
        ["--n"],
        ["--iters", "0"],
        ["--block-size", "100"],
        ["--unknown-flag"],
    ],
)
def test_rejects_bad_arguments(workloads_binary, workload, args):
    # Arguments are validated before the GPU is touched, so this behaves the
    # same with or without a GPU.
    result = run(workloads_binary, workload, *args)
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""
    assert result.stderr.startswith(f"{workload}: ")


def test_reduction_requires_power_of_two_block(workloads_binary):
    result = run(workloads_binary, "reduction", "--block-size", "96")
    assert result.returncode == EXIT_USAGE
    assert "power of two" in result.stderr


@no_gpu_only
def test_device_info_without_gpu_exits_cleanly(workloads_binary):
    result = run(workloads_binary, "--device-info")
    assert result.returncode == EXIT_NO_DEVICE
    assert result.stdout == ""
    assert "no usable CUDA device" in result.stderr


@no_gpu_only
@pytest.mark.parametrize("workload", WORKLOADS)
def test_workload_without_gpu_exits_cleanly(workloads_binary, workload):
    result = run(workloads_binary, workload, "--n", "1024", "--json")
    assert result.returncode == EXIT_NO_DEVICE
    assert result.stdout == ""  # no partial or fabricated JSON
    assert "no usable CUDA device" in result.stderr


# --- GPU required ----------------------------------------------------------------


@pytest.mark.gpu
def test_device_info_reports_gpu(workloads_binary):
    info = json.loads(run(workloads_binary, "--device-info").stdout)
    assert info["schema"] == "probe.device_info/1"
    device = info["device"]
    assert device["count"] >= 1
    assert device["name"]
    major, _minor = (int(part) for part in device["compute_capability"].split("."))
    assert major >= 7
    assert device["global_mem_bytes"] > 0


def check_timing(doc, iters, phases):
    gpu = doc["timing"]["gpu"]
    assert gpu["method"] == "cuda_events"
    assert gpu["phase_order"] == phases
    raw = doc["samples"]
    for name in phases:
        assert len(raw["gpu"][name]) == iters
        assert all(t >= 0 for t in raw["gpu"][name])
    assert len(raw["host"]["end_to_end"]) == iters

    for i in range(iters):
        # Phases are contiguous intervals on one stream, so they add up to
        # the GPU total, up to event-timer rounding.
        parts = sum(raw["gpu"][name][i] for name in phases)
        assert parts == pytest.approx(raw["gpu"]["total"][i], abs=0.01)
    # host.end_to_end is recorded but deliberately not checked against
    # gpu.total: the two clocks measure different things, and how they relate
    # is to be studied on real hardware before anything is asserted about it.
    assert all(t >= 0 for t in raw["host"]["end_to_end"])

    # The C++ statistics agree with an independent Python computation.
    for name in phases:
        assert gpu["phases"][name]["p50"] == pytest.approx(summarize(raw["gpu"][name])["p50"])


@pytest.mark.gpu
def test_vector_add_valid_and_complete(workloads_binary):
    iters = 5
    doc = run_json(workloads_binary, "vector_add", "--n", "1048576", "--iters", str(iters))
    assert doc["workload"] == "vector_add"
    assert doc["valid"] is True
    assert doc["validation"]["iterations_checked"] == iters
    assert doc["validation"]["max_abs_error"] == 0.0
    check_timing(doc, iters, ["h2d", "kernel", "d2h"])


@pytest.mark.gpu
def test_vector_add_size_not_a_multiple_of_block(workloads_binary):
    # 1,000,003 is prime, so the last block is partially full: this exercises
    # the bounds check and the tail of the output.
    doc = run_json(workloads_binary, "vector_add", "--n", "1000003", "--iters", "3")
    assert doc["valid"] is True
    assert doc["parameters"]["grid_size"] == (1000003 + 255) // 256


@pytest.mark.gpu
def test_raw_samples_are_opt_in(workloads_binary):
    result = run(workloads_binary, "vector_add", "--n", "4096", "--iters", "2", "--json")
    assert result.returncode == 0
    assert "samples" not in json.loads(result.stdout)


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 2, 255, 257, 1000, 4097, 65537, 1000003])
def test_reduction_awkward_sizes(workloads_binary, n):
    iters = 3
    doc = run_json(workloads_binary, "reduction", "--n", str(n), "--iters", str(iters))
    assert doc["workload"] == "reduction"
    assert doc["valid"] is True, doc["invalid_reasons"]
    assert doc["validation"]["max_abs_error"] <= doc["validation"]["atol"]
    check_timing(doc, iters, ["h2d", "kernel", "d2h"])

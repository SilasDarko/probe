"""Contract tests for the probe-workloads executable.

All tests need the binary to be built (they skip otherwise). Argument
validation and the no-GPU path run on any machine; tests marked ``gpu`` need
real NVIDIA hardware.
"""

import json
import subprocess

import pytest

from probe.hardware import nvidia_gpu_available

EXIT_USAGE = 2
EXIT_NO_DEVICE = 3

no_gpu_only = pytest.mark.skipif(nvidia_gpu_available(), reason="checks the no-GPU path")


def run(binary, *args):
    return subprocess.run([str(binary), *args], capture_output=True, text=True, timeout=300)


def run_json(binary, *args):
    result = run(binary, *args, "--json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# --- any machine ---------------------------------------------------------------


def test_version_reports_build_and_architectures(workloads_binary):
    result = run(workloads_binary, "--version")
    assert result.returncode == 0
    assert result.stdout.startswith("probe-workloads 0.1.0 (")
    assert "cuda architectures" in result.stdout


def test_unknown_command_is_a_usage_error(workloads_binary):
    assert run(workloads_binary, "--bogus").returncode == EXIT_USAGE
    assert run(workloads_binary, "no_such_workload").returncode == EXIT_USAGE


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
def test_vector_add_rejects_bad_arguments(workloads_binary, args):
    # Arguments are validated before the GPU is touched, so this behaves the
    # same with or without a GPU.
    result = run(workloads_binary, "vector_add", *args)
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""
    assert result.stderr.startswith("vector_add: ")


@no_gpu_only
def test_device_info_without_gpu_exits_cleanly(workloads_binary):
    result = run(workloads_binary, "--device-info")
    assert result.returncode == EXIT_NO_DEVICE
    assert result.stdout == ""
    assert "no usable CUDA device" in result.stderr


@no_gpu_only
def test_vector_add_without_gpu_exits_cleanly(workloads_binary):
    result = run(workloads_binary, "vector_add", "--n", "1024", "--json")
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


@pytest.mark.gpu
def test_vector_add_json_is_valid_and_complete(workloads_binary):
    iters = 5
    run_result = run_json(
        workloads_binary, "vector_add", "--n", "1048576", "--warmup", "2", "--iters", str(iters)
    )
    assert run_result["schema"] == "probe.run/1"
    assert run_result["workload"] == "vector_add"
    assert run_result["valid"] is True
    assert run_result["validation"]["passed"] is True
    assert run_result["validation"]["iterations_checked"] == iters
    assert run_result["validation"]["max_abs_error"] == 0.0

    timing = run_result["timing"]
    assert timing["method"] == "cuda_events"
    samples = timing["samples"]
    for phase in ("h2d", "kernel", "d2h", "total"):
        assert len(samples[phase]) == iters
        assert all(t > 0 for t in samples[phase])

    # The three phases are contiguous intervals on one stream, so they should
    # add up to the total, up to event-timer rounding.
    for i in range(iters):
        parts = samples["h2d"][i] + samples["kernel"][i] + samples["d2h"][i]
        assert parts == pytest.approx(samples["total"][i], abs=0.01)


@pytest.mark.gpu
def test_vector_add_size_not_a_multiple_of_block(workloads_binary):
    # 1,000,003 is prime, so the last block is partially full: this exercises
    # the bounds check and the tail of the output.
    run_result = run_json(
        workloads_binary, "vector_add", "--n", "1000003", "--block-size", "256", "--iters", "3"
    )
    assert run_result["valid"] is True
    assert run_result["parameters"]["grid_size"] == (1000003 + 255) // 256

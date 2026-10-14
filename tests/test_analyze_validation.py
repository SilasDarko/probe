"""Tests for scripts/analyze_validation.py on a synthetic run directory.

Every document here is synthetic test data (see tests/synthetic.py). The test
checks that the analysis computes the relationships it reports, not any
property of real hardware.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from synthetic import SYNTHETIC_GPU, default_phases, make_result

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_validation.py"


@pytest.fixture(scope="module")
def analyze():
    spec = importlib.util.spec_from_file_location("analyze_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(directory: Path, steps: dict[str, tuple[int | None, str]]) -> None:
    entries = []
    for name, (code, stdout) in steps.items():
        base = directory / name
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{base}.stdout").write_text(stdout)
        Path(f"{base}.stderr").write_text("")
        entries.append(
            {"name": name, "argv": [], "cwd": "", "extra_env": {}, "exit_code": code,
             "note": None, "seconds": 0.1, "started_utc": ""}
        )  # fmt: skip
    (directory / "steps.json").write_text(json.dumps(entries))
    (directory / "meta.json").write_text(json.dumps({"quick": False, "env": {}}))


@pytest.fixture
def report(tmp_path, analyze):
    doc = make_result()
    reduction = make_result(workload="reduction")
    reduction["validation"]["max_abs_error"] = 1e-4
    reduction["validation"]["atol"] = 1e-3
    reduction["parameters"]["passes"] = 2

    slow_start = default_phases(30)
    slow_start["kernel"][0] *= 3.0  # first measured iteration three times slower
    warmup0 = make_result(phases=slow_start)
    warmup0["config"]["warmup"] = 0

    device = {
        "schema": "probe.device_info/1",
        "device": copy.deepcopy(doc["environment"]["device"]),
    }
    write_run(
        tmp_path,
        {
            "env/nvidia-smi": (0, "| Driver Version: 550.00   CUDA Version: 12.2 |"),
            "env/nvidia-smi-query": (
                0,
                "name, compute_cap, driver_version, memory.total [MiB]\n"
                f"{SYNTHETIC_GPU}, 8.6, 550.00, 1\n",
            ),
            "env/nvcc": (0, "Cuda compilation tools, release 12.0, V12.0.140"),
            "env/git-commit": (0, "abc123\n"),
            "build/cuobjdump": (
                0,
                "ELF 1: x.sm_75.cubin\nELF 2: x.sm_86.cubin\nPTX 1: x.sm_90.ptx",
            ),
            "device/device-info": (0, json.dumps(device)),
            "vector_add/n33": (0, json.dumps(doc)),
            "vector_add/n1": (3, ""),
            "reduction/prime-n1009": (0, json.dumps(reduction)),
            "sanitizer/memcheck-vector_add-n33": (0, "========= ERROR SUMMARY: 0 errors\n"),
            "timing/vector_add": (0, json.dumps(doc)),
            "warmup/vector_add-w0": (0, json.dumps(warmup0)),
            "noise/vector_add-run1": (0, json.dumps(doc)),
            "noise/vector_add-run2": (0, json.dumps(doc)),
            "noise/vector_add-run3": (0, json.dumps(doc)),
            "workflow/vec-compare-a": (0, "+0.0% -> PASS: no change beyond the noise floor"),
        },
    )
    return analyze.report(tmp_path)


def test_step_failures_are_listed(report):
    assert "1 with a non-zero or missing exit code" in report
    assert "| vector_add/n1 | exit 3 |" in report


def test_environment_comparison(report):
    assert f"| GPU name | {SYNTHETIC_GPU} | {SYNTHETIC_GPU} | yes |" in report
    assert "| compute capability | 8.6 | 8.6 | yes |" in report
    assert "| CUDA version supported by driver | 12.2 | 12.2 | yes |" in report
    assert "| NVIDIA driver build | 550.00 | not recorded by Probe | GAP |" in report
    assert "| CUDA runtime / toolkit | 12.0 | 12.0 | yes |" in report


def test_architecture_selection(report):
    assert "native `sm_86` SASS is present" in report


def test_vector_add_and_sanitizer_rows(report):
    assert "| 33 | exit 0 | True | 20 | 0 | 0 |" in report
    assert "| memcheck-vector_add-n33 | exit 0 | ERROR SUMMARY: 0 errors |" in report


def test_reduction_error_versus_bound(report):
    # error / bound = 0.1; 0.5 - 1e-4 > 1e-3, so a dropped element is detectable
    assert "| prime | 1009 | exit 0 | True | 2 | 1.00e-04 | 1.00e-03 | 0.1 | yes |" in report


def test_timing_relationships(report):
    # Synthetic data: gpu_total is exactly the sum of phases and host is
    # exactly 0.05 ms (50 us) above it.
    assert "Sum of phases minus GPU total (us): min 0.0, median 0.0, max 0.0" in report
    assert "Host end-to-end minus GPU total (us): min 50.0, median 50.0" in report
    assert "0 of 20 iterations negative" in report


def test_warmup_ratio(report, analyze):
    assert "| vector_add-w0 | exit 0 | 0 |" in report
    row = next(line for line in report.splitlines() if line.startswith("| vector_add-w0"))
    assert float(row.split("|")[-2]) > 2.0  # first iteration about 3x the kernel time


def test_identical_runs_are_an_aa_pass(report):
    assert "Between-run SD of the medians: 0.0 us" in report
    assert "0 of 3 unchanged pairs differ by more than the noise floor; verdicts: PASS 3." in report


def test_workflow_verdicts(report):
    assert "| vec-compare-a | exit 0 | PASS: no change beyond the noise floor |" in report


def test_empty_directory_does_not_crash(tmp_path, analyze):
    text = analyze.report(tmp_path)
    assert "0 steps recorded" in text

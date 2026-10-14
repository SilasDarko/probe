"""CLI behavior and exit codes (probe.cli), using synthetic result files.

Comparisons use --current FILE, so no GPU is needed. Tests that invoke the
real probe-workloads binary check only what this machine can show.
"""

import json
import subprocess
import sys

import pytest

from probe import exit_codes
from probe.cli import main
from probe.hardware import nvidia_gpu_available

from synthetic import default_phases, jitter, make_result


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run every CLI call in an empty directory: .probe/ and results/ land there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROBE_STORE", raising=False)
    return tmp_path


def write(path, doc):
    path.write_text(json.dumps(doc))
    return path


def save(workdir, name="main", **kwargs):
    path = write(workdir / f"{name}-input.json", make_result(**kwargs))
    assert main(["baseline", "save", "--name", name, "--from", str(path)]) == exit_codes.OK


def compare(workdir, current_doc, *extra):
    path = write(workdir / "current.json", current_doc)
    return main(["compare", "vector-add", "--baseline", "main", "--current", str(path), *extra])


# --- baseline commands ---------------------------------------------------------------


def test_baseline_save_list_show_delete(workdir, capsys):
    save(workdir)
    assert (workdir / ".probe" / "baselines" / "main.json").is_file()

    assert main(["baseline", "list"]) == exit_codes.OK
    out = capsys.readouterr().out
    assert "main" in out and "vector_add" in out

    assert main(["baseline", "show", "main"]) == exit_codes.OK
    out = capsys.readouterr().out
    assert "result: VALID" in out and "kernel" in out

    assert main(["baseline", "delete", "main"]) == exit_codes.OK
    assert main(["baseline", "show", "main"]) == exit_codes.BAD_FILE


def test_store_option_and_environment_variable(workdir, monkeypatch):
    path = write(workdir / "r.json", make_result())
    assert main(["baseline", "save", "--name", "a", "--from", str(path), "--store", "s1"]) == 0
    assert (workdir / "s1" / "baselines" / "a.json").is_file()
    monkeypatch.setenv("PROBE_STORE", str(workdir / "s2"))
    assert main(["baseline", "save", "--name", "b", "--from", str(path)]) == 0
    assert (workdir / "s2" / "baselines" / "b.json").is_file()


def test_save_refuses_invalid_run(workdir, capsys):
    path = write(workdir / "bad.json", make_result(valid=False))
    code = main(["baseline", "save", "--name", "bad", "--from", str(path)])
    assert code == exit_codes.INVALID_RESULT
    assert "invalid run" in capsys.readouterr().err
    assert not (workdir / ".probe" / "baselines" / "bad.json").exists()


def test_save_checks_workload_of_file(workdir):
    path = write(workdir / "r.json", make_result(workload="reduction"))
    code = main(["baseline", "save", "vector-add", "--name", "x", "--from", str(path)])
    assert code == exit_codes.INCOMPATIBLE


def test_save_argument_errors(workdir):
    assert main(["baseline", "save", "--name", "x"]) == exit_codes.USAGE  # nothing to save
    path = write(workdir / "r.json", make_result())
    code = main(["baseline", "save", "--name", "x", "--from", str(path), "--size", "10"])
    assert code == exit_codes.USAGE
    assert main(["baseline", "save", "--name", "../x", "--from", str(path)]) == exit_codes.USAGE


def test_save_from_missing_or_malformed_file(workdir):
    assert main(["baseline", "save", "--name", "x", "--from", "nope.json"]) == exit_codes.BAD_FILE
    (workdir / "junk.json").write_text("[1, 2")
    assert main(["baseline", "save", "--name", "x", "--from", "junk.json"]) == exit_codes.BAD_FILE


def test_list_reports_unreadable_baselines(workdir, capsys):
    save(workdir)
    (workdir / ".probe" / "baselines" / "broken.json").write_text("{}")
    assert main(["baseline", "list"]) == exit_codes.BAD_FILE
    assert "UNREADABLE" in capsys.readouterr().out


# --- compare: verdicts and exit codes ------------------------------------------------------


def test_compare_unchanged_passes(workdir, capsys):
    save(workdir)
    assert compare(workdir, make_result(), "--fail-if-regression", "10") == exit_codes.OK
    out = capsys.readouterr().out
    assert "-> PASS" in out
    for label in ("H2D", "kernel", "D2H", "GPU total", "host end-to-end"):
        assert label in out


def test_compare_regression_fails_only_when_asked(workdir, capsys):
    save(workdir)
    slower = make_result(phases=default_phases(20, kernel_scale=1.2))  # kernel +20%
    assert compare(workdir, slower) == exit_codes.OK  # informational without a threshold
    assert "-> REGRESSION" in capsys.readouterr().out
    assert compare(workdir, slower, "--fail-if-regression", "10") == exit_codes.REGRESSION
    # GPU total rose by about 15%: a 20% threshold is not exceeded.
    assert compare(workdir, slower, "--fail-if-regression", "20") == exit_codes.OK


def test_compare_gates_on_selected_metric(workdir, capsys):
    save(workdir)
    slower = make_result(phases=default_phases(20, kernel_scale=1.2))
    code = compare(workdir, slower, "--metric", "kernel", "--fail-if-regression", "19")
    assert code == exit_codes.REGRESSION  # kernel itself rose by 20%
    out = capsys.readouterr().out
    assert "* kernel" in out
    assert compare(workdir, slower, "--metric", "d2h", "--fail-if-regression", "1") == 0


def test_compare_improvement_does_not_fail(workdir, capsys):
    save(workdir)
    faster = make_result(phases=default_phases(20, kernel_scale=0.8))
    assert compare(workdir, faster, "--fail-if-regression", "5") == exit_codes.OK
    assert "-> IMPROVEMENT" in capsys.readouterr().out


def test_compare_inconclusive(workdir, capsys):
    noisy = {
        "h2d": jitter(0.40, 0.3, 20),
        "kernel": jitter(1.80, 1.5, 20, 3),
        "d2h": jitter(0.20, 0.15, 20, 5),
    }
    noisier_and_slower = {name: [t * 1.15 for t in v] for name, v in noisy.items()}
    path = write(workdir / "base.json", make_result(phases=noisy))
    assert main(["baseline", "save", "--name", "main", "--from", str(path)]) == 0

    current = make_result(phases=noisier_and_slower)
    assert compare(workdir, current, "--fail-if-regression", "10") == exit_codes.OK
    assert "-> INCONCLUSIVE" in capsys.readouterr().out
    code = compare(workdir, current, "--fail-if-regression", "10", "--fail-on-inconclusive")
    assert code == exit_codes.INCONCLUSIVE


def test_compare_threshold_boundary(workdir):
    flat = {"h2d": [0.5] * 20, "kernel": [3.0] * 20, "d2h": [0.5] * 20}  # total 4.0
    path = write(workdir / "base.json", make_result(phases=flat))
    assert main(["baseline", "save", "--name", "main", "--from", str(path)]) == 0
    current = make_result(phases={**flat, "kernel": [3.5] * 20})  # total 4.5: +12.5% exactly
    assert compare(workdir, current, "--fail-if-regression", "12.5") == exit_codes.OK
    assert compare(workdir, current, "--fail-if-regression", "12.4") == exit_codes.REGRESSION


# --- compare: refusals -----------------------------------------------------------------------


def test_compare_missing_or_inconsistent_baseline(workdir):
    assert compare(workdir, make_result()) == exit_codes.BAD_FILE
    save(workdir)
    path = workdir / ".probe" / "baselines" / "main.json"
    doc = json.loads(path.read_text())
    doc["result"]["timing"]["gpu"]["total"]["p50"] = 0.001
    path.write_text(json.dumps(doc))
    assert compare(workdir, make_result()) == exit_codes.BAD_FILE


def test_compare_refuses_invalid_or_malformed_current(workdir):
    save(workdir)
    assert compare(workdir, make_result(valid=False)) == exit_codes.INVALID_RESULT
    no_samples = make_result()
    del no_samples["samples"]
    assert compare(workdir, no_samples) == exit_codes.BAD_FILE
    (workdir / "current.json").write_text("not json")
    code = main(["compare", "vector-add", "--baseline", "main", "--current", "current.json"])
    assert code == exit_codes.BAD_FILE


def test_compare_workload_mismatch(workdir):
    save(workdir)
    code = main(["compare", "reduction", "--baseline", "main", "--current", "x.json"])
    assert code == exit_codes.INCOMPATIBLE


def test_compare_parameter_mismatch_cannot_be_overridden(workdir, capsys):
    save(workdir)
    code = compare(workdir, make_result(n=2048), "--allow-environment-mismatch")
    assert code == exit_codes.INCOMPATIBLE
    assert "FATAL: parameter n" in capsys.readouterr().err


def test_compare_gpu_mismatch_needs_override(workdir, capsys):
    save(workdir, gpu="GPU A")
    assert compare(workdir, make_result(gpu="GPU B")) == exit_codes.INCOMPATIBLE
    err = capsys.readouterr().err
    assert "FATAL: GPU name" in err and "--allow-environment-mismatch" in err

    code = compare(workdir, make_result(gpu="GPU B"), "--allow-environment-mismatch")
    assert code == exit_codes.OK
    err = capsys.readouterr().err
    assert "WARNING (overridden): GPU name" in err
    assert "ENVIRONMENT MISMATCH OVERRIDDEN" in err


def test_compare_warning_does_not_block(workdir, capsys):
    save(workdir, driver="12.2")
    assert compare(workdir, make_result(driver="12.4")) == exit_codes.OK
    assert "warning: cuda driver version" in capsys.readouterr().err


def test_compare_argument_errors(workdir):
    save(workdir)
    assert compare(workdir, make_result(), "--metric", "nonsense") == exit_codes.USAGE
    assert compare(workdir, make_result(), "--size", "10") == exit_codes.USAGE
    with pytest.raises(SystemExit) as info:
        compare(workdir, make_result(), "--fail-if-regression", "-5")
    assert info.value.code == exit_codes.USAGE
    with pytest.raises(SystemExit) as info:
        main(["compare", "matmul", "--baseline", "main"])
    assert info.value.code == exit_codes.USAGE


# --- the real binary on this machine -----------------------------------------------------------


def test_run_without_binary(workdir, capsys):
    code = main(["run", "vector-add", "--binary", str(workdir / "missing")])
    assert code == exit_codes.WORKLOAD_FAILED
    assert "probe-workloads not found" in capsys.readouterr().err


def test_bad_workload_option_reaches_binary_as_usage_error(workdir, workloads_binary):
    code = main(["run", "reduction", "--block-size", "96", "--binary", str(workloads_binary)])
    assert code == exit_codes.USAGE


@pytest.mark.skipif(nvidia_gpu_available(), reason="checks the no-GPU path")
def test_run_and_compare_without_gpu(workdir, workloads_binary, capsys):
    binary = ["--binary", str(workloads_binary)]
    assert main(["run", "vector-add", "--size", "1024", *binary]) == exit_codes.NO_GPU
    assert "no usable CUDA device" in capsys.readouterr().err
    assert not (workdir / "results").exists()  # nothing fabricated

    save(workdir)
    assert main(["compare", "vector-add", "--baseline", "main", *binary]) == exit_codes.NO_GPU


@pytest.mark.gpu
def test_run_save_compare_on_gpu(workdir, workloads_binary, capsys):
    binary = ["--binary", str(workloads_binary)]
    args = ["--size", "1048576", "--iters", "20"]
    assert main(["baseline", "save", "vector-add", "--name", "gpu", *args, *binary]) == 0
    code = main(["compare", "vector-add", "--baseline", "gpu", *binary])
    assert code == exit_codes.OK
    assert "-> " in capsys.readouterr().out
    assert list((workdir / "results").glob("vector-add-*.json"))


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "probe", "--version"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "probe 0.1.0"

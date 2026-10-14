"""Tests for Probe_GPU_Validation.ipynb.

The notebook's decisions (which failures block, what counts as sanitizer
clean, whether CUDA workloads actually ran) live in its helper cell. These
tests execute that cell's real code against synthetic run directories. All
documents here are synthetic test data; none of it is a measurement.
"""

import json
import re
import tarfile
from pathlib import Path

import pytest

from synthetic import make_result

NOTEBOOK = Path(__file__).resolve().parents[1] / "Probe_GPU_Validation.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text())


@pytest.fixture
def nb(notebook, tmp_path, monkeypatch):
    """Namespace produced by running the helper cell with ROOT = tmp_path."""
    monkeypatch.setenv("PROBE_NB_ROOT", str(tmp_path))
    helper = next(c for c in notebook["cells"] if "helpers" in c["metadata"].get("tags", []))
    namespace: dict = {}
    exec("".join(helper["source"]), namespace)
    return namespace


# --- structure -------------------------------------------------------------------------


def test_is_a_clean_nbformat_4_notebook(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_every_code_cell_compiles(notebook):
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell {i}", "exec")


def test_authorship(notebook):
    assert notebook["metadata"]["authors"] == [{"name": "Silas Offei Darko"}]
    text = NOTEBOOK.read_text()
    assert "Author: Silas Offei Darko" in text


def test_stages_in_order(notebook):
    headings = [
        "".join(c["source"]).splitlines()[0]
        for c in notebook["cells"]
        if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Stage")
    ]
    assert [int(re.search(r"Stage (\d+)", h).group(1)) for h in headings] == list(range(12))


def test_uses_the_repository_scripts_and_paths(notebook):
    source = "".join("".join(c["source"]) for c in notebook["cells"])
    for script in ("scripts/validate_gpu.py", "scripts/analyze_validation.py"):
        assert script in source
        assert (Path(__file__).resolve().parents[1] / script).is_file()
    assert "pip install -e '.[dev]'" in source
    assert "-DCMAKE_BUILD_TYPE=Release" in source
    assert 'REPO / "build" / "workloads" / "probe-workloads"' in source
    assert "No NVIDIA GPU is attached to this Colab runtime." in source


# --- helper logic ---------------------------------------------------------------------------


def test_find_run_dir(nb):
    out = "recording to /c/probe/validation-runs/X\n...\nrecorded in /c/probe/validation-runs/X\n"
    assert nb["find_run_dir"](out) == Path("/c/probe/validation-runs/X")
    assert nb["find_run_dir"]("no such line") is None


def test_small_helpers(nb):
    assert nb["gpu_slug"]("Tesla T4") == "tesla-t4"
    assert nb["gpu_slug"](None) == "unknown-gpu"
    assert nb["version_tuple"]("cmake version 3.31.6") == (3, 31, 6)
    assert nb["version_tuple"](">=3.11") == (3, 11)
    assert nb["ANSI_ESCAPE"].sub("", "\x1b[31m13 failed\x1b[0m") == "13 failed"


def step(name, code, note=None):
    return {"name": name, "exit_code": code, "note": note, "argv": [], "seconds": 0.1}


def write_run(run_dir: Path, steps: list[tuple[dict, str, str]]) -> Path:
    for s, stdout, stderr in steps:
        base = run_dir / s["name"]
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{base}.stdout").write_text(stdout)
        Path(f"{base}.stderr").write_text(stderr)
    (run_dir / "steps.json").write_text(json.dumps([s for s, _, _ in steps]))
    return run_dir


@pytest.mark.parametrize(
    ("code", "note", "text", "expected"),
    [
        (0, None, "========= ERROR SUMMARY: 0 errors", "clean"),
        (0, None, "========= RACECHECK SUMMARY: 0 hazards displayed", "clean"),
        (99, None, "========= ERROR SUMMARY: 3 errors", "errors reported"),
        (0, None, "========= ERROR SUMMARY: 2 errors", "errors reported"),
        (0, None, "", "did not complete"),  # exit 0 alone is not proof
        (13, None, "Unable to find injection library", "did not complete"),
        (None, "command not found: compute-sanitizer", "", "unavailable"),
    ],
)
def test_sanitizer_verdict(nb, code, note, text, expected):
    assert nb["sanitizer_verdict"](step("sanitizer/x", code, note), text) == expected


def test_classify_successful_run(nb, tmp_path):
    vec, red = make_result(), make_result(workload="reduction")
    run = write_run(
        tmp_path / "run",
        [
            (step("env/nvidia-smi", None, "command not found: nvidia-smi"), "", ""),  # not blocking
            (step("vector_add/n33", 0), json.dumps(vec), ""),
            (step("reduction/prime-n1009", 0), json.dumps(red), ""),
            (step("sanitizer/memcheck-vector_add-n33", 0), "ERROR SUMMARY: 0 errors", ""),
            (step("workflow/vec-compare-a", 0), "x -> PASS: no change beyond the noise floor", ""),
            (step("workflow/vec-compare-b", 0), "x -> INCONCLUSIVE: change exceeds", ""),
        ],
    )
    summary = nb["classify_run"](run)
    assert summary["blocking"] == []
    assert summary["valid"] == {"vector_add": 1, "reduction": 1}
    assert summary["sanitizer"] == {"memcheck-vector_add-n33": "clean"}
    assert summary["workflow"] == {
        "vec-compare-a": "exit 0, PASS",
        "vec-compare-b": "exit 0, INCONCLUSIVE",  # reported as recorded, not forced to PASS
    }
    assert nb["workloads_executed"](summary)


def test_classify_failed_run(nb, tmp_path):
    invalid = make_result(valid=False)
    run = write_run(
        tmp_path / "run",
        [
            (step("build/pytest", 1), "1 failed", ""),
            (step("vector_add/n1", 3), "", "no usable CUDA device"),
            (step("vector_add/n33", 4), json.dumps(invalid), ""),
            (step("sanitizer/racecheck-reduction-n257", 99), "RACECHECK SUMMARY: 2 hazards", ""),
        ],
    )
    summary = nb["classify_run"](run)
    assert summary["blocking"] == [
        "build/pytest: exit 1",
        "vector_add/n1: exit 3",
        "vector_add/n33: exit 4",
        "sanitizer/racecheck-reduction-n257: compute-sanitizer reported errors",
    ]
    assert summary["valid"] == {"vector_add": 0, "reduction": 0}
    assert summary["attempted"] == {"vector_add": 2, "reduction": 0}
    assert not nb["workloads_executed"](summary)


def test_completion_requires_both_workloads(nb, tmp_path):
    run = write_run(tmp_path / "run", [(step("vector_add/n33", 0), json.dumps(make_result()), "")])
    assert not nb["workloads_executed"](nb["classify_run"](run))
    assert not nb["workloads_executed"]({})


def test_summary_wording(nb):
    nb["STATUS"].update({"tools": {"compute-sanitizer": "UNAVAILABLE", "nsys": "UNAVAILABLE",
                                   "ncu": "AVAILABLE"}})  # fmt: skip
    not_done = nb["summary_lines"](False, "No NVIDIA GPU is attached to this Colab runtime.")
    assert not_done[0] == "STAGE 5 NOT COMPLETE"
    assert "Reason: No NVIDIA GPU is attached to this Colab runtime." in not_done
    text = "\n".join(not_done)
    assert "compute-sanitizer unavailable in this environment" in text
    assert "not reached" in text  # stages never run are not reported as passed
    assert "FINISHED" not in text

    nb["STATUS"]["full"] = {"valid": {"vector_add": 9, "reduction": 27},
                            "attempted": {"vector_add": 9, "reduction": 27},
                            "sanitizer": {}, "workflow": {}}  # fmt: skip
    done = nb["summary_lines"](True)
    assert done[0] == "STAGE 5 HARDWARE VALIDATION FINISHED"
    assert any("9 of 9 sizes valid on the GPU" in line for line in done)


def test_archive_excludes_build_trees(nb, tmp_path):
    work, logs = nb["WORK"], nb["LOGS"]
    (logs / "stage0.log").write_text("log")
    run = tmp_path / "probe" / "validation-runs" / "X"
    (run / "cmake-build" / "workloads").mkdir(parents=True)
    (run / "cmake-build" / "workloads" / "probe-workloads").write_text("binary")
    (run / "vector_add").mkdir()
    (run / "vector_add" / "n1.stdout").write_text("{}")
    nb["STATUS"]["full"] = {"dir": str(run)}
    archive = nb["make_archive"]()
    assert archive.parent == tmp_path and archive.name.startswith("probe-stage5-colab-")
    names = tarfile.open(archive).getnames()
    assert "probe-stage5/logs/stage0.log" in names
    assert "validation-runs/full-X/vector_add/n1.stdout" in names
    assert not any("cmake-build" in n for n in names)
    assert work.is_dir()

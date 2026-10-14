"""Tests for baseline storage and integrity checks (probe.baseline)."""

import json
import math

import pytest

from probe import exit_codes
from probe.baseline import (
    delete_baseline,
    list_baselines,
    load_baseline,
    save_baseline,
)
from probe.exit_codes import ProbeError
from probe.result_schema import raw_series, stored_stats
from probe.stats import summarize

from synthetic import make_result


def expect_error(code, fn, *args, **kwargs):
    with pytest.raises(ProbeError) as info:
        fn(*args, **kwargs)
    assert info.value.code == code, str(info.value)
    return str(info.value)


def edit_saved(store, name, mutate):
    """Simulate a hand edit or corruption of a saved baseline file."""
    path = store / "baselines" / f"{name}.json"
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc))


# --- save and load ----------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    result = make_result()
    path = save_baseline(tmp_path, "main", result)
    assert path == tmp_path / "baselines" / "main.json"

    doc = load_baseline(tmp_path, "main")
    assert doc["schema"] == "probe.baseline"
    assert doc["baseline_schema_version"] == 1
    assert doc["name"] == "main"
    assert doc["result"] == result


def test_saved_baseline_contains_everything_needed(tmp_path):
    save_baseline(tmp_path, "main", make_result())
    r = load_baseline(tmp_path, "main")["result"]
    assert r["schema_version"] == 2
    assert r["build"]["probe_version"] and r["build"]["cuda_architectures"]
    assert r["workload"] and r["parameters"]
    assert r["provenance"]["git_commit"]
    assert r["environment"]["device"]["name"] and r["environment"]["host"]
    assert r["config"]["warmup"] == 3 and r["config"]["iterations"] == 20
    assert r["valid"] is True and r["validation"]["passed"] is True
    assert set(raw_series(r)) == {"h2d", "kernel", "d2h", "gpu_total", "host_end_to_end"}


def test_stored_statistics_are_reconstructible_from_raw_samples(tmp_path):
    save_baseline(tmp_path, "main", make_result())
    r = load_baseline(tmp_path, "main")["result"]
    stored = stored_stats(r)
    for metric, samples in raw_series(r).items():
        assert summarize(samples) == pytest.approx(stored[metric]), metric


def test_refuses_invalid_run(tmp_path):
    message = expect_error(
        exit_codes.INVALID_RESULT, save_baseline, tmp_path, "bad", make_result(valid=False)
    )
    assert "invalid run" in message
    assert not (tmp_path / "baselines" / "bad.json").exists()


def test_refuses_result_without_raw_samples(tmp_path):
    result = make_result()
    del result["samples"]
    assert "raw samples" in expect_error(exit_codes.BAD_FILE, save_baseline, tmp_path, "x", result)


def test_refuses_result_without_provenance(tmp_path):
    result = make_result()
    del result["provenance"]
    assert "provenance" in expect_error(exit_codes.BAD_FILE, save_baseline, tmp_path, "x", result)


@pytest.mark.parametrize("name", ["", "../escape", "a/b", ".hidden", "x" * 65, "sp ace"])
def test_rejects_unsafe_names(tmp_path, name):
    expect_error(exit_codes.USAGE, save_baseline, tmp_path, name, make_result())


def test_overwrite_requires_force(tmp_path):
    save_baseline(tmp_path, "main", make_result(commit="a" * 40))
    expect_error(exit_codes.USAGE, save_baseline, tmp_path, "main", make_result(commit="b" * 40))
    save_baseline(tmp_path, "main", make_result(commit="b" * 40), overwrite=True)
    loaded = load_baseline(tmp_path, "main")
    assert loaded["result"]["provenance"]["git_commit"] == "b" * 40


def test_list_and_delete(tmp_path):
    assert list_baselines(tmp_path) == []
    save_baseline(tmp_path, "b-second", make_result())
    save_baseline(tmp_path, "a-first", make_result(workload="reduction"))
    (tmp_path / "baselines" / "broken.json").write_text("{not json")

    entries = list_baselines(tmp_path)
    assert [name for name, _, _ in entries] == ["a-first", "b-second", "broken"]
    assert entries[0][1]["result"]["workload"] == "reduction"
    assert entries[2][1] is None and "not valid JSON" in entries[2][2]

    delete_baseline(tmp_path, "a-first")
    assert [name for name, _, _ in list_baselines(tmp_path)] == ["b-second", "broken"]
    expect_error(exit_codes.BAD_FILE, delete_baseline, tmp_path, "a-first")


def test_missing_baseline(tmp_path):
    assert "not found" in expect_error(exit_codes.BAD_FILE, load_baseline, tmp_path, "nope")


# --- tampering and corruption ----------------------------------------------------------


@pytest.fixture
def saved(tmp_path):
    save_baseline(tmp_path, "main", make_result())
    return tmp_path


def test_detects_edited_aggregate(saved):
    def edit(doc):
        doc["result"]["timing"]["gpu"]["phases"]["kernel"]["p50"] *= 0.9

    edit_saved(saved, "main", edit)
    message = expect_error(exit_codes.BAD_FILE, load_baseline, saved, "main")
    assert "kernel.p50" in message and "recomputed from raw samples" in message


def test_detects_edited_raw_sample(saved):
    def edit(doc):
        doc["result"]["samples"]["gpu"]["kernel"][0] = 99.0

    edit_saved(saved, "main", edit)
    assert "kernel" in expect_error(exit_codes.BAD_FILE, load_baseline, saved, "main")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_detects_non_finite_samples(saved, token):
    # Python's json module writes and reads these non-standard tokens, so a
    # corrupted file can contain them.
    path = saved / "baselines" / "main.json"
    doc = json.loads(path.read_text())
    doc["result"]["samples"]["gpu"]["kernel"][3] = float(token.replace("Infinity", "inf"))
    path.write_text(json.dumps(doc))
    assert token in path.read_text()
    message = expect_error(exit_codes.BAD_FILE, load_baseline, saved, "main")
    assert "sample 3 is not a finite non-negative duration" in message


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["result"]["samples"]["gpu"]["kernel"].__setitem__(0, "1.8"), "not a number"),
        (lambda d: d["result"]["samples"]["gpu"]["kernel"].pop(), "expected 20 raw samples"),
        (lambda d: d["result"]["samples"]["gpu"].pop("d2h"), "raw samples incomplete"),
        (lambda d: d["result"].pop("timing"), "missing key 'timing'"),
        (lambda d: d["result"].__setitem__("schema_version", 1), "schema_version"),
        (lambda d: d.__setitem__("baseline_schema_version", 2), "baseline_schema_version"),
        (lambda d: d.__setitem__("schema", "something.else"), "schema"),
        (lambda d: d.__setitem__("name", "other"), "records the name"),
        (lambda d: d["result"].__setitem__("timing", []), "malformed result"),
        (lambda d: d.pop("result"), "result is not a JSON object"),
    ],
)
def test_detects_malformed_files(saved, mutate, expected):
    edit_saved(saved, "main", mutate)
    assert expected in expect_error(exit_codes.BAD_FILE, load_baseline, saved, "main")


def test_detects_invalid_run_edited_into_file(saved):
    def edit(doc):
        doc["result"]["valid"] = False
        doc["result"]["invalid_reasons"] = ["edited"]

    edit_saved(saved, "main", edit)
    assert "invalid run" in expect_error(exit_codes.BAD_FILE, load_baseline, saved, "main")


def test_detects_non_json(saved):
    (saved / "baselines" / "main.json").write_text("")
    assert "not valid JSON" in expect_error(exit_codes.BAD_FILE, load_baseline, saved, "main")


def test_written_files_are_strict_json(saved):
    text = (saved / "baselines" / "main.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard token {c}"))
    assert not any(math.isnan(x) for x in json.loads(text)["result"]["samples"]["gpu"]["kernel"])

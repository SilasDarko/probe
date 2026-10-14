"""Baselines: known-good results stored as plain JSON files.

Layout: <store>/baselines/<name>.json, where <store> is --store, or
$PROBE_STORE, or ./.probe. Names are global (not per workload); a baseline
records its workload and compare refuses to use it for another one.

A baseline file wraps one complete result document, including raw samples:

    {
      "schema": "probe.baseline",
      "baseline_schema_version": 1,
      "name": "main",
      "saved_utc": "...",
      "result": { ...probe.run document... }
    }

Every load re-validates the file: structure, validity, and that the stored
statistics follow from the stored raw samples. That catches hand edits and
corruption; it is a consistency check, not a cryptographic signature.
"""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from probe import exit_codes
from probe.exit_codes import ProbeError
from probe.result_schema import check_raw_samples, check_run_result

BASELINE_SCHEMA = "probe.baseline"
BASELINE_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def default_store() -> Path:
    return Path(os.environ.get("PROBE_STORE", ".probe"))


def baseline_dir(store: Path) -> Path:
    return store / "baselines"


def check_name(name: str) -> None:
    # Restricting the character set also rules out path traversal ("../x").
    if not _NAME.match(name):
        raise ProbeError(
            exit_codes.USAGE,
            f"invalid baseline name '{name}': use letters, digits, '.', '_' or '-' "
            "(max 64 characters, starting with a letter or digit)",
        )


def require_usable_result(result: object, what: str) -> None:
    """Raise unless `result` can be used as a baseline or comparison input.

    Checked in this order, so the reported reason is the most fundamental:
      1. structure            -> BAD_FILE
      2. the run was valid    -> INVALID_RESULT
      3. raw samples present and consistent, provenance present -> BAD_FILE
    """
    problems = check_run_result(result)
    if problems:
        raise ProbeError(exit_codes.BAD_FILE, f"{what} is not usable: " + "; ".join(problems))
    if not result["valid"]:
        raise ProbeError(
            exit_codes.INVALID_RESULT,
            f"{what} is an invalid run: " + "; ".join(result["invalid_reasons"]),
        )
    problems = check_raw_samples(result)
    if "provenance" not in result:
        problems.append("no provenance (it was not produced by 'probe run')")
    if problems:
        raise ProbeError(exit_codes.BAD_FILE, f"{what} is not usable: " + "; ".join(problems))


def save_baseline(store: Path, name: str, result: dict, overwrite: bool = False) -> Path:
    """Save `result` as baseline `name`. Refuses invalid runs (INVALID_RESULT)."""
    check_name(name)
    require_usable_result(result, "result")

    path = baseline_dir(store) / f"{name}.json"
    if path.exists() and not overwrite:
        raise ProbeError(
            exit_codes.USAGE, f"baseline '{name}' already exists (use --force to replace it)"
        )
    doc = {
        "schema": BASELINE_SCHEMA,
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "name": name,
        "saved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and rename, so an interrupted save never
    # leaves a half-written baseline behind.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def read_json(path: Path, what: str) -> object:
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise ProbeError(exit_codes.BAD_FILE, f"{what} not found: {path}") from exc
    except OSError as exc:
        raise ProbeError(exit_codes.BAD_FILE, f"cannot read {what} {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProbeError(exit_codes.BAD_FILE, f"{what} {path} is not valid JSON: {exc}") from exc


def validate_baseline(doc: object, name: str) -> dict:
    """Return the wrapped result if `doc` is a sound baseline called `name`."""
    if not isinstance(doc, dict):
        raise ProbeError(exit_codes.BAD_FILE, f"baseline '{name}' is not a JSON object")
    if doc.get("schema") != BASELINE_SCHEMA:
        raise ProbeError(exit_codes.BAD_FILE, f"baseline '{name}' has schema {doc.get('schema')!r}")
    if doc.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        raise ProbeError(
            exit_codes.BAD_FILE,
            f"baseline '{name}' has baseline_schema_version "
            f"{doc.get('baseline_schema_version')!r}, expected {BASELINE_SCHEMA_VERSION}",
        )
    if doc.get("name") != name:
        raise ProbeError(
            exit_codes.BAD_FILE, f"baseline file '{name}' records the name {doc.get('name')!r}"
        )
    result = doc.get("result")
    try:
        require_usable_result(result, f"baseline '{name}'")
    except ProbeError as exc:
        # A stored baseline that is unusable is a bad file, whatever the reason.
        raise ProbeError(exit_codes.BAD_FILE, str(exc)) from exc
    return doc


def load_baseline(store: Path, name: str) -> dict:
    check_name(name)
    path = baseline_dir(store) / f"{name}.json"
    return validate_baseline(read_json(path, f"baseline '{name}'"), name)


def list_baselines(store: Path) -> list[tuple[str, dict | None, str | None]]:
    """(name, baseline or None, error or None) for every file in the store."""
    directory = baseline_dir(store)
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.json")):
        name = path.stem
        try:
            entries.append((name, load_baseline(store, name), None))
        except ProbeError as exc:
            entries.append((name, None, str(exc)))
    return entries


def delete_baseline(store: Path, name: str) -> Path:
    check_name(name)
    path = baseline_dir(store) / f"{name}.json"
    if not path.is_file():
        raise ProbeError(exit_codes.BAD_FILE, f"baseline '{name}' not found in {store}")
    path.unlink()
    return path

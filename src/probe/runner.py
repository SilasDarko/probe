"""Runs probe-workloads and turns its output into a result document."""

import json
import os
import subprocess
from pathlib import Path

from probe import exit_codes
from probe.environment import host_info, provenance
from probe.exit_codes import ProbeError
from probe.result_schema import check_run_result

# CLI workload name -> probe-workloads command name.
WORKLOADS = {"vector-add": "vector_add", "reduction": "reduction"}

# CLI option -> probe-workloads flag.
WORKLOAD_FLAGS = {
    "size": "--n",
    "block_size": "--block-size",
    "warmup": "--warmup",
    "iters": "--iters",
    "seed": "--seed",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_workload(name: str) -> str:
    """Accept "vector-add" or "vector_add"; return the CLI form."""
    normalized = name.replace("_", "-")
    if normalized not in WORKLOADS:
        raise ValueError(f"unknown workload '{name}' (choose from {', '.join(WORKLOADS)})")
    return normalized


def find_binary(explicit: Path | None) -> Path:
    """--binary, then $PROBE_WORKLOADS_BIN, then the repository's build dir."""
    if explicit is not None:
        candidate = explicit
    elif os.environ.get("PROBE_WORKLOADS_BIN"):
        candidate = Path(os.environ["PROBE_WORKLOADS_BIN"])
    else:
        candidate = _REPO_ROOT / "build" / "workloads" / "probe-workloads"
    if not candidate.is_file():
        raise ProbeError(
            exit_codes.WORKLOAD_FAILED,
            f"probe-workloads not found at {candidate} "
            "(build it with CMake, or pass --binary / set PROBE_WORKLOADS_BIN)",
        )
    return candidate


def workload_args(options: dict[str, int | None]) -> list[str]:
    args = []
    for key, flag in WORKLOAD_FLAGS.items():
        value = options.get(key)
        if value is not None:
            args += [flag, str(value)]
    return args


def run_workload(
    binary: Path, workload: str, options: dict[str, int | None], command: list[str]
) -> dict:
    """Run one workload and return its result document, augmented with
    provenance and host details. Raw samples are always recorded: they are
    what baselines and comparisons are computed from.

    Returns invalid results too (validation failed); callers decide what to
    do with them. Raises ProbeError for every case with no usable result."""
    argv = [str(binary), WORKLOADS[workload], *workload_args(options), "--json", "--raw-samples"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except OSError as exc:
        raise ProbeError(exit_codes.WORKLOAD_FAILED, f"could not run {binary}: {exc}") from exc

    stderr = proc.stderr.strip()
    if proc.returncode == 3:
        raise ProbeError(exit_codes.NO_GPU, stderr or "no usable CUDA device")
    if proc.returncode == 2:
        raise ProbeError(exit_codes.USAGE, stderr or "probe-workloads rejected the arguments")
    if proc.returncode not in (0, 4):
        raise ProbeError(
            exit_codes.WORKLOAD_FAILED,
            f"probe-workloads exited with code {proc.returncode}: {stderr}",
        )

    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(
            exit_codes.WORKLOAD_FAILED, f"probe-workloads printed invalid JSON: {exc}"
        ) from exc
    problems = check_run_result(doc)
    if problems:
        raise ProbeError(
            exit_codes.WORKLOAD_FAILED, "unexpected result structure: " + "; ".join(problems)
        )

    doc["environment"]["host"] = host_info()
    doc["provenance"] = provenance(command, Path.cwd())
    return doc

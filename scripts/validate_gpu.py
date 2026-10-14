#!/usr/bin/env python3
"""Stage 5 hardware validation: run every check, record everything, judge nothing.

Run from the repository root on a Linux machine with an NVIDIA GPU:

    python3 scripts/validate_gpu.py            # full run (about 15-30 minutes)
    python3 scripts/validate_gpu.py --quick    # smoke test of the procedure

Each step's command, exit code, duration, stdout and stderr are saved under
validation-runs/<UTC timestamp>/ (or --out). A failing step is recorded and the
script moves on, so one problem does not hide the rest, and steps.json is
rewritten after every step so an interrupted session keeps what it gathered.
At the end everything except the CMake build tree is packed into a .tar.gz
next to the run directory.

This script interprets nothing. Analysis is separate:

    python3 scripts/analyze_validation.py validation-runs/<timestamp>

Standard library only: it runs before anything is installed.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# D. vector_add sizes: 1, warp boundaries (31/32/33), block boundaries
# (255/256/257), a prime, and a large size.
VECTOR_ADD_SIZES = [1, 31, 32, 33, 255, 256, 257, 1000003, 10000000]

# E. reduction sizes by category. "detection_boundary" brackets the n where
# the analytical bound passes 0.5 (about 466,000 at block size 256), beyond
# which a single dropped element is no longer guaranteed to be detected.
REDUCTION_SIZES = {
    "tiny": [1, 2, 3, 31, 32, 33],
    "block_boundary": [255, 256, 257, 511, 512, 513, 65535, 65536, 65537],
    "prime": [1009, 65521, 999983, 1000003],
    "detection_boundary": [400000, 466000, 470000, 500000],
    "large": [1 << 20, 1 << 22, 10000000, 1 << 24],
}

# compute-sanitizer is slow; it runs on the awkward sizes, not the large ones.
SANITIZE_VECTOR_ADD = [1, 31, 33, 255, 257, 1000003]
SANITIZE_REDUCTION = [1, 2, 33, 255, 257, 513, 65537]
SANITIZER_TOOLS = ["memcheck", "racecheck", "synccheck", "initcheck"]

WARMUPS = [0, 1, 5, 10, 50]
NOISE_RUNS = 5

# Large sizes used for timing, warmup and noise experiments.
TIMING_WORKLOADS = {"vector_add": 10000000, "reduction": 1 << 24}


class Recorder:
    def __init__(self, out: Path, env: dict[str, str]):
        self.out = out
        self.env = env
        self.steps: list[dict] = []

    def run(self, name, argv, *, cwd=REPO, extra_env=None, timeout=1800):
        argv = [str(a) for a in argv]
        base = self.out / name
        base.parent.mkdir(parents=True, exist_ok=True)
        env = dict(self.env)
        env.update({k: str(v) for k, v in (extra_env or {}).items()})

        started = datetime.now(UTC).isoformat(timespec="seconds")
        t0 = time.monotonic()
        stdout, stderr, code, note = "", "", None, None
        try:
            proc = subprocess.run(
                argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except FileNotFoundError:
            note = f"command not found: {argv[0]}"
        except subprocess.TimeoutExpired as exc:
            note = f"timed out after {timeout} s"
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else ""
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else ""
        seconds = round(time.monotonic() - t0, 3)

        Path(f"{base}.stdout").write_text(stdout)
        Path(f"{base}.stderr").write_text(stderr if note is None else f"{note}\n{stderr}")
        entry = {
            "name": name,
            "argv": argv,
            "cwd": str(cwd),
            "extra_env": {k: str(v) for k, v in (extra_env or {}).items()},
            "exit_code": code,
            "note": note,
            "seconds": seconds,
            "started_utc": started,
        }
        self.steps.append(entry)
        (self.out / "steps.json").write_text(json.dumps(self.steps, indent=2) + "\n")
        status = note or f"exit {code}"
        print(f"[{len(self.steps):3d}] {name:<55} {status:<22} {seconds:8.1f} s", flush=True)
        return entry


def tool_env() -> dict[str, str]:
    env = dict(os.environ)
    cuda_bin = Path("/usr/local/cuda/bin")
    if cuda_bin.is_dir():  # where Colab and NVIDIA's packages put nvcc and the sanitizer
        env["PATH"] = f"{cuda_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def workload(binary: Path, name: str, n: int, warmup: int, iters: int, *extra: str) -> list:
    return [
        binary,
        name,
        "--n",
        n,
        "--warmup",
        warmup,
        "--iters",
        iters,
        "--json",
        "--raw-samples",
        *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="few sizes and iterations")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or REPO / "validation-runs" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rec = Recorder(out, tool_env())
    q = args.quick
    print(f"recording to {out}{' (quick mode)' if q else ''}\n")

    (out / "meta.json").write_text(
        json.dumps(
            {
                "started_utc": stamp,
                "quick": q,
                "python": sys.version,
                "repo": str(REPO),
                "env": {
                    k: os.environ.get(k)
                    for k in ("CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING")
                },
            },
            indent=2,
        )
        + "\n"
    )

    # --- A. environment ---------------------------------------------------------
    nvsmi_query = "name,compute_cap,driver_version,memory.total,pci.bus_id,persistence_mode"
    for name, argv in [
        ("nvidia-smi", ["nvidia-smi"]),
        ("nvidia-smi-query", ["nvidia-smi", f"--query-gpu={nvsmi_query}", "--format=csv"]),
        ("nvidia-smi-q", ["nvidia-smi", "-q"]),
        ("nvcc", ["nvcc", "--version"]),
        ("uname", ["uname", "-a"]),
        ("lscpu", ["lscpu"]),
        ("os-release", ["cat", "/etc/os-release"]),
        ("cmake", ["cmake", "--version"]),
        ("python", [sys.executable, "--version"]),
        ("gcc", ["gcc", "--version"]),
        ("gxx", ["g++", "--version"]),
        ("git-commit", ["git", "rev-parse", "HEAD"]),
        ("git-status", ["git", "status", "--porcelain"]),
        ("compute-sanitizer", ["compute-sanitizer", "--version"]),
    ]:
        rec.run(f"env/{name}", argv)

    # --- B. clean build -----------------------------------------------------------
    build = out / "cmake-build"  # step logs go to build/, the CMake tree here
    rec.run(
        "build/configure",
        ["cmake", "-S", REPO, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DPROBE_REQUIRE_CUDA=ON"],
    )
    rec.run("build/compile", ["cmake", "--build", build, "-j"], timeout=3600)
    rec.run("build/ctest", ["ctest", "--test-dir", build, "--output-on-failure"])
    binary = build / "workloads" / "probe-workloads"
    rec.run("build/cuobjdump", ["cuobjdump", "--list-elf", "--list-ptx", binary])
    rec.run("build/version", [binary, "--version"])
    rec.run(
        "build/pip-install", [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[dev]"]
    )
    rec.run(
        "build/pytest",
        [sys.executable, "-m", "pytest", "-v", "-rs", "-p", "no:cacheprovider", REPO / "tests"],
        extra_env={"PROBE_BUILD_DIR": build},
    )

    # --- C. device info -------------------------------------------------------------
    rec.run("device/device-info", [binary, "--device-info"])
    rec.run("device/human-summary", [binary, "vector_add", "--n", 1000003, "--iters", 5])

    # --- D. vector_add correctness --------------------------------------------------------
    for n in VECTOR_ADD_SIZES[:4] if q else VECTOR_ADD_SIZES:
        rec.run(f"vector_add/n{n}", workload(binary, "vector_add", n, 3, 20))

    # --- E. reduction correctness ---------------------------------------------------------
    for category, sizes in REDUCTION_SIZES.items():
        for n in sizes[:1] if q else sizes:
            rec.run(f"reduction/{category}-n{n}", workload(binary, "reduction", n, 3, 20))

    # --- D/E. compute-sanitizer ------------------------------------------------------------
    for tool in SANITIZER_TOOLS[:1] if q else SANITIZER_TOOLS:
        cases = [("vector_add", n) for n in SANITIZE_VECTOR_ADD]
        cases += [("reduction", n) for n in SANITIZE_REDUCTION]
        for wl, n in cases[:2] if q else cases:
            rec.run(
                f"sanitizer/{tool}-{wl}-n{n}",
                [
                    "compute-sanitizer",
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "99",
                    binary,
                    wl,
                    "--n",
                    n,
                    "--warmup",
                    0,
                    "--iters",
                    2,
                    "--json",
                ],
                timeout=1800,
            )

    # --- F. timing ------------------------------------------------------------------------
    for wl, n in TIMING_WORKLOADS.items():
        rec.run(f"timing/{wl}", workload(binary, wl, n, 5, 20 if q else 200))

    # --- G. warmup --------------------------------------------------------------------------
    for wl, n in TIMING_WORKLOADS.items():
        for w in [0, 5] if q else WARMUPS:
            rec.run(f"warmup/{wl}-w{w}", workload(binary, wl, n, w, 30))

    # --- H. noise: independent processes ------------------------------------------------------
    for wl, n in TIMING_WORKLOADS.items():
        for i in range(1, (2 if q else NOISE_RUNS) + 1):
            rec.run(f"noise/{wl}-run{i}", workload(binary, wl, n, 3, 20 if q else 100))

    # --- I. baseline workflow ------------------------------------------------------------------
    workflow = out / "workflow"
    workflow.mkdir(exist_ok=True)
    probe = [sys.executable, "-m", "probe"]
    where = ["--store", workflow / "store", "--binary", binary]
    iters = "20" if q else "100"
    for tag, wl, n in (("vec", "vector-add", 10000000), ("red", "reduction", 1 << 24)):
        name = f"{tag}-main"
        steps = [
            ("save", ["baseline", "save", wl, "--name", name, "--size", n, "--iters", iters]),
            ("show", ["baseline", "show", name]),
            ("compare-a", ["compare", wl, "--baseline", name]),
            ("compare-b", ["compare", wl, "--baseline", name]),  # a separate process
        ]
        for label, cmd in steps:
            rec.run(f"workflow/{tag}-{label}", [*probe, *cmd, *where], cwd=workflow)
    rec.run("workflow/list", [*probe, "baseline", "list", *where], cwd=workflow)

    # --- pack -----------------------------------------------------------------------------------
    archive = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(
            out,
            arcname=out.name,
            filter=lambda info: None if info.name.startswith(f"{out.name}/cmake-build") else info,
        )
    failed = [s for s in rec.steps if s["exit_code"] != 0]
    print(f"\n{len(rec.steps)} steps, {len(failed)} with a non-zero exit or no exit code")
    for s in failed:
        print(f"  {s['name']}: {s['note'] or 'exit ' + str(s['exit_code'])}")
    print(f"\nrecorded in {out}\npacked (without cmake-build/) into {archive}")
    print(f"next: python3 scripts/analyze_validation.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

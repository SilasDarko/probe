"""Provenance and host details the workload binary cannot know about."""

import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from probe import __version__


def _git(args: list[str], cwd: Path) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run([git, *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state(cwd: Path) -> dict:
    """Commit of the repository the command is run from, and whether the
    working tree had uncommitted changes (which the commit alone does not
    describe)."""
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd) if commit else None
    return {"git_commit": commit, "git_dirty": bool(status) if status is not None else None}


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def host_info() -> dict:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "python": platform.python_version(),
    }


def provenance(command: list[str], cwd: Path) -> dict:
    return {
        "probe_cli_version": __version__,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        **git_state(cwd),
    }

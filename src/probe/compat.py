"""Can two results be compared?

Mismatches fall into three groups:

fatal        never overridable: comparing them would compare different
             problems, not the same problem on different code
             (workload, workload parameters)
environment  fatal by default, overridable with --allow-environment-mismatch:
             the same problem on a different machine or build
             (GPU model, compute capability, build type)
warning      reported, comparison proceeds: things that can plausibly change
             performance but are often exactly what is being evaluated
             (CUDA runtime/driver, compiler and flags, compiled architectures,
             warmup/iteration counts, seed, host OS/CPU, Probe version)

A differing git commit is not a mismatch at all: comparing commits is the
point of a baseline.
"""

from dataclasses import dataclass, field


@dataclass
class Compatibility:
    fatal: list[str] = field(default_factory=list)
    environment: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def blocked(self, allow_environment_mismatch: bool) -> bool:
        return bool(self.fatal) or (bool(self.environment) and not allow_environment_mismatch)


def _get(doc: dict, *path: str):
    for key in path:
        if not isinstance(doc, dict):
            return None
        doc = doc.get(key)
    return doc


def _differs(label: str, base, cur, out: list[str]) -> None:
    if base != cur:
        out.append(f"{label}: baseline {base!r}, current {cur!r}")


def check_compatibility(baseline: dict, current: dict) -> Compatibility:
    c = Compatibility()

    _differs("workload", baseline["workload"], current["workload"], c.fatal)
    base_params, cur_params = baseline["parameters"], current["parameters"]
    for key in sorted(set(base_params) | set(cur_params)):
        _differs(f"parameter {key}", base_params.get(key), cur_params.get(key), c.fatal)

    device = ("environment", "device")
    for key in ("name", "compute_capability"):
        _differs(
            f"GPU {key}", _get(baseline, *device, key), _get(current, *device, key), c.environment
        )
    _differs(
        "build type",
        _get(baseline, "build", "build_type"),
        _get(current, "build", "build_type"),
        c.environment,
    )

    for key in ("cuda_runtime_version", "cuda_driver_version"):
        _differs(
            key.replace("_", " "),
            _get(baseline, *device, key),
            _get(current, *device, key),
            c.warnings,
        )
    for key in (
        "cuda_architectures",
        "cuda_compiler",
        "cxx_compiler",
        "cuda_flags",
        "probe_version",
    ):
        _differs(
            key.replace("_", " "),
            _get(baseline, "build", key),
            _get(current, "build", key),
            c.warnings,
        )
    for key in ("warmup", "iterations", "seed"):
        _differs(key, _get(baseline, "config", key), _get(current, "config", key), c.warnings)
    for key in ("os", "os_release", "machine", "cpu_model"):
        _differs(
            f"host {key}",
            _get(baseline, "environment", "host", key),
            _get(current, "environment", "host", key),
            c.warnings,
        )
    return c

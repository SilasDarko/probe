#!/usr/bin/env python3
"""Turn a validate_gpu.py run directory into a Markdown report.

    python3 scripts/analyze_validation.py validation-runs/<timestamp>

Writes <dir>/report.md and prints it. Every number comes from the recorded
files; missing or failed steps are reported as such, never filled in.

Precision: CUDA events resolve about 0.5 us, so GPU times are shown to 1 us
(0.001 ms). Ratios are shown to two significant figures.
"""

import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

from probe.compare import Verdict, compare_series, noise_floor
from probe.result_schema import GPU_TOTAL, HOST_END_TO_END, raw_series
from probe.stats import mad

MAD_TO_SIGMA = 1.4826
SE_FACTOR_P50 = math.sqrt(math.pi / 2)  # 1.2533


# --- loading ------------------------------------------------------------------------


class Run:
    def __init__(self, directory: Path):
        self.dir = directory
        steps_file = directory / "steps.json"
        self.steps = json.loads(steps_file.read_text()) if steps_file.is_file() else []
        self.by_name = {s["name"]: s for s in self.steps}

    def step(self, name):
        return self.by_name.get(name)

    def text(self, name, stream="stdout") -> str | None:
        path = self.dir / f"{name}.{stream}"
        return path.read_text(errors="replace") if path.is_file() else None

    def doc(self, name) -> dict | None:
        """The result JSON of a workload step (exit 0 or 4), else None."""
        step = self.step(name)
        if step is None or step["exit_code"] not in (0, 4):
            return None
        try:
            return json.loads(self.text(name) or "")
        except json.JSONDecodeError:
            return None

    def names(self, prefix):
        return [s["name"] for s in self.steps if s["name"].startswith(prefix)]


def ms(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def us(v) -> str:
    return "-" if v is None else f"{v * 1000:.1f}"


def sig2(v) -> str:
    return "-" if v is None else f"{v:.2g}"


def outcome(step) -> str:
    if step is None:
        return "not run"
    if step["note"]:
        return step["note"]
    return f"exit {step['exit_code']}"


def table(headers, rows) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


# --- sections -----------------------------------------------------------------------------


def section_steps(run: Run) -> list[str]:
    failed = [s for s in run.steps if s["exit_code"] != 0]
    lines = ["## Step outcomes", ""]
    lines.append(
        f"{len(run.steps)} steps recorded, {len(failed)} with a non-zero or missing exit code."
    )
    if failed:
        lines += [
            "",
            *table(
                ["step", "outcome", "seconds"],
                [(s["name"], outcome(s), s["seconds"]) for s in failed],
            ),
        ]
    return lines


def parse_nvsmi_query(text: str | None) -> dict:
    if not text:
        return {}
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if len(rows) < 2:
        return {}
    keys = [k.strip() for k in rows[0].split(",")]
    values = [v.strip() for v in rows[1].split(",")]
    return dict(zip(keys, values, strict=False))


def section_environment(run: Run) -> list[str]:
    lines = ["## Environment: Probe versus the system tools", ""]
    smi = parse_nvsmi_query(run.text("env/nvidia-smi-query"))
    header = run.text("env/nvidia-smi") or ""
    smi_cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
    nvcc = re.search(r"release ([\d.]+)", run.text("env/nvcc") or "")
    info = None
    try:
        info = json.loads(run.text("device/device-info") or "")["device"]
    except (json.JSONDecodeError, KeyError, TypeError):
        lines.append(
            "`probe-workloads --device-info` gave no JSON "
            f"({outcome(run.step('device/device-info'))})."
        )
        return lines

    def get(key):
        return smi.get(key) or smi.get(f"{key} [MiB]")

    mem_smi = get("memory.total")
    rows = [
        ("GPU name", get("name"), info["name"], "yes" if get("name") == info["name"] else "NO"),
        (
            "compute capability",
            get("compute_cap"),
            info["compute_capability"],
            "yes" if get("compute_cap") == info["compute_capability"] else "NO",
        ),
        (
            "memory (MiB)",
            mem_smi,
            f"{info['global_mem_bytes'] / 2**20:.0f}",
            "compare" if mem_smi else "-",
        ),
        (
            "CUDA version supported by driver",
            smi_cuda.group(1) if smi_cuda else None,
            info["cuda_driver_version"],
            "yes" if smi_cuda and smi_cuda.group(1) == info["cuda_driver_version"] else "NO",
        ),
        ("NVIDIA driver build", get("driver_version"), "not recorded by Probe", "GAP"),
        (
            "CUDA runtime / toolkit",
            nvcc.group(1) if nvcc else None,
            info["cuda_runtime_version"],
            "yes" if nvcc and nvcc.group(1) == info["cuda_runtime_version"] else "NO",
        ),
    ]
    lines += table(["field", "system tools", "Probe", "agrees"], rows)
    return lines


def section_architecture(run: Run) -> list[str]:
    lines = ["## Architecture: which code the GPU should load", ""]
    listing = run.text("build/cuobjdump") or ""
    sass = sorted(set(re.findall(r"\.sm_(\d+)\.cubin", listing)))
    ptx = sorted(set(re.findall(r"\.sm_(\d+)\.ptx", listing)))
    try:
        cc = json.loads(run.text("device/device-info") or "")["device"]["compute_capability"]
    except (json.JSONDecodeError, KeyError, TypeError):
        cc = None
    lines.append(
        f"SASS embedded: {', '.join('sm_' + s for s in sass) or 'none found'}; "
        f"PTX embedded: {', '.join('compute_' + p for p in ptx) or 'none found'}."
    )
    if cc:
        target = cc.replace(".", "")
        if target in sass:
            lines.append(
                f"Device is cc {cc}: native `sm_{target}` SASS is present, "
                "so no PTX JIT is expected."
            )
        elif any(int(p) <= int(target) for p in ptx):
            lines.append(
                f"Device is cc {cc}: no `sm_{target}` SASS; the driver would JIT-compile PTX."
            )
        else:
            lines.append(
                f"Device is cc {cc}: no exact SASS; CUDA may still load SASS of the same major "
                "version with a lower minor version. Check the run outcomes."
            )
        lines.append(
            "This is what CUDA's selection rules predict from the embedded images; "
            "the image actually loaded is not observed directly."
        )
    return lines


def section_vector_add(run: Run) -> list[str]:
    lines = ["## vector_add correctness", ""]
    rows = []
    for name in run.names("vector_add/"):
        doc, step = run.doc(name), run.step(name)
        n = name.split("/n")[1]
        if doc is None:
            rows.append((n, outcome(step), "-", "-", "-", "-", "-"))
            continue
        v = doc["validation"]
        rows.append(
            (
                n,
                outcome(step),
                doc["valid"],
                v["iterations_checked"],
                v["iterations_failed"],
                v["max_abs_error"],
                doc["parameters"].get("grid_size", "-"),
            )
        )
    lines += table(["n", "outcome", "valid", "checked", "failed", "max |error|", "grid"], rows)
    return lines


def section_correctness_summary(run: Run) -> list[str]:
    """Both workloads in one table: did each size pass, and by what margin."""
    lines = ["## Correctness summary", ""]
    rows = []
    for name in [*run.names("vector_add/"), *run.names("reduction/")]:
        workload, case = name.split("/", 1)
        size = case.rsplit("n", 1)[1]
        doc, step = run.doc(name), run.step(name)
        if doc is None:
            rows.append((workload, size, f"NO ({outcome(step)})", "-", "-", "-"))
            continue
        v = doc["validation"]
        passed = "yes" if step["exit_code"] == 0 and doc["valid"] else f"NO ({outcome(step)})"
        err, atol, rtol = v["max_abs_error"], v["atol"], v["rtol"]
        if atol == 0 and rtol == 0:
            allowed = "0 (exact)"
            ratio = "exact match" if err == 0 else "MISMATCH"
        else:
            allowed = f"{atol:.2e}" if rtol == 0 else f"atol {atol:.2e}, rtol {rtol:.2e}"
            ratio = sig2(err / atol) if rtol == 0 and atol else "-"
        rows.append((workload, size, passed, f"{err:.2e}", allowed, ratio))
    if not rows:
        return [*lines, "not run"]
    lines += table(
        ["workload", "size", "passed", "observed error", "allowed error", "observed / bound"], rows
    )
    return lines


def section_sanitizer(run: Run) -> list[str]:
    lines = ["## compute-sanitizer", ""]
    rows = []
    for name in run.names("sanitizer/"):
        text = (run.text(name) or "") + (run.text(name, "stderr") or "")
        summary = re.findall(r"(ERROR SUMMARY: [^\n]*|RACECHECK SUMMARY: [^\n]*)", text)
        if not summary:  # e.g. the sanitizer could not start; show its own message
            summary = [
                ln.strip("= ").strip() for ln in text.splitlines() if ln.startswith("=====")
            ][1:2]
        detail = "; ".join(summary) or "no sanitizer output"
        rows.append((name.split("/", 1)[1], outcome(run.step(name)), detail))
    lines += table(["case", "outcome", "sanitizer summary"], rows) if rows else ["not run"]
    return lines


def section_reduction(run: Run) -> list[str]:
    lines = ["## reduction: observed error versus analytical bound", ""]
    rows = []
    for name in run.names("reduction/"):
        category, n = name.split("/", 1)[1].rsplit("-n", 1)
        doc = run.doc(name)
        if doc is None:
            rows.append((category, n, outcome(run.step(name)), *["-"] * 6))
            continue
        v = doc["validation"]
        err, bound = v["max_abs_error"], v["atol"]
        ratio = err / bound if bound else None
        # A dropped element x (|x| >= 0.5) is guaranteed to be detected iff 0.5 - |err| > atol.
        detect = "yes" if 0.5 - err > bound else "no"
        rows.append(
            (
                category,
                n,
                outcome(run.step(name)),
                doc["valid"],
                doc["parameters"].get("passes", "-"),
                f"{err:.2e}",
                f"{bound:.2e}",
                sig2(ratio),
                detect,
            )
        )
    lines += table(
        [
            "category",
            "n",
            "outcome",
            "valid",
            "passes",
            "observed |error|",
            "bound (atol)",
            "error / bound",
            "1-element loss detectable",
        ],
        rows,
    )
    return lines


def section_timing(run: Run) -> list[str]:
    lines = ["## Timing relationships", ""]
    for name in run.names("timing/"):
        doc = run.doc(name)
        lines += [f"### {name.split('/')[1]}", ""]
        if doc is None or "samples" not in doc:
            lines += [f"no data ({outcome(run.step(name))})", ""]
            continue
        series = raw_series(doc)
        phases = doc["timing"]["gpu"]["phase_order"]
        total = np.array(series[GPU_TOTAL])
        parts = sum(np.array(series[p]) for p in phases)
        gap = parts - total
        host_minus_gpu = np.array(series[HOST_END_TO_END]) - total
        rows = [
            (
                m,
                ms(float(np.median(series[m]))),
                ms(MAD_TO_SIGMA * mad(series[m])),
                ms(min(series[m])),
                ms(max(series[m])),
            )
            for m in [*phases, GPU_TOTAL, HOST_END_TO_END]
        ]
        lines += table(["metric", "p50 ms", "robust sigma ms", "min ms", "max ms"], rows)
        lines += ["", f"n = {len(total)} iterations.", ""]
        lines += [f"Timing accounting ({' + '.join(phases)} versus GPU total), per iteration:", ""]
        lines += table(
            ["", "sum of phases ms", "GPU total ms", "difference us"],
            [
                ("median", ms(float(np.median(parts))), ms(float(np.median(total))),
                 us(float(np.median(gap)))),
                ("min", ms(float(parts.min())), ms(float(total.min())), us(float(gap.min()))),
                ("max", ms(float(parts.max())), ms(float(total.max())), us(float(gap.max()))),
            ],
        )  # fmt: skip
        lines += [
            "",
            "Sum of phases minus GPU total (us): "
            f"min {us(gap.min())}, median {us(float(np.median(gap)))}, max {us(gap.max())}; "
            f"largest |gap| is {sig2(float(np.max(np.abs(gap) / total)) * 100)}% of the total.",
        ]
        lines += [
            "",
            "Host end-to-end minus GPU total (us): "
            f"min {us(host_minus_gpu.min())}, median {us(float(np.median(host_minus_gpu)))}, "
            f"p95 {us(float(np.percentile(host_minus_gpu, 95)))}, max {us(host_minus_gpu.max())}; "
            f"{int((host_minus_gpu < 0).sum())} of {len(total)} iterations negative.",
            "",
        ]
    return lines


def section_warmup(run: Run) -> list[str]:
    lines = [
        "## Warmup",
        "",
        "Each row is a separate process. Values are GPU total per measured iteration; the "
        "ratio compares an iteration with the median of iterations 5 and later.",
        "",
    ]
    rows = []
    for name in run.names("warmup/"):
        doc = run.doc(name)
        label = name.split("/")[1]
        if doc is None:
            rows.append((label, outcome(run.step(name)), *["-"] * 6))
            continue
        total = raw_series(doc)[GPU_TOTAL]
        settled = statistics.median(total[5:]) if len(total) > 5 else statistics.median(total)
        rows.append(
            (
                label,
                outcome(run.step(name)),
                doc["config"]["warmup"],
                ms(total[0]),
                ms(total[1]),
                ms(total[2]),
                ms(settled),
                sig2(total[0] / settled if settled else None),
            )
        )
    lines += table(
        [
            "run",
            "outcome",
            "warmup",
            "iter 0 ms",
            "iter 1 ms",
            "iter 2 ms",
            "median iter 5+ ms",
            "iter 0 / settled",
        ],
        rows,
    )
    return lines


def noise_tables(run: Run, workload: str) -> list[str]:
    docs = [(n, run.doc(n)) for n in run.names(f"noise/{workload}-")]
    docs = [(n, d) for n, d in docs if d is not None and "samples" in d]
    lines = [f"### {workload}", ""]
    if len(docs) < 2:
        return [*lines, f"fewer than two usable runs ({len(docs)})", ""]
    for metric in (GPU_TOTAL, "kernel"):
        rows, medians, ses = [], [], []
        for name, doc in docs:
            x = raw_series(doc)[metric]
            med = statistics.median(x)
            se = SE_FACTOR_P50 * MAD_TO_SIGMA * mad(x) / math.sqrt(len(x))
            medians.append(med)
            ses.append(se)
            rows.append((name.split("-")[-1], len(x), ms(med), ms(MAD_TO_SIGMA * mad(x)), us(se)))
        lines += [f"**{metric}**", ""]
        lines += table(
            ["run", "n", "median ms", "robust sigma ms", "predicted SE of median us"], rows
        )
        between = statistics.stdev(medians)
        predicted = statistics.mean(ses)
        lines += [
            "",
            f"Between-run SD of the medians: {us(between)} us; range "
            f"{us(max(medians) - min(medians))} us; "
            f"mean within-run predicted SE: {us(predicted)} us; "
            f"ratio between / predicted: {sig2(between / predicted if predicted else None)}. "
            "A ratio near 1 is what independent iterations and stable runs predict; "
            "well above 1 means run-to-run variation the within-run model does not see.",
            "",
        ]

        pair_rows, counts = [], {v: 0 for v in Verdict}
        for (na, a), (nb, b) in itertools.combinations(docs, 2):
            xa, xb = raw_series(a)[metric], raw_series(b)[metric]
            result = compare_series(metric, xa, xb, "p50", 5.0)
            counts[result.verdict] += 1
            exceeds = abs(result.delta) > noise_floor(xa, xb, "p50")
            pair_rows.append(
                (
                    f"{na.split('-')[-1]} vs {nb.split('-')[-1]}",
                    f"{result.percent:+.2f}%",
                    us(abs(result.delta)),
                    us(result.noise_floor),
                    "yes" if exceeds else "no",
                    result.verdict.value,
                )
            )
        lines += ["Same code, compared pairwise (A/A) with the current rule at T = 5%:", ""]
        lines += table(
            ["pair", "change", "|delta| us", "noise floor us", "beyond noise floor", "verdict"],
            pair_rows,
        )
        beyond = sum(1 for r in pair_rows if r[4] == "yes")
        lines += [
            "",
            f"{beyond} of {len(pair_rows)} unchanged pairs differ by more than the noise "
            f"floor; verdicts: {', '.join(f'{v.value} {c}' for v, c in counts.items() if c)}.",
            "",
        ]
    return lines


def section_noise(run: Run) -> list[str]:
    lines = ["## Noise: within-run versus between-run", ""]
    for workload in ("vector_add", "reduction"):
        lines += noise_tables(run, workload)
    return lines


def section_workflow(run: Run) -> list[str]:
    lines = ["## Baseline workflow", ""]
    rows = []
    for name in run.names("workflow/"):
        out = run.text(name) or ""
        verdict = re.findall(r"-> (PASS|INCONCLUSIVE|REGRESSION|IMPROVEMENT): ([^\n]*)", out)
        detail = "; ".join(f"{v}: {why}" for v, why in verdict)
        err = (run.text(name, "stderr") or "").strip().splitlines()
        rows.append(
            (name.split("/")[1], outcome(run.step(name)), detail or (err[-1] if err else ""))
        )
    lines += (
        table(["step", "outcome", "verdict or last stderr line"], rows) if rows else ["not run"]
    )
    return lines


def report(directory: Path) -> str:
    run = Run(directory)
    meta = (
        json.loads((directory / "meta.json").read_text())
        if (directory / "meta.json").is_file()
        else {}
    )
    commit = (run.text("env/git-commit") or "").strip() or "unknown"
    lines = [
        f"# GPU validation report: {directory.name}",
        "",
        f"Commit {commit}. Quick mode: {meta.get('quick')}. "
        f"CUDA_MODULE_LOADING={meta.get('env', {}).get('CUDA_MODULE_LOADING')}.",
        "",
    ]
    for section in (
        section_steps,
        section_environment,
        section_architecture,
        section_correctness_summary,
        section_vector_add,
        section_sanitizer,
        section_reduction,
        section_timing,
        section_warmup,
        section_noise,
        section_workflow,
    ):
        lines += [*section(run), ""]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    directory = Path(sys.argv[1])
    text = report(directory)
    (directory / "report.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

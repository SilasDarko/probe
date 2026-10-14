"""Plain-text rendering of results and comparisons for the terminal."""

import math

from probe.compare import MetricComparison
from probe.result_schema import GPU_TOTAL, HOST_END_TO_END, metric_names, stored_stats

_LABELS = {
    "h2d": "H2D",
    "kernel": "kernel",
    "d2h": "D2H",
    GPU_TOTAL: "GPU total",
    HOST_END_TO_END: "host end-to-end",
}


def label(metric: str) -> str:
    return _LABELS.get(metric, metric)


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f} ms"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f} ms"


def fmt_pct(value: float) -> str:
    if math.isinf(value):
        return "n/a (baseline 0)"
    return f"{value:+.1f}%"


def short_commit(doc: dict) -> str:
    prov = doc.get("provenance", {})
    commit = prov.get("git_commit")
    if not commit:
        return "unknown commit"
    return commit[:10] + (" (dirty)" if prov.get("git_dirty") else "")


def describe(doc: dict) -> str:
    params = "  ".join(f"{k}={v}" for k, v in doc["parameters"].items())
    device = doc["environment"]["device"]
    config = doc["config"]
    return (
        f"{doc['workload']}  {params}\n"
        f"GPU {device['name']} (cc {device['compute_capability']}), "
        f"CUDA runtime {device['cuda_runtime_version']}, driver {device['cuda_driver_version']}\n"
        f"build {doc['build']['build_type']}, architectures {doc['build']['cuda_architectures']}\n"
        f"warmup {config['warmup']}, iterations {config['iterations']}, "
        f"commit {short_commit(doc)}"
    )


def result_summary(doc: dict) -> str:
    lines = [describe(doc), ""]
    lines.append(f"  {'metric':<16} {'p50':>12} {'p95':>12} {'min':>12}")
    stats = stored_stats(doc)
    for metric in metric_names(doc):
        s = stats[metric]
        if s is None:
            lines.append(f"  {label(metric):<16} {'(no statistics)':>12}")
            continue
        lines.append(
            f"  {label(metric):<16} {fmt_ms(s['p50']):>12} {fmt_ms(s['p95']):>12} "
            f"{fmt_ms(s['min']):>12}"
        )
    lines.append("")
    if doc["valid"]:
        v = doc["validation"]
        lines.append(f"result: VALID ({v['iterations_checked']} iterations validated)")
    else:
        lines.append("result: INVALID, timings must not be used")
        lines += [f"  - {reason}" for reason in doc["invalid_reasons"]]
    return "\n".join(lines)


def comparison_table(rows: list[MetricComparison], gated_metric: str) -> str:
    stat = rows[0].stat if rows else "p50"
    header = (
        f"  {'metric':<16} {'base ' + stat:>12} {'cur ' + stat:>12} {'delta':>13} "
        f"{'change':>9} {'base p95':>12} {'cur p95':>12} {'noise floor':>12} "
        f"{'detectable':>12}  verdict"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for r in rows:
        marker = "*" if r.metric == gated_metric else " "
        lines.append(
            f"{marker} {label(r.metric):<16} {fmt_ms(r.baseline):>12} {fmt_ms(r.current):>12} "
            f"{fmt_delta(r.delta):>13} {fmt_pct(r.percent):>9} {fmt_ms(r.baseline_p95):>12} "
            f"{fmt_ms(r.current_p95):>12} {fmt_ms(r.noise_floor):>12} "
            f"{fmt_ms(r.min_detectable):>12}  {r.verdict.value}"
        )
    lines.append(
        "  (* = gated metric; noise floor: estimated from the samples; detectable: the larger "
        "of the noise floor and the clock's resolution)"
    )
    return "\n".join(lines)

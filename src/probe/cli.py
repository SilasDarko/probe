"""The ``probe`` command line.

Workflow model: every command that needs a measurement runs the workload
itself, unless it is given an explicit result file. There is no hidden "last
run" state:

    probe run vector-add --size 10000000          # writes results/<...>.json
    probe baseline save vector-add --name main    # runs, then saves as 'main'
    probe baseline save --name main --from results/<...>.json   # promotes a file
    probe compare vector-add --baseline main      # runs with main's parameters
    probe compare vector-add --baseline main --current results/<...>.json
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from probe import __version__, exit_codes
from probe.baseline import (
    default_store,
    delete_baseline,
    list_baselines,
    load_baseline,
    read_json,
    require_usable_result,
    save_baseline,
)
from probe.compare import (
    DEFAULT_THRESHOLD_PCT,
    SUPPORTED_STATS,
    Verdict,
    compare_series,
)
from probe.compat import check_compatibility
from probe.exit_codes import ProbeError
from probe.report import comparison_table, describe, fmt_pct, label, result_summary
from probe.result_schema import GPU_TOTAL, metric_names, raw_series
from probe.runner import canonical_workload, find_binary, run_workload

DEFAULT_METRIC = GPU_TOTAL


def _workload(text: str) -> str:
    try:
        return canonical_workload(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got '{text}'") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _percentage(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got '{text}'") from exc
    if not value >= 0:  # also rejects NaN
        raise argparse.ArgumentTypeError("must be a non-negative percentage")
    return value


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--store", type=Path, default=None, help="baseline store (default: $PROBE_STORE or .probe)"
    )
    common.add_argument("--binary", type=Path, default=None, help="path to probe-workloads")

    workload_options = argparse.ArgumentParser(add_help=False)
    group = workload_options.add_argument_group("workload options")
    group.add_argument("--size", type=_non_negative_int, help="number of elements")
    group.add_argument("--block-size", dest="block_size", type=_non_negative_int)
    group.add_argument("--warmup", type=_non_negative_int)
    group.add_argument("--iters", type=_non_negative_int, help="measured iterations")
    group.add_argument("--seed", type=_non_negative_int)

    parser = argparse.ArgumentParser(
        prog="probe", description="GPU performance regression and diagnostics toolkit."
    )
    parser.add_argument("--version", action="version", version=f"probe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common, workload_options], help="run a workload")
    run.add_argument("workload", type=_workload)
    run.add_argument("--output", type=Path, help="result file (default: results/<auto>.json)")

    baseline = sub.add_parser("baseline", help="manage baselines")
    bsub = baseline.add_subparsers(dest="baseline_command", required=True)
    save = bsub.add_parser(
        "save", parents=[common, workload_options], help="run a workload and save it"
    )
    save.add_argument("workload", nargs="?", type=_workload)
    save.add_argument("--name", required=True)
    save.add_argument("--from", dest="from_file", type=Path, help="save an existing result file")
    save.add_argument("--force", action="store_true", help="replace an existing baseline")
    bsub.add_parser("list", parents=[common], help="list baselines")
    show = bsub.add_parser("show", parents=[common], help="show one baseline")
    show.add_argument("name")
    delete = bsub.add_parser("delete", parents=[common], help="delete one baseline")
    delete.add_argument("name")

    compare = sub.add_parser(
        "compare", parents=[common, workload_options], help="compare against a baseline"
    )
    compare.add_argument("workload", type=_workload)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--current", type=Path, help="compare this result file instead of running")
    compare.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"metric that decides the outcome (default: {DEFAULT_METRIC}); "
        "a phase name such as kernel, or gpu_total / host_end_to_end",
    )
    compare.add_argument("--stat", choices=SUPPORTED_STATS, default="p50")
    compare.add_argument(
        "--fail-if-regression",
        dest="fail_if_regression",
        type=_percentage,
        metavar="PCT",
        help="exit 1 if the metric regresses by more than PCT percent",
    )
    compare.add_argument(
        "--fail-on-inconclusive",
        action="store_true",
        help="exit 8 if the result for the metric is inconclusive",
    )
    compare.add_argument("--allow-environment-mismatch", action="store_true")
    compare.add_argument("--output", type=Path, help="where to write the current result file")
    return parser


# --- helpers -----------------------------------------------------------------------


def _store(args) -> Path:
    return args.store if args.store is not None else default_store()


def _options(args) -> dict[str, int | None]:
    return {key: getattr(args, key) for key in ("size", "block_size", "warmup", "iters", "seed")}


def _default_output(workload: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("results") / f"{workload}-{stamp}.json"


def _write_result(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


def _run(args, workload: str, options: dict, argv: list[str]) -> dict:
    binary = find_binary(args.binary)
    return run_workload(binary, workload, options, ["probe", *argv])


# --- commands -----------------------------------------------------------------------


def cmd_run(args, argv) -> int:
    doc = _run(args, args.workload, _options(args), argv)
    path = args.output or _default_output(args.workload)
    _write_result(doc, path)
    print(result_summary(doc))
    print(f"\nresult written to {path}")
    return exit_codes.OK if doc["valid"] else exit_codes.INVALID_RESULT


def cmd_baseline_save(args, argv) -> int:
    if args.from_file is not None:
        if any(v is not None for v in _options(args).values()):
            raise ProbeError(exit_codes.USAGE, "workload options cannot be combined with --from")
        doc = read_json(args.from_file, "result file")
        if args.workload is not None and isinstance(doc, dict):
            recorded = doc.get("workload", "").replace("_", "-")
            if recorded != args.workload:
                raise ProbeError(
                    exit_codes.INCOMPATIBLE,
                    f"{args.from_file} is a '{recorded}' result, not '{args.workload}'",
                )
    else:
        if args.workload is None:
            raise ProbeError(exit_codes.USAGE, "give a workload to run, or --from FILE")
        doc = _run(args, args.workload, _options(args), argv)

    path = save_baseline(_store(args), args.name, doc, overwrite=args.force)
    print(describe(doc))
    print(f"\nbaseline '{args.name}' saved to {path}")
    return exit_codes.OK


def cmd_baseline_list(args, argv) -> int:
    entries = list_baselines(_store(args))
    if not entries:
        print(f"no baselines in {_store(args)}")
        return exit_codes.OK
    status = exit_codes.OK
    for name, doc, error in entries:
        if doc is None:
            print(f"{name:<20} UNREADABLE: {error}")
            status = exit_codes.BAD_FILE
            continue
        r = doc["result"]
        params = ", ".join(
            f"{k}={v}" for k, v in r["parameters"].items() if k in ("n", "block_size")
        )
        gpu = r["environment"]["device"]["name"]
        commit = (r["provenance"].get("git_commit") or "-")[:10]
        print(
            f"{name:<20} {r['workload']:<12} {params:<28} {gpu:<28} {commit:<10}"
            f"  saved {doc['saved_utc']}"
        )
    return status


def cmd_baseline_show(args, argv) -> int:
    doc = load_baseline(_store(args), args.name)
    print(f"baseline '{args.name}' (saved {doc['saved_utc']})\n")
    print(result_summary(doc["result"]))
    return exit_codes.OK


def cmd_baseline_delete(args, argv) -> int:
    path = delete_baseline(_store(args), args.name)
    print(f"deleted baseline '{args.name}' ({path})")
    return exit_codes.OK


def cmd_compare(args, argv) -> int:
    baseline = load_baseline(_store(args), args.baseline)["result"]
    base_workload = baseline["workload"].replace("_", "-")
    if base_workload != args.workload:
        raise ProbeError(
            exit_codes.INCOMPATIBLE,
            f"baseline '{args.baseline}' is a '{base_workload}' baseline, not '{args.workload}'",
        )

    if args.current is not None:
        if any(v is not None for v in _options(args).values()):
            raise ProbeError(exit_codes.USAGE, "workload options cannot be combined with --current")
        current = read_json(args.current, "current result file")
    else:
        # Re-run with the baseline's own parameters, so the two runs measure
        # the same problem. Explicit options override (and are then checked).
        options = {
            "size": baseline["parameters"].get("n"),
            "block_size": baseline["parameters"].get("block_size"),
            "warmup": baseline["config"]["warmup"],
            "iters": baseline["config"]["iterations"],
            "seed": baseline["config"]["seed"],
        }
        options.update({k: v for k, v in _options(args).items() if v is not None})
        current = _run(args, args.workload, options, argv)
        path = args.output or _default_output(args.workload)
        _write_result(current, path)
        print(f"current result written to {path}\n")

    require_usable_result(current, "current result")

    compat = check_compatibility(baseline, current)
    for line in compat.fatal:
        print(f"FATAL: {line}", file=sys.stderr)
    for line in compat.environment:
        prefix = "WARNING (overridden)" if args.allow_environment_mismatch else "FATAL"
        print(f"{prefix}: {line}", file=sys.stderr)
    for line in compat.warnings:
        print(f"warning: {line}", file=sys.stderr)
    if compat.blocked(args.allow_environment_mismatch):
        hint = "" if compat.fatal else " (use --allow-environment-mismatch to compare anyway)"
        raise ProbeError(exit_codes.INCOMPATIBLE, "runs are not comparable" + hint)
    if compat.environment:
        print(
            "!!! ENVIRONMENT MISMATCH OVERRIDDEN: these runs come from different GPUs or "
            "builds; differences may reflect the environment, not the code !!!",
            file=sys.stderr,
        )

    metrics = metric_names(baseline)
    if args.metric not in metrics:
        raise ProbeError(
            exit_codes.USAGE,
            f"unknown metric '{args.metric}' for {args.workload} "
            f"(choose from {', '.join(metrics)})",
        )

    threshold = (
        args.fail_if_regression if args.fail_if_regression is not None else DEFAULT_THRESHOLD_PCT
    )
    base_series, cur_series = raw_series(baseline), raw_series(current)
    rows = [compare_series(m, base_series[m], cur_series[m], args.stat, threshold) for m in metrics]
    gated = next(r for r in rows if r.metric == args.metric)

    print(f"baseline '{args.baseline}': {describe(baseline)}\n")
    print(f"current: commit {current['provenance'].get('git_commit') or 'unknown'}\n")
    print(comparison_table(rows, args.metric))
    print(
        f"\n{label(gated.metric)} {gated.stat}: {fmt_pct(gated.percent)}, "
        f"threshold {threshold:g}%"
        + ("" if args.fail_if_regression is not None else " (default, informational)")
        + f" -> {gated.verdict.value}: {gated.reason}"
    )

    if gated.verdict is Verdict.REGRESSION and args.fail_if_regression is not None:
        return exit_codes.REGRESSION
    if gated.verdict is Verdict.INCONCLUSIVE and args.fail_on_inconclusive:
        return exit_codes.INCONCLUSIVE
    return exit_codes.OK


COMMANDS = {
    ("run", None): cmd_run,
    ("baseline", "save"): cmd_baseline_save,
    ("baseline", "list"): cmd_baseline_list,
    ("baseline", "show"): cmd_baseline_show,
    ("baseline", "delete"): cmd_baseline_delete,
    ("compare", None): cmd_compare,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)  # exits with code 2 on usage errors
    handler = COMMANDS[(args.command, getattr(args, "baseline_command", None))]
    try:
        return handler(args, argv)
    except ProbeError as exc:
        print(f"probe: {exc}", file=sys.stderr)
        return exc.code

"""Agent optimization loop: crash-isolated candidate evals with a keep/revert experiment log.

The loop protocol (docs/loop.md): pick a target op, edit one candidate module under
`popcorn/impls/`, run `try` after every change, and keep or revert on its verdict.
Each case runs in its own subprocess so a crashing kernel cannot poison the parent's
CUDA context. The log is an untracked scratchpad in the user cache; evidence enters
the report database only through the normal grid (`python -m popcorn.bench run`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from itertools import chain, repeat
from pathlib import Path
from typing import Any

import popcorn.kernels  # noqa: F401
from popcorn import KERNELS
from popcorn.bench.compare import compare_inputs
from popcorn.bench.grid import make_inputs, sample_cases
from popcorn.bench.model import Case, Record
from popcorn.bench.store import Store, user_reports
from popcorn.core.config import device_name
from popcorn.core.dispatcher import Dispatcher
from popcorn.core.fingerprint import fingerprint
from popcorn.core.sources import resolve

IMPROVEMENT = 0.99  # a keep needs at least a 1% total-time gain over the best kept row


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _op(name: str) -> Dispatcher:
    if name not in KERNELS:
        raise SystemExit(f"unknown op {name!r}; registered: {', '.join(sorted(KERNELS))}")
    return KERNELS[name]


def _candidate(op: str, impl: str | None) -> tuple[str, Any]:
    """The candidate callable: `--impl`, or the first-party convention `popcorn.impls.<op>_{tl,cu}.<op>`."""
    paths = [impl] if impl else [f"popcorn.impls.{op}_{suffix}.{op}" for suffix in ("tl", "cu")]
    failures = []
    for path in paths:
        try:
            return path, resolve(path)
        except Exception as error:
            failures.append(f"  {path}: {type(error).__name__}: {error}")
    raise SystemExit("no candidate implementation:\n" + "\n".join(failures))


def _log_path(override: str | None, op: str) -> Path:
    return Path(override) if override else user_reports().parent / "loop" / f"{op}.jsonl"


def _read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _pair(bench: Mapping[str, Any], grad: bool, prefix: str = "") -> float | None:
    forward = bench.get(f"{prefix}fwd_ms")
    if forward is None:
        return None
    return forward + (bench.get(f"{prefix}bwd_ms", 0.0) if grad else 0.0)


def _total(rows: Sequence[Mapping[str, Any]], grad: bool, source: str) -> float | None:
    """Summed per-case time in ms for "mine", "ref", or "vs"; None when any timing is missing."""
    total = 0.0
    for row in rows:
        bench = ((row.get("vs") or {}) if source == "vs" else row).get("bench") or {}
        value = _pair(bench, grad, "ref_" if source == "ref" else "")
        if value is None:
            return None
        total += value
    return total


# ---------------------------------------------------------------- child (one case, own process)


def _time_backend(op: Dispatcher, backend: str, case: Case, device: str, grad: bool, reps: int) -> dict[str, Any]:
    """Advisory incumbent timing on the same inputs; its correctness remains the grid's job."""
    try:
        chosen = op[backend]
        inputs = make_inputs(op, case, device, 0, grad)
        if reason := chosen.rejects(op._values(inputs), inputs):
            return {"backend": backend, "status": "skip", "reason": reason, "bench": {}}
        result = compare_inputs(
            lambda **arguments: chosen.invoke(arguments),
            op.reference,
            repeat(inputs, reps),
            backward=grad,
            repeats=reps,
        )
    except Exception as error:
        return {"backend": backend, "status": "error", "reason": f"{type(error).__name__}: {error}", "bench": {}}
    return {"backend": backend, "status": result.status, "reason": result.reason, "bench": result.bench}


def cmd_child(args: argparse.Namespace) -> None:
    op = _op(args.op)
    case = Case.from_config(json.loads(args.case))
    grad = not args.forward_only
    candidate = resolve(args.impl)
    try:
        first = make_inputs(op, case, args.device, 0, grad)
    except Exception as error:
        payload: dict[str, Any] = {"status": "error", "reason": f"inputs: {type(error).__name__}: {error}", "bench": {}}
    else:
        trials = chain((first,), (make_inputs(op, case, args.device, seed, grad) for seed in range(1, args.reps)))
        result = compare_inputs(candidate, op.reference, trials, backward=grad, repeats=args.reps)
        payload = {"status": result.status, "reason": result.reason, "bench": result.bench, "bench_error": result.bench_error}
        if args.vs:
            payload["vs"] = _time_backend(op, args.vs, case, args.device, grad, args.reps)
    Path(args.out).write_text(json.dumps(payload))


# ------------------------------------------------------------------------------- try (parent)


def _run_child(args: argparse.Namespace, impl: str, case: Case) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = Path(handle.name)
    command = [sys.executable, "-m", "popcorn.bench.loop", "child", "--op", args.op, "--impl", impl]
    command += ["--case", json.dumps(case.config()), "--device", args.device, "--reps", str(args.reps), "--out", str(out)]
    if args.forward_only:
        command.append("--forward-only")
    if args.vs:
        command += ["--vs", args.vs]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        out.unlink(missing_ok=True)
        return {"status": "hang", "reason": f"no result within {args.timeout}s; worker killed", "bench": {}}
    try:
        return json.loads(out.read_text())
    except (OSError, ValueError):
        tail = "; ".join((process.stderr or process.stdout).strip().splitlines()[-8:])
        return {"status": "crash", "reason": f"worker exit {process.returncode}: {tail}", "bench": {}}
    finally:
        out.unlink(missing_ok=True)


def _line(case: Case, row: Mapping[str, Any]) -> str:
    parts = [f"  [{row.get('status', '?')}] {case}"]
    bench = row.get("bench") or {}
    if "fwd_ms" in bench:
        timing = f"fwd {bench['fwd_ms']:.3f}ms"
        if "bwd_ms" in bench:
            timing += f" bwd {bench['bwd_ms']:.3f}ms"
        if "ref_fwd_ms" in bench:
            reference = f"{bench['ref_fwd_ms']:.3f}"
            if "bwd_ms" in bench and "ref_bwd_ms" in bench:
                reference += f"/{bench['ref_bwd_ms']:.3f}"
            timing += f" (ref {reference})"
        parts.append(timing)
    if (vs := row.get("vs")) is not None:
        vs_bench = vs.get("bench") or {}
        if "fwd_ms" in vs_bench:
            timing = f"{vs['backend']} {vs_bench['fwd_ms']:.3f}"
            if "bwd_ms" in vs_bench:
                timing += f"/{vs_bench['bwd_ms']:.3f}"
            parts.append(timing + "ms")
        else:
            parts.append(f"{vs['backend']} [{vs.get('status')}]")
    if reason := row.get("reason"):
        parts.append(f"({reason[:160]})")
    return "  ".join(parts)


def _baseline(entries: Sequence[Mapping[str, Any]], sample: str, grad: bool, device: str, reps: int) -> float | None:
    kept = [
        entry["total_ms"]
        for entry in entries
        if entry.get("verdict") == "keep"
        and entry.get("sample") == sample
        and entry.get("grad") == grad
        and entry.get("device") == device
        and entry.get("reps") == reps
        and entry.get("total_ms") is not None
    ]
    return min(kept, default=None)


def _verdict(rows: Sequence[Mapping[str, Any]], total: float | None, baseline: float | None) -> tuple[str, str]:
    failing = [row for row in rows if row.get("status") != "pass"]
    if failing:
        first = failing[0]
        return "revert", f"{len(failing)}/{len(rows)} cases not passing; first: [{first['status']}] {first.get('reason', '')}"
    if total is None:
        return "revert", "passed but not timed (bench_error); rerun"
    if baseline is None:
        return "keep", f"first passing baseline: {total:.3f}ms"
    if total < baseline * IMPROVEMENT:
        return "keep", f"{total:.3f}ms improves best kept {baseline:.3f}ms by {(1 - total / baseline) * 100:.1f}%"
    return "revert", f"{total:.3f}ms does not improve best kept {baseline:.3f}ms by >=1%"


def cmd_try(args: argparse.Namespace) -> None:
    op = _op(args.op)
    impl, candidate = _candidate(args.op, args.impl)
    grad = not args.forward_only
    sample = sample_cases(op, args.sample)
    if not sample:
        raise SystemExit(f"{op.name}: empty case grid")
    device = device_name(args.device)
    mode = "forward+backward" if grad else "forward only"
    print(f"{op.name} <- {impl} ({len(sample)} cases, {mode}, {args.reps} reps, {args.timeout}s/case timeout)")
    rows = []
    for case in sample:
        rows.append(_run_child(args, impl, case))
        print(_line(case, rows[-1]))

    total = _total(rows, grad, "mine")
    reference = _total(rows, grad, "ref")
    incumbent = _total(rows, grad, "vs") if args.vs else None
    log = _log_path(args.log, op.name)
    sample_key = hashlib.sha1(",".join(case.case_id for case in sample).encode()).hexdigest()[:12]
    baseline = _baseline(_read_log(log), sample_key, grad, device, args.reps)
    verdict, reason = _verdict(rows, total, baseline)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "op": op.name,
        "impl": impl,
        "impl_hash": fingerprint(candidate),
        "device": device,
        "grad": grad,
        "sample": sample_key,
        "reps": args.reps,
        "tag": args.tag,
        "note": args.note,
        "total_ms": total,
        "ref_total_ms": reference,
        "vs": args.vs,
        "vs_total_ms": incumbent,
        "verdict": verdict,
        "reason": reason,
        "results": [{"case": str(case), **row} for case, row in zip(sample, rows)],
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as sink:
        sink.write(json.dumps(entry, sort_keys=True) + "\n")
    references = (("ref", reference), (args.vs, incumbent))
    context = "".join(f"  {name} {value:.3f}ms" for name, value in references if value is not None)
    print(f"VERDICT: {verdict.upper()}  {reason}{context}  -> {log}")
    raise SystemExit(0 if verdict == "keep" else 1)


# ------------------------------------------------------------------------------------ targets


def _best_speedups(records: Sequence[Record]) -> list[float]:
    """Per (case, grad mode), the best recorded implementation speedup over the reference."""
    best: dict[tuple[str, bool], float] = {}
    for record in records:
        if record.result.status != "pass":
            continue
        mine = _pair(record.result.bench, record.result.grad)
        reference = _pair(record.result.bench, record.result.grad, "ref_")
        if mine and reference:
            key = (record.case_id, record.result.grad)
            best[key] = max(best.get(key, 0.0), reference / mine)
    return list(best.values())


def cmd_targets(args: argparse.Namespace) -> None:
    store = Store()
    rows: list[tuple[float, float, str]] = []
    for name in args.ops or sorted(KERNELS):
        op = _op(name)
        records = [record for record in store.merged(name) if record.impl != "torch"]
        if args.hardware:
            records = [record for record in records if args.hardware.lower() in record.environment.device.lower()]
        tested: dict[tuple[str, bool], list[Record]] = defaultdict(list)
        for record in records:
            tested[(record.case_id, record.result.grad)].append(record)
        passing = {key for key, group in tested.items() if any(record.result.status == "pass" for record in group)}
        grad_keys = {key for key in tested if key[1]}
        coverage = len(passing) / len(tested) if tested else 0.0
        grad_coverage = len(passing & grad_keys) / len(grad_keys) if grad_keys else 0.0
        speedups = _best_speedups(records)
        median = statistics.median(speedups) if speedups else None
        backends = ",".join(sorted({record.impl for record in records})) or "-"
        line = (
            f"{name:<28} {backends[:24]:<24} {len(tested):>6} {coverage:>7.0%} {grad_coverage:>7.0%} "
            f"{f'{median:.2f}x' if median is not None else '-':>8} {len(op.available_backends()) - 1:>4}"
        )
        rows.append((median or 0.0, coverage, line))
    rows.sort()
    print(f"{'op':<28} {'recorded backends':<24} {'tested':>6} {'pass%':>7} {'grad%':>7} {'x best':>8} {'reg':>4}")
    for _, _, line in rows[: args.top] if args.top else rows:
        print(line)
    print("\nheadroom reads top-down: no recorded rows, low best-backend speedup, or low grad coverage.")


# ------------------------------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> None:
    if args.log:
        paths = [Path(args.log)]
    else:
        directory = user_reports().parent / "loop"
        paths = [directory / f"{name}.jsonl" for name in args.ops] if args.ops else sorted(directory.glob("*.jsonl"))
    shown = 0
    for path in paths:
        entries = _read_log(path)
        if not entries:
            continue
        shown += 1
        kept = [entry for entry in entries if entry.get("verdict") == "keep"]
        timed = [entry for entry in kept if entry.get("total_ms") is not None]
        streak = next((index for index, entry in enumerate(reversed(entries)) if entry.get("verdict") == "keep"), len(entries))
        summary = f"{entries[-1].get('op', path.stem)}: {len(entries)} experiments, {len(kept)} kept, {streak} since last keep"
        if timed:
            best = min(timed, key=lambda entry: entry["total_ms"])
            summary += f"; best {best['total_ms']:.3f}ms ({best.get('tag') or best['ts']})"
        print(summary)
        for entry in entries[-args.last :]:
            marker = "+" if entry.get("verdict") == "keep" else "-"
            total = f"{entry['total_ms']:.3f}ms" if entry.get("total_ms") is not None else "-"
            print(f"  {marker} {(entry.get('tag') or entry['ts']):<26} {total:>10}  {entry.get('reason', '')[:90]}")
    if not shown:
        print("no experiment logs found")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m popcorn.bench.loop",
        description="Optimization loop: rank targets, evaluate a candidate after each edit, track keep/revert history.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    targets = sub.add_parser("targets", help="rank ops by recorded headroom (worst-served first)")
    targets.add_argument("ops", nargs="*", help="ops to rank (default: all registered)")
    targets.add_argument("--hardware", help="only count report rows whose device name contains this substring")
    targets.add_argument("--top", type=_positive, help="show only the first N rows")
    targets.set_defaults(fn=cmd_targets)

    trial = sub.add_parser("try", help="evaluate one candidate on a fixed case sample; exit 0 = keep, 1 = revert")
    trial.add_argument("op")
    trial.add_argument("--impl", help="dotted path to the candidate (default: popcorn.impls.<op>_tl.<op>, then _cu)")
    trial.add_argument(
        "--sample",
        "--cases",
        dest="sample",
        type=_positive,
        default=4,
        help="deterministic random Cartesian sample size (stable across runs; --cases is deprecated)",
    )
    trial.add_argument("--reps", type=_positive, default=5, help="seeded correctness draws and timing reps")
    trial.add_argument("--device", default="cuda")
    trial.add_argument("--forward-only", action="store_true", help="skip backward grading and timing")
    trial.add_argument("--vs", help="also time this registered backend on the same inputs (advisory)")
    trial.add_argument("--timeout", type=_positive, default=240, help="seconds per case worker before it is killed")
    trial.add_argument("--tag", default="", help="short experiment identifier for the log")
    trial.add_argument("--note", default="", help="what was tried, for the log")
    trial.add_argument("--log", help="experiment log path (default: user cache, per op)")
    trial.set_defaults(fn=cmd_try)

    status = sub.add_parser("status", help="summarize experiment logs: kept/reverted, best, plateau streak")
    status.add_argument("ops", nargs="*", help="ops to show (default: every op with a log)")
    status.add_argument("--last", type=_positive, default=5, help="recent experiments to list per op")
    status.add_argument("--log", help="read this log file instead of the user cache")
    status.set_defaults(fn=cmd_status)

    child = sub.add_parser("child", help="internal: grade one case in an isolated process")
    child.add_argument("--op", required=True)
    child.add_argument("--impl", required=True)
    child.add_argument("--case", required=True, help="case config JSON")
    child.add_argument("--device", required=True)
    child.add_argument("--reps", type=_positive, required=True)
    child.add_argument("--forward-only", action="store_true")
    child.add_argument("--vs")
    child.add_argument("--out", required=True, help="write the result JSON here")
    child.set_defaults(fn=cmd_child)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

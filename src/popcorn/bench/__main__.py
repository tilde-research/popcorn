"""CLI harness: run/merge/view/map locally or submit as a slurm array."""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch

import popcorn.kernels  # noqa: F401
from popcorn import KERNELS
from popcorn.bench import hub
from popcorn.bench.grid import backward_safe, case_plan, sample_cases
from popcorn.bench.live import DEFAULT_PORT, LiveServer
from popcorn.bench.model import Case, Record
from popcorn.bench.plan import EFFORT, EXHAUSTED, BudgetFrontier, adaptive_plan
from popcorn.bench.readme import refresh
from popcorn.bench.service import INCOMPLETE
from popcorn.bench.store import (
    BUNDLED_REPORTS,
    PUBLISHED,
    Store,
    audit_records,
    matching,
    prefer,
    read,
    read_file,
    scrub,
    unmeasured,
    user_reports,
    write,
)
from popcorn.bench.sweep import estimated, infeasible
from popcorn.bench.viewer import render
from popcorn.core.config import device_name
from popcorn.core.dispatcher import Dispatcher, Implementation
from popcorn.core.sources import available, installed_version

SBATCH = """\
#!/bin/bash
#SBATCH --job-name=popcorn
#SBATCH --array=0-{last}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
{qos_line}#SBATCH --time={time}
#SBATCH --output=logs/popcorn_%A_%a.log

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"
export TRITON_CACHE_DIR="/tmp/triton_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TRITON_CACHE_DIR"

{python} -m popcorn.bench run {ops} --reps {reps} {limit} {hardware} --timeout {timeout} {only} \
    --shard "$SLURM_ARRAY_TASK_ID/{shards}" --out {shard_dir}/"$SLURM_ARRAY_TASK_ID".jsonl
"""

HARDWARE_HELP = "fail unless the resolved device name contains this substring (e.g. H100)"


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return port


def _shard(value: str) -> tuple[int, int]:
    try:
        index, count = map(int, value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("must be I/K") from None
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("must satisfy 0 <= I < K")
    return index, count


def _target_device(args: argparse.Namespace) -> str:
    """Pin the device and confirm it is the hardware the run claims to target.

    A bare `cuda` follows whichever index happens to be current, and falls back to a CPU
    name when no GPU is visible, so rows could be stamped with hardware nobody intended.
    """
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise SystemExit(f"invalid --device {args.device!r}: {error}") from None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {args.device} requested but no CUDA device is visible")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    name = device_name(device)
    if args.hardware and args.hardware.lower() not in name.lower():
        raise SystemExit(f"--hardware {args.hardware!r} does not match {name!r} on {device}")
    print(f"recording on {device} ({name})")
    return str(device)


def _selected(only: str, ops: list[str], backend: str | None = None) -> list[tuple[Dispatcher, str, Case]]:
    """Rebuild exactly the cases named by recorded rows, from the config each row carries.

    Rebuilding beats intersecting with a fresh grid: Cartesian samples shift as pools change,
    so the sampled subset shifts whenever a pool or the sampler changes, and rows recorded
    before such a change would silently drop out of a rerun.
    """
    work = []
    seen = set()
    skipped: Counter[str] = Counter()
    for record in read_file(only):
        triple = (record.op, record.impl, record.case_id)
        if triple in seen or record.impl == "torch" or backend not in (None, record.impl):
            continue
        if ops and record.op not in ops:
            continue
        seen.add(triple)
        if record.op not in KERNELS:
            skipped[f"op {record.op!r} is not registered"] += 1
            continue
        op = KERNELS[record.op]
        if record.impl not in op.available_backends():
            skipped[f"impl {record.impl!r} is unavailable here"] += 1
            continue
        work.append((op, record.impl, Case.from_config(record.config)))
    print(f"--only: {len(work)} of {len(seen)} selected cases are runnable")
    for reason, count in skipped.most_common():
        print(f"  {count} skipped: {reason}")
    return work


def _work(
    ops: list[str], backend: str | None = None, limit: int | None = None, only: str | None = None
) -> list[tuple[Dispatcher, str, Case]]:
    if only is not None:
        if limit is not None:
            raise SystemExit("--only names exact cases; --limit does not apply")
        return _selected(only, ops, backend)
    work = []
    for op in (KERNELS[name] for name in ops or sorted(KERNELS)):
        op_cases = sample_cases(op, limit) if limit is not None else case_plan(op).flatten()
        for candidate in op.available_backends():
            if candidate != "torch" and backend in (None, candidate):
                work.extend((op, candidate, case) for case in op_cases)
    return work


def _isolated(
    op: Dispatcher,
    backend: str,
    case: Case,
    device: str,
    reps: int,
    timeout: int,
    grad: bool | None = None,
) -> Record:
    """Run one case in a worker process so a hang or a hard crash costs only that case."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = Path(handle.name)
    command = [sys.executable, "-m", "popcorn.bench", "run-case", "--op", op.name, "--impl", backend]
    command += ["--case", json.dumps(case.config()), "--device", device, "--reps", str(reps), "--out", str(out)]
    if grad is not None:
        command.append("--grad" if grad else "--no-grad")
    try:
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            record = op.bench.incomplete(backend, case, device, f"no result within {timeout}s")
            record.result.status = "timeout"
            if grad is not None:
                record.result.grad = grad
            return record
        try:
            return Record.from_dict(json.loads(out.read_text()))
        except (OSError, ValueError):
            tail = "; ".join((process.stderr or process.stdout).strip().splitlines()[-8:])
            record = op.bench.incomplete(backend, case, device, f"worker exit {process.returncode}: {tail}")
            record.result.status = "crash"
            if grad is not None:
                record.result.grad = grad
            return record
    finally:
        out.unlink(missing_ok=True)


def cmd_run_case(args: argparse.Namespace) -> None:
    """One case, one process: the isolated half of `run`. Not meant to be called directly."""
    op = KERNELS[args.op]
    record = op.bench.run_case(args.impl, Case.from_config(json.loads(args.case)), args.device, args.reps, grad=args.grad)
    Path(args.out).write_text(json.dumps(record.to_dict(), sort_keys=True))


def _publish(records: list[Record]) -> None:
    write(records, BUNDLED_REPORTS, PUBLISHED)
    matrix, _ = refresh(ops=KERNELS)
    print(matrix)


def _append(path: Path, record: Record) -> None:
    with path.open("a") as sink:
        sink.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def cmd_run(args: argparse.Namespace) -> None:
    args.device = _target_device(args)
    work = _work(args.ops, args.backend, args.limit, args.only)
    if args.shard:
        index, count = args.shard
        work = work[index * len(work) // count : (index + 1) * len(work) // count]
    if not work:
        raise SystemExit("no matching non-torch backend cases")
    isolation = "in-process" if args.in_process else f"{args.timeout}s/case worker"
    print(f"{len(work)} case-backend pairs, {args.reps} reps each ({isolation})")
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")
        for op, backend, case in work:
            _append(out, op.bench.incomplete(backend, case, args.device))

    records = []
    poisoned = False
    for index, (op, backend, case) in enumerate(work):
        if args.in_process:
            try:
                record = op.bench.run_case(backend, case, args.device, args.reps)
            except Exception as error:
                record = op.bench.incomplete(backend, case, args.device, f"{type(error).__name__}: {error}")
        else:
            record = _isolated(op, backend, case, args.device, args.reps, args.timeout)
        records.append(record)
        if out:
            _append(out, record)
        try:
            torch.zeros(1, device=args.device).item()
        except Exception as error:
            reason = f"device poisoned: {type(error).__name__}: {error}"
            record.result.status, record.result.reason = "crash", reason
            if out:
                _append(out, record)
            else:
                records.extend(
                    other.bench.incomplete(candidate, other_case, args.device, reason)
                    for other, candidate, other_case in work[index + 1 :]
                )
            print(f"{reason} after {record.op}:{record.impl} {record.case}")
            poisoned = True
            break
    if not out:
        _publish(records)
    failures = [
        record
        for record in records
        if record.result.status in ("fail", "crash", "error", "timeout") or record.result.bench_error
    ]
    for record in failures[:20]:
        result = record.result
        reason = result.reason or f"benchmark: {result.bench_error}"
        print(f"\n[{result.status}] {record.op}:{record.impl} {record.case}\n  {reason}")
    raise SystemExit(1 if failures or poisoned else 0)


def _pending(directory: Path, stored: Mapping[tuple, Record]) -> tuple[int, int, str]:
    """Rows in a shard directory, how many the store lacks, and why it could not be read.

    A shard run writes rows and leaves publishing to a separate merge step, so a merge
    that never ran (or died) is otherwise invisible: `shards/` is gitignored and the
    store reader does not recurse into it. Directories written by an older report schema
    are reported rather than raised on, so one stale directory cannot mask a live one.
    """
    rows = 0
    pending = 0
    for source in sorted(directory.glob("*.jsonl")):
        try:
            records = read_file(source)
        except ValueError as error:
            return rows, pending, f"{source.name}: {error}"
        rows += len(records)
        pending += sum(1 for record in records if record.key not in stored or prefer(record, stored[record.key]))
    return rows, pending, ""


def _require_clean(records: list[Record]) -> None:
    bad = [record for record in records if record.result.status in ("fail", "crash", "error") or record.result.bench_error]
    if bad:
        counts = Counter(record.result.status if not record.result.bench_error else "bench_error" for record in bad)
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        raise SystemExit(f"refusing to publish {len(bad)} unsuccessful row(s): {summary}")


def _require_passing_reports() -> None:
    current, passing, problems = audit_records(KERNELS.values())
    total = current + len(problems.get("missing", [])) + len(problems.get("stale", []))
    if passing != total:
        raise SystemExit(
            f"refusing to publish: only {passing}/{total} implementations have a current passing row; "
            "run `uv run python scripts/check_records.py --require-pass` for details"
        )


def cmd_merge(args: argparse.Namespace) -> None:
    if args.scrub_incomplete:
        if args.sources or args.check:
            raise SystemExit("merge --scrub-incomplete rewrites the whole store; pass no sources or --check")
        dropped = scrub(BUNDLED_REPORTS, lambda record: record.result.reason == INCOMPLETE)
        refresh(ops=KERNELS)
        print(f"dropped {dropped} incomplete sentinel row(s); README badges refreshed")
        return
    if args.check:
        if args.sources:
            raise SystemExit("merge --check scans the shard directories; pass no sources")
        directories = sorted(path for path in (BUNDLED_REPORTS / "shards").glob("*") if path.is_dir())
        if not directories:
            print("no shard directories")
            return
        stored = {record.key: record for record in read(BUNDLED_REPORTS)}
        orphaned = []
        for directory in directories:
            rows, pending, error = _pending(directory, stored)
            if error:
                print(f"{directory.name}: unreadable after {rows} rows -> {error}")
            else:
                print(f"{directory.name}: {rows} rows, {pending} not in the store")
            if pending:
                orphaned.append(directory)
        if orphaned:
            listing = " ".join(f"{path}/*.jsonl" for path in orphaned)
            raise SystemExit(f"{len(orphaned)} shard directory(ies) never landed; merge with:\n  {listing}")
        return
    if not args.sources:
        raise SystemExit("merge needs shard files, or --check to scan for unmerged ones")
    if args.expect is not None and len(args.sources) != args.expect:
        raise SystemExit(f"expected {args.expect} shard files, found {len(args.sources)}")
    records = [record for source in args.sources for record in read_file(source)]
    if not records:
        raise SystemExit("no report rows to merge")
    if args.publish:
        _require_clean(records)
    _publish(records)
    if args.publish:
        _require_passing_reports()
        revision = hub.publish(message=f"Merge {len(records)} rows from {len(args.sources)} shards")
        print(f"published {hub.repository()} at {revision}; pin updated")


def _impl(op: Dispatcher, name: str) -> Implementation:
    return next(candidate for candidate in op._impls if candidate.name == name)


def _usable_records(store: Store, work: list[tuple[Dispatcher, str, Case]], device: str) -> list[Record]:
    """Rows produced by this hardware, dependency set, and current source fingerprints."""
    usable = []
    selected: dict[str, set[str]] = {}
    for op, impl, _ in work:
        selected.setdefault(op.name, set()).add(impl)
    for name, selected_impls in selected.items():
        op = KERNELS[name]
        implementations = {impl.name: impl for impl in op._impls}
        for record in store.merged(name):
            environment = record.environment
            implementation = implementations.get(record.impl)
            if (
                implementation is not None
                and record.impl in selected_impls
                and environment.device == device
                and environment.torch == torch.__version__
                and environment.backend_version == installed_version(record.impl)
                and matching(op.fingerprint, environment.ref_hash)
                and matching(implementation.fingerprint, environment.impl_hash)
            ):
                usable.append(record)
    return usable


def _record_index(records: list[Record]) -> dict[tuple[str, str, str, bool], Record]:
    """Best usable row per op, implementation, case, and gradient mode."""
    best: dict[tuple[str, str, str, bool], Record] = {}
    for record in records:
        key = (record.op, record.impl, record.case_id, record.result.grad)
        if key not in best or prefer(record, best[key]):
            best[key] = record
    return best


def _fill_pending(record: Record | None) -> bool:
    """Retry absent and inconclusive work, but retain rejections and exhausted budgets.

    A case that prunes the shapes above it has to be settled for itself too, or every sweep
    pays the same OOM or the whole timeout again to rediscover a boundary already recorded.
    """
    return record is None or (record.result.status not in ("skip", *EXHAUSTED) and unmeasured(record))


def cmd_fill(args: argparse.Namespace) -> None:
    """Benchmark the shared case plan this machine has no timed row for, into the user cache.

    The plan combines pairwise coverage with named curves without materializing the full product.
    An observed OOM or timeout prunes only coordinate-wise larger shapes in the same stratum.
    """
    args.device = _target_device(args)
    device = device_name(args.device)
    store = Store()
    planned = _work(args.ops, args.backend, args.limit)
    work = [(op, impl, case) for op, impl, case in planned if available(impl)]
    if not work:
        raise SystemExit("no matching installed non-torch backend cases")
    current = _usable_records(store, work, device)
    cached = {} if args.force else _record_index(current)
    frontiers: dict[tuple[str, str, bool], BudgetFrontier] = {}
    reference_frontiers: dict[tuple[str, bool], BudgetFrontier] = {}
    if not args.force:
        for record in current:
            if record.result.status in EXHAUSTED:
                key = (record.op, record.impl, record.result.grad)
                frontiers.setdefault(key, BudgetFrontier()).add(Case.from_config(record.config))
        references = list({op.name: (op, "torch", case) for op, _, case in work}.values())
        for record in _usable_records(store, references, device):
            if record.result.status in EXHAUSTED:
                reference_frontiers.setdefault((record.op, record.result.grad), BudgetFrontier()).add(
                    Case.from_config(record.config)
                )
    capacity = (
        torch.cuda.get_device_properties(torch.device(args.device)).total_memory
        if torch.device(args.device).type == "cuda"
        else sys.maxsize
    )
    todo = []
    skipped = 0
    for op, impl, case in work:
        grad = not _impl(op, impl).forward_only and backward_safe(op, case)
        if not _fill_pending(cached.get((op.name, impl, case.case_id, grad))):
            continue
        reference_frontier = reference_frontiers.get((op.name, grad))
        if infeasible(op, case, grad, capacity) or (
            reference_frontier is not None and reference_frontier.blocker(case) is not None
        ):
            skipped += 1
            continue
        todo.append((op, impl, case, grad))
    todo.sort(key=lambda item: estimated(item[0], item[2]))
    cached_count = len(work) - len(todo) - skipped
    print(
        f"{len(work)} case-impl pairs: {cached_count} already cached, {skipped} infeasible or reference-pruned, {len(todo)} to consider"
    )
    isolation = "in-process" if args.in_process else f"{args.timeout}s/case worker"
    if not todo:
        return
    print(f"measuring with monotonic budget pruning ({isolation}) -> {store.user}")
    live = None
    if getattr(args, "live", None) is not None:
        live = LiveServer(
            args.live,
            device=device,
            ops=sorted({op.name for op, _, _, _ in todo}),
            total=len(todo),
            cached=cached_count,
            skipped=skipped,
        )
        try:
            live.start()
        except OSError as error:
            raise SystemExit(f"could not bind live benchmark server to 127.0.0.1:{args.live}: {error}") from None
        print(f"streaming live results at http://{live.host}:{live.port}")
    pending: list[Record] = []
    tally: Counter[str] = Counter()
    pruned = 0
    measured = 0
    try:
        for op, impl, case, grad in todo:
            key = (op.name, impl, grad)
            frontier = frontiers.setdefault(key, BudgetFrontier())
            if frontier.blocker(case) is not None:
                pruned += 1
                if live is not None:
                    live.pruned()
                continue
            if args.in_process:
                try:
                    record = op.bench.run_case(impl, case, args.device, args.reps, grad=grad)
                except Exception as error:
                    record = op.bench.incomplete(impl, case, args.device, f"{type(error).__name__}: {error}", grad=grad)
            else:
                record = _isolated(op, impl, case, args.device, args.reps, args.timeout, grad=grad)
            pending.append(record)
            measured += 1
            tally[record.result.status] += 1
            if live is not None:
                live.record(record)
            if record.result.status in EXHAUSTED:
                frontier.add(case)
            if len(pending) >= args.flush:
                store.write_user(pending)
                pending.clear()
            try:
                torch.zeros(1, device=args.device).item()
            except Exception as error:
                store.write_user(pending)
                raise SystemExit(f"device poisoned after {measured} measurements: {type(error).__name__}: {error}") from None
        if pending:
            store.write_user(pending)
    except BaseException as error:
        if live is not None:
            live.close(error)
        raise
    else:
        if live is not None:
            live.close()
    print(f"cached {measured} row(s), pruned {pruned} dominated shape(s) -> {store.user}")
    for status, count in tally.most_common():
        print(f"  {status}: {count}")


def cmd_pull(args: argparse.Namespace) -> None:
    """Fetch the published reports so dispatch and the generated artifacts have evidence."""
    files, revision = hub.pull(revision=args.revision, repo=args.repo)
    print(f"{files} report files from {args.repo or hub.repository()} at {revision} -> {BUNDLED_REPORTS}")


def cmd_publish(args: argparse.Namespace) -> None:
    """Publish the current Parquet store and move its pinned revision."""
    _require_passing_reports()
    revision = hub.publish(repo=args.repo, message=args.message)
    print(f"published {hub.repository() if args.repo is None else args.repo} at {revision}; pin updated")


def cmd_view(args: argparse.Namespace) -> None:
    records = read(BUNDLED_REPORTS)
    if args.user:
        merged = {record.key: record for record in records}
        merged |= {record.key: record for record in read(user_reports())}
        records = list(merged.values())
    out = Path(args.out)
    out.write_text(render(records))
    print(f"{len(records)} rows -> {out}")


def cmd_map(args: argparse.Namespace) -> None:
    """Fill validity regions by running the adaptive planner to fixpoint."""
    args.device = _target_device(args)
    store = Store()
    ops = [KERNELS[name] for name in args.ops] if args.ops else list(KERNELS.values())
    total = 0
    for op in ops:
        backends = [name for name in op.available_backends() if name != "torch" and args.backend in (None, name)]
        for backend in backends:
            registered = next(candidate for candidate in op._impls if candidate.name == backend)
            rounds = 0
            while rounds < args.rounds:
                records = store.merged(op.name)
                todo = adaptive_plan(
                    op,
                    records,
                    backend,
                    effort=args.effort,
                    device=args.device,
                    grad=not registered.forward_only,
                ).flatten()
                if args.shard:
                    index, count = args.shard
                    todo = todo[index::count]
                if not todo:
                    break
                print(f"{op.name}:{backend} round {rounds}: {len(todo)} probes ({args.effort})")
                written = []
                poisoned = False
                for index, case in enumerate(todo):
                    try:
                        record = op.bench.run_case(backend, case, args.device, args.reps, benchmark=args.bench)
                    except Exception as error:
                        record = op.bench.incomplete(backend, case, args.device, f"{type(error).__name__}: {error}")
                    written.append(record)
                    try:
                        torch.zeros(1, device=args.device).item()
                    except Exception as error:
                        reason = f"device poisoned: {type(error).__name__}: {error}"
                        record.result.status, record.result.reason = "crash", reason
                        print(reason)
                        written.extend(op.bench.incomplete(backend, rest, args.device, reason) for rest in todo[index + 1 :])
                        poisoned = True
                        break
                store.write_user(written)
                op.tuner.forget()
                total += len(written)
                rounds += 1
                if poisoned:
                    break
    print(f"mapped {total} rows -> {store.user}")


def cmd_submit(args: argparse.Namespace) -> None:
    total = len(_work(args.ops, limit=args.limit, only=args.only))
    if not total:
        raise SystemExit("no matching non-torch backend cases")
    shards = min(args.array, total)
    shard_dir = BUNDLED_REPORTS / "shards" / uuid.uuid4().hex
    script = SBATCH.format(
        last=shards - 1,
        time=args.time,
        qos_line=f"#SBATCH --qos={args.qos}\n" if args.qos else "",
        python=shlex.quote(sys.executable),
        ops=" ".join(map(shlex.quote, args.ops)),
        reps=args.reps,
        limit=f"--limit {args.limit}" if args.limit else "",
        hardware=f"--hardware {shlex.quote(args.hardware)}" if args.hardware else "",
        timeout=args.timeout,
        only=f"--only {shlex.quote(str(Path(args.only).resolve()))}" if args.only else "",
        shards=shards,
        shard_dir=shlex.quote(str(shard_dir)),
    )
    path = Path("logs") / "popcorn_bench.slurm"
    path.parent.mkdir(exist_ok=True)
    path.write_text(script)
    print(f"{total} case-backend pairs across {shards} shards ({path})")
    if args.dry_run:
        return
    submitted = subprocess.run(["sbatch", str(path)], capture_output=True, text=True, check=True).stdout
    job = submitted.strip().rsplit(" ", 1)[-1]
    print(submitted.strip())
    merge = (
        f'cd "$SLURM_SUBMIT_DIR" && {shlex.quote(sys.executable)} -m popcorn.bench merge '
        f"--expect {shards} {'--publish ' if args.publish else ''}{shlex.quote(str(shard_dir))}/*.jsonl"
    )
    subprocess.run(
        [
            "sbatch",
            f"--dependency={'afterok' if args.publish else 'afterany'}:{job}",
            *([f"--qos={args.qos}"] if args.qos else []),
            "--job-name=popcorn-merge",
            "--output=logs/popcorn_merge_%j.log",
            f"--wrap={merge}",
        ],
        check=True,
    )


def main(prog: str = "python -m popcorn.bench") -> None:
    parser = argparse.ArgumentParser(prog=prog, description="Popcorn correctness + benchmark harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the shared case plan locally (or one shard of it)")
    run.add_argument("ops", nargs="*", help="ops to test (default: all registered)")
    run.add_argument("--backend", help="restrict to one backend")
    run.add_argument("--device", default="cuda")
    run.add_argument("--hardware", help=HARDWARE_HELP)
    run.add_argument("--reps", type=_positive, default=10)
    run.add_argument("--limit", type=_positive, help="use a deterministic random Cartesian sample of at most N cases per op")
    run.add_argument("--shard", type=_shard, help="I/K: run the I-th of K deterministic slices")
    run.add_argument("--out", help="write raw rows to this JSONL instead of the reports store")
    run.add_argument("--timeout", type=_positive, default=30, help="seconds per case worker before it is killed")
    run.add_argument("--in-process", action="store_true", help="run cases without a worker: no timeout, crashes end the run")
    run.add_argument("--only", help="restrict to the op/impl/case rows recorded in this JSONL (e.g. a shard file)")
    run.set_defaults(fn=cmd_run)

    run_case = sub.add_parser("run-case", help=argparse.SUPPRESS)
    run_case.add_argument("--op", required=True)
    run_case.add_argument("--impl", required=True)
    run_case.add_argument("--case", required=True, help="JSON case config")
    run_case.add_argument("--device", default="cuda")
    run_case.add_argument("--reps", type=_positive, default=10)
    run_case.add_argument("--grad", action=argparse.BooleanOptionalAction, default=None)
    run_case.add_argument("--out", required=True)
    run_case.set_defaults(fn=cmd_run_case)

    merge = sub.add_parser("merge", help="fold shard JSONL files into the reports store")
    merge.add_argument("--expect", type=_positive, help="fail unless this many shard files exist")
    merge.add_argument("--check", action="store_true", help="report shard directories that were never merged; write nothing")
    merge.add_argument("--publish", action="store_true", help=f"upload the merged reports to {hub.REPO} and update the pin")
    merge.add_argument(
        "--scrub-incomplete",
        action="store_true",
        help="drop rows the harness scheduled but never measured, then refresh the README badges",
    )
    merge.add_argument("sources", nargs="*")
    merge.set_defaults(fn=cmd_merge)

    fill = sub.add_parser("fill", help="measure planned coverage and curves not timed on this machine")
    fill.add_argument("ops", nargs="*", help="ops to measure (default: all registered)")
    fill.add_argument("--backend", help="restrict to one backend")
    fill.add_argument("--device", default="cuda")
    fill.add_argument("--hardware", help=HARDWARE_HELP)
    fill.add_argument("--reps", type=_positive, default=10)
    fill.add_argument("--limit", type=_positive, help="use a deterministic random Cartesian sample of at most N cases per op")
    fill.add_argument("--timeout", type=_positive, default=30, help="seconds per case worker before it is killed")
    fill.add_argument("--in-process", action="store_true", help="run cases without a worker: no timeout, crashes end the run")
    fill.add_argument("--force", action="store_true", help="measure every combination, including ones already cached")
    fill.add_argument("--flush", type=_positive, default=50, help="write to the cache every N rows")
    fill.add_argument(
        "--live",
        nargs="?",
        const=DEFAULT_PORT,
        type=_port,
        metavar="PORT",
        help=f"stream this run to the kernel explorer on localhost (default port: {DEFAULT_PORT})",
    )
    fill.set_defaults(fn=cmd_fill)

    pull = sub.add_parser("pull", help="fetch the published reports from the hub")
    pull.add_argument("--revision", help=f"revision to fetch (default: the {hub.REVISION} pin, else the default branch)")
    pull.add_argument("--repo", help=f"dataset repository to fetch from (default: {hub.REPO})")
    pull.set_defaults(fn=cmd_pull)

    publish = sub.add_parser("publish", help="upload the current Parquet store to the hub and update its pin")
    publish.add_argument("--repo", help=f"dataset repository to update (default: {hub.REPO})")
    publish.add_argument("--message", default="Update reports", help="dataset commit message")
    publish.set_defaults(fn=cmd_publish)

    view = sub.add_parser("view", help="render the reports store as a standalone HTML page")
    view.add_argument("--out", default=str(BUNDLED_REPORTS / "index.html"))
    view.add_argument("--user", action="store_true", help="include rows from the user cache, newest winning")
    view.set_defaults(fn=cmd_view)

    submit = sub.add_parser("submit", help="submit the shared case plan as a slurm array + merge job")
    submit.add_argument("ops", nargs="*")
    submit.add_argument("--array", type=_positive, default=8, help="number of shards")
    submit.add_argument("--hardware", help=f"passed to each array task: {HARDWARE_HELP}")
    submit.add_argument("--reps", type=_positive, default=10)
    submit.add_argument(
        "--limit", type=_positive, help="use a deterministic random Cartesian sample of at most N cases per op"
    )
    submit.add_argument("--qos", default=None, help="slurm QoS for the array and merge jobs (default: cluster default)")
    submit.add_argument("--time", default="2:00:00")
    submit.add_argument("--timeout", type=_positive, default=30, help="passed to each array task: seconds per case worker")
    submit.add_argument("--only", help="passed to each array task: restrict to the rows recorded in this JSONL")
    submit.add_argument("--publish", action="store_true", help="the merge job uploads to the hub and updates the pin")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(fn=cmd_submit)

    mapping = sub.add_parser("map", help="adaptively probe validity regions to fixpoint")
    mapping.add_argument("ops", nargs="*", help="ops to map (default: all registered)")
    mapping.add_argument("--backend", help="restrict to one backend")
    mapping.add_argument("--device", default="cuda")
    mapping.add_argument("--hardware", help=HARDWARE_HELP)
    mapping.add_argument("--effort", choices=sorted(EFFORT), default="standard")
    mapping.add_argument("--reps", type=_positive, default=5, help="timing reps per raced probe")
    mapping.add_argument("--rounds", type=_positive, default=8, help="max plan/run iterations per backend")
    mapping.add_argument("--bench", action="store_true", help="also time passing probes")
    mapping.add_argument("--shard", type=_shard, help="I/K: run the I-th of K deterministic slices of each plan")
    mapping.set_defaults(fn=cmd_map)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

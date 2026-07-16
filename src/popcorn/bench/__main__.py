"""CLI harness: run/merge/view the grid locally or submit it as a slurm array."""

import argparse
import json
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import torch

import popcorn.kernels  # noqa: F401
from popcorn import KERNELS
from popcorn.bench.grid import cases
from popcorn.bench.model import Case, Record
from popcorn.bench.report import refresh
from popcorn.bench.store import BUNDLED_REPORTS, read, read_file, user_reports, write
from popcorn.bench.viewer import render
from popcorn.core.dispatcher import Dispatcher

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

{python} -m popcorn.bench run {ops} --reps {reps} {limit} \
    --shard "$SLURM_ARRAY_TASK_ID/{shards}" --out {shard_dir}/"$SLURM_ARRAY_TASK_ID".jsonl
"""


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _shard(value: str) -> tuple[int, int]:
    try:
        index, count = map(int, value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("must be I/K") from None
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("must satisfy 0 <= I < K")
    return index, count


def _work(ops: list[str], backend: str | None = None, limit: int | None = None) -> list[tuple[Dispatcher, str, Case]]:
    work = []
    for op in (KERNELS[name] for name in ops or sorted(KERNELS)):
        op_cases = cases(op, limit)
        for candidate in op.available_backends():
            if candidate != "torch" and backend in (None, candidate):
                work.extend((op, candidate, case) for case in op_cases)
    return work


def _publish(records: list[Record]) -> None:
    write(records, BUNDLED_REPORTS)
    print(refresh(ops=KERNELS))


def _append(path: Path, record: Record) -> None:
    with path.open("a") as sink:
        sink.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def cmd_run(args: argparse.Namespace) -> None:
    work = _work(args.ops, args.backend, args.limit)
    if args.shard:
        index, count = args.shard
        work = work[index * len(work) // count : (index + 1) * len(work) // count]
    if not work:
        raise SystemExit("no matching non-torch backend cases")
    print(f"{len(work)} case-backend pairs, {args.reps} reps each")
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("")
        for op, backend, case in work:
            _append(out, op.bench.incomplete(backend, case, args.device))

    records = []
    poisoned = False
    for index, (op, backend, case) in enumerate(work):
        try:
            record = op.bench.run_case(backend, case, args.device, args.reps)
        except Exception as error:
            record = op.bench.incomplete(backend, case, args.device, f"{type(error).__name__}: {error}")
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
            print(f"{reason} after {record.op}:{record.backend} {record.case}")
            poisoned = True
            break
    if not out:
        _publish(records)
    failures = [
        record for record in records if record.result.status in ("fail", "crash", "error") or record.result.bench_error
    ]
    for record in failures[:20]:
        result = record.result
        reason = result.reason or f"benchmark: {result.bench_error}"
        print(f"\n[{result.status}] {record.op}:{record.backend} {record.case}\n  {reason}")
    raise SystemExit(1 if failures or poisoned else 0)


def cmd_merge(args: argparse.Namespace) -> None:
    if args.expect is not None and len(args.sources) != args.expect:
        raise SystemExit(f"expected {args.expect} shard files, found {len(args.sources)}")
    records = [record for source in args.sources for record in read_file(source)]
    if not records:
        raise SystemExit("no report rows to merge")
    _publish(records)


def cmd_view(args: argparse.Namespace) -> None:
    records = read(BUNDLED_REPORTS)
    if args.user:
        merged = {record.key: record for record in records}
        merged |= {record.key: record for record in read(user_reports())}
        records = list(merged.values())
    out = Path(args.out)
    out.write_text(render(records))
    print(f"{len(records)} rows -> {out}")


def cmd_submit(args: argparse.Namespace) -> None:
    total = len(_work(args.ops, limit=args.limit))
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
        f"--expect {shards} {shlex.quote(str(shard_dir))}/*.jsonl"
    )
    subprocess.run(
        [
            "sbatch",
            f"--dependency=afterany:{job}",
            *([f"--qos={args.qos}"] if args.qos else []),
            "--job-name=popcorn-merge",
            "--output=logs/popcorn_merge_%j.log",
            f"--wrap={merge}",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m popcorn.bench", description="Popcorn correctness + benchmark harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the grid locally (or one shard of it)")
    run.add_argument("ops", nargs="*", help="ops to test (default: all registered)")
    run.add_argument("--backend", help="restrict to one backend")
    run.add_argument("--device", default="cuda")
    run.add_argument("--reps", type=_positive, default=10)
    run.add_argument("--limit", type=_positive, help="subsample the grid to at most N cases per op")
    run.add_argument("--shard", type=_shard, help="I/K: run the I-th of K deterministic slices")
    run.add_argument("--out", help="write raw rows to this JSONL instead of the reports store")
    run.set_defaults(fn=cmd_run)

    merge = sub.add_parser("merge", help="fold shard JSONL files into the reports store")
    merge.add_argument("--expect", type=_positive, help="fail unless this many shard files exist")
    merge.add_argument("sources", nargs="+")
    merge.set_defaults(fn=cmd_merge)

    view = sub.add_parser("view", help="render the reports store as a standalone HTML page")
    view.add_argument("--out", default=str(BUNDLED_REPORTS / "index.html"))
    view.add_argument("--user", action="store_true", help="include rows from the user cache, newest winning")
    view.set_defaults(fn=cmd_view)

    submit = sub.add_parser("submit", help="submit the grid as a slurm array + merge job")
    submit.add_argument("ops", nargs="*")
    submit.add_argument("--array", type=_positive, default=8, help="number of shards")
    submit.add_argument("--reps", type=_positive, default=10)
    submit.add_argument("--limit", type=_positive, help="subsample the grid to at most N cases per op")
    submit.add_argument("--qos", default=None, help="slurm QoS for the array and merge jobs (default: cluster default)")
    submit.add_argument("--time", default="2:00:00")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(fn=cmd_submit)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

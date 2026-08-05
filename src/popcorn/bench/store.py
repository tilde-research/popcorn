"""Persist published Parquet reports, JSONL shards, and the user cache."""

import json
import os
import inspect
import typing
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from popcorn.bench.grid import backward_safe, case_plan
from popcorn.bench.model import SCHEMA, Record
from popcorn.core.config import Call
from popcorn.core.typecheck import matches

try:
    import fcntl
except ImportError:
    fcntl = None

BUNDLED_REPORTS = Path(__file__).parents[1] / "reports"
PUBLISHED = ".parquet"
SCRATCH = ".jsonl"
FETCHED = ".revision"
# Nested fields ride as JSON text in one column each: the readers want whole rows back,
# not projections into a gauge, and a stable string beats a struct that shifts with the model.
NESTED = ("config", "fwd", "bwd", "bench")
# Column types are stated rather than inferred: a column that happens to be all-null in one
# op's rows (an absent backend_version, say) would otherwise land as pyarrow's null type and
# refuse to concatenate with the same column elsewhere.
_COLUMNS = {
    "schema": pa.int32(),
    "op": pa.string(),
    "impl": pa.string(),
    "case": pa.string(),
    "case_id": pa.string(),
    "device": pa.string(),
    "torch": pa.string(),
    "backend_version": pa.string(),
    "ts": pa.string(),
    "ref_hash": pa.string(),
    "impl_hash": pa.string(),
    "status": pa.string(),
    "reason": pa.string(),
    "grad": pa.bool_(),
    "benchmarked": pa.bool_(),
    "bench_error": pa.string(),
    "reps": pa.int32(),
} | {name: pa.string() for name in NESTED}
_SCHEMA = pa.schema(_COLUMNS)
_WRITE_LOCK = Lock()
_warned = False


def user_reports() -> Path:
    """Per-user report cache, namespaced by report schema.

    The generation segment tracks `SCHEMA`, so a schema bump lands in a fresh
    directory instead of feeding rows to a reader that would reject them; caches
    from older generations are left untouched beside it.
    """
    root = os.getenv("POPCORN_CACHE_DIR")
    if root is None:
        root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "popcorn"
    return Path(root).expanduser() / f"v{SCHEMA}" / "reports"


def conclusive(record: Record) -> bool:
    return record.result.status in ("pass", "fail", "crash", "oom")


def unmeasured(record: Record | None, benchmark: bool = True) -> bool:
    """Whether a benchmark run should still produce this row: absent, inconclusive, or untimed.

    One definition serves the lazy per-call path and the grid-wide `fill`, so the two cannot
    disagree about what "already cached" means.
    """
    status = record.result.status if record else None
    if benchmark and status == "pass" and record is not None and not record.result.benchmarked:
        return True
    return status not in ("pass", "fail", "crash")


def matching(expected: str | None, recorded: str | None) -> bool:
    """Fingerprints agree; only currently unhashable code may omit its hash."""
    return expected is None or expected == recorded


def prefer(record: Record, current: Record, priority: int = 0, current_priority: int = 0) -> bool:
    rank = (conclusive(record), record.environment.ts, priority)
    current_rank = (conclusive(current), current.environment.ts, current_priority)
    return rank >= current_rank


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    with _WRITE_LOCK:
        lock = path.with_suffix(path.suffix + ".lock")
        with lock.open("a") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def read_parquet(path: Path) -> list[Record]:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        for name in NESTED:
            row[name] = json.loads(row[name])
    return [Record.from_dict(row) for row in rows]


def write_parquet(records: Iterable[Record], path: Path) -> None:
    rows = []
    for record in records:
        row = record.to_dict()
        for name in NESTED:
            row[name] = json.dumps(row.get(name, {}), sort_keys=True)
        rows.append(row)
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def read_file(path: Path | str) -> list[Record]:
    path = Path(path)
    try:
        if path.suffix == PUBLISHED:
            return read_parquet(path)
        return [Record.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]
    except FileNotFoundError:
        # Concurrent writers replace report files atomically, and over NFS the old entry
        # can vanish between an existence check and the read. Absent and just-replaced
        # answer the same; the replacement serves the next read.
        return []


def read(directory: Path | str) -> list[Record]:
    directory = Path(directory)
    paths = sorted(directory.glob(f"*{PUBLISHED}")) + sorted(directory.glob(f"*{SCRATCH}"))
    return [record for path in paths for record in read_file(path)]


def bundled_path(directory: Path, op: str) -> Path:
    """Where one op's published rows live, preferring Parquet and falling back to JSONL."""
    published = directory / f"{op}{PUBLISHED}"
    return published if published.exists() else directory / f"{op}{SCRATCH}"


def _audit_rows(op: str, directory: Path, extra_directories: Iterable[Path]) -> list[Record]:
    return [record for source in (directory, *extra_directories) for record in read_file(bundled_path(source, op))]


def audit_records(
    ops: Iterable[Any],
    directory: Path = BUNDLED_REPORTS,
    extra_directories: Iterable[Path] = (),
) -> tuple[int, int, dict[str, list[str]]]:
    """Count current and passing implementation pairs, with missing/stale details."""
    extra_directories = tuple(extra_directories)
    current = 0
    passing = 0
    problems = defaultdict(list)
    for op in ops:
        rows = _audit_rows(op.name, directory, extra_directories)
        for impl in op._impls:
            if impl.name == "torch":
                continue
            pair = f"{op.name}:{impl.name}"
            present = [row for row in rows if row.impl == impl.name]
            fresh = [
                row
                for row in present
                if matching(op.fingerprint, row.environment.ref_hash) and matching(impl.fingerprint, row.environment.impl_hash)
            ]
            if not fresh:
                kind = "stale" if present else "missing"
                detail = f" ({len(present)} rows, none matching the current fingerprint)" if present else ""
                problems[kind].append(f"{pair}{detail}")
                continue
            current += 1
            if any(row.result.status == "pass" for row in fresh):
                passing += 1
            else:
                statuses = ", ".join(
                    f"{status}={count}" for status, count in sorted(Counter(row.result.status for row in fresh).items())
                )
                problems["no_pass"].append(f"{pair} ({statuses})")
    return current, passing, dict(problems)


def _latest_rows(records: Iterable[Record]) -> list[Record]:
    latest: dict[tuple[Any, ...], Record] = {}
    for record in records:
        key = record.key[1:]
        current = latest.get(key)
        if current is None or prefer(record, current):
            latest[key] = record
    return list(latest.values())


def _timed(record: Record) -> bool:
    return (
        record.result.status == "pass"
        and record.result.benchmarked
        and not record.result.bench_error
        and isinstance(record.result.bench.get("fwd_ms"), (int, float))
    )


def _dtype_allowed(annotation: Any, dtype: Any) -> bool | None:
    if dtypes := getattr(annotation, "dtypes", None):
        return str(dtype).removeprefix("torch.") in dtypes
    decisions = [_dtype_allowed(child, dtype) for child in typing.get_args(annotation)]
    known = [decision for decision in decisions if decision is not None]
    return any(known) if known else None


def static_allowed(op: Any, impl: Any, dtype: Any, args: Mapping[str, Any], present: frozenset[str]) -> bool:
    """Whether static adapter gates admit a case without materializing its tensors."""
    specs = {spec.param: spec for spec in op.specs}
    for name, annotation in getattr(impl, "gates", {}).items():
        if (spec := specs.get(name)) is not None:
            if spec.optional and name not in present:
                if not matches(None, annotation):
                    return False
                continue
            if _dtype_allowed(annotation, dtype) is not True:
                return False
            continue
        value = args.get(name, op._signature.parameters[name].default)
        if value is inspect.Parameter.empty or not matches(value, annotation):
            return False
    return True


def _series_allowed(op: Any, impl: Any, series: Any) -> bool:
    """Whether static adapter gates admit every fixed property of a series."""
    return static_allowed(op, impl, series.dtype, dict(series.args), series.present)


def audit_evidence(
    ops: Iterable[Any],
    directory: Path = BUNDLED_REPORTS,
    min_curve_points: int = 2,
    extra_directories: Iterable[Path] = (),
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Audit timed evidence, named curve depth, and failed benchmark attempts."""
    if min_curve_points < 1:
        raise ValueError("min_curve_points must be positive")
    extra_directories = tuple(extra_directories)
    counts = Counter(pairs=0, timed=0, curves=0, complete_curves=0)
    problems: defaultdict[str, list[str]] = defaultdict(list)
    for op in ops:
        rows = _audit_rows(op.name, directory, extra_directories)
        curves = [series for series in case_plan(op).series if series.kind == "curve" and series.cases]
        for impl in op._impls:
            if impl.name == "torch":
                continue
            counts["pairs"] += 1
            pair = f"{op.name}:{impl.name}"
            fresh = _latest_rows(
                row
                for row in rows
                if row.impl == impl.name
                and matching(op.fingerprint, row.environment.ref_hash)
                and matching(impl.fingerprint, row.environment.impl_hash)
            )
            if not fresh:
                continue  # missing/stale is reported by audit_records
            if any(_timed(row) for row in fresh):
                counts["timed"] += 1
            else:
                problems["no_timing"].append(pair)

            benchmark_errors = [row for row in fresh if row.result.bench_error]
            if benchmark_errors:
                problems["bench_error"].append(f"{pair} ({len(benchmark_errors)} rows)")
            bad = Counter(row.result.status for row in fresh if row.result.status in {"fail", "crash", "error"})
            if bad:
                details = ", ".join(f"{status}={count}" for status, count in sorted(bad.items()))
                problems["bad_status"].append(f"{pair} ({details})")

            by_case: defaultdict[tuple[str, bool], list[Record]] = defaultdict(list)
            for row in fresh:
                by_case[(row.case_id, row.result.grad)].append(row)
            for series in curves:
                if not _series_allowed(op, impl, series):
                    continue
                expected = {
                    case.case_id: by_case[(case.case_id, not impl.forward_only and backward_safe(op, case))]
                    for case in series.cases
                }
                eligible = [
                    case
                    for case in series.cases
                    if not ((case_rows := expected[case.case_id]) and all(row.result.status == "skip" for row in case_rows))
                ]
                if len(eligible) < min_curve_points:
                    # Once every planned point has an explicit row, all-skip points
                    # describe a gated region, not a missing implementation curve.
                    attempted = all(expected[case.case_id] for case in series.cases)
                    if attempted:
                        continue
                counts["curves"] += 1
                timed = sum(any(_timed(row) for row in expected[case.case_id]) for case in eligible)
                needed = min(min_curve_points, len(eligible))
                if timed >= needed:
                    counts["complete_curves"] += 1
                    continue
                problems["incomplete_curve"].append(
                    f"{pair} {series.name} ({timed}/{needed} timed points; {len(series.cases)} planned)"
                )
    return dict(counts), dict(problems)


def audit_outcomes(directories: Iterable[Path]) -> dict[str, list[str]]:
    """Report disallowed outcomes in newly generated report directories."""
    rows = [record for directory in directories for record in read(directory)]
    problems: defaultdict[str, list[str]] = defaultdict(list)
    statuses = Counter(
        (record.op, record.impl, record.result.status) for record in rows if record.result.status in {"fail", "crash", "error"}
    )
    for (op, impl, status), count in sorted(statuses.items()):
        problems["bad_status"].append(f"{op}:{impl} ({status}={count})")
    errors = Counter((record.op, record.impl) for record in rows if record.result.bench_error)
    for (op, impl), count in sorted(errors.items()):
        problems["bench_error"].append(f"{op}:{impl} ({count} rows)")
    return dict(problems)


def _dump(records: list[Record], path: Path, suffix: str) -> None:
    if suffix == PUBLISHED:
        write_parquet(records, path)
        return
    path.write_text("\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records) + "\n")


def _preserve_revision(directory: Path) -> None:
    """Keep local report edits across editable rebuilds until the tracked pin changes."""
    pin = directory / "REVISION"
    if pin.exists():
        (directory / FETCHED).write_text(pin.read_text())
    else:
        (directory / FETCHED).unlink(missing_ok=True)


def write(records: Iterable[Record], directory: Path | str, suffix: str = SCRATCH) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.op].append(record)
    for op, incoming in sorted(grouped.items()):
        path = directory / f"{op}{suffix}"
        with _locked(path):
            merged = {record.key: record for record in read_file(path)}
            for record in incoming:
                if record.key not in merged or prefer(record, merged[record.key]):
                    merged[record.key] = record
            order = sorted(merged, key=lambda key: tuple("" if value is None else str(value) for value in key))
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                _dump([merged[key] for key in order], temporary, suffix)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    if suffix == PUBLISHED:
        _preserve_revision(directory)


def scrub(directory: Path | str, drop: Callable[[Record], bool]) -> int:
    """Rewrite a reports directory without the rows `drop` selects, returning how many fell.

    `write` can only add or replace rows by key, so removal needs its own path. It exists
    for rows that are bookkeeping rather than evidence (the harness sentinel that never
    became a measurement); files left with no rows are removed outright.
    """
    directory = Path(directory)
    dropped = 0
    for path in [*sorted(directory.glob(f"*{PUBLISHED}")), *sorted(directory.glob(f"*{SCRATCH}"))]:
        with _locked(path):
            rows = read_file(path)
            kept = [row for row in rows if not drop(row)]
            if len(kept) == len(rows):
                continue
            dropped += len(rows) - len(kept)
            if not kept:
                path.unlink()
                continue
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                _dump(kept, temporary, path.suffix)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    if dropped:
        _preserve_revision(directory)
    return dropped


class Store:
    """Merged view over the bundled reports and the user cache, newest row winning."""

    def __init__(self, bundled: Path | str | None = None, user: Path | str | None = None) -> None:
        self.bundled = Path(bundled) if bundled is not None else BUNDLED_REPORTS
        self._user = Path(user) if user is not None else None

    @property
    def user(self) -> Path:
        return self._user or user_reports()

    def announce_if_empty(self) -> None:
        """Say so, once, when there is no evidence to dispatch on.

        Published reports are fetched rather than checked in, so a fresh clone has none
        until `popcorn.bench pull` runs. Without rows every call resolves to the torch
        reference, which is correct but forfeits every implementation; that is too large a
        change in behaviour to leave for the reader to infer from a slow benchmark.
        """
        global _warned
        if _warned or any(self.bundled.glob(f"*{PUBLISHED}")) or any(self.bundled.glob(f"*{SCRATCH}")):
            return
        _warned = True
        warnings.warn(
            f"no reports in {self.bundled}: every call will use the torch reference. "
            "Fetch the published evidence with `python -m popcorn.bench pull`.",
            RuntimeWarning,
            stacklevel=3,
        )

    def merged(self, op: str) -> tuple[Record, ...]:
        self.announce_if_empty()
        merged: dict[tuple, tuple[int, Record]] = {}
        paths = dict.fromkeys((bundled_path(self.bundled, op), self.user / f"{op}{SCRATCH}"))
        for priority, path in enumerate(paths):
            for record in read_file(path):
                key = record.key[1:]
                if key not in merged or prefer(record, merged[key][1], priority, merged[key][0]):
                    merged[key] = (priority, record)
        return tuple(record for _, record in merged.values())

    def exact(
        self,
        op: str,
        impl: str,
        call: Call,
        torch_version: str,
        backend_version: str | None,
        records: Iterable[Record] | None = None,
        ref_hash: str | None = None,
        impl_hash: str | None = None,
    ) -> Record | None:
        found = [
            record
            for record in (self.merged(op) if records is None else records)
            if record.impl == impl
            and record.environment.device == call.device_name
            and record.environment.torch == torch_version
            and record.environment.backend_version == backend_version
            and record.result.grad == call.grad
            and record.config == call.config
            and matching(ref_hash, record.environment.ref_hash)
            and matching(impl_hash, record.environment.impl_hash)
        ]
        return max(found, key=lambda record: (conclusive(record), record.environment.ts), default=None)

    def reference_bench(
        self,
        op: str,
        case_id: str,
        config: Mapping[str, Any],
        device: str,
        torch_version: str,
        grad: bool,
        ref_hash: str | None,
    ) -> dict[str, float] | None:
        """Reusable torch timing for this exact reference case, never correctness evidence."""
        found = [
            record
            for record in self.merged(op)
            if record.impl == "torch"
            and record.case_id == case_id
            and record.config == config
            and record.environment.device == device
            and record.environment.torch == torch_version
            and record.environment.backend_version is None
            and record.result.grad == grad
            and record.result.status == "pass"
            and record.result.benchmarked
            and matching(ref_hash, record.environment.ref_hash)
        ]
        record = max(found, key=lambda row: row.environment.ts, default=None)
        if record is None:
            return None
        bench = {name: value for name, value in record.result.bench.items() if name.startswith("ref_")}
        bench.update(
            {f"ref_{name}": value for name, value in record.result.bench.items() if name.startswith(("fwd_", "bwd_"))}
        )
        return bench if "ref_fwd_ms" in bench else None

    def write_user(self, records: Iterable[Record]) -> None:
        write(records, self.user)

    def write_bundled(self, records: Iterable[Record]) -> None:
        write(records, self.bundled, PUBLISHED)

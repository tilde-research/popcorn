"""Report persistence: bundled package reports plus the per-user cache."""

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from popcorn.bench.model import SCHEMA, Record
from popcorn.core.config import Call

try:
    import fcntl
except ImportError:
    fcntl = None

BUNDLED_REPORTS = Path(__file__).parents[1] / "reports"
_WRITE_LOCK = Lock()


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


def matching(expected: str | None, recorded: str | None) -> bool:
    """Fingerprints agree; None on either side (unhashable code, legacy row) matches everything."""
    return expected is None or recorded is None or expected == recorded


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


def read_file(path: Path | str) -> list[Record]:
    path = Path(path)
    if not path.exists():
        return []
    return [Record.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def read(directory: Path | str) -> list[Record]:
    return [record for path in sorted(Path(directory).glob("*.jsonl")) for record in read_file(path)]


def write(records: Iterable[Record], directory: Path | str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record.op].append(record)
    for op, incoming in sorted(grouped.items()):
        path = directory / f"{op}.jsonl"
        with _locked(path):
            merged = {record.key: record for record in read_file(path)}
            for record in incoming:
                if record.key not in merged or prefer(record, merged[record.key]):
                    merged[record.key] = record
            order = sorted(merged, key=lambda key: tuple("" if value is None else str(value) for value in key))
            lines = [json.dumps(merged[key].to_dict(), sort_keys=True) for key in order]
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                temporary.write_text("\n".join(lines) + "\n")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


class Store:
    """Merged view over the bundled reports and the user cache, newest row winning."""

    def __init__(self, bundled: Path | str | None = None, user: Path | str | None = None) -> None:
        self.bundled = Path(bundled) if bundled is not None else BUNDLED_REPORTS
        self._user = Path(user) if user is not None else None

    @property
    def user(self) -> Path:
        return self._user or user_reports()

    def merged(self, op: str) -> tuple[Record, ...]:
        merged: dict[tuple, tuple[int, Record]] = {}
        paths = dict.fromkeys((self.bundled / f"{op}.jsonl", self.user / f"{op}.jsonl"))
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

    def write_user(self, records: Iterable[Record]) -> None:
        write(records, self.user)

    def write_bundled(self, records: Iterable[Record]) -> None:
        write(records, self.bundled)

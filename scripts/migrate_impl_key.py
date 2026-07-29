#!/usr/bin/env python
"""Migrate report JSONL from schema 2 `backend` to schema 3 `impl`.

Usage:
    uv run python scripts/migrate_impl_key.py [REPORT.jsonl ...]
"""

import json
import sys
from pathlib import Path

from popcorn.bench.store import BUNDLED_REPORTS


def migrate(row: dict) -> dict | None:
    if row.get("schema") == 3:
        return None
    if row.get("schema") != 2 or "backend" not in row:
        raise ValueError(f"cannot migrate row with schema {row.get('schema')!r}")
    return {("impl" if key == "backend" else key): (3 if key == "schema" else value) for key, value in row.items()}


def migrate_file(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    migrated = [migrate(row) for row in rows]
    if not any(row is not None for row in migrated):
        return 0
    merged = [new or old for new, old in zip(migrated, rows)]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged))
    return sum(row is not None for row in migrated)


def main() -> None:
    paths = [Path(name) for name in sys.argv[1:]] or sorted(BUNDLED_REPORTS.glob("*.jsonl"))
    total = 0
    for path in paths:
        count = migrate_file(path)
        total += count
        print(f"{path.name}: {count} row(s) migrated")
    print(f"{total} row(s) migrated across {len(paths)} file(s)")


if __name__ == "__main__":
    main()

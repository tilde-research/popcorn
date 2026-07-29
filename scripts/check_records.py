#!/usr/bin/env python
"""Check that pinned reports represent every registered implementation.

Usage:
    uv run python scripts/check_records.py [--quiet] [--require-pass]
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import popcorn.kernels  # noqa: F401  # populates KERNELS
from popcorn import KERNELS
from popcorn.bench.hub import pinned
from popcorn.bench.store import BUNDLED_REPORTS, bundled_path, read_file


def _matches(expected: str | None, recorded: str | None) -> bool:
    """Current hashable code requires its exact fingerprint; unhashable code has no gate."""
    return expected is None or expected == recorded


def audit(directory: Path = BUNDLED_REPORTS) -> tuple[int, int, dict[str, list[str]]]:
    """Return current and passing pair counts, plus problems keyed by kind."""
    current = 0
    passing = 0
    problems = defaultdict(list)
    for op in KERNELS.values():
        rows = read_file(bundled_path(directory, op.name))
        for impl in op._impls:
            if impl.name == "torch":
                continue  # the reference needs no evidence to be chosen as the fallback
            pair = f"{op.name}:{impl.name}"
            present = [row for row in rows if row.impl == impl.name]
            fresh = [
                row
                for row in present
                if _matches(op.fingerprint, row.environment.ref_hash) and _matches(impl.fingerprint, row.environment.impl_hash)
            ]
            if fresh:
                current += 1
                if any(row.result.status == "pass" for row in fresh):
                    passing += 1
                else:
                    statuses = ", ".join(
                        f"{status}={count}" for status, count in Counter(row.result.status for row in fresh).items()
                    )
                    problems["no_pass"].append(f"{pair} ({statuses})")
            elif present:
                problems["stale"].append(f"{pair} ({len(present)} rows, none matching the current fingerprint)")
            else:
                problems["missing"].append(pair)
    return current, passing, dict(problems)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print nothing when every pair is covered")
    parser.add_argument("--require-pass", action="store_true", help="also fail when a current pair has no passing row")
    args = parser.parse_args()

    current, passing, problems = audit()
    total = current + len(problems.get("missing", [])) + len(problems.get("stale", []))
    revision = pinned() or "the default branch"
    blocking = problems.get("missing", []) + problems.get("stale", [])
    if args.require_pass:
        blocking += problems.get("no_pass", [])
    if not blocking:
        if not args.quiet:
            print(f"records: {current}/{total} current fingerprints represented at {revision}")
            print(f"passes:  {passing}/{total} implementations have a current passing row")
        return
    report = [
        f"records: {current}/{total} current fingerprints represented at {revision}",
        f"passes:  {passing}/{total} implementations have a current passing row",
    ]
    for pair in problems.get("missing", []):
        report.append(f"  no rows: {pair}")
    for pair in problems.get("stale", []):
        report.append(f"  stale:   {pair}")
    if args.require_pass:
        for pair in problems.get("no_pass", []):
            report.append(f"  no pass: {pair}")
    report.append(
        "\nBenchmark the gaps on a GPU and publish, or pull a revision that already covers them:\n"
        "  uv run python -m popcorn.bench run --out reports.jsonl\n"
        "  uv run python -m popcorn.bench merge --publish reports.jsonl\n"
        "  uv run python -m popcorn.bench pull"
    )
    raise SystemExit("\n".join(report))


if __name__ == "__main__":
    main()

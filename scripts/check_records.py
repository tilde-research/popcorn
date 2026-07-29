#!/usr/bin/env python
"""Check that pinned reports represent every registered implementation.

Usage:
    uv run python scripts/check_records.py [--quiet] [--require-pass]
"""

import argparse

import popcorn.kernels  # noqa: F401  # populates KERNELS
from popcorn import KERNELS
from popcorn.bench.hub import pinned
from popcorn.bench.store import audit_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print nothing when every pair is covered")
    parser.add_argument("--require-pass", action="store_true", help="also fail when a current pair has no passing row")
    args = parser.parse_args()

    current, passing, problems = audit_records(KERNELS.values())
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

#!/usr/bin/env python
"""Check fingerprint, timing, curve, and outcome gates for benchmark reports.

Usage:
    uv run python scripts/check_records.py [--quiet] [--release]
"""

import argparse
from pathlib import Path

import popcorn.kernels  # noqa: F401  # populates KERNELS
from popcorn import KERNELS
from popcorn.bench.hub import pinned
from popcorn.bench.store import BUNDLED_REPORTS, audit_evidence, audit_outcomes, audit_records

# These are implementation-wide limitations documented in ISSUES.md, not cases that
# happened to fail. Fingerprint coverage remains mandatory even for an exception.
QUALITY_EXCEPTIONS = {
    "mesa_net:fla": "ISSUES.md: approximate solver cannot match the exact reference",
}


def _pair(detail: str) -> str:
    return detail.split(" ", 1)[0]


def _without_exceptions(details: list[str]) -> tuple[list[str], list[str]]:
    allowed = [detail for detail in details if _pair(detail) in QUALITY_EXCEPTIONS]
    return [detail for detail in details if _pair(detail) not in QUALITY_EXCEPTIONS], allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print nothing when every pair is covered")
    parser.add_argument("--require-pass", action="store_true", help="also fail when a current pair has no passing row")
    parser.add_argument("--require-timing", action="store_true", help="require a timed passing row for every pair")
    parser.add_argument("--require-curves", action="store_true", help="require two timed points in every named curve")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="reject bad rows in --include directories, or --directory when none are included",
    )
    parser.add_argument("--release", action="store_true", help="enable every release evidence gate")
    parser.add_argument("--directory", type=Path, default=BUNDLED_REPORTS, help="reports directory to audit")
    parser.add_argument("--include", action="append", type=Path, default=[], help="merge another reports directory")
    args = parser.parse_args()

    require_pass = args.require_pass or args.release
    require_timing = args.require_timing or args.release
    require_curves = args.require_curves or args.release
    require_clean = args.require_clean or (args.release and bool(args.include))
    current, passing, problems = audit_records(KERNELS.values(), args.directory, args.include)
    evidence, evidence_problems = (
        audit_evidence(KERNELS.values(), args.directory, extra_directories=args.include)
        if require_timing or require_curves or args.release
        else ({}, {})
    )
    outcome_problems = audit_outcomes(args.include or [args.directory]) if require_clean else {}
    total = current + len(problems.get("missing", [])) + len(problems.get("stale", []))
    revision = pinned() or "the default branch"
    blocking = problems.get("missing", []) + problems.get("stale", [])
    allowed = []
    gates = []
    if require_pass:
        gates.append(("no pass", problems.get("no_pass", [])))
    if require_timing:
        gates.append(("no timing", evidence_problems.get("no_timing", [])))
    if require_curves:
        gates.append(("curve", evidence_problems.get("incomplete_curve", [])))
    if args.release:
        gates.append(("benchmark error", evidence_problems.get("bench_error", [])))
    filtered = []
    for label, details in gates:
        blocked, exceptions = _without_exceptions(details)
        filtered.append((label, blocked))
        blocking += blocked
        allowed += exceptions
    if require_clean:
        clean_gates = [
            ("new benchmark error", outcome_problems.get("bench_error", [])),
            ("new bad status", outcome_problems.get("bad_status", [])),
        ]
        filtered.extend(clean_gates)
        blocking += [detail for _, details in clean_gates for detail in details]
    if not blocking:
        if not args.quiet:
            print(f"records: {current}/{total} current fingerprints represented at {revision}")
            print(f"passes:  {passing}/{total} implementations have a current passing row")
            if require_timing:
                print(f"timings: {evidence.get('timed', 0)}/{total} implementations have a timed pass")
            if require_curves:
                print(
                    f"curves:  {evidence.get('complete_curves', 0)}/{evidence.get('curves', 0)} "
                    "named implementation curves have at least two timed points"
                )
            if allowed:
                for pair in sorted({_pair(detail) for detail in allowed}):
                    print(f"allowed: {pair} ({QUALITY_EXCEPTIONS[pair]})")
        return
    report = [
        f"records: {current}/{total} current fingerprints represented at {revision}",
        f"passes:  {passing}/{total} implementations have a current passing row",
    ]
    if require_timing:
        report.append(f"timings: {evidence.get('timed', 0)}/{total} implementations have a timed pass")
    if require_curves:
        report.append(
            f"curves:  {evidence.get('complete_curves', 0)}/{evidence.get('curves', 0)} "
            "named implementation curves have at least two timed points"
        )
    for pair in problems.get("missing", []):
        report.append(f"  no rows: {pair}")
    for pair in problems.get("stale", []):
        report.append(f"  stale:   {pair}")
    for label, details in filtered:
        report.extend(f"  {label}: {detail}" for detail in details)
    for pair in sorted({_pair(detail) for detail in allowed}):
        report.append(f"  allowed: {pair} ({QUALITY_EXCEPTIONS[pair]})")
    report.append(
        "\nBenchmark the gaps on a GPU and publish, or pull a revision that already covers them:\n"
        "  uv run python -m popcorn.bench run --out reports.jsonl\n"
        "  uv run python -m popcorn.bench merge --publish reports.jsonl\n"
        "  uv run python -m popcorn.bench pull"
    )
    raise SystemExit("\n".join(report))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Regenerate README badges from the bundled reports.

Usage:
    uv run python scripts/update_readme.py [--check]
"""

import argparse

import popcorn.kernels  # noqa: F401
from popcorn import KERNELS
from popcorn.bench.readme import refresh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the badge block is out of date; write nothing")
    args = parser.parse_args()

    matrix, stale = refresh(ops=KERNELS, write=not args.check)
    print(matrix)
    if args.check and stale:
        raise SystemExit(
            "README badges are out of date. Regenerate and commit the result:\n  uv run python scripts/update_readme.py"
        )


if __name__ == "__main__":
    main()

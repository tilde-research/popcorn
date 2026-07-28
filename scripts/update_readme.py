#!/usr/bin/env python
"""Regenerate the README badges from the bundled reports and print the detailed
per-backend matrix. Runs nothing; see bench_hardware.py to produce rows."""

import popcorn.kernels  # noqa: F401
from popcorn import KERNELS
from popcorn.bench.readme import refresh


def main() -> None:
    print(refresh(ops=KERNELS))


if __name__ == "__main__":
    main()

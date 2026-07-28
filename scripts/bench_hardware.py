#!/usr/bin/env python
"""Owns the bundled report rows. Nothing else writes them.

Runs the full correctness + benchmark grid on this machine's GPU: a thin wrapper
over `python -m popcorn.bench run` putting every registered op (or the ones you
name) against every available implementation, upserting into
`src/popcorn/reports/`. All `run` flags pass through, e.g.:

    scripts/bench_hardware.py                     # everything, 10 reps
    scripts/bench_hardware.py rms_norm swiglu     # two ops
    scripts/bench_hardware.py --limit 24 --reps 5 # quicker sweep

Derived output is left alone: run update_readme.py for the badges and
update_site.py for the kernel explorer data once the rows land.
"""

import sys

import torch

from popcorn.bench.__main__ import main
from popcorn.core.config import device_name


def run() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("bench_hardware.py needs a CUDA device; none is available")
    print(f"benchmarking on {device_name()}")
    sys.argv = [sys.argv[0], "run", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    run()

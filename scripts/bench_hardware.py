#!/usr/bin/env python
"""Run the benchmark grid on the current CUDA device.

Usage:
    uv run python scripts/bench_hardware.py [OP ...] [BENCH OPTIONS]
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

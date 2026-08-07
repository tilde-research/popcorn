"""Installed Popcorn command-line entry point."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

_MODULE_COMMANDS = {
    "bench": "popcorn.bench.__main__",
    "loop": "popcorn.bench.loop",
    "profile": "popcorn.impls._profile",
}
SWEEP_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_sweep.py"
_HELP = """\
usage: popcorn <command> [args]

commands:
  bench     correctness, benchmarking, reports, and publishing
  sweep     resumable multi-node benchmark sweep (source checkout)
  loop      kernel optimization experiment loop
  profile   profile one kernel case under Nsight Compute

Run `popcorn <command> --help` for command-specific options.
"""


def _run_module(module_name: str, command: str, args: list[str]) -> None:
    module = importlib.import_module(module_name)
    previous = sys.argv
    sys.argv = [f"popcorn {command}", *args]
    try:
        module.main(prog=f"popcorn {command}")
    finally:
        sys.argv = previous


def _load_sweep() -> Any:
    if not SWEEP_SCRIPT.is_file():
        raise SystemExit("popcorn sweep is available only from a Popcorn source checkout")
    spec = importlib.util.spec_from_file_location("popcorn_sweep_cli", SWEEP_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load sweep command from {SWEEP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_sweep(args: list[str]) -> None:
    module = _load_sweep()
    previous = sys.argv
    sys.argv = ["popcorn sweep", *args]
    try:
        module.main()
    finally:
        sys.argv = previous


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_HELP, end="")
        return

    command, *forwarded = args
    if command == "sweep":
        _run_sweep(forwarded)
        return
    if module_name := _MODULE_COMMANDS.get(command):
        _run_module(module_name, command, forwarded)
        return

    print(f"popcorn: unknown command {command!r}\n", file=sys.stderr)
    print(_HELP, file=sys.stderr, end="")
    raise SystemExit(2)

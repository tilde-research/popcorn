"""Nsight Compute harness for kernel work.

`python -m popcorn.impls._profile <op> [name=value ...] [+name ...]` builds one
concrete case (dims default to the largest tested size), warms the op up so JIT
builds and autotuning stay out of the capture, then re-runs itself under `ncu`
with profiling scoped to the annotated call. `name=value` overrides a dim or a
scalar argument; `+name` enables an optional tensor.

For custom scripts, run them under `ncu --profile-from-start off` and wrap the
region of interest in `annotate()`.
"""

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import warnings
from collections.abc import Iterator
from typing import Any

import torch

import popcorn.kernels  # noqa: F401
from popcorn import KERNELS
from popcorn.bench.grid import _dim_pool, make_inputs
from popcorn.bench.model import Case


@contextlib.contextmanager
def annotate() -> Iterator[None]:
    """The region ncu captures under --profile-from-start off."""
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _case(op: Any, tokens: list[str], batch: tuple[int, ...] | None, dtype: torch.dtype) -> Case:
    overrides, present = {}, set()
    for token in tokens:
        if token.startswith("+"):
            present.add(token[1:])
            continue
        name, eq, text = token.partition("=")
        if not eq:
            raise SystemExit(f"expected name=value or +name, got {token!r}")
        overrides[name] = _value(text)
    optional = {spec.param for spec in op.specs if spec.optional}
    if unknown := (set(overrides) - op._dims - set(op.arg_pools)) | (present - optional):
        raise SystemExit(f"{op.name}: unknown names {sorted(unknown)}")
    dims = tuple((name, overrides.get(name, max(_dim_pool(op, name)))) for name in sorted(op._dims))
    if required := [name for name, pool in op.arg_pools.items() if not pool and name not in overrides]:
        raise SystemExit(f"{op.name}: pass {', '.join(f'{name}=...' for name in sorted(required))} (no tested default)")
    args = tuple((name, overrides[name] if name in overrides else op.arg_pools[name][0]) for name in sorted(op.arg_pools))
    if batch is None:
        batch = (2, 2048) if any(... in spec.tokens for spec in op.specs) else ()
    return Case(dims, batch, dtype, args, frozenset(present))


def _once(op: Any, inputs: dict[str, Any], backend: str, grads: list[torch.Tensor] | None) -> list[torch.Tensor]:
    outputs = op(**inputs, backend=backend)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    floating = [out for out in outputs if isinstance(out, torch.Tensor) and out.is_floating_point()]
    if grads is not None:
        torch.autograd.backward(floating, grads)
    return floating


def _worker(args: argparse.Namespace) -> None:
    warnings.simplefilter("ignore")
    op = KERNELS[args.op]
    case = _case(op, args.case, args.batch, getattr(torch, args.dtype))
    inputs = make_inputs(op, case, grad=args.grad)
    outputs = _once(op, inputs, args.backend, None)
    grads = [torch.randn_like(out) for out in outputs] if args.grad else None
    _once(op, inputs, args.backend, grads)
    print(f"profiling {op.name}:{args.backend} {case} grad={args.grad}", file=sys.stderr)
    with annotate():
        _once(op, inputs, args.backend, grads)


def _batch(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m popcorn.impls._profile", description="Profile one op case under Nsight Compute."
    )
    parser.add_argument("op", choices=sorted(KERNELS))
    parser.add_argument("case", nargs="*", metavar="name=value|+name", help="dim/arg overrides, +name for optional tensors")
    parser.add_argument("--backend", default="popcorn")
    parser.add_argument("--batch", type=_batch, help="leading batch dims, e.g. 8,2048 (default: largest tested)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--grad", action="store_true", help="profile the backward as well")
    parser.add_argument("--set", default="basic", help="ncu section set: basic, detailed, full")
    parser.add_argument("--kernel", help="ncu kernel-name regex filter")
    parser.add_argument("--out", help="write an .ncu-rep file instead of printing to stdout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    ncu = shutil.which("ncu")
    if not ncu:
        raise SystemExit("ncu not found on PATH (load Nsight Compute or the CUDA toolkit module)")
    command = [ncu, "--profile-from-start", "off", "--target-processes", "all", "--set", args.set]
    if args.kernel:
        command += ["--kernel-name", f"regex:{args.kernel}"]
    if args.out:
        command += ["--force-overwrite", "--export", args.out]
    command += [sys.executable, "-m", "popcorn.impls._profile", *sys.argv[1:], "--worker"]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()

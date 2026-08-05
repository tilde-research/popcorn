#!/usr/bin/env python
"""Popcorn benchmark sweep: one submitit job, one isolated backend at a time, resumable.

Usage:
    uv run python scripts/bench_sweep.py submit [--nodes N] [--curves-only] [--watch]
    uv run python scripts/bench_sweep.py resume [RUN] [--watch]
    uv run python scripts/bench_sweep.py watch [RUN]
    uv run python scripts/bench_sweep.py status [RUN]

A single Slurm job (`--nodes`, capped at 6, with 8 GPUs each) walks
the phases in order: torch references first, then each backend family in a freshly
installed environment. Task 0 coordinates: it builds each phase environment, plans
worker manifests from the current store (cached rows skipped, infeasible shapes dropped,
cheapest first), and folds shards into the run cache every few minutes. Every task
supervises one in-process GPU worker and restarts it when it dies; a sentinel written
before each case converts the killer into a `crash` row on restart, and a per-case
watchdog converts hangs into `timeout` rows. Near the walltime Slurm raises SIGUSR2,
workers drain after their current case, and the coordinator merges and resubmits the
job; planning is idempotent, so every attempt is a no-op for finished work and `resume`
is the same operation by hand. Results stay in the run cache until reviewed and merged;
this script never publishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from popcorn.bench.sweep import (
    GPUS_PER_NODE,
    MAX_NODES,
    PHASES,
    SweepConfig,
    affinity_units,
    distribute,
    estimated as _estimated,
    expected_totals as _expected_totals,
    infeasible as _infeasible,
    minutes as _minutes,
    new_state as _new_state,
    phase_spec,
    phase_work as _build_phase_work,
    render,
    stratum as _stratum,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
RUNS = ROOT / "logs" / "sweeps"
STATE = "state.json"
CONFIG = "config.json"
CONTROL = "control.json"
TERMINAL = {"completed", "failed", "paused"}
SETTLED = {"completed", "skipped"}
WATCHDOG_EXIT = 90
POISON_EXIT = 91
BAD_OUTCOMES = ("fail", "crash", "error")
REFERENCE_TIMEOUT = 60
# CUDA reports these asynchronously and never recovers the context: every later kernel in the
# process fails too, so the worker must recycle after recording the case that surfaced one.
POISON_MARKS = ("illegal memory access", "device-side assert", "unspecified launch failure", "misaligned address")
MAX_ATTEMPTS = 40
MAX_RESTARTS = 60
MERGE_EVERY = 600.0
STALL_LIMIT = 1800.0


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nodes(value: str) -> int:
    parsed = _positive(value)
    if parsed > MAX_NODES:
        raise argparse.ArgumentTypeError(f"must not exceed {MAX_NODES}")
    return parsed


def _read_json(path: Path, default: Any = None) -> Any:
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            if attempt < 2:
                time.sleep(0.05)
    return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_records(path: Path) -> list[Any]:
    """Shard rows, tolerating the torn trailing line a killed worker can leave behind."""
    from popcorn.bench.model import Record

    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            records.append(Record.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue  # torn write; the case simply reruns
    return records


def _append_record(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as sink:
        sink.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        sink.flush()


# --------------------------------------------------------------------------- state


def _phase(state: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in state["phases"] if item["name"] == name)


def _save_state(run: Path, state: dict[str, Any]) -> None:
    _write_json(run / STATE, state)


# --------------------------------------------------------------------------- planning


def expected_totals(phases: list[str], only: set[str] | None = None, curves_only: bool = False) -> dict[str, int]:
    """Registered work per phase from the full grid, before cache and budget pruning."""
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS

    return _expected_totals(KERNELS.values(), phases, only, curves_only)


def _phase_work(phase_name: str, only: set[str] | None = None, curves_only: bool = False) -> list[tuple[Any, str, Any, bool]]:
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS

    try:
        return _build_phase_work(KERNELS.values(), phase_name, only, curves_only)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


def plan_phase(
    run: Path,
    phase_name: str,
    workers: int,
    only: set[str] | None = None,
    curves_only: bool = False,
) -> dict[str, Any]:
    import torch

    from popcorn.bench.__main__ import _fill_pending, _record_index, _usable_records
    from popcorn.bench.model import Case
    from popcorn.bench.plan import EXHAUSTED, BudgetFrontier
    from popcorn.bench.store import Store
    from popcorn.core.config import device_name

    phase_dir = run / phase_name
    manifests = phase_dir / "manifests"
    progress = phase_dir / "progress"
    manifests.mkdir(parents=True, exist_ok=True)
    progress.mkdir(parents=True, exist_ok=True)
    for path in (*manifests.glob("*.jsonl"), *progress.glob("*.json")):
        path.unlink()

    _merge_shards(run, phase_name)
    work = _phase_work(phase_name, only, curves_only)
    ops = {op.name: op for op, _, _, _ in work}
    triples = list({(op.name, impl): (op, impl, case) for op, impl, case, _ in work}.values())
    device = device_name("cuda")
    capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    store = Store()
    current = _usable_records(store, triples, device)
    cached = _record_index(current)
    frontiers: dict[tuple[str, str, bool], BudgetFrontier] = {}
    for record in current:
        if record.result.status in EXHAUSTED:
            key = (record.op, record.impl, record.result.grad)
            frontiers.setdefault(key, BudgetFrontier()).add(Case.from_config(record.config))

    reference_frontiers: dict[tuple[str, bool], BudgetFrontier] = {}
    if phase_name != "reference":
        references = list({op.name: (op, "torch", case) for op, _, case, _ in work}.values())
        for record in _usable_records(store, references, device):
            if record.result.status in EXHAUSTED:
                key = (record.op, record.result.grad)
                reference_frontiers.setdefault(key, BudgetFrontier()).add(Case.from_config(record.config))

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    exact_cached = 0
    infeasible = 0
    prior_pruned = 0
    for op, impl, case, grad in work:
        record = cached.get((op.name, impl, case.case_id, grad))
        if not _fill_pending(record):
            exact_cached += 1
            continue
        if _infeasible(op, case, grad, capacity):
            infeasible += 1
            continue
        reference_frontier = reference_frontiers.get((op.name, grad))
        if reference_frontier is not None and reference_frontier.blocker(case) is not None:
            prior_pruned += 1
            continue
        frontier = frontiers.setdefault((op.name, impl, grad), BudgetFrontier())
        if frontier.blocker(case) is not None:
            prior_pruned += 1
            continue
        key = (op.name, impl, grad, *_stratum(case))
        group = groups.setdefault(key, {"op": op.name, "impl": impl, "grad": grad, "cases": []})
        group["cases"].append(case)

    # Cheapest first within each stratum, so a worker's frontier meets the OOM or timeout
    # boundary at the small end and prunes everything beyond it instead of paying for it.
    for group in groups.values():
        op = ops[group["op"]]
        group["cases"] = [case.config() for case in sorted(group["cases"], key=lambda case: _estimated(op, case))]

    units = affinity_units(list(groups.values()), workers)
    bins, loads = distribute(units, workers)
    for rank, assigned in enumerate(bins):
        path = manifests / f"{rank}.jsonl"
        groups_for_worker = [group for chunk in assigned for group in chunk]
        path.write_text("".join(json.dumps(group, sort_keys=True) + "\n" for group in groups_for_worker))

    summary = {
        "phase": phase_name,
        "device": device,
        "torch": torch.__version__,
        "total": len(work),
        "cached": exact_cached,
        "infeasible": infeasible,
        "prior_pruned": prior_pruned,
        "scheduled": sum(loads),
        "groups": len(groups),
        "chunks": len(units),
        "workers": workers,
        "max_worker_cases": max(loads, default=0),
    }
    _write_json(phase_dir / "plan.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


# --------------------------------------------------------------------------- environments


def _run_logged(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as sink:
        sink.write("$ " + shlex.join(command) + "\n")
        sink.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT, check=True)


def install_environment(phase_name: str, destination: Path, log: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    shutil.rmtree(destination, ignore_errors=True)
    env = os.environ | {
        "POPCORN_SKIP_REPORTS": "1",
        "MAX_JOBS": os.getenv("MAX_JOBS", "8"),
        "CMAKE_BUILD_PARALLEL_LEVEL": os.getenv("CMAKE_BUILD_PARALLEL_LEVEL", "8"),
    }
    # Pin the interpreter to the driver's: fingerprints and torch wheels must not drift
    # with whatever default python uv happens to prefer on the node.
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    _run_logged([uv, "venv", "--clear", "--python", version, str(destination)], log, env)
    python = destination / "bin" / "python"
    spec = phase_spec(phase_name)
    if spec.build_requirements:
        _run_logged(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--torch-backend=auto",
                "torch",
                "setuptools",
                "wheel",
                "packaging",
            ],
            log,
            env,
        )
    _run_logged(
        [uv, "pip", "install", "--python", str(python), "--torch-backend=auto", "-e", spec.target, *spec.packages],
        log,
        env,
    )
    return python


def _lock_fingerprint(phase_name: str) -> str:
    spec = phase_spec(phase_name)
    digest = hashlib.sha256(phase_name.encode())
    digest.update((spec.extra or "").encode())
    digest.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    for package in spec.packages:
        digest.update(package.encode())
    for name in ("uv.lock", "pyproject.toml"):
        path = ROOT / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _ensure_environment(phase_name: str, destination: Path, log: Path) -> Path:
    """Build the phase environment, or reuse it when the same attempt was already built.

    A drained job resubmits itself mid-phase; rebuilding an identical environment on the
    next attempt would cost minutes per phase for nothing, so a marker records what the
    environment contains and a matching marker short-circuits the build.
    """
    marker = destination.with_name(destination.name + ".json")
    fingerprint = _lock_fingerprint(phase_name)
    python = destination / "bin" / "python"
    if python.exists() and _read_json(marker) == fingerprint:
        return python
    marker.unlink(missing_ok=True)
    python = install_environment(phase_name, destination, log)
    _write_json(marker, fingerprint)
    return python


def _cuda_mismatch(nvcc_banner: str | None, torch_cuda: str | None) -> str | None:
    """Why flash-attn-3 cannot compile here, or None when the toolchain lines up."""
    if nvcc_banner is None:
        return "nvcc is not on PATH; flash-attn-3 builds from source"
    if torch_cuda is None:
        return "torch has no CUDA runtime; flash-attn-3 builds against it"
    matched = re.search(r"release (\d+)\.(\d+)", nvcc_banner)
    if matched is None:
        return "could not parse the nvcc version banner"
    nvcc_version = (int(matched[1]), int(matched[2]))
    torch_version = tuple(int(part) for part in torch_cuda.split(".")[:2])
    if nvcc_version < torch_version:
        return (
            f"nvcc {nvcc_version[0]}.{nvcc_version[1]} is older than torch's CUDA {torch_cuda}; "
            "flash-attn-3 cannot compile against it"
        )
    return None


def preflight_fa3() -> str | None:
    """Skip reason for the fa3 phase, checked before the expensive source build."""
    import torch

    nvcc = shutil.which("nvcc")
    banner = None
    if nvcc is not None:
        banner = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False).stdout
    return _cuda_mismatch(banner, torch.version.cuda)


def preflight_cudnn(python: Path) -> str | None:
    """Skip reason when the isolated cuDNN frontend cannot execute on this host."""
    code = """
import torch
from popcorn.kernels import attn

q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
attn(q, k, v, causal=True, backend="cudnn").float().square().mean().backward()
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        env=os.environ | {"POPCORN_SKIP_REPORTS": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return f"cuDNN frontend preflight failed: {lines[-1] if lines else f'exit {result.returncode}'}"


# --------------------------------------------------------------------------- worker


def _budget(op: Any, case: Any, timeout: int, phase_name: str | None = None) -> int:
    """Per-case watchdog seconds: a flat floor that grows with input bytes.

    The floor must absorb Triton compile and autotune, which re-run per shape and dwarf
    small cases, so it never shrinks below --timeout. Bytes extend it linearly so a large
    but feasible case is not misclassified as a hang, capped at 3x so a true hang cannot
    hold a GPU for long. Frontier pruning keeps repeat timeouts from recurring above the
    first one. Torch references have neither compile nor autotune, so they use a short cap.
    """
    if phase_name == "reference":
        return min(timeout, REFERENCE_TIMEOUT)
    scaled = timeout + 60 * _estimated(op, case) / 2**30
    return int(min(scaled, 3 * timeout))


def _convert_sentinel(sentinel: Path, kernels: Any, device: str, done: set[Any]) -> Any:
    """The crash row for a case a dead worker left behind, or None when there is none.

    The sentinel is written before a case runs and cleared after its row lands, so on
    restart an existing sentinel means the previous worker died mid-case: the case gets a
    `crash` row (conclusive, like `_isolated` would record) instead of being retried into
    the same crash forever.
    """
    from popcorn.bench.model import Case

    payload = _read_json(sentinel)
    if payload is None:
        return None
    sentinel.unlink(missing_ok=True)
    op = kernels.get(payload.get("op"))
    if op is None:
        return None
    case = Case.from_config(payload["config"])
    key = (payload["op"], payload["impl"], case.case_id, bool(payload["grad"]))
    if key in done:
        return None
    record = op.bench.incomplete(
        payload["impl"],
        case,
        device,
        reason="worker process died running this case; see the worker log",
        grad=bool(payload["grad"]),
    )
    record.result.status = "crash"
    return record


def run_worker(run: Path, phase_name: str, rank: int, reps: int, timeout: int, drain: Path, hardware: str | None) -> None:
    # One compile-cache directory per rank, recycled at each phase change: per-phase dirs
    # accumulated until they filled node boot disks (prolog then drains the node), while a
    # blanket wipe on start would re-autotune after every crash restart. The marker keeps
    # phases isolated (extension builds are torch-specific) but restarts warm.
    scratch = Path(f"/tmp/popcorn_{rank}")
    marker = scratch / "phase"
    if not marker.exists() or marker.read_text() != phase_name:
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        marker.write_text(phase_name)
    for variable, cache_name in (
        ("TRITON_CACHE_DIR", str(scratch / "triton")),
        ("TORCH_EXTENSIONS_DIR", str(scratch / "extensions")),
        ("TORCHINDUCTOR_CACHE_DIR", str(scratch / "inductor")),
    ):
        os.environ[variable] = cache_name
        Path(cache_name).mkdir(parents=True, exist_ok=True)

    import torch

    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.bench.__main__ import _target_device
    from popcorn.bench.model import Case
    from popcorn.bench.plan import EXHAUSTED, BudgetFrontier

    phase_dir = run / phase_name
    manifest = phase_dir / "manifests" / f"{rank}.jsonl"
    shard = phase_dir / "shards" / f"{rank}.jsonl"
    sentinel = phase_dir / "sentinels" / f"{rank}.json"
    progress = phase_dir / "progress" / f"{rank}.json"
    groups = [json.loads(line) for line in manifest.read_text().splitlines() if line] if manifest.exists() else []
    device = _target_device(argparse.Namespace(device="cuda", hardware=hardware))

    existing = _read_records(shard)
    done = {(record.op, record.impl, record.case_id, record.result.grad) for record in existing}
    crashed = _convert_sentinel(sentinel, KERNELS, device, done)
    if crashed is not None:
        _append_record(shard, crashed)
        existing.append(crashed)
        done.add((crashed.op, crashed.impl, crashed.case_id, crashed.result.grad))
    statuses = Counter(record.result.status for record in existing)
    bench_errors = sum(bool(record.result.bench_error) for record in existing)
    measured = len(existing)
    pruned = 0
    frontiers: dict[tuple[str, str, bool], BudgetFrontier] = {}
    for record in existing:
        if record.result.status in EXHAUSTED:
            key = (record.op, record.impl, record.result.grad)
            frontiers.setdefault(key, BudgetFrontier()).add(Case.from_config(record.config))

    def report(status: str, error: str = "") -> None:
        _write_json(
            progress,
            {
                "status": status,
                "measured": measured,
                "pruned": pruned,
                "bench_errors": bench_errors,
                "statuses": dict(statuses),
                "error": error,
            },
        )

    last_update = 0.0
    report("running")
    try:
        for group in groups:
            op = KERNELS[group["op"]]
            impl = group["impl"]
            grad = bool(group["grad"])
            frontier = frontiers.setdefault((op.name, impl, grad), BudgetFrontier())
            for config in group["cases"]:
                case = Case.from_config(config)
                key = (op.name, impl, case.case_id, grad)
                if key in done:
                    continue
                if drain.exists():
                    report("drained")
                    return
                if frontier.blocker(case) is not None:
                    pruned += 1
                else:
                    budget = _budget(op, case, timeout, phase_name)

                    def abort() -> None:
                        hung = op.bench.incomplete(impl, case, device, reason=f"no result within {budget}s", grad=grad)
                        hung.result.status = "timeout"
                        _append_record(shard, hung)
                        sentinel.unlink(missing_ok=True)
                        os._exit(WATCHDOG_EXIT)

                    _write_json(sentinel, {"op": op.name, "impl": impl, "grad": grad, "config": case.config()})
                    watchdog = threading.Timer(budget, abort)
                    watchdog.daemon = True
                    watchdog.start()
                    try:
                        record = op.bench.run_case(impl, case, device, reps, grad=grad)
                    except torch.OutOfMemoryError as error:
                        record = op.bench.incomplete(impl, case, device, reason=str(error).splitlines()[0], grad=grad)
                        record.result.status = "oom"
                    except Exception as error:  # a poisoned case must not sink the worker
                        record = op.bench.incomplete(impl, case, device, reason=f"{type(error).__name__}: {error}", grad=grad)
                    finally:
                        watchdog.cancel()
                    _append_record(shard, record)
                    sentinel.unlink(missing_ok=True)
                    done.add(key)
                    measured += 1
                    statuses[record.result.status] += 1
                    bench_errors += bool(record.result.bench_error)
                    if record.result.status in EXHAUSTED:
                        frontier.add(case)
                        torch.cuda.empty_cache()
                    if any(mark in (record.result.reason or "") for mark in POISON_MARKS):
                        report("running")
                        os._exit(POISON_EXIT)
                now = time.monotonic()
                if now - last_update >= 1:
                    report("running")
                    last_update = now
        report("completed")
    except Exception as error:
        report("failed", error=f"{type(error).__name__}: {error}")
        raise


# --------------------------------------------------------------------------- job internals


class SweepTask:
    """The submitit payload: every Slurm task runs this, rank 0 coordinates.

    Holds only the run path and reloads this script from disk before running: cloudpickle
    serializes __main__ functions by value, so without the reload a drained job's resubmit
    would forever re-run the bytecode of the original submit, and a coordinator fix could
    never reach a live run.
    """

    def __init__(self, run: str) -> None:
        self.run = run

    def __call__(self) -> None:
        import importlib.util

        import submitit

        spec = importlib.util.spec_from_file_location("popcorn_bench_sweep", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        environment = submitit.JobEnvironment()
        module.sweep_main(Path(self.run), environment.global_rank, environment.num_tasks, str(environment.job_id))


def sweep_main(run: Path, rank: int, world: int, job: str) -> None:
    raw_config = _read_json(run / CONFIG)
    if raw_config is None:
        raise SystemExit(f"missing {run / CONFIG}")
    config = SweepConfig.from_mapping(raw_config)
    drain = run / f"drain-{job}"
    requeue = run / f"requeue-{job}"

    def request_drain(with_requeue: bool):
        def handle(signum: int, frame: Any) -> None:
            if with_requeue:
                requeue.touch()
            drain.touch()

        return handle

    # SIGUSR2 is the walltime warning submitit configures: drain and resubmit. SIGTERM is
    # scancel or preemption: drain, but leave resumption to a human `resume`.
    signal.signal(signal.SIGUSR2, request_drain(with_requeue=True))
    signal.signal(signal.SIGTERM, request_drain(with_requeue=False))
    os.environ["POPCORN_CACHE_DIR"] = str(run / "cache")
    os.environ["POPCORN_SKIP_REPORTS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        if rank == 0:
            _coordinate(run, config, world, job, drain, requeue)
        else:
            _follow(run, rank, job, drain)
    finally:
        # Leave node /tmp the way we found it; a drained attempt pays one re-autotune.
        shutil.rmtree(f"/tmp/popcorn_{rank}", ignore_errors=True)


def _quality_error(statuses: dict[str, int], bench_errors: int = 0) -> str:
    problems = [f"{status}={statuses[status]}" for status in BAD_OUTCOMES if statuses.get(status)]
    if bench_errors:
        problems.append(f"bench_error={bench_errors}")
    return ", ".join(problems)


def _aggregate_progress(phase_dir: Path) -> dict[str, Any]:
    measured = 0
    pruned = 0
    bench_errors = 0
    statuses: Counter[str] = Counter()
    failed = []
    terminal = 0
    for path in sorted((phase_dir / "progress").glob("*.json")):
        row = _read_json(path, {})
        measured += int(row.get("measured", 0))
        pruned += int(row.get("pruned", 0))
        bench_errors += int(row.get("bench_errors", 0))
        statuses.update(row.get("statuses", {}))
        if row.get("status") in ("completed", "drained", "failed"):
            terminal += 1
        if row.get("status") == "failed":
            failed.append(f"{path.stem}: {row.get('error', 'worker failed')}")
    return {
        "measured": measured,
        "pruned": pruned,
        "bench_errors": bench_errors,
        "statuses": dict(statuses),
        "failed": failed,
        "terminal": terminal,
    }


def _ranks_settled(phase_dir: Path, world: int, started: float) -> tuple[bool, str]:
    """Whether every rank finished the phase, treating long-silent ranks as lost."""
    files = list((phase_dir / "progress").glob("*.json"))
    aggregate = _aggregate_progress(phase_dir)
    if len(files) >= world and aggregate["terminal"] >= world:
        return True, ""
    stamps = []
    for path in files:
        try:
            stamps.append(path.stat().st_mtime)
        except FileNotFoundError:
            # A worker on another node replaced the file between glob and stat; NFS
            # surfaces that as a vanished path. It reappears on the next pass.
            continue
    newest = max(stamps, default=started)
    if time.time() - newest > STALL_LIMIT:
        missing = world - aggregate["terminal"]
        return True, f"{missing} worker(s) went silent; their remainder replans next attempt"
    return False, ""


def _merge_shards(run: Path, phase_name: str, unlink: bool = True) -> int:
    """Fold worker shards into the run cache; incremental calls leave shards in place."""
    from popcorn.bench.store import user_reports, write

    shards = sorted((run / phase_name / "shards").glob("*.jsonl"))
    records = [record for shard in shards for record in _read_records(shard)]
    if records:
        write(records, user_reports())
        if unlink:
            for shard in shards:
                shard.unlink()
    return len(records)


def _supervise(
    run: Path,
    phase_name: str,
    rank: int,
    python: Path,
    reps: int,
    timeout: int,
    hardware: str | None,
    drain: Path,
    tick: Any = None,
) -> None:
    """Keep one worker subprocess alive until it finishes cleanly or wears out its restarts."""
    phase_dir = run / phase_name
    log = phase_dir / "logs" / f"{rank}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(SCRIPT),
        "worker",
        "--run",
        str(run),
        "--phase",
        phase_name,
        "--rank",
        str(rank),
        "--reps",
        str(reps),
        "--timeout",
        str(timeout),
        "--drain",
        str(drain),
    ]
    if hardware:
        command += ["--hardware", hardware]
    restarts = 0
    while True:
        with log.open("a") as sink:
            sink.write("$ " + shlex.join(command) + "\n")
            sink.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=sink, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if tick is not None:
                    tick()
                time.sleep(2)
        if process.returncode == 0:
            return
        if drain.exists():
            return  # the job is ending; whatever was lost replans on the next attempt
        restarts += 1
        if restarts >= MAX_RESTARTS:
            progress = phase_dir / "progress" / f"{rank}.json"
            row = _read_json(progress, {}) or {}
            row.update(status="failed", error=f"worker kept dying (exit {process.returncode} after {restarts} restarts)")
            _write_json(progress, row)
            return
        time.sleep(5)


def _run_phase(
    run: Path,
    phase_name: str,
    state: dict[str, Any],
    phase: dict[str, Any],
    world: int,
    python: Path,
    config: SweepConfig,
    drain: Path,
    requeue: Path,
) -> None:
    phase_dir = run / phase_name
    started = time.time()
    last_merge = time.monotonic()
    last_print = 0.0
    quality_error = ""

    def tick() -> None:
        nonlocal last_merge, last_print, quality_error
        aggregate = _aggregate_progress(phase_dir)
        phase.update(
            measured=aggregate["measured"],
            pruned=aggregate["pruned"],
            bench_errors=aggregate["bench_errors"],
            statuses=aggregate["statuses"],
        )
        if phase_name == "reference" and not quality_error:
            quality_error = _quality_error(aggregate["statuses"], aggregate["bench_errors"])
            if quality_error:
                drain.touch()
                phase["message"] = f"stopping on invalid reference evidence: {quality_error}"
        _save_state(run, state)
        now = time.monotonic()
        if now - last_merge >= MERGE_EVERY:
            _merge_shards(run, phase_name, unlink=False)
            last_merge = now
        if now - last_print >= 30:
            print(render(state), flush=True)
            last_print = now

    _supervise(run, phase_name, 0, python, config.reps, config.timeout, config.hardware, drain, tick)
    drained_at = None
    while True:
        settled, note = _ranks_settled(phase_dir, world, started)
        if settled:
            if note:
                requeue.touch()
                drain.touch()
            break
        # Draining workers stop within one case; wait a little for their final flush, but
        # never long enough to eat the walltime grace the merge below still needs.
        if drain.exists():
            drained_at = drained_at or time.monotonic()
            if time.monotonic() - drained_at > 120:
                note = "drained before every worker reported back"
                break
        tick()
        time.sleep(2)

    aggregate = _aggregate_progress(phase_dir)
    phase.update(
        measured=aggregate["measured"],
        pruned=aggregate["pruned"],
        bench_errors=aggregate["bench_errors"],
        statuses=aggregate["statuses"],
    )
    merged = _merge_shards(run, phase_name, unlink=True)
    quality_error = quality_error or (
        _quality_error(aggregate["statuses"], aggregate["bench_errors"]) if phase_name == "reference" else ""
    )
    if quality_error:
        raise RuntimeError(f"invalid reference evidence: {quality_error}")
    if drain.exists():
        message = f"drained; {merged:,} rows merged so far"
        phase.update(status="paused", message=f"{message}; {note}" if note else message)
    elif aggregate["failed"]:
        raise RuntimeError("; ".join([*aggregate["failed"][:3], note]) if note else "; ".join(aggregate["failed"][:3]))
    else:
        message = f"{merged:,} observed rows saved"
        phase.update(status="completed", message=f"{message}; {note}" if note else message)


def _coordinate(run: Path, config: SweepConfig, world: int, job: str, drain: Path, requeue: Path) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = _read_json(run / STATE) or _new_state(run, list(config.phases))
    state.update(job_id=job, status="running", workers=world, finished=None)
    state["started"] = state.get("started") or now
    state["attempts"] = int(state.get("attempts", 0)) + 1
    _save_state(run, state)
    environment = run / "environment"
    control = run / CONTROL
    only = set(config.ops) if config.ops else None
    failures = []

    for phase_name in config.phases:
        phase = _phase(state, phase_name)
        if phase["status"] in SETTLED:
            continue
        if drain.exists():
            break
        phase_dir = run / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        try:
            if phase_name == "fa3" and (reason := preflight_fa3()):
                phase.update(status="skipped", message=reason)
                continue
            phase.update(status="installing", message="building isolated environment")
            _save_state(run, state)
            python = _ensure_environment(phase_name, environment, phase_dir / "install.log")
            if phase_name == "cudnn" and (reason := preflight_cudnn(python)):
                phase.update(status="skipped", message=reason)
                continue

            phase.update(status="planning", message="planning from the current store")
            _save_state(run, state)
            command = [str(python), str(SCRIPT), "plan", "--run", str(run), "--phase", phase_name, "--workers", str(world)]
            if only:
                command += ["--ops", *sorted(only)]
            if config.curves_only:
                command.append("--curves-only")
            _run_logged(command, phase_dir / "plan.log", os.environ.copy())
            summary = _read_json(phase_dir / "plan.json", {})
            phase.update(
                status="running",
                total=summary.get("total", 0),
                cached=summary.get("cached", 0),
                prior_pruned=summary.get("prior_pruned", 0),
                infeasible=summary.get("infeasible", 0),
                measured=0,
                pruned=0,
                bench_errors=0,
                statuses={},
                message=f"{summary.get('groups', 0):,} strata; max shard {summary.get('max_worker_cases', 0):,}",
            )
            _save_state(run, state)
            if not summary.get("scheduled", 0):
                phase.update(status="completed", message="all rows already cached, pruned, or infeasible")
                continue

            _write_json(
                control,
                {
                    "job": job,
                    "status": "ready",
                    "phase": phase_name,
                    "python": str(python),
                    "reps": config.reps,
                    "timeout": config.timeout,
                    "hardware": config.hardware,
                },
            )
            _run_phase(run, phase_name, state, phase, world, python, config, drain, requeue)
        except Exception as error:
            traceback.print_exc()
            phase.update(status="failed", message=f"{type(error).__name__}: {error}")
            failures.append(phase_name)
        finally:
            _save_state(run, state)
            print(render(state), flush=True)

    _write_json(control, {"job": job, "status": "done"})
    unfinished = [item["name"] for item in state["phases"] if item["status"] not in SETTLED]
    if drain.exists() and requeue.exists() and unfinished:
        if state["attempts"] >= MAX_ATTEMPTS:
            state["status"] = "failed"
            state["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _phase(state, unfinished[0])["message"] = f"gave up after {MAX_ATTEMPTS} attempts"
        else:
            try:
                new_job = _launch(run, config)
                state.update(status="requeued", job_id=new_job)
                print(f"walltime drain: resubmitted as job {new_job}", flush=True)
            except Exception as error:
                state["status"] = "paused"
                state["message"] = (
                    f"resubmit failed ({type(error).__name__}: {error}); continue with: bench_sweep.py resume {run}"
                )
    elif failures or (drain.exists() and unfinished):
        state["status"] = "paused" if not failures and unfinished else "failed"
        state["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        state["status"] = "completed"
        state["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_state(run, state)
    print(render(state), flush=True)
    print(f"run cache: {run / 'cache'}", flush=True)


def _ready_phase(control: Any, job: str, handled: set[str]) -> str | None:
    """The phase a follower should start working on, if the control file names one."""
    if not isinstance(control, dict) or control.get("job") != job or control.get("status") != "ready":
        return None
    phase = control.get("phase")
    return phase if isinstance(phase, str) and phase not in handled else None


def _follow(run: Path, rank: int, job: str, drain: Path) -> None:
    """Non-zero ranks: supervise one worker per phase as the coordinator announces them."""
    handled: set[str] = set()
    while True:
        control = _read_json(run / CONTROL, {})
        if isinstance(control, dict) and control.get("job") == job and control.get("status") == "done":
            return
        phase_name = _ready_phase(control, job, handled)
        if phase_name is not None:
            python = Path(control["python"])
            _supervise(run, phase_name, rank, python, control["reps"], control["timeout"], control.get("hardware"), drain)
            handled.add(phase_name)
        time.sleep(5)


# --------------------------------------------------------------------------- submission


def _slurm_parameters(config: SweepConfig | dict[str, Any]) -> dict[str, Any]:
    config = config if isinstance(config, SweepConfig) else SweepConfig.from_mapping(config)
    return {
        "job_name": "popcorn-sweep",
        "nodes": config.nodes,
        "ntasks_per_node": GPUS_PER_NODE,
        "gpus_per_task": 1,
        "cpus_per_task": 8,
        "qos": config.qos,
        "time": _minutes(config.walltime),
        "exclusive": True,
        # Singleton serializes every job named popcorn-sweep, so a drained job's resubmit
        # queues behind nothing else of ours and usage can never exceed one allocation.
        "dependency": "singleton",
        # Walltime warning: enough notice to finish the slowest case budget and merge.
        "signal_delay_s": 900,
        "stderr_to_stdout": True,
    }


def _launch(run: Path, config: SweepConfig) -> str:
    import submitit

    if config.local:
        executor: Any = submitit.LocalExecutor(folder=str(run / "submitit"))
        # Without gpus_per_node the local controller exports CUDA_VISIBLE_DEVICES=""
        # and the task sees no GPUs; without timeout_min it drains every 2 minutes.
        executor.update_parameters(gpus_per_node=1, timeout_min=24 * 60)
    else:
        executor = submitit.SlurmExecutor(folder=str(run / "submitit"))
        executor.update_parameters(**_slurm_parameters(config))
    job = executor.submit(SweepTask(str(run)))
    return str(job.job_id)


def _queued_sweeps() -> list[str]:
    return subprocess.run(
        ["squeue", "--noheader", "--user", os.getenv("USER", ""), "--name", "popcorn-sweep", "--format=%i"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def _submit(run: Path, config: SweepConfig, dry_run: bool, revalidate: bool = False) -> None:
    run.mkdir(parents=True, exist_ok=True)
    _write_json(run / CONFIG, config.to_dict())
    state = _read_json(run / STATE) or _new_state(run, list(config.phases))
    if revalidate:
        for phase in state["phases"]:
            if phase["status"] in SETTLED:
                phase.update(status="pending", message="revalidating current source, environment, and case plan")
    state["workers"] = config.workers
    state.pop("slurm_state", None)
    only = set(config.ops) if config.ops else None
    totals = expected_totals(list(config.phases), only, config.curves_only)
    for phase in state["phases"]:
        if phase["status"] not in SETTLED:
            phase["total"] = totals.get(phase["name"], 0)
    _save_state(run, state)
    if dry_run:
        print(json.dumps({"run": str(run), "config": config.to_dict(), "totals": totals}, indent=2, sort_keys=True))
        return
    if not config.local and (queued := _queued_sweeps()):
        raise SystemExit(
            f"refusing to submit another sweep while job(s) {', '.join(queued)} are active; "
            f"this guard keeps usage at {config.nodes} nodes"
        )
    job_id = _launch(run, config)
    state.update(job_id=job_id, status="queued")
    _save_state(run, state)
    workers = state["workers"]
    print(f"submitted job {job_id}: {'local, 1 worker' if config.local else f'{config.nodes} nodes / {workers} workers'}")
    print(f"run: {run}")


def latest_run() -> Path:
    found = sorted(path for path in RUNS.glob("*") if (path / STATE).exists())
    if not found:
        raise SystemExit("no sweep runs found")
    return found[-1]


def _slurm_state(job_id: str) -> str | None:
    try:
        queued = subprocess.run(
            ["squeue", "--noheader", "--jobs", job_id, "--format=%T"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if queued:
            return queued.splitlines()[0]
        accounted = subprocess.run(
            ["sacct", "--jobs", job_id, "--noheader", "-X", "--format=State"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return accounted.splitlines()[0].strip().rstrip("+") if accounted else None
    except OSError:
        return None


def watch(run: Path) -> None:
    last = ""
    try:
        while True:
            state = _read_json(run / STATE)
            if state is None:
                time.sleep(2)  # tolerate the brief NFS replace window
                continue
            if state.get("status") not in TERMINAL and (job_id := state.get("job_id")):
                slurm = _slurm_state(job_id)
                if slurm and slurm not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                    state["slurm_state"] = slurm
                    fresh = _read_json(run / STATE, state)
                    if fresh.get("job_id") == job_id and fresh.get("status") not in ("completed", "requeued"):
                        state["status"] = (
                            "completed"
                            if slurm == "COMPLETED" and all(phase["status"] in SETTLED for phase in state["phases"])
                            else "failed"
                        )
                        _save_state(run, state)
            text = render(state)
            if text != last:
                if sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                print(text, flush=True)
                last = text
            if state.get("status") in TERMINAL:
                return
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\nwatch stopped; the job continues. Resume with:\n  uv run python scripts/bench_sweep.py watch {run}")


def _fresh_run() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return RUNS / stamp


def _phase_list(names: list[str] | None) -> list[str]:
    if not names:
        return list(PHASES)
    return [name for name in PHASES if name in set(names)]


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Resumable isolated-backend sweep on at most {MAX_NODES} H100 nodes.")
    sub = parser.add_subparsers(dest="command", required=True)

    def submission_flags(command: argparse.ArgumentParser) -> None:
        # Defaults resolve in the handler so `resume` can inherit the run's own settings.
        command.add_argument("--qos")
        command.add_argument("--time", help="walltime per attempt (default 24:00:00); drained work resubmits itself")
        command.add_argument("--reps", type=_positive)
        command.add_argument("--timeout", type=_positive, help="per-case watchdog floor in seconds (default 300)")
        command.add_argument("--nodes", type=_nodes, help=f"Slurm nodes (default and maximum: {MAX_NODES})")
        command.add_argument("--curves-only", action="store_true", help="run named curve cases without coverage samples")
        command.add_argument("--phases", nargs="*", choices=PHASES, help="run only these phases (default: all)")
        command.add_argument("--ops", nargs="*", help="run only these ops (default: all registered)")
        command.add_argument("--local", action="store_true", help="run in-process via submitit's LocalExecutor (1 worker)")
        command.add_argument("--watch", action="store_true")
        command.add_argument("--dry-run", action="store_true")

    submit = sub.add_parser("submit", help="submit a fresh sweep")
    submission_flags(submit)

    resume = sub.add_parser("resume", help="resubmit an existing run; finished work is skipped")
    resume.add_argument("run", nargs="?", type=Path)
    submission_flags(resume)

    watching = sub.add_parser("watch", help="live overall and per-phase progress")
    watching.add_argument("run", nargs="?", type=Path)

    status = sub.add_parser("status", help="print the progress snapshot once")
    status.add_argument("run", nargs="?", type=Path)

    planning = sub.add_parser("plan", help="internal: build one phase's worker manifests")
    planning.add_argument("--run", required=True, type=Path)
    planning.add_argument("--phase", required=True, choices=PHASES)
    planning.add_argument("--workers", required=True, type=_positive)
    planning.add_argument("--ops", nargs="*")
    planning.add_argument("--curves-only", action="store_true")

    worker = sub.add_parser("worker", help="internal: run one assigned GPU worker")
    worker.add_argument("--run", required=True, type=Path)
    worker.add_argument("--phase", required=True, choices=PHASES)
    worker.add_argument("--rank", required=True, type=int)
    worker.add_argument("--reps", required=True, type=_positive)
    worker.add_argument("--timeout", required=True, type=_positive)
    worker.add_argument("--drain", required=True, type=Path)
    worker.add_argument("--hardware")

    args = parser.parse_args()
    if args.command in ("submit", "resume"):
        run = (args.run.resolve() if args.run else latest_run()) if args.command == "resume" else _fresh_run()
        previous = _read_json(run / CONFIG, {})
        local = bool(args.local or previous.get("local"))
        config = SweepConfig.from_mapping(
            previous,
            phases=_phase_list(args.phases or previous.get("phases")),
            ops=args.ops or previous.get("ops"),
            reps=args.reps or previous.get("reps") or 10,
            timeout=args.timeout or previous.get("timeout") or 300,
            qos=args.qos or previous.get("qos") or "staff-prod",
            walltime=args.time or previous.get("walltime") or "24:00:00",
            hardware=None if local else previous.get("hardware") or "H100",
            local=local,
            nodes=args.nodes or previous.get("nodes") or MAX_NODES,
            curves_only=bool(args.curves_only or previous.get("curves_only")),
        )
        _submit(run, config, args.dry_run, revalidate=args.command == "resume")
        if args.watch and not args.dry_run:
            watch(run)
    elif args.command == "watch":
        watch(args.run.resolve() if args.run else latest_run())
    elif args.command == "status":
        run = args.run.resolve() if args.run else latest_run()
        state = _read_json(run / STATE)
        if state is None:
            raise SystemExit(f"no sweep state at {run}")
        print(render(state))
    elif args.command == "plan":
        plan_phase(
            args.run.resolve(),
            args.phase,
            args.workers,
            set(args.ops) if args.ops else None,
            args.curves_only,
        )
    elif args.command == "worker":
        run_worker(args.run.resolve(), args.phase, args.rank, args.reps, args.timeout, args.drain, args.hardware)


if __name__ == "__main__":
    main()

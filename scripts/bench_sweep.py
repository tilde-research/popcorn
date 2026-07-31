#!/usr/bin/env python
"""Run the complete benchmark sweep in one fixed 10-node Slurm allocation.

Usage:
    uv run python scripts/bench_sweep.py submit --watch
    uv run python scripts/bench_sweep.py watch [RUN]
    uv run python scripts/bench_sweep.py resume RUN --watch

The coordinator runs references first, then reinstalls one isolated environment
per backend family. Results stay in the run cache until explicitly reviewed and
merged; this script never publishes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs" / "sweeps"
PHASES = ("reference", "popcorn", "fa3", "fla", "liger", "quack", "unsloth")
NODES = 10
GPUS_PER_NODE = 8
WORKERS = NODES * GPUS_PER_NODE
STATE = "state.json"
TERMINAL = {"completed", "failed"}


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _new_state(run: Path) -> dict[str, Any]:
    return {
        "run": str(run),
        "job_id": None,
        "status": "created",
        "started": None,
        "finished": None,
        "phases": [
            {
                "name": name,
                "status": "pending",
                "total": 0,
                "cached": 0,
                "prior_pruned": 0,
                "measured": 0,
                "pruned": 0,
                "statuses": {},
                "message": "",
            }
            for name in PHASES
        ],
    }


def _phase(state: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in state["phases"] if item["name"] == name)


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = min(width, round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(state: dict[str, Any]) -> str:
    total = sum(int(phase.get("total", 0)) for phase in state.get("phases", []))
    done = sum(
        int(phase.get("cached", 0))
        + int(phase.get("prior_pruned", 0))
        + int(phase.get("measured", 0))
        + int(phase.get("pruned", 0))
        for phase in state.get("phases", [])
    )
    slurm = f", Slurm {state['slurm_state']}" if state.get("slurm_state") else ""
    lines = [
        f"Popcorn sweep job {state.get('job_id') or '-'}: {state.get('status', 'unknown')}{slurm} "
        f"(fixed {NODES} nodes / {WORKERS} H100 workers)",
        f"Overall {_bar(done, total, 32)} {done:,}/{total:,}",
    ]
    for index, phase in enumerate(state.get("phases", []), start=1):
        total = int(phase.get("total", 0))
        done = (
            int(phase.get("cached", 0))
            + int(phase.get("prior_pruned", 0))
            + int(phase.get("measured", 0))
            + int(phase.get("pruned", 0))
        )
        status = phase.get("status", "pending")
        marker = {"completed": "✓", "failed": "✗", "running": "▶", "installing": "↓", "planning": "…"}.get(status, "·")
        progress = f"{done:,}/{total:,}" if total else status
        line = f"{marker} {index}/{len(PHASES)} {phase['name']:<10} {_bar(done, total)} {progress}"
        details = []
        if phase.get("cached"):
            details.append(f"cached {phase['cached']:,}")
        if phase.get("measured"):
            details.append(f"ran {phase['measured']:,}")
        pruned = int(phase.get("prior_pruned", 0)) + int(phase.get("pruned", 0))
        if pruned:
            details.append(f"OOM-pruned {pruned:,}")
        statuses = phase.get("statuses", {})
        if statuses:
            details.append(" ".join(f"{name}={count:,}" for name, count in sorted(statuses.items())))
        if details:
            line += " | " + ", ".join(details)
        if phase.get("message"):
            line += f" | {phase['message']}"
        lines.append(line)
    return "\n".join(lines)


def distribute(units: list[tuple[int, Any]], workers: int) -> tuple[list[list[Any]], list[int]]:
    """Longest-processing-time assignment for deterministic static balancing."""
    if workers < 1:
        raise ValueError("workers must be positive")
    bins: list[list[Any]] = [[] for _ in range(workers)]
    loads = [0] * workers
    ordered = sorted(enumerate(units), key=lambda item: (-item[1][0], item[0]))
    for _, (weight, payload) in ordered:
        target = min(range(workers), key=lambda index: (loads[index], index))
        bins[target].append(payload)
        loads[target] += weight
    return bins, loads


def affinity_units(groups: list[dict[str, Any]], workers: int) -> list[tuple[int, list[dict[str, Any]]]]:
    """Split only heavy op/implementation bundles, keeping compile affinity."""
    total = sum(len(group["cases"]) for group in groups)
    target = max(1, math.ceil(total / workers))
    affinities: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for group in groups:
        affinities.setdefault((group["op"], group["impl"]), []).append(group)

    units = []
    for affinity in sorted(affinities):
        members = affinities[affinity]
        weight = sum(len(group["cases"]) for group in members)
        chunks = min(len(members), max(1, math.ceil(weight / target)))
        bins, loads = distribute([(len(group["cases"]), group) for group in members], chunks)
        units.extend((load, chunk) for chunk, load in zip(bins, loads) if chunk)
    return units


def reference_modes(op: Any) -> tuple[bool, ...]:
    """Gradient modes needed to compare every registered implementation."""
    modes = {not impl.forward_only for impl in op._impls if impl.name != "torch"}
    return tuple(sorted(modes or {True}, reverse=True))


def expected_totals() -> dict[str, int]:
    """Registered pairwise work, before cache and OOM pruning."""
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.bench.grid import covering_cases

    totals = {name: 0 for name in PHASES}
    families = set()
    for op in KERNELS.values():
        count = len(covering_cases(op))
        totals["reference"] += count * len(reference_modes(op))
        for impl in op._impls:
            if impl.name == "torch":
                continue
            family = impl.name.split(":", 1)[0]
            families.add(family)
            if family in totals:
                totals[family] += count
    missing = families - set(PHASES)
    if missing:
        raise RuntimeError(f"sweep phase order is missing registered backend family/families: {', '.join(sorted(missing))}")
    return totals


def _merge_shards(run: Path, phase_name: str) -> int:
    from popcorn.bench.store import read_file, user_reports, write

    shards = sorted((run / phase_name / "shards").glob("*.jsonl"))
    records = [record for shard in shards for record in read_file(shard)]
    if records:
        write(records, user_reports())
        for shard in shards:
            shard.unlink()
    return len(records)


def _phase_work(phase_name: str) -> list[tuple[Any, str, Any, bool]]:
    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.bench.grid import covering_cases
    from popcorn.core.sources import available, unavailable_reason

    installed_families = {
        impl.name.split(":", 1)[0]
        for op in KERNELS.values()
        for impl in op._impls
        if impl.name != "torch" and available(impl.name)
    }
    expected_external = set() if phase_name in ("reference", "popcorn") else {phase_name}
    unexpected = installed_families - {"popcorn"} - expected_external
    if unexpected:
        raise SystemExit(f"backend environment is not isolated for {phase_name!r}; also found {', '.join(sorted(unexpected))}")

    work = []
    registered = 0
    unavailable = set()
    for op in (KERNELS[name] for name in sorted(KERNELS)):
        cases = covering_cases(op)
        if phase_name == "reference":
            for grad in reference_modes(op):
                work.extend((op, "torch", case, grad) for case in cases)
            continue
        implementations = [impl for impl in op._impls if impl.name != "torch" and impl.name.split(":", 1)[0] == phase_name]
        registered += len(implementations)
        for impl in implementations:
            if not available(impl.name):
                unavailable.add(unavailable_reason(impl.name) or impl.name)
                continue
            work.extend((op, impl.name, case, not impl.forward_only) for case in cases)
    if phase_name != "reference" and not registered:
        raise SystemExit(f"no implementations are registered for backend family {phase_name!r}")
    if phase_name != "reference" and not work:
        details = "; ".join(sorted(unavailable))
        raise SystemExit(f"backend family {phase_name!r} is unavailable after installation: {details}")
    return work


def _stratum(case: Any) -> tuple[Any, ...]:
    return (
        str(case.dtype),
        json.dumps(dict(case.args), sort_keys=True, separators=(",", ":")),
        tuple(sorted(case.present)),
        len(case.batch),
    )


def plan_phase(run: Path, phase_name: str, workers: int) -> dict[str, Any]:
    import torch

    from popcorn.bench.__main__ import _fill_pending, _record_index, _usable_records
    from popcorn.bench.model import Case
    from popcorn.bench.plan import OOMFrontier
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
    work = _phase_work(phase_name)
    triples = list({(op.name, impl): (op, impl, case) for op, impl, case, _ in work}.values())
    device = device_name("cuda")
    store = Store()
    current = _usable_records(store, triples, device)
    cached = _record_index(current)
    frontiers: dict[tuple[str, str, bool], OOMFrontier] = {}
    for record in current:
        if record.result.status == "oom":
            key = (record.op, record.impl, record.result.grad)
            frontiers.setdefault(key, OOMFrontier()).add(Case.from_config(record.config))

    reference_frontiers: dict[tuple[str, bool], OOMFrontier] = {}
    if phase_name != "reference":
        references = list({op.name: (op, "torch", case) for op, _, case, _ in work}.values())
        for record in _usable_records(store, references, device):
            if record.result.status == "oom":
                key = (record.op, record.result.grad)
                reference_frontiers.setdefault(key, OOMFrontier()).add(Case.from_config(record.config))

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    exact_cached = 0
    prior_pruned = 0
    for op, impl, case, grad in work:
        record = cached.get((op.name, impl, case.case_id, grad))
        if not _fill_pending(record):
            exact_cached += 1
            continue
        reference_frontier = reference_frontiers.get((op.name, grad))
        if reference_frontier is not None and reference_frontier.blocker(case) is not None:
            prior_pruned += 1
            continue
        frontier = frontiers.setdefault((op.name, impl, grad), OOMFrontier())
        if frontier.blocker(case) is not None:
            prior_pruned += 1
            continue
        key = (op.name, impl, grad, *_stratum(case))
        group = groups.setdefault(key, {"op": op.name, "impl": impl, "grad": grad, "cases": []})
        group["cases"].append(case.config())

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


def _worker_progress(
    path: Path,
    *,
    status: str,
    measured: int,
    pruned: int,
    statuses: Counter[str],
    error: str = "",
) -> None:
    _write_json(
        path,
        {
            "status": status,
            "measured": measured,
            "pruned": pruned,
            "statuses": dict(statuses),
            "error": error,
        },
    )


def run_worker(run: Path, phase_name: str, rank: int, reps: int, timeout: int) -> None:
    job = os.getenv("SLURM_JOB_ID", "local")
    caches = {
        "TRITON_CACHE_DIR": f"/tmp/triton_{job}_{rank}",
        "TORCH_EXTENSIONS_DIR": f"/tmp/torch_extensions_{job}_{rank}",
        "TORCHINDUCTOR_CACHE_DIR": f"/tmp/torchinductor_{job}_{rank}",
    }
    for variable, cache_name in caches.items():
        os.environ[variable] = cache_name
        cache = Path(cache_name)
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True)

    import popcorn.kernels  # noqa: F401
    from popcorn import KERNELS
    from popcorn.bench.__main__ import _isolated, _target_device
    from popcorn.bench.model import Case
    from popcorn.bench.plan import OOMFrontier
    from popcorn.bench.store import read_file

    phase_dir = run / phase_name
    manifest = phase_dir / "manifests" / f"{rank}.jsonl"
    shard = phase_dir / "shards" / f"{rank}.jsonl"
    progress = phase_dir / "progress" / f"{rank}.json"
    shard.parent.mkdir(parents=True, exist_ok=True)
    groups = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    existing = read_file(shard)
    done = {(record.op, record.impl, record.case_id, record.result.grad) for record in existing}
    statuses = Counter(record.result.status for record in existing)
    measured = len(existing)
    pruned = 0
    frontiers: dict[tuple[str, str, bool], OOMFrontier] = {}
    for record in existing:
        if record.result.status == "oom":
            key = (record.op, record.impl, record.result.grad)
            frontiers.setdefault(key, OOMFrontier()).add(Case.from_config(record.config))

    device = _target_device(argparse.Namespace(device="cuda", hardware="H100"))
    last_update = 0.0
    _worker_progress(progress, status="running", measured=measured, pruned=pruned, statuses=statuses)
    try:
        for group in groups:
            op = KERNELS[group["op"]]
            impl = group["impl"]
            grad = bool(group["grad"])
            frontier = frontiers.setdefault((op.name, impl, grad), OOMFrontier())
            for config in group["cases"]:
                case = Case.from_config(config)
                key = (op.name, impl, case.case_id, grad)
                if key in done:
                    continue
                if frontier.blocker(case) is not None:
                    pruned += 1
                else:
                    record = _isolated(op, impl, case, device, reps, timeout, grad=grad)
                    with shard.open("a") as sink:
                        sink.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
                    done.add(key)
                    measured += 1
                    statuses[record.result.status] += 1
                    if record.result.status == "oom":
                        frontier.add(case)
                now = time.monotonic()
                if now - last_update >= 1:
                    _worker_progress(
                        progress,
                        status="running",
                        measured=measured,
                        pruned=pruned,
                        statuses=statuses,
                    )
                    last_update = now
        _worker_progress(progress, status="completed", measured=measured, pruned=pruned, statuses=statuses)
    except Exception as error:
        _worker_progress(
            progress,
            status="failed",
            measured=measured,
            pruned=pruned,
            statuses=statuses,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _aggregate_progress(phase_dir: Path) -> dict[str, Any]:
    measured = 0
    pruned = 0
    statuses: Counter[str] = Counter()
    failed = []
    for path in sorted((phase_dir / "progress").glob("*.json")):
        row = _read_json(path, {})
        measured += int(row.get("measured", 0))
        pruned += int(row.get("pruned", 0))
        statuses.update(row.get("statuses", {}))
        if row.get("status") == "failed":
            failed.append(f"{path.stem}: {row.get('error', 'worker failed')}")
    return {
        "measured": measured,
        "pruned": pruned,
        "statuses": dict(statuses),
        "failed": failed,
    }


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
    _run_logged([uv, "venv", "--clear", str(destination)], log, env)
    python = destination / "bin" / "python"
    target = "." if phase_name in ("reference", "popcorn") else f".[{phase_name}]"
    if phase_name == "fa3":
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
        [uv, "pip", "install", "--python", str(python), "--torch-backend=auto", "-e", target],
        log,
        env,
    )
    return python


def _save_state(run: Path, state: dict[str, Any]) -> None:
    _write_json(run / STATE, state)


def coordinate(run: Path, reps: int, timeout: int) -> None:
    run.mkdir(parents=True, exist_ok=True)
    state = _read_json(run / STATE, _new_state(run))
    state["status"] = "running"
    state["started"] = state.get("started") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    state["finished"] = None
    _save_state(run, state)
    environment = run / "environment"
    failures = []
    cache = run / "cache"
    os.environ["POPCORN_CACHE_DIR"] = str(cache)
    os.environ["POPCORN_SKIP_REPORTS"] = "1"

    for phase_name in PHASES:
        phase = _phase(state, phase_name)
        if phase.get("status") == "completed":
            continue
        phase_dir = run / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        try:
            phase.update(status="installing", message="rebuilding isolated environment")
            _save_state(run, state)
            python = install_environment(phase_name, environment, phase_dir / "install.log")

            phase.update(status="planning", message="building balanced OOM-stratum shards")
            _save_state(run, state)
            _run_logged(
                [
                    str(python),
                    str(Path(__file__).resolve()),
                    "plan",
                    "--run",
                    str(run),
                    "--phase",
                    phase_name,
                    "--workers",
                    str(WORKERS),
                ],
                phase_dir / "plan.log",
                os.environ.copy(),
            )
            summary = _read_json(phase_dir / "plan.json", {})
            phase.update(
                status="running",
                total=summary.get("total", 0),
                cached=summary.get("cached", 0),
                prior_pruned=summary.get("prior_pruned", 0),
                measured=0,
                pruned=0,
                statuses={},
                message=f"{summary.get('groups', 0):,} strata; max shard {summary.get('max_worker_cases', 0):,}",
            )
            _save_state(run, state)
            if not summary.get("scheduled", 0):
                phase.update(status="completed", message="all rows already cached or OOM-pruned")
                continue

            logs = phase_dir / "logs"
            logs.mkdir(exist_ok=True)
            worker_command = (
                f"exec {shlex.quote(str(python))} {shlex.quote(str(Path(__file__).resolve()))} worker "
                f"--run {shlex.quote(str(run))} --phase {shlex.quote(phase_name)} "
                f'--rank "$SLURM_PROCID" --reps {reps} --timeout {timeout}'
            )
            command = [
                "srun",
                f"--nodes={NODES}",
                f"--ntasks={WORKERS}",
                f"--ntasks-per-node={GPUS_PER_NODE}",
                "--gpus-per-task=1",
                "--cpus-per-task=8",
                "--kill-on-bad-exit=1",
                f"--output={logs}/%t.log",
                f"--error={logs}/%t.log",
                "bash",
                "-c",
                worker_command,
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy())
            last_print = 0.0
            while process.poll() is None:
                aggregate = _aggregate_progress(phase_dir)
                phase.update(
                    measured=aggregate["measured"],
                    pruned=aggregate["pruned"],
                    statuses=aggregate["statuses"],
                )
                _save_state(run, state)
                now = time.monotonic()
                if now - last_print >= 30:
                    print(render(state), flush=True)
                    last_print = now
                time.sleep(2)
            aggregate = _aggregate_progress(phase_dir)
            phase.update(
                measured=aggregate["measured"],
                pruned=aggregate["pruned"],
                statuses=aggregate["statuses"],
            )
            merged = _merge_shards(run, phase_name)
            if process.returncode:
                details = "; ".join(aggregate["failed"][:3]) or f"srun exited {process.returncode}"
                raise RuntimeError(details)
            phase.update(status="completed", message=f"{merged:,} observed rows saved")
        except Exception as error:
            phase.update(status="failed", message=f"{type(error).__name__}: {error}")
            failures.append(phase_name)
        finally:
            shutil.rmtree(environment, ignore_errors=True)
            _save_state(run, state)
            print(render(state), flush=True)

    state["status"] = "failed" if failures else "completed"
    state["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_state(run, state)
    print(render(state), flush=True)
    print(f"run cache: {cache}", flush=True)
    if failures:
        raise SystemExit(f"failed phases: {', '.join(failures)}")


def _job_script(run: Path, qos: str, walltime: str, reps: int, timeout: int) -> str:
    # Keep the virtualenv entry point. Resolving its symlink would drop the
    # environment's site-packages and run the coordinator with bare uv Python.
    python = Path(sys.executable).absolute()
    script = Path(__file__).resolve()
    return f"""\
#!/usr/bin/env bash
#SBATCH --job-name=popcorn-sweep
#SBATCH --qos={qos}
#SBATCH --nodes={NODES}
#SBATCH --ntasks={WORKERS}
#SBATCH --ntasks-per-node={GPUS_PER_NODE}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --time={walltime}
#SBATCH --output={run}/coordinator.log

set -euo pipefail
cd {shlex.quote(str(ROOT))}
export PYTHONUNBUFFERED=1
exec {shlex.quote(str(python))} {shlex.quote(str(script))} coordinate \
    --run {shlex.quote(str(run))} --reps {reps} --timeout {timeout}
"""


def _submit(run: Path, qos: str, walltime: str, reps: int, timeout: int, dry_run: bool) -> None:
    run.mkdir(parents=True, exist_ok=True)
    state = _read_json(run / STATE, _new_state(run))
    state.pop("slurm_state", None)
    totals = expected_totals()
    for phase in state["phases"]:
        phase["total"] = totals[phase["name"]]
    job = run / "job.slurm"
    job.write_text(_job_script(run, qos, walltime, reps, timeout))
    _save_state(run, state)
    if dry_run:
        print(job)
        return
    queued = subprocess.run(
        ["squeue", "--noheader", "--user", os.getenv("USER", ""), "--name", "popcorn-sweep", "--format=%i"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if queued:
        raise SystemExit(
            f"refusing to submit another sweep while job(s) {', '.join(queued)} are active; "
            f"this guard keeps usage at {NODES} nodes"
        )
    submitted = subprocess.run(["sbatch", "--parsable", str(job)], capture_output=True, text=True, check=True)
    job_id = submitted.stdout.strip().split(";", 1)[0]
    state["job_id"] = job_id
    state["status"] = "queued"
    _save_state(run, state)
    print(f"submitted job {job_id}: exactly {NODES} nodes / {WORKERS} H100 workers")
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
                raise SystemExit(f"no sweep state at {run}")
            if state.get("status") not in TERMINAL and (job_id := state.get("job_id")):
                slurm = _slurm_state(job_id)
                if slurm and slurm not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
                    state["slurm_state"] = slurm
                    state["status"] = (
                        "completed"
                        if slurm == "COMPLETED" and all(phase["status"] == "completed" for phase in state["phases"])
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
        print(f"\nwatch stopped; Slurm job continues. Resume with:\n  uv run python scripts/bench_sweep.py watch {run}")


def _fresh_run() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return RUNS / stamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential isolated-backend sweep on exactly ten H100 nodes.")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="submit a fresh fixed-size sweep")
    submit.add_argument("--qos", default="staff-prod")
    submit.add_argument("--time", default="24:00:00")
    submit.add_argument("--reps", type=_positive, default=10)
    submit.add_argument("--timeout", type=_positive, default=300)
    submit.add_argument("--watch", action="store_true")
    submit.add_argument("--dry-run", action="store_true")

    resume = sub.add_parser("resume", help="resume incomplete phases in an existing run")
    resume.add_argument("run", type=Path)
    resume.add_argument("--qos", default="staff-prod")
    resume.add_argument("--time", default="24:00:00")
    resume.add_argument("--reps", type=_positive, default=10)
    resume.add_argument("--timeout", type=_positive, default=300)
    resume.add_argument("--watch", action="store_true")
    resume.add_argument("--dry-run", action="store_true")

    watching = sub.add_parser("watch", help="show the live overall and per-backend progress bars")
    watching.add_argument("run", nargs="?", type=Path)

    coordinator = sub.add_parser("coordinate", help="run phase coordination inside an allocation")
    coordinator.add_argument("--run", required=True, type=Path)
    coordinator.add_argument("--reps", required=True, type=_positive)
    coordinator.add_argument("--timeout", required=True, type=_positive)

    planning = sub.add_parser("plan", help="build one phase's worker manifests")
    planning.add_argument("--run", required=True, type=Path)
    planning.add_argument("--phase", required=True, choices=PHASES)
    planning.add_argument("--workers", required=True, type=_positive)

    worker = sub.add_parser("worker", help="run one assigned GPU worker")
    worker.add_argument("--run", required=True, type=Path)
    worker.add_argument("--phase", required=True, choices=PHASES)
    worker.add_argument("--rank", required=True, type=int)
    worker.add_argument("--reps", required=True, type=_positive)
    worker.add_argument("--timeout", required=True, type=_positive)

    args = parser.parse_args()
    if args.command == "submit":
        run = _fresh_run()
        _submit(run, args.qos, args.time, args.reps, args.timeout, args.dry_run)
        if args.watch and not args.dry_run:
            watch(run)
    elif args.command == "resume":
        run = args.run.resolve()
        _submit(run, args.qos, args.time, args.reps, args.timeout, args.dry_run)
        if args.watch and not args.dry_run:
            watch(run)
    elif args.command == "watch":
        watch(args.run.resolve() if args.run else latest_run())
    elif args.command == "coordinate":
        coordinate(args.run.resolve(), args.reps, args.timeout)
    elif args.command == "plan":
        plan_phase(args.run.resolve(), args.phase, args.workers)
    elif args.command == "worker":
        run_worker(args.run.resolve(), args.phase, args.rank, args.reps, args.timeout)


if __name__ == "__main__":
    main()

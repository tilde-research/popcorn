---
title: Benchmarking
description: Record timings and route dispatch on data
---

`benchmark` validates eligible implementations, then records latency and peak memory relative
to the reference. Dispatch uses the resulting measurements.

Use `report` to print one result per backend:

```python
from popcorn.bench import report

for record in rms_norm.benchmark(x, weight):
    report(record)
```

```
fla      torch      0.1 ms -> fla    0.0 ms   (8.4x)
         peak         0.33 GB ->      0.27 GB
         bwd           0.4 ms ->    0.1 ms   (4.0x)
         err 1.56e-02 on scale 5.2
liger    torch      0.1 ms -> liger    0.0 ms   (8.2x)
         peak         0.33 GB ->      0.27 GB
         bwd           0.4 ms ->    0.1 ms   (4.7x)
         err 1.56e-02 on scale 5.2
```

`report` prints and returns the formatted text. Raw status, gauges, and measurements are
available through `record.result`.

Use `compare` for an isolated correctness and performance comparison:

```python
from popcorn.bench import compare, report

report(compare(fast, slow, {"x": x, "weight": weight}), "rms_norm", mine="popcorn", reference="torch")
```

To benchmark missing cases during dispatch, pass `bench=True` or set `POPCORN_BENCH=1`:

```python
output = rms_norm(x, weight, bench=True)
```

```bash
POPCORN_BENCH=1 python train.py
```

`benchmark` and runtime benchmarking write only to the user cache. They never modify bundled
reports. Runtime benchmarking is synchronous and may compile every eligible implementation.

## Fill dimension coverage

Measure systematic coverage on the current device with:

```bash
popcorn bench fill
```

`fill` runs the shared case plan. Pairwise coverage exercises every value and two-axis
interaction without evaluating the full Cartesian product. Named production and long-context
curves vary one axis at a time while their other dimensions stay fixed. Full-model temporal
curves stop at 32K; reduced-shape frontier curves overlap from 8K and continue through 1M.

Smaller shapes run first. Cases that cannot fit the harness's native and fp64 correctness copies
with workspace headroom are omitted. A reference,
implementation OOM, or timeout skips only later cases that are at least as large in every
dimension within the same gradient mode and case stratum. Skipped shapes are not stored as
inferred results. Use `--limit N` for a deterministic random sample or `--force` to remeasure
cached cases.

## Follow a local fill in the kernel explorer

Project the rows from one `fill` process into the website while they are measured:

```bash
popcorn bench fill rms_norm --live
```

Open `rms_norm` in the [kernel explorer](https://tilde-research.github.io/popcorn/kernels),
select the **Local** tab, and press **Check**. The browser may ask for permission to connect
to loopback. The tab verifies that port 8765 is Popcorn, then shows this session's progress
and recent results. It does not replace or modify the published plots.

Use the same custom port on both sides when 8765 is unavailable:

```bash
popcorn bench fill rms_norm --live 9000
```

The website remembers the last port entered. The server binds only to `127.0.0.1`, accepts
read-only connections from the Popcorn site and local site development, and exits with the
`fill` command. Only rows produced by that command are streamed; the existing user cache is
not replayed. If every planned case is already cached, add `--force` to remeasure it.

## Full multi-node sweep

Maintainers can run the complete H100 sweep with:

```bash
uv run popcorn sweep submit --nodes 6 --watch
```

`--nodes` defaults to 6 and cannot exceed 10. `submit` queues one exclusive, singleton submitit
job named `popcorn-sweep`, with eight GPU workers per node, so sweeps cannot stack allocations.
The coordinator measures the Torch references, then runs `popcorn`, `cudnn`, `fa3`, `fla`,
`liger`, `quack`, `transformer_engine`, and `unsloth`, each in a freshly installed environment.
`cudnn` is skipped when the frontend cannot load one unambiguous CUDA runtime; `fa3` is skipped
when `nvcc` cannot compile against torch's CUDA build.
Use `--curves-only` for a release fill that skips sparse coverage samples already represented by
the pinned database.

On a cluster with Pyxis, a CUDA 13 development image with Git resolves both host problems:

```bash
uv run popcorn sweep resume logs/sweeps/<run> \
  --container-image 'nvcr.io#nvidia/pytorch:25.09-py3' --watch
```

Every Slurm task runs in the image with the user's home mounted and the repository as its working
directory. The configured image is inherited by later resumes and invalidates reusable phase
environments when changed. Container execution is not supported with `--local`.
The H100 sweep limits FA3's source build to the FP16/BF16 dense and varlen features its adapters
expose, omitting SM80, FP8, paged/append-KV, local-window, softcap, and unequal-head-dimension variants.

On one GPU without Slurm, the same orchestrator can collect the NVIDIA adapter curves in isolated
environments:

```bash
uv run popcorn sweep submit --local --phases reference cudnn transformer_engine \
  --ops attn softmax --curves-only --reps 3 --watch
```

It does not publish. The resulting run and mergeable report cache stay under `logs/sweeps/`.

The same explicit case plan drives the CLI, sweep, and site. Coverage samples remain sparse;
production and long-context series provide plot-ready one-axis curves. Cases whose correctness
copies cannot leave device workspace are dropped at plan time, and each stratum runs cheapest first so budget
frontiers meet the OOM or timeout boundary before paying for anything beyond it.
Static adapter gates also remove impossible dtype and scalar cases before either the backend or
its reference work is scheduled.
The sweep records fp16 cases forward-only when backward would reduce 32K or more output
positions into an input. This fanout comes from the input and output shape annotations, with
the temporal-axis check retained when an output shape cannot be resolved.

Workers run cases in process. A crash costs one case: a sentinel written before each case
converts it to a `crash` row when the worker restarts. A hang costs one watchdog budget and
records `timeout`. References cap that budget at 60 seconds; backend budgets retain the
configured compile floor and may grow with input size. Shards fold into the run cache
incrementally, and near the walltime the
job drains after the cases in flight, merges, and resubmits itself. Planning is idempotent,
so every attempt skips finished work. Reference timings are reused only for exact device,
Torch version, fingerprint, case, and gradient-mode matches; correctness still executes the
reference for every backend. Stopping the watcher does not cancel the job:

```bash
uv run popcorn sweep watch
uv run popcorn sweep resume [logs/sweeps/<run>] --watch
```

The sweep does not publish results. Review the run cache, then merge its JSONL files into the
Parquet store; `--scrub-incomplete` removes rows the harness scheduled but never measured:

```bash
POPCORN_CACHE_DIR=logs/sweeps/<run>/cache uv run popcorn bench view --user
uv run popcorn bench merge logs/sweeps/<run>/cache/v*/reports/*.jsonl
uv run python scripts/check_records.py --release \
  --include logs/sweeps/<run>/cache/v3/reports
```

## Inspect results

API-triggered benchmarks log one line per op, backend, and case on `popcorn.bench`:

```python
import logging

logging.basicConfig()
logging.getLogger("popcorn.bench").setLevel(logging.INFO)
```

Render recorded data with `popcorn bench view --user`. This combines the local
cache with bundled reports. The same data powers the
[kernel explorer](https://tilde-research.github.io/popcorn/kernels), where named curves and
sparse coverage samples are shown separately.

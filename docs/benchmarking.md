---
title: Benchmarking
description: Record timings and route dispatch on data
---

`benchmark` validates before timing and records implementation and reference latency and
peak memory. Future calls use these measurements for dispatch.

Each returned record carries the backend it came from, so `report` labels the block for you:

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

`report` prints and returns the text. Reach for `record.result` when you want the raw
status, gauges, or `bench` dictionary instead.

To compare two callables end to end (correctness + timing + memory) and print a one-block summary:

```python
from popcorn.bench import compare, report

report(compare(fast, slow, {"x": x, "weight": weight}), "rms_norm", mine="popcorn", reference="torch")
```

To validate, benchmark, and select the exact fastest implementation on first use, pass
`bench=True` to the call or enable it for the whole application:

```python
output = rms_norm(x, weight, bench=True)
```

```bash
POPCORN_BENCH=1 python train.py
```

Both APIs write to the user cache and never modify the package's bundled reports. First use
is synchronous and may compile every eligible implementation.

## Fill dimension coverage

To expand the cache systematically on the current device:

```bash
python -m popcorn.bench fill
```

`fill` builds a pairwise covering array over every dimension pool in `core/dims.py`, dtypes,
scalar test arguments, and optional inputs. This covers every value and every two-axis interaction
without attempting the full Cartesian product. It runs smaller shapes first and keeps an OOM
frontier for each implementation and case stratum. A later shape is skipped only when it is at
least as large in every dimension as an observed OOM. A mixed tradeoff, such as lowering `hidden`
while raising `seq`, still runs.

Only observed results are stored. Shapes skipped by the OOM frontier do not become inferred report
rows. Use `--limit N` when you specifically want the older deterministic random sample, or
`--force` to ignore cached exact cases and rebuild the frontier from this run.

## Full ten-node sweep

Maintainers can run the complete H100 database sweep with:

```bash
uv run python scripts/bench_sweep.py submit --watch
```

This submits one exclusive allocation with exactly 10 nodes and 80 workers. A submission guard
refuses to start while another `popcorn-sweep` job is queued or running. Inside that allocation,
the coordinator benchmarks torch references first, then runs `popcorn`, `fa3`, `fla`, `liger`,
`quack`, and `unsloth` in order. It deletes and reinstalls the environment once before every
phase. Backends are never installed together and the environment is not rebuilt per case.
Later phases reuse reference timing only for an exact device, Torch version, reference fingerprint,
case, and gradient-mode match. Every backend still executes the reference during correctness
grading; cached timing never substitutes for validation.

Work is balanced by OOM-comparable strata. Small op/implementation bundles stay together to reuse
their compile cache; only bundles large enough to cause a tail are split across workers. The live
view reports installation and planning, cached and newly measured rows, status counts, OOM-pruned
cases, each backend phase, and overall completion. Stopping the watcher does not cancel Slurm:

```bash
uv run python scripts/bench_sweep.py watch
uv run python scripts/bench_sweep.py resume logs/sweeps/<run> --watch
```

Each phase saves partial results before the next environment is installed, so `resume` skips
completed phases and cached cases. Nothing is published automatically. Review the run cache first,
then fold its JSONL files into the bundled Parquet store:

```bash
POPCORN_CACHE_DIR=logs/sweeps/<run>/cache uv run python -m popcorn.bench view --user
uv run python -m popcorn.bench merge logs/sweeps/<run>/cache/v*/reports/*.jsonl
```

## Watching it happen

API-triggered benchmarks log on `popcorn.bench`: one line per op, backend, and case,
followed by where the rows were written.

```python
import logging

logging.basicConfig()
logging.getLogger("popcorn.bench").setLevel(logging.INFO)
```

To browse recorded rows instead, render the report database as HTML:
`python -m popcorn.bench view --user` folds your local cache into the bundled reports. The
same data powers the [kernel explorer](https://tilde-research.github.io/popcorn/kernels).

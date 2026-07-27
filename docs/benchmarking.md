---
title: Benchmarking
description: Record timings and route dispatch on data
---

`benchmark` validates before timing and records implementation and reference latency and
peak memory. Future calls use these measurements for dispatch.

```python
for result in rms_norm.benchmark(x, weight):
    print(result.status, result.bench)
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

## Watching it happen

Every recorded result is also logged on the `popcorn.bench` logger — one line per op,
backend, and case with status, forward/backward milliseconds, and the failure reason if any
— followed by where the rows were written:

```python
import logging

logging.basicConfig()
logging.getLogger("popcorn.bench").setLevel(logging.INFO)
```

To browse recorded rows instead, render the report database as HTML:
`python -m popcorn.bench view --user` folds your local cache into the bundled reports. The
same data powers the [kernel explorer](https://tilde-research.github.io/popcorn/kernels/).

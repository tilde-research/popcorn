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

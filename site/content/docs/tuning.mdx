---
title: Tuning
description: Select one implementation for a region of shapes
---

Ordinary dispatch tunes each call automatically from the nearest compatible benchmark. To
select one implementation for a broader region, query the recorded data explicitly:

```python
from popcorn import Range

best = rms_norm.tuner.best(
    device="cuda",
    dtype="bfloat16",
    grad=False,
    normalized_shape=Range(1024, 8192),
)
output = best(x, weight)
```

`best` returns the implementation with the lowest median recorded latency across the region.
It only reads existing benchmark data; it never runs benchmarks itself.

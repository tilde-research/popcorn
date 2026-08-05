---
title: Tuning
description: Select one implementation for a region of shapes
---

Normal dispatch selects an implementation from the nearest compatible benchmark. Use the
tuner to select one implementation for a broader region:

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

`best` returns the implementation with the lowest median recorded latency in the region. It
reads existing data and does not run benchmarks.

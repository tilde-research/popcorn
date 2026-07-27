---
title: Dispatching
description: How each call resolves to one implementation
---

Each call resolves to one implementation in three steps:

1. **Eligibility.** An implementation is a candidate only when its package is installed at a
   supported version, it has a backward pass if the call needs gradients, and the inputs
   satisfy its declared shape, dtype, and value constraints.
2. **Correctness.** Implementations with a recorded failure for this exact case are excluded.
   Selecting one that has no recorded pass emits `UnvalidatedWarning` once (see
   [Validation](./validation.md)).
3. **Speed.** Among the remaining candidates, the fastest wins, judged by the nearest recorded
   benchmark on the same device, dtype, and gradient mode. The reference competes on equal
   terms: when it measures fastest, it is selected. Without benchmark data, candidates are
   tried in registration order, reference last.

The decision is memoized per call configuration (shapes, dtypes, device, gradient mode, and
scalar arguments), so dispatch adds negligible overhead in steady state. New validation
records, benchmarks, or registrations invalidate the memo.

## Overriding

Nothing is ever pinned on the kernel; you override selection per call or per scope:

```python
output = rms_norm(x, weight, backend="fla")   # force this call
output = rms_norm["fla"](x, weight)           # equivalent

with rms_norm["fla"] as rms_norm:             # every call in the block
    ...                                       # nestable and context-local

rms_norm.available_backends()                 # ('fla', 'liger', 'quack', 'torch')
print(rms_norm)                               # signature and per-implementation constraints
```

A forced implementation never falls back: it raises `DispatchError` when it cannot serve the
call, whether because its package is missing (the error names the install extra), the inputs
are unsupported, or the case has a recorded failure (the error carries the recorded reason).

## Validation flags

Two flags harden dispatch, per call as keyword arguments or process-wide as environment
variables:

| Per call | Process-wide | Effect |
| --- | --- | --- |
| `validate=True` | `POPCORN_VALIDATE=1` | Validate unrecorded cases synchronously before dispatch; only implementations with a recorded pass are selected. |
| `bench=True` | `POPCORN_BENCH=1` | Additionally record timings and select the exact fastest implementation. Takes precedence over `validate`. |

The keyword flags only enable: a call cannot opt out of a mode set in the environment. Both
modes run synchronously inside the call, so a first encounter with a new configuration may
compile and check every candidate before returning. Results land in the user cache,
`${XDG_CACHE_HOME:-~/.cache}/popcorn` by default, overridden with `POPCORN_CACHE_DIR`.

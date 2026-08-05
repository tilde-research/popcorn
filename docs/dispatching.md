---
title: Dispatching
description: How each call resolves to one implementation
---

Dispatch resolves each call in three steps:

1. **Eligibility.** The package version, gradient support, input constraints, and recorded
   shape validity must allow the call.
2. **Correctness.** A recorded failure excludes the implementation. Selecting an
   implementation without a recorded pass emits `UnvalidatedWarning` once. See
   [Validation](./validation.md).
3. **Speed.** The nearest compatible benchmark determines the fastest candidate. Measurements
   match device, dtype, and gradient mode. The reference is selected when it is fastest.
   Without benchmark data, registration order applies, with the reference last.

Decisions are cached by shape, dtype, device, gradient mode, and scalar arguments. New
validation records, benchmarks, or registrations invalidate the cache.

## Overrides

Select a backend per call or within a context:

```python
output = rms_norm(x, weight, backend="fla")   # force this call
output = rms_norm["fla"](x, weight)           # equivalent

with rms_norm["fla"] as rms_norm:             # every call in the block
    ...                                       # nestable and context-local

rms_norm.available_backends()                 # ('fla', 'liger', 'quack', 'torch')
print(rms_norm)                               # signature and per-implementation constraints
```

An explicit backend never falls back. It raises `DispatchError` if the package is missing,
the inputs are unsupported, or the case has a recorded failure.
The public API calls this choice a backend; report rows store the same name in the `impl` field.

## Runtime validation

Use keyword arguments or environment variables to validate during dispatch:

| Per call | Process-wide | Effect |
| --- | --- | --- |
| `validate=True` | `POPCORN_VALIDATE=1` | Validate unrecorded cases before selection. |
| `bench=True` | `POPCORN_BENCH=1` | Validate, time, and select the fastest implementation. This takes precedence over `validate`. |

Keyword arguments cannot disable a mode set by the environment. Both modes are synchronous,
so the first call for a configuration may compile and check every candidate. Results are
stored under `${XDG_CACHE_HOME:-~/.cache}/popcorn` by default. Set `POPCORN_CACHE_DIR` to
override the location.

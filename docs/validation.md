---
title: Validation
description: Compare implementations against the reference
---

Validation compares eligible implementations with a higher-precision reference. It checks
forward outputs and, when required, gradients.

```python
records = rms_norm.validate(x, weight)
assert all(record.result.status == "pass" for record in records)
```

Each record identifies its implementation. Use `report` to print the results:

```python
from popcorn.bench import report

for record in rms_norm.validate(x, weight):
    report(record)
```

Results are keyed by case, device, PyTorch version, backend version, and gradient mode. A
kernel-code fingerprint invalidates records after functional changes; comments, formatting,
and the running Python version do not affect it. Automatic dispatch excludes known failures. An implementation without a
validation record remains available but emits `UnvalidatedWarning`.

`validate` always runs the comparison. To validate only missing cases during dispatch, pass
`validate=True` or set `POPCORN_VALIDATE=1`:

```python
output = rms_norm(x, weight, validate=True)
```

```bash
POPCORN_VALIDATE=1 python train.py
```

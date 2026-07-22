---
title: Validation
description: Compare implementations against the reference
---

Validation compares eligible implementations with higher-precision reference results. It
checks forward outputs and, when inputs require gradients, backward results.

```python
results = rms_norm.validate(x, weight)
assert all(result.status == "pass" for result in results)
```

Results are stored per case, device, PyTorch version, backend version, and gradient mode,
and are stamped with a fingerprint of the kernel code: results recorded for a since-edited
reference or implementation are ignored (comments and formatting don't count). A known
failure is removed from automatic dispatch; an unvalidated implementation remains usable but
emits `UnvalidatedWarning`.

`validate` always re-runs the comparison. To instead validate on first use, only where a
conclusive record is missing, pass `validate=True` to the call or enable it for the whole
application:

```python
output = rms_norm(x, weight, validate=True)
```

```bash
POPCORN_VALIDATE=1 python train.py
```

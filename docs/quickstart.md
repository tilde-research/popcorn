---
title: Quick start
description: Install Popcorn and dispatch your first kernel
---

## Install

Install using `uv` with:

```bash
uv add popcorn              # first-party only
uv add "popcorn[liger]"     # + Liger-Kernel backends
uv add "popcorn[fla]"       # + FLA backends
uv add "popcorn[all]"       # every backend
```

A backend is eligible only when its package is installed at a declared supported version:
auto-dispatch skips unavailable ones, and forcing one raises with the install hint or
version error. First-party kernels (the `popcorn` backend) are always included.

> **Note:** you may be able to use `pip` to install Popcorn (with `pip install popcorn`)
> but this path is not officially supported.

Update using `uv` with:

```bash
uv add --upgrade popcorn
```

## Call a kernel

Import a kernel and call it with tensors. Popcorn dispatches the call to an eligible
implementation, using benchmark data when available and the registered reference as a
fallback.

```python
import torch
from popcorn.kernels import rms_norm

x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

output = rms_norm(x, weight)
```

To override automatic dispatching:

```python
output = rms_norm["liger"](x, weight)
# same as:
output = rms_norm(x, weight, backend="liger")
```

## Next steps

- [Dispatching](./dispatching.md) — how a call resolves to one implementation
- [Validation](./validation.md) — compare implementations against the reference
- [Benchmarking](./benchmarking.md) — record timings and route on data
- [Kernel explorer](https://tilde-research.github.io/popcorn/kernels/) — every kernel: math, backends, measured performance

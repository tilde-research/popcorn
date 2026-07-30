---
title: Quick start
description: Install Popcorn and dispatch your first kernel
---

## Install

Install Popcorn from PyPI using `uv`:

```bash
uv pip install --torch-backend=auto popcorn           # first-party only
uv pip install --torch-backend=auto "popcorn[liger]"  # + Liger-Kernel backends
uv pip install --torch-backend=auto "popcorn[fla]"    # + FLA backends
```

`--torch-backend=auto` detects the available CPU, CUDA, ROCm, or XPU environment and selects the
matching PyTorch wheel index.

> [!WARNING]
> `pip install popcorn` is not generally supported. Install the correct PyTorch build for your
> hardware first, then install Popcorn with pip. We recommend `uv` because
> `--torch-backend=auto` selects PyTorch for you. The `fa3` extra must be installed from source;
> see [CONTRIBUTING.md](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md).

Add one extra per backend you want. There is no "everything" extra: the backends pin mutually
exclusive requirements, so installing them together does not resolve.

Installing also fetches the benchmark cache that dispatch selects implementations with, from the
[popcorn-reports](https://huggingface.co/datasets/tilde-research/popcorn-reports) dataset at the
revision this version pins. Set `POPCORN_SKIP_REPORTS=1` to install without it; popcorn then falls
back to the reference for every call and says so. Refresh it at any time with:

```bash
uv run python -m popcorn.bench pull
```

The shipped cache covers the hardware it was recorded on. To measure what your own machine has no
timing for and add it to your local cache, use `python -m popcorn.bench fill`. Already-cached
combinations are skipped, and `--force` re-measures them.

A backend is eligible only when its package is installed at a declared supported version:
auto-dispatch skips unavailable ones, and forcing one raises with the install hint or
version error. First-party kernels (the `popcorn` backend) are always included.

Upgrade Popcorn and refresh its pinned report cache with:

```bash
uv pip install --upgrade --torch-backend=auto popcorn
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

- [Dispatching](./dispatching.md): how a call resolves to one implementation
- [Validation](./validation.md): compare implementations against the reference
- [Benchmarking](./benchmarking.md): record timings and route on data
- [Kernel explorer](https://tilde-research.github.io/popcorn/kernels): every kernel, its math, backends, and measured performance

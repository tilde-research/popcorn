---
title: Quick start
description: Install Popcorn and dispatch your first kernel
---

## Install

Install Popcorn from PyPI with `uv`:

```bash
uv pip install popcorn           # first-party kernels
uv pip install "popcorn[liger]"  # Liger backends
uv pip install "popcorn[fla]"    # FLA backends
uv pip install "popcorn[cudnn]"  # cuDNN attention
uv pip install "popcorn[transformer_engine]"  # Transformer Engine softmax
```

> [!NOTE]
> Add `--torch-backend=auto` to let `uv` select a PyTorch build for your hardware.

> [!WARNING]
> `pip install popcorn` is not generally supported. If required, install the appropriate
> PyTorch build before installing Popcorn.

Install one extra for each required backend. Some backends have incompatible dependency
requirements, so there is no combined extra.

Installation fetches the pinned benchmark data from
[popcorn-reports](https://huggingface.co/datasets/tilde-research/popcorn-reports).
Set `POPCORN_SKIP_REPORTS=1` to install without reports. Automatic dispatch then uses the
reference implementation. Refresh the reports with:

```bash
uv run popcorn bench pull
```

Reports apply only to the hardware on which they were recorded. Use
`popcorn bench fill` to measure missing cases on the current device. See
[Benchmarking](./benchmarking.md) for details.

A backend is eligible only when a supported package version is installed. Automatic dispatch
skips unavailable backends. Forcing an unavailable backend raises an installation or version
error. First-party kernels under the `popcorn` backend are always available.

Upgrade Popcorn and its pinned reports with:

```bash
uv pip install --upgrade popcorn
```

## Call a kernel

Import and call a kernel. Popcorn selects an eligible implementation using benchmark data,
with the registered reference as the fallback.

```python
import torch
from popcorn.kernels import rms_norm

x = torch.randn(2, 512, 4096, device="cuda", dtype=torch.bfloat16)
weight = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

output = rms_norm(x, weight)
```

To select a backend explicitly:

```python
output = rms_norm["liger"](x, weight)
# same as:
output = rms_norm(x, weight, backend="liger")
```

## Next steps

- [Dispatching](./dispatching.md): Implementation selection
- [Validation](./validation.md): Correctness checks
- [Benchmarking](./benchmarking.md): Performance measurement
- [Kernel explorer](https://tilde-research.github.io/popcorn/kernels): Kernels, backends, and results

---
title: Registration
description: Define kernels and bind implementations
---

A kernel starts with a reference that defines its public signature and semantics.
`@register_kernel` replaces the reference function with a dispatcher while retaining the
reference as a universally available implementation.

```python
import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def squared_relu(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    return torch.relu(x).square()
```

Implementations can be bound directly by dotted import path and are imported lazily on
first use.

```python
squared_relu.register("custom", source="custom_ops.squared_relu")
```

Use an adapter when the source signature differs, and declare shape, dtype, or forward-only
constraints at registration. See
[CONTRIBUTING.md](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md) for
the complete registration contract.

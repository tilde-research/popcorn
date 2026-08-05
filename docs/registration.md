---
title: Registration
description: Define kernels and bind implementations
---

A kernel begins with a reference that defines its public signature and semantics.
`@register_kernel` returns a dispatcher and retains the reference as the universal fallback.

```python
import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def squared_relu(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    return torch.relu(x).square()
```

Bind an implementation by dotted import path. The source is loaded on first use.

```python
squared_relu.register("custom", source="custom_ops.squared_relu")
```

Use an adapter when the source signature differs. Declare dtype, value, and forward-only
constraints at registration; shape validity comes from benchmark reports. See
[CONTRIBUTING.md](https://github.com/tilde-research/popcorn/blob/main/CONTRIBUTING.md) for
the complete registration contract.

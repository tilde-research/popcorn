---
title: torch.compile (Experimental)
description: Experimental integration between torch.compile and Popcorn
---

> [!WARNING]
> This integration is experimental. Enable it for selected operations and measure
> end-to-end performance.

`popcorn.compile.enable()` installs an Inductor rewrite pass. It replaces subgraphs that match
a Popcorn reference with the corresponding `torch.ops.popcorn` call. Unmatched graphs are
unchanged.

```python
import popcorn.compile

popcorn.compile.enable(ops=["rms_norm", "swiglu"])  # or enable() for every kernel
model = torch.compile(model)
```

## Behavior

- Patterns are traced from each reference for each dtype and optional-argument combination.
- A rewrite requires an exact ATen graph match and an eligible backend beyond the reference.
- Training uses `popcorn::<kernel>_backward`, which adds one dispatched forward call per
  backward pass.
- Initial pattern tracing may take several seconds per kernel. Limit `ops` to the kernels in
  the workload. `disable()` removes the pass while retaining traced patterns.

## Performance

The custom-op boundary can prevent Inductor from fusing adjacent operations. In H100
measurements, speedup over plain Inductor was 0.6x to 0.8x for matched `rms_norm` training
and 1.4x to 2x for matched `attn`. Results depend on the model, shapes, and available
backends. Measure the complete workload before enabling the pass in production.

## Limits

- References with input-dependent graph structure, such as a Python loop over sequence
  length, do not match. Call those kernels directly.
- Without compatible benchmark records, dispatch uses registration order. Run with
  `POPCORN_BENCH=1` on representative inputs before relying on automatic selection.

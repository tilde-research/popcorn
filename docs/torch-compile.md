---
title: torch.compile
description: Route compiled models through Popcorn
---

Compiled models can route through Popcorn without source changes. `popcorn.compile.enable()`
installs an inductor pass that pattern-matches subgraphs computing a Popcorn kernel — a
hand-written RMSNorm, a swiglu, an attention block — and rewrites them into the kernel's
`torch.ops.popcorn` binding, which dispatches as usual at run time.

```python
import popcorn.compile

popcorn.compile.enable(ops=["rms_norm", "swiglu"])  # or enable() for every kernel
model = torch.compile(model)
```

Patterns are traced from each kernel's reference, per dtype and optional-argument
combination: a subgraph is rewritten when it decomposes to exactly the ATen sequence of the
reference, and a miss is silently left to inductor. Inference and training graphs both
match; in training, the backward is served by `popcorn::<kernel>_backward`, which replays
the forward through the dispatcher and differentiates through the selected backend (one
extra forward per backward). A match is also declined when no backend beyond the reference
could serve the call — rewriting only to route back to the reference would just add a
custom-op boundary. Tracing costs a few seconds per kernel on first `enable`, so scope
`ops=` to what you use; `disable()` uninstalls the pass and keeps the patterns for a later
`enable`.

> **A rewrite is not automatically a win.** For small memory-bound kernels (norms, glu
> blocks) inductor's fused codegen is often at roofline and beats any dispatched backend,
> while the custom-op boundary blocks fusion into neighboring ops — on an H100, a matched
> `rms_norm` train step measures 0.6–0.8x plain inductor. The rewrite pays off when a
> backend holds an algorithmic advantage the compiler cannot recover: matched `attn` routes
> to flash-attention and measures 1.4–2x plain inductor, forward and training alike.
> Measure end to end, and prefer `enable(ops=[...])` scoped to attention-class kernels.

Two structural limits are worth knowing. References whose traced graph shape depends on the
input — a python loop over sequence length, as in the linear-attention scans — can never
pattern-match; call those ops directly (eager `gla` dispatch beats compiled-unrolled
inductor by ~50x). And in regions with no recorded benchmarks the tuner falls back to
registration order, so the routed backend is not necessarily the fastest for your shapes:
run `POPCORN_BENCH=1` once on representative inputs to pin routing to data.

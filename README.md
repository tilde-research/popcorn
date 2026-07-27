<p align="center">
  <img src="images/popcorn-banner.png" alt="Popcorn"/>
</p>

<!-- popcorn:badges -->
<p align="center">
  <img src="https://img.shields.io/badge/kernels-96-blue" alt="kernels"/>
  <img src="https://img.shields.io/badge/backends-8-blue" alt="backends"/>
  <img src="https://img.shields.io/badge/implementations-128-blue" alt="implementations"/>
  <img src="https://img.shields.io/badge/grid%20rows-31%2C099-blue" alt="grid rows"/>
</p>
<!-- /popcorn:badges -->

***Popcorn*** 🍿 is a development and dispatch library for high-performance machine learning kernels. It unifies **96 kernels** and more than **100 optimized implementations** behind a single API and automatically selects the fastest backend for your hardware. Every backend is rigorously validated against a reference implementation and benchmarked in a consistent suite, so **speed never comes at the cost of correctness**.
Popcorn also ships a growing collection of first-party kernels, along with the environment and harness to build, test, benchmark, and deploy new kernels and backends. Contributions are welcome.

---
* [Terminology](#terminology)
* [Installation](#installation)
    * [Installing Backends](#installing-backends)
    * [Updating](#updating)
* [Usage](#usage)
    * [Quick start](#quick-start)
    * [Dispatching](#dispatching)
    * [Registration](#registration)
    * [Validation](#validation)
    * [Benchmarking](#benchmarking)
    * [Tuning](#tuning)
    * [torch.compile](#torchcompile)
* [Supported kernels](#supported-kernels)
* [Limitations](#limitations)
* [Contributing](#contributing)
* [Acknowledgement](#acknowledgement)

## Terminology
Due to common overloading of some frequently used terms, consult the following definitions:
<table>
  <tr>
    <td><b>Kernel</b></td>
    <td>A pure unit of work with a fixed signature and semantics defined by a <i>ground-truth</i> reference.</td>
  </tr>
  <tr>
    <td><b>Input</b></td>
    <td>A valid configuration of arguments for a kernel call.</td>
  </tr>
  <tr>
    <td><b>Case</b></td>
    <td>The set of all inputs with matching tensor metadata (shape, type) and other arguments. i.e. the data inside tensors is abstracted away.</td>
  </tr>
  <tr>
    <td><b>Implementation</b></td>
    <td>A function that matches the output of a <i>kernel's reference</i> for a subset of valid cases.</td>
  </tr>
  <tr>
    <td><b>Backend</b></td>
    <td>A library or collection of <i>kernel implementations</i>.</td>
  </tr>
</table>


## Installation

Install using `uv` with:
```bash
uv add popcorn              # first-party only
uv add "popcorn[liger]"     # + Liger-Kernel backends
uv add "popcorn[fla]"       # + FLA backends
uv add "popcorn[all]"       # every backend
```

A backend is eligible only when its package is installed at a declared supported version: auto-dispatch skips unavailable ones, and forcing one raises with the install hint or version error. First-party kernels (the `popcorn` backend) are always included.

> [!NOTE]
> You may be able to use `pip` to install Popcorn (with `pip install popcorn`) but this path is not officially supported.

### Installing Backends
To install the prerequisites for an additional backend:
```bash
uv add "popcorn[<new_extra>]"
```

### Updating
Update using `uv` with:
```bash
uv add --upgrade popcorn
```


## Usage
### Quick Start

Import a kernel and call it with tensors. Popcorn dispatches the call to an eligible implementation, using benchmark data when available and the registered reference as a fallback.

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

### Dispatching

Each call resolves to one implementation in three steps:

1. **Eligibility.** An implementation is a candidate only when its package is installed at a supported version, it has a backward pass if the call needs gradients, and the inputs satisfy its dtype/value gates.
2. **Correctness.** Exact recorded failures are excluded. Automatic selection further requires an exact pass row or membership in a fitted validity region (learned from the report table); otherwise the reference serves the call.
3. **Speed.** Among the remaining candidates, the fastest wins, judged by the nearest recorded benchmark on the same device, dtype, and gradient mode. The reference competes on equal terms: when it measures fastest — or when timings are within a small indifference margin — it is selected.

The decision is memoized per call configuration (shapes, dtypes, device, gradient mode, and scalar arguments), so dispatch adds negligible overhead in steady state. New validation records, benchmarks, or registrations invalidate the memo.

Nothing is ever pinned on the kernel; you override selection per call or per scope:

```python
output = rms_norm(x, weight, backend="fla")   # force this call
output = rms_norm["fla"](x, weight)           # equivalent

with rms_norm["fla"] as rms_norm:             # every call in the block
    ...                                       # nestable and context-local

rms_norm.available_backends()                 # ('fla', 'liger', 'quack', 'torch')
print(rms_norm)                               # signature and per-implementation constraints
```

A forced implementation never falls back: it raises `DispatchError` when it cannot serve the call, whether because its package is missing (the error names the install extra), the inputs are unsupported, or the case has a recorded failure (the error carries the recorded reason).

#### Lazy measurement

`bench=True` or `POPCORN_BENCH=1` measures and records on first encounter with a missing conclusive row, then uses the new evidence for selection. Results land in the user cache, `${XDG_CACHE_HOME:-~/.cache}/popcorn` by default, overridden with `POPCORN_CACHE_DIR`. Validity regions are filled offline with `python -m popcorn.bench map`.

### Registration

A kernel starts with a reference that defines its public signature and semantics. `@register_kernel` replaces the reference function with a dispatcher while retaining the reference as a universally available implementation.

```python
import torch
from jaxtyping import Float
from torch import Tensor

from popcorn import register_kernel


@register_kernel
def squared_relu(x: Float[Tensor, "... hidden"]) -> Float[Tensor, "... hidden"]:
    return torch.relu(x).square()
```

Implementations can be bound directly by dotted import path and are imported lazily on first use.

```python
squared_relu.register("custom", source="custom_ops.squared_relu")
```

Use an adapter when the source signature differs, and declare shape, dtype, or forward-only constraints at registration. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete registration contract.

### Validation

Validation compares eligible implementations with higher-precision reference results. It checks forward outputs and, when inputs require gradients, backward results.

```python
results = rms_norm.validate(x, weight)
assert all(result.status == "pass" for result in results)
```

Results are stored per case, device, PyTorch version, backend version, and gradient mode, and are stamped with a fingerprint of the kernel code: results recorded for a since-edited reference or implementation are ignored (comments and formatting don't count). A known failure is removed from automatic dispatch; without a pass row or fitted region, the reference is used.

`validate` always re-runs the comparison. To measure on first use where a conclusive record is missing:

```python
output = rms_norm(x, weight, bench=True)
```

```bash
POPCORN_BENCH=1 python train.py
```

### Benchmarking

`benchmark` validates before timing and records implementation and reference latency and peak memory. Future calls use these measurements for dispatch.

```python
for result in rms_norm.benchmark(x, weight):
    print(result.status, result.bench)
```

To validate, benchmark, and select the exact fastest implementation on first use, pass `bench=True` to the call or enable it for the whole application:

```python
output = rms_norm(x, weight, bench=True)
```

```bash
POPCORN_BENCH=1 python train.py
```

Both APIs write to the user cache and never modify the package's bundled reports. First use is synchronous and may compile every eligible implementation.

Every recorded result is also logged on the `popcorn.bench` logger — one line per op, backend, and case with status, forward/backward milliseconds, and the failure reason if any — followed by where the rows were written. Enable it to watch validation and benchmarking as they happen:

```python
import logging

logging.basicConfig()
logging.getLogger("popcorn.bench").setLevel(logging.INFO)
```

To browse recorded rows instead, render the report database as HTML: `python -m popcorn.bench view --user` folds your local cache into the bundled reports.

### Tuning

Ordinary dispatch tunes each call automatically from the nearest compatible benchmark. To select one implementation for a broader region, query the recorded data explicitly:

```python
from popcorn import Range

best = rms_norm.tuner.best(
    device="cuda",
    dtype="bfloat16",
    grad=False,
    normalized_shape=Range(1024, 8192),
)
output = best(x, weight)
```

`best` returns the implementation with the lowest median recorded latency across the region. It only reads existing benchmark data; it never runs benchmarks itself.

### torch.compile

Compiled models can route through Popcorn without source changes. `popcorn.compile.enable()` installs an inductor pass that pattern-matches subgraphs computing a Popcorn kernel — a hand-written RMSNorm, a swiglu, an attention block — and rewrites them into the kernel's `torch.ops.popcorn` binding, which dispatches as usual at run time.

```python
import popcorn.compile

popcorn.compile.enable(ops=["rms_norm", "swiglu"])  # or enable() for every kernel
model = torch.compile(model)
```

Patterns are traced from each kernel's reference, per dtype and optional-argument combination: a subgraph is rewritten when it decomposes to exactly the ATen sequence of the reference, and a miss is silently left to inductor. Inference and training graphs both match; in training, the backward is served by `popcorn::<kernel>_backward`, which replays the forward through the dispatcher and differentiates through the selected backend (one extra forward per backward). A match is also declined when no backend beyond the reference could serve the call — rewriting only to route back to the reference would just add a custom-op boundary. Tracing costs a few seconds per kernel on first `enable`, so scope `ops=` to what you use; `disable()` uninstalls the pass and keeps the patterns for a later `enable`.

> [!IMPORTANT]
> A rewrite is not automatically a win. For small memory-bound kernels (norms, glu blocks) inductor's fused codegen is often at roofline and beats any dispatched backend, while the custom-op boundary blocks fusion into neighboring ops — on an H100, a matched `rms_norm` train step measures 0.6–0.8x plain inductor. The rewrite pays off when a backend holds an algorithmic advantage the compiler cannot recover: matched `attn` routes to flash-attention and measures 1.4–2x plain inductor, forward and training alike. Measure end to end, and prefer `enable(ops=[...])` scoped to attention-class kernels.

Two structural limits are worth knowing. References whose traced graph shape depends on the input — a python loop over sequence length, as in the linear-attention scans — can never pattern-match; call those ops directly (eager `gla` dispatch beats compiled-unrolled inductor by ~50x). And in regions with no recorded benchmarks the tuner falls back to registration order, so the routed backend is not necessarily the fastest for your shapes: run `POPCORN_BENCH=1` once on representative inputs to pin routing to data.

## Supported Kernels

## Limitations

### Correctness

Popcorn validates implementations across a broad grid of inputs and does its best to warn when an exact case has not been validated. This reduces the risk of numerical errors but does not guarantee correctness for every possible input. Every comparison also assumes that the registered reference is correct: Popcorn measures agreement with that reference, not correctness in the abstract, therefore "correctness" here is the measure of similarity to the reference. Validate representative production inputs, particularly unusual shapes, dtypes, and optional arguments.

### Performance
Popcorn benchmarks kernels in isolation. This is useful for comparing implementations in an apples-to-apples fashion, but these results do not necessarily predict end-to-end model performance. PyTorch and `torch.compile` may fuse or rewrite surrounding native operations, while a backend call or custom-op boundary can limit those optimizations. Tensor layouts, memory traffic, synchronization, and compilation overhead can also change the result in a real workload.
It's important to keep in mind that dispatch currently optimizes recorded execution time, not peak memory. For some workloads, we may prefer a slower implementation with a smaller memory footprint. Benchmark representative model code and explicitly tune or select a backend when latency and memory requirements are both important.

## Contributing
New kernels and backends follow a guided flow: [CONTRIBUTING.md](CONTRIBUTING.md). Benchmarks come from contributors' machines; run the grid on your hardware and commit the reports.

## Acknowledgement

Popcorn stands on the shoulders of the open-source kernel ecosystem. The optimized backends it dispatches to are built and maintained by their authors:

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [quack](https://github.com/Dao-AILab/quack)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [Unsloth](https://github.com/unslothai/unsloth)

Adapted files credit their origin in the header comment.

## Citation

```bibtex
@software{popcorn,
  author = {{Tilde Research}},
  title  = {Popcorn: kernel dispatch for PyTorch},
  url    = {https://github.com/tilde-research/popcorn},
  year   = {2026}
}
```

## License

[Apache-2.0](LICENSE).
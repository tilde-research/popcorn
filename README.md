<p align="center">
  <img src="images/popcorn-banner.png" alt="Popcorn"/>
</p>

<!-- popcorn:badges -->
<p align="center">
  <img src="https://img.shields.io/badge/kernels-96-blue" alt="kernels"/>
  <img src="https://img.shields.io/badge/backends-6-blue" alt="backends"/>
  <img src="https://img.shields.io/badge/implementations-111-blue" alt="implementations"/>
  <img src="https://img.shields.io/badge/grid%20rows-19%2C016-blue" alt="grid rows"/>
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

1. **Eligibility.** An implementation is a candidate only when its package is installed at a supported version, it has a backward pass if the call needs gradients, and the inputs satisfy its declared shape, dtype, and value constraints.
2. **Correctness.** Implementations with a recorded failure for this exact case are excluded. Selecting one that has no recorded pass emits `UnvalidatedWarning` once (see [Validation](#validation)).
3. **Speed.** Among the remaining candidates, the fastest wins, judged by the nearest recorded benchmark on the same device, dtype, and gradient mode. The reference competes on equal terms: when it measures fastest, it is selected. Without benchmark data, candidates are tried in registration order, reference last.

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

#### Validation flags

Two flags harden dispatch, per call as keyword arguments or process-wide as environment variables:

| Per call | Process-wide | Effect |
|---|---|---|
| `validate=True` | `POPCORN_VALIDATE=1` | Validate unrecorded cases synchronously before dispatch; only implementations with a recorded pass are selected. |
| `bench=True` | `POPCORN_BENCH=1` | Additionally record timings and select the exact fastest implementation. Takes precedence over `validate`. |

The keyword flags only enable: a call cannot opt out of a mode set in the environment. Both modes run synchronously inside the call, so a first encounter with a new configuration may compile and check every candidate before returning. Results land in the user cache, `${XDG_CACHE_HOME:-~/.cache}/popcorn` by default, overridden with `POPCORN_CACHE_DIR`.

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

Results are stored per case, device, PyTorch version, backend version, and gradient mode, and are stamped with a fingerprint of the kernel code: results recorded for a since-edited reference or implementation are ignored (comments and formatting don't count). A known failure is removed from automatic dispatch; an unvalidated implementation remains usable but emits `UnvalidatedWarning`.

`validate` always re-runs the comparison. To instead validate on first use, only where a conclusive record is missing, pass `validate=True` to the call or enable it for the whole application:

```python
output = rms_norm(x, weight, validate=True)
```

```bash
POPCORN_VALIDATE=1 python train.py
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

## Supported Kernels

<!-- popcorn:matrix -->
| kernel | torch | fa3 | fla | liger | popcorn | quack |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `abc` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `add_rms_norm` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `addmm` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `attn` | ✔ | ✔* | ✔* | ✘ | ✘ | ✘ |
| `attn_varlen` | ✔ | ✔* | ✔* | ✘ | ✘ | ✘ |
| `based` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `bias_gelu` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `bit_linear` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `chunk_global_cumsum` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `chunk_local_cumsum` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `comba` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `cross_entropy` | ✔ | ✘ | ✔* | ✔ | ✘ | ✔* |
| `delta_rule` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `deltaformer` | ✔ | ✘ | ✘ | ✘ | ✘ | ✘ |
| `dyt` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `embedding` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `forgetting_attn` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `gated_delta_product` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `gated_delta_rule` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `gated_oja_rule` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `geglu` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `gelu` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `gla` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `group_norm` | ✔ | ✘ | ✔* | ✔* | ✘ | ✘ |
| `group_norm_linear` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `grpo` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `grpo_offpolicy` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `gsa` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `hadamard_transform` | ✔ | ✘ | ✘ | ✘ | ✘ | ✔ |
| `hgrn` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `int8_int2_matmul` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `iplr_delta_rule` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `jsd` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `kda` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `kda_gate` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `kda_gate_cumsum` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `kl_div` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `l2_norm` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `layer_norm` | ✔ | ✘ | ✔* | ✔* | ✘ | ✘ |
| `layer_norm_gated` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `layer_norm_linear` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `layer_norm_linear_quant` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `layer_norm_swish_linear` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `lightning_attn` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `linear_attn` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `linear_cross_entropy` | ✔ | ✘ | ✔* | ✔* | ✘ | ✔* |
| `linear_jsd` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `linear_kl_div` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `llama4_rope` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `log_linear_attn` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `log_sigmoid` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `logsumexp` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `matmul` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `mean_pooling` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `mesa_net` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `mesa_net_decode` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `momoe` | ✔ | ✘ | ✘ | ✘ | ✔* | ✘ |
| `multi_token_attention` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `neighborhood_attn` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `nsa` | ✔ | ✘ | ✔* | ✘ | ✔* | ✘ |
| `nsa_compression` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `path_attn` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `poly_norm` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `qwen2vl_mrope` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `rebased` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `retention` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rms_norm` | ✔ | ✘ | ✔ | ✔* | ✘ | ✔* |
| `rms_norm_gated` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rms_norm_linear` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rms_norm_linear_quant` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `rms_norm_swish_linear` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rope` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `rotary_embedding` | ✔ | ✘ | ✔ | ✘ | ✔ | ✔ |
| `rwkv4` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `rwkv6` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rwkv7` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `rwkv7_addcmul` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rwkv7_channel_mixing` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `rwkv7_gate_output` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `rwkv7_k_update` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `selective_log_softmax` | ✔ | ✘ | ✘ | ✔ | ✘ | ✘ |
| `sigmoid` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `simple_gla` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `softmax` | ✔ | ✘ | ✔ | ✔* | ✘ | ✘ |
| `solve_tril` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `sparsemax` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `sqrelu` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `swiglu` | ✔ | ✘ | ✔ | ✔ | ✔ | ✘ |
| `swiglu_linear` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `swiglu_mlp` | ✔ | ✘ | ✘ | ✘ | ✘ | ✘ |
| `swish` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `titans_linear` | ✔ | ✘ | ✔* | ✘ | ✘ | ✘ |
| `token_shift` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `ttt` | ✔ | ✘ | ✔ | ✘ | ✘ | ✘ |
| `tvd` | ✔ | ✘ | ✘ | ✔* | ✘ | ✘ |
| `wall_attn` | ✔ | ✘ | ✘ | ✘ | ✔* | ✘ |
<!-- /popcorn:matrix -->

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
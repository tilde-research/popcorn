<p align="center">
  <img src="images/popcorn-banner.png" alt="Popcorn"/>
</p>

<!-- popcorn:badges -->
<p align="center">
  <img src="https://img.shields.io/badge/kernels-96-blue" alt="kernels"/>
  <img src="https://img.shields.io/badge/backends-6-blue" alt="backends"/>
  <img src="https://img.shields.io/badge/implementations-128-blue" alt="implementations"/>
  <img src="https://img.shields.io/badge/grid%20rows-31%2C176-blue" alt="grid rows"/>
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

## Documentation



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

The agent optimization loop (`python -m popcorn.bench.loop`) is inspired by [AutoKernel](https://github.com/RightNow-AI/autokernel)'s edit–evaluate–keep/revert cycle for autonomous kernel search.

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
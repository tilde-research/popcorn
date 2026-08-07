<p align="center">
  <img src="https://raw.githubusercontent.com/tilde-research/popcorn/main/images/popcorn-banner.png" alt="Popcorn"/>
</p>

<!-- popcorn:badges -->
<p align="center">
  <a href="https://pypi.org/project/popcorn/"><img src="https://img.shields.io/pypi/v/popcorn" alt="PyPI"/></a>
  <a href="https://github.com/tilde-research/popcorn/releases/latest"><img src="https://img.shields.io/github/v/release/tilde-research/popcorn" alt="release"/></a>
  <img src="https://img.shields.io/badge/kernels-96-blue" alt="kernels"/>
  <img src="https://img.shields.io/badge/backends-8-blue" alt="backends"/>
  <img src="https://img.shields.io/badge/implementations-222-blue" alt="implementations"/>
  <img src="https://img.shields.io/badge/grid%20rows-152%2C116-blue" alt="grid rows"/>
</p>
<!-- /popcorn:badges -->

--------------------------------------------------------------------------------

<h3 align="center">
  <a href="https://tilde-research.github.io/popcorn/docs">Documentation</a>
  &nbsp;|&nbsp;
  <a href="https://tilde-research.github.io/popcorn/kernels">Kernel Explorer</a>
</h3>

<!-- * [About](#about)
* [Installation](#installation)
    * [Installing Backends](#installing-backends)
    * [Updating](#updating)
* [Limitations](#limitations)
* [Contributing](#contributing)
* [Acknowledgement](#acknowledgement) -->

## About
***Popcorn*** 🍿 is a development and dispatch library for high-performance machine learning kernels. It unifies **96 kernels** and more than **100 optimized implementations** behind a single API, then selects among eligible backends using hardware-specific benchmark records. Each implementation is checked against a PyTorch reference; failed and unmeasured cases remain visible evidence rather than being presented as successful validation.
Popcorn also ships a growing collection of first-party kernels, along with the environment and harness to build, test, benchmark, and deploy new kernels and backends. Contributions are welcome.

## Installation

Install Popcorn from PyPI using `uv`:
```bash
uv pip install popcorn           # first-party only
uv pip install "popcorn[liger]"  # + Liger-Kernel backends
uv pip install "popcorn[fla]"    # + FLA backends
uv pip install "popcorn[cudnn]"  # + cuDNN attention
uv pip install "popcorn[transformer_engine]"  # + Transformer Engine softmax
```

> [!NOTE]
> Add `--torch-backend=auto` to let `uv` select a PyTorch build for your hardware.

> [!WARNING]
> `pip install popcorn` may work but is not generally supported. Install the desired PyTorch build for your hardware first, then install Popcorn with pip.

Add one extra per backend you want. There is no "everything" extra: the backends pin mutually exclusive requirements, so installing them together does not resolve.

Installing also fetches the benchmark cache that dispatch selects implementations with, from the [popcorn-reports](https://huggingface.co/datasets/tilde-research/popcorn-reports) dataset at the revision this version pins. Set `POPCORN_SKIP_REPORTS=1` to install without it; popcorn then falls back to the reference for every call and says so. Refresh it at any time with:
```bash
popcorn bench pull
```

A backend is eligible only when its package is installed at a declared supported version: auto-dispatch skips unavailable ones, and forcing one raises with the install hint or version error. First-party kernels (the `popcorn` backend) are always included.

### Installing Backends
To install the prerequisites for an additional backend:
```bash
uv pip install "popcorn[<new_extra>]"
```

### Updating
Upgrade Popcorn and refresh its pinned report cache with:
```bash
uv pip install --upgrade popcorn
```
## Limitations

### Correctness

Popcorn validates implementations across a broad grid of inputs and does its best to warn when an exact case has not been validated. This reduces the risk of numerical errors but does not guarantee correctness for every possible input. Every comparison also assumes that the registered reference is correct: Popcorn measures agreement with that reference, not correctness in the abstract, therefore "correctness" here is the measure of similarity to the reference. Validate representative production inputs, particularly unusual shapes, dtypes, and optional arguments.

### Performance
Popcorn benchmarks kernels in isolation. This is useful for comparing implementations in an apples-to-apples fashion, but these results do not necessarily predict end-to-end model performance. PyTorch and `torch.compile` may fuse or rewrite surrounding native operations, while a backend call or custom-op boundary can limit those optimizations. Tensor layouts, memory traffic, synchronization, and compilation overhead can also change the result in a real workload.
It's important to keep in mind that dispatch currently optimizes recorded execution time, not peak memory. For some workloads, we may prefer a slower implementation with a smaller memory footprint. Benchmark representative model code and explicitly tune or select a backend when latency and memory requirements are both important.

## Contributing
New kernels and backends follow a guided flow: [CONTRIBUTING.md](CONTRIBUTING.md). Benchmarks come from contributors' machines; run the grid on your hardware and include its matrix rows for review. Maintainers publish accepted report evidence to the pinned dataset.

## Acknowledgement

Popcorn stands on the shoulders of the open-source kernel ecosystem. The optimized backends it dispatches to are built and maintained by their authors:

- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [quack](https://github.com/Dao-AILab/quack)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Unsloth](https://github.com/unslothai/unsloth)

The agent optimization loop (`popcorn loop`) is inspired by [AutoKernel](https://github.com/RightNow-AI/autokernel)'s edit–evaluate–keep/revert cycle for autonomous kernel search.

## Citation

```bibtex
@software{popcorn,
  author = {Timor Averbuch},
  title  = {Popcorn: kernel dispatch for PyTorch},
  url    = {https://github.com/tilde-research/popcorn},
  year   = {2026}
}
```

## License

[Apache-2.0](LICENSE).
---
title: Overview
description: A development and dispatch library for high-performance ML kernels
---

**Popcorn** provides a common API for high-performance machine learning kernels. It includes
96 kernels and more than 100 optimized implementations, selected using hardware-specific
benchmark data.

Each kernel has a PyTorch reference. Validation failures and unmeasured cases remain explicit
and are never treated as successful results. Popcorn also provides tools to develop, test,
benchmark, and deploy first-party kernels.

## Terminology

These terms have specific meanings in Popcorn:

| Term | Meaning |
| --- | --- |
| **Kernel** | A pure unit of work with a fixed signature and semantics defined by a *ground-truth* reference. |
| **Input** | A valid configuration of arguments for a kernel call. |
| **Case** | A call configuration defined by tensor metadata and other arguments. Tensor values are excluded. |
| **Implementation** | A function that matches the output of a *kernel's reference* for a subset of valid cases. |
| **Backend** | A library or collection of *kernel implementations*. |

## Guides

- [Quick start](./quickstart.md): Install Popcorn and call a kernel
- [Dispatching](./dispatching.md): Understand implementation selection
- [Registration](./registration.md): Define kernels and bind implementations
- [Validation](./validation.md): Compare implementations with the reference
- [Benchmarking](./benchmarking.md): Record performance data
- [Tuning](./tuning.md): Select an implementation for a shape range
- [torch.compile (Experimental)](./torch-compile.md): Route compiled models through Popcorn
- [Optimization loop (Experimental)](./loop.md): Evaluate candidate kernels in isolated processes

See the [kernel explorer](https://tilde-research.github.io/popcorn/kernels) for kernel
definitions, backends, and measured performance.
